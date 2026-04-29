"""
models/ridge_nested.py — Ridge Regression with explicit nested GridSearchCV.

Registered as "ridge_nested".
Usage: --model_file ridge_nested  [--ridge_alphas "1,10,100,1e3,1e4,1e5"]
                                  [--ridge_k_inner N]
                                  [--pca]  [--n_components N]

Design
------
Wraps a Pipeline(StandardScaler → [PCA →] Ridge) in a GridSearchCV so that
cross_validate()'s outer loop sees a single sklearn estimator.  The inner CV
(--ridge_k_inner folds, KFold with shuffle) tunes alpha; the outer CV
(RepeatedKFold configured in cv.py) evaluates generalisation.

After each outer fold, ``estimator.best_params_`` and ``estimator.best_score_``
are available for inspection via ``scores['estimator']``.

When to prefer this over "ridge"
---------------------------------
• You need explicit control of inner vs outer CV folds as separate CLI params.
• You want to inspect ``best_params_`` per outer fold.
• You are testing a non-linear model that lacks a built-in CV variant.
• You want to benchmark nested CV against RidgeCV's analytic path.

Note: GridSearchCV re-fits a Ridge model for every alpha candidate on every
inner fold, which is substantially slower than RidgeCV's analytic path for
the same candidate set.  Prefer "ridge" unless you specifically need nested CV.
"""

import argparse
from typing import List

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


def _parse_alphas(s: str) -> List[float]:
    """Parse comma-separated floats into a list, e.g. '1,10,100' → [1., 10., 100.]."""
    try:
        return [float(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--ridge_alphas must be comma-separated floats, got: '{s}'"
        )


@register("ridge_nested")
class RidgeNestedModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("ridge_nested options")
        g.add_argument(
            "--ridge_alphas",
            type=str,
            default="1,10,100,1000,10000,100000",
            help=(
                "Comma-separated alpha candidates for GridSearchCV. "
                "(default: '1,10,100,1e3,1e4,1e5')"
            ),
        )
        g.add_argument(
            "--ridge_k_inner",
            type=int,
            default=5,
            help=(
                "Number of inner CV folds used by GridSearchCV for alpha selection. "
                "Inner folds use KFold with shuffle=True, random_state=42. "
                "(default: 5)"
            ),
        )
        g.add_argument(
            "--pca",
            action="store_true",
            default=False,
            help="Prepend PCA to the pipeline (default: off).",
        )
        g.add_argument(
            "--n_components",
            type=int,
            default=500,
            help="PCA components (only used when --pca is set; default: 500).",
        )

    @classmethod
    def build_estimator(cls, args: argparse.Namespace):
        """
        Return GridSearchCV( Pipeline(StandardScaler → [PCA →] Ridge) ).

        The Pipeline step names follow sklearn's make_pipeline() convention
        (lowercase class name): ``'standardscaler'``, ``'pca'`` (optional),
        ``'ridge'``.  GridSearchCV's ``param_grid`` uses double-underscore
        notation to reach the Ridge estimator inside the Pipeline:

            ``{'ridge__alpha': [1, 10, 100, ...]}``

        Parameters
        ----------
        args.ridge_alphas  : str   — comma-separated alpha candidates
        args.ridge_k_inner : int   — inner CV folds for GridSearchCV
        args.pca           : bool  — include PCA step
        args.n_components  : int   — PCA dimensionality

        Returns
        -------
        GridSearchCV
            Unfitted; ``refit=True`` so the best pipeline is re-fitted on the
            full outer training fold and can be used for prediction.
        """
        alphas   = _parse_alphas(args.ridge_alphas)
        inner_cv = KFold(n_splits=args.ridge_k_inner, shuffle=True, random_state=42)

        steps = [StandardScaler()]
        if args.pca:
            steps.append(
                PCA(
                    n_components=args.n_components,
                    svd_solver="randomized",
                    random_state=42,
                )
            )
        steps.append(Ridge())
        pipe = make_pipeline(*steps)

        # Double-underscore notation: 'ridge__alpha' sets Ridge(alpha=...) inside pipe
        param_grid = {"ridge__alpha": alphas}

        return GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            refit=True,    # refit best model on full outer training fold
            n_jobs=-1,
            verbose=1,
        )
