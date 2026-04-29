"""
models/ridge.py — Ridge Regression with built-in cross-validated alpha selection.

Registered as "ridge".
Usage: --model_file ridge  [--ridge_alphas "1,10,100,1e3,1e4,1e5"]
                           [--ridge_cv_folds N]
                           [--pca]  [--n_components N]

Design
------
Uses ``RidgeCV`` rather than wrapping ``Ridge`` in ``GridSearchCV``.
sklearn's ``RidgeCV`` selects alpha via an efficient analytic LOO or k-fold
path (QR-based; no re-fitting per candidate), making it significantly cheaper
than a ``GridSearchCV`` loop for the same candidate set.

Pipeline layout (always):
    StandardScaler  →  [PCA →]  RidgeCV

Because all preprocessing lives inside the Pipeline, sklearn's
``cross_validate()`` guarantees that the scaler (and PCA, if used) are
fitted only on the training fold of each outer split — no data leakage.

After fitting, ``pipeline[-1].alpha_`` holds the alpha chosen for that fold
and is printed by ``cv.py``'s per-fold diagnostics loop.

Nested CV alternative
---------------------
If you need explicit nested CV with a ``GridSearchCV`` inner loop (e.g. to
inspect ``best_params_`` per fold or to benchmark against ``RidgeCV``),
use ``--model_file ridge_nested`` instead.
"""

import argparse
from typing import List

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_alphas(s: str) -> List[float]:
    """
    Parse a comma-separated string of floats into a Python list.

    Parameters
    ----------
    s : str
        E.g. ``"1,10,100,1e3,1e4,1e5"``

    Returns
    -------
    list of float
        E.g. ``[1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]``

    Raises
    ------
    argparse.ArgumentTypeError
        If any token cannot be converted to float.
    """
    try:
        return [float(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--ridge_alphas must be comma-separated floats, got: '{s}'"
        )


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

@register("ridge")
class RidgeModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("ridge options")
        g.add_argument(
            "--ridge_alphas",
            type=str,
            default="1,10,100,1000,10000,100000",
            help=(
                "Comma-separated regularisation strength candidates for RidgeCV. "
                "RidgeCV selects the best alpha via internal k-fold CV on each "
                "outer training fold.  (default: '1,10,100,1e3,1e4,1e5')"
            ),
        )
        g.add_argument(
            "--ridge_cv_folds",
            type=int,
            default=5,
            help=(
                "Number of inner CV folds for RidgeCV alpha selection. "
                "Pass 0 to use leave-one-out (RidgeCV default, LOO). "
                "(default: 5)"
            ),
        )
        g.add_argument(
            "--pca",
            action="store_true",
            default=False,
            help=(
                "Prepend a PCA dimensionality-reduction step to the pipeline. "
                "Recommended for very high-dimensional FC matrices. "
                "(default: off)"
            ),
        )
        g.add_argument(
            "--n_components",
            type=int,
            default=500,
            help=(
                "Number of PCA components to retain. "
                "Clamped to min(n_components, n_features, n_samples) at fit time. "
                "Only used when --pca is set.  (default: 500)"
            ),
        )

    @classmethod
    def build_estimator(cls, args: argparse.Namespace):
        """
        Return an unfitted Pipeline:  StandardScaler → [PCA →] RidgeCV.

        Parameters
        ----------
        args.ridge_alphas   : str
            Comma-separated alpha candidates, e.g. ``"1,10,100,1e3,1e4,1e5"``.
        args.ridge_cv_folds : int
            Inner CV folds for alpha selection. ``0`` → LOO (``cv=None``).
        args.pca            : bool
            Whether to include a PCA step between scaler and RidgeCV.
        args.n_components   : int
            PCA dimensionality (only relevant when ``args.pca`` is True).

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline ready for ``cross_validate()``.
        """
        alphas = _parse_alphas(args.ridge_alphas)
        # cv=None → RidgeCV uses efficient leave-one-out; cv=k → k-fold
        inner_cv = args.ridge_cv_folds if args.ridge_cv_folds > 0 else None

        steps = [StandardScaler()]
        if args.pca:
            steps.append(
                PCA(
                    n_components=args.n_components,
                    svd_solver="randomized",
                    random_state=42,
                )
            )
        steps.append(
            RidgeCV(
                alphas=alphas,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                # store_cv_results is only compatible with LOO (cv=None);
                # when cv is an integer k-fold, the attribute is unavailable.
                store_cv_results=(inner_cv is None),
            )
        )
        return make_pipeline(*steps)
