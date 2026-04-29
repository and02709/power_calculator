"""
models/svr.py — Support Vector Regression with nested GridSearchCV.

Registered as "svr".
Usage: --model_file svr  [--svr_C_vals "0.1,1,10,100"]
                         [--svr_kernel K]
                         [--svr_epsilon E]
                         [--svr_k_inner N]
                         [--pca]  [--n_components N]

Design
------
SVR has no built-in CV analogue, so hyperparameter tuning uses a
GridSearchCV inner loop.  The C (regularisation) parameter is searched;
kernel and epsilon are fixed per run via CLI flags.

Performance note
----------------
SVR scales as O(n²) to O(n³) in the number of training samples.  PCA is
strongly recommended for large FC matrices (--pca --n_components 200).
The default inner grid searches only over C; add epsilon to param_grid
here if needed.

Pipeline layout:
    StandardScaler  →  [PCA →]  SVR(kernel, epsilon)
    wrapped in GridSearchCV over C
"""

import argparse
from typing import List

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from models.base import CVModel, register


def _parse_floats(s: str, flag: str) -> List[float]:
    try:
        return [float(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{flag} must be comma-separated floats, got: '{s}'"
        )


@register("svr")
class SVRModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("svr options")
        g.add_argument(
            "--svr_C_vals",
            type=str,
            default="0.1,1,10,100",
            help=(
                "Comma-separated C (regularisation) candidates for GridSearchCV. "
                "Larger C = less regularisation, tighter fit. "
                "(default: '0.1,1,10,100')"
            ),
        )
        g.add_argument(
            "--svr_kernel",
            type=str,
            default="rbf",
            choices=["rbf", "linear", "poly", "sigmoid"],
            help="SVR kernel (default: rbf).",
        )
        g.add_argument(
            "--svr_epsilon",
            type=float,
            default=0.1,
            help=(
                "Epsilon in the ε-insensitive loss tube. "
                "Predictions within epsilon of the true value incur no penalty. "
                "(default: 0.1)"
            ),
        )
        g.add_argument(
            "--svr_gamma",
            type=str,
            default="scale",
            help=(
                "Kernel coefficient for rbf/poly/sigmoid. "
                "'scale' uses 1 / (n_features × X.var()), 'auto' uses 1 / n_features. "
                "(default: 'scale')"
            ),
        )
        g.add_argument(
            "--svr_k_inner",
            type=int,
            default=5,
            help="Number of inner CV folds for C tuning via GridSearchCV (default: 5).",
        )
        g.add_argument(
            "--pca",
            action="store_true",
            default=False,
            help=(
                "Prepend PCA before SVR.  Strongly recommended for high-dimensional "
                "FC matrices to keep training tractable.  (default: off)"
            ),
        )
        g.add_argument(
            "--n_components",
            type=int,
            default=200,
            help=(
                "PCA components (only used when --pca is set). "
                "Default is lower than other models (200) because SVR is O(n²/n³). "
                "(default: 200)"
            ),
        )

    @classmethod
    def build_estimator(cls, args: argparse.Namespace):
        """
        Return GridSearchCV( Pipeline(StandardScaler → [PCA →] SVR) ).

        param_grid searches over ``svr__C``; kernel, epsilon, and gamma are
        fixed from CLI flags.

        Parameters
        ----------
        args.svr_C_vals   : str   — comma-separated C candidates
        args.svr_kernel   : str   — SVR kernel type
        args.svr_epsilon  : float — epsilon-tube width
        args.svr_gamma    : str   — kernel coefficient
        args.svr_k_inner  : int   — inner CV folds
        args.pca          : bool
        args.n_components : int

        Returns
        -------
        GridSearchCV(Pipeline)
        """
        C_vals   = _parse_floats(args.svr_C_vals, "--svr_C_vals")
        inner_cv = KFold(n_splits=args.svr_k_inner, shuffle=True, random_state=42)

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
            SVR(
                kernel=args.svr_kernel,
                epsilon=args.svr_epsilon,
                gamma=args.svr_gamma,
            )
        )
        pipe = make_pipeline(*steps)

        param_grid = {"svr__C": C_vals}

        return GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=-1,
            verbose=1,
        )
