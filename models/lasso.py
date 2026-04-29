"""
models/lasso.py — Lasso Regression with built-in cross-validated alpha selection.

Registered as "lasso".
Usage: --model_file lasso  [--lasso_n_alphas N]  [--lasso_cv_folds N]
                           [--lasso_max_iter N]
                           [--pca]  [--n_components N]

Design
------
Uses ``LassoCV`` rather than wrapping ``Lasso`` in ``GridSearchCV``.
``LassoCV`` searches a log-spaced grid of ``n_alphas`` candidates using an
efficient coordinate-descent path (the full regularisation path is computed
once per inner fold rather than once per candidate), making it significantly
cheaper than a ``GridSearchCV`` loop for the same candidate count.

Pipeline layout:
    StandardScaler  →  [PCA →]  LassoCV

After fitting, ``pipeline[-1].alpha_`` holds the alpha chosen for that fold.
"""

import argparse

from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("lasso")
class LassoModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("lasso options")
        g.add_argument(
            "--lasso_n_alphas",
            type=int,
            default=100,
            help=(
                "Number of alpha candidates on a log scale for LassoCV. "
                "LassoCV generates this many candidates automatically between "
                "alpha_max (where all coefficients are zero) and alpha_max/1000. "
                "Passed as the 'alphas' integer parameter to LassoCV. "
                "(default: 100)"
            ),
        )
        g.add_argument(
            "--lasso_cv_folds",
            type=int,
            default=5,
            help=(
                "Number of inner CV folds for LassoCV alpha selection. "
                "(default: 5)"
            ),
        )
        g.add_argument(
            "--lasso_max_iter",
            type=int,
            default=5000,
            help=(
                "Maximum number of coordinate-descent iterations. "
                "Increase if convergence warnings appear. "
                "(default: 5000)"
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
        Return an unfitted Pipeline:  StandardScaler → [PCA →] LassoCV.

        Parameters
        ----------
        args.lasso_n_alphas  : int  — number of log-spaced alpha candidates
        args.lasso_cv_folds  : int  — inner CV folds for alpha selection
        args.lasso_max_iter  : int  — max coordinate-descent iterations
        args.pca             : bool — include PCA step
        args.n_components    : int  — PCA dimensionality

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
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
            LassoCV(
                alphas=args.lasso_n_alphas,   # int → log-spaced grid of that size
                cv=args.lasso_cv_folds,
                max_iter=args.lasso_max_iter,
                random_state=42,
                n_jobs=-1,
            )
        )
        return make_pipeline(*steps)
