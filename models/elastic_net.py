"""
models/elastic_net.py — ElasticNet with cross-validated alpha and l1_ratio selection.

Registered as "elastic_net".
Usage: --model_file elastic_net  [--en_l1_ratios "0.1,0.5,0.9,1.0"]
                                 [--en_n_alphas N]
                                 [--en_cv_folds N]
                                 [--en_max_iter N]
                                 [--pca]  [--n_components N]

Design
------
Uses ``ElasticNetCV``, which simultaneously searches both ``alpha`` and
``l1_ratio`` via an efficient coordinate-descent path.  This is much faster
than a nested ``GridSearchCV`` over two hyperparameter grids.

Pipeline layout:
    StandardScaler  →  [PCA →]  ElasticNetCV

Interpretation of l1_ratio
--------------------------
l1_ratio = 1.0  →  pure Lasso (all L1 penalty)
l1_ratio = 0.0  →  pure Ridge (all L2 penalty; note: use ridge plugin instead)
l1_ratio ∈ (0, 1) → mix of L1 and L2

After fitting, ``pipeline[-1].alpha_`` and ``pipeline[-1].l1_ratio_`` hold
the values chosen for that fold.
"""

import argparse
from typing import List

from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


def _parse_floats(s: str, flag: str) -> List[float]:
    """Parse a comma-separated string of floats, raising ArgumentTypeError on failure."""
    try:
        return [float(x.strip()) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{flag} must be comma-separated floats, got: '{s}'"
        )


@register("elastic_net")
class ElasticNetModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("elastic_net options")
        g.add_argument(
            "--en_l1_ratios",
            type=str,
            default="0.1,0.5,0.7,0.9,0.95,1.0",
            help=(
                "Comma-separated l1_ratio candidates for ElasticNetCV. "
                "1.0 = pure Lasso; 0.0 = pure Ridge (use ridge plugin instead). "
                "(default: '0.1,0.5,0.7,0.9,0.95,1.0')"
            ),
        )
        g.add_argument(
            "--en_n_alphas",
            type=int,
            default=100,
            help=(
                "Number of alpha candidates on a log scale (per l1_ratio value). "
                "(default: 100)"
            ),
        )
        g.add_argument(
            "--en_cv_folds",
            type=int,
            default=5,
            help="Number of inner CV folds for ElasticNetCV (default: 5).",
        )
        g.add_argument(
            "--en_max_iter",
            type=int,
            default=5000,
            help=(
                "Maximum coordinate-descent iterations per (alpha, l1_ratio) pair. "
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
        Return an unfitted Pipeline:  StandardScaler → [PCA →] ElasticNetCV.

        Parameters
        ----------
        args.en_l1_ratios : str  — comma-separated l1_ratio candidates
        args.en_n_alphas  : int  — log-spaced alpha candidates per l1_ratio
        args.en_cv_folds  : int  — inner CV folds
        args.en_max_iter  : int  — max coordinate-descent iterations
        args.pca          : bool — include PCA step
        args.n_components : int  — PCA dimensionality

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
        l1_ratios = _parse_floats(args.en_l1_ratios, "--en_l1_ratios")

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
            ElasticNetCV(
                l1_ratio=l1_ratios,
                alphas=args.en_n_alphas,      # int → log-spaced grid of that size
                cv=args.en_cv_folds,
                max_iter=args.en_max_iter,
                random_state=42,
                n_jobs=-1,
            )
        )
        return make_pipeline(*steps)
