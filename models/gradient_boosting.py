"""
models/gradient_boosting.py — Gradient Boosting Regressor.

Registered as "gradient_boosting".
Usage: --model_file gradient_boosting  [--gb_n_estimators N]
                                       [--gb_lr LR]
                                       [--gb_max_depth N]
                                       [--gb_subsample F]
                                       [--gb_tune]  [--gb_k_inner N]
                                       [--pca]  [--n_components N]

Design
------
By default returns a Pipeline with fixed hyperparameters.  If --gb_tune is
set, wraps the pipeline in a GridSearchCV that searches over learning_rate
and max_depth (the two most impactful GBM hyperparameters), keeping
n_estimators fixed to control compute cost.

Pipeline layout:
    StandardScaler  →  [PCA →]  GradientBoostingRegressor

For faster alternatives on large datasets, swap GradientBoostingRegressor
for HistGradientBoostingRegressor (sklearn) or install lightgbm/xgboost and
replace the estimator here — the Pipeline and GridSearchCV wrapper remain
identical.
"""

import argparse

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("gradient_boosting")
class GradientBoostingModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("gradient_boosting options")
        g.add_argument(
            "--gb_n_estimators",
            type=int,
            default=300,
            help="Number of boosting stages (trees).  (default: 300)",
        )
        g.add_argument(
            "--gb_lr",
            type=float,
            default=0.05,
            help=(
                "Learning rate / shrinkage applied to each tree's contribution. "
                "Lower values require more trees but generalise better. "
                "(default: 0.05)"
            ),
        )
        g.add_argument(
            "--gb_max_depth",
            type=int,
            default=4,
            help=(
                "Maximum depth of each individual tree.  "
                "Shallower trees reduce overfitting.  (default: 4)"
            ),
        )
        g.add_argument(
            "--gb_subsample",
            type=float,
            default=0.8,
            help=(
                "Fraction of training samples used to fit each tree (stochastic GBM). "
                "Values < 1.0 reduce variance.  (default: 0.8)"
            ),
        )
        g.add_argument(
            "--gb_tune",
            action="store_true",
            default=False,
            help=(
                "Wrap the pipeline in GridSearchCV to tune learning_rate and "
                "max_depth via nested CV.  Expensive — use sparingly.  (default: off)"
            ),
        )
        g.add_argument(
            "--gb_k_inner",
            type=int,
            default=3,
            help=(
                "Inner CV folds for GridSearchCV (only used when --gb_tune is set). "
                "(default: 3)"
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
            default=200,
            help=(
                "PCA components (only used when --pca is set). "
                "Default 200 — GBM can be slow on high-dimensional input.  (default: 200)"
            ),
        )

    @classmethod
    def build_estimator(cls, args: argparse.Namespace):
        """
        Return a Pipeline or GridSearchCV(Pipeline) for Gradient Boosting.

        Without --gb_tune
        ~~~~~~~~~~~~~~~~~
        Pipeline(StandardScaler → [PCA →] GradientBoostingRegressor) with
        hyperparameters fixed from CLI flags.

        With --gb_tune
        ~~~~~~~~~~~~~~
        GridSearchCV searching over:
          ``gradientboostingregressor__learning_rate`` : [0.01, 0.05, 0.1]
          ``gradientboostingregressor__max_depth``     : [3, 4, 6]
        n_estimators is held fixed to control compute cost.

        Parameters
        ----------
        args.gb_n_estimators : int
        args.gb_lr           : float
        args.gb_max_depth    : int
        args.gb_subsample    : float
        args.gb_tune         : bool
        args.gb_k_inner      : int
        args.pca             : bool
        args.n_components    : int

        Returns
        -------
        Pipeline  or  GridSearchCV(Pipeline)
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
            GradientBoostingRegressor(
                n_estimators=args.gb_n_estimators,
                learning_rate=args.gb_lr,
                max_depth=args.gb_max_depth,
                subsample=args.gb_subsample,
                random_state=42,
            )
        )
        pipe = make_pipeline(*steps)

        if not args.gb_tune:
            return pipe

        param_grid = {
            "gradientboostingregressor__learning_rate": [0.01, 0.05, 0.1],
            "gradientboostingregressor__max_depth":     [3, 4, 6],
        }
        inner_cv = KFold(n_splits=args.gb_k_inner, shuffle=True, random_state=42)
        return GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=1,   # GBM is serial internally; avoid nested parallelism
            verbose=1,
        )
