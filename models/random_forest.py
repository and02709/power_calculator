"""
models/random_forest.py — Random Forest Regressor with optional nested CV.

Registered as "random_forest".
Usage: --model_file random_forest  [--rf_n_estimators N]
                                   [--rf_max_features STR]
                                   [--rf_tune]  [--rf_k_inner N]
                                   [--pca]  [--n_components N]

Design
------
By default, returns a Pipeline(StandardScaler → [PCA →] RandomForestRegressor)
with fixed hyperparameters.  If --rf_tune is set, wraps the pipeline in a
GridSearchCV that searches over n_estimators and max_features.

Random Forest does not have a built-in CV analogue (unlike Ridge/Lasso), so
hyperparameter tuning requires a full GridSearchCV inner loop.  The OOB score
(oob_score_) provides an independent estimate of generalisation error and is
logged per fold by cv.py for diagnostic purposes; it is NOT the same as the
outer CV R².

Pipeline layout:
    StandardScaler  →  [PCA →]  RandomForestRegressor(oob_score=True)
"""

import argparse
from typing import List

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("random_forest")
class RandomForestModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("random_forest options")
        g.add_argument(
            "--rf_n_estimators",
            type=int,
            default=500,
            help=(
                "Number of trees.  If --rf_tune is set, this value is used as "
                "the single candidate (add more via --rf_tune_n_estimators). "
                "(default: 500)"
            ),
        )
        g.add_argument(
            "--rf_max_features",
            type=str,
            default="1.0",
            help=(
                "Fraction of features to consider at each split, or 'sqrt'/'log2'. "
                "If --rf_tune is set, a small grid around this value is searched. "
                "(default: '1.0' — use all features)"
            ),
        )
        g.add_argument(
            "--rf_tune",
            action="store_true",
            default=False,
            help=(
                "Wrap the pipeline in GridSearchCV for nested hyperparameter tuning. "
                "Searches over rf_n_estimators and rf_max_features candidates. "
                "(default: off — fixed hyperparameters, much faster)"
            ),
        )
        g.add_argument(
            "--rf_k_inner",
            type=int,
            default=3,
            help=(
                "Inner CV folds for GridSearchCV (only used when --rf_tune is set). "
                "Keep small (3–5) — RF is expensive to fit. "
                "(default: 3)"
            ),
        )
        g.add_argument(
            "--pca",
            action="store_true",
            default=False,
            help=(
                "Prepend PCA before the Random Forest.  Not always beneficial — "
                "RF handles high-dimensional data well, but PCA can reduce training "
                "time substantially for very large FC matrices.  (default: off)"
            ),
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
        Return a Pipeline or GridSearchCV( Pipeline ) for Random Forest.

        Without --rf_tune
        ~~~~~~~~~~~~~~~~~
        Returns Pipeline(StandardScaler → [PCA →] RandomForestRegressor).
        ``oob_score=True`` is always set so the OOB R² is available after
        fitting for diagnostic inspection in cv.py.

        With --rf_tune
        ~~~~~~~~~~~~~~
        Returns GridSearchCV(Pipeline(...)) searching over:
          ``randomforestregressor__n_estimators`` : [args.rf_n_estimators / 2,
                                                      args.rf_n_estimators,
                                                      args.rf_n_estimators * 2]
          ``randomforestregressor__max_features``  : ['sqrt', 0.3, 1.0]

        Parameters
        ----------
        args.rf_n_estimators : int
        args.rf_max_features : str   — float as string or 'sqrt'/'log2'
        args.rf_tune         : bool
        args.rf_k_inner      : int
        args.pca             : bool
        args.n_components    : int

        Returns
        -------
        Pipeline  or  GridSearchCV(Pipeline)
        """
        # Parse max_features: convert numeric strings to float
        try:
            max_features = float(args.rf_max_features)
        except ValueError:
            max_features = args.rf_max_features  # 'sqrt' or 'log2'

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
            RandomForestRegressor(
                n_estimators=args.rf_n_estimators,
                max_features=max_features,
                oob_score=True,   # always on for diagnostic logging
                n_jobs=-1,
                random_state=42,
            )
        )
        pipe = make_pipeline(*steps)

        if not args.rf_tune:
            return pipe

        # Build a modest tuning grid centred on the requested n_estimators
        n = args.rf_n_estimators
        param_grid = {
            "randomforestregressor__n_estimators": [max(50, n // 2), n, n * 2],
            "randomforestregressor__max_features": ["sqrt", 0.3, 1.0],
        }
        inner_cv = KFold(n_splits=args.rf_k_inner, shuffle=True, random_state=42)
        return GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            refit=True,
            n_jobs=1,   # RF already uses n_jobs=-1 internally; avoid over-subscription
            verbose=1,
        )
