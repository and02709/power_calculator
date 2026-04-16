from typing import Optional
"""
models/gradient_boosting.py — Gradient Boosting Regressor (sklearn).

Registered as "gradient_boosting".
Usage: --model_file gradient_boosting  [--gb_n_estimators N] [--gb_lr LR] ...

For a faster drop-in, install lightgbm/xgboost and swap the underlying
estimator while keeping this CVModel shell.
"""


import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("gradient_boosting")
class GradientBoostingModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("gradient_boosting options")
        g.add_argument("--gb_n_estimators", type=int,   default=300,
                       help="Number of boosting stages (default: 300)")
        g.add_argument("--gb_lr",           type=float, default=0.05,
                       help="Learning rate / shrinkage (default: 0.05)")
        g.add_argument("--gb_max_depth",    type=int,   default=4,
                       help="Max tree depth (default: 4)")
        g.add_argument("--gb_subsample",    type=float, default=0.8,
                       help="Fraction of samples per tree (default: 0.8)")
        g.add_argument("--n_components",    type=int,   default=200,
                       help="PCA components (default: 200)")
        g.add_argument("--no_pca", action="store_true", help="Skip PCA")

    def __init__(self, args: argparse.Namespace) -> None:
        self._n_estimators = args.gb_n_estimators
        self._lr           = args.gb_lr
        self._max_depth    = args.gb_max_depth
        self._subsample    = args.gb_subsample
        self._n_components = args.n_components
        self._use_pca      = not args.no_pca
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._model: Optional[GradientBoostingRegressor] = None

    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if self._use_pca:
            n_comp = min(self._n_components, X.shape[0], X.shape[1])
            print(f"[GB] PCA n_components={n_comp}")
            self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            return self._pca.fit_transform(X_sc)
        return X_sc

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(X)
        return self._pca.transform(X_sc) if self._use_pca else X_sc

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pp = self._preprocess_train(X_train)
        print(
            f"[GB] GradientBoostingRegressor n_estimators={self._n_estimators} "
            f"lr={self._lr} max_depth={self._max_depth} subsample={self._subsample}"
        )
        self._model = GradientBoostingRegressor(
            n_estimators=self._n_estimators,
            learning_rate=self._lr,
            max_depth=self._max_depth,
            subsample=self._subsample,
            random_state=42,
        )
        self._model.fit(X_pp, y_train)
        print(f"[GB] Train R²: {self._model.train_score_[-1]:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._preprocess_test(X_test))

    def extra_info(self) -> dict:
        return {
            "gb_n_estimators": self._n_estimators,
            "gb_lr": self._lr,
            "gb_max_depth": self._max_depth,
            "use_pca": self._use_pca,
        }
