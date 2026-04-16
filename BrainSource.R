"""
models/random_forest.py — PCA + Random Forest (original cv.py behaviour).

Registered as "random_forest".
Usage: --model_file random_forest  [--n_components N] [--n_estimators N]
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("random_forest")
class RandomForestModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("random_forest options")
        g.add_argument(
            "--n_components", type=int, default=500,
            help="PCA components before RF (default: 500)"
        )
        g.add_argument(
            "--n_estimators", type=int, default=500,
            help="Number of RF trees (default: 500)"
        )

    def __init__(self, args: argparse.Namespace) -> None:
        self._n_components = args.n_components
        self._n_estimators = args.n_estimators
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._rf: RandomForestRegressor | None = None
        self._oob_r2: float | None = None
        self._var_explained: float | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        n_comp = min(self._n_components, X.shape[0], X.shape[1])
        print(f"[RF] StandardScaler + PCA (n_components={n_comp}) ...")
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
        X_pca = self._pca.fit_transform(X_sc)
        self._var_explained = float(
            np.cumsum(self._pca.explained_variance_ratio_)[-1]
        ) * 100
        print(f"[RF] PCA variance explained: {self._var_explained:.1f}%")
        return X_pca

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        assert self._scaler is not None and self._pca is not None
        return self._pca.transform(self._scaler.transform(X))

    # ------------------------------------------------------------------
    # CVModel interface
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pca = self._preprocess_train(X_train)
        print(f"[RF] Fitting RandomForestRegressor (n_estimators={self._n_estimators}, n_jobs=-1) ...")
        self._rf = RandomForestRegressor(
            n_estimators=self._n_estimators,
            n_jobs=-1,
            random_state=42,
            oob_score=True,
        )
        self._rf.fit(X_pca, y_train)
        self._oob_r2 = self._rf.oob_score_
        print(f"[RF] OOB R²: {self._oob_r2:.4f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._rf is not None
        return self._rf.predict(self._preprocess_test(X_test))

    def extra_info(self) -> dict:
        return {
            "oob_r2": self._oob_r2,
            "pca_var_explained_pct": self._var_explained,
            "n_components_actual": (
                self._pca.n_components_ if self._pca is not None else None
            ),
        }
