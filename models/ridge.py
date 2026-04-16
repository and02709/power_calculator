from typing import Optional
"""
models/ridge.py — Ridge Regression (PCA preprocessing optional).

Registered as "ridge".
Usage: --model_file ridge  [--ridge_alpha A] [--n_components N | --no_pca]
"""


import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("ridge")
class RidgeModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("ridge options")
        g.add_argument(
            "--ridge_alpha", type=float, default=1.0,
            help="Regularisation strength α (default: 1.0)"
        )
        g.add_argument(
            "--n_components", type=int, default=500,
            help="PCA components before Ridge (default: 500; use --no_pca to skip)"
        )
        g.add_argument(
            "--no_pca", action="store_true",
            help="Skip PCA and feed raw (scaled) features directly to Ridge"
        )

    def __init__(self, args: argparse.Namespace) -> None:
        self._alpha = args.ridge_alpha
        self._n_components = args.n_components
        self._use_pca = not args.no_pca
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._model: Optional[Ridge] = None

    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if self._use_pca:
            n_comp = min(self._n_components, X.shape[0], X.shape[1])
            print(f"[Ridge] PCA n_components={n_comp}")
            self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            return self._pca.fit_transform(X_sc)
        return X_sc

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(X)
        if self._use_pca:
            return self._pca.transform(X_sc)
        return X_sc

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pp = self._preprocess_train(X_train)
        print(f"[Ridge] Fitting Ridge (alpha={self._alpha}) ...")
        self._model = Ridge(alpha=self._alpha)
        self._model.fit(X_pp, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._preprocess_test(X_test))

    def extra_info(self) -> dict:
        return {"ridge_alpha": self._alpha, "use_pca": self._use_pca}
