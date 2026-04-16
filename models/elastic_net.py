"""
models/svr.py — Support Vector Regression.

Registered as "svr".
Usage: --model_file svr  [--svr_C C] [--svr_kernel K] [--svr_epsilon E] [--n_components N | --no_pca]

Note: SVR can be slow on large datasets. PCA (default: 200 components) is
strongly recommended to keep training tractable.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from models.base import CVModel, register


@register("svr")
class SVRModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("svr options")
        g.add_argument("--svr_C",       type=float, default=1.0,
                       help="Regularisation C (default: 1.0)")
        g.add_argument("--svr_kernel",  type=str,   default="rbf",
                       choices=["rbf", "linear", "poly", "sigmoid"],
                       help="SVR kernel (default: rbf)")
        g.add_argument("--svr_epsilon", type=float, default=0.1,
                       help="Epsilon in ε-SVR tube (default: 0.1)")
        g.add_argument("--svr_gamma",   type=str,   default="scale",
                       help="Kernel coefficient for rbf/poly/sigmoid (default: scale)")
        g.add_argument("--n_components", type=int, default=200,
                       help="PCA components before SVR (default: 200)")
        g.add_argument("--no_pca", action="store_true", help="Skip PCA")

    def __init__(self, args: argparse.Namespace) -> None:
        self._C = args.svr_C
        self._kernel = args.svr_kernel
        self._epsilon = args.svr_epsilon
        self._gamma = args.svr_gamma
        self._n_components = args.n_components
        self._use_pca = not args.no_pca
        self._scaler: StandardScaler | None = None
        self._pca: PCA | None = None
        self._model: SVR | None = None
        self._n_sv: int | None = None

    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if self._use_pca:
            n_comp = min(self._n_components, X.shape[0], X.shape[1])
            print(f"[SVR] PCA n_components={n_comp}")
            self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            return self._pca.fit_transform(X_sc)
        return X_sc

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(X)
        return self._pca.transform(X_sc) if self._use_pca else X_sc

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pp = self._preprocess_train(X_train)
        print(f"[SVR] Fitting SVR (C={self._C}, kernel={self._kernel}, epsilon={self._epsilon}) ...")
        self._model = SVR(C=self._C, kernel=self._kernel,
                         epsilon=self._epsilon, gamma=self._gamma)
        self._model.fit(X_pp, y_train)
        self._n_sv = int(self._model.n_support_[0]) if hasattr(self._model, "n_support_") else None
        if self._n_sv is not None:
            print(f"[SVR] Support vectors: {self._n_sv}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._preprocess_test(X_test))

    def extra_info(self) -> dict:
        return {
            "svr_C": self._C,
            "svr_kernel": self._kernel,
            "svr_epsilon": self._epsilon,
            "n_support_vectors": self._n_sv,
            "use_pca": self._use_pca,
        }
