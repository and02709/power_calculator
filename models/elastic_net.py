from typing import Optional
"""
models/elastic_net.py — ElasticNet (L1 + L2).

Registered as "elastic_net".
Usage: --model_file elastic_net  [--en_alpha A] [--en_l1_ratio R] [--n_components N | --no_pca]
"""


import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("elastic_net")
class ElasticNetModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("elastic_net options")
        g.add_argument(
            "--en_alpha", type=float, default=0.01,
            help="Overall regularisation strength (default: 0.01)"
        )
        g.add_argument(
            "--en_l1_ratio", type=float, default=0.5,
            help="L1 ratio: 0=Ridge, 1=Lasso (default: 0.5)"
        )
        g.add_argument(
            "--en_max_iter", type=int, default=5000,
            help="Max iterations (default: 5000)"
        )
        g.add_argument(
            "--n_components", type=int, default=500,
            help="PCA components (default: 500)"
        )
        g.add_argument("--no_pca", action="store_true", help="Skip PCA")

    def __init__(self, args: argparse.Namespace) -> None:
        self._alpha = args.en_alpha
        self._l1_ratio = args.en_l1_ratio
        self._max_iter = args.en_max_iter
        self._n_components = args.n_components
        self._use_pca = not args.no_pca
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._model: Optional[ElasticNet] = None

    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if self._use_pca:
            n_comp = min(self._n_components, X.shape[0], X.shape[1])
            self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            return self._pca.fit_transform(X_sc)
        return X_sc

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(X)
        return self._pca.transform(X_sc) if self._use_pca else X_sc

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pp = self._preprocess_train(X_train)
        print(f"[ElasticNet] alpha={self._alpha}, l1_ratio={self._l1_ratio}")
        self._model = ElasticNet(
            alpha=self._alpha, l1_ratio=self._l1_ratio,
            max_iter=self._max_iter, random_state=42,
        )
        self._model.fit(X_pp, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._preprocess_test(X_test))

    def extra_info(self) -> dict:
        return {
            "en_alpha": self._alpha,
            "en_l1_ratio": self._l1_ratio,
            "use_pca": self._use_pca,
        }
