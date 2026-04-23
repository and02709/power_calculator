from typing import Optional
"""
models/lasso.py — Lasso Regression (L1).

Registered as "lasso".
Usage: --model_file lasso  [--lasso_alpha A] [--lasso_max_iter N] [--n_components N ] [--pca]
"""


import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


@register("lasso")
class LassoModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("lasso options")
        g.add_argument(
            "--lasso_alpha", type=float, default=0.01,
            help="L1 regularisation strength (default: 0.01)"
        )
        g.add_argument(
            "--lasso_max_iter", type=int, default=5000,
            help="Max iterations for coordinate descent (default: 5000)"
        )
        g.add_argument(
            "--n_components", type=int, default=500,
            help="PCA components before Lasso (default: 500)"
        )
        g.add_argument(
            "--pca", action="store_true", default=False,
            help="Apply PCA preprocessing (default: off)"
        )

    def __init__(self, args: argparse.Namespace) -> None:
        self._alpha = args.lasso_alpha
        self._max_iter = args.lasso_max_iter
        self._n_components = args.n_components
        self._use_pca = args.pca
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._model: Optional[Lasso] = None

    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if self._use_pca:
            n_comp = min(self._n_components, X.shape[0], X.shape[1])
            print(f"[Lasso] PCA n_components={n_comp}")
            self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            return self._pca.fit_transform(X_sc)
        return X_sc

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(X)
        return self._pca.transform(X_sc) if self._use_pca else X_sc

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pp = self._preprocess_train(X_train)
        print(f"[Lasso] Fitting Lasso (alpha={self._alpha}, max_iter={self._max_iter}) ...")
        self._model = Lasso(alpha=self._alpha, max_iter=self._max_iter, random_state=42)
        self._model.fit(X_pp, y_train)
        n_nonzero = int(np.sum(self._model.coef_ != 0))
        print(f"[Lasso] Non-zero coefficients: {n_nonzero} / {len(self._model.coef_)}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._preprocess_test(X_test))

    def extra_info(self) -> dict:
        n_nz = int(np.sum(self._model.coef_ != 0)) if self._model is not None else None
        return {
            "lasso_alpha": self._alpha,
            "use_pca": self._use_pca,
            "n_nonzero_coef": n_nz,
        }
