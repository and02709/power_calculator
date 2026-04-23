"""
models/neural_network.py — Multi-Layer Perceptron Regressor (sklearn).

Registered as "neural_network".
Usage: --model_file neural_network  [--nn_hidden_layers "256,128"] [--nn_lr LR] ...

For GPU-backed deep networks, swap the sklearn MLP for a PyTorch model here
while keeping the same CVModel interface.
"""


import argparse
from typing import Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


def _parse_layers(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--nn_hidden_layers must be comma-separated ints, got: '{s}'"
        )


@register("neural_network")
class NeuralNetworkModel(CVModel):

    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("neural_network options")
        g.add_argument(
            "--nn_hidden_layers", type=str, default="256,128",
            help="Comma-separated hidden layer sizes (default: '256,128')"
        )
        g.add_argument(
            "--nn_activation", type=str, default="relu",
            choices=["relu", "tanh", "logistic"],
            help="Activation function (default: relu)"
        )
        g.add_argument(
            "--nn_lr", type=float, default=1e-3,
            help="Initial learning rate (default: 0.001)"
        )
        g.add_argument(
            "--nn_max_iter", type=int, default=500,
            help="Max training epochs (default: 500)"
        )
        g.add_argument(
            "--nn_alpha", type=float, default=1e-4,
            help="L2 regularisation term (default: 1e-4)"
        )
        g.add_argument(
            "--n_components", type=int, default=500,
            help="PCA components before MLP (default: 500)"
        )
        g.add_argument("--pca", action="store_true", default=False,
                       help="Apply PCA preprocessing (default: off)")

    def __init__(self, args: argparse.Namespace) -> None:
        self._hidden_layers = _parse_layers(args.nn_hidden_layers)
        self._activation = args.nn_activation
        self._lr = args.nn_lr
        self._max_iter = args.nn_max_iter
        self._alpha = args.nn_alpha
        self._n_components = args.n_components
        self._use_pca = args.pca
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._model: Optional[MLPRegressor] = None

    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if self._use_pca:
            n_comp = min(self._n_components, X.shape[0], X.shape[1])
            print(f"[NN] PCA n_components={n_comp}")
            self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            return self._pca.fit_transform(X_sc)
        return X_sc

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(X)
        return self._pca.transform(X_sc) if self._use_pca else X_sc

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pp = self._preprocess_train(X_train)
        print(
            f"[NN] Fitting MLP hidden_layers={self._hidden_layers} "
            f"activation={self._activation} lr={self._lr} max_iter={self._max_iter}"
        )
        self._model = MLPRegressor(
            hidden_layer_sizes=self._hidden_layers,
            activation=self._activation,
            learning_rate_init=self._lr,
            max_iter=self._max_iter,
            alpha=self._alpha,
            solver="adam",
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20,
            verbose=False,
        )
        self._model.fit(X_pp, y_train)
        print(f"[NN] Training completed after {self._model.n_iter_} epochs")
        print(f"[NN] Best validation loss: {self._model.best_loss_:.6f}")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._model is not None
        return self._model.predict(self._preprocess_test(X_test))

    def extra_info(self) -> dict:
        return {
            "nn_hidden_layers": str(self._hidden_layers),
            "nn_activation": self._activation,
            "nn_lr": self._lr,
            "nn_epochs_trained": (
                self._model.n_iter_ if self._model is not None else None
            ),
            "nn_best_val_loss": (
                self._model.best_loss_ if self._model is not None else None
            ),
            "use_pca": self._use_pca,
        }
