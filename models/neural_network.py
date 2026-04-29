"""
models/neural_network.py — Multi-Layer Perceptron Regressor.

Registered as "neural_network".
Usage: --model_file neural_network  [--nn_hidden_layers "256,128"]
                                    [--nn_activation relu]
                                    [--nn_lr LR]
                                    [--nn_max_iter N]
                                    [--nn_alpha A]
                                    [--pca]  [--n_components N]

Design
------
Returns a Pipeline(StandardScaler → [PCA →] MLPRegressor) with fixed
hyperparameters.  No inner GridSearchCV is used by default because MLP is
expensive to train and has a large hyperparameter space; a fixed architecture
is often sufficient for neuroimaging regression.

To add hyperparameter search, follow the pattern in ridge_nested.py:
wrap the pipeline in a GridSearchCV and use ``mlpregressor__hidden_layer_sizes``
etc. in the param_grid.

Pipeline layout:
    StandardScaler  →  [PCA →]  MLPRegressor

MLPRegressor is configured with:
  - adam solver (adaptive learning rate)
  - early stopping (monitors 10% validation split)
  - n_iter_no_change=20 for patience
"""

import argparse
from typing import Tuple

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CVModel, register


def _parse_layers(s: str) -> Tuple[int, ...]:
    """
    Parse a comma-separated string of ints into a tuple of layer sizes.

    Parameters
    ----------
    s : str
        E.g. ``"256,128"`` → ``(256, 128)``

    Raises
    ------
    argparse.ArgumentTypeError
        If any token is not a positive integer.
    """
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
            "--nn_hidden_layers",
            type=str,
            default="256,128",
            help=(
                "Comma-separated hidden layer sizes, e.g. '256,128' for two layers. "
                "(default: '256,128')"
            ),
        )
        g.add_argument(
            "--nn_activation",
            type=str,
            default="relu",
            choices=["relu", "tanh", "logistic"],
            help="Activation function for hidden layers (default: relu).",
        )
        g.add_argument(
            "--nn_lr",
            type=float,
            default=1e-3,
            help="Initial learning rate for the adam solver (default: 0.001).",
        )
        g.add_argument(
            "--nn_max_iter",
            type=int,
            default=500,
            help=(
                "Maximum training epochs.  Early stopping may terminate sooner. "
                "(default: 500)"
            ),
        )
        g.add_argument(
            "--nn_alpha",
            type=float,
            default=1e-4,
            help="L2 regularisation coefficient (default: 1e-4).",
        )
        g.add_argument(
            "--pca",
            action="store_true",
            default=False,
            help=(
                "Prepend PCA before the MLP.  Recommended for very high-dimensional "
                "FC matrices to reduce input dimensionality.  (default: off)"
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
        Return an unfitted Pipeline:  StandardScaler → [PCA →] MLPRegressor.

        Parameters
        ----------
        args.nn_hidden_layers : str   — e.g. '256,128'
        args.nn_activation    : str   — 'relu', 'tanh', or 'logistic'
        args.nn_lr            : float — initial learning rate
        args.nn_max_iter      : int   — max epochs
        args.nn_alpha         : float — L2 regularisation
        args.pca              : bool
        args.n_components     : int

        Returns
        -------
        sklearn.pipeline.Pipeline
        """
        hidden = _parse_layers(args.nn_hidden_layers)

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
            MLPRegressor(
                hidden_layer_sizes=hidden,
                activation=args.nn_activation,
                solver="adam",
                learning_rate_init=args.nn_lr,
                max_iter=args.nn_max_iter,
                alpha=args.nn_alpha,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                verbose=False,
            )
        )
        return make_pipeline(*steps)
