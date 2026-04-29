"""
models/base.py — Abstract base class and global registry for CV model plugins.

Interface version: 2 (sklearn cross_validate edition)

Every plugin file (e.g. models/ridge.py) must:
  1. Subclass CVModel
  2. Decorate the class with @register("<name>")  (name = file stem)
  3. Implement cli_args(parser) — register argparse flags
  4. Implement build_estimator(args) — return an unfitted sklearn estimator

Interface change from v1
------------------------
v1 required three methods on each plugin:
    fit(X_train, y_train)     — plugins managed their own preprocessing
    predict(X_test)           — and a manual train/test split

v2 requires one factory classmethod:
    build_estimator(args) -> sklearn estimator

    The returned object is passed directly to sklearn's cross_validate(),
    which handles fold generation, fitting, and scoring.  Returning a
    Pipeline (rather than a bare estimator) guarantees that preprocessing
    steps (StandardScaler, PCA) are fitted only on the training fold of
    each split, preventing data leakage.

    For models that require hyperparameter tuning, build_estimator() should
    return a GridSearchCV wrapping a Pipeline.  cross_validate()'s outer
    loop then sees a single estimator; GridSearchCV manages the inner CV
    automatically on each outer training fold.
"""

import abc
import argparse
from typing import Dict, Type

from sklearn.base import BaseEstimator  # for type hints only


# ---------------------------------------------------------------------------
# Global registry: file stem -> class
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type["CVModel"]] = {}


def register(name: str):
    """
    Class decorator — registers a CVModel subclass under *name*.

    *name* must match the plugin's filename stem (e.g. ``ridge`` for
    ``models/ridge.py``) so that ``--model_file ridge`` resolves correctly.
    """
    def _inner(cls: Type["CVModel"]):
        if name in _REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return _inner


def get_model_class(name: str) -> Type["CVModel"]:
    """Look up a registered model class by name, raising ValueError on miss."""
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {available}\n"
            "To add a new model, create models/<name>.py and subclass CVModel."
        )
    return _REGISTRY[name]


def list_models():
    """Return a sorted list of all registered model names."""
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class CVModel(abc.ABC):
    """
    Plugin interface — one subclass per model file.

    Lifecycle (v2)
    --------------
    1. ``cli_args(parser)``      — add model-specific argparse flags
    2. ``build_estimator(args)`` — return an unfitted sklearn Pipeline or
                                   GridSearchCV for use with cross_validate()

    Plugins do NOT implement fit() / predict() directly.  sklearn's
    cross_validate() calls those on the returned estimator.

    Example minimal plugin
    ----------------------
    .. code-block:: python

        @register("my_model")
        class MyModel(CVModel):

            @classmethod
            def cli_args(cls, parser):
                parser.add_argument("--my_alpha", type=float, default=1.0)

            @classmethod
            def build_estimator(cls, args):
                from sklearn.linear_model import Ridge
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                return make_pipeline(StandardScaler(), Ridge(alpha=args.my_alpha))
    """

    @classmethod
    @abc.abstractmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add model-specific CLI arguments to *parser*."""

    @classmethod
    @abc.abstractmethod
    def build_estimator(cls, args: argparse.Namespace) -> BaseEstimator:
        """
        Construct and return an **unfitted** sklearn estimator.

        The estimator must implement the sklearn fit/predict API.  It will
        be passed directly to ``cross_validate()``, which calls ``.fit()``
        on each outer training fold.

        Returns
        -------
        Pipeline
            For models with a fixed hyperparameter or a built-in CV variant
            (e.g. ``RidgeCV``, ``LassoCV``).  The Pipeline must start with
            ``StandardScaler`` (and optionally ``PCA``) to ensure that
            preprocessing is fitted only on the training fold.

        GridSearchCV(Pipeline(...))
            For models whose hyperparameter is tuned via explicit nested CV.
            The ``param_grid`` keys must use sklearn's ``stepname__param``
            double-underscore convention to reach estimator params inside
            the Pipeline.
        """
