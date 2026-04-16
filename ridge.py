"""
models/base.py — Abstract base class and global registry for CV model plugins.

Every plugin file (e.g. models/ridge.py) must:
  1. Subclass CVModel
  2. Call register() at module level

The registry is keyed by the plugin *file stem* (e.g. "ridge", "lasso",
"random_forest") so that `--model_file ridge` resolves to models/ridge.py.
"""

from __future__ import annotations

import abc
import argparse
from typing import Dict, Type

import numpy as np


# ---------------------------------------------------------------------------
# Global registry: stem -> class
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type["CVModel"]] = {}


def register(name: str):
    """Class decorator that registers a CVModel subclass under `name`."""
    def _inner(cls: Type[CVModel]):
        if name in _REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = cls
        return cls
    return _inner


def get_model_class(name: str) -> Type["CVModel"]:
    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available models: {available}\n"
            f"To add a new model, create models/<name>.py and subclass CVModel."
        )
    return _REGISTRY[name]


def list_models():
    return sorted(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class CVModel(abc.ABC):
    """
    Interface every model plugin must implement.

    Lifecycle
    ---------
    1. cli_args(parser)  — add model-specific argparse flags
    2. __init__(args)    — construct from parsed namespace
    3. fit(X_train, y_train)
    4. predict(X_test)   -> np.ndarray
    5. extra_info()      -> dict   (optional; logged to stamp file)
    """

    @classmethod
    @abc.abstractmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add model-specific CLI arguments to *parser*."""

    @abc.abstractmethod
    def __init__(self, args: argparse.Namespace) -> None:
        """Construct the model from the parsed CLI namespace."""

    @abc.abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data (already preprocessed)."""

    @abc.abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predictions for test data (already preprocessed)."""

    def extra_info(self) -> dict:
        """
        Return a dict of key/value pairs to include in the stamp file.
        Override in subclasses to expose model-specific diagnostics
        (e.g. OOB R², number of support vectors, etc.).
        """
        return {}
