from typing import Optional
"""
models/TEMPLATE.py — Copy this file to add a new model plugin.

Steps
-----
1. Copy this file to  models/<your_model_name>.py
   (use lowercase_with_underscores for the name, e.g. models/xgboost_reg.py)

2. Replace every occurrence of "TEMPLATE" / "template" with your model name.

3. Fill in the three required methods:
     cli_args  — add argparse flags your model needs
     __init__  — store hyperparameters; build the sklearn/PyTorch/... estimator
     fit       — train on (X_train, y_train)
     predict   — return predictions for X_test

4. Optionally override extra_info() to surface diagnostics in the stamp file.

5. Run:
     python3 cv.py WRKDIR FILEDIR NUMFILES KFOLDS EPSILON INDEX \\
         --model_file <your_model_name>

That's it.  No changes to cv.py, cv.sh, or any other file are needed.
"""


import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -- The two imports you always need ----------------------------------------
from models.base import CVModel, register


# Change "template" to your chosen model name (must match the filename stem).
@register("template")
class TemplateModel(CVModel):

    # ------------------------------------------------------------------
    # 1. CLI flags
    # ------------------------------------------------------------------
    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("template options")
        # Add your model's hyperparameter flags here, e.g.:
        g.add_argument("--template_param", type=float, default=1.0,
                       help="Example hyperparameter (default: 1.0)")
        g.add_argument("--n_components", type=int, default=500,
                       help="PCA components (default: 500; --no_pca to skip)")
        g.add_argument("--no_pca", action="store_true",
                       help="Skip PCA and use raw scaled features")

    # ------------------------------------------------------------------
    # 2. Constructor
    # ------------------------------------------------------------------
    def __init__(self, args: argparse.Namespace) -> None:
        self._param = args.template_param
        self._n_components = args.n_components
        self._use_pca = not args.no_pca

        # Internal state
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._model = None  # replace with your estimator type

    # ------------------------------------------------------------------
    # 3. Preprocessing helpers (copy-paste boilerplate, usually unchanged)
    # ------------------------------------------------------------------
    def _preprocess_train(self, X: np.ndarray) -> np.ndarray:
        self._scaler = StandardScaler()
        X_sc = self._scaler.fit_transform(X)
        if self._use_pca:
            n_comp = min(self._n_components, X.shape[0], X.shape[1])
            print(f"[TEMPLATE] PCA n_components={n_comp}")
            self._pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
            return self._pca.fit_transform(X_sc)
        return X_sc

    def _preprocess_test(self, X: np.ndarray) -> np.ndarray:
        X_sc = self._scaler.transform(X)
        return self._pca.transform(X_sc) if self._use_pca else X_sc

    # ------------------------------------------------------------------
    # 4. fit / predict
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        X_pp = self._preprocess_train(X_train)
        print(f"[TEMPLATE] Fitting with param={self._param} ...")

        # ------ Replace this block with your model ------
        # from sklearn.linear_model import Ridge
        # self._model = Ridge(alpha=self._param)
        # self._model.fit(X_pp, y_train)
        raise NotImplementedError("Replace this stub with your model's fit() call.")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Call fit() before predict()"
        return self._model.predict(self._preprocess_test(X_test))

    # ------------------------------------------------------------------
    # 5. Optional diagnostics for the stamp file
    # ------------------------------------------------------------------
    def extra_info(self) -> dict:
        return {"template_param": self._param, "use_pca": self._use_pca}
