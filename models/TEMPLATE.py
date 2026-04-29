"""
models/TEMPLATE.py — Copy this file to add a new model plugin (v2 interface).

Steps
-----
1.  Copy to  models/<your_model_name>.py
    (use lowercase_with_underscores, e.g. models/xgboost_reg.py)

2.  Replace every occurrence of "template" / "TEMPLATE" with your model name.

3.  Fill in ``cli_args()`` with any argparse flags your model needs, and
    ``build_estimator()`` to return an unfitted sklearn Pipeline or GridSearchCV.

4.  Run locally to smoke-test:
        python3 scripts/cv_reference.py --model_file <your_model_name>

5.  Submit to the cluster:
        python3 cv.py WRKDIR FILEDIR NUMFILES INDEX --model_file <your_model_name>

No changes to cv.py, cv.sh, or any other file are needed.

Interface summary (v2)
----------------------
Plugins do NOT implement fit() / predict() directly.
sklearn's cross_validate() calls those on the estimator returned by build_estimator().

Fixed hyperparameter  →  return make_pipeline(StandardScaler(), YourModel(param=val))
Built-in CV variant   →  return make_pipeline(StandardScaler(), YourModelCV(...))
Tuned via grid search →  return GridSearchCV(make_pipeline(...), param_grid, cv=inner_cv)
"""

import argparse

from sklearn.linear_model import Ridge         # Replace with your estimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
        # Add your model's hyperparameter flags here.
        g.add_argument(
            "--template_param",
            type=float,
            default=1.0,
            help="Example hyperparameter (default: 1.0).",
        )
        # Optional PCA flags — include if your model benefits from dimensionality
        # reduction (copy this block verbatim; cv.py passes --pca / --n_components).
        g.add_argument(
            "--pca",
            action="store_true",
            default=False,
            help="Prepend PCA to the pipeline (default: off).",
        )
        g.add_argument(
            "--n_components",
            type=int,
            default=500,
            help="PCA components (only used when --pca is set; default: 500).",
        )

    # ------------------------------------------------------------------
    # 2. Estimator factory
    # ------------------------------------------------------------------
    @classmethod
    def build_estimator(cls, args: argparse.Namespace):
        """
        Return an unfitted sklearn estimator.

        The object returned here is passed directly to cross_validate().
        sklearn calls .fit(X_train, y_train) and .predict(X_test) on each fold.

        Common patterns
        ---------------
        Fixed hyperparameter (no tuning):
            return make_pipeline(StandardScaler(), Ridge(alpha=args.template_param))

        Built-in CV estimator (e.g. RidgeCV):
            from sklearn.linear_model import RidgeCV
            return make_pipeline(StandardScaler(), RidgeCV(alphas=[1, 10, 100]))

        Nested GridSearchCV:
            from sklearn.model_selection import GridSearchCV, KFold
            pipe = make_pipeline(StandardScaler(), Ridge())
            param_grid = {'ridge__alpha': [1, 10, 100]}
            return GridSearchCV(pipe, param_grid, cv=KFold(5), scoring='neg_root_mean_squared_error')
        """
        from sklearn.decomposition import PCA  # import here to keep top-level clean

        steps = [StandardScaler()]
        if args.pca:
            steps.append(PCA(n_components=args.n_components,
                             svd_solver="randomized", random_state=42))

        # ---- Replace this line with your estimator ----
        steps.append(Ridge(alpha=args.template_param))
        # -----------------------------------------------

        return make_pipeline(*steps)
