"""
models/custom.py — User-defined custom model plugin.

Registered as "custom".
Usage: --model_file custom  [--custom_script PATH]
                            [--pca]  [--n_components N]

Purpose
-------
This plugin is an extension point for individual users who want to plug in their
own machine-learning method without modifying any core pipeline files.

How it works
------------
1.  The user creates a Python script (default: custom_model.py in the working
    directory, or any path passed via --custom_script).

2.  That script must define a top-level function:

        def build_estimator(args):
            \"\"\"
            Return an unfitted sklearn-compatible estimator.

            Parameters
            ----------
            args : argparse.Namespace
                All CLI arguments parsed by cv.py, including any extra flags
                added by cli_args() below.  Use getattr(args, 'my_flag', default)
                to access custom flags safely.

            Returns
            -------
            sklearn estimator
                A Pipeline, GridSearchCV, or any object implementing
                sklearn's fit/predict API.  cross_validate() will call
                .fit(X_train, y_train) on each outer fold.
            \"\"\"
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            # Replace with your own estimator:
            return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

3.  Run the pipeline normally:
        python3 cv.py WRKDIR FILEDIR NUMFILES INDEX --model_file custom

    Or with a non-default script location:
        python3 cv.py ... --model_file custom --custom_script /path/to/my_model.py

Passing extra CLI flags to your script
---------------------------------------
To accept additional hyperparameter flags, also define in your script:

    def cli_args(parser):
        \"\"\"Optionally add argparse flags for your model.\"\"\"
        g = parser.add_argument_group("custom model options")
        g.add_argument("--my_alpha", type=float, default=1.0,
                       help="Example hyperparameter.")

If cli_args() is not defined in your script, no extra flags are registered
and any unrecognised flags forwarded by PWR.sh are silently ignored.

Minimal working example (custom_model.py)
-----------------------------------------
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def build_estimator(args):
        alpha    = getattr(args, 'my_alpha', 0.5)
        l1_ratio = getattr(args, 'my_l1', 0.5)
        return make_pipeline(StandardScaler(),
                             ElasticNet(alpha=alpha, l1_ratio=l1_ratio))

    def cli_args(parser):
        g = parser.add_argument_group("custom model options")
        g.add_argument("--my_alpha", type=float, default=0.5)
        g.add_argument("--my_l1",   type=float, default=0.5)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from models.base import CVModel, register

# Default script name searched relative to the current working directory.
_DEFAULT_SCRIPT = "custom_model.py"


def _load_user_module(script_path: str):
    """
    Dynamically load a Python source file and return the module object.

    Parameters
    ----------
    script_path : str
        Absolute or relative path to the user's Python script.

    Returns
    -------
    module
        The loaded module, with attributes accessible via ``module.build_estimator``,
        ``module.cli_args``, etc.

    Raises
    ------
    FileNotFoundError
        If the script does not exist at the resolved path.
    ImportError
        If the script raises an exception during import.
    """
    p = Path(script_path).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"[custom] Custom model script not found: {p}\n"
            f"Create '{p.name}' in your working directory or pass "
            f"--custom_script /full/path/to/your_script.py"
        )
    spec = importlib.util.spec_from_file_location("_custom_user_model", str(p))
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        raise ImportError(f"[custom] Error loading {p}: {e}") from e
    return mod


@register("custom")
class CustomModel(CVModel):
    """
    Proxy plugin that delegates to a user-supplied Python script.

    The script must define ``build_estimator(args)``.  It may optionally
    define ``cli_args(parser)`` to register extra CLI flags.
    """

    # Cache the loaded module so the file is only read once per process.
    _user_module = None
    _user_script: str = _DEFAULT_SCRIPT

    # ------------------------------------------------------------------
    # 1. CLI flags
    # ------------------------------------------------------------------
    @classmethod
    def cli_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("custom model options")
        g.add_argument(
            "--custom_script",
            type=str,
            default=_DEFAULT_SCRIPT,
            help=(
                f"Path to your Python script that defines build_estimator(args). "
                f"The script may also define cli_args(parser) to add extra flags. "
                f"(default: {_DEFAULT_SCRIPT!r} in the current working directory)"
            ),
        )
        g.add_argument(
            "--pca",
            action="store_true",
            default=False,
            help="Prepend PCA to the pipeline — only used if your build_estimator() reads args.pca. (default: off)",
        )
        g.add_argument(
            "--n_components",
            type=int,
            default=500,
            help="PCA components — only used if your build_estimator() reads args.n_components. (default: 500)",
        )

        # ── Eagerly load the user module and delegate cli_args ────────────────
        # We need to do this here (during Pass 2 of cv.py's two-pass parse) so
        # that any flags defined in the user's cli_args() are registered before
        # the final parse_known_args() call.  The module path comes from the
        # already-parsed --custom_script value (or the default).
        import sys as _sys
        # Extract --custom_script from sys.argv manually (argparse not yet run)
        script = _DEFAULT_SCRIPT
        argv = _sys.argv
        for i, tok in enumerate(argv):
            if tok in ("--custom_script", "--custom-script") and i + 1 < len(argv):
                script = argv[i + 1]
                break
        cls._user_script = script

        try:
            mod = _load_user_module(script)
            cls._user_module = mod
        except FileNotFoundError as e:
            # If no script exists yet, skip — build_estimator will raise a
            # clear error later when cross_validate() is actually called.
            print(f"[WARN] {e}", file=sys.stderr)
            return

        if hasattr(mod, "cli_args"):
            try:
                mod.cli_args(parser)
            except Exception as e:
                print(f"[WARN] custom cli_args() raised: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 2. Estimator factory
    # ------------------------------------------------------------------
    @classmethod
    def build_estimator(cls, args: argparse.Namespace):
        """
        Delegate to the user script's ``build_estimator(args)`` function.

        Parameters
        ----------
        args : argparse.Namespace
            Full parsed CLI namespace, including any flags added by the
            user script's ``cli_args()`` and the standard cv.py flags.

        Returns
        -------
        sklearn estimator
            Whatever the user's ``build_estimator()`` returns.
        """
        script = getattr(args, "custom_script", cls._user_script)

        # Load (or re-use cached) user module
        if cls._user_module is None or cls._user_script != script:
            cls._user_module = _load_user_module(script)
            cls._user_script = script

        mod = cls._user_module

        if not hasattr(mod, "build_estimator"):
            raise AttributeError(
                f"[custom] '{Path(script).resolve()}' must define a "
                "top-level function: build_estimator(args) -> sklearn estimator.\n"
                "See models/custom.py docstring for a minimal working example."
            )

        print(f"[INFO] custom: delegating to build_estimator() in {Path(script).resolve()}")
        estimator = mod.build_estimator(args)
        return estimator
