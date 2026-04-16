# models/__init__.py
# Auto-register all model plugins in this package.

from importlib import import_module
from pathlib import Path

# Import every .py file in this directory so plugins self-register.
_pkg_dir = Path(__file__).parent
for _f in sorted(_pkg_dir.glob("*.py")):
    if _f.stem not in ("__init__", "base"):
        import_module(f"models.{_f.stem}")
