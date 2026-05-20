from __future__ import annotations

import importlib
import sys
from pathlib import Path

from .__cli__ import app

# RL project root: keep screenshots' import side-effect style but with rl_master.
root = Path(__file__).resolve().parents[4]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

for module in ["archive", "extract", "parse", "filter", "deduplicate", "generate", "make", "convert", "analyze"]:
    imported = importlib.import_module(f".{module}", package=__package__)
    if hasattr(imported, "app"):
        app.add_typer(imported.app, name=module)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
