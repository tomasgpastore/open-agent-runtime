import os
import sys

from assistant_cli.cli import run as run_cli
from assistant_cli.tui import run_tui


if __name__ == "__main__":
    ui_mode = os.getenv("ASSISTANT_UI", "textual").lower()
    if "--classic" in sys.argv or ui_mode in {"classic", "cli"}:
        run_cli()
    else:
        run_tui()
