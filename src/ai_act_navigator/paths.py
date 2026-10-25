"""Where the project keeps its data, independent of the folder you start a command from.

Paths like Path("data/chroma") are relative to the *current* folder: run a command from
somewhere else and it silently creates a second, empty data/ folder there. Building every
path from this file's own location avoids that.
"""

from pathlib import Path

# This file is src/ai_act_navigator/paths.py, so the project folder is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"  # git-ignored: everything in it can be recreated
