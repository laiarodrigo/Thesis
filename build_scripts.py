#!/usr/bin/env python3
# Legacy wrapper for scripts/build_project_db.py
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "scripts" / "project" / "build_project_db.py"), run_name="__main__")
