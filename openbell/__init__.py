"""OpenBell - autonomous paper-trading agent for the US cash session.

WHAT: Marks the `openbell` folder as a Python package and records its version.
HOW:  Python runs this file on the first `import openbell...`. The only thing it defines
      is `__version__`, which `doctor.py` prints at the top of its report.
WHY:  Having a package (instead of loose scripts) lets the modules import each other with
      relative imports (`from .engine import Engine`) and lets the app start with
      `python -m openbell <command>`, which runs `openbell/__main__.py`.
"""
# Bumped by hand on a release; shown in the `doctor` report header so a pasted report
# says which build produced it.
__version__ = "0.1.0"
