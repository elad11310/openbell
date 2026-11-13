"""Test package.

WHAT: Marks `tests/` as a Python package.
HOW:  Empty apart from this docstring.
WHY:  test_live_protocol.py does `from tests import fake_services`, which needs `tests`
      to be importable as a package. Run the suite from the repository root with
      `python -m pytest -q`.
"""
