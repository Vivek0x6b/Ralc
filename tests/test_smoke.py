"""Smoke test: the package imports and exposes a version."""

import ralc


def test_version():
    assert ralc.__version__ == "0.0.1"
