"""Offline tests of the data scripts: no network, no download, nothing written outside a temporary directory.

    python -m pytest data/tests -q
"""
import os
import sys

import pytest

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DATA not in sys.path:
    sys.path.insert(0, DATA)


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    """A private $PUREBYTE_DATA for one test."""
    home = tmp_path / "purebyte-data"
    home.mkdir()
    monkeypatch.setenv("PUREBYTE_DATA", str(home))
    return home


@pytest.fixture
def F():
    from pbdata import paths
    return paths.fingerprint_module()
