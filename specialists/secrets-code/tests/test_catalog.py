"""The catalogue loader of the generator (secretsgen/catalog.py): a missing catalogue is reported, never a crash."""
import os

from secretsgen import catalog


def test_a_missing_catalogue_on_another_drive_is_reported_not_raised(monkeypatch, tmp_path):
    """On Windows `os.path.relpath` raises when $PUREBYTE_DATA is on another drive than the working directory: the
    message falls back to the absolute path instead of crashing the import."""
    def other_drive(path, start=None):
        raise ValueError("path is on mount 'D:', start on mount 'C:'")
    monkeypatch.setattr(catalog.os.path, "relpath", other_drive)
    p = str(tmp_path / "catalogs" / "gitleaks.json")
    assert catalog._shown(p) == p
    monkeypatch.setattr(catalog, "_CAT", p)
    monkeypatch.setattr(catalog, "_MISSING", [])
    assert catalog._load() == ([], [], []) and catalog._MISSING == [p]
    assert not os.path.exists(p)
