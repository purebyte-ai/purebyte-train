"""Helpers for the data tests."""
import os
import shutil
import subprocess

import pytest


def git_repo(path, files, message="snapshot"):
    """A small local git repository with `files` ({relative path: bytes}) committed; returns (path, commit)."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_COMMITTER_NAME="t",
               GIT_COMMITTER_EMAIL="t@example.invalid", GIT_AUTHOR_DATE="2026-01-01T00:00:00Z",
               GIT_COMMITTER_DATE="2026-01-01T00:00:00Z")
    os.makedirs(path, exist_ok=True)
    run = lambda *a: subprocess.run(["git", *a], cwd=path, env=env, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "core.autocrlf", "false")
    for rel, data in files.items():
        p = os.path.join(path, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    run("add", "-A")
    run("commit", "-q", "-m", message)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, check=True).stdout.decode().strip()
    return str(path), commit
