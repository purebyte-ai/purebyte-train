"""Archive members in memory (deb, zip, nested zip, tar.xz) and the path patterns of `take:` blocks."""
import io
import lzma
import os
import tarfile
import zipfile

import pytest

from pbdata import archives, globs
from pbdata.util import DataError


@pytest.mark.parametrize("path,pattern,ok", [
    ("a/b/c.py", "*.py", True), ("a/b/c.py", "a/*.py", False), ("a/b/c.py", "a/**/*.py", True),
    ("c.py", "**/c.py", True), ("a/b/c.py", "**/c.py", True), ("src/pip/_vendor/x.py", "src/pip/_vendor/**", True),
    ("Dockerfile", "Dockerfile", True), ("x/Dockerfile", "Dockerfile", True), ("lib/x.so.6", "lib/*.so*", True),
    ("lib/x.so", "lib/[xy].so", True), ("lib/z.so", "lib/[xy].so", False),
])
def test_globs(path, pattern, ok):
    assert globs.match(path, pattern) is ok


def test_selected():
    assert globs.selected("a.py", None, None)
    assert not globs.selected("a.py", ["*.c"], None)
    assert not globs.selected("vendor/a.py", ["*.py"], ["vendor/**"])


def _tar_xz(files):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return lzma.compress(raw.getvalue())


def _ar(members):
    out = b"!<arch>\n"
    for name, data in members:
        header = f"{name + '/':16s}{0:<12d}{0:<6d}{0:<6d}{'100644':<8s}{len(data):<10d}`\n".encode()
        out += header + data + (b"\n" if len(data) % 2 else b"")
    return out


def test_deb_members(tmp_path):
    data = _tar_xz({"./usr/bin/tool": b"\x7fELF" + b"t" * 100, "./usr/lib/libx.so.1": b"\x7fELF" + b"x" * 50,
                    "./usr/share/doc/copyright": b"licence text"})
    deb = _ar([("debian-binary", b"2.0\n"), ("control.tar.xz", _tar_xz({"./control": b"Package: t\n"})),
               ("data.tar.xz", data)])
    p = tmp_path / "t_1.0_amd64.deb"
    p.write_bytes(deb)
    got = dict(archives.iter_members(str(p), include=["usr/bin/*", "usr/lib/*.so*"]))
    assert set(got) == {"usr/bin/tool", "usr/lib/libx.so.1"}
    assert archives.read_member(str(p), "usr/share/doc/copyright") == b"licence text"
    with pytest.raises(DataError):
        archives.read_member(str(p), "usr/bin/missing")


def test_zip_with_nested_zip(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("encodings/utf_8.pyc", b"\xcb\x0d\x0d\x0a" + b"p" * 40)
    p = tmp_path / "python-embed.zip"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("python.exe", b"MZ" + b"e" * 100)
        z.writestr("python313.zip", inner.getvalue())
    got = dict(archives.iter_members(str(p), include=["*.exe", "*.pyc"], nested=["python3*.zip"]))
    assert set(got) == {"python.exe", "python313.zip!encodings/utf_8.pyc"}
    assert archives.read_member(str(p), "python313.zip!encodings/utf_8.pyc").startswith(b"\xcb\x0d")


def test_tarball_and_raw(tmp_path):
    p = tmp_path / "go.tar.xz"
    p.write_bytes(_tar_xz({"go/bin/go": b"\x7fELF" + b"g" * 10, "go/README": b"hi"}))
    assert [n for n, _ in archives.iter_members(str(p), include=["go/bin/*"])] == ["go/bin/go"]
    r = tmp_path / "tool.exe"
    r.write_bytes(b"MZ....")
    assert list(archives.iter_members(str(r))) == [("tool.exe", b"MZ....")]
    assert archives.detect_format("x.whl") == "zip" and archives.detect_format("x.deb") == "deb"
    assert archives.detect_format(os.path.join("a", "b.tar.gz")) == "tar"
