"""Read the members of an archive in memory: .deb, wheels / zips / jars / nupkgs, tarballs, raw files.

Nothing is extracted to disk, so member names from an archive can never write outside the cache. Members are yielded in
sorted order (deterministic), regular files only, each capped in size.

    iter_members(path, fmt=None, include=None, exclude=None, nested=None, max_bytes=...)
        -> (member path, bytes)     nested members are named `outer!inner`

Formats: zip-like (zip, whl, jar, nupkg, apk, aar, vsix), tar (plain, gz, bz2, xz, zst), deb (the data.tar.* member of
the ar archive), raw (the file itself). `.zst` needs Python 3.14 (`compression.zstd`) or the `zstd` command on PATH.
"""
import io
import os
import shutil
import subprocess
import tarfile
import zipfile

from . import globs
from .util import DataError

ZIP_EXT = (".zip", ".whl", ".jar", ".nupkg", ".apk", ".aar", ".vsix", ".egg")
TAR_EXT = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".tar.zst", ".tzst")
MAX_MEMBER = 256 << 20            # a member larger than this is skipped (never a context file we want)
MAX_RATIO = 200                   # zip-bomb guard: uncompressed / compressed


def entry_format(e):
    """The archive format of a manifest entry. Cached files are named by their sha256 and have no extension, so the
    format comes from the entry (explicit `format`, the type, or the name in its URL), never from the cache path."""
    if e.get("format"):
        return e["format"]
    if e.get("type") == "deb":
        return "deb"
    if e.get("type") == "pypi":
        return "zip"
    name = e.get("filename") or e["url"].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return detect_format(name)


def detect_format(path):
    name = os.path.basename(path).lower()
    if name.endswith(".deb") or name.endswith(".udeb"):
        return "deb"
    if name.endswith(ZIP_EXT):
        return "zip"
    if name.endswith(TAR_EXT):
        return "tar"
    return "raw"


def _zstd_decompress(data):
    try:
        from compression import zstd                  # Python 3.14+
        return zstd.decompress(data)
    except ImportError:
        pass
    exe = shutil.which("zstd")
    if not exe:
        raise DataError("this archive is zstd-compressed: use Python 3.14 or newer, or put the `zstd` command on PATH")
    return subprocess.run([exe, "-dc"], input=data, stdout=subprocess.PIPE, check=True).stdout


def _open_tar(data, name):
    lname = name.lower()
    if lname.endswith((".zst", ".tzst")):
        return tarfile.open(fileobj=io.BytesIO(_zstd_decompress(data)), mode="r:")
    return tarfile.open(fileobj=io.BytesIO(data), mode="r:*")


def deb_data_tar(data):
    """The (name, bytes) of the data.tar.* member of a Debian binary package (an `ar` archive)."""
    if data[:8] != b"!<arch>\n":
        raise DataError("not a Debian package (no ar header)")
    pos = 8
    while pos + 60 <= len(data):
        header = data[pos:pos + 60]
        name = header[:16].decode("ascii", "replace").strip().rstrip("/")
        size = int(header[48:58].decode("ascii").strip())
        body = data[pos + 60:pos + 60 + size]
        if name.startswith("data.tar"):
            return name, body
        pos += 60 + size + (size & 1)
    raise DataError("Debian package without a data.tar member")


def _norm(member):
    m = member.replace("\\", "/")
    while m.startswith("./"):
        m = m[2:]
    return m.lstrip("/")


def _iter_tar(tf, include, exclude, max_bytes):
    members = sorted((m for m in tf.getmembers() if m.isfile()), key=lambda m: _norm(m.name))
    for m in members:
        name = _norm(m.name)
        if not name or not globs.selected(name, include, exclude) or m.size > max_bytes:
            continue
        f = tf.extractfile(m)
        if f is not None:
            yield name, f.read()


def _iter_zip(zf, include, exclude, max_bytes, nested, prefix=""):
    infos = sorted((i for i in zf.infolist() if not i.is_dir()), key=lambda i: _norm(i.filename))
    for info in infos:
        name = _norm(info.filename)
        if not name or info.file_size > max_bytes:
            continue
        if info.compress_size and info.file_size / max(1, info.compress_size) > MAX_RATIO:
            continue
        if nested and globs.match_any(name, nested):
            inner = zf.read(info)
            if inner[:4] == b"PK\x03\x04":
                with zipfile.ZipFile(io.BytesIO(inner)) as z2:
                    yield from _iter_zip(z2, include, exclude, max_bytes, None, prefix + name + "!")
            continue
        full = prefix + name
        if globs.selected(full, include, exclude) or (prefix and globs.selected(name, include, exclude)):
            yield full, zf.read(info)


def iter_members(path, fmt=None, include=None, exclude=None, nested=None, max_bytes=MAX_MEMBER):
    fmt = fmt or detect_format(path)
    if fmt == "raw":
        name = os.path.basename(path)
        if globs.selected(name, include, exclude) and os.path.getsize(path) <= max_bytes:
            with open(path, "rb") as f:
                yield name, f.read()
        return
    if fmt == "zip":
        with zipfile.ZipFile(path) as zf:
            yield from _iter_zip(zf, include, exclude, max_bytes, nested)
        return
    with open(path, "rb") as f:
        data = f.read()
    if fmt == "deb":
        name, body = deb_data_tar(data)
        with _open_tar(body, name) as tf:
            yield from _iter_tar(tf, include, exclude, max_bytes)
        return
    if fmt == "tar":
        with _open_tar(data, os.path.basename(path)) as tf:
            yield from _iter_tar(tf, include, exclude, max_bytes)
        return
    raise DataError(f"unknown archive format `{fmt}` for {path}")


def read_member(path, member, fmt=None):
    """The bytes of one member (`outer!inner` for nested zips), matched by exact name."""
    outer, _, inner = member.partition("!")
    fmt = fmt or detect_format(path)
    data = None
    if fmt == "zip":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if _norm(info.filename) == outer:
                    data = zf.read(info)
                    break
    else:
        for name, body in iter_members(path, fmt=fmt):
            if name == outer:
                data = body
                break
    if data is None:
        raise DataError(f"{os.path.basename(path)} has no member `{outer}`")
    if not inner:
        return data
    with zipfile.ZipFile(io.BytesIO(data)) as z2:
        return z2.read(inner)
