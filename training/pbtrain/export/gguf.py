"""Minimal GGUF v3 writer (spec/FORMAT.md §2). Written by hand, like the engine's reader: the container is simple
(header, typed metadata, tensor table, aligned data) and a dependency on ggml would buy nothing.

Only two tensor types are written: F32 (GGML type 0) and I8 (GGML type 24, raw bytes: packed ternary codes and 4-bit
n-gram rows). Tensors are written in the given order; shapes are row-major here and reversed in the file.
"""
import struct

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGNMENT = 32
T_U32, T_I32, T_F32, T_BOOL, T_STRING, T_ARRAY, T_U64 = 4, 5, 6, 7, 8, 9, 10
GGML_F32, GGML_I8 = 0, 24


def _w_str(buf, s):
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b)) + b


def _w_val(buf, v):
    if isinstance(v, bool):
        buf += struct.pack("<IB", T_BOOL, int(v))
    elif isinstance(v, (int, np.integer)):
        v = int(v)
        buf += struct.pack("<IQ", T_U64, v) if v > 2 ** 31 - 1 else struct.pack("<Ii", T_I32, v)
    elif isinstance(v, (float, np.floating)):
        buf += struct.pack("<If", T_F32, float(v))
    elif isinstance(v, str):
        buf += struct.pack("<I", T_STRING)
        _w_str(buf, v)
    elif isinstance(v, (list, tuple)):
        if not v:
            buf += struct.pack("<IIQ", T_ARRAY, T_I32, 0)
        elif all(isinstance(x, str) for x in v):
            buf += struct.pack("<IIQ", T_ARRAY, T_STRING, len(v))
            for x in v:
                _w_str(buf, x)
        elif any(isinstance(x, (float, np.floating)) for x in v):
            buf += struct.pack("<IIQ", T_ARRAY, T_F32, len(v)) + np.asarray(v, "<f4").tobytes()
        else:
            buf += struct.pack("<IIQ", T_ARRAY, T_I32, len(v)) + np.asarray(v, "<i4").tobytes()
    else:
        raise TypeError(f"unsupported metadata value: {type(v)}")


def write(path, meta: dict, tensors: dict):
    """meta: key -> value; tensors: name -> np.ndarray (float32, or uint8/int8 raw). Returns the file size."""
    head = bytearray(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), len(meta)))
    for k, v in meta.items():
        _w_str(head, k)
        _w_val(head, v)
    infos, blobs, off = bytearray(), [], 0
    for name, arr in tensors.items():
        a = np.ascontiguousarray(arr)
        if a.dtype not in (np.float32, np.uint8, np.int8):
            raise TypeError(f"{name}: dtype {a.dtype} not supported (F32 or raw bytes)")
        if a.ndim == 0 or a.ndim > 4 or min(a.shape) < 1:
            raise ValueError(f"{name}: shape {a.shape} (rank 1..4, every dimension >= 1)")
        _w_str(infos, name)
        dims = tuple(reversed(a.shape))
        infos += struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", GGML_F32 if a.dtype == np.float32 else GGML_I8, off)
        blobs.append(a.tobytes())
        off += a.nbytes + (-a.nbytes % ALIGNMENT)
    body = head + infos
    body += b"\x00" * ((-len(body)) % ALIGNMENT)
    for b in blobs:
        body += b + b"\x00" * ((-len(b)) % ALIGNMENT)
    with open(path, "wb") as f:
        f.write(body)
    return len(body)


_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B", 10: "<Q", 11: "<q", 12: "<d"}


def read_metadata(path):
    """{key: value} of a GGUF file (for checks and tests; the engine's reader is the authority). Reads the header and
    the metadata only, never the tensor data."""
    with open(path, "rb") as f:
        def take(n):
            b = f.read(n)
            if len(b) != n:
                raise ValueError(f"{path}: truncated GGUF header")
            return b

        def unpack(fmt):
            return struct.unpack(fmt, take(struct.calcsize(fmt)))

        def value(t):
            if t == T_STRING:
                return take(unpack("<Q")[0]).decode("utf-8")
            if t == T_ARRAY:
                et, n = unpack("<IQ")
                return [value(et) for _ in range(n)]
            if t not in _SCALARS:
                raise ValueError(f"{path}: unknown GGUF metadata type {t}")
            v = unpack(_SCALARS[t])[0]
            return bool(v) if t == T_BOOL else v

        magic, ver, nt, nm = unpack("<IIQQ")
        if magic != GGUF_MAGIC or ver != GGUF_VERSION:
            raise ValueError(f"{path}: not a GGUF v3 file")
        meta = {}
        for _ in range(nm):
            key = take(unpack("<Q")[0]).decode("utf-8")
            meta[key] = value(unpack("<I")[0])
    return meta
