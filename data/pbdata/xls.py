"""A minimal reader of legacy Excel workbooks (.xls: BIFF8 inside an OLE2 compound file), standard library only.

It reads what frequency tables need: every sheet's cells as numbers (NUMBER, RK, MULRK, cached FORMULA results) or
text (LABELSST through the shared string table, LABEL). Formatting, dates and formulas are ignored.

    read_xls(path) -> [(sheet name, {row: {col: value}})]     in workbook order
    row_values(cells, row, ncols) -> list                     missing cells are '' (as in xlrd)
"""
import struct

from .util import DataError

_SIG = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_END_OF_CHAIN = {0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC}


class _CompoundFile:
    def __init__(self, data):
        if data[:8] != _SIG:
            raise DataError("not an OLE2 compound file (.xls)")
        self.data = data
        self.sec = 1 << struct.unpack_from("<H", data, 30)[0]
        self.mini = 1 << struct.unpack_from("<H", data, 32)[0]
        n_fat, dir_start = struct.unpack_from("<II", data, 44)
        self.cutoff, mini_fat_start, n_mini_fat, difat_start, n_difat = struct.unpack_from("<IIIII", data, 56)
        difat = list(struct.unpack_from("<109I", data, 76))
        per = self.sec // 4 - 1
        sid = difat_start
        for _ in range(n_difat):
            vals = struct.unpack_from(f"<{per + 1}I", self._sector(sid))
            difat += vals[:per]
            sid = vals[per]
        self.fat = []
        for sid in difat[:n_fat]:
            self.fat += struct.unpack_from(f"<{self.sec // 4}I", self._sector(sid))
        self.entries = []
        d = self._chain(dir_start)
        for i in range(0, len(d) - 127, 128):
            e = d[i:i + 128]
            name_len = struct.unpack_from("<H", e, 64)[0]
            name = e[:max(0, name_len - 2)].decode("utf-16-le", "replace")
            start, size = struct.unpack_from("<IQ", e, 116)
            self.entries.append((name, e[66], start, size))
        root = self.entries[0]
        self.mini_stream = self._chain(root[2])[:root[3]] if root[3] else b""
        raw = self._chain(mini_fat_start) if n_mini_fat else b""
        self.mini_fat = list(struct.unpack_from(f"<{len(raw) // 4}I", raw)) if raw else []

    def _sector(self, sid):
        off = (sid + 1) * self.sec
        return self.data[off:off + self.sec]

    def _chain(self, start):
        out, sid, seen = [], start, set()
        while sid not in _END_OF_CHAIN and sid < len(self.fat) and sid not in seen:
            seen.add(sid)
            out.append(self._sector(sid))
            sid = self.fat[sid]
        return b"".join(out)

    def stream(self, name):
        for n, etype, start, size in self.entries:
            if etype == 2 and n == name:
                if size >= self.cutoff:
                    return self._chain(start)[:size]
                out, sid = [], start
                while sid not in _END_OF_CHAIN and sid < len(self.mini_fat):
                    out.append(self.mini_stream[sid * self.mini:(sid + 1) * self.mini])
                    sid = self.mini_fat[sid]
                return b"".join(out)[:size]
        raise DataError(f"the workbook has no `{name}` stream")


def _records(stream):
    pos = 0
    while pos + 4 <= len(stream):
        rtype, size = struct.unpack_from("<HH", stream, pos)
        yield rtype, stream[pos + 4:pos + 4 + size]
        pos += 4 + size


class _Segments:
    """The SST record and its CONTINUE records, read as one stream that knows where each record starts."""

    def __init__(self, parts):
        self.parts, self.i, self.pos = parts, 0, 0

    def read(self, n):
        out = b""
        while n > 0:
            while self.i < len(self.parts) and self.pos >= len(self.parts[self.i]):
                self.i += 1
                self.pos = 0
            if self.i >= len(self.parts):
                raise DataError("truncated shared string table")
            chunk = self.parts[self.i][self.pos:self.pos + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return out

    def read_chars(self, count, high):
        """`count` characters. When character data continues in the next record, that record starts with an option
        byte that says whether the rest is 8-bit or 16-bit."""
        chars = []
        while count > 0:
            if self.pos >= len(self.parts[self.i]):
                self.i += 1
                self.pos = 0
                if self.i >= len(self.parts):
                    raise DataError("truncated shared string table")
                high = self.parts[self.i][0] & 1
                self.pos = 1
            width = 2 if high else 1
            take = min(count, (len(self.parts[self.i]) - self.pos) // width)
            if take == 0:
                self.pos = len(self.parts[self.i])
                continue
            raw = self.parts[self.i][self.pos:self.pos + take * width]
            self.pos += take * width
            chars.append(raw.decode("utf-16-le" if high else "latin-1"))
            count -= take
        return "".join(chars)


def _sst(parts):
    seg = _Segments(parts)
    _, unique = struct.unpack("<II", seg.read(8))
    out = []
    for _ in range(unique):
        cch = struct.unpack("<H", seg.read(2))[0]
        flags = seg.read(1)[0]
        runs = struct.unpack("<H", seg.read(2))[0] if flags & 0x08 else 0
        ext = struct.unpack("<I", seg.read(4))[0] if flags & 0x04 else 0
        out.append(seg.read_chars(cch, flags & 1) if cch else "")
        if runs:
            seg.read(4 * runs)
        if ext:
            seg.read(ext)
    return out


def _rk(v):
    if v & 2:                                   # a 30-bit signed integer
        val = float(struct.unpack("<i", struct.pack("<I", v & 0xFFFFFFFC))[0] >> 2)
    else:                                       # the high 30 bits of a double
        val = struct.unpack("<d", struct.pack("<Q", (v & 0xFFFFFFFC) << 32))[0]
    return val / 100.0 if v & 1 else val


def _short_string(data):
    """A BIFF8 string with a 16-bit length: length, option byte, characters."""
    cch, flags = struct.unpack_from("<HB", data, 0)
    body = data[3:]
    return body[:cch * 2].decode("utf-16-le") if flags & 1 else body[:cch].decode("latin-1")


def _sheet_name(data):
    cch, flags = data[6], data[7]
    body = data[8:]
    return body[:cch * 2].decode("utf-16-le") if flags & 1 else body[:cch].decode("latin-1")


def read_xls(path):
    with open(path, "rb") as f:
        cf = _CompoundFile(f.read())
    try:
        wb = cf.stream("Workbook")
    except DataError:
        wb = cf.stream("Book")
    recs = list(_records(wb))
    sst, sheets, i = [], [], 0
    while i < len(recs):
        rtype, data = recs[i]
        if rtype == 0x00FC:                                  # SST, continued by CONTINUE records
            parts = [data]
            while i + 1 < len(recs) and recs[i + 1][0] == 0x003C:
                i += 1
                parts.append(recs[i][1])
            sst = _sst(parts)
        elif rtype == 0x0085:                                # BOUNDSHEET
            sheets.append((_sheet_name(data), struct.unpack_from("<I", data, 0)[0]))
        i += 1
    out = []
    for name, offset in sheets:
        cells = {}
        for rtype, data in _records(wb[offset:]):
            if rtype == 0x000A:                              # end of this sheet
                break
            if rtype == 0x00FD:                              # LABELSST
                r, c, _, k = struct.unpack_from("<HHHI", data, 0)
                cells.setdefault(r, {})[c] = sst[k]
            elif rtype == 0x0203:                            # NUMBER
                r, c, _, v = struct.unpack_from("<HHHd", data, 0)
                cells.setdefault(r, {})[c] = v
            elif rtype == 0x027E:                            # RK
                r, c, _, v = struct.unpack_from("<HHHI", data, 0)
                cells.setdefault(r, {})[c] = _rk(v)
            elif rtype == 0x00BD:                            # MULRK
                r, c0 = struct.unpack_from("<HH", data, 0)
                for j in range((len(data) - 6) // 6):
                    cells.setdefault(r, {})[c0 + j] = _rk(struct.unpack_from("<I", data, 4 + 6 * j + 2)[0])
            elif rtype == 0x0204:                            # LABEL
                r, c, _ = struct.unpack_from("<HHH", data, 0)
                cells.setdefault(r, {})[c] = _short_string(data[6:])
            elif rtype == 0x0006:                            # FORMULA: its cached number, if it is one
                r, c, _ = struct.unpack_from("<HHH", data, 0)
                if data[12:14] != b"\xff\xff":
                    cells.setdefault(r, {})[c] = struct.unpack_from("<d", data, 6)[0]
        out.append((name, cells))
    return out


def ncols(cells):
    return 1 + max((c for row in cells.values() for c in row), default=-1)


def row_values(cells, row, n):
    r = cells.get(row, {})
    return [r.get(c, "") for c in range(n)]
