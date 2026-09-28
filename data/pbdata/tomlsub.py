"""TOML for Python < 3.11: `tomllib` when it exists, otherwise a reader of the subset that tool configuration files
such as gitleaks.toml use (tables, arrays of tables, strings of the four kinds, integers, floats, booleans, arrays).
"""
import re


def loads(text):
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        return _Parser(text).parse()


_ESC = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}


class _Parser:
    def __init__(self, text):
        self.s, self.i = text, 0

    def err(self, msg):
        line = self.s.count("\n", 0, self.i) + 1
        raise ValueError(f"TOML line {line}: {msg}")

    def ws(self, newlines=False):
        while self.i < len(self.s):
            c = self.s[self.i]
            if c in " \t" or (newlines and c in "\r\n"):
                self.i += 1
            elif c == "#":
                while self.i < len(self.s) and self.s[self.i] != "\n":
                    self.i += 1
            else:
                break

    def parse(self):
        root, cur = {}, None
        cur = root
        while True:
            self.ws(newlines=True)
            if self.i >= len(self.s):
                return root
            if self.s.startswith("[[", self.i):
                self.i += 2
                keys = self.keypath("]]")
                cur = self._table(root, keys[:-1])
                cur = cur.setdefault(keys[-1], [])
                cur.append({})
                cur = cur[-1]
            elif self.s[self.i] == "[":
                self.i += 1
                keys = self.keypath("]")
                cur = self._table(root, keys)
            else:
                keys = self.keypath("=")
                self.ws()
                tbl = self._table(cur, keys[:-1])
                tbl[keys[-1]] = self.value()
                self.ws()
                if self.i < len(self.s) and self.s[self.i] not in "\r\n":
                    self.err("expected the end of the line")

    def _table(self, base, keys):
        t = base
        for k in keys:
            t = t.setdefault(k, {})
            if isinstance(t, list):
                t = t[-1]
        return t

    def keypath(self, end):
        keys = []
        while True:
            self.ws()
            if self.s[self.i] in "\"'":
                keys.append(self.string())
            else:
                m = re.compile(r"[A-Za-z0-9_-]+").match(self.s, self.i)
                if not m:
                    self.err("bad key")
                keys.append(m.group(0))
                self.i = m.end()
            self.ws()
            if self.s.startswith(end, self.i):
                self.i += len(end)
                return keys
            if self.s[self.i] == ".":
                self.i += 1
                continue
            self.err(f"expected `{end}`")

    def string(self):
        s = self.s
        for q in ('"""', "'''"):
            if s.startswith(q, self.i):
                j = s.find(q, self.i + 3)
                while j >= 0 and s.startswith(q[0], j + 3):     # up to two quotes may end the content
                    j += 1
                raw = s[self.i + 3:j]
                self.i = j + 3
                if raw.startswith("\r\n"):
                    raw = raw[2:]
                elif raw.startswith("\n"):
                    raw = raw[1:]
                return raw if q == "'''" else self._unescape(raw, multiline=True)
        q = s[self.i]
        j = self.i + 1
        if q == "'":
            k = s.index("'", j)
            self.i = k + 1
            return s[j:k]
        out = []
        while s[j] != '"':
            if s[j] == "\\":
                out.append(s[j:j + 2] if s[j + 1] not in "uU" else s[j:j + (6 if s[j + 1] == "u" else 10)])
                j += len(out[-1])
            else:
                out.append(s[j])
                j += 1
        self.i = j + 1
        return self._unescape("".join(out))

    def _unescape(self, raw, multiline=False):
        if multiline:
            raw = re.sub(r"\\[ \t]*\r?\n\s*", "", raw)
        out, j = [], 0
        while j < len(raw):
            c = raw[j]
            if c == "\\":
                n = raw[j + 1]
                if n in _ESC:
                    out.append(_ESC[n])
                    j += 2
                elif n == "u":
                    out.append(chr(int(raw[j + 2:j + 6], 16)))
                    j += 6
                elif n == "U":
                    out.append(chr(int(raw[j + 2:j + 10], 16)))
                    j += 10
                else:
                    self.err(f"bad escape \\{n}")
            else:
                out.append(c)
                j += 1
        return "".join(out)

    def value(self):
        s = self.s
        c = s[self.i]
        if c in "\"'":
            return self.string()
        if c == "[":
            self.i += 1
            arr = []
            while True:
                self.ws(newlines=True)
                if s[self.i] == "]":
                    self.i += 1
                    return arr
                arr.append(self.value())
                self.ws(newlines=True)
                if s[self.i] == ",":
                    self.i += 1
        if c == "{":
            self.i += 1
            tbl = {}
            while True:
                self.ws()
                if s[self.i] == "}":
                    self.i += 1
                    return tbl
                keys = self.keypath("=")
                self.ws()
                self._table(tbl, keys[:-1])[keys[-1]] = self.value()
                self.ws()
                if s[self.i] == ",":
                    self.i += 1
        m = re.compile(r"true|false|[+-]?(?:\d[\d_]*)(?:\.\d[\d_]*)?(?:[eE][+-]?\d+)?").match(s, self.i)
        if not m:
            self.err("unsupported value")
        self.i = m.end()
        tok = m.group(0)
        if tok in ("true", "false"):
            return tok == "true"
        tok = tok.replace("_", "")
        return float(tok) if any(ch in tok for ch in ".eE") else int(tok)
