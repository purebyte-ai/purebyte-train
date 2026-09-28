"""Path patterns for `take:` blocks.

    *        any run of characters except `/`
    **       any run of characters including `/` (`**/` also matches nothing: `**/x` matches `x` and `a/b/x`)
    ?        one character except `/`
    [abc]    a character class

A pattern without `/` is matched against the file name only (`*.py`, `Dockerfile`); a pattern with `/` against the
whole POSIX path relative to the source root (`src/**/*.c`). Matching is case-sensitive.
"""
import functools
import re


@functools.lru_cache(maxsize=4096)
def _compile(pattern):
    out, i, n = [], 0, len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i:i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if pattern[i:i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pattern.find("]", i + 1)
            if j < 0:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1:j]
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append("[" + body.replace("\\", "\\\\") + "]")
                i = j
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("(?s:" + "".join(out) + r")\Z")


def match(path, pattern):
    path = path.replace("\\", "/")
    target = path if "/" in pattern else path.rsplit("/", 1)[-1]
    return _compile(pattern).match(target) is not None


def match_any(path, patterns):
    return any(match(path, p) for p in (patterns or ()))


def selected(path, include=None, exclude=None):
    """True when `path` matches an include pattern (all paths when `include` is empty) and no exclude pattern."""
    if include and not match_any(path, include):
        return False
    return not match_any(path, exclude)
