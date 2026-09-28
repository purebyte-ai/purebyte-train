"""The deterministic corpus writer shared by build_code_corpus.py and build_binary_corpus.py.

Same inputs -> same bytes. Nothing depends on the platform, the Python version, dictionary order or a random generator:

1. SELECTION. Each group has a declared share of the target size. Its candidates are visited in the order of a sha256
   key (`util.order_key`: seed, group, content id), and taken - after the builder's transform (UTF-8 check, exam
   exclusion, span choice) - until the group's quota is filled. A group that falls short is NOT topped up with another
   group: the shortfall is reported and the guard decides.
2. INTERLEAVING. Accepted pieces are dealt out by deficit (at each step the group furthest below its share goes next),
   so every prefix of the corpus - and every span the training stack cuts from it - has the declared mix.
3. VERIFICATION. The finished corpus is checked as a whole against every exam fingerprint set (a junction between two
   files could in theory form a shared block); pieces that touch a shared block are removed and the corpus is written
   again, until it is clean.
4. GUARDS. Real share vs declared share, distinct files per group, and homogeneity of the ten byte-deciles.

Outputs: <name>.bin, <name>.manifest.json (inputs with their pins, output sha256, groups, guard, per-exam exclusion
counts) and <name>.index.tsv.gz (offset, length, group, source, path, content id of every piece).
"""
import gzip
import io
import os
import platform
import shutil

import numpy as np

from .util import DataError, ensure_dir, human, log, order_key, sha256_file, write_json


def _data_path(path):
    """How a build manifest records one of its outputs: relative to $PUREBYTE_DATA when it is there (`corpora/...`),
    the absolute path otherwise (`--out` elsewhere), so that `--verify` finds what was really written."""
    from . import paths
    root, p = os.path.abspath(paths.data_dir()), os.path.abspath(path)
    try:
        rel = os.path.relpath(p, root)
    except ValueError:                                       # another drive on Windows
        return p.replace("\\", "/")
    return p.replace("\\", "/") if rel.startswith("..") else rel.replace("\\", "/")


class Candidate:
    __slots__ = ("key", "group", "source", "path", "size", "cid", "ref")

    def __init__(self, group, source, path, size, cid, ref, seed):
        self.group, self.source, self.path, self.size, self.cid, self.ref = group, source, path, size, cid, ref
        self.key = order_key(seed, group, cid)


def dedup(candidates):
    """One candidate per content id: the first by (source, path). Duplicates across sources are common (vendored
    copies, identical licence files) and would over-weight their content."""
    best = {}
    for c in candidates:
        cur = best.get(c.cid)
        if cur is None or (c.source, c.path) < (cur.source, cur.path):
            best[c.cid] = c
    return list(best.values()), len(candidates) - len(best)


def interleave(pieces, groups):
    """Deal pieces out by deficit. `pieces`: {group: [(length, payload)]} in acceptance order -> [(group, payload)]."""
    order = list(groups)
    total = sum(n for g in order for n, _ in pieces.get(g, []))
    target = {g: (sum(n for n, _ in pieces.get(g, [])) / total if total else 0.0) for g in order}
    queues = {g: list(pieces.get(g, [])) for g in order}
    heads = {g: 0 for g in order}
    placed = {g: 0 for g in order}
    carried, out = 0, []
    while True:
        live = [g for g in order if heads[g] < len(queues[g])]
        if not live:
            return out
        g = min(live, key=lambda g: ((placed[g] / carried if carried else 0.0) - target[g], order.index(g)))
        n, payload = queues[g][heads[g]]
        heads[g] += 1
        out.append((g, payload))
        placed[g] += n
        carried += n


def decile_deviation(seq, real):
    """Largest deviation of a group's share inside one byte-decile from its global share."""
    total = sum(n for _, n in seq)
    if not total:
        return 0.0
    dec = [dict() for _ in range(10)]
    acc = 0
    for g, n in seq:
        dec[min(9, int(acc * 10 // total))][g] = dec[min(9, int(acc * 10 // total))].get(g, 0) + n
        acc += n
    worst = 0.0
    for d in dec:
        by = sum(d.values())
        if by:
            for g, share in real.items():
                worst = max(worst, abs(d.get(g, 0) / by - share))
    return worst


class CorpusWriter:
    def __init__(self, name, out_dir, groups, target_bytes, seed=0, tolerance=0.03, decile_tolerance=0.03):
        """groups: ordered {name: {"share": float, "min_files": int}}."""
        self.name, self.out_dir, self.groups = name, out_dir, dict(groups)
        self.target, self.seed = int(target_bytes), seed
        self.tolerance, self.decile_tolerance = tolerance, decile_tolerance
        total_share = sum(g["share"] for g in self.groups.values())
        if abs(total_share - 1.0) > 1e-6:
            raise DataError(f"{name}: group shares add up to {total_share:.4f}, not 1")
        self.bin_path = os.path.join(out_dir, f"{name}.bin")
        self.index_path = os.path.join(out_dir, f"{name}.index.tsv.gz")
        self.manifest_path = os.path.join(out_dir, f"{name}.manifest.json")
        self.spool_dir = os.path.join(out_dir, f".{name}.spool")

    # ------------------------------------------------------------------ 1. selection
    def select(self, candidates, stream, transform, progress_every=5000):
        """candidates: [Candidate]; stream(pool) yields (candidate, raw bytes) in pool order (it may read ahead in
        batches; it is closed as soon as the quota is filled); transform(c, raw) -> (bytes or None, reason).
        Returns {group: [(length, (spool offset, candidate))]} and the counters."""
        if os.path.isdir(self.spool_dir):
            shutil.rmtree(self.spool_dir)
        ensure_dir(self.spool_dir)
        by_group = {g: [] for g in self.groups}
        for c in candidates:
            if c.group in by_group:
                by_group[c.group].append(c)
        pieces, stats = {}, {}
        for g, spec in self.groups.items():
            quota = int(self.target * spec["share"])
            pool = sorted(by_group[g], key=lambda c: (c.key, c.source, c.path))
            st = dict(quota=quota, candidates=len(pool), bytes=0, files=0, dropped={}, sources={})
            got = []
            feed = stream(pool)
            with open(os.path.join(self.spool_dir, g), "wb") as spool:
                for i, (c, raw) in enumerate(feed):
                    if st["bytes"] >= quota:
                        break
                    out, reason = transform(c, raw)
                    if out is None:
                        st["dropped"][reason] = st["dropped"].get(reason, 0) + 1
                        continue
                    off = spool.tell()
                    spool.write(out)
                    got.append((len(out), (off, c)))
                    st["bytes"] += len(out)
                    st["files"] += 1
                    src = st["sources"].setdefault(c.source, [0, 0])
                    src[0] += 1
                    src[1] += len(out)
                    if progress_every and (i + 1) % progress_every == 0:
                        log(f"    {g}: {i + 1} candidates seen, {human(st['bytes'])} of {human(quota)}")
            if hasattr(feed, "close"):
                feed.close()                                   # stop reading ahead: the quota is filled
            st["short"] = st["bytes"] < quota
            log(f"  {g:12s} {st['files']:7d} files {human(st['bytes']):>10s} of {human(quota):>10s}"
                + ("   SHORT: not enough candidates" if st["short"] else ""))
            pieces[g], stats[g] = got, st
        return pieces, stats

    # ------------------------------------------------------------------ 2+3. write and verify
    def _write(self, order):
        ensure_dir(self.out_dir)
        tmp = self.bin_path + ".tmp"
        spools = {g: open(os.path.join(self.spool_dir, g), "rb") for g in self.groups}
        rows, off = [], 0
        try:
            with open(tmp, "wb") as out:
                for g, (length, (spool_off, c)) in order:
                    spools[g].seek(spool_off)
                    out.write(spools[g].read(length))
                    rows.append((off, length, g, c.source, c.path, c.cid))
                    off += length
        finally:
            for f in spools.values():
                f.close()
        os.replace(tmp, self.bin_path)
        return rows

    def write(self, pieces, content_filter=None, max_repairs=4):
        """Interleave, write, verify against the exams and repair. Returns (rows, verification report)."""
        order = interleave({g: [(n, (n, p)) for n, p in pieces[g]] for g in self.groups}, self.groups)
        order = [(g, pl) for g, pl in order]
        verification = dict(rounds=0, pieces_removed=0, shared_regions=0)
        for _ in range(max_repairs + 1):
            rows = self._write(order)
            if content_filter is None or not content_filter.sets:
                return rows, verification
            verification["rounds"] += 1
            with open(self.bin_path, "rb") as f:
                data = f.read()
            regions = content_filter.verify_regions(data)
            del data
            if not regions:
                verification["shared_regions"] = 0
                return rows, verification
            verification["shared_regions"] = len(regions)
            starts = np.array([r[0] for r in rows], dtype=np.int64)
            bad = set()
            for a, z in regions:
                i = int(np.searchsorted(starts, a, side="right")) - 1
                while i < len(rows) and rows[i][0] < z:
                    bad.add(i)
                    i += 1
            log(f"  verification: {len(regions)} region(s) share a block with an exam; removing {len(bad)} piece(s)")
            verification["pieces_removed"] += len(bad)
            order = [o for i, o in enumerate(order) if i not in bad]
        raise DataError(f"{self.name}: the corpus still shares blocks with an exam after {max_repairs} repairs")

    def write_index(self, rows):
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as gz:
            gz.write(b"offset\tlength\tgroup\tsource\tpath\tcontent_id\n")
            for off, n, g, src, path, cid in rows:
                safe = path.replace("\t", " ").replace("\n", " ")
                gz.write(f"{off}\t{n}\t{g}\t{src}\t{safe}\t{cid}\n".encode("utf-8"))
        with open(self.index_path, "wb") as f:
            f.write(buf.getvalue())

    # ------------------------------------------------------------------ 4. guards and manifest
    def guards(self, rows, stats):
        total = sum(r[1] for r in rows) or 1
        real = {g: sum(r[1] for r in rows if r[2] == g) / total for g in self.groups}
        files = {g: sum(1 for r in rows if r[2] == g) for g in self.groups}
        failures = []
        dev_max = 0.0
        for g, spec in self.groups.items():
            d = abs(real[g] - spec["share"])
            dev_max = max(dev_max, d)
            if d > self.tolerance:
                failures.append(f"{g}: declared share {spec['share']:.3f}, real {real[g]:.3f}")
            if files[g] < spec.get("min_files", 0):
                failures.append(f"{g}: {files[g]} distinct files < {spec['min_files']}")
        dec = decile_deviation([(r[2], r[1]) for r in rows], real)
        if dec > self.decile_tolerance:
            failures.append(f"a byte-decile deviates {dec:.3f} from the global mix (> {self.decile_tolerance})")
        groups = {g: dict(share_declared=spec["share"], share_real=round(real[g], 5), bytes=stats[g]["bytes"],
                          files=files[g], candidates=stats[g]["candidates"], quota=stats[g]["quota"],
                          short=stats[g]["short"], dropped=dict(sorted(stats[g]["dropped"].items())),
                          sources={s: dict(files=v[0], bytes=v[1]) for s, v in sorted(stats[g]["sources"].items())})
                  for g, spec in self.groups.items()}
        return dict(ok=not failures, failures=failures, deviation_max=round(dev_max, 5),
                    decile_deviation_max=round(dec, 5), tolerance=self.tolerance,
                    decile_tolerance=self.decile_tolerance), groups

    def finish(self, rows, stats, manifest):
        """Index, guards and the build manifest. Returns the manifest dict (also written to disk)."""
        self.write_index(rows)
        guard, groups = self.guards(rows, stats)
        out_bytes = os.path.getsize(self.bin_path)
        import sys
        manifest = dict(manifest)
        manifest.update(
            name=self.name,
            output=dict(path=_data_path(self.bin_path), bytes=out_bytes, sha256=sha256_file(self.bin_path),
                        pieces=len(rows), index=dict(path=_data_path(self.index_path),
                                                     sha256=sha256_file(self.index_path))),
            target_bytes=self.target, seed=self.seed, groups=groups, guard=guard,
            environment=dict(python=platform.python_version(), numpy=np.__version__,
                             platform=sys.platform))
        write_json(self.manifest_path, manifest)
        shutil.rmtree(self.spool_dir, ignore_errors=True)
        return manifest
