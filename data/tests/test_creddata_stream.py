"""The CredData DEV stream (pbdata/creddata_stream.py, build_creddata_stream.py) and the CredData helpers
(pbdata/creddata.py), offline: a synthetic CredData of six small repositories stands in for the real one. Its values are
look-alikes generated at test time; nothing is downloaded."""
import collections
import csv
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import build_creddata_stream as B
from helpers import git_repo
from pbdata import creddata, manifest, paths
from pbdata import creddata_stream as CS
from pbdata.util import DataError, write_json

FIELDS = ["Id", "FileID", "Domain", "RepoName", "FilePath", "LineStart", "LineEnd", "GroundTruth", "ValueStart",
          "ValueEnd", "CryptographyKey", "PredefinedPattern", "Category"]
SMALL = dict(n=400, seq_len=128, seed=0, replant=0.5)


def repo_ids(subset, k):
    """k repository ids (8 hex, like CredData's) that fall in one DEV subset."""
    out, i = [], 1
    while len(out) < k:
        rid = f"{i * 2654435761 % 2 ** 32:08x}"
        if creddata.dev_subset(rid) == subset and rid not in CS.EXCLUDED_REPOS:
            out.append(rid)
        i += 1
    return out


class Syn:
    """A synthetic CredData checkout: meta/<repo>.csv and data/<repo>/src/<FileID>.py."""

    def __init__(self, root, seed=0):
        self.root = root
        self.rng = np.random.default_rng(seed)
        self.train = repo_ids("train", 2)
        self.select = repo_ids("select", 1)
        self.heldout = repo_ids("heldout", 1)
        self.test = ["fe0d0001", "fe0d0002"]
        self.split = {r: "dev" for r in self.train + self.select + self.heldout}
        self.split.update({r: "test" for r in self.test})
        self.rows = collections.defaultdict(list)
        self.labelled = set()          # values the stream labels (the whole URL for the URL credential)
        self.never = set()             # bytes that must never reach a window

    def word(self, n=24):
        return "zq" + "".join(self.rng.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), n))

    def uuid(self):
        return "-".join("".join(self.rng.choice(list("0123456789abcdef"), k)) for k in (8, 4, 4, 4, 12))

    def file(self, repo, k, units):
        """units: [(line text, GroundTruth, (ValueStart, ValueEnd) or None, Category)], one every 12 lines."""
        fid = f"{k:02x}{repo[:6]}"
        path = f"data/{repo}/src/{fid}.py"
        lines = [f"setting_{i} = {int(self.rng.integers(0, 10 ** 6))}  # marker {repo} f{k}" for i in range(48)]
        for j, (text, gt, pos, cat) in enumerate(units):
            ln = 8 + 12 * j
            lines[ln - 1] = text
            vs, ve = pos if pos else ("", "")
            self.rows[repo].append(dict(Id=str(len(self.rows[repo]) + 1), FileID=fid, Domain="GitHub", RepoName=repo,
                                        FilePath=path, LineStart=str(ln), LineEnd=str(ln), GroundTruth=gt,
                                        ValueStart=str(vs), ValueEnd=str(ve), CryptographyKey="",
                                        PredefinedPattern="", Category=cat))
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(("\n".join(lines) + "\n").encode())
        return p

    @staticmethod
    def assign(name, value):
        text = f'{name} = "{value}"'
        start = len(f'{name} = "')
        return text, (start, start + len(value))

    @staticmethod
    def unit(name, value, label, category):
        """A labelled `name = "value"` line, with the value's position."""
        text, pos = Syn.assign(name, value)
        return text, label, pos, category

    def write_meta(self):
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        for repo in self.split:
            with open(self.root / "meta" / f"{repo}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, FIELDS)
                w.writeheader()
                w.writerows(self.rows[repo])


@pytest.fixture
def syn(tmp_path):
    s = Syn(tmp_path / "CredData")
    a, b = s.train
    token, pw, u1, u2 = s.word(), s.word(10), s.uuid(), s.uuid()
    shared, xtok, pw2, sec = s.word(), s.word(), s.word(12), s.word()
    url = f"postgres://svc:{pw}@db.internal.invalid:5432/app"
    s.file(a, 0, [s.unit("api_token", token, "T", "Token"), s.unit("user_name", s.word(), "F", "Other")])
    t, (vs, _) = s.assign("db_url", url)
    s.file(a, 1, [(t, "T", (vs + url.index(pw), vs + url.index(pw) + len(pw)), "URL Credentials"),
                  s.unit("note", xtok, "X", "Other")])
    s.file(a, 2, [s.unit("request_id", u1, "T", "UUID"),                  # a UUID under an identifier's name
                  s.unit("api_key", u2, "T", "UUID"),                     # ... and under a credential name
                  s.unit("secret_key", shared, "T", "Secret")])           # the value of a TEST credential
    s.file(b, 0, [s.unit("db_password", pw2, "T", "Password"), s.unit("password_hint", s.word(), "F", "Password")])
    s.file(b, 1, [(f"client_secret: {sec}", "T", None, "Secret")])       # a T row without a position
    for repo in s.select + s.heldout + s.test[1:]:
        v = s.word()
        s.file(repo, 0, [s.unit("api_token", v, "T", "Token")])
        s.never |= {v, f"marker {repo}"}
    s.file(s.test[0], 0, [s.unit("secret_key", shared, "T", "Secret")])
    s.never |= {f"marker {s.test[0]}", u1, shared, xtok}
    s.labelled |= {token, url, u2, pw2, sec}
    s.u2 = u2
    s.write_meta()
    return s


@pytest.fixture
def generator(data_home):
    """The secrets generator with (empty) catalogues from the test's $PUREBYTE_DATA; its state is restored after."""
    src = os.path.join(paths.REPO, "specialists", "secrets-code")
    if src not in sys.path:
        sys.path.insert(0, src)
    from secretsgen import catalog as C
    keep = {k: getattr(C, k) for k in ("_DIR", "_CAT", "_CAT_TH", "RULES", "FPS_CURATED", "WORDS", "PROVIDERS",
                                       "EXAMPLES_TH")}
    missing, compiled = list(C._MISSING), dict(C._COMPILED)
    cat = data_home / "catalogs"
    cat.mkdir()
    (cat / "gitleaks.json").write_text(json.dumps({"rules": [], "tps": [], "fps": []}))
    (cat / "trufflehog.json").write_text(json.dumps({"providers": [], "patterns": [], "examples": []}))
    G, C2 = CS.load_generator(str(data_home))
    yield G, C2
    for k, v in keep.items():
        setattr(C, k, v)
    C._MISSING[:] = missing
    C._COMPILED.clear()
    C._COMPILED.update(compiled)


def build(syn, G, **params):
    keyed = CS.keyed_hasher()
    hashes, info = CS.test_value_hashes(str(syn.root), syn.split, keyed)
    macro = creddata.load_scorer().macro_category
    return CS.build(str(syn.root), syn.split, G, macro, hashes, keyed, **dict(SMALL, **params))


def label_runs(x, s):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], s.astype(np.int8), [0]])))
    return [bytes(x[a:z]) for a, z in zip(edges[::2], edges[1::2])]


# ------------------------------------------------------------------------------------------------ the stream
def test_only_dev_train_reaches_the_windows(syn, generator):
    arrays, meta = build(syn, generator[0])
    blob = arrays["X"].tobytes()
    for v in syn.never:
        assert v.encode() not in blob                          # TEST, DEV select / heldout, X lines, ignored UUIDs
    assert meta["repos"] == sorted(syn.train) and meta["windows"] == SMALL["n"]
    c = meta["counts"]
    assert c["T_units_hash_in_TEST"] == 1                      # the value shared with a TEST repository
    assert c["X_units"] == 1 and c["uuid_ignored_not_credential_name"] == 1
    assert c["uuid_positive_credential_name"] >= 1 and c["url_widened"] >= 1


def test_meta_files_of_other_repositories_are_never_opened(syn, generator):
    for repo in syn.select + syn.heldout:                     # unreadable: opening one would raise
        (syn.root / "meta" / f"{repo}.csv").write_bytes(b"\xff\xfe\x00 not a csv")
    build(syn, generator[0])
    for repo in syn.test:
        (syn.root / "meta" / f"{repo}.csv").write_bytes(b"\xff\xfe\x00 not a csv")
    rows, repos = CS.dev_train_rows(str(syn.root / "meta"), syn.split)
    assert repos == sorted(syn.train) and {r["RepoName"] for r in rows} == set(syn.train)


def test_labels_are_the_values_and_the_contrast_is_negative(syn, generator):
    arrays, _ = build(syn, generator[0])
    X, S, K = arrays["X"], arrays["S"], arrays["kind"]
    known = [v.encode() for v in syn.labelled]
    part_of_known = lambda r: any(r in v for v in known)  # noqa: E731  (a span can be clipped at a window's edge)
    assert np.array_equal(arrays["Y"].astype(bool), S.any(1))
    seen = collections.Counter()
    for x, s, k in zip(X, S, K):
        runs = label_runs(x, s)
        seen[int(k)] += 1
        if k == 0:
            assert runs and all(part_of_known(r) for r in runs)
        elif k == 1:
            assert any(not part_of_known(r) for r in runs)     # a generated value, labelled
        elif k == 2:
            assert all(part_of_known(r) for r in runs)         # only the other real values: the hole is negative
        else:
            assert not runs
    assert all(seen[k] > 0 for k in range(5)) and seen[1] == seen[2]
    whole = [label_runs(x, s) for x, s, k in zip(X, S, K) if k == 0]
    assert any(syn.u2.encode() in r for rs in whole for r in rs)                        # a UUID under `api_key`
    assert any(r.startswith(b"postgres://") and r.endswith(b"/app") for rs in whole for r in rs)   # the whole URL


def test_same_inputs_same_stream(syn, generator):
    d1 = CS.array_digests(build(syn, generator[0])[0])
    d2 = CS.array_digests(build(syn, generator[0])[0])
    d3 = CS.array_digests(build(syn, generator[0], seed=1)[0])
    assert d1 == d2 and d1["X"] != d3["X"]


def test_the_generator_switch_does_not_leak_into_the_stream(syn, generator):
    G = generator[0]
    d1 = CS.array_digests(build(syn, G)[0])
    old = G.INVENTED_PREFIXES
    G.INVENTED_PREFIXES = 0.9
    try:
        d2 = CS.array_digests(build(syn, G)[0])
    finally:
        G.INVENTED_PREFIXES = old
    assert d1 == d2


def test_hard_check(syn, generator, tmp_path):
    a, s = syn.train[0], syn.select[0]
    for repo in (syn.test[0], s, syn.heldout[0]):
        with pytest.raises(DataError):
            CS.assert_dev_train(repo, syn.split)
    with pytest.raises(DataError):
        CS.assert_dev_train(a, syn.split, f"data/{syn.train[1]}/src/x.py")    # a file of another repository
    with pytest.raises(DataError):
        CS.assert_dev_train("0000beef", syn.split)                          # not in the split
    arrays, meta = build(syn, generator[0])
    out = str(tmp_path / "stream.npz")
    B.write_npz(out, arrays, meta)
    assert CS.check_file(out, syn.split)["repos"] == sorted(syn.train)
    for bad in (dict(meta, repos=meta["repos"] + [syn.test[0]]),            # a TEST repository recorded
                dict(meta, split=dict(meta["split"], sha256="0" * 64))):    # built on another split
        B.write_npz(out, arrays, bad)
        with pytest.raises(DataError):
            CS.check_file(out, syn.split)
    B.write_npz(out, dict(arrays, Y=np.zeros_like(arrays["Y"])), meta)      # Y is not S.any(1)
    with pytest.raises(DataError):
        CS.check_file(out, syn.split)


def test_a_missing_file_stops_the_build(syn, generator):
    next((syn.root / "data" / syn.train[1] / "src").iterdir()).unlink()
    with pytest.raises(DataError, match="missing"):
        build(syn, generator[0])


def test_the_trainer_reads_it(syn, generator, tmp_path):
    """`train.real_mode: spans` (pbtrain/train/streams.py) accepts the file as written."""
    pytest.importorskip("torch")
    paths._training_on_path()
    from pbtrain.train.streams import load_stream
    arrays, meta = build(syn, generator[0])
    out = str(tmp_path / "stream.npz")
    B.write_npz(out, arrays, meta)
    r = load_stream(out, 0.25, "spans", 1.0, 16, SMALL["seq_len"], 5, 0, False)
    assert r["n"] == 4 and len(r["tag"]) == SMALL["n"] and int(r["tag"].max()) < 5


def test_exam_overlap_report_and_drop(syn, generator, data_home, F):
    arrays, meta = build(syn, generator[0])
    fdir = data_home / "exams" / "fingerprints"
    fdir.mkdir(parents=True)
    exam = next((syn.root / "data" / syn.train[0] / "src").iterdir()).read_bytes()      # a DEV file is also an exam
    fp = F.fingerprints(exam)
    F.save(str(fdir / "synthetic-exam.npz"), fp, dict(name="synthetic-exam", role="test"))
    F.save(str(fdir / "synthetic-dev.npz"), fp, dict(name="synthetic-dev", role="dev"))
    a1, m1, r1 = B.exam_overlap(arrays, meta, "report", allow_missing=True)
    hits = r1["checked"]["synthetic-exam"]["windows"]
    assert hits > 0 and len(a1["X"]) == SMALL["n"] and "synthetic-dev" not in r1["checked"]
    a2, m2, r2 = B.exam_overlap(arrays, meta, "drop", allow_missing=True)
    assert r2["dropped"] == hits and len(a2["X"]) == SMALL["n"] - hits == m2["windows"]
    assert not F.window_hits(a2["X"], fp).any() and np.array_equal(a2["Y"].astype(bool), a2["S"].any(1))


def test_output_stays_out_of_the_repository(data_home):
    with pytest.raises(DataError):
        B.output_path(os.path.join(paths.REPO, "data", "stream.npz"))
    assert B.output_path(None).startswith(str(data_home))


def test_released_record_is_the_pinned_split():
    m, entry, parts = creddata.yaml_partitions()
    split = creddata.pinned_split(parts)
    assert creddata.split_digest(split) == CS.RELEASED["split_sha256"]
    train = [r for r, p in split.items() if p == "dev" and creddata.dev_subset(r) == "train"]
    assert len(split) == 337 and len(train) == 117 and not set(train) & set(CS.EXCLUDED_REPOS)
    assert CS.RELEASED["params"] == CS.DEFAULTS


def test_compare_with_released_names_the_reason():
    same, reasons = CS.compare_with_released({}, dict(CS.DEFAULTS, seed=1), {}, dict(rules=120, compiled=96, sha256=""))
    assert not same and len(reasons) == 3 and "Python 3.14" in reasons[2]


# ------------------------------------------------------------------------------------------------ CredData helpers
def test_units_agree_with_the_public_scorer(syn):
    rows = creddata.read_meta(str(syn.root / "meta"))
    ours = creddata.units(rows)
    theirs = creddata.load_scorer().CredData._units(rows)
    assert ours.keys() == theirs.keys()
    assert all(ours[k]["label"] == theirs[k]["label"] and ours[k]["cats"] == theirs[k]["cats"]
               and ours[k]["repo"] == theirs[k]["repo"] for k in ours)


def test_line_form():
    s = Syn(pathlib.Path("."), seed=3)
    pw, tok = s.word(10), s.word()
    key = ("data/r/x.py", 1, 1)
    text, (a, z) = Syn.assign("db_password", pw)
    form, value = creddata.line_form([text], key, [dict(ValueStart=str(a), ValueEnd=str(z))])
    assert value == pw and form["name"] == "db_password" and form["name_kind"] == "password"
    assert form["syntax"] == "named_quoted" and form["n"] == len(pw)
    form, value = creddata.line_form([f'mode = "{s.word(6)}", token = {tok}'], key, [dict(ValueStart="", ValueEnd="")])
    assert value == tok and form["name"] == "token" and form["syntax"] == "named_unquoted"   # the credential name wins
    url = f"https://svc:{pw}@host.invalid/x"
    line = f'endpoint = "{url}"'
    a = line.index(pw)
    form, value = creddata.line_form([line], key, [dict(ValueStart=str(a), ValueEnd=str(a + len(pw)))])
    assert value == pw and form["syntax"] == "url_userinfo"
    long_name = "service_" * 6 + "api_key"
    form, _ = creddata.line_form([f"{long_name} = {tok}"], key, [dict(ValueStart="", ValueEnd="")])
    assert form["name"] == long_name[-32:]
    assert creddata.value_shape(s.uuid()) == "uuid" and creddata.value_shape("${DB_PASSWORD}") == "placeholder/template"


def test_file_scope_and_repository_ids():
    words = ["test", "mock", "/src", "code"]
    assert creddata.file_scope("src/app/settings", words) == "/src/"
    assert creddata.file_scope("tests/unit/mock_client", words) == "/test/mock/"
    assert creddata.file_scope("lib/util", words) == "/_/"
    assert creddata.new_repo_id("00" * 24) == f"{__import__('binascii').crc32(bytes(24)):08x}"
    assert creddata.is_licence("LICENSE.md") and creddata.is_licence("COPYING") and not creddata.is_licence("src")
    assert not creddata.is_licence("licensemanager")


def test_partial_fetch_takes_exactly_the_labelled_files(tmp_path):
    words = ["test", "mock", "/src", "code"]
    files = {"src/app/settings.py": b"setting = 1\r\n", "tests/unit/mock_client.py": b"client = None\n",
             "src/other.py": b"other = 2\n", "lib/unlabelled.py": b"x = 3\n", "COPYING": b"licence text\n",
             "docs/mixes/LICENSE": b"mixed licence\n"}
    up, commit = git_repo(str(tmp_path / "upstream"), files)
    for k in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        subprocess.run(["git", "config", k, "true"], cwd=up, check=True, capture_output=True)
    sid = commit + "0badc0de"                                  # a snapshot.json key: commit + suffix
    rid = creddata.new_repo_id(sid)
    fid = lambda p: hashlib.sha256(p.encode()).hexdigest()[:8]  # noqa: E731

    def dest(p):
        stem, ext = os.path.splitext(p)
        return f"data/{rid}{creddata.file_scope(stem, words)}{fid(p)}{ext.lower()}"

    wanted = {fid("src/app/settings.py"): dest("src/app/settings.py"),
              fid("tests/unit/mock_client.py"): dest("tests/unit/mock_client.py"),
              fid("src/other.py"): f"data/{rid}/_/{fid('src/other.py')}.py"}          # another scope: not taken
    root = tmp_path / "CredData"
    st = creddata.fetch_repository(str(root), sid, pathlib.Path(up).as_uri(), wanted, words, str(tmp_path / "scratch"))
    assert st["error"] == "" and st["got"] == 2 and st["mismatched"] == 1 and st["licences"] == 2
    for p in ("src/app/settings.py", "tests/unit/mock_client.py"):
        assert (root / dest(p)).read_bytes() == files[p]          # the committed bytes (no line-ending conversion)
    assert (root / "data" / rid / "COPYING").read_bytes() == files["COPYING"]
    assert (root / "data" / rid / "LICENSE").read_bytes() == files["docs/mixes/LICENSE"]
    written = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file()}
    assert written == {dest("src/app/settings.py"), dest("tests/unit/mock_client.py"), f"data/{rid}/COPYING",
                       f"data/{rid}/LICENSE"}
    assert not (tmp_path / "scratch" / rid).exists()
    assert creddata.missing_files(str(root), {rid: wanted}) == {rid: 1}


def test_meta_paths_cannot_leave_the_checkout(tmp_path):
    for bad in ("data/../x.py", "/etc/x", "meta/x.csv", "data//x.py"):
        with pytest.raises(DataError):
            creddata.data_path(str(tmp_path), bad)


def test_obfuscation_runs_once_and_never_shows_a_value(tmp_path, monkeypatch):
    root, base = tmp_path / "CredData", tmp_path
    (root / "data").mkdir(parents=True)
    (root / "data" / "f.txt").write_bytes(b"abc\n")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "base62.py").write_text("")                         # CredData's requirement, faked
    monkeypatch.setenv("PYTHONPATH", str(lib))
    monkeypatch.setattr(creddata, "TEXT_MODE_WRITES_CRLF", False)
    (root / "obfuscate_creds.py").write_text(
        "import sys\nopen('data/f.txt', 'ab').write(b'x')\nsys.stderr.write('The same value: zqhidden\\n')\n")
    st = creddata.obfuscate(str(root), str(base))
    assert st["status"] == "done" and st["values_left_identical"] == 1
    creddata.obfuscate(str(root), str(base))                    # done: never twice
    assert (root / "data" / "f.txt").read_bytes() == b"abc\nx"
    write_json(str(base / creddata.OBFUSCATION_STATE), dict(status="running"))
    with pytest.raises(DataError, match="interrupted"):
        creddata.obfuscate(str(root), str(base))
    os.remove(base / creddata.OBFUSCATION_STATE)
    (root / "obfuscate_creds.py").write_text(
        "import sys\nsys.stderr.write('The same value: zqhidden\\nboom\\n')\nsys.exit(3)\n")
    with pytest.raises(DataError) as e:
        creddata.obfuscate(str(root), str(base))
    assert "boom" in str(e.value) and "zqhidden" not in str(e.value)
    monkeypatch.setattr(creddata, "TEXT_MODE_WRITES_CRLF", True)
    os.remove(base / creddata.OBFUSCATION_STATE)
    with pytest.raises(DataError, match="Windows"):
        creddata.obfuscate(str(root), str(base))


def test_official_output_is_captured_and_never_shows_a_value(tmp_path):
    """download_data.py ends with obfuscate_creds.py, which logs every value it leaves unchanged: only progress lines
    are passed on, the value lines are counted, and a failure shows only lines that cannot hold a value."""
    script = tmp_path / "fake_download.py"
    script.write_text(
        "import sys\n"
        "print('2026-09-25 | INFO | download_data.py:103 | Downloaded: 1/2', flush=True)\n"
        "print('Cloning into tmp/x...', flush=True)\n"
        "sys.stderr.write('2026-09-25 | WARNING | obfuscate_creds.py:240 | The same value: zqhidden, row\\n')\n"
        "sys.stderr.write('a line that repeats a value: zqhidden\\n')\n"
        "print('2026-09-25 | INFO | download_data.py:270 | Done! All files saved to data', flush=True)\n"
        "sys.exit(int(sys.argv[1]))\n", encoding="utf-8")
    shown = []
    code, same, tail = creddata.run_captured([sys.executable, str(script), "0"], str(tmp_path), show=shown.append)
    assert code == 0 and same == 1
    assert shown == ["  CredData: Downloaded: 1/2", "  CredData: Done! All files saved to data"]
    code, same, tail = creddata.run_captured([sys.executable, str(script), "3"], str(tmp_path), show=shown.append)
    assert code == 3 and "zqhidden" not in " ".join(shown + tail) and "Cloning into tmp/x..." in tail


@pytest.mark.skipif(os.name == "nt", reason="CredData's download script runs on Linux, macOS or WSL only")
def test_the_official_method_runs_captured(tmp_path, monkeypatch, capsys):
    import fetch_creddata
    root, base = tmp_path / "CredData", tmp_path
    root.mkdir()
    (root / "obfuscate_creds.py").write_text("", encoding="utf-8")
    (root / "download_data.py").write_text(
        "import os, sys\nos.makedirs('data')\nsys.stderr.write('The same value: zqhidden, row\\n')\n", encoding="utf-8")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "base62.py").write_text("")
    monkeypatch.syspath_prepend(str(lib))
    fetch_creddata.run_official_download(str(root), str(base), 1)
    st = creddata.obfuscation_state(str(base))
    assert st["status"] == "done" and st["method"] == "official" and st["values_left_identical"] == 1
    assert "zqhidden" not in capsys.readouterr().out


def test_identity_overlap_report_and_drop():
    """Windows of CredData DEV repositories that are also repositories of another evaluation set are reported, and
    left out with `drop` (the identity rule: any commit of an evaluation repository is that repository)."""
    meta = dict(repos=["aaaa0001", "aaaa0002"], windows=6)
    arrays = dict(X=np.zeros((6, 8), np.uint8), S=np.zeros((6, 8), np.uint8), Y=np.array([1, 0, 1, 0, 0, 1], np.uint8),
                  kind=np.array([0, 3, 1, 2, 4, 0], np.uint8), repo=np.array([0, 0, 1, 1, 1, 0], np.int16))
    parts = {"dev": {"aaaa0001": dict(url="https://github.com/Example/Tool"),
                     "aaaa0002": dict(url="https://github.com/example/other")}}
    exam_repos = {"github.com/example/tool": ["code-fp-z"]}
    a, m, rep = B.identity_overlap(arrays, meta, parts, exam_repos, "report")
    assert rep["windows"] == 3 and rep["positive_windows"] == 2 and len(a["X"]) == 6
    assert rep["repositories"]["aaaa0001"]["exam_sets"] == ["code-fp-z"]
    a, m, rep = B.identity_overlap(arrays, meta, parts, exam_repos, "drop")
    assert rep["dropped"] == 3 and len(a["X"]) == 3 and m["windows"] == 3 and m["exam_identity_dropped"] == 3
    assert set(a["repo"].tolist()) == {1} and m["repos_used"] == 1


def test_five_code_fp_repositories_are_creddata_dev_train_repositories():
    """The overlap data/README.md discloses: five repositories of code-fp-a/b/c are CredData DEV repositories of the
    `train` subset (the stream's), and three more of code-fp-c are CredData TEST repositories."""
    from pbdata.identity import norm_repo
    m, entry, parts = creddata.yaml_partitions()
    exam_repos = B.exam_repositories(manifest.load_exams())
    dev = {rid for rid, r in parts["dev"].items() if norm_repo(r["url"]) in exam_repos}
    test = {rid for rid, r in parts["test"].items() if norm_repo(r["url"]) in exam_repos}
    assert dev == {"f623c7b3", "3b7cbdd3", "be221a57", "349ac2b1", "6e2ed0e4"}
    assert all(creddata.dev_subset(r) == "train" for r in dev)
    assert len(test) == 3
    assert {exam_repos[norm_repo(parts["dev"][r]["url"])][0] for r in dev} == {"code-fp-a", "code-fp-b", "code-fp-c"}
