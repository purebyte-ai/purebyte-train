"""The benchmark's scorer (benchmarks/piimb/score.py), on made-up sentences: the leaderboard's character metric (merged
intervals, micro per task, the mean of the six tasks), byte-to-character spans, the CLI's report, the reference points
and the exam metrics the pii recipe's criteria read. No benchmark file is needed."""
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(scope="module")
def sc():
    spec = importlib.util.spec_from_file_location("piimb_score", os.path.join(REPO, "benchmarks", "piimb", "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(tmp_path, sc):
    rows = []
    for i, task in enumerate(sc.TASKS):
        for j in range(2):
            text = f"Kalo Mire sent {j} items to Ruvo Satize on day {i}."
            rows.append(dict(uid=f"{task}:{j}", task_name=task, text=text,
                             entities=[dict(start=0, end=9, label="NAME"), dict(start=len(text) - 25,
                                                                               end=len(text) - 14, label="NAME")]))
    p = tmp_path / "test_sentences.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(tmp_path), sc.load(str(tmp_path), check=False)


def test_the_metric_merges_and_micro_averages(tmp_path, sc):
    _, rows = _rows(tmp_path, sc)
    exact = {n: list(r["gold"]) for n, r in enumerate(rows)}
    res = sc.score(rows, exact, chars=True)
    assert all(res[t]["f2"] == 1.0 and res[t]["precision"] == 1.0 for t in sc.TASKS)
    assert res["_mean"]["f2"] == 1.0 and res["_mean"]["tasks"] == 6
    overlapping = {n: [(0, 5), (3, 9)] for n in range(len(rows))}      # two overlapping predictions: one merged span
    r2 = sc.score(rows, overlapping, chars=True)
    t = sc.TASKS[0]
    assert r2[t]["precision"] == 1.0 and r2[t]["predicted_chars"] == 18
    assert abs(r2[t]["recall"] - 18 / 40) < 1e-12
    assert abs(r2[t]["f2"] - sc.f_beta(1.0, 18 / 40, 2)) < 1e-12


def test_reference_points(tmp_path, sc):
    _, rows = _rows(tmp_path, sc)
    everything = {n: [(0, len(r["text"]))] for n, r in enumerate(rows)}
    res = sc.score(rows, everything, chars=True)
    t = sc.TASKS[0]
    gold = sum(b - a for r in rows if r["task"] == t for a, b in sc.merge(r["gold"]))
    total = sum(len(r["text"]) for r in rows if r["task"] == t)
    assert res[t]["recall"] == 1.0 and abs(res[t]["precision"] - gold / total) < 1e-12 and res[t]["fpr"] == 1.0
    assert sc.score(rows, {}, chars=True)["_mean"]["f2"] == 0.0


def test_bytes_become_characters_rounding_outwards(sc):
    text = "Zoë Ñandú"
    b = text.encode("utf-8")
    assert sc.byte_to_char(text, [(0, len(b))]) == [(0, len(text))]
    assert sc.byte_to_char(text, [(3, 4)]) == [(2, 3)], "a boundary inside ë takes the whole character"
    assert sc.byte_to_char("plain ascii", [(1, 4)]) == [(1, 4)]


def test_the_cli_report_is_read_by_file_name(tmp_path, sc):
    _, rows = _rows(tmp_path, sc)
    rep = tmp_path / "purebyte.jsonl"
    lines = [dict(file=f"inputs/0000/{0}.txt", spans=[], findings=[dict(start=0, end=9, kind="PII")]),
             dict(file="inputs/0000/1.txt", findings=[]),
             dict(file="inputs/0000/2.txt", not_scanned="too large"),
             dict(file="elsewhere/notes.md", findings=[dict(start=0, end=3)])]
    rep.write_text("".join(json.dumps(x) + "\n" for x in lines) + "purebyte: summary line\n", encoding="utf-8")
    pred, info = sc.read_cli(str(rep), set(range(len(rows))))
    assert pred == {0: [(0, 9)], 1: []} and info == dict(scanned=2, not_scanned=1)
    assert sc.input_name(1234) == os.path.join("0001", "1234.txt")


def test_the_exam_metrics_of_the_recipe(tmp_path, sc):
    _, rows = _rows(tmp_path, sc)
    files = []
    for k, frac in enumerate((1.0, 0.5)):
        pred = {n: [(a, a + int((b - a) * frac)) for a, b in r["gold"]] for n, r in enumerate(rows)}
        res = sc.score(rows, pred, chars=True)
        p = tmp_path / f"scores{k}.json"
        p.write_text(json.dumps(dict(models={f"seed{k}": dict(all=res)})), encoding="utf-8")
        files.append(str(p))
    out = tmp_path / "metrics.json"
    sc.main(["metrics", "--released", files[0], "--seeds", *files, "--out", str(out)])
    m = json.loads(out.read_text(encoding="utf-8"))
    for key in ("f2_mean6_seeds_mean", "f2_ai4privacy_en_seeds_mean", "precision_mean6_seeds_mean"):
        assert key in m, key                                    # the metric names of the recipe's criteria
    assert m["f2_mean6"] == 1.0 and m["n_seeds"] == 2 and m["f2_mean6_seeds_min"] < m["f2_mean6_seeds_max"] == 1.0


def test_repeated_uids_are_matched_in_order(sc, tmp_path):
    """A uid that the benchmark repeats: the k-th line of the spans file goes to its k-th sentence."""
    uid_to_ns = {"a": [0, 2], "b": [1]}
    p = tmp_path / "spans.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in (
        {"uid": "a", "spans": [[0, 3]]}, {"uid": "b", "spans": [[1, 2]]}, {"uid": "a", "spans": [[4, 6]]},
        {"uid": "a", "spans": [[7, 8]]})), encoding="utf-8")   # a third line of `a` has no sentence: ignored
    assert sc.read_spans(str(p), uid_to_ns) == {0: [(0, 3)], 1: [(1, 2)], 2: [(4, 6)]}
