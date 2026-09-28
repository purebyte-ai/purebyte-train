"""The Quick start on a fresh checkout: `python data/build_smoke_data.py` into an empty $PUREBYTE_DATA, then the smoke
recipes of secrets-code and pii run in the factory (their `data` and `baselines` stages here, which read every
stand-in; training/tests/test_factory_smoke.py runs the whole chain of the secrets-code recipe). The stand-ins never
train anything else. And the catalogue rules the generator validates with depend on the Python version exactly as
data/README.md says."""
import os
import shutil
import subprocess
import sys
import types

import pytest
import yaml
from pbtrain import paths
from pbtrain.factory import spec as S
from pbtrain.factory.chain import StageError
from pbtrain.factory.stages import Context, run

REPO = paths.REPO
SMOKE = {"secrets-code": os.path.join(paths.SPECIALISTS_SRC, "secrets-code", "recipes", "smoke-cpu.yaml"),
         "pii": os.path.join(paths.SPECIALISTS_SRC, "pii", "recipes", "smoke-cpu.yaml")}


@pytest.fixture(scope="module")
def fresh(tmp_path_factory):
    """An empty $PUREBYTE_DATA after the documented step, run as a user runs it."""
    root = tmp_path_factory.mktemp("fresh")
    data, work = str(root / "data"), str(root / "work")
    env = dict(os.environ, PUREBYTE_DATA=data, PUREBYTE_WORK=work, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, os.path.join(REPO, "data", "build_smoke_data.py")], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    old = {k: os.environ.get(k) for k in ("PUREBYTE_DATA", "PUREBYTE_WORK")}
    os.environ["PUREBYTE_DATA"], os.environ["PUREBYTE_WORK"] = data, work
    yield types.SimpleNamespace(data=data, work=work, log=r.stdout)
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _recipe(tmp_path, name, **data_over):
    t = yaml.safe_load(open(SMOKE[name], encoding="utf-8"))
    t["data"].update(data_over)
    p = tmp_path / f"{name}-smoke.yaml"
    p.write_text(yaml.safe_dump(t), encoding="utf-8")
    return S.load(str(p))


@pytest.mark.parametrize("name", sorted(SMOKE))
def test_the_smoke_recipes_run_on_a_fresh_checkout(fresh, tmp_path, name):
    out = run(S.load(SMOKE[name]), ["data", "baselines"], str(tmp_path / "runs"), Context(log=lambda *a: None))
    d = out["data"]
    assert d["n_train"] > 0 and out["baselines"]["rival_f1"] >= 0.0
    assert d["corpus"]["build_manifest"]["made_up"] is True
    used = d["manifest"]["catalogs" if name == "secrets-code" else "resources"]["smoke_standins"]
    assert used and all(u.startswith(("catalogs/", "pii/")) for u in used)


@pytest.mark.parametrize("name", sorted(SMOKE))
def test_standins_never_train_another_recipe(fresh, tmp_path, name):
    other = os.path.join(fresh.data, "corpora", "not-smoke.bin")
    shutil.copyfile(os.path.join(fresh.data, "corpora", "smoke-context.bin"), other)
    with pytest.raises(StageError, match="made-up stand-ins"):
        run(_recipe(tmp_path, name, path=other), ["data"], str(tmp_path / "runs"), Context(log=lambda *a: None))


def test_catalogue_rules_and_the_python_version():
    """The generator validates a sampled value with its catalogue rule when the rule compiles: a global inline flag
    after the start is refused on every version (3.10 alone would compile it), `\\z` compiles on Python 3.14 only."""
    from secretsgen import catalog
    assert catalog._compile(r"abc(?i)def") is None
    assert catalog._compile(r"(?i)abcdef") is not None
    assert (catalog._compile(r"abc\z") is not None) == (sys.version_info >= (3, 14))
