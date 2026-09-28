# specialists

One folder per **AI Specialist**: a tiny model that does one thing very well, on a CPU, privately and measurably. The
training stack (`training/`) is generic; everything about a task lives here. The three folders below are the example
specialists of the first release, and templates for new ones: a new specialist is one more folder, made by hand or
by **Vibe Training** (describe it to an AI coding agent, which follows the golden path of
[AGENTS.md](../AGENTS.md#the-golden-path-vibe-training)).

| Example specialist | Detects | Status |
|---|---|---|
| [secrets-code](secrets-code/README.md) | credentials in source code and configuration | 1.0.0 released (one model; optional ensemble of three) |
| [secrets-bin](secrets-bin/README.md) | credentials embedded in compiled binaries | 1.0.0 released (one model) |
| [pii](pii/README.md) | personal data in any text, masked as one type (`PII`), for redaction by copy | 1.0.0 released (one model); an experimental typed recipe (12 types) |

A specialist folder:

| Path | Role |
|---|---|
| `specialist.yaml` | name, the plugin, what the exported model declares (profile, labels, window, n-gram precision), the release policy (ensemble, vote), the exams |
| `plugin.py` | the vertical: builds the windows and spans, its guards, manifest and classic rival (contract: `training/pbtrain/verticals/base.py`) |
| `<generator package>/` | the data generator the plugin imports |
| `recipes/*.yaml` | task files: the release recipes and candidates, a CPU smoke recipe |
| `README.md`, `MODEL_CARD.template.md` | documentation and the card of released models |
| `tests/` | the plugin's offline tests |

Create one with `python -m pbtrain new-vertical NAME --entities a,b` and follow
[training/docs/new-specialist.md](../training/docs/new-specialist.md).
