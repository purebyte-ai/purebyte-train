# Security policy

The training stack of PureByte's AI Specialists downloads public datasets that contain credentials and personal data
(real ones, in some public benchmarks) and produces model files that other people run. We treat vulnerabilities in it
with priority.

## Supported versions

| Version | Security fixes |
|---|---|
| The `main` branch and the latest tagged release | Yes |
| Older releases | No; please upgrade |

## Reporting a vulnerability

**Please do not open a public issue for a vulnerability.** Report it privately:

1. Preferably through GitHub: the repository's **Security** tab, then **Report a vulnerability**.
2. Or by email to **pablo@purebyte.ai**.

Please include the commit or version, your platform (`python -m pbtrain doctor`), the steps that reproduce the problem
and the impact you expect. **Never include real credentials or real personal data**: use look-alike values of the
same shape.

What happens next:

- We acknowledge the report within 3 working days and keep you informed as we investigate.
- We agree on a disclosure date with you. By default we aim to publish a fix and an advisory within 90 days.
- We credit reporters in the advisory unless they prefer otherwise.

## What is in scope

- **The data scripts**: a download that is not verified against its pinned commit or checksum, a path from a
  manifest or an archive that escapes its data directory, or network access beyond the pinned sources.
- **Leaks of dataset values**: anything that could print, log or write in clear a credential or a piece of personal
  data from a dataset, or put one into the repository, a run report, a model card or a released bundle.
- **Released bundles**: a public bundle that carries internal paths, run names or data, or a checksum that does not
  match its file.
- **Exported model files**: a file written by `pbtrain` that the runtime would load but that does not follow the
  model format specification.

Vulnerabilities in the runtime (the engine, the CLI, the HTTP server, the integrations) belong to
[purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte/security/policy).

## What is not a vulnerability

- **A weak model**, a missed credential or a false alarm of a specialist you trained: that is a quality issue. Please
  open an issue with masked snippets only.
- Credentials or personal data inside the public datasets themselves: report them to the dataset's maintainers.

## Hardening notes for users

- Keep `$PUREBYTE_DATA` and `$PUREBYTE_WORK` outside any directory you publish or sync.
- Never train on data you are not allowed to use, and never commit data, runs or weights.
- Publish only public bundles made with `python -m pbtrain release TASK --public --semver X.Y.Z`, which refuse
  internal paths and names in their card and metadata.
