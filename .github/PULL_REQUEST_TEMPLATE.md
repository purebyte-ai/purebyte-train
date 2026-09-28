## What this changes

<!-- One or two sentences on what changes for people who train AI Specialists, and why. Link the issue if there is one. -->

## How it was tested

<!-- Commands you ran and their relevant output (training/tests, the specialist's tests, data/tests). -->

## Checklist

- [ ] One purpose per pull request; tests and documentation are in this change.
- [ ] I did not change the trainer's parameter order, random draws or floating-point order, or I say why here (the
      golden weight fingerprints match only on the machine that recorded them, where the maintainers compare them;
      they are regenerated only on purpose).
- [ ] No criterion, seed or operating point was chosen by looking at an exam.
- [ ] No real credentials or personal data anywhere: code, tests, fixtures, logs, screenshots or this description.
- [ ] No dataset files, model weights or training runs.
- [ ] Nothing new is sent over the network (the data scripts download only their pinned sources).
- [ ] User-visible changes are recorded in `CHANGELOG.md`.
- [ ] Commits are signed off (`git commit -s`).
