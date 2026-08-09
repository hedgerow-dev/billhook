# Contributing to billhook

billhook is a referee. A contribution that makes the measurement less honest is
worse than no contribution, so the bar here is procedural as much as technical.

## Development setup

Python 3.10 or later. Then:

```bash
pip install -e '.[dev]'
```

Three gates must pass before any PR is considered. All are configured in
`pyproject.toml`; mypy runs strict.

```bash
pytest -q
ruff check src tests
mypy src
```

## Spec-first culture

`specs/` is the product. Code that contradicts a spec is a bug in the code.

Behavioural claims in code, docs, and commit messages cite requirement IDs
(`OT-BCH-*`, `OT-EVL-*`). If you are adding behaviour those requirements govern,
the change needs a spec entry or an amendment, not just code. See
[specs/README.md](specs/README.md) for which spec owns which prefix, and note
that some cited IDs are owned by open-thicket and are never redefined here.

## The sacred rules

These exist because each was once broken somewhere, and a PR that breaks one is
rejected regardless of how good the rest of it is.

- **Never degrade an unavailable detector into a silent non-detection**
  (OT-BCH-006). A missing API key, model, or package raises
  `AdapterUnavailable` and lands in the "Not run" table. Scoring an absent tool
  as one that never fires credits it with a perfect false-positive rate.
- **Never emit recall without an FPR** (OT-EVL-004). The renderer raises rather
  than print a recall cell beside an empty FPR cell. Keep it that way.
- **Never drop a detector or suite from a published table for scoring badly**
  (OT-BCH-003, OT-EVL-003). The full result set is reported, including ours.
- **Never write evaluation false-negative text into the repository**
  (OT-EVL-010). Counts and per-group rates, never the missed strings. There is
  no `--dump-misses` flag and there will not be one.
- **Never commit corpora** (OT-RUL-007, open-thicket's, binding here). Datasets
  are fetched at run time, checksum-pinned, and gitignored.
- **Lazy-import optional dependencies** (OT-BCH-009). `import billhook` must
  not import torch, transformers, llm_guard, or thicket. `pip install billhook`
  pulls nothing.

## Commits and PRs

Small commits. Imperative subject line ("Add acme-guard adapter", not "Added
some stuff"). The body explains the why, because the diff already shows the
what. If the change touches scored behaviour, cite the requirement IDs it
affects in the body.

## Where to start

- Detector work: read [docs/ADAPTERS.md](docs/ADAPTERS.md), then
  `src/billhook/detectors.py`. An adapter is an afternoon by design.
- Disputing a published result: [docs/CORRECTIONS.md](docs/CORRECTIONS.md).
- How the numbers are produced: [docs/BENCHMARK.md](docs/BENCHMARK.md).

If you find a way this harness flatters one tool over another, that is a bug,
and the most useful issue you could file.
