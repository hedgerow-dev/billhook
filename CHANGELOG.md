# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-09

First public release, extracted from the
[open-thicket](https://github.com/hedgerow-dev/open-thicket) repository so that
the referee is not shipped inside one of the tools it scores.

### Added

- `billhook.fetch`: fetcher for checksum-pinned evaluation datasets. Every
  pinned file carries a SHA-256 and an upstream licence as reported by the
  host; `NOASSERTION` is recorded where the host detects no licence rather
  than assumed away. Fetch refuses unpinned files, and `--verify` re-checks
  downloaded bytes against the pin table. Datasets include BIPIA, NotInject,
  InjecGuard, JailbreakBench, the Dolly-15k benign splits, and the
  llmail-inject challenge set (partially pinned: the phase-1 file and raw
  submission logs are not fetched because no result needs them).
- `billhook.run`: single-detector runner producing reproducible artifacts
  (schema `billhook/eval-result/1`) with Wilson score intervals on recall,
  FPR, and precision; per-group breakdowns carrying counts and rates only,
  never sample text; latency percentiles recorded in every artifact and gated
  on the reference environment (the project CI runner, `ubuntu-latest`
  x86_64) before they are publishable; a dirty-tree release gate
  (`--release` refuses to write from a dirty working tree); and a recorded
  reproduce command.
- `billhook.compare`: cross-detector comparison at matched false-positive
  rate (SPEC-011). Thresholds are matched on a shared benign corpus via
  `--benign-suite` (SPEC-010 §5), and the table names the basis and its
  granularity so a column read off too few negatives is printed as the 0% FPR
  column it really is. Includes the default-precision column, and the
  "not run" table lives in the artifact rather than on stderr: a detector
  that cannot be loaded is reported, never scored as a silent zero.
- Detector adapters in `billhook.detectors`:
  - `regex-floor`: ten obvious patterns in-tree, the honesty baseline every
    comparison includes.
  - `thicket`: adapter for open-thicket's deterministic engine, optional and
    never in the default comparison.
  - `huggingface:<model>`: any sequence-classification classifier.
  - `llm-guard:<scanner>`: the upstream package only; no compatibility-shim
    substitution.
  - `normalize+<spec>`: wraps any detector in open-thicket's normalization to
    isolate preprocessing lift from detection.
  - All adapters beyond `regex-floor` are optional extras that report
    `AdapterUnavailable` until their backing package is installed.
  - `import billhook` pulls no detector, model, or framework; a test asserts
    the import-purity list.
- The spec set in `specs/`: SPEC-010 (evaluation) and SPEC-011 (benchmark
  harness), with every cited requirement ID resolving, and the packaging
  rules pointed at this distribution rather than `thicket.bench`.
- `docs/BENCHMARK.md`: the published benchmark methodology.

[0.1.0]: https://github.com/hedgerow-dev/billhook/releases/tag/v0.1.0
