# SPEC-011: The comparison harness

**Status:** Approved

## 1. Why this exists

Prompt-injection defenses cannot currently be compared. Every vendor publishes
self-reported numbers on private corpora at unstated operating points. PINT is
Lakera's own benchmark. `garak` is a red-teaming tool, not a comparator. Peer-
reviewed work (arXiv:2504.11168) reports up to 100% evasion against every major
commercial guardrail, and buyers have reasonably concluded that guardrails do not
work, but no one can say *which* work, *how well*, or *at what cost*.

The field is missing a referee, and no vendor can be one: measuring guardrails
honestly is against the interest of everyone selling a guardrail.

This project is unusually well placed to fill that gap, for one reason. Its own
detector scores **9.3% recall on BIPIA**, and that number was produced by this
harness, published in this repo's README, and used to kill this project's own
product positioning twice. A benchmark whose author's tool ranks badly *in the
benchmark's own table* is the only kind anyone should trust.

## 2. What it is

`thicket bench`, a CLI that takes any set of detectors, runs them over pinned
public corpora, and emits a reproducible comparison at **matched false-positive
rate**.

```
OT-BCH-001 MUST compare detectors at matched FPR, not at each detector's own
  default threshold. Comparing thresholded booleans across tools measures
  vendor threshold choices, not detection ability.
  Verify: unit
OT-BCH-002 MUST report recall, FPR, precision, sample size, and a 95% Wilson
  interval for n < 500, for every detector on every suite (OT-EVL-004/005).
  Verify: unit
OT-BCH-003 MUST report the full result set for every detector run, never a
  subset (OT-EVL-003). A detector may not be dropped from a published table
  for scoring badly, including ours.
  Verify: audit
OT-BCH-004 MUST record per detector: name, version, kind, and what it requires
  to run (offline / model download / network + API key). A number without its
  cost is not a comparison.
  Verify: unit
OT-BCH-005 MUST include a naive regex baseline in every published comparison.
  A tool that barely beats grep for "ignore previous instructions" is the most
  useful finding a benchmark can produce, and it is invisible without a floor.
  Verify: unit
OT-BCH-006 MUST NOT require an API key, a model download, or network access to
  run the deterministic detectors. Adapters needing those degrade to a clearly
  reported "not run", never a silent zero.
  Verify: unit
OT-BCH-007 MUST refuse to write a comparison artifact from a dirty tree
  (OT-EVL-002) and MUST record every detector's version in the artifact.
  Verify: unit
```

## 3. The detector contract

The whole harness rests on this being small enough that anyone can implement it
for their own tool in an afternoon.

```python
@dataclass(frozen=True)
class Judgement:
    detected: bool         # the detector's own default decision
    score: float | None    # continuous risk in [0,1]; enables OT-BCH-001
    error: str | None      # adapter failure, never silently a non-detection

class Detector(Protocol):
    name: str
    version: str
    kind: str        # deterministic | classifier | hosted | hybrid
    requires: str    # offline | model-download | network+key
    def scan(self, text: str, surface: str) -> Judgement: ...
```

```
OT-BCH-008 SHOULD expose a continuous score. A detector returning only a
  boolean can still be run, but appears in the table at its own fixed
  operating point with that limitation stated, because it cannot be
  threshold-matched.
  Verify: unit
```

That caveat is not a footnote. Most commercial APIs return a boolean or a coarse
band, which is precisely why cross-tool comparison has not happened, and the
table must say so rather than quietly comparing incomparable things.

## 4. Adapters shipped

| Adapter | Kind | Requires | Notes |
|---|---|---|---|
| `thicket` | deterministic | offline | This project's engine. Ships with its 9.3% BIPIA result visible. |
| `regex-floor` | deterministic | offline | ~10 patterns of the most obvious override phrasings. The honesty floor (OT-BCH-005). |
| `huggingface:<model>` | classifier | model download | Any sequence-classification model. Verified against `protectai/deberta-v3-base-prompt-injection-v2`. |
| `llm-guard:<scanner>` | classifier | model download | Drives real `llm-guard` if installed, else this repo's shim. |
| `thicket+normalize` | deterministic | offline | Any detector wrapped in our normalization, to measure preprocessing lift separately from detection. |

```
OT-BCH-009 MUST keep every adapter optional at import. `import billhook` must
  not import torch, transformers, llm_guard, or thicket.
  Verify: unit
OT-BCH-012 MUST ship the harness inside the installed distribution. A user who
  runs `pip install billhook` MUST be able to run the documented compare
  command with no source checkout and no extras.
  Verify: integration (installed-wheel smoke test)
OT-BCH-013 MUST NOT occupy a generic top-level import name. The harness lives
  under `billhook`, not `evals` or `bench`.
  Verify: unit
```

`OT-BCH-012` records a real defect found on 2026-08-01: the harness lived at
repo-root `evals/` while packaging declared `where = ["src"]`, so it was absent
from the wheel and the first command in the README could not work for anyone who
installed the package. This is the same failure the predecessor project shipped -
a documented quick start that only works from a checkout, and it is precisely
what the installed-wheel gate in SPEC-009 exists to catch.

`OT-BCH-013` is the reason the fix was a move rather than an added include.
Installing a top-level `evals` package into a user's `site-packages` would
collide with any other project using that entirely generic name. `billhook` is
distinctive enough to sit at top level, which is why the harness no longer
needs to hide inside another project's namespace.

## 5. What this harness will not do

Stated so the scope cannot drift into something dishonest:

- **It does not measure attack success.** It measures whether a detector emits a
  detection on a piece of text. It runs no agent and calls no model under attack.
  AgentDojo-style end-to-end ASR is a different quantity and is out of scope
  until it is actually run end to end (SPEC-010 §4).
- **It does not rank tools by a single score.** There is no leaderboard number.
  Recall at 1% FPR, recall at 5% FPR, and cost-to-run are different questions and
  a buyer with a latency budget and no GPU needs a different answer than one
  without.
- **It does not evaluate harmful-content classification.** Different capability,
  different benchmarks (SPEC-003 §2).
- **It does not accept vendor-supplied numbers.** Every figure in a published
  table is produced by running the detector in this harness.

## 6. Publication rules

The harness inherits SPEC-010 §2 wholesale. Additionally:

```
OT-BCH-010 MUST publish the exact command, corpus checksums, detector versions,
  and git SHA alongside any comparison table, such that a third party can
  reproduce it without contacting us.
  Verify: audit
OT-BCH-011 MUST offer any vendor whose tool appears in a published table the
  same reproduction path, and MUST correct a result shown to be a harness
  error rather than defending it.
  Verify: audit
```

`OT-BCH-011` is the load-bearing one for credibility. The harness will sometimes
be wrong about someone's product. The response to that determines whether this is
a referee or a marketing asset.
