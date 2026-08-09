# billhook, comparing injection detectors honestly

An independent harness for measuring prompt-injection defenses against each
other, on pinned public corpora, at matched false-positive rate.

```bash
pip install billhook
python -m billhook.fetch --fetch bipia --fetch notinject
python -m billhook.compare --suite bipia --detector regex-floor
```

## Why this exists

You cannot currently compare guardrails. Vendors publish self-reported numbers,
on private corpora, at unstated operating points. PINT is Lakera's own benchmark.
`garak` is a red-teaming tool, not a comparator. Published research reports up to
100% evasion against every major commercial guardrail, and buyers have reasonably
concluded the whole category doesn't work, without being able to say *which*
tools work, *how well*, or *at what cost*.

The field needs a referee, and no vendor can be one.

We are unusually well placed only because our own detector is **bad**: it scores
9.3% recall on BIPIA, that number was produced by this harness, and it is printed
in our own README. A benchmark whose author's tool ranks poorly in its own table
is the only kind worth trusting.

## The three things this does differently

### 1. Matched false-positive rate

Every tool ships its own threshold. Comparing default outputs measures **vendor
risk appetite**, not detection ability, a detector that fires on everything
posts excellent recall and is worthless.

So we take each detector's continuous score, pick the threshold that yields a
fixed FPR *on that suite's own negatives*, and read recall there. Detectors that
expose only a boolean cannot be threshold-matched; they are run, reported at
their shipped operating point, and **marked**, never silently compared as though
they were.

### 2. A naive baseline floor

Every comparison includes `regex-floor`: ten obvious patterns, about thirty lines
of code. If a funded commercial product cannot clearly beat grep for "ignore
previous instructions", that is the most useful thing a benchmark can tell you,
and it is invisible without a floor.

### 3. Cost is part of the result

Every row carries what the detector *requires*, offline, a model download, or a
network call plus an API key. A 60% recall that needs a GPU and ships your
prompts to a third party is not comparable to a 40% that runs air-gapped in a
millisecond. The table refuses to pretend otherwise.

## Adding your detector

Three attributes and one method. If your tool is a Python package, this is an
afternoon.

```python
from evals.detectors import Judgement

class MyDetector:
    name = "acme-guard"
    version = "2.1.0"
    kind = "classifier"          # deterministic | classifier | hosted | hybrid
    requires = "network+key"     # offline | model-download | network+key

    def scan(self, text: str, surface: str) -> Judgement:
        risk = acme.check(text)          # your call
        return Judgement(detected=risk > 0.8, score=risk)
```

`score` should be a continuous risk in `[0, 1]`. Supplying it is what lets your
tool be threshold-matched instead of pinned to whatever cut-off you shipped -
in our experience this usually helps the tool, because default thresholds are
frequently mis-set for the deployment surface.

Built-in specs: `thicket`, `regex-floor`, `huggingface:<model>`,
`llm-guard:<scanner>`, and `normalize+<any spec>` to measure what deobfuscation
preprocessing contributes separately from detection.

## Reproducing and disputing a result

Every published table ships the exact command, corpus checksums, detector
versions, and git SHA. You do not need to contact us to reproduce it.

**If we get your tool wrong, we will correct it.** The harness will sometimes be
wrong, a bad adapter, a wrong label index, an out-of-distribution corpus. Open
an issue with a reproduction and we will fix the harness and re-publish. That
commitment is `OT-BCH-011` in `specs/011-benchmark-harness.md`, and it is the
thing that decides whether this is a referee or a marketing asset.

## What this does not measure

- **Not attack success.** It measures whether a detector flags a piece of text.
  No agent is run, no model is attacked. AgentDojo-style end-to-end attack
  success rate is a different quantity and out of scope until it is genuinely
  run end to end.
- **No single ranking score.** There is no leaderboard number. Recall at 1% FPR,
  recall at 5% FPR, and cost-to-run answer different questions, and a team with
  a latency budget and no GPU needs a different answer than one without.
- **Not harmful-content classification.** Different capability, different
  benchmarks.
- **No vendor-supplied numbers.** Every figure is produced by running the tool
  in this harness.

## A methodological warning we learned the hard way

**The choice of negative corpus dominates the result**, and it is the easiest
place to accidentally flatter a tool.

We measured the same classifier twice. Against *out-of-distribution* negatives
(generic Wikipedia-style passages) it looked strong. Against *in-distribution*
negatives, the benchmark's own clean emails, same format as the positives, it
collapsed, because it was firing on the document format rather than on injection
semantics.

Both runs were "correct". Only the second one answers the question a deployer
actually has, which is "what happens to my real traffic". When you read any
guardrail benchmark, including this one, check what the negatives were.
