# billhook

**An independent harness for comparing prompt-injection defenses.**

```bash
pip install billhook
python -m billhook.fetch --fetch bipia --fetch notinject
python -m billhook.compare --suite bipia --detector regex-floor
```

> **Status: pre-release.** Not yet published to PyPI.

---

## The problem this addresses

You cannot currently compare guardrails. Vendors self-report, on private corpora,
at unstated operating points. PINT is Lakera's own benchmark. `garak` is a
red-teaming tool, not a comparator. Published research reports high evasion rates
against every major commercial guardrail, so buyers have reasonably concluded the
category doesn't work, without being able to say which tools work, how well, or
at what cost.

## What it does differently

**Matched false-positive rate.** Every tool ships its own threshold, so comparing
defaults measures vendor risk appetite, not detection ability. billhook takes each
detector's continuous score and reads recall at a fixed FPR on the suite's own
negatives. Boolean-only detectors are reported at their shipped operating point
and *marked*, never silently compared as if matched.

**A naive baseline.** Every comparison includes `regex-floor`: ten obvious
patterns, thirty lines. If a funded product can't clearly beat grep for "ignore
previous instructions", that's the most useful thing a benchmark can report, and
it's invisible without a floor.

**Cost in the table.** Every row carries what the detector needs, offline, a
model download, or a network call and an API key. Recall that needs a GPU and
ships prompts to a third party isn't comparable to recall that runs air-gapped.

**Never a silent zero.** A detector that can't be loaded is reported as "not run",
with the reason. Degrading an unavailable tool into one that simply never fires
would score it as perfectly permissive.

## It depends on nothing

`pip install billhook` pulls no detector, no model, no framework. The only
detector in-tree is `regex-floor`, because a comparison without a baseline can't
tell you whether a tool is good or whether the corpus is easy.

Everything else is an optional adapter that reports itself unavailable until you
install its backing package:

| Spec | Needs |
|---|---|
| `regex-floor` | nothing |
| `thicket` | `pip install open-thicket` |
| `huggingface:<model>` | `pip install 'billhook[hf]'` |
| `llm-guard:<scanner>` | `pip install llm-guard` |
| `lakera:guard` | `BILLHOOK_LAKERA_API_KEY` in the environment |
| `normalize+<spec>` | `pip install open-thicket` |

Adding your own takes about thirty lines: satisfy the `Detector` protocol in
[detectors.py](src/billhook/detectors.py) and register it in `load()`.

## Who makes this

billhook is built by [Hedgerow](https://hedgerow.dev), which also makes
open-thicket, a deterministic injection detector that this harness scores as
`thicket`. That is a conflict of interest and worth stating plainly rather than
burying.

Two things keep it honest. open-thicket is not in the default comparison and has
to be asked for by name, so no result here quietly favours it. And its measured
recall is poor, which came out of this harness and is published in open-thicket's
own README and limitations doc rather than hidden.

If you find a way this harness flatters one tool over another, that's a bug and
the most useful issue you could file.

## Research built on it

`evals/experiments/` holds worked examples of the harness in use: a study of
where a deployed prompt-injection classifier
(`protectai/deberta-v3-base-prompt-injection-v2`) over-fires, why, and what it
costs at a real operating point. Each script prints one result and names what it
establishes; `evals/experiments/README.md` records the exact scoring conditions,
and `docs/FINDINGS-*.md` write the results up, including the retractions.

```bash
pip install -e ".[experiments]"
python -m billhook.fetch --fetch bipia --fetch llmail-inject
python evals/experiments/occlusion_triggers.py        # the single-word triggers
python evals/experiments/trigger_label_association.py # where it learned them
```

These evaluate a third-party model, not open-thicket, so they do not bear on the
conflict of interest above.

## License

Apache 2.0. Copyright 2026 Hedgerow BV.
