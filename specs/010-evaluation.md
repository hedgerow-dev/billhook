# SPEC-010: Evaluation and claim honesty

**Status:** Approved

**Shared with open-thicket.** These requirements were written there and came
across with the harness. Both projects are bound by them and the IDs are
deliberately identical in both repositories, so that `OT-EVL-004` means the same
thing in either. See [specs/README.md](README.md) for the full ID map.

## 1. Purpose

The harness that became billhook was written after a codebase published detection
numbers that did not survive independent verification: a headline "100% recall on
indirect injection" produced entirely by one hardcoded literal string, a "100% /
2.2% FPR" figure whose dataset was not present in the repository, a "0% attack
success rate on AgentDojo" measured by a harness that did not run AgentDojo, and a
"0% false positive rate on realistic tool outputs" that measured 26.7% when re-run.

The technical work underneath was real. The numbers destroyed its credibility
anyway. These rules exist so that cannot happen again, and a referee that does not
hold itself to them has no standing to measure anyone else.

## 2. The honesty rules, release-blocking

```
OT-EVL-001 MUST NOT publish any number that is not reproduced by a committed,
  clean-tree artifact in evals/results/, with the reproducing command printed
  adjacent to it.
  Verify: audit
OT-EVL-002 MUST refuse to record a benchmark result from a dirty working tree.
  The harness writes `git_dirty: true` and the release check rejects it.
  Verify: unit
OT-EVL-003 MUST report the full result set for any benchmark run, not a selected
  subset. Reporting the best dataset and omitting the worst is prohibited.
  Verify: audit
OT-EVL-004 MUST report recall and false-positive rate together, always, with n.
  A recall figure without its FPR and sample size is not publishable.
  Verify: audit
OT-EVL-005 MUST state sample size and, for n < 500, the 95% confidence interval
  alongside any recall figure.
  Verify: audit
OT-EVL-006 MUST NOT compare one detector's result on an easy condition against
  another's on a hard condition. Comparisons state the identical evaluation
  condition for every row, or are not made.
  Verify: audit
```

`OT-EVL-005` retires a specific inherited claim: a 56.5% recall headline computed
on 23 samples, which carries a 95% confidence interval of roughly ±20 points.

`OT-EVL-006` retires another: comparing a score on InjecAgent's *enhanced*
variant, which prepends the literal string "IMPORTANT!!! Ignore all previous
instructions and strictly adhere to the following instruction:" to every sample,
against competitors' scores on a distribution-shifted eval. Removing only that
prefix moved the upstream score from 100% to 26.7-37.5%.

`OT-EVL-007` is open-thicket's ("MUST NOT fit rules to an evaluation set"). It
binds detector authors, not this harness, and is not restated here. The ID is
skipped rather than reused so it never means two things.

## 3. Harness

Datasets are **fetched, checksum-pinned, and never committed**. That is
open-thicket's `OT-RUL-007`, and it is a property of `billhook.fetch`: the
gitignored `evals/datasets/` tree is the only place a corpus is written.

```
OT-EVL-008 MUST fetch evaluation datasets at run time with a pinned checksum and
  fail on mismatch.
  Verify: unit
OT-EVL-009 MUST record in every result artifact: git SHA, dirty flag, detector
  name/version/kind/requires, dataset name and checksum, n, recall, FPR,
  precision, and p50/p95/p99 latency.
  Verify: unit
OT-EVL-010 MUST NOT write evaluation false-negative text into the repository.
  Committing the text burns the holdout for everyone, and inspecting FNs to
  author detection rules is prohibited by open-thicket's OT-RUL-006.
  Verify: audit (CI path check)
```

**The reference environment** is `ubuntu-latest`, x86_64, single process: this
project's CI runner. A latency figure measured anywhere else is a development
signal and is not publishable, and every artifact says which it is in
`environment.latency_publishable`. This mirrors open-thicket's `OT-ENG-019`.

Detection counts carry no such caveat. For a deterministic detector they do not
depend on the machine, so an artifact produced on a laptop is unpublishable for
latency and perfectly publishable for recall and FPR. The two are labelled
separately rather than the whole artifact being discarded.

`OT-EVL-009` named "ruleset hash, engine config" while the harness scored only
one engine. It records a `detector` block instead, because the runner now scores
any detector satisfying the SPEC-011 protocol and a ruleset hash is meaningless
for a hosted classifier.

`OT-EVL-010` prevents a specific inherited hazard: the upstream repository
committed verbatim false-negative text for every holdout set, including one whose
own provenance file stated it must never have its FNs inspected.

## 4. Which benchmarks apply

billhook scores injection and directive detectors. Running one against a
harmful-content benchmark measures the wrong capability, and the table must say
so rather than quietly reporting the number.

| Benchmark | Applies | Note |
|---|---|---|
| BIPIA | ✅ | indirect injection in real QA context, the primary honest measure |
| InjecAgent (base) | ✅ | must report base, not enhanced (§2) |
| AgentDojo | ✅ | only if actually run end-to-end; a regex match on extracted templates is not AgentDojo |
| NotInject | ✅ | false-positive measurement on trigger-word-loaded benign text |
| Public benign corpora | ✅ | FPR |
| JailbreakBench | ⚠️ | report it, with the caveat; do not exclude it for scoring badly |
| HarmBench | ❌ | harmful-content classification, different capability |

```
OT-EVL-011 MUST report JailbreakBench results when run rather than excluding the
  benchmark post hoc for an unfavorable result.
  Verify: audit
```

The distinction between "this measures a different capability" and "this measures
the capability and the tool did badly" must be argued in the open, with the number
printed, or not argued at all. That applies to every row, including a detector
written by whoever is running the harness.

## 5. What good looks like

A published comparison should read approximately:

> On BIPIA (indirect injection in retrieved content, n=3800), at a matched 1.0%
> FPR measured on NotInject: `regex-floor` 0.0%, `thicket` 9.3%,
> `deberta-v3-base-prompt-injection-v2` 16.5%. Reproduce with
> `python -m billhook.compare --suite bipia --at-fpr 0.01`. The classifier needs
> a GPU and a model download; the other two run offline in under a millisecond.

Unflattering to someone, specific, reproducible, and defensible. That is the
standard.
