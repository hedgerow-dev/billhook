# First comparison: four injection detectors on indirect-injection corpora

**2026-08-01.** Produced by `thicket bench` on a clean tree. Reproduce with the
command in §6. Every detector that ran appears below, including ours, which is
the worst performer on one suite and the best on another.

---

## 1. The headline

**No detector tested is usable for detecting indirect prompt injection in
retrieved content.** The best recall any of the four achieved on BIPIA, holding
false positives to 5%, was **9.4%**.

That is the finding. It is not that one tool beats another; it is that the
category does not currently work on this surface, and buyers should plan
accordingly, which means constraining what an agent can *do* with retrieved
content rather than relying on detecting the injection inside it.

## 2. BIPIA, indirect injection in retrieved email content

n = 3750 attacks / 50 clean emails. Surface: tool result.

| Detector | Kind | Requires | default R / FPR | R @ 1% FPR | R @ 5% FPR |
|---|---|---|---|---|---|
| `deberta-v3-base-prompt-injection-v2` | classifier | model download | 61.1% / **62.0%** | 1.0% | 2.1% |
| `normalize+deberta-v3-…` | classifier | model download | 83.7% / **80.0%** | 1.3% | 4.7% |
| `thicket` | deterministic | offline | 9.3% / 4.0% | 0.0% | **9.4%** |
| `regex-floor` | deterministic | offline | 0.0% / 0.0% | 0.0% | 0.0% |

Read the default column and the matched column together, because they tell
opposite stories.

At its **shipped threshold** the classifier looks far stronger than the
deterministic tools, 61.1% recall against 9.3%. But it is flagging **62% of the
clean emails** to get there. Once both tools are held to the same false-positive
budget, that advantage inverts: at 5% FPR the classifier finds 2.1% and the regex
engine finds 9.4%.

This is the entire argument for matched-FPR comparison. A recall number quoted
without its false-positive rate is not a measurement, and every tool ships a
threshold chosen by its vendor for a surface that may not be yours.

## 3. The false-positive picture is the useful part

Same detectors, three different benign corpora:

| Benign corpus | n | `regex-floor` | `thicket` | `deberta-v3-…` |
|---|---:|---:|---:|---:|
| dolly-15k context (ordinary prose passages) | 4467 | 0.4% | 1.4% | **0.6%** |
| NotInject (benign, trigger-word-loaded) | 339 | 0.0% | 0.0% | **43.4%** |
| BIPIA clean emails (formatted business email) | 50 | 0.0% | 4.0% | **62.0%** |

The classifier is **fine on ordinary prose** (0.6%) and **collapses on anything
that superficially resembles an attack surface**, 43% on benign text that merely
contains injection-adjacent trigger words, 62% on structured emails.

That pattern says the model keys substantially on surface features rather than on
whether an instruction is actually being injected. NotInject exists precisely to
measure this failure mode, and it is where the two deterministic tools score 0%.

### Being fair to the model

`protectai/deberta-v3-base-prompt-injection-v2` is Apache-2.0, free, and widely
used (~285k downloads/month; it is the model LLM Guard's `PromptInjection`
scanner shipped). Its model card **does** disclose a false-positive surface:

> we do not recommend using this scanner for system prompts, as it produces
> false-positives

It does **not** warn about documents or retrieved content, and its listed
training data is prompt-style. So scanning retrieved emails with it is
out-of-distribution use, but it is also exactly what indirect-injection defense
requires, and it is a mainstream deployment pattern.

Note also that **the model and LLM Guard were both archived** (2026-07-09,
following Palo Alto's acquisition of Protect AI). This will not be fixed.

**Actionable version:** if you are running this model over RAG or tool output,
measure your own false-positive rate before trusting the default threshold. It
may be flagging most of your legitimate traffic.

## 4. InjecAgent (base), attacker instructions in tool output

n = 1054 attacks / 17 clean templates.

| Detector | default R / FPR | R @ 1% FPR | R @ 5% FPR |
|---|---|---|---|
| `deberta-v3-base-prompt-injection-v2` | 65.7% / 58.8% | **16.5%** | **16.5%** |
| `normalize+deberta-v3-…` | 84.7% / 64.7% | 10.0% | 10.0% |
| `thicket` | 33.1% / 5.9% | 14.5% | 14.5% |
| `regex-floor` | 0.0% / 0.0% | 0.0% | 0.0% |

Here the classifier is genuinely ahead at matched FPR (16.5% vs 14.5%), though
both are low and the 17 negatives make the FPR estimate very coarse, this suite
cannot resolve a 1% operating point at all, which is why both columns are
identical.

## 5. Three secondary findings

**Preprocessing made things worse, not better.** `normalize+` wraps a detector in
our Unicode normalization and takes the max score. It raised default recall
substantially (61.1% → 83.7% on BIPIA), and raised the false-positive rate just
as much (62.0% → 80.0%), so at matched FPR the gain is small on BIPIA and
*negative* on InjecAgent. Feeding decoded variants to a classifier is an
intuitive design that does not pay for itself.

**The corpora are not trivially easy.** `regex-floor`, ten obvious patterns like
"ignore previous instructions", scores **0.0%** on both attack suites. Whatever
these benchmarks contain, it is not the textbook phrasing. That is a validity
check worth running on any injection benchmark before trusting results from it.

**Cost is not correlated with capability here.** The two offline detectors need
no GPU, no model download, and no network. The classifier needs all three and, at
matched FPR on the primary suite, finds less.

## 6. Reproducing this

```bash
pip install billhook open-thicket "billhook[hf]"
python -m billhook.fetch --fetch bipia --fetch injecagent-base \
    --fetch notinject --fetch dolly15k
python -m billhook.compare \
    --suite bipia --suite injecagent-base --suite notinject --suite dolly-context \
    --detector regex-floor --detector thicket \
    --detector huggingface:protectai/deberta-v3-base-prompt-injection-v2 \
    --detector normalize+huggingface:protectai/deberta-v3-base-prompt-injection-v2 \
    --at-fpr 0.01,0.05
```

Corpus checksums, detector versions, and the git SHA are recorded in the JSON
artifact alongside this document in `evals/results/`.

## 7. Limits of this comparison

- **Detection, not prevention.** No agent was run and no model was attacked. This
  does not measure whether an attack would have succeeded.
- **One classifier.** PromptGuard 2 is gated (`gated: manual`) and could not be
  obtained. Commercial APIs were not tested, several return only a boolean and
  cannot be threshold-matched at all, which is itself part of why this field
  lacks comparable numbers.
- **Small negative sets on two suites.** BIPIA ships 50 clean emails and
  InjecAgent 17 clean templates. A 1% FPR target is not resolvable on either;
  treat those columns as directional.
- **The negative corpus dominates the result.** We measured the same classifier
  against out-of-distribution negatives (generic prose) and in-distribution ones
  (the benchmark's own clean emails) and got opposite readings. Only the second
  answers a deployer's question. When reading any guardrail benchmark, including
  this one, check what the negatives were.

## 8. If we got your tool wrong

Open an issue with a reproduction. We will fix the harness and re-publish. That
is `OT-BCH-011` in `specs/011-benchmark-harness.md`, and honoring it is the only
thing separating this from marketing.
