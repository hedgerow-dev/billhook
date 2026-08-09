# Guardrail misses are nearly independent, and stacking guardrails still buys almost nothing

**Date:** 2026-08-02 · **Status:** verified, 2 of 4 predictions falsified

Every guardrail benchmark, including ours, reports **one row per model**. A row
per model cannot answer the question deployments actually face, which is not
"which detector is best" but "if I run two, am I safer?" That question is about
the *off-diagonal*: do two detectors fail on the same inputs, or different ones?

"Defense in depth" is the standard recommendation and it is priced on an
assumption nobody has measured: that failures are independent, so the chance
both miss is roughly the product of the chances each misses.

The gap is explicit rather than merely unfilled. SAGE
([arXiv 2607.22926](https://arxiv.org/html/2607.22926v1)) states that correlated
detector failures invalidate the independence bound, that "without measured
diversity, the defensible joint-failure bound may equal the strongest single
layer", and that evaluation should probe correlated blind spots. It measures
none. Neither does any guardrail benchmark we can find.

Reproduce:

```bash
python evals/experiments/ensemble_independence.py        # 5% budget
python evals/experiments/ensemble_independence.py 0.01   # sensitivity
python evals/experiments/ensemble_independence.py 0.10
```

10 detectors, 553 benign records (415 clusters), 500 LLMail-Inject
ground-truth-fired attacks, 500 BIPIA static attacks, 200 held-out cluster
splits.

---

## The short version

1. **Misses really are nearly independent.** Median joint-miss rate is
   1.02-1.08× the independence prediction across 45 pairs. The diversity that
   defense-in-depth assumes is genuinely there.
2. **It almost never converts into recall.** At a matched 5% false-positive
   budget the best *pair* beats the best *single* by **+1.0 points** on LLMail
   and **−1.4** on BIPIA. At 10%: +1.0 and +0.0.
3. **It converts at tight budgets only.** At 1%, the best pair gains **+12.8**
   points on LLMail. This is the one condition where stacking pays.
4. **More layers is worse.** All ten members at a matched 5% budget score
   **94.0%** on LLMail, below the 98.8% of the single best detector.
5. **The apparent benefit is an operating-point shift.** Stack all ten the way
   deployments actually do, each at its own 5% threshold, and the ensemble runs
   at **25.0%** false positives, five times the budget it was supposed to hold.

The one-line reading: *stacking guardrails mostly moves you along the same ROC
curve rather than to a better one, and the gains people report come from not
having held the false-positive rate fixed.*

---

## A. Why the thresholds had to be redone

Every member is placed at a **common false-positive budget on a held-out
calibration split**, never at its shipped cut-off. Comparing default thresholds
measures vendor risk appetite (OT-BCH-001), and a trigger-happy member would
otherwise register as diversity when it is just a lower threshold.

Per-member operating points, threshold at a 5% budget, median over 200 cluster
splits:

| member | threshold | FPR eval | llmail | bipia | dolly | rec llmail | rec bipia |
|---|---:|---:|---:|---:|---:|---:|---:|
| protectai-v2 | 9.93e-01 | 4.6% | 0.0% | 48.1% | 0.0% | 21.2% | 45.2% |
| protectai-v1 | 1.31e-07 | 3.6% | 1.0% | 9.5% | 4.4% | 12.0% | 9.8% |
| testsavant-base | 3.78e-02 | 4.2% | 0.0% | 25.5% | 3.7% | 43.2% | 37.8% |
| testsavant-large | 1.28e-04 | 4.7% | 0.0% | 0.0% | 8.7% | 28.8% | 19.6% |
| proventra | 7.39e-04 | 4.9% | 0.0% | 14.1% | 6.3% | **98.8%** | 37.0% |
| tihilya | 1.00e-04 | 4.7% | 0.0% | 37.7% | 2.6% | 43.0% | 54.4% |
| fmops | 1.00e+00 | 4.5% | 4.5% | 28.6% | 0.7% | 88.0% | 53.0% |
| devndeploy | 1.00e+00 | 4.4% | 0.0% | 50.0% | 0.6% | 61.6% | **82.8%** |
| regex-floor | 3.33e-01 | 0.0% | 0.0% | 0.0% | 0.0% | 7.4% | 0.0% |
| thicket | 4.00e-01 | 2.5% | 3.0% | 3.3% | 1.4% | 55.2% | 8.8% |

The per-corpus FPR columns reproduce §A of the classifier-surface writeup from a
different direction: at a *pooled* threshold, the transactional-email column
(`bipia`) is where five of eight classifiers spend their entire budget.

**A protocol bug that would have inverted the headline.** The obvious way to set
a threshold, take the (1−budget) order statistic of benign scores, names a
threshold without checking what it does. `regex-floor` emits two distinct scores
on benign text, its 95th percentile is `0.0`, and `score >= 0.0` flags **100%**
of inputs. The first run of this experiment therefore handed the floor a
100%-FPR operating point, scored 100% recall from it, and **selected
`regex-floor` as the best single detector in the fleet.**

The fix walks the distinct scores upward and takes the first one whose
*achieved* FPR meets the budget. Where too much benign mass ties at a member's
maximum score, no finite threshold meets the budget; that member is reported
untunable rather than scored at a budget it never hit. This was caught before
publication, not after, and it is recorded because an order statistic is the
default in every comparison of this kind and it is wrong for any detector whose
scores are discrete.

---

## B. Misses are nearly independent; false positives are not

Independence gap = P(both err) / [P(A errs) × P(B errs)]. **1.0 is independent**,
above 1.0 means failures cluster.

| error type | pairs | median gap | IQR | max gap | median φ | gap > 1 |
|---|---:|---:|---|---:|---:|---:|
| **missed attacks**, LLMail | 45 | **1.08** | [1.04, 1.12] | 1.75 | +0.091 | 37/45 |
| **missed attacks**, BIPIA | 45 | **1.02** | [1.00, 1.09] | 1.42 | +0.058 | 31/45 |
| **false positives** | 45 (36 defined) | **2.17** | [1.06, 4.25] | 13.93 | +0.034 | 28/36 |

The gap is a ratio, so it is undefined when a member never errs. That happens
for 9 of the 45 false-positive pairs, all of them involving `regex-floor`,
whose false-positive rate here is 0.0%. Those pairs are written as `null` in the
artifact and excluded from the median; the false-negative gaps are all defined.

**P1 confirmed in sign, and the magnitude is the finding.** Joint misses exceed
independence for 37 of 45 pairs, but only barely: the median pair misses
together 8% more often than chance, and φ > 0.3 for just 3 of 45. On the
false-negative side these detectors are close to independent.

**False positives behave completely differently.** Median gap 2.17, reaching
13.9. And the correlation is concentrated, not universal, in exactly the models
that fail on transactional email:

| pair | joint FP | independent | gap | φ |
|---|---:|---:|---:|---:|
| testsavant-base + tihilya | 3.1% | 0.2% | 13.93 | **0.64** |
| testsavant-base + devndeploy | 2.5% | 0.2% | 11.47 | 0.52 |
| protectai-v2 + devndeploy | 2.7% | 0.3% | 10.59 | 0.51 |
| tihilya + devndeploy | 2.5% | 0.2% | 10.62 | 0.49 |
| protectai-v2 + fmops | 2.2% | 0.2% | 8.80 | 0.41 |
| protectai-v2 + testsavant-base | 1.6% | 0.2% | 6.86 | 0.30 |

**P3 falsified, and it was our own prediction about our own prior finding.** §B
of the classifier-surface writeup identifies protectai-v2 and testsavant-base as
the two fleet members that key on the register of an automated notification. We
predicted that shared shortcut would make them the most FP-correlated pair in
the fleet. They rank **6th of 45**. The shortcut is real and pre-registered
out-of-sample, but it does not dominate false-positive co-occurrence: whatever
makes these models fire together on benign email is broader than the register
feature we characterised, and `tihilya`, which does *not* have the shortcut,
correlates with `testsavant-base` more strongly than `protectai-v2` does.

---

## C. The main result: matched-budget ensembles

Every member is tightened by a common quantile until the OR of them hits the
budget **on the calibration half**, then evaluated on the other half. All 10
singles and all 45 pairs are searched; the winner is not a nominated candidate.

**Split counts differ between tables and small differences follow from it.**
The per-member table in §A is the median over 200 cluster splits; the ensemble
search below runs 50, because each configuration costs a 20-step threshold
search per split. `devndeploy` therefore reads 82.8% BIPIA recall in §A and
85.0% here. Same quantity, different split count, not a discrepancy to resolve.

**At a 5% budget:**

| ensemble | FPR eval | rec llmail | rec bipia | naive FPR |
|---|---:|---:|---:|---:|
| best single on LLMail: proventra | 5.3% | 98.8% | 37.0% | 5.3% |
| best pair: proventra + fmops | 5.3% | **99.8%** | 60.7% | 10.1% |
| best single on BIPIA: devndeploy | 5.3% | 63.4% | **85.0%** | 5.3% |
| best pair: devndeploy + regex-floor | 5.0% | 62.4% | 83.6% | 5.3% |
| **all ten members** | 4.7% | 94.0% | 58.9% | **25.0%** |

Best pair minus best single: **+1.0** points on LLMail, **−1.4** on BIPIA. Ten
members is *worse than one* on both.

**Budget sensitivity, and it matters:**

| budget | LLMail: pair − single | BIPIA: pair − single | all-ten, matched | all-ten, naive FPR |
|---:|---:|---:|---|---:|
| 1% | **+12.8** | +9.2 | no feasible point | 6.5% |
| 5% | +1.0 | −1.4 | 94.0% / 58.9% | 25.0% |
| 10% | +1.0 | +0.0 | 99.8% / 78.4% | 43.2% |

**P2 confirmed at 5% and 10%, falsified at 1%.** Stacking pays when the budget
is tight enough that a single detector has to sit at an extreme threshold and
bleeds recall for it. Once the budget is loose enough for the best single
detector to approach its ceiling, a second detector adds nothing that the
required tightening does not immediately take back.

**Why ten layers cannot hit 1%.** Under a single shared tightening knob the
ten-member OR has no feasible non-silent operating point at a 1% budget, so the
protocol reports it silent. This is **granularity, not blindness**, and the
distinction matters:

| per-member quantile | joint FPR | rec llmail | rec bipia |
|---:|---:|---:|---:|
| 0.10% | 0.0% | 0.0% | 0.0% |
| 0.20% | 1.4% | 82.4% | 35.0% |
| 1.00% | 6.7% | 97.6% | 67.0% |
| 5.00% | 24.2% | 100.0% | 96.2% |

The stack jumps from silent to 1.4% with nothing in between. A ten-way OR is
simply too coarse an instrument to be placed at 1%. Do not read the 0.0% as
"ten scanners detect nothing."

**Interpretation, offered as interpretation.** The union of two detectors does
cover more attacks, which §B's near-independence predicts. Holding the joint
false-positive rate fixed requires tightening both, and the recall that
tightening costs is close to what the union gained. We measured both endpoints;
we did not isolate the cancellation, and it is stated as the reading most
consistent with the data rather than as a mechanism.

**P4 confirmed on correlation, and it does not help.** The prediction was that
pairing a neural classifier with a non-neural one decorrelates errors. It does:
pairs containing `regex-floor` or `thicket` show median FP gap 0.00 against 3.00
for neural-neural pairs, and BIPIA miss gap 1.00 against 1.07. But those pairs
have *lower* median LLMail recall (49.0% for `regex-floor` pairs vs 65.8%
without). **Diversity without capability buys nothing.** A decorrelated member
that catches almost nothing contributes almost nothing, which is the practical
answer to "should I add a regex layer to my classifier".

---

## D. The shared blind spot

Attacks missed by **every one of the ten** members, each at its own 5% budget:

| corpus | missed by all | rate |
|---|---:|---:|
| LLMail-Inject (ground-truth-fired) | 0 / 500 | 0.0% |
| BIPIA (static) | 21 / 500 | 4.2% |

Zero on LLMail is the near-independence result made concrete: for every one of
500 attacks that actually fired a tool call, *something* in the fleet saw it.

The 21 BIPIA survivors are not distinguished by length (median 559 characters
against 551 corpus-wide). Their hashes are recorded in the artifact. We do not
have an account of what they share, and are not offering one.

---

## Limitations

- **BIPIA attacks contain BIPIA benign text.** The attack corpus is built as
  clean email + attack string, and those same 50 clean emails are in the benign
  pool. Attack and benign samples are therefore not independent on this corpus.
  This is inherent to BIPIA's construction and matches `calibration_set_matters.py`,
  but BIPIA recall and BIPIA FPR should not be read as independent measurements.
- **One benign mixture.** LLMail benign + BIPIA clean + dolly prose, pooled.
  `calibration_set_matters.py` showed the choice of benign corpus moves the
  operating point enormously; a different mixture would move these thresholds
  and could move the ensemble comparison with them.
- **OR-aggregation only.** Deployments chain scanners and block if any fires, so
  OR is the realistic policy, but score-averaging or a trained stacker could do
  better than either member. Not tested.
- **A common tightening knob, not per-member optimisation.** A free per-member
  search would overfit the calibration half and is not reproducible in a
  deployment. It would also likely find better ensembles than we report, so the
  matched-budget numbers here are a lower bound on what stacking can achieve.
- **Two of ten members are degenerate.** `fmops` and `devndeploy` flag most
  benign prose at their shipped thresholds. Matched-FPR thresholding is what
  makes them comparable at all, and it works, but their thresholds sit at 1.00.

---

## What a deployer should take from this

1. **Adding a second guardrail is not free protection.** At a 5% or 10% budget
   it bought ~1 point in this fleet. Measure before you buy.
2. **If you stack, re-calibrate.** Ten scanners at their individual 5%
   thresholds ran at 25% false positives here. The layer count silently changes
   your operating point.
3. **Stacking helps most where budgets are tightest.** +12.8 points at 1%, +1.0
   at 10%. If you cannot tolerate false positives, an ensemble is worth it.
4. **More layers is not more safety.** Ten members underperformed one.
5. **A cheap decorrelated layer is still a weak layer.** `regex-floor`
   decorrelates beautifully and adds nothing.

---

## Corrections log

- **2026-08-02**, the first run set thresholds with the (1−budget) order
  statistic, which returns `0.0` for detectors with discrete scores and flags
  100% of inputs. `regex-floor` was thereby scored at 100% FPR / 100% recall and
  reported as the fleet's best single detector. Replaced with a threshold that
  verifies its achieved FPR; untunable members are now reported as such. Caught
  before publication.
- **2026-08-02**, the first ensemble table nominated candidate pairs by LLMail
  recall only and so never evaluated the strongest BIPIA members. It reported
  the best BIPIA pair as +23.7 points over "the best single", which was not the
  best single on that corpus. Replaced with an exhaustive search over all 10
  singles and all 45 pairs; the true BIPIA figure is **−1.4**. The error
  inflated the case for stacking, which is the direction that flatters the
  interesting result, and is exactly why the search is now exhaustive.
