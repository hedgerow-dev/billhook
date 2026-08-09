# Experiment: does open-thicket normalization lift a real classifier?

**Date:** 2026-08-01 · **Reproduce:** `python evals/experiments/deobfuscation_lift.py`
**Result: the hypothesis is largely falsified.** Recorded here because it argues
against our own product, and SPEC-010 exists so that such results get published.

## Hypothesis under test

open-thicket scores 42% on BIPIA substitution ciphers and 42% on emoji
substitution against 2–4% on plain-language categories. The proposed reading was
that we are a *deobfuscation* engine rather than a detector, and that placing our
normalization in front of an ML classifier, which published work reports failing
on encoded input, would lift that classifier meaningfully. If true, the
positioning becomes "we make the classifier you already run work on obfuscated
input", and our own low recall stops mattering.

## Method

**Classifier:** `protectai/deberta-v3-base-prompt-injection-v2` (Apache-2.0,
ungated, ~285k downloads/month). Chosen because it is the model LLM Guard's
`PromptInjection` scanner used, so the claim under test is directly about the
users our migration guide targets. Meta's PromptGuard 2 is `gated: manual` and
could not be obtained.

**Corpus:** BIPIA via the harness's own `load_bipia`, so construction is
identical to the committed artifacts. 1000 obfuscated positives (all of Base
Encoding, Emoji Substitution, Substitution Ciphers, Reverse Text), 275
plain-language positives (25 sampled per family), 500 negatives (BIPIA's 50 clean
emails plus 450 dolly-15k contexts, for usable FPR resolution).

**Arms:**

| Arm | Definition | Inference cost |
|---|---|---|
| `baseline` | classifier on raw text | 1× |
| `+norm` | max(raw, classifier on canonical `normalize()`) | 2×, shippable |
| `+max(all)` | max over all thicket variants (27–57 per sample) | 27–57×, ceiling |

**Comparison at matched FPR.** Taking a max over variants can only *add*
positives, so both recall and false positives rise together; comparing raw recall
would be meaningless. The decision threshold is therefore swept to hold the
false-positive rate fixed across arms.

## Results, recall at matched FPR

| Target FPR | Arm | Obfuscated | Plain-language |
|---:|---|---:|---:|
| 1% | baseline | 14.9% | 6.9% |
| 1% | **+norm** | **15.3%** | 9.5% |
| 1% | +max(all) | 10.5% *(vs 12.5% subsample baseline)* | n/a |
| 2% | baseline | 22.0% | 17.5% |
| 2% | **+norm** | **29.9%** | 22.5% |
| 2% | +max(all) | 19.0% *(vs 15.0%)* | n/a |
| 5% | baseline | 53.0% | 46.2% |
| 5% | **+norm** | **59.0%** | 43.6% |
| 5% | +max(all) | 37.5% *(vs 47.5%)* | n/a |

## Findings

**1. The cheap arm gives a real but modest lift.** `+norm` adds ~6–8 points of
recall on obfuscated content at 2–5% FPR, for one extra inference. At 1% FPR the
lift is +0.4pp, i.e. noise. This is worth something, but it is not "we make your
classifier work", it is a few points on a subset of attacks, and it requires
running the classifier anyway.

**2. Naive variant fan-out actively hurts.** `+max(all)` is *worse than baseline*
at every operating point. Generating 27–57 variants produces enough
classifier-tripping text on the negatives to raise the FPR floor, so holding FPR
constant forces a much higher threshold and net recall falls. This is a useful
negative result in its own right: throwing every decoding at a classifier and
taking the max is an intuitive design that does not work.

**3. The finding that matters, the classifier beats our engine on our own best
surface.** At 5% FPR, deberta alone scores **53%** on the obfuscated categories.
open-thicket's rule engine scores 42.4% / 41.6% / 23.2% / 3.6% on those same four
categories (~28% weighted). The premise that ML classifiers are weak on
obfuscation and we are strong is not supported by this measurement. They are
better than us at it.

## Validity checks

Both ran before recording the negative result, since a flawed experiment would
have made it meaningless:

- `normalize()` alters the text in **1000/1000** obfuscated samples, so the
  treatment was genuinely applied in every case.
- **0/150** sampled inputs exceed the classifier's 512-token limit (median 131
  tokens, p90 183), so truncation is not suppressing the effect.

Limitations: thresholds are selected on the same negatives used for evaluation
(no held-out split), which inflates all arms equally and so does not affect the
comparison. `+max(all)` is measured on a 200-sample subsample with its own
baseline reported alongside. Single classifier, single corpus.

## Consequence

The "deobfuscation preprocessing" positioning is not supported. What survives is
narrower and should be stated plainly: open-thicket is a **deterministic,
offline prefilter** needing no model, no GPU, and no network, which is a real
advantage only where a model cannot be run at all, or where cost per request
dominates.

This paragraph read "sub-millisecond prefilter, roughly 50× faster than the
classifier" until 2026-08-08. Neither figure is measured anywhere in this
document, and no latency artifact from the SPEC-003 §6.1 reference environment
exists, so both fell under OT-ENG-019. Restore them with a citation if an
artifact backs them.
