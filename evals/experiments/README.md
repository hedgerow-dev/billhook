# Experiment environment and exact conditions

Every finding in `docs/FINDINGS-classifier-surface-failure.md` is surface
sensitive: it depends on short strings, and tokenization or truncation choices
can move the numbers. So the conditions are recorded here rather than left
implicit.

## Environment

| | |
|---|---|
| Python | 3.14.6 |
| torch | 2.13.0 |
| transformers | 5.14.1 |
| tokenizers | 0.22.2 |
| device | CPU |
| precision | fp32 (no autocast, no quantization) |

## Scoring

All adapters live in `src/billhook/detectors.py`. For HuggingFace models:

```python
enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, dim=-1)
score = float(probs[0, injection_index])
```

- **max_length 512, truncation on.** Verified not to bind on this corpus:
  0/150 sampled BIPIA inputs exceed it (median 131 tokens, p90 183). It does
  bind on longer dolly passages, which is disclosed where relevant.
- **No padding** for single-item scoring, so batch composition cannot shift a
  score.
- **`injection_index` is resolved empirically, not from the config.** One model
  (`fmops/distilbert-prompt-injection`) ships unnamed `LABEL_0`/`LABEL_1`, and
  a flipped index would invert its entire row. The adapter prefers a label whose
  name contains `INJECT`, and every model was hand-checked against known
  injections and known benign text before its numbers were used.
- **Determinism**: `model.eval()`, no sampling, no dropout. Scores are
  bit-identical across reruns; this was verified rather than assumed.

## Model revisions

Models are loaded by bare id, so they resolve to whatever revision is in the
local HuggingFace cache. To pin exactly:

```bash
python - <<'PY'
from huggingface_hub import model_info
for m in ["protectai/deberta-v3-base-prompt-injection-v2",
          "protectai/deberta-v3-base-prompt-injection",
          "testsavantai/prompt-injection-defender-base-v0",
          "proventra/mdeberta-v3-base-prompt-injection",
          "tihilya/modernbert-base-prompt-injection-detection",
          "fmops/distilbert-prompt-injection",
          "devndeploy/bert-prompt-injection-detector"]:
    print(f"{m}  {model_info(m).sha}")
PY
```

Note that `protectai/deberta-v3-base-prompt-injection-v2` and LLM Guard are
**archived** (last repository activity 2026-07-08), so their revisions are
frozen and will not drift.

## Corpora

Fetched and checksum-pinned by `billhook.fetch`, never committed. Digests are
recorded in every result artifact under `evals/results/`.

## What each experiment establishes

| Script | Establishes |
|---|---|
| `classifier_fleet_survey.py` | Per-model false-positive rates by surface, AUC with bootstrap CI |
| `template_independence.py` | 50 BIPIA emails are 26 templates; 31/50 vs 16/26 |
| `clustered_bootstrap.py` | Email AUC CI resampled at template level: [0.437, 0.525], still straddling 0.5 |
| `format_instability.py` | Same content, five serializations, 20-62% |
| `register_trigger.py` | The minimal-pair ladder, agent gradient, and the not-an-evasion check |
| `register_across_fleet.py` | Which models show the pattern, gated on staying quiet on neutral prose |
| `register_heldout.py` | Out-of-sample validation, set frozen and hashed before scoring |
| `register_counterexamples.py` | **The control that narrowed our own claim.** Run this before believing us |
| `scaffolding_causal_test.py` | Additive control that killed the header mechanism |
| `notification_domain_sweep.py` | 7 domains x 4 nouns x {The,Your}. Kills the noun-domain hypothesis: `Your` lifts 9/28 to 19/28 across HR, cloud and healthcare, and the grid is incoherent within domains |
| `llmail_external_validation.py` | Second benign corpus and adaptive attacks. 0.0% FPR on 203 LLMail-Inject emails vs 62% on BIPIA, so the failure is a benign SUBGROUP effect |
| `corpus_format_crossed.py` | Crosses content against serialization. BIPIA falls 62% -> 28% stripped of its envelope, LLMail rises only to 2.0% wrapped in it: content load-bearing, format an amplifier |
| `agentivity_ladder.py` | Hypothesis 11: agentivity is neither necessary (4/7 fire with `you` as agent) nor sufficient. Retires the agency reading of the by-phrase gradient |
| `calibration_set_matters.py` | The same model at a 5% budget, calibrated three ways. Which benign corpus you tune on decides whether it looks excellent or useless |
| `ensemble_independence.py` | The off-diagonal of the fleet survey: do detectors fail on the SAME inputs? Misses are near-independent (median gap 1.02-1.08), false positives are not (2.17). At a matched 5% budget the best pair beats the best single by +1.0 / -1.4, and all ten members lose to one. Takes an optional budget argument |
| `occlusion_triggers.py` | The first POSITIVE cue, not another elimination. Leave-one-word-out surfaces the tokens the model leans on (`preauthorized`, `card`, `your`), a control rules out both token rarity and financial meaning, and `The lawn has been authorized` fires. See `docs/FINDINGS-token-triggers.md` |
| `trigger_label_association.py` | Where those tokens got their meaning: in public injection-classification corpora the firing tokens are enriched in the attack class (`authorized` 2.84x) and the silent financial words are depleted (`charged` 0.22x). Training-distribution proxies, not v2's actual training set. Needs pyarrow |

## A note on what these do not establish

None of it isolates the feature. The elimination ladder shows that eleven
candidate explanations are individually insufficient. That is a weaker and more
defensible claim than naming a mechanism, and it is deliberately the only claim
made.
