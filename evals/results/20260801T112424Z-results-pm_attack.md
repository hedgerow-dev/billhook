# open-thicket evaluation results

> **NOT PUBLISHABLE.** OT-ENG-019: latency fields were not measured on the SPEC-003 §6.1 reference environment and are not publishable. Detection counts are.

- Reproduce: `python -m evals.run --suite attack`
- Commit: `df5e4fa6a9e69bccb57d96c2e7c3fb3662e5775a` (dirty: `False`)
- Ruleset: `sha256:8eede3ff126a98cde3d56bd5335cf12270cc0829c5750a8f84a624cc7534c4c4` (56 enabled of 57)
- Generated: 2026-08-01T11:24:24+00:00

Whether the engine emits a detection on a piece of text. It does not run an agent and does not call a model, so it does not measure attack success or attack prevention. AgentDojo is deliberately absent: SPEC-010 §4 admits it only if actually run end to end.

## Results

Every rate carries its sample size, and every rate on n < 500 carries a 95% Wilson interval in brackets (OT-EVL-005). Recall and FPR always appear together (OT-EVL-004); where a suite has no negatives of its own, the FPR column names the companion benign suite it is paired with.

| Suite | Surface | n | Recall | FPR | Precision |
|---|---|---:|---|---|---|
| `bipia` | tool_result | 3800 | 9.3% | 4.0% [1.1-13.5] | 99.4% |
| `injecagent-base` | tool_result | 1071 | 33.1% | 5.9% [1.0-27.0] | 99.7% |

### `bipia`

**Measures.** Whether the engine flags an indirect-injection instruction sitting in retrieved content. SPEC-010 §4 names this our primary honest measure. It does not measure whether an agent would have obeyed the instruction.

**Construction.** Positives: each of the 50 emails in BIPIA's email/test.jsonl concatenated with each of the 75 attack instructions in text_attack_test.json, joined by a newline, attack appended at the end (50 x 75 = 3750). Negatives: the same 50 emails unmodified. Scanned as tool-result content via Engine.scan_tool_result. This is BIPIA's data and insertion procedure applied to the external content only; it is not a byte-identical replay of BIPIA's model-prompt template.

**Counts.** n=3800 (positive 3750, negative 50); TP 347, FN 3403, FP 2, TN 48; engine errors 0.

**Dataset.** `bipia` sha256 `6a6c9058bc0d824b…`

| Group | n+ | recall | n- | FPR |
|---|---:|---|---:|---|
| Base Encoding | 250 | 23.2% [18.4-28.8] | 0 | n/a |
| Business Intelligence | 250 | 3.2% [1.6-6.2] | 0 | n/a |
| Conversational Agent | 250 | 3.2% [1.6-6.2] | 0 | n/a |
| Emoji Substitution | 250 | 41.6% [35.7-47.8] | 0 | n/a |
| Entertainment | 250 | 2.0% [0.9-4.6] | 0 | n/a |
| Information Dissemination | 250 | 2.0% [0.9-4.6] | 0 | n/a |
| Language Translation | 250 | 4.0% [2.2-7.2] | 0 | n/a |
| Marketing & Advertising | 250 | 2.0% [0.9-4.6] | 0 | n/a |
| Misinformation & Propaganda | 250 | 2.0% [0.9-4.6] | 0 | n/a |
| Research Assistance | 250 | 2.0% [0.9-4.6] | 0 | n/a |
| Reverse Text | 250 | 3.6% [1.9-6.7] | 0 | n/a |
| Scams & Fraud | 250 | 2.8% [1.4-5.7] | 0 | n/a |
| Sentiment Analysis | 250 | 2.0% [0.9-4.6] | 0 | n/a |
| Substitution Ciphers | 250 | 42.4% [36.4-48.6] | 0 | n/a |
| Task Automation | 250 | 2.8% [1.4-5.7] | 0 | n/a |
| clean email | 0 | n/a | 50 | 4.0% [1.1-13.5] |

### `injecagent-base`

**Measures.** Whether the engine flags an attacker instruction embedded in a tool-calling agent's tool output. Detection here is not the same as InjecAgent's own attack-success-rate metric, which requires running an agent against a model; this harness runs neither.

**Construction.** Positives: the `Tool Response` field of every base test case in test_cases_dh_base.json and test_cases_ds_base.json, each verified on load to contain its `Attacker Instruction`. Negatives: the distinct `Tool Response Template` values with the `<Attacker Instruction>` placeholder replaced by the empty string. BASE variant only; the enhanced variant is not fetched (OT-EVL-006).

**Counts.** n=1071 (positive 1054, negative 17); TP 349, FN 705, FP 1, TN 16; engine errors 0.

**Dataset.** `injecagent-base` sha256 `de600ef41e90bc3b…`

| Group | n+ | recall | n- | FPR |
|---|---:|---|---:|---|
| dh/Data Security Harm | 187 | 34.8% [28.3-41.8] | 0 | n/a |
| dh/Financial Harm | 153 | 24.2% [18.1-31.5] | 0 | n/a |
| dh/Physical Harm | 170 | 30.6% [24.2-37.9] | 0 | n/a |
| dh/clean template | 0 | n/a | 17 | 5.9% [1.0-27.0] |
| ds/Financial Data | 102 | 32.4% [24.1-41.9] | 0 | n/a |
| ds/Others | 255 | 31.8% [26.4-37.7] | 0 | n/a |
| ds/Physical Data | 187 | 43.3% [36.4-50.5] | 0 | n/a |

## Latency (informational)

NOT the SPEC-003 §6.1 reference environment. Under OT-ENG-019 the latency fields in this artifact are a development signal and are not publishable. The detection counts are unaffected: they are deterministic and environment-independent.

These are per-scan wall times over each suite's own heterogeneous input sizes. They are a distribution, not a fixed-size measurement; SPEC-003 §6.1's controlled figures come from `evals/bench.py`.

| Suite | scans | mean | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| `bipia` | 3800 | 2.484 | 2.389 | 4.010 | 4.692 | 6.947 |
| `injecagent-base` | 1071 | 0.999 | 0.880 | 2.298 | 2.848 | 3.732 |

<sub>Times in milliseconds. `databricks-dolly-15k` © Databricks, CC BY-SA 3.0.</sub>
