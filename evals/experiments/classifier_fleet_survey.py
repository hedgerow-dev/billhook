"""Does the document-format failure generalise across the classifier fleet?

FINDINGS-classifier-surface-failure.md establishes, for ONE model, that:
  - false positives are surface-specific (0.1% on prompts, 62% on email)
  - the trigger is metadata scaffolding, not content (length-controlled)
  - on email the scores carry no usable signal (AUC ~0.49, CI straddles 0.5)

n=1 is an anecdote. This script runs the identical three tests across every
injection classifier available locally and prints one table, so the question
"is this one bad model or is it the whole ecosystem?" gets an answer.

It scores ONLY models already in the HuggingFace cache. Nothing is downloaded:
in a network-restricted environment a download attempt fails slowly and
confusingly, and a model that silently fails to load must never be reported as
a model that scored zero.

    python evals/experiments/classifier_fleet_survey.py
    python evals/experiments/classifier_fleet_survey.py --list

Populate the cache first, on a machine with normal internet:

    hf download protectai/deberta-v3-base-prompt-injection
    hf download fmops/distilbert-prompt-injection
    ...
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

SEED = 20260801
N_ATTACKS = 600
N_BOOTSTRAP = 2000
HEADER_KEYS = (
    "SUBJECT", "EMAIL_FROM", "EMAIL_TO", "RECEIVED DATE", "FROM", "TO", "CC", "DATE",
)

#: Candidates. Ungated ones first; the rest need terms accepted on the model page.
CANDIDATES = [
    "protectai/deberta-v3-base-prompt-injection-v2",
    "protectai/deberta-v3-base-prompt-injection",
    "fmops/distilbert-prompt-injection",
    "proventra/mdeberta-v3-base-prompt-injection",
    "testsavantai/prompt-injection-defender-base-v0",
    "tihilya/modernbert-base-prompt-injection-detection",
    "devndeploy/bert-prompt-injection-detector",
    # gated, need `hf auth login` and accepted terms
    "protectai/deberta-v3-small-prompt-injection-v2",
    "qualifire/prompt-injection-sentinel",
    "vijil/vijil_dome_prompt_injection_detection",
    "rogue-security/prompt-injection-jailbreak-sentinel-v2",
]


def cached_models() -> set[str]:
    """Model ids present in the local HF cache, without touching the network."""
    found = set()
    for root in (Path.home() / ".cache/huggingface/hub",):
        if not root.is_dir():
            continue
        for d in root.glob("models--*"):
            found.add(d.name[len("models--"):].replace("--", "/"))
    return found


def split_record(record: str) -> tuple[str, str]:
    """BIPIA emails are PIPE-delimited; a line-based regex would eat the body."""
    heads, body = [], []
    for seg in record.split("|"):
        m = re.match(r"\s*([A-Z_ ]{2,20})\s*:", seg)
        key = m.group(1).strip() if m else None
        if key in HEADER_KEYS:
            heads.append(seg.strip())
        elif key == "CONTENT":
            body.append(seg.split(":", 1)[1] if ":" in seg else seg)
        else:
            body.append(seg)
    return " | ".join(heads), "\n".join(body).strip()


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum(
        (1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg
    ) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="show which candidates are cached, then exit")
    args = ap.parse_args()

    available = cached_models()
    runnable = [m for m in CANDIDATES if m in available]
    missing = [m for m in CANDIDATES if m not in available]

    print(f"cached and runnable : {len(runnable)}")
    for m in runnable:
        print(f"   + {m}")
    if missing:
        print(f"not cached (skipped): {len(missing)}")
        for m in missing:
            print(f"   - {m}")
        print("\n   populate with:  hf download <model-id>")
    if args.list:
        return
    if not runnable:
        print("\nNothing to run.")
        return

    from billhook.detectors import AdapterUnavailable, load
    from billhook.fetch import DEFAULT_DEST
    from billhook.run import SUITES

    random.seed(SEED)
    raw = DEFAULT_DEST / "bipia"
    emails = [
        json.loads(line)["context"]
        for line in (raw / "email_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attack_strings = [
        a for v in json.loads((raw / "text_attack_test.json").read_text()).values()
        for a in v
    ]
    attack_texts = random.sample(
        [f"{e}\n{a}" for e in emails for a in attack_strings], N_ATTACKS
    )
    bodies = [split_record(e)[1] for e in emails]
    length_matched = [
        e[: len(b)] if len(b) < len(e) else e for e, b in zip(emails, bodies, strict=False)
    ]

    prose = [
        s.text for s in SUITES["dolly-context"].loader(DEFAULT_DEST / "dolly15k")
        if not s.positive
    ]
    prose = random.sample(prose, 600)
    notinject = [
        s.text for s in SUITES["notinject"].loader(DEFAULT_DEST / "notinject")
        if not s.positive
    ]

    rows = []
    for model_id in runnable:
        try:
            det = load(f"huggingface:{model_id}")
        except AdapterUnavailable as exc:
            print(f"SKIP {model_id}: {exc}")
            continue
        print(f"\nscoring {model_id} ...", flush=True)

        def sc(texts: list[str], _d: object = det) -> list[float]:
            # _d binds the loop variable at definition time (ruff B023):
            # a late-bound closure would score every model with the last
            # detector loaded and silently produce identical rows.
            return [_d.scan(t, "tool_result").score or 0.0 for t in texts]  # type: ignore[attr-defined]

        def fpr(v: list[float]) -> float:
            return 100.0 * sum(x >= 0.5 for x in v) / max(1, len(v))

        s_att = sc(attack_texts)
        s_mail = sc(emails)
        s_body = sc(bodies)
        s_len = sc(length_matched)
        s_prose = sc(prose)
        s_ni = sc(notinject)

        boots = []
        for _ in range(N_BOOTSTRAP):
            nb = [random.choice(s_mail) for _ in s_mail]
            pb = [random.choice(s_att) for _ in range(200)]
            boots.append(auc(pb, nb))
        boots.sort()
        lo, hi = boots[int(0.025 * N_BOOTSTRAP)], boots[int(0.975 * N_BOOTSTRAP) - 1]

        rows.append({
            "model": model_id,
            "fpr_prose": fpr(s_prose),
            "fpr_notinject": fpr(s_ni),
            "fpr_email": fpr(s_mail),
            "median_email": statistics.median(s_mail),
            "fpr_body": fpr(s_body),
            "fpr_lenmatched": fpr(s_len),
            "recall": fpr(s_att),
            "auc_email": auc(s_att, s_mail),
            "auc_ci": [lo, hi],
            "auc_prose": auc(s_att, s_prose),
        })

    print("\n" + "=" * 108)
    print("A. False positives by surface (threshold 0.5)")
    print("=" * 108)
    print(f"{'model':46} {'prose':>8} {'notinject':>10} {'email':>8} {'med':>7} {'recall':>8}")
    for r in rows:
        print(f"{r['model'][:46]:46} {r['fpr_prose']:7.1f}% {r['fpr_notinject']:9.1f}% "
              f"{r['fpr_email']:7.1f}% {r['median_email']:7.3f} {r['recall']:7.1f}%")

    print("\n" + "=" * 108)
    print("B. Header ablation with length control (clean emails)")
    print("=" * 108)
    print(f"{'model':46} {'as-shipped':>11} {'headers off':>12} {'len-matched':>12} {'delta':>8}")
    for r in rows:
        delta = r["fpr_email"] - r["fpr_body"]
        print(f"{r['model'][:46]:46} {r['fpr_email']:10.1f}% {r['fpr_body']:11.1f}% "
              f"{r['fpr_lenmatched']:11.1f}% {delta:7.1f}")

    print("\n" + "=" * 108)
    print("C. Separability, is any threshold usable on email?")
    print("=" * 108)
    print(f"{'model':46} {'AUC email':>10} {'95% CI':>18} {'AUC prose':>10}  verdict")
    for r in rows:
        lo, hi = r["auc_ci"]
        verdict = "CHANCE" if lo <= 0.5 <= hi else "separates"
        print(f"{r['model'][:46]:46} {r['auc_email']:9.3f} "
              f"[{lo:.3f}, {hi:.3f}]{'':>3} {r['auc_prose']:9.3f}  {verdict}")

    out = REPO / "evals" / "results" / "fleet-survey.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nartifact -> {out}")


if __name__ == "__main__":
    main()
