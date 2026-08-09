"""Why does a prompt-injection classifier fire on clean business emails?

One model at 62% FPR is an anecdote. The mechanism is the finding. Four tests,
none of which need a second model:

  A. FPR across every benign surface available -> is the failure surface-specific?
  B. Header ablation -> is it the document FORMAT rather than the content?
  C. Separability (AUC) per surface -> is ANY threshold usable there?
  D. Bootstrap CI on C -> is the separability result real at this sample size?

C is what matters to a deployer, and D is what keeps C honest. A high FPR at the
default threshold is a tuning problem. Chance-level AUC is not tunable at all -
but with only 50 negatives available, C without D would be an overclaim, and the
CI does in fact change the wording of the conclusion.

Run:  python evals/experiments/classifier_surface_failure.py
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from billhook.detectors import load  # noqa: E402
from billhook.fetch import DEFAULT_DEST  # noqa: E402
from billhook.run import SUITES  # noqa: E402

SEED = 20260801
MODEL = "huggingface:protectai/deberta-v3-base-prompt-injection-v2"
N_ATTACK_SAMPLE = 750
N_BOOTSTRAP = 2000


def auc(pos: list[float], neg: list[float]) -> float:
    """AUROC via pairwise comparison. 0.5 == chance, i.e. no usable threshold."""
    if not pos or not neg:
        return float("nan")
    wins = sum(
        (1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg
    )
    return wins / (len(pos) * len(neg))


HEADER_KEYS = (
    "SUBJECT", "EMAIL_FROM", "EMAIL_TO", "RECEIVED DATE", "FROM", "TO", "CC", "DATE",
)


def split_record(record: str) -> tuple[str, str]:
    """Split a BIPIA email into (headers, body).

    BIPIA records are PIPE-delimited, not newline-delimited: the whole record is
    effectively one line. An earlier version of this experiment used a
    `^SUBJECT:.*$` MULTILINE regex, which therefore matched from the header to
    the end of the record and deleted most of the email, 8 of 50 records were
    cut below 10% of their original length. Any FPR drop measured that way
    confounds "headers removed" with "text destroyed", so it is not usable.
    """
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


def main() -> None:
    random.seed(SEED)
    det = load(MODEL)
    cache: dict[str, float] = {}

    def score(t: str) -> float:
        if t not in cache:
            cache[t] = det.scan(t, "tool_result").score or 0.0
        return cache[t]

    def fpr(scores: list[float], thr: float = 0.5) -> float:
        return 100.0 * sum(s >= thr for s in scores) / max(1, len(scores))

    def median(scores: list[float]) -> float:
        return sorted(scores)[len(scores) // 2]

    bipia = SUITES["bipia"]
    bipia_rows = list(bipia.loader(DEFAULT_DEST / bipia.dataset))
    clean_emails = [s.text for s in bipia_rows if not s.positive]
    attacks = random.sample(
        [s.text for s in bipia_rows if s.positive], N_ATTACK_SAMPLE
    )

    surfaces: dict[str, list[str]] = {}
    for name in ("notinject", "dolly-prompt", "dolly-context", "jailbreakbench"):
        suite = SUITES[name]
        rows = [
            s.text
            for s in suite.loader(DEFAULT_DEST / suite.dataset)
            if not s.positive
        ]
        surfaces[name] = rows if len(rows) <= 1200 else random.sample(rows, 1200)
    surfaces["bipia-clean-email"] = clean_emails

    print("=" * 74)
    print("A. False-positive rate by benign surface (threshold 0.5)")
    print("=" * 74)
    for name, rows in surfaces.items():
        sc = [score(t) for t in rows]
        print(
            f"  {name:22} n={len(rows):>5}  FPR={fpr(sc):6.1f}%   "
            f"median={median(sc):.3f}"
        )

    print()
    print("=" * 74)
    print("B. Header ablation, with a length-matched control")
    print("=" * 74)
    print("  Removing headers also shortens the text. The control cuts the full")
    print("  record to the same length while KEEPING the headers, so the two")
    print("  effects can be told apart.")
    print()
    parsed = [split_record(t) for t in clean_emails]
    bodies = [b for _, b in parsed]
    headers = [h for h, _ in parsed]
    length_matched = [
        e[: len(b)] if len(b) < len(e) else e for e, b in zip(clean_emails, bodies, strict=False)
    ]
    variants = {
        "as shipped (full record)": clean_emails,
        "body only (headers removed)": bodies,
        "length-matched, headers KEPT": length_matched,
        "headers only, body discarded": [
            h if h else e[:60] for h, e in zip(headers, clean_emails, strict=False)
        ],
    }
    for name, rows in variants.items():
        sc = [score(t) for t in rows]
        mean_len = sum(len(t) for t in rows) / len(rows)
        print(
            f"  {name:30} FPR={fpr(sc):6.1f}%  median={median(sc):.4f}  "
            f"meanlen={mean_len:.0f}"
        )
    down = sum(1 for e, b in zip(clean_emails, bodies, strict=False) if score(e) >= 0.5 > score(b))
    up = sum(1 for e, b in zip(clean_emails, bodies, strict=False) if score(b) >= 0.5 > score(e))
    print(f"\n  flagged -> clean when headers removed: {down}/{len(clean_emails)}")
    print(f"  clean -> flagged when headers removed: {up}/{len(clean_emails)}")

    print()
    print("=" * 74)
    print("C/D. Separability per surface, with bootstrap CI")
    print("=" * 74)
    print("  AUC 0.5 = chance. An interval containing 0.5 means no threshold on")
    print("  that surface separates attacks from benign better than a coin flip.")
    print()
    a_scores = [score(t) for t in attacks]
    for label, negs in (
        ("bipia clean emails", clean_emails),
        ("notinject", surfaces["notinject"]),
        ("dolly-context (ordinary prose)", surfaces["dolly-context"]),
    ):
        n_scores = [score(t) for t in negs]
        observed = auc(a_scores, n_scores)
        boots = []
        for _ in range(N_BOOTSTRAP):
            nb = [random.choice(n_scores) for _ in n_scores]
            pb = [random.choice(a_scores) for _ in range(200)]
            boots.append(auc(pb, nb))
        boots.sort()
        lo, hi = boots[int(0.025 * N_BOOTSTRAP)], boots[int(0.975 * N_BOOTSTRAP) - 1]
        verdict = "CHANCE" if lo <= 0.5 <= hi else "separates"
        print(
            f"  attacks vs {label:32} AUC={observed:.3f} "
            f"95% CI [{lo:.3f}, {hi:.3f}]  {verdict}"
        )

    print(f"\n({len(cache)} unique texts scored)")


if __name__ == "__main__":
    main()
