"""Is the classifier's score driven by content, or by serialization?

This experiment exists because it killed our previous one.

We had reported a "header ablation": removing `SUBJECT:`/`EMAIL_FROM:` fields
from BIPIA emails dropped protectai-v2's false-positive rate 62% -> 30%, with a
length-matched control (62% -> 58%) appearing to rule out length. We concluded
the headers were the trigger.

Two controls refuted it:

  ADDITIVE   prepending real email headers to 400 clean dolly passages produced
             0/400 new false positives. If headers caused FPs, they would cause
             them here.
  INTERACTION re-attaching each email's OWN header to its OWN body scores 38%,
             not the original 62%. If headers were the cause, restoring them
             would restore the number. It does not.

So the original "ablation" was substantially measuring the parse-and-rejoin
(whitespace, `|` structure, `CONTENT:` prefix) rather than header removal.

What survives is a stronger claim: the SAME email body, serialized five ways,
spans 22%-62% FPR. The score tracks surface form, not content. A detector whose
output swings 40 points on serialization is not measuring what it claims to.

Run:  python evals/experiments/format_instability.py
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from billhook.detectors import load  # noqa: E402
from billhook.fetch import DEFAULT_DEST  # noqa: E402

MODEL = "huggingface:protectai/deberta-v3-base-prompt-injection-v2"
HEADER_KEYS = (
    "SUBJECT", "EMAIL_FROM", "EMAIL_TO", "RECEIVED DATE", "FROM", "TO", "CC", "DATE",
)
SYNTHETIC = (
    "SUBJECT: Quarterly summary|EMAIL_FROM: Dana Whitfield "
    "<dana.whitfield@northbridge.example>"
)


def split_record(record: str) -> tuple[str, str]:
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
    det = load(MODEL)
    raw = DEFAULT_DEST / "bipia"
    emails = [
        json.loads(line)["context"]
        for line in (raw / "email_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    parsed = [split_record(e) for e in emails]
    headers = [h for h, _ in parsed]
    bodies = [b for _, b in parsed]

    def measure(texts: list[str]) -> tuple[float, float]:
        s = [det.scan(t, "tool_result").score or 0.0 for t in texts]
        return 100.0 * sum(x >= 0.5 for x in s) / len(s), statistics.median(s)

    variants = [
        ("full record, exactly as shipped", emails),
        ("body only (headers stripped)", bodies),
        (
            "body + its OWN header re-attached",
            [f"{h}|CONTENT: {b}" if h else b for h, b in zip(headers, bodies, strict=False)],
        ),
        (
            "body + a FOREIGN email's header",
            [
                f"{headers[(i + 7) % len(headers)]}|CONTENT: {b}"
                if headers[(i + 7) % len(headers)]
                else b
                for i, b in enumerate(bodies)
            ],
        ),
        (
            "body + one synthetic generic header",
            [f"{SYNTHETIC}|CONTENT: {b}" for b in bodies],
        ),
    ]

    print("=" * 72)
    print("SAME 50 EMAILS, SAME CONTENT, DIFFERENT SERIALIZATION")
    print("=" * 72)
    results = []
    for name, texts in variants:
        f, m = measure(texts)
        results.append(f)
        print(f"  {name:36} FPR={f:6.1f}%  median={m:.4f}")

    print()
    print(f"  spread: {min(results):.1f}% - {max(results):.1f}%  "
          f"({max(results) - min(results):.0f} points, identical content)")
    print()
    print("  Re-attaching each email's own header recovers only part of the")
    print("  original rate. The 'header ablation' therefore did not isolate")
    print("  headers; it isolated the reserialization. See module docstring.")


if __name__ == "__main__":
    main()
