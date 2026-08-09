"""How many INDEPENDENT observations are behind the 62% email false-positive rate?

BIPIA ships 50 clean emails, but several are near-duplicates of one another:
the same Mercury or Deel notification template with a different amount or
merchant. If the flagged ones happened to be the duplicated ones, "62% of
business emails" would really be "one unlucky template, counted eight times",
and the headline would be an artifact of corpus construction rather than a
property of the model.

This checks that, because it is the most obvious way the finding could be
wrong and it costs nothing to rule out.

Result: it is not an artifact. Clustering at 0.75 similarity gives 26 distinct
templates, and weighting each template once instead of each email once moves
the rate from 62.0% to 61.5%.

The clusters are also strikingly bimodal, which is what pointed at a lexical
trigger in the first place: templates score ~1.0 or ~0.0 almost without
exception, so the model is not uncertain about these emails. It is confidently
sorting them into two groups, and register_trigger.py identifies what separates
the groups.

The honest consequence for the writeup is that the effective sample on this
surface is 26, not 50, which is why every rate here carries a bootstrap
interval.

Run:  python evals/experiments/template_independence.py
"""
from __future__ import annotations

import difflib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from billhook.detectors import load  # noqa: E402
from billhook.fetch import DEFAULT_DEST  # noqa: E402

MODEL = "huggingface:protectai/deberta-v3-base-prompt-injection-v2"
SIMILARITY = 0.75
PREFIX = 400  # chars compared when clustering


def main() -> None:
    det = load(MODEL)
    raw = DEFAULT_DEST / "bipia"
    emails = [
        json.loads(line)["context"]
        for line in (raw / "email_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = [det.scan(e, "tool_result").score or 0.0 for e in emails]

    used = [False] * len(emails)
    clusters: list[list[int]] = []
    for i in range(len(emails)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(emails)):
            if used[j]:
                continue
            ratio = difflib.SequenceMatcher(
                None, emails[i][:PREFIX], emails[j][:PREFIX]
            ).ratio()
            if ratio > SIMILARITY:
                group.append(j)
                used[j] = True
        clusters.append(group)

    print("=" * 72)
    print(f"{len(emails)} emails -> {len(clusters)} templates (>{SIMILARITY} similarity)")
    print("=" * 72)
    print(f"{'n':>3}  {'mean score':>10}  {'flagged':>8}   first 56 chars")
    for c in sorted(clusters, key=len, reverse=True)[:10]:
        mean = statistics.mean(scores[i] for i in c)
        flagged = sum(1 for i in c if scores[i] >= 0.5)
        sample = emails[c[0]][:56].replace("\n", " ")
        print(f"{len(c):>3}  {mean:10.3f}  {flagged:>4}/{len(c):<3}   {sample!r}")

    per_email = 100.0 * sum(1 for s in scores if s >= 0.5) / len(scores)
    per_template = 100.0 * sum(
        1 for c in clusters if statistics.mean(scores[i] for i in c) >= 0.5
    ) / len(clusters)

    print()
    print(f"  FPR, each EMAIL counted once     : {per_email:.1f}%  (n={len(emails)})")
    print(f"  FPR, each TEMPLATE counted once  : {per_template:.1f}%  (n={len(clusters)})")
    print()
    if abs(per_email - per_template) < 5:
        print("  Near-duplicates are NOT inflating the rate. The finding holds,")
        print(f"  but the effective sample is {len(clusters)}, not {len(emails)}.")
    else:
        print("  WARNING: the two rates diverge. The headline is substantially a")
        print("  corpus-construction artifact and must be reported per template.")

    bimodal = sum(
        1
        for c in clusters
        if not 0.1 < statistics.mean(scores[i] for i in c) < 0.9
    )
    print()
    print(f"  templates scoring <0.1 or >0.9   : {bimodal}/{len(clusters)}")
    print("  The model is not uncertain about these emails. It is confidently")
    print("  sorting them. See register_trigger.py for what separates the groups.")


if __name__ == "__main__":
    main()
