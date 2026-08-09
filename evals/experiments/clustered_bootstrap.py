"""Template-clustered bootstrap CI for the email-surface AUC.

Reproduces the exact scoring conditions of classifier_fleet_survey.py
(same seed, same 600-attack sample, same adapter) but resamples at the
template level: the 26 near-duplicate clusters identified by
template_independence.py's rule, applied to BOTH the negatives (emails)
and the positives (email+attack cells inherit their email's cluster).

Prints the plain i.i.d. bootstrap alongside as a sanity check; it should
land near the published [0.389, 0.578].
"""
from __future__ import annotations

import difflib
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from billhook.detectors import load  # noqa: E402
from billhook.fetch import DEFAULT_DEST  # noqa: E402

SEED = 20260801
N_ATTACKS = 600
N_BOOTSTRAP = 2000
SIMILARITY = 0.75
PREFIX = 400
MODEL = "huggingface:protectai/deberta-v3-base-prompt-injection-v2"


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    return sum(
        (1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg
    ) / (len(pos) * len(neg))


def main() -> None:
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
    # Same construction order and same first draw after the seed as the
    # fleet survey, so the same 600 grid cells are selected.
    grid = [(i, j) for i in range(len(emails)) for j in range(len(attack_strings))]
    cells = random.sample(grid, N_ATTACKS)

    det = load(MODEL)

    def sc(t: str) -> float:
        return det.scan(t, "tool_result").score or 0.0

    print(f"scoring {len(emails)} emails + {len(cells)} attack cells ...", flush=True)
    s_mail = [sc(e) for e in emails]
    s_att = [sc(f"{emails[i]}\n{attack_strings[j]}") for i, j in cells]

    # template_independence.py's clustering rule, verbatim.
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
    print(f"{len(emails)} emails -> {len(clusters)} templates")

    cluster_of = {}
    for c_idx, members in enumerate(clusters):
        for m in members:
            cluster_of[m] = c_idx
    att_by_cluster: dict[int, list[float]] = {}
    for (i, _j), s in zip(cells, s_att, strict=True):
        att_by_cluster.setdefault(cluster_of[i], []).append(s)

    point = auc(s_att, s_mail)
    print(f"point AUC (all {len(s_att)} x {len(s_mail)}): {point:.4f}")

    # Plain i.i.d. bootstrap, replicating the published procedure.
    iid = []
    for _ in range(N_BOOTSTRAP):
        nb = [random.choice(s_mail) for _ in s_mail]
        pb = [random.choice(s_att) for _ in range(200)]
        iid.append(auc(pb, nb))
    iid.sort()
    print(f"i.i.d. bootstrap 95% CI : [{iid[50]:.4f}, {iid[1949]:.4f}]")

    # Template-clustered bootstrap: draw 26 clusters with replacement;
    # a drawn cluster contributes all its emails and all its attack cells.
    cl = []
    n_c = len(clusters)
    for _ in range(N_BOOTSTRAP):
        drawn = [random.randrange(n_c) for _ in range(n_c)]
        nb = [s_mail[m] for c in drawn for m in clusters[c]]
        pb = [s for c in drawn for s in att_by_cluster.get(c, [])]
        cl.append(auc(pb, nb))
    cl = [x for x in cl if x == x]  # drop NaN (no positives drawn)
    cl.sort()
    lo, hi = cl[int(0.025 * len(cl))], cl[int(0.975 * len(cl)) - 1]
    print(f"clustered bootstrap 95% CI ({len(cl)} valid draws): [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
