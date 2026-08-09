"""The benign corpus you calibrate on decides whether this detector is good.

`llmail_external_validation.py` produced a result we did not predict: with the
threshold calibrated on held-out LLMail benign email, protectai-v2 reaches 92.5%
recall at a 5% false-positive budget. On BIPIA transactional email the same
model, same protocol, is at 2.1%. Both numbers are real and neither is the whole
picture.

This experiment puts the two side by side and adds the condition a practitioner
actually faces: benign traffic that contains BOTH kinds of mail.

Design. Every corpus is split into calibration and evaluation halves at the
CLUSTER level, so no near-duplicate ever spans the split. The threshold is the
95th percentile of whichever calibration pool the condition names. Evaluation is
always on the held-out halves of both benign corpora plus both attack sets, so
the four conditions differ only in what the threshold was tuned on.

  A  human-authored business email      (LLMail-Inject FP test set, n=203)
  B  transactional business email       (BIPIA email_test, n=50)
  MIX  A + B pooled
  SHIPPED  no calibration, the 0.5 default

What we expect: calibrating on A alone puts the threshold near zero, which is
excellent against attacks and catastrophic the moment a real notification
arrives. Calibrating on B alone pushes the threshold to the top of the range and
buys silence at the cost of the entire detection capability. MIX should show
whether any middle ground exists, which is the question a deployment actually
has to answer.

Scores are cached to evals/results/calibration-matrix.json so the analysis can
be rerun without re-scoring.

Note on handling: attack corpora are adversarial text. Scored and counted, never
printed, never executed or followed.

Run:  python evals/experiments/calibration_set_matters.py
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

from billhook.detectors import load  # noqa: E402
from billhook.fetch import DEFAULT_DEST  # noqa: E402

MODEL = "huggingface:protectai/deberta-v3-base-prompt-injection-v2"
SEED = 20260802
N_ATTACKS = 600
N_SPLITS = 200
FPR_BUDGET = 0.05
SHIPPED = 0.5
SIMILARITY = 0.75
PREFIX = 400
CACHE = REPO / "evals" / "results" / "calibration-matrix.json"


def cluster(texts: list[str]) -> list[list[int]]:
    """The clustering rule from template_independence.py, unchanged."""
    used = [False] * len(texts)
    groups: list[list[int]] = []
    for i in range(len(texts)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(texts)):
            if used[j]:
                continue
            ratio = difflib.SequenceMatcher(
                None, texts[i][:PREFIX], texts[j][:PREFIX]
            ).ratio()
            if ratio > SIMILARITY:
                group.append(j)
                used[j] = True
        groups.append(group)
    return groups


def corpora() -> dict[str, list[str]]:
    random.seed(SEED)
    llm = DEFAULT_DEST / "llmail-inject"
    bip = DEFAULT_DEST / "bipia"

    benign_a = json.loads((llm / "emails_for_fp_tests.json").read_text())
    labelled = json.loads(
        (llm / "labelled_unique_submissions_phase2.json").read_text()
    )
    attacks_a = random.sample(
        [t for t, m in labelled.items()
         if m.get("reason") == "api_triggered"
         and str(m.get("attack_attempt")) == "True"],
        N_ATTACKS,
    )

    benign_b = [
        json.loads(line)["context"]
        for line in (bip / "email_test.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    ]
    strings = [
        a for v in json.loads((bip / "text_attack_test.json").read_text()).values()
        for a in v
    ]
    attacks_b = random.sample(
        [f"{e}\n{a}" for e in benign_b for a in strings], N_ATTACKS
    )
    return {
        "benign_a": benign_a, "benign_b": benign_b,
        "attacks_a": attacks_a, "attacks_b": attacks_b,
    }


def scores(data: dict[str, list[str]]) -> dict[str, list[float]]:
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if all(len(cached.get(k, [])) == len(v) for k, v in data.items()):
            print(f"using cached scores from {CACHE.name}")
            return cached
    det = load(MODEL)
    out = {}
    for key, texts in data.items():
        print(f"  scoring {key} ({len(texts)}) ...", flush=True)
        out[key] = [det.scan(t, "tool_result").score or 0.0 for t in texts]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=1))
    return out


def write_per_example(
    data: dict[str, list[str]],
    s: dict[str, list[float]],
    groups: dict[str, list[list[int]]],
) -> None:
    """One row per scored input, so every table here can be re-derived without
    running inference. Text is hashed rather than copied: the attack corpus is
    adversarial and the benign corpora have their own licences."""
    out = REPO / "evals" / "results" / "calibration-per-example.jsonl"
    cluster_of: dict[str, dict[int, int]] = {}
    for key, gs in groups.items():
        cluster_of[key] = {i: c for c, members in enumerate(gs) for i in members}
    with out.open("w", encoding="utf-8") as fh:
        for key, texts in data.items():
            for i, (text, score) in enumerate(zip(texts, s[key], strict=True)):
                fh.write(json.dumps({
                    "corpus": key,
                    "index": i,
                    "cluster": cluster_of.get(key, {}).get(i),
                    "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                    "chars": len(text),
                    "score": score,
                    "flagged_at_shipped": score >= SHIPPED,
                    "label": "attack" if key.startswith("attacks") else "benign",
                    "model": MODEL,
                }) + "\n")
    print(f"per-example artifact -> {out.name}")


def main() -> None:
    data = corpora()
    s = scores(data)
    groups_a = cluster(data["benign_a"])
    groups_b = cluster(data["benign_b"])
    write_per_example(data, s, {"benign_a": groups_a, "benign_b": groups_b})
    print(f"\nbenign A: {len(data['benign_a'])} emails -> {len(groups_a)} clusters")
    print(f"benign B: {len(data['benign_b'])} emails -> {len(groups_b)} clusters")

    def rate(v: list[float], thr: float) -> float:
        return 100.0 * sum(x >= thr for x in v) / max(1, len(v))

    conditions = ("A", "B", "MIX", "SHIPPED")
    acc: dict[str, dict[str, list[float]]] = {
        c: {"thr": [], "fpr_a": [], "fpr_b": [], "rec_a": [], "rec_b": []}
        for c in conditions
    }

    random.seed(SEED)
    for _ in range(N_SPLITS):
        def halves(groups: list[list[int]], key: str) -> tuple[list[float], list[float]]:
            idx = list(range(len(groups)))
            random.shuffle(idx)
            cut = len(idx) // 2
            cal = [s[key][i] for g in idx[:cut] for i in groups[g]]
            test = [s[key][i] for g in idx[cut:] for i in groups[g]]
            return cal, test

        cal_a, test_a = halves(groups_a, "benign_a")
        cal_b, test_b = halves(groups_b, "benign_b")

        for cond in conditions:
            if cond == "SHIPPED":
                thr = SHIPPED
            else:
                pool = {"A": cal_a, "B": cal_b, "MIX": cal_a + cal_b}[cond]
                pool = sorted(pool)
                k = max(0, round((1 - FPR_BUDGET) * len(pool)) - 1)
                thr = pool[k]
            acc[cond]["thr"].append(thr)
            acc[cond]["fpr_a"].append(rate(test_a, thr))
            acc[cond]["fpr_b"].append(rate(test_b, thr))
            acc[cond]["rec_a"].append(rate(s["attacks_a"], thr))
            acc[cond]["rec_b"].append(rate(s["attacks_b"], thr))

    label = {
        "A": "human business email",
        "B": "transactional email",
        "MIX": "both, pooled",
        "SHIPPED": "none (0.5 default)",
    }
    print("\n" + "=" * 92)
    print("Threshold calibrated at a 5% budget on ... then evaluated held-out")
    print("=" * 92)
    print(f"{'calibrated on':<22} {'threshold':>11} {'FPR human':>10} "
          f"{'FPR trans.':>11} {'rec adapt.':>11} {'rec static':>11}")
    print("-" * 92)
    for cond in conditions:
        a = acc[cond]
        m = statistics.median
        print(f"{label[cond]:<22} {m(a['thr']):>11.2e} {m(a['fpr_a']):>9.1f}% "
              f"{m(a['fpr_b']):>10.1f}% {m(a['rec_a']):>10.1f}% "
              f"{m(a['rec_b']):>10.1f}%")

    print("\n  spread across the 200 splits (5th-95th percentile):")
    for cond in conditions:
        a = acc[cond]

        def band(v: list[float]) -> str:
            v = sorted(v)
            return f"[{v[int(0.05 * len(v))]:5.1f}, {v[int(0.95 * len(v)) - 1]:5.1f}]"

        print(f"    {label[cond]:<22} FPR trans. {band(a['fpr_b'])}  "
              f"rec adapt. {band(a['rec_a'])}")
    print(f"\n  clusters: A {len(groups_a)} ({len(groups_a) // 2} calibration / "
          f"{len(groups_a) - len(groups_a) // 2} eval), "
          f"B {len(groups_b)} ({len(groups_b) // 2} / "
          f"{len(groups_b) - len(groups_b) // 2})")
    print("  thresholds are recomputed per split; the table shows the median.")
    print("  a record is flagged when score >= threshold, so a threshold of")
    print("  exactly 1.0 still flags every score saturated at 1.0.")

    # TPR at fixed FPR, which AUC does not tell you and deployment needs.
    print("\n" + "=" * 92)
    print("TPR AT FIXED FALSE-POSITIVE RATES (thresholds from the full benign set)")
    print("=" * 92)
    print(f"{'benign reference':<24} {'FPR budget':>11} {'threshold':>11} "
          f"{'rec adapt.':>11} {'rec static':>11}")
    print("-" * 92)
    for bkey, bname in (("benign_a", "human business email"),
                        ("benign_b", "transactional email")):
        pool = sorted(s[bkey])
        for budget in (0.001, 0.01, 0.05):
            k = max(0, round((1 - budget) * len(pool)) - 1)
            thr = pool[k]
            print(f"{bname:<24} {budget:>10.1%} {thr:>11.2e} "
                  f"{rate(s['attacks_a'], thr):>10.1f}% "
                  f"{rate(s['attacks_b'], thr):>10.1f}%")

    print("\n" + "=" * 92)
    print("READING")
    print("=" * 92)
    a, b, mix = acc["A"], acc["B"], acc["MIX"]
    m = statistics.median
    print(f"  Calibrate on human business mail : {m(a['rec_a']):.1f}% recall, but "
          f"{m(a['fpr_b']):.1f}% of transactional mail is flagged.")
    print(f"  Calibrate on transactional mail  : {m(b['fpr_b']):.1f}% FPR there, but "
          f"recall collapses to {m(b['rec_a']):.1f}%.")
    print(f"  Pool them                        : {m(mix['fpr_b']):.1f}% FPR on "
          f"transactional, {m(mix['rec_a']):.1f}% recall on adaptive attacks.")
    print()
    print("  Same model, same protocol, same 5% budget. The benign distribution")
    print("  you calibrate against is what decides the answer.")


if __name__ == "__main__":
    main()
