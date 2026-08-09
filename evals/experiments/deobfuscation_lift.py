"""Does open-thicket normalization lift a real classifier on obfuscated injection?

Classifier: protectai/deberta-v3-base-prompt-injection-v2 (Apache-2.0, ungated,
~285k downloads/month). This is the model LLM Guard's PromptInjection scanner
used, so "we lift the classifier your migrating users already run" is the claim
under test.

Arms:
  baseline    classifier(raw text)                      1 inference
  arm_norm    max(raw, classifier(canonical normalize)) 2 inferences  <- shippable
  arm_max     max over ALL thicket variants             27-57 inferences <- ceiling

Max-over-variants can only ADD positives, so recall AND false positives both
rise. Comparing raw recall would be meaningless. We therefore sweep the decision
threshold and report recall at MATCHED false-positive rate.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO = Path("/Users/kennethcox/Developer/projects/open-thicket")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from thicket.normalize import normalize, variants  # noqa: E402

from billhook.run import load_bipia  # noqa: E402

MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
OBFUSCATION_GROUPS = {
    "Base Encoding", "Emoji Substitution", "Substitution Ciphers", "Reverse Text",
}
PLAIN_PER_GROUP = 25
N_NEG_DOLLY = 450
MAX_ARM_POSITIVES = 200   # arm_max is 30x cost; subsample it
SEED = 20260801


def variant_texts(text: str) -> list[str]:
    out = []
    for v in variants(text):
        t = v.text if hasattr(v, "text") else (v[1] if isinstance(v, tuple) else v)
        if isinstance(t, str) and t:
            out.append(t)
    return out


def main() -> None:
    random.seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    torch.set_num_threads(8)
    inj_idx = next(
        (i for i, lab in model.config.id2label.items() if "INJECT" in str(lab).upper()), 1
    )
    print(f"model labels={model.config.id2label} injection_index={inj_idx}", flush=True)

    cache: dict[str, float] = {}

    def score(texts: list[str]) -> list[float]:
        todo = [t for t in dict.fromkeys(texts) if t not in cache]
        for i in range(0, len(todo), 16):
            batch = todo[i : i + 16]
            enc = tok(batch, truncation=True, max_length=512, padding=True,
                      return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)[:, inj_idx]
            for t, p in zip(batch, probs.tolist(), strict=False):
                cache[t] = p
        return [cache[t] for t in texts]

    # ---- assemble the sample ------------------------------------------------
    pos_obf, pos_plain_by_group, neg = [], {}, []
    for s in load_bipia(REPO / "evals" / "datasets" / "bipia"):
        if s.positive:
            fam = s.group.split("/")[-1].strip()
            (pos_obf if fam in OBFUSCATION_GROUPS
             else pos_plain_by_group.setdefault(fam, [])).append((fam, s.text))
        else:
            neg.append(s.text)

    pos_plain = []
    for fam, rows in pos_plain_by_group.items():
        pos_plain.extend(random.sample(rows, min(PLAIN_PER_GROUP, len(rows))))

    dolly = REPO / "evals" / "datasets" / "dolly15k"
    dolly_ctx = []
    for line in (next(dolly.glob("*.jsonl"))).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        ctx = (rec.get("context") or "").strip()
        if len(ctx) > 40:
            dolly_ctx.append(ctx)
    neg = neg + random.sample(dolly_ctx, min(N_NEG_DOLLY, len(dolly_ctx)))

    print(f"positives: {len(pos_obf)} obfuscated, {len(pos_plain)} plain | "
          f"negatives: {len(neg)}", flush=True)

    # ---- score --------------------------------------------------------------
    def arm_scores(rows: list[str], do_max: bool) -> tuple[list[float], list[float], list[float]]:
        base = score(rows)
        norm_s = score([normalize(t) for t in rows])
        a_norm = [max(b, n) for b, n in zip(base, norm_s, strict=False)]
        a_max = []
        if do_max:
            for t in rows:
                vs = variant_texts(t)[:40]
                a_max.append(max(score(vs)) if vs else 0.0)
        return base, a_norm, a_max

    obf_txt = [t for _, t in pos_obf]
    plain_txt = [t for _, t in pos_plain]
    sub = random.sample(range(len(obf_txt)), min(MAX_ARM_POSITIVES, len(obf_txt)))

    print("scoring obfuscated positives...", flush=True)
    o_base, o_norm, _ = arm_scores(obf_txt, False)
    print("scoring obfuscated positives (max arm, subsample)...", flush=True)
    _, _, o_max = arm_scores([obf_txt[i] for i in sub], True)
    print("scoring plain positives...", flush=True)
    p_base, p_norm, _ = arm_scores(plain_txt, False)
    print("scoring negatives...", flush=True)
    n_base, n_norm, n_max = arm_scores(neg, True)

    # ---- recall at matched FPR ---------------------------------------------
    def thr_for_fpr(negs: list[float], target: float) -> float:
        s = sorted(negs, reverse=True)
        k = int(len(s) * target)
        return (s[k] if k < len(s) else 0.0) + 1e-9

    def recall(scores: list[float], thr: float) -> float:
        return 100.0 * sum(x >= thr for x in scores) / max(1, len(scores))

    print("\n" + "=" * 78)
    print("RECALL AT MATCHED FALSE-POSITIVE RATE (BIPIA, tool-result content)")
    print("=" * 78)
    print(f"{'target FPR':>10} | {'arm':<10} | {'obfuscated':>11} | {'plain-lang':>11}")
    print("-" * 78)
    for target in (0.01, 0.02, 0.05):
        t_b = thr_for_fpr(n_base, target)
        t_n = thr_for_fpr(n_norm, target)
        t_m = thr_for_fpr(n_max, target)
        o_sub_base = [o_base[i] for i in sub]
        rows = [
            ("baseline", recall(o_base, t_b), recall(p_base, t_b)),
            ("+norm", recall(o_norm, t_n), recall(p_norm, t_n)),
            ("+max(all)", recall(o_max, t_m), float("nan")),
        ]
        for name, ro, rp in rows:
            rp_s = "     n/a" if rp != rp else f"{rp:10.1f}%"
            print(f"{target*100:9.0f}% | {name:<10} | {ro:10.1f}% | {rp_s}")
        print(f"{'':>10} | {'(max arm subsample baseline)':<10}   "
              f"{recall(o_sub_base, t_b):9.1f}%")
        print("-" * 78)

    _out = open("/tmp/deobf_result.json", "w")
    json.dump(
        {"model": MODEL, "n_obf": len(o_base), "n_plain": len(p_base),
         "n_neg": len(n_base), "n_max_sub": len(o_max),
         "scores": {"obf_base": o_base, "obf_norm": o_norm, "obf_max": o_max,
                    "plain_base": p_base, "plain_norm": p_norm,
                    "neg_base": n_base, "neg_norm": n_norm, "neg_max": n_max}},
        _out,
    )
    print("\nraw scores -> /tmp/deobf_result.json")


if __name__ == "__main__":
    main()
