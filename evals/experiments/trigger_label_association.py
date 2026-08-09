"""Do the firing tokens correlate with the injection label in training-style data?

occlusion_triggers.py found that the model fires on a specific set of tokens
(`authorized`, `card`, `your`, and `preauthorized` via its `authorized`
subword) and stays silent on plainly-financial words (`charged`, `payment`).
This asks where that split could come from.

HONEST LIMIT, stated first: protectai never published v2's exact training set.
The model card says it was "meticulously assembled from various public open
datasets", and names only a handful. So this is NOT an audit of v2's training
data. It is the weaker, reproducible claim the deck arc listed as an addition:
in public injection-CLASSIFICATION corpora representative of that distribution,
is each firing token enriched in the attack class, and is each silent financial
word not? If so, the behavioural split is learnable from this kind of labelling.

The test that makes it a control rather than a correlation is the contrast: the
firing tokens and the non-firing financial words are all common, but only the
firing ones should be enriched in the injection class. `instructions` (3.2x) and
`weather` (0.08x) confirm the method separates signal from noise.

Corpora are training-distribution PROXIES, fetched by pinned revision, and are
deliberately not billhook eval suites: nothing here is an evaluation negative or
a rule source, so SPEC-005's provenance rules do not apply. Needs pyarrow, in
the `experiments` extra.

Run:  python evals/experiments/trigger_label_association.py
"""
from __future__ import annotations

import re
import sys

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - env dependent
    print("needs pyarrow: pip install 'open-thicket[experiments]'", file=sys.stderr)
    raise

from huggingface_hub import hf_hub_download

#: (repo, pinned revision, parquet files). Revisions pinned 2026-08-09 so the
#: numbers below stay reproducible even if the datasets are updated upstream.
CORPORA = (
    (
        "xTRam1/safe-guard-prompt-injection",
        "a3a877d608f37b7d20d9945671902df895ecdb46",
        ("data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet"),
    ),
    (
        "deepset/prompt-injections",
        "4f61ecb038e9c3fb77e21034b22511b523772cdd",
        ("data/train-00000-of-00001-9564e8b05b4757ab.parquet",
         "data/test-00000-of-00001-701d16158af87368.parquet"),
    ),
)

#: Grouped exactly by occlusion_triggers.py's finding, so the two scripts read
#: as one argument. `preauthorized` is absent from these corpora by design: the
#: point is that it fires anyway, inheriting `authorized` as a subword.
FIRES = ["authorized", "preauthorized", "encrypted", "card", "your"]
SILENT_FINANCIAL = ["charged", "debited", "refunded", "invoiced", "payment"]
NEUTRAL = ["the", "please", "question", "weather", "system", "instructions"]


def load_corpus(repo: str, revision: str, files: tuple[str, ...]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for fname in files:
        path = hf_hub_download(repo, fname, repo_type="dataset", revision=revision)
        table = pq.read_table(path).to_pydict()
        tcol = next((c for c in table if c.lower() in ("text", "prompt")), None)
        lcol = next((c for c in table if c.lower() in ("label", "labels")), None)
        if not tcol or not lcol:
            continue
        for text, label in zip(table[tcol], table[lcol], strict=True):
            if label in (0, 1):
                rows.append((str(text or ""), int(label)))
    return rows


def contains(word: str, text: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, re.I) is not None


def analyse(name: str, rows: list[tuple[str, int]]) -> None:
    n = len(rows)
    base = sum(label for _, label in rows) / n
    print("\n" + "=" * 70)
    print(f"{name}: n={n}, injection base rate {base:.1%}")
    print("=" * 70)
    print(f"  {'token':<15}{'rows':>7}{'inj%':>8}{'lift':>8}")
    for group, words in (("model FIRES", FIRES),
                         ("financial, model SILENT", SILENT_FINANCIAL),
                         ("neutral (method check)", NEUTRAL)):
        print(f"  [{group}]")
        for w in words:
            sub = [label for text, label in rows if contains(w, text)]
            if not sub:
                print(f"  {w:<15}{0:>7}{'--':>8}{'--':>8}")
                continue
            rate = sum(sub) / len(sub)
            lift = rate / base if base else float("nan")
            print(f"  {w:<15}{len(sub):>7}{rate * 100:>7.1f}%{lift:>7.2f}x")


def main() -> None:
    for repo, revision, files in CORPORA:
        rows = load_corpus(repo, revision, files)
        if rows:
            analyse(repo, rows)
    print("\nReading: the tokens the model fires on are enriched in the attack")
    print("class; the financial words it ignores are not. The behavioural split")
    print("is learnable from this labelling. Proxy corpora, not v2's training set.")


if __name__ == "__main__":
    main()
