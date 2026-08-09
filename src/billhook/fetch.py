"""Checksum-pinned evaluation dataset fetching, OT-EVL-008, OT-RUL-007.

Evaluation datasets are **fetched at run time, checksum-pinned, and never
committed**. ``evals/datasets/`` is gitignored and this module is the only
supported way to populate it::

    python -m billhook.fetch --list
    python -m billhook.fetch --fetch bipia
    python -m billhook.fetch --verify bipia
    python -m billhook.fetch --pin <name>          # print digests for a new pin

Why this is a module and not a `curl` in a Makefile
---------------------------------------------------

The upstream repository committed roughly 10 MB of third-party corpora with no
LICENSE file and with terms asserted only in prose, while its own documentation
claimed they were not committed (SPEC-005 §3). OT-RUL-007 is therefore both a
reproducibility control and a licensing control, and it needs a single
enforceable entry point rather than a convention.

A pin is a promise about bytes, not about meaning. Every entry records the
upstream licence as reported by the host, and every entry that is not clearly
permissive says so. Read it before you fetch: this tool will download a corpus
for you, it will not tell you that you are allowed to.

**The corpus is not authored from these files.** Deriving a rule from a labelled
corpus, or inspecting evaluation false negatives to author one, is prohibited
(OT-RUL-006) and unenforceable after the fact, which is why
`evals/provenance_check.py` carries a tripwire for it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

__all__ = [
    "DATASETS",
    "DEFAULT_DEST",
    "ChecksumMismatch",
    "Dataset",
    "PinnedFile",
    "UnpinnedDataset",
    "fetch",
    "main",
    "verify",
]

# Working-data root. Package-relative paths break once installed (there is no
# repo root in site-packages), so datasets and results resolve against the
# caller's working directory and are overridable. See OT-BCH-012.
_REPO_ROOT = Path(os.environ.get("BILLHOOK_HOME", Path.cwd()))

#: Gitignored by SPEC-005 / OT-RUL-007. Nothing here is ever committed.
DEFAULT_DEST = _REPO_ROOT / "evals" / "datasets"

_TIMEOUT_S = 120.0
_CHUNK = 1 << 16
_USER_AGENT = "billhook/0.1 (+https://github.com/hedgerow-dev/billhook)"


class ChecksumMismatch(RuntimeError):
    """A downloaded file did not match its pin. The file is discarded."""


class UnpinnedDataset(RuntimeError):
    """A dataset entry has no recorded checksum, so it cannot be fetched."""


@dataclass(frozen=True, slots=True)
class PinnedFile:
    """One file, pinned by SHA-256 at an immutable upstream revision."""

    url: str
    name: str
    sha256: str | None
    size_bytes: int | None = None

    @property
    def pinned(self) -> bool:
        return bool(self.sha256)


@dataclass(frozen=True, slots=True)
class Dataset:
    """A fetchable evaluation corpus.

    Attributes:
        applies: Why this corpus measures injection detection, per SPEC-010 §4.
            Datasets that measure a different capability are not listed here at
            all, running a harmful-content benchmark against an injection
            detector measures the wrong thing (SPEC-010 §4).
        licence: The upstream licence as reported by the host. ``NOASSERTION``
            means the host found no licence file: the corpus may still be
            usable, but that is a judgement for a human, not for this tool.
    """

    name: str
    homepage: str
    applies: str
    licence: str
    files: tuple[PinnedFile, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def pinned(self) -> bool:
        return bool(self.files) and all(f.pinned for f in self.files)


DATASETS: dict[str, Dataset] = {
    "bipia": Dataset(
        name="bipia",
        homepage="https://github.com/microsoft/BIPIA",
        applies=(
            "Indirect injection in a real QA context. SPEC-010 §4 names this the "
            "primary honest measure."
        ),
        licence="NOASSERTION (GitHub's licence detector finds no SPDX licence file)",
        notes=(
            "Pinned at commit a004b69ec0dd446e0afd461d98cb5e96e120a5d0. Review the "
            "repository's own terms before using it in published material, the "
            "absence of a detected licence is exactly the condition OT-RUL-007 exists "
            "to stop us from papering over."
        ),
        files=(
            PinnedFile(
                url=(
                    "https://raw.githubusercontent.com/microsoft/BIPIA/"
                    "a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/text_attack_test.json"
                ),
                name="text_attack_test.json",
                sha256="75750e7b4e8b34e8f9d88d89b357aeaaf02bd07f9e493ccd37eda74a0cd7c7f8",
                size_bytes=6322,
            ),
            PinnedFile(
                url=(
                    "https://raw.githubusercontent.com/microsoft/BIPIA/"
                    "a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/email/test.jsonl"
                ),
                name="email_test.jsonl",
                sha256="217b403faaa1d0cb12c24892bea39f4a0b9e2819919ac7e2b902006e28278cdb",
                size_bytes=30334,
            ),
        ),
    ),
    "injecagent-base": Dataset(
        name="injecagent-base",
        homepage="https://github.com/uiuc-kang-lab/InjecAgent",
        applies=(
            "Tool-calling agent injection. SPEC-010 §2 requires reporting the BASE "
            "variant: the 'enhanced' variant prepends a fixed literal string to every "
            "sample, and any score computed on it is not comparable to anything."
        ),
        licence="MIT",
        notes=(
            "Pinned at commit f19c9f2c79a41046eb13c03c51a24c567a8ffa07. The enhanced "
            "split is deliberately not pinned here, OT-EVL-006 retires the claim it "
            "was used to make, and the easiest way to keep it retired is not to have "
            "the file."
        ),
        files=(
            PinnedFile(
                url=(
                    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/"
                    "f19c9f2c79a41046eb13c03c51a24c567a8ffa07/data/test_cases_dh_base.json"
                ),
                name="test_cases_dh_base.json",
                sha256="0a8186468d21389af432e8c7b399ae42264d1b93a07b65c7a489468508604305",
                size_bytes=672918,
            ),
            PinnedFile(
                url=(
                    "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/"
                    "f19c9f2c79a41046eb13c03c51a24c567a8ffa07/data/test_cases_ds_base.json"
                ),
                name="test_cases_ds_base.json",
                sha256="4daab35c62a3845e8b9400f4dca58b9c9f37e57cd33b2337552557fbb26282e9",
                size_bytes=765370,
            ),
        ),
    ),
    "notinject": Dataset(
        name="notinject",
        homepage="https://huggingface.co/datasets/leolee99/NotInject",
        applies=(
            "False-positive measurement on benign text deliberately loaded with "
            "injection trigger words. SPEC-010 §4."
        ),
        licence="MIT (dataset card `license: mit`; InjecGuard repository LICENSE is MIT)",
        notes=(
            "Pinned against the InjecGuard *source repository* at commit "
            "1b5751e88bf7475acbedfc8eda795ce060307c84, not against the Hugging Face "
            "hub copy. The hub copy of NotInject ships only Parquet, which would make "
            "the harness depend on pyarrow to read a 339-row corpus; the repository "
            "ships the same three splits as JSON. Both carry the same content and the "
            "same MIT licence. 3 x 113 = 339 rows, matching the figure in the paper "
            "(arXiv:2410.22770) and in SPEC-010 §5."
        ),
        files=(
            PinnedFile(
                url=(
                    "https://raw.githubusercontent.com/SaFoLab-WISC/InjecGuard/"
                    "1b5751e88bf7475acbedfc8eda795ce060307c84/datasets/NotInject_one.json"
                ),
                name="NotInject_one.json",
                sha256="69b535596d95102424e9c5946944feb4f2d596687eb8213f2ecad75478e5ffdd",
                size_bytes=26117,
            ),
            PinnedFile(
                url=(
                    "https://raw.githubusercontent.com/SaFoLab-WISC/InjecGuard/"
                    "1b5751e88bf7475acbedfc8eda795ce060307c84/datasets/NotInject_two.json"
                ),
                name="NotInject_two.json",
                sha256="6043d94e75b48d8e7682d25dc79eaf45359e1e561ce520e3b8fd5625a91060c6",
                size_bytes=29902,
            ),
            PinnedFile(
                url=(
                    "https://raw.githubusercontent.com/SaFoLab-WISC/InjecGuard/"
                    "1b5751e88bf7475acbedfc8eda795ce060307c84/datasets/NotInject_three.json"
                ),
                name="NotInject_three.json",
                sha256="ef01eff0d761d2e34571b3fdbcec08c30cd93efe8d0e1a2eb5c2baeb1873b070",
                size_bytes=35765,
            ),
        ),
    ),
    "jailbreakbench": Dataset(
        name="jailbreakbench",
        homepage="https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        applies=(
            "SPEC-010 §4 marks this ⚠️, not ✅. JBB-Behaviors is a list of harmful "
            "*request goals*, it measures harmful-content classification, which is "
            "a different capability from injection detection (SPEC-003 §2 has no "
            "harmful-content category). It is pinned and run anyway because "
            "OT-EVL-011 forbids excluding a benchmark post hoc for an unfavorable "
            "result: the number gets printed and the caveat gets argued in the open."
        ),
        licence="MIT (repository LICENSE file; dataset card `license: mit`)",
        notes=(
            "Pinned at revision 886acc352a31533ffbcf4ef22c744658688086fc. Both splits "
            "are pinned deliberately: the harmful split alone would let us report a "
            "recall without its FPR, which OT-EVL-004 forbids."
        ),
        files=(
            PinnedFile(
                url=(
                    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/"
                    "886acc352a31533ffbcf4ef22c744658688086fc/data/harmful-behaviors.csv"
                ),
                name="harmful-behaviors.csv",
                sha256="4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
                size_bytes=23116,
            ),
            PinnedFile(
                url=(
                    "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/"
                    "886acc352a31533ffbcf4ef22c744658688086fc/data/benign-behaviors.csv"
                ),
                name="benign-behaviors.csv",
                sha256="3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
                size_bytes=20570,
            ),
        ),
    ),
    "dolly15k": Dataset(
        name="dolly15k",
        homepage="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        applies=(
            "General benign corpus for false-positive measurement (SPEC-010 §4, "
            "'Public benign corpora'). 15,011 human-authored instruction records, "
            "each with an optional retrieved-passage `context` field, which makes it "
            "usable as a benign control on the prompt surface *and* on the "
            "tool-result / RAG surface, where the engine runs extra directive "
            "patterns and therefore has a different FPR."
        ),
        licence="CC BY-SA 3.0 (dataset card; records are human-authored by Databricks employees)",
        notes=(
            "Pinned at revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a. Chosen over "
            "the Alpaca-derived corpora the predecessor used because those are "
            "CC BY-NC 4.0, non-commercial, and this project ships under Apache 2.0. "
            "CC BY-SA 3.0 obliges attribution, which the result artifact carries."
        ),
        files=(
            PinnedFile(
                url=(
                    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
                    "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl"
                ),
                name="databricks-dolly-15k.jsonl",
                sha256="2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec",
                size_bytes=13085339,
            ),
        ),
    ),
    "llmail-inject": Dataset(
        name="llmail-inject",
        homepage="https://huggingface.co/datasets/microsoft/llmail-inject-challenge",
        applies=(
            "A second email surface, and the only adaptive one available. The benign "
            "half is business email sourced independently of BIPIA, which is what "
            "lets a result separate a benign SUBGROUP effect from a property of email "
            "in general. The attack half was written by people adapting to live "
            "defences, so recall on it is not comparable to recall on BIPIA's static "
            "strings and must not be pooled with it."
        ),
        licence="MIT",
        notes=(
            "Pinned at revision 1063bdf01ec8762b812d5e06ee768a06faa5a6f7. Only two "
            "of the eleven published files are pinned here: the false-positive email "
            "set, and the phase-2 labelled submissions. The phase-1 file (448 MB) and "
            "the raw submission logs (1.7 GB) are not fetched, no result needs them. "
            "Score only records whose `reason` is `api_triggered`: those attacks "
            "actually fired the tool call, whereas the rest carry an LLM judge's "
            "label. The attack text is adversarial by construction and is input to a "
            "classifier, never to an agent."
        ),
        files=(
            PinnedFile(
                url=(
                    "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/"
                    "resolve/1063bdf01ec8762b812d5e06ee768a06faa5a6f7/"
                    "data/emails_for_fp_tests.json"
                ),
                name="emails_for_fp_tests.json",
                sha256="4ddd950b5dbaa8548f5597c886d8e09a051ba07f80a9291fdcca9c2397d22abe",
                size_bytes=55486,
            ),
            PinnedFile(
                url=(
                    "https://huggingface.co/datasets/microsoft/llmail-inject-challenge/"
                    "resolve/1063bdf01ec8762b812d5e06ee768a06faa5a6f7/"
                    "data/labelled_unique_submissions_phase2.json"
                ),
                name="labelled_unique_submissions_phase2.json",
                sha256="f89af984e345430c3b357903890e30867bf4676f4ef10c138cc7bad218e890b8",
                size_bytes=68586935,
            ),
        ),
    ),
    "agentdojo": Dataset(
        name="agentdojo",
        homepage="https://github.com/ethz-spylab/agentdojo",
        applies=(
            "Agent-level attack success. SPEC-010 §4 admits it ONLY if actually run "
            "end to end: a regex match against extracted templates is not AgentDojo, "
            "and reporting it as such is the specific claim OT-EVL-001 retires."
        ),
        licence="MIT",
        notes=(
            "NOT PINNED, and deliberately so. AgentDojo is a harness, not a file: it "
            "has to be installed and executed against a model. There is nothing "
            "honest to checksum here, and pinning a data file would invite exactly "
            "the shortcut SPEC-010 §4 forbids."
        ),
    ),
}


def _sha256_and_copy(response: IO[bytes], out_path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with out_path.open("wb") as handle:
        while chunk := response.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
            handle.write(chunk)
    return digest.hexdigest(), size


def _download(url: str, out_path: Path) -> tuple[str, int]:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            return _sha256_and_copy(response, out_path)
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc


def fetch(name: str, dest: Path = DEFAULT_DEST, *, force: bool = False) -> list[Path]:
    """Fetch a pinned dataset, verifying every file's SHA-256.

    Downloads to a temporary file first and only moves it into place once the
    digest matches, so a mismatch can never leave a partially-trusted file where
    the harness would pick it up.

    Args:
        name: Key in :data:`DATASETS`.
        dest: Directory root. A subdirectory named after the dataset is created.
        force: Re-download files that are already present and already verify.

    Returns:
        The paths of the files now on disk.

    Raises:
        KeyError: unknown dataset.
        UnpinnedDataset: the entry has no recorded checksum (OT-EVL-008).
        ChecksumMismatch: a download did not match its pin. Nothing is kept.
    """
    dataset = DATASETS[name]
    if not dataset.pinned:
        raise UnpinnedDataset(
            f"{name} has no recorded checksum, so fetching it would produce an "
            f"unreproducible benchmark (OT-EVL-008). Pin it first:\n"
            f"    python -m billhook.fetch --pin {name}\n"
            f"{dataset.notes}"
        )

    target_dir = dest / name
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for pinned in dataset.files:
        final = target_dir / pinned.name
        if final.exists() and not force and _digest_of(final) == pinned.sha256:
            written.append(final)
            continue
        with tempfile.NamedTemporaryFile(dir=target_dir, delete=False, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
        try:
            actual, size = _download(pinned.url, tmp_path)
            if actual != pinned.sha256:
                raise ChecksumMismatch(
                    f"{name}/{pinned.name}: expected sha256 {pinned.sha256}, got {actual} "
                    f"({size} bytes from {pinned.url}). The file has been discarded. "
                    f"Either upstream changed under a pinned revision, or you are not "
                    f"talking to upstream."
                )
            shutil.move(str(tmp_path), str(final))
            written.append(final)
        finally:
            tmp_path.unlink(missing_ok=True)
    return written


def _digest_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(name: str, dest: Path = DEFAULT_DEST) -> list[str]:
    """Re-verify an already-fetched dataset. Returns a list of problems (empty is good)."""
    dataset = DATASETS[name]
    problems: list[str] = []
    if not dataset.pinned:
        return [f"{name}: not pinned, nothing to verify"]
    for pinned in dataset.files:
        path = dest / name / pinned.name
        if not path.exists():
            problems.append(f"{name}/{pinned.name}: missing")
            continue
        actual = _digest_of(path)
        if actual != pinned.sha256:
            problems.append(f"{name}/{pinned.name}: sha256 {actual}, expected {pinned.sha256}")
    return problems


def pin(name: str) -> int:
    """Download a dataset's files to a temp dir and print digests to record."""
    dataset = DATASETS[name]
    if not dataset.files:
        print(
            f"{name} has no file list yet. Add PinnedFile entries with the exact "
            f"immutable URLs you intend to pin, then re-run --pin.\n{dataset.notes}",
            file=sys.stderr,
        )
        return 2
    with tempfile.TemporaryDirectory() as tmpdir:
        for pinned_file in dataset.files:
            out = Path(tmpdir) / pinned_file.name
            actual, size = _download(pinned_file.url, out)
            print(f'{pinned_file.name}\n    sha256="{actual}",\n    size_bytes={size},')
    print(
        "\nRecord these in evals/fetch.py together with the upstream licence. "
        "A pin without a licence note is half a pin.",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m billhook.fetch",
        description="Fetch checksum-pinned evaluation datasets (OT-EVL-008).",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--fetch", metavar="NAME")
    parser.add_argument("--verify", metavar="NAME")
    parser.add_argument("--pin", metavar="NAME")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.list or not (args.fetch or args.verify or args.pin):
        for dataset in DATASETS.values():
            state = "pinned" if dataset.pinned else "NOT PINNED"
            print(f"{dataset.name:18} [{state}]  licence: {dataset.licence}")
            print(f"{'':18}   {dataset.homepage}")
            print(f"{'':18}   applies: {dataset.applies}")
            if dataset.notes:
                print(f"{'':18}   note: {dataset.notes}")
            print()
        return 0

    try:
        if args.pin:
            return pin(args.pin)
        if args.verify:
            problems = verify(args.verify, args.dest)
            for problem in problems:
                print(problem, file=sys.stderr)
            return 1 if problems else 0
        paths = fetch(args.fetch, args.dest, force=args.force)
        for path in paths:
            print(path)
        return 0
    except KeyError:
        print("unknown dataset; try --list", file=sys.stderr)
        return 2
    except (UnpinnedDataset, ChecksumMismatch, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
