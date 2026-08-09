"""Checksum-pinned dataset fetching, OT-EVL-008, OT-RUL-007.

Every test here runs against a `file://` URL. Nothing in the test suite reaches
the network, and nothing in the test suite downloads an evaluation corpus.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:  # pragma: no cover - path bootstrap
        sys.path.insert(0, _p)

from billhook import fetch as ef  # noqa: E402


@pytest.fixture()
def local_dataset(tmp_path: Path, monkeypatch):
    """A one-file dataset served from disk, correctly pinned."""
    payload = b"synthetic evaluation payload\n"
    source = tmp_path / "source.jsonl"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    dataset = ef.Dataset(
        name="synthetic",
        homepage="https://example.com/synthetic",
        applies="test double",
        licence="CC0",
        files=(
            ef.PinnedFile(
                url=source.as_uri(), name="source.jsonl", sha256=digest, size_bytes=len(payload)
            ),
        ),
    )
    monkeypatch.setitem(ef.DATASETS, "synthetic", dataset)
    return dataset, source, digest, payload


def test_fetch_writes_the_file_when_the_checksum_matches(local_dataset, tmp_path: Path) -> None:
    _dataset, _source, _digest, payload = local_dataset
    written = ef.fetch("synthetic", tmp_path / "datasets")
    assert len(written) == 1
    assert written[0].read_bytes() == payload
    assert written[0].parent.name == "synthetic"


def test_fetch_raises_and_keeps_nothing_when_the_checksum_mismatches(
    local_dataset, tmp_path: Path, monkeypatch
) -> None:
    """OT-EVL-008: fail on mismatch. A partially-trusted file must not survive."""
    dataset, _source, _digest, _payload = local_dataset
    wrong = replace(dataset, files=(replace(dataset.files[0], sha256="0" * 64),))
    monkeypatch.setitem(ef.DATASETS, "synthetic", wrong)

    dest = tmp_path / "datasets"
    with pytest.raises(ef.ChecksumMismatch, match="expected sha256"):
        ef.fetch("synthetic", dest)
    assert list((dest / "synthetic").iterdir()) == []


def test_fetch_refuses_an_unpinned_dataset(tmp_path: Path) -> None:
    """OT-EVL-008. Uses `agentdojo`, which is unpinned on purpose and always will be.

    This named `notinject` until Phase 4 pinned it (B2). `agentdojo` cannot
    suffer the same fate: SPEC-010 §4 admits it only if actually run end to end,
    so there is deliberately no file to checksum, which makes it the stable
    fixture for this path.
    """
    with pytest.raises(ef.UnpinnedDataset, match="unreproducible benchmark"):
        ef.fetch("agentdojo", tmp_path)


def test_fetch_is_idempotent_and_skips_a_verified_file(local_dataset, tmp_path: Path) -> None:
    dest = tmp_path / "datasets"
    first = ef.fetch("synthetic", dest)
    mtime = first[0].stat().st_mtime_ns
    second = ef.fetch("synthetic", dest)
    assert second == first
    assert second[0].stat().st_mtime_ns == mtime


def test_fetch_redownloads_a_corrupted_file(local_dataset, tmp_path: Path) -> None:
    _dataset, _source, _digest, payload = local_dataset
    dest = tmp_path / "datasets"
    written = ef.fetch("synthetic", dest)
    written[0].write_bytes(b"tampered")
    assert ef.verify("synthetic", dest)
    ef.fetch("synthetic", dest)
    assert written[0].read_bytes() == payload
    assert ef.verify("synthetic", dest) == []


def test_verify_reports_a_missing_file(local_dataset, tmp_path: Path) -> None:
    problems = ef.verify("synthetic", tmp_path / "nothing-here")
    assert problems and "missing" in problems[0]


def test_unknown_dataset_raises_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        ef.fetch("no-such-dataset", tmp_path)


# -- registry hygiene -------------------------------------------------------


def test_default_destination_is_the_gitignored_datasets_directory() -> None:
    """OT-RUL-007. Fetched corpora land somewhere git will never pick them up."""
    assert ef.DEFAULT_DEST.name == "datasets"
    gitignore = (_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "evals/datasets/" in gitignore


def test_every_registered_dataset_records_a_licence_and_a_reason_to_exist() -> None:
    for dataset in ef.DATASETS.values():
        assert dataset.licence, f"{dataset.name}: no licence recorded"
        assert dataset.applies, f"{dataset.name}: no statement of what it measures"
        assert dataset.homepage.startswith("https://")


def test_unpinned_datasets_say_why_in_their_notes() -> None:
    for dataset in ef.DATASETS.values():
        if not dataset.pinned:
            assert "NOT PINNED" in dataset.notes, f"{dataset.name}"


def test_pinned_digests_are_full_sha256() -> None:
    for dataset in ef.DATASETS.values():
        for pinned in dataset.files:
            assert pinned.sha256 is not None
            assert len(pinned.sha256) == 64
            assert set(pinned.sha256) <= set("0123456789abcdef")


def test_injecagent_enhanced_split_is_not_pinned() -> None:
    """OT-EVL-006 retires the claim built on it; not having the file keeps it retired."""
    urls = [f.url for f in ef.DATASETS["injecagent-base"].files]
    assert all("enhanced" not in url for url in urls)


def test_harmful_content_benchmarks_are_absent_from_the_registry() -> None:
    """SPEC-010 §4: HarmBench measures a different capability. Not ours to run."""
    names = " ".join(ef.DATASETS).lower()
    assert "harmbench" not in names


def test_main_list_runs_without_network(capsys) -> None:
    assert ef.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "bipia" in out and "NOT PINNED" in out


def test_main_fetches_every_name_it_is_given(local_dataset, tmp_path: Path, capsys) -> None:
    """The documented quick start passes --fetch twice, and both must arrive.

    A non-repeatable flag silently kept only the last name, so the corpus the
    next command needed was never downloaded.
    """
    dataset, source, digest, payload = local_dataset
    second = ef.Dataset(
        name="synthetic-two",
        homepage="https://example.com/synthetic-two",
        applies="test double",
        licence="CC0",
        files=(ef.PinnedFile(url=source.as_uri(), name="source.jsonl", sha256=digest),),
    )
    ef.DATASETS["synthetic-two"] = second
    dest = tmp_path / "datasets"
    try:
        argv = ["--fetch", "synthetic", "--fetch", "synthetic-two", "--dest", str(dest)]
        assert ef.main(argv) == 0
    finally:
        del ef.DATASETS["synthetic-two"]
    assert (dest / "synthetic" / "source.jsonl").read_bytes() == payload
    assert (dest / "synthetic-two" / "source.jsonl").read_bytes() == payload


def test_main_reports_an_unknown_dataset_before_touching_the_network(capsys) -> None:
    assert ef.main(["--fetch", "no-such-dataset"]) == 2
    assert "unknown dataset 'no-such-dataset'" in capsys.readouterr().err
