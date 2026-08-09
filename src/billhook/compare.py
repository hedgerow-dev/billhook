"""Cross-detector comparison at matched false-positive rate (SPEC-011).

    python -m billhook.compare --suite bipia --suite notinject \\
        --detector regex-floor --detector thicket

Why matched FPR. Every tool ships its own threshold, so comparing default
outputs measures vendor risk appetite rather than detection ability, a
detector that fires on everything looks excellent on recall and is useless. We
therefore take each detector's continuous score, choose the threshold that
yields a fixed FPR on a benign corpus, and read recall at that point.

Detectors that expose only a boolean cannot be threshold-matched. They are run
and reported at their own fixed point, marked, and never silently compared as
if they were (OT-BCH-008).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from billhook.detectors import (
    DEFAULT_COMPARISON,
    AdapterUnavailable,
    Detector,
    Judgement,
    load,
)
from billhook.fetch import DEFAULT_DEST
from billhook.run import SUITES, DirtyTree, Sample, git_state, wilson_interval

DEFAULT_FPR_POINTS = (0.01, 0.05)


@dataclass
class DetectorRun:
    """Every judgement one detector made on one suite, kept unthresholded."""

    detector: str
    version: str
    kind: str
    requires: str
    suite: str
    pos_scores: list[float] = field(default_factory=list)
    neg_scores: list[float] = field(default_factory=list)
    pos_flags: list[bool] = field(default_factory=list)
    neg_flags: list[bool] = field(default_factory=list)
    groups: dict[str, list[int]] = field(default_factory=dict)
    errors: int = 0
    scoreless: bool = False

    @property
    def n_positive(self) -> int:
        return len(self.pos_flags)

    @property
    def n_negative(self) -> int:
        return len(self.neg_flags)

    def default_rates(self) -> tuple[float | None, float | None]:
        """Recall and FPR at the detector's own shipped threshold."""
        rec = (sum(self.pos_flags) / self.n_positive) if self.n_positive else None
        fpr = (sum(self.neg_flags) / self.n_negative) if self.n_negative else None
        return rec, fpr

    def threshold_for_fpr(self, target: float) -> float | None:
        """Lowest threshold whose FPR does not exceed `target`.

        Returns None when the detector has no scores, or when even the
        strictest threshold cannot reach the target, the honest answer is
        "this operating point is unreachable", not a silently clamped number.
        """
        if self.scoreless or not self.neg_scores:
            return None
        allowed = int(len(self.neg_scores) * target)
        ranked = sorted(self.neg_scores, reverse=True)
        if allowed >= len(ranked):
            return None
        cut = ranked[allowed]
        if allowed == 0 and cut >= 1.0:
            return None
        return cut + 1e-9

    def recall_at(self, threshold: float | None) -> float | None:
        if threshold is None or not self.pos_scores:
            return None
        return sum(s >= threshold for s in self.pos_scores) / len(self.pos_scores)

    def fpr_at(self, threshold: float | None) -> float | None:
        if threshold is None or not self.neg_scores:
            return None
        return sum(s >= threshold for s in self.neg_scores) / len(self.neg_scores)


def run_detector(det: Detector, samples: Sequence[Sample], surface: str,
                 suite: str) -> DetectorRun:
    run = DetectorRun(
        detector=det.name, version=det.version, kind=det.kind,
        requires=det.requires, suite=suite,
    )
    saw_score = False
    for sample in samples:
        j: Judgement = det.scan(sample.text, surface)
        if j.error:
            run.errors += 1
        if j.score is not None:
            saw_score = True
        value = j.score if j.score is not None else (1.0 if j.detected else 0.0)
        counts = run.groups.setdefault(sample.group, [0, 0, 0, 0])
        if sample.positive:
            run.pos_scores.append(value)
            run.pos_flags.append(j.detected)
            counts[0] += 1
            counts[1] += int(j.detected)
        else:
            run.neg_scores.append(value)
            run.neg_flags.append(j.detected)
            counts[2] += 1
            counts[3] += int(j.detected)
    run.scoreless = not saw_score
    return run


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def render(runs: list[DetectorRun], fpr_points: Sequence[float],
           benign_suite: str | None) -> str:
    """The comparison table. Every detector that ran appears (OT-BCH-003)."""
    out: list[str] = []
    by_suite: dict[str, list[DetectorRun]] = {}
    for r in runs:
        by_suite.setdefault(r.suite, []).append(r)

    out.append("## Comparison\n")
    out.append(
        "Recall at matched false-positive rate. Each detector's threshold is "
        "chosen on that suite's own negatives to hit the target FPR, so the "
        "columns are comparable; a detector's shipped default threshold is "
        "reported separately because it reflects vendor risk appetite rather "
        "than detection ability.\n"
    )
    for suite, suite_runs in by_suite.items():
        out.append(f"\n### `{suite}`\n")
        header = ["Detector", "Kind", "Requires", "n+", "n-", "default R/FPR"]
        header += [f"R @ {p * 100:.0f}% FPR" for p in fpr_points]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for r in sorted(suite_runs, key=lambda x: x.detector):
            rec, fpr = r.default_rates()
            row = [
                f"`{r.detector}`", r.kind, r.requires,
                str(r.n_positive), str(r.n_negative),
                f"{_pct(rec)} / {_pct(fpr)}",
            ]
            for p in fpr_points:
                if r.scoreless:
                    row.append("n/a¹")
                    continue
                thr = r.threshold_for_fpr(p)
                row.append(_pct(r.recall_at(thr)))
            out.append("| " + " | ".join(row) + " |")

        if any(r.scoreless for r in suite_runs):
            out.append(
                "\n¹ boolean-only detector: cannot be threshold-matched, so it "
                "appears only at its own shipped operating point (OT-BCH-008)."
            )
        for r in suite_runs:
            if r.n_positive and r.n_positive < 500:
                rec, _ = r.default_rates()
                if rec is not None:
                    lo, hi = wilson_interval(
                        int(sum(r.pos_flags)), r.n_positive
                    )
                    out.append(
                        f"\n- `{r.detector}` default recall 95% CI on n={r.n_positive}: "
                        f"[{lo * 100:.1f}-{hi * 100:.1f}]"
                    )
            if r.errors:
                out.append(
                    f"\n- ⚠ `{r.detector}` reported {r.errors} adapter error(s); "
                    f"those items counted as non-detections."
                )
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m billhook.compare",
        description="Compare injection detectors at matched FPR (SPEC-011).",
    )
    ap.add_argument("--detector", action="append", default=[], metavar="SPEC")
    ap.add_argument("--suite", action="append", default=[], metavar="NAME")
    ap.add_argument("--at-fpr", default=",".join(str(p) for p in DEFAULT_FPR_POINTS))
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per suite (development only; "
                         "artifacts recorded from a capped run are marked)")
    ap.add_argument("--out-dir", default="evals/results")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--release", action="store_true",
                    help="refuse to write from a dirty tree (OT-BCH-007)")
    args = ap.parse_args(argv)

    detector_specs = args.detector or list(DEFAULT_COMPARISON)
    suite_names = args.suite or ["bipia", "notinject"]
    fpr_points = tuple(float(x) for x in args.at_fpr.split(","))

    detectors: list[Detector] = []
    unavailable: list[tuple[str, str]] = []
    for spec in detector_specs:
        try:
            detectors.append(load(spec))
        except AdapterUnavailable as exc:
            unavailable.append((spec, str(exc)))
            print(f"SKIP {spec}: {exc}", file=sys.stderr)

    if not detectors:
        print("no detectors could be loaded", file=sys.stderr)
        return 2

    runs: list[DetectorRun] = []
    for suite_name in suite_names:
        suite = SUITES[suite_name]
        root = DEFAULT_DEST / suite.dataset
        if not root.is_dir():
            print(f"{root} is missing. Fetch it first:\n"
                  f"    python -m billhook.fetch --fetch {suite.dataset}",
                  file=sys.stderr)
            return 2
        samples = list(suite.loader(root))
        if args.limit:
            samples = samples[: args.limit]
        for det in detectors:
            print(f"  {suite_name} x {det.name} ({len(samples)} samples)...",
                  file=sys.stderr, flush=True)
            runs.append(run_detector(det, samples, suite.surface, suite_name))

    table = render(runs, fpr_points, None)
    print(table)

    git = git_state()
    if args.release and git.get("dirty"):
        raise DirtyTree("refusing to write a comparison from a dirty tree")

    if not args.no_write:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "git": git,
            "capped": args.limit,
            "fpr_points": list(fpr_points),
            "unavailable": [{"spec": s, "reason": r} for s, r in unavailable],
            "detectors": [
                {"name": d.name, "version": d.version, "kind": d.kind,
                 "requires": d.requires} for d in detectors
            ],
            "runs": [
                {"detector": r.detector, "suite": r.suite,
                 "n_positive": r.n_positive, "n_negative": r.n_negative,
                 "errors": r.errors, "scoreless": r.scoreless,
                 "default": r.default_rates(),
                 "matched": {str(p): r.recall_at(r.threshold_for_fpr(p))
                             for p in fpr_points}}
                for r in runs
            ],
        }
        (out_dir / f"{stamp}-comparison.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / f"{stamp}-comparison.md").write_text(table, encoding="utf-8")
        print(f"\nartifact -> {out_dir}/{stamp}-comparison.{{json,md}}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
