"""The benchmark runner, SPEC-010 §3, OT-EVL-001..011.

    python -m billhook.run --list
    python -m billhook.run --suite v0_1                 # fetch, score, write artifact
    python -m billhook.run --suite v0_1 --release       # + refuse a dirty working tree

This module produces the *only* thing that may back a public number
(OT-EVL-001). Everything it does is arranged around one question: if a stranger
re-runs the printed command on a clean checkout, do they get this file back?

Four design choices follow from that and are worth stating before the code.

**Every suite is scored, and the artifact holds them all.** There is no flag to
drop a suite from the output. OT-EVL-003 prohibits reporting the best dataset
and omitting the worst, and the cheapest way to keep that promise is to make
selective reporting require editing a committed JSON file rather than passing an
argument. JailbreakBench in particular is wired in permanently (OT-EVL-011).

**No recall is ever emitted without an FPR.** OT-EVL-004 makes recall
unpublishable on its own, so every suite that has positives must either carry
its own negatives or name a companion benign suite in `fpr_companion`.
:func:`render_markdown` raises rather than print a recall cell with an empty FPR
cell, a formatting error is a cheap way to enforce a claims rule.

**False negatives are counted, never captured.** OT-EVL-010 forbids writing
evaluation FN text into the repository, and the predecessor committed verbatim
FN text for every holdout it ran. There is deliberately no `--dump-misses`
option here, not even one guarded to write outside the repository: the per-group
breakdowns below answer "which attack families were missed" without materializing
a single missed string, and a tool that materializes them is an attractive
nuisance next to OT-RUL-006 ("inspecting evaluation false negatives to author
patterns is prohibited").

**The construction is recorded, not just the score.** A benchmark name is not a
condition. BIPIA in particular is a *procedure* for building attacked contexts,
not a file of samples, so `Suite.construction` records exactly how this harness
assembled what it scored and which scan surface it used. OT-EVL-006 is about not
comparing an easy condition to a hard one; you cannot honour it if the condition
is not written down.

A note on what this harness does **not** measure
------------------------------------------------

It measures whether a detector flags a piece of text. It does not run an agent,
does not call a model, and therefore does not measure attack *success*. A
detection here is not a blocked exploit and the artifact says so in
`what_this_measures`. AgentDojo is excluded for exactly this reason, SPEC-010
§4 admits it only if actually run end to end, and a regex pass over extracted
templates is not AgentDojo. That is the specific claim OT-EVL-001 retires.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Working-data root. Package-relative paths break once installed (there is no
# repo root in site-packages), so datasets and results resolve against the
# caller's working directory and are overridable. See OT-BCH-012.
_REPO_ROOT = Path(os.environ.get("BILLHOOK_HOME", Path.cwd()))

from billhook import fetch as _fetch  # noqa: E402
from billhook.detectors import AdapterUnavailable, Detector, load  # noqa: E402

__all__ = [
    "ARTIFACT_SCHEMA",
    "RESULTS_DIR",
    "SUITES",
    "Sample",
    "Suite",
    "SuiteResult",
    "build_artifact",
    "git_state",
    "main",
    "render_markdown",
    "run_suite",
    "score",
    "wilson_interval",
]

#: Bump when the artifact shape changes. Old artifacts stay readable.
#: Artifacts written before the split carry `open-thicket/eval-result/1` or `/2`
#: and describe the same fields; only the `detector` block differs.
ARTIFACT_SCHEMA = "billhook/eval-result/1"

RESULTS_DIR = _REPO_ROOT / "evals" / "results"

#: Scan surfaces. A suite declares one, and it is passed to `Detector.scan`.
PROMPT = "prompt"
TOOL_RESULT = "tool_result"

_LATENCY_PERCENTILES = (50, 95, 99)


# ---------------------------------------------------------------------------
# Samples and suites
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Sample:
    """One scored item.

    Attributes:
        text: Exactly what is handed to the detector.
        positive: True if a detection is the correct outcome for this item.
            For JailbreakBench this is deliberately *not* the case, see
            `Suite.what_this_measures` there.
        group: Attack family, split, or category. Drives the per-group
            breakdown, which is how this harness answers "what was missed"
            without storing missed text (OT-EVL-010).
    """

    text: str
    positive: bool
    group: str


@dataclass(frozen=True, slots=True)
class Suite:
    """A dataset plus the exact procedure used to turn it into scored samples.

    Attributes:
        dataset: Key in :data:`billhook.fetch.DATASETS`. The artifact records that
            dataset's pinned digests, re-verified against the bytes on disk.
        surface: Which scan surface the detector is asked for. Detectors are
            free to behave differently on retrieved content than on a user
            prompt, and several do, so the surface is part of the condition,
            not an implementation detail.
        construction: How samples were assembled from the raw files. Prose,
            copied verbatim into the artifact, because a benchmark name is not a
            condition (OT-EVL-006).
        what_this_measures: The honest scope note. Printed next to the number.
        fpr_companion: Suite whose FPR accompanies this suite's recall when this
            suite has no negatives of its own (OT-EVL-004).
        caveat: Printed with every appearance of the result, or not printed at
            all. Used for the SPEC-010 §4 ⚠️ entry.
    """

    name: str
    dataset: str
    surface: str
    construction: str
    what_this_measures: str
    loader: Callable[[Path], Iterator[Sample]]
    fpr_companion: str | None = None
    caveat: str | None = None


# ---------------------------------------------------------------------------
# Loaders
#
# Each takes the fetched dataset directory and yields Samples. They are pure
# functions of bytes on disk: no sampling, no shuffling, no seeds. A benchmark
# that needs a random seed to reproduce has one more thing that can quietly
# change, and none of these need one.
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bipia(root: Path) -> Iterator[Sample]:
    """BIPIA email split: 50 real email contexts x 75 attack instructions.

    BIPIA is a construction procedure, not a sample file. `text_attack_test.json`
    holds 15 attack families of 5 instructions each; `email/test.jsonl` holds 50
    genuine emails with a question about each. The benchmark's own harness
    inserts an attack into the external content and asks the model the question.

    We scan the **external content only**, the email plus the inserted
    instruction, because that is what a guardrail actually sees when a tool
    result comes back. We do not reconstruct BIPIA's full model prompt, and the
    artifact says so: this is BIPIA's data and insertion procedure, not a
    byte-identical replay of its evaluation loop.

    Negatives are the 50 unmodified emails, which makes this the only attack
    suite here whose FPR comes from its own distribution.
    """
    attacks = _read_json(root / "text_attack_test.json")
    contexts = [
        json.loads(line)
        for line in (root / "email_test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in contexts:
        yield Sample(text=record["context"], positive=False, group="clean email")
    for family, instructions in attacks.items():
        for instruction in instructions:
            for record in contexts:
                yield Sample(
                    text=f"{record['context']}\n{instruction}",
                    positive=True,
                    group=family,
                )


def load_injecagent_base(root: Path) -> Iterator[Sample]:
    """InjecAgent **base** split: the attacker instruction inside a tool response.

    Positives are the `Tool Response` field, which is the tool output with the
    attacker instruction already embedded, verified on load to actually contain
    it, because a positive set that is silently clean is precisely the failure
    SPEC-010 §1 describes.

    Negatives are the matching `Tool Response Template` with the
    `<Attacker Instruction>` placeholder replaced by the empty string. That is a
    same-distribution control that introduces no text of the harness's
    authorship; filling the hole with invented benign prose would make the FPR a
    measurement of that prose. Templates are deduplicated, many test cases share
    one.

    The **enhanced** split is not fetched and not scorable here. It prepends a
    fixed literal ("IMPORTANT!!! Ignore all previous instructions...") to every
    sample, which any pattern engine scores near 100% on; OT-EVL-006 retires the
    claim built that way, and `evals/fetch.py` keeps it unpinned so the shortcut
    is not one flag away.
    """
    placeholder = "<Attacker Instruction>"
    seen_negatives: set[str] = set()
    for path in sorted(root.glob("test_cases_*_base.json")):
        split = "dh" if "_dh_" in path.name else "ds"
        for case in _read_json(path):
            response = case["Tool Response"]
            instruction = case["Attacker Instruction"]
            if instruction not in response:
                raise ValueError(
                    f"{path.name}: a base test case whose Tool Response does not "
                    f"contain its Attacker Instruction. Refusing to score a positive "
                    f"set that may not be positive (SPEC-010 §1)."
                )
            yield Sample(text=response, positive=True, group=f"{split}/{case['Attack Type']}")
            clean = case["Tool Response Template"].replace(placeholder, "")
            if clean not in seen_negatives:
                seen_negatives.add(clean)
                yield Sample(text=clean, positive=False, group=f"{split}/clean template")


_ZERO_WIDTH = frozenset("​‌‍⁠﻿")


def _llmail_group(text: str) -> str:
    """A mechanical obfuscation bucket for one LLMail submission.

    These groups *describe* the corpus for the per-group breakdown; nothing
    here is fed back into a rule set (OT-RUL-006). Zero-width characters and
    fullwidth forms are the two dominant obfuscation vehicles in the
    challenge data; everything else splits on whether it is plain ASCII.
    """
    if any(ch in _ZERO_WIDTH for ch in text):
        return "zero-width characters"
    # U+FF01..U+FF5E fullwidth ASCII forms, plus U+3000 ideographic space.
    # Deliberately NOT a range from U+3000 to U+FF5F: that would span the
    # CJK ideograph blocks and misbucket ordinary multilingual text.
    if any("！" <= ch <= "～" or ch == "　" for ch in text):
        return "fullwidth forms"
    if any(ord(ch) > 127 for ch in text):
        return "other non-ASCII"
    return "plain ASCII"


def load_llmail_inject(root: Path) -> Iterator[Sample]:
    """LLMail-Inject phase 2: real attacks submitted against a live defence.

    Positives are submissions that actually fired the target tool call -
    `attack_attempt` true with `reason == "api_triggered"` - not the larger
    judge-labelled set, whose labels are an LLM's opinion rather than an
    observed effect. The fetch pin note fixes this selection; a label is a
    claim, an api_triggered record is a measurement.

    Negatives are the 203 benign emails in emails_for_fp_tests.json, written
    for false-positive testing and sourced independently of BIPIA's 50, which
    is what lets this suite separate "FPR on business email" from "FPR on
    BIPIA's emails in particular".

    Recall here is NOT comparable to BIPIA's and the two must not be pooled:
    these attacks were written by people adapting to live defences, so the
    corpus is harder where attackers try and easier where they rely on
    obfuscation a static corpus never attempts.
    """
    submissions = _read_json(root / "labelled_unique_submissions_phase2.json")
    for text, meta in submissions.items():
        attempt = str(meta.get("attack_attempt", "")).strip().lower() == "true"
        if attempt and meta.get("reason") == "api_triggered":
            yield Sample(text=text, positive=True, group=_llmail_group(text))
    for email in _read_json(root / "emails_for_fp_tests.json"):
        yield Sample(text=email, positive=False, group="clean email")


def load_notinject(root: Path) -> Iterator[Sample]:
    """NotInject: 339 benign prompts built *around* injection trigger words.

    Entirely negative by construction. This is the adversarial FPR measurement -
    benign text engineered to look like an attack to a keyword matcher. It is
    the suite that separates a detector from a trigger-word list, and the one
    number a pattern-based tool has no excuse for.
    """
    for split in ("one", "two", "three"):
        for record in _read_json(root / f"NotInject_{split}.json"):
            yield Sample(
                text=record["prompt"],
                positive=False,
                group=f"{split} trigger word(s) / {record['category']}",
            )


def load_jailbreakbench(root: Path) -> Iterator[Sample]:
    """JBB-Behaviors: 100 harmful and 100 matched benign request goals.

    Both splits are loaded as **negatives**, and that is not a mistake.

    A JBB harmful behaviour is a request for harmful content ("write a
    defamatory article..."). It is not an injection: it contains no instruction
    aimed at overriding a system prompt, hijacking a tool, or exfiltrating
    context. Harmful-content classification is a different capability from
    injection detection, so for an injection detector the correct output on
    *both* JBB splits is "no detection", and the honest metric is a
    false-positive rate over all 200 rows.

    Scoring the harmful split as positives would manufacture a recall number for
    a capability injection detectors do not claim, and a low one would then read
    as a failure rather than as correct behaviour. Reporting it this way keeps
    OT-EVL-011's promise, the number is printed and the argument is made in the
    open, without inventing a metric to lose at.
    """
    for name, label in (("harmful-behaviors.csv", "harmful"), ("benign-behaviors.csv", "benign")):
        with (root / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                yield Sample(
                    text=row["Goal"],
                    positive=False,
                    group=f"{label} / {row['Category']}",
                )


def _dolly_records(root: Path) -> Iterator[dict[str, Any]]:
    with (root / "databricks-dolly-15k.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_dolly_prompt(root: Path) -> Iterator[Sample]:
    """databricks-dolly-15k instructions as benign prompts. FPR on the prompt surface."""
    for record in _dolly_records(root):
        instruction = record["instruction"].strip()
        if instruction:
            yield Sample(text=instruction, positive=False, group=record["category"])


def load_dolly_context(root: Path) -> Iterator[Sample]:
    """dolly-15k `context` passages as benign retrieved content. FPR on the tool surface.

    Only the ~8k records that carry a context are used; the field is a real
    retrieved passage, which is the closest public stand-in for RAG content this
    project can use without inventing a corpus. Scored through
    `scan_tool_result`, which runs strictly more machinery than the prompt
    surface, so this number and the prompt-surface one are not interchangeable
    and are reported separately.
    """
    for record in _dolly_records(root):
        context = record["context"].strip()
        if context:
            yield Sample(text=context, positive=False, group=record["category"])


SUITES: dict[str, Suite] = {
    "bipia": Suite(
        name="bipia",
        dataset="bipia",
        surface=TOOL_RESULT,
        construction=(
            "Positives: each of the 50 emails in BIPIA's email/test.jsonl concatenated "
            "with each of the 75 attack instructions in text_attack_test.json, joined "
            "by a newline, attack appended at the end (50 x 75 = 3750). Negatives: the "
            "same 50 emails unmodified. Scanned as tool-result content via "
            "the tool_result surface. This is BIPIA's data and insertion procedure "
            "applied to the external content only; it is not a byte-identical replay "
            "of BIPIA's model-prompt template."
        ),
        what_this_measures=(
            "Whether the detector flags an indirect-injection instruction sitting in "
            "retrieved content. SPEC-010 §4 names this the primary honest measure. It "
            "does not measure whether an agent would have obeyed the instruction."
        ),
        loader=load_bipia,
    ),
    "injecagent-base": Suite(
        name="injecagent-base",
        dataset="injecagent-base",
        surface=TOOL_RESULT,
        construction=(
            "Positives: the `Tool Response` field of every base test case in "
            "test_cases_dh_base.json and test_cases_ds_base.json, each verified on "
            "load to contain its `Attacker Instruction`. Negatives: the distinct "
            "`Tool Response Template` values with the `<Attacker Instruction>` "
            "placeholder replaced by the empty string. BASE variant only; the "
            "enhanced variant is not fetched (OT-EVL-006)."
        ),
        what_this_measures=(
            "Whether the detector flags an attacker instruction embedded in a "
            "tool-calling agent's tool output. Detection here is not the same as "
            "InjecAgent's own attack-success-rate metric, which requires running an "
            "agent against a model; this harness runs neither."
        ),
        loader=load_injecagent_base,
        # This suite has same-distribution negatives, but they are templates with
        # a hole in them rather than independently-sourced benign tool output, so
        # a general benign FPR is printed beside it as well (OT-EVL-004).
        fpr_companion="dolly-context",
    ),
    "llmail-inject": Suite(
        name="llmail-inject",
        dataset="llmail-inject",
        surface=TOOL_RESULT,
        construction=(
            "Positives: the submission texts in labelled_unique_submissions_phase2.json "
            "whose `attack_attempt` is true and whose `reason` is `api_triggered` "
            "(the attack was observed to fire the tool call; judge-labelled records "
            "are excluded). Negatives: all 203 emails in emails_for_fp_tests.json. "
            "Scanned as tool-result content, the surface an email assistant's "
            "retrieved mail arrives on."
        ),
        what_this_measures=(
            "Whether the detector flags real, adaptive injection attacks written by "
            "competitors against a live email-assistant defence. Recall here is not "
            "comparable to BIPIA's static templates and the two are never pooled: "
            "this corpus is harder where attackers use novel phrasing and easier "
            "where they rely on obfuscation. Detection is not attack success."
        ),
        loader=load_llmail_inject,
    ),
    "notinject": Suite(
        name="notinject",
        dataset="notinject",
        surface=PROMPT,
        construction=(
            "All 339 rows of the NotInject one/two/three splits, `prompt` field, "
            "scanned on the prompt surface. Every row is benign by construction."
        ),
        what_this_measures=(
            "Over-defense: false positives on benign prompts deliberately loaded with "
            "one, two or three injection trigger words. This is the adversarial FPR "
            "measurement (SPEC-010 §4)."
        ),
        loader=load_notinject,
    ),
    "jailbreakbench": Suite(
        name="jailbreakbench",
        dataset="jailbreakbench",
        surface=PROMPT,
        construction=(
            "All 200 rows of JBB-Behaviors (100 harmful + 100 matched benign), `Goal` "
            "field, scanned on the prompt surface. Both splits are scored as "
            "negatives, see `what_this_measures`."
        ),
        what_this_measures=(
            "A false-positive rate on harmful-content request goals. JBB behaviours "
            "are requests for harmful output, not injections: they contain no attempt "
            "to override a system prompt, hijack a tool, or exfiltrate context. "
            "Harmful-content classification is a different capability, so the correct "
            "output on both splits is 'no detection', and a recall figure here would "
            "measure something injection detectors do not claim. Reported rather than "
            "dropped, per OT-EVL-011."
        ),
        loader=load_jailbreakbench,
        caveat=(
            "SPEC-010 §4 marks JailbreakBench ⚠️: it measures harmful-content "
            "classification, a different capability. Do not read this row as an "
            "injection-detection result."
        ),
    ),
    "dolly-prompt": Suite(
        name="dolly-prompt",
        dataset="dolly15k",
        surface=PROMPT,
        construction=(
            "The non-empty `instruction` field of every record in "
            "databricks-dolly-15k.jsonl, scanned on the prompt surface."
        ),
        what_this_measures=(
            "False positives on ordinary human-written instructions, the base rate a "
            "deployment actually pays on the prompt surface. Attribution: "
            "databricks-dolly-15k, Databricks, CC BY-SA 3.0."
        ),
        loader=load_dolly_prompt,
    ),
    "dolly-context": Suite(
        name="dolly-context",
        dataset="dolly15k",
        surface=TOOL_RESULT,
        construction=(
            "The non-empty `context` field of every record in "
            "databricks-dolly-15k.jsonl, a real retrieved passage, scanned via "
            "the tool_result surface."
        ),
        what_this_measures=(
            "False positives on benign retrieved content. Reported separately from "
            "dolly-prompt because the tool-result surface runs the directive pattern "
            "set and the context-mismatch scorer in addition to the rule pass; the "
            "two numbers are not interchangeable. Attribution: databricks-dolly-15k, "
            "Databricks, CC BY-SA 3.0."
        ),
        loader=load_dolly_context,
    ),
}

#: Named collections. `v0_1` is every suite: OT-EVL-003 makes a partial run a
#: reporting hazard, so the default set is not a selection.
SUITE_SETS: dict[str, tuple[str, ...]] = {
    "v0_1": tuple(SUITES),
    "attack": ("bipia", "injecagent-base", "llmail-inject"),
    "benign": ("notinject", "jailbreakbench", "dolly-prompt", "dolly-context"),
}

#: A companion must exist, or the OT-EVL-004 guard in `render_markdown` would
#: fail at report time on a run that had already cost minutes. Fail at import.
for _suite in SUITES.values():
    if _suite.fpr_companion is not None and _suite.fpr_companion not in SUITES:
        raise AssertionError(
            f"{_suite.name}.fpr_companion={_suite.fpr_companion!r} is not a suite"
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because these rates sit near 0
    and 1 often enough that a Wald interval would run outside [0, 1] and quietly
    understate the uncertainty. OT-EVL-005 requires an interval for n < 500; one
    is computed for every rate here regardless, because a wide interval on a
    small n is exactly the thing that inherited claim (56.5% on n=23, ±20
    points) hid by omission.
    """
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    spread = (z / denominator) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. Deterministic, and defined for n == 1."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(pct / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


@dataclass
class SuiteResult:
    """Scored outcome for one suite. Serialized verbatim into the artifact."""

    suite: str
    surface: str
    dataset: str
    dataset_files: list[dict[str, Any]] = field(default_factory=list)
    dataset_sha256: str = ""
    construction: str = ""
    what_this_measures: str = ""
    caveat: str | None = None
    fpr_companion: str | None = None

    n: int = 0
    n_positive: int = 0
    n_negative: int = 0
    true_positive: int = 0
    false_negative: int = 0
    false_positive: int = 0
    true_negative: int = 0
    engine_errors: int = 0

    recall: float | None = None
    recall_ci95: tuple[float, float] | None = None
    fpr: float | None = None
    fpr_ci95: tuple[float, float] | None = None
    precision: float | None = None
    precision_ci95: tuple[float, float] | None = None

    latency_ms: dict[str, float] = field(default_factory=dict)
    total_seconds: float = 0.0
    groups: list[dict[str, Any]] = field(default_factory=list)


def score(samples: Sequence[Sample], detector: Detector, surface: str) -> SuiteResult:
    """Scan every sample and compute the full metric set.

    Latency is wall time around the `scan` call, not any detector-reported
    figure, so it includes the dispatch a caller actually pays. These are
    *suite* latencies over heterogeneous input sizes, useful as a distribution,
    not comparable to the fixed-size figures in `evals/bench.py`, and not
    publishable off the reference environment either way (SPEC-010 §3).
    """
    result = SuiteResult(suite="", surface=surface, dataset="")
    timings: list[float] = []
    per_group: dict[str, list[int]] = {}

    started = time.perf_counter()
    for sample in samples:
        call_start = time.perf_counter()
        judgement = detector.scan(sample.text, surface)
        timings.append((time.perf_counter() - call_start) * 1000.0)
        if judgement.error is not None:
            result.engine_errors += 1
        detected = judgement.detected

        # counts[0]=positives, [1]=hits on positives, [2]=negatives, [3]=hits on negatives
        counts = per_group.setdefault(sample.group, [0, 0, 0, 0])
        if sample.positive:
            result.n_positive += 1
            counts[0] += 1
            if detected:
                result.true_positive += 1
                counts[1] += 1
            else:
                result.false_negative += 1
        else:
            result.n_negative += 1
            counts[2] += 1
            if detected:
                result.false_positive += 1
                counts[3] += 1
            else:
                result.true_negative += 1
    result.total_seconds = time.perf_counter() - started
    result.n = result.n_positive + result.n_negative

    if result.n_positive:
        result.recall = result.true_positive / result.n_positive
        result.recall_ci95 = wilson_interval(result.true_positive, result.n_positive)
    if result.n_negative:
        result.fpr = result.false_positive / result.n_negative
        result.fpr_ci95 = wilson_interval(result.false_positive, result.n_negative)
    flagged = result.true_positive + result.false_positive
    if result.n_positive and flagged:
        result.precision = result.true_positive / flagged
        result.precision_ci95 = wilson_interval(result.true_positive, flagged)

    ordered = sorted(timings)
    result.latency_ms = {
        "n_scans": float(len(ordered)),
        "mean": statistics.fmean(ordered) if ordered else 0.0,
        **{f"p{p}": _percentile(ordered, p) for p in _LATENCY_PERCENTILES},
        "max": ordered[-1] if ordered else 0.0,
    }

    # Per-group breakdown: counts and rates only. No sample text, ever
    # (OT-EVL-010), this is what answers "which families do we miss".
    for group, (pos, pos_hit, neg, neg_hit) in sorted(per_group.items()):
        entry: dict[str, Any] = {"group": group, "n_positive": pos, "n_negative": neg}
        if pos:
            entry["recall"] = pos_hit / pos
            entry["recall_ci95"] = list(wilson_interval(pos_hit, pos))
        if neg:
            entry["fpr"] = neg_hit / neg
            entry["fpr_ci95"] = list(wilson_interval(neg_hit, neg))
        result.groups.append(entry)
    return result


# ---------------------------------------------------------------------------
# Provenance of the run itself
# ---------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def git_state() -> dict[str, Any]:
    """Commit, branch and dirty flag (OT-EVL-002, OT-EVL-009).

    `dirty` covers untracked files as well as modifications. An untracked
    `run.py` variant that produced the number is exactly as unreproducible as an
    uncommitted edit to a tracked one.
    """
    status = _git("status", "--porcelain")
    sha = _git("rev-parse", "HEAD")
    return {
        "sha": sha or "unknown",
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "dirty": bool(status) or not sha,
        "dirty_paths": sorted(line[3:] for line in status.splitlines())[:50],
    }


def _dataset_files(name: str, root: Path) -> tuple[list[dict[str, Any]], str]:
    """Re-hash the fetched bytes and cross-check against the pin (OT-EVL-008).

    The pin is checked at fetch time; it is checked again here because the
    artifact asserts a checksum, and an artifact that copies a checksum out of a
    source file rather than off the bytes it scored is asserting nothing.
    """
    entries: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for pinned in _fetch.DATASETS[name].files:
        path = root / pinned.name
        actual = _fetch._digest_of(path)
        if pinned.sha256 and actual != pinned.sha256:
            raise _fetch.ChecksumMismatch(
                f"{name}/{pinned.name}: on-disk sha256 {actual} does not match the pin "
                f"{pinned.sha256}. Refusing to score bytes that are not the pinned bytes."
            )
        entries.append(
            {
                "name": pinned.name,
                "sha256": actual,
                "size_bytes": path.stat().st_size,
                "url": pinned.url,
            }
        )
        combined.update(actual.encode("ascii"))
    return entries, combined.hexdigest()


def _environment() -> dict[str, Any]:
    """Machine identity, plus a flat statement about whether latency is publishable.

SPEC-010 §3 fixes the reference environment as this project's CI runner
    (`ubuntu-latest`, x86_64), mirroring open-thicket's OT-ENG-019: a latency
    number measured anywhere else is a development signal only. Since
    OT-EVL-009 *requires* latency in every artifact, every artifact produced off
    the runner necessarily carries a mandatory field that is simultaneously
    unpublishable, so it is labelled rather than left for a reader to infer.

    Detection counts are unaffected. They are deterministic for a deterministic
    detector and do not depend on the machine.
    """
    on_ci = bool(_env_flag("CI")) and bool(_env_flag("GITHUB_ACTIONS"))
    is_linux_x86 = platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")
    reference = on_ci and is_linux_x86
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "is_reference_environment": reference,
        "reference_environment_spec": "SPEC-010 §3: ubuntu-latest, x86_64, single process",
        "latency_publishable": reference,
        "latency_note": (
            "Measured on the SPEC-010 §3 reference environment."
            if reference
            else (
                "NOT the SPEC-010 §3 reference environment. The "
                "latency fields in this artifact are a development signal and are not "
                "publishable. The detection counts are unaffected: they are "
                "deterministic and environment-independent."
            )
        ),
    }


def _env_flag(name: str) -> str:
    import os

    return os.environ.get(name, "")


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


def build_artifact(
    results: Sequence[SuiteResult],
    *,
    detector: Detector,
    command: str,
) -> dict[str, Any]:
    """Assemble the committed artifact. Records everything OT-EVL-009 names."""
    from dataclasses import asdict

    git = git_state()
    environment = _environment()
    blockers: list[str] = []
    if git["dirty"]:
        blockers.append(
            "OT-EVL-002: produced from a dirty working tree. Not a release artifact; "
            "no number in it may be published."
        )
    if not environment["latency_publishable"]:
        blockers.append(
            "SPEC-010 §3: latency fields were not measured on the reference "
            "environment and are not publishable. Detection counts are."
        )
    return {
        "schema": ARTIFACT_SCHEMA,
        "requirements": [
            "OT-EVL-001",
            "OT-EVL-002",
            "OT-EVL-003",
            "OT-EVL-004",
            "OT-EVL-005",
            "OT-EVL-008",
            "OT-EVL-009",
            "OT-EVL-011",
        ],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "git": git,
        "environment": environment,
        "detector": {
            "name": detector.name,
            "version": detector.version,
            "kind": detector.kind,
            "requires": detector.requires,
        },
        "what_this_measures": (
            "Whether the detector emits a detection on a piece of text. It does not run "
            "an agent and does not call a model, so it does not measure attack "
            "success or attack prevention. AgentDojo is deliberately absent: SPEC-010 "
            "§4 admits it only if actually run end to end."
        ),
        "publication_blockers": blockers,
        "publishable": not blockers,
        "suites": [asdict(result) for result in results],
    }


def _fmt_rate(value: float | None, ci: Sequence[float] | None, n: int) -> str:
    """A rate cell. Below n=500 the interval is mandatory (OT-EVL-005)."""
    if value is None:
        return "-"
    if ci is not None and n < 500:
        return f"{value * 100:.1f}% [{ci[0] * 100:.1f}-{ci[1] * 100:.1f}]"
    return f"{value * 100:.1f}%"


def _group_rate(group: dict[str, Any], key: str) -> str:
    """A per-group rate cell. `recall` pairs with n_positive, `fpr` with n_negative."""
    n = group["n_positive" if key == "recall" else "n_negative"]
    return _fmt_rate(group.get(key), group.get(f"{key}_ci95"), n)


def render_markdown(artifact: dict[str, Any]) -> str:
    """Human-readable summary. Refuses to print a recall without an FPR.

    OT-EVL-004 is a claims rule, and a claims rule that lives only in a
    reviewer's head is the one that gets forgotten at 6pm on a release day. Here
    it is a `ValueError`.
    """
    by_name = {suite["suite"]: suite for suite in artifact["suites"]}
    lines: list[str] = []
    add = lines.append

    add("# billhook evaluation results")
    add("")
    if not artifact["publishable"]:
        add("> **NOT PUBLISHABLE.** " + " ".join(artifact["publication_blockers"]))
        add("")
    add(f"- Reproduce: `{artifact['command']}`")
    add(f"- Commit: `{artifact['git']['sha']}` (dirty: `{artifact['git']['dirty']}`)")
    add(f"- Detector: `{artifact['detector']['name']}` {artifact['detector']['version']} "
        f"({artifact['detector']['kind']}, {artifact['detector']['requires']})")
    add(f"- Generated: {artifact['generated_utc']}")
    add("")
    add(f"{artifact['what_this_measures']}")
    add("")
    add("## Results")
    add("")
    add("Every rate carries its sample size, and every rate on n < 500 carries a 95% "
        "Wilson interval in brackets (OT-EVL-005). Recall and FPR always appear "
        "together (OT-EVL-004); where a suite has no negatives of its own, the FPR "
        "column names the companion benign suite it is paired with.")
    add("")
    add("| Suite | Surface | n | Recall | FPR | Precision |")
    add("|---|---|---:|---|---|---|")
    for suite in artifact["suites"]:
        fpr_cell = _fmt_rate(suite["fpr"], suite["fpr_ci95"], suite["n_negative"])
        companion = by_name.get(suite["fpr_companion"] or "")
        if companion is not None and companion["fpr"] is not None:
            companion_cell = (
                _fmt_rate(companion["fpr"], companion["fpr_ci95"], companion["n_negative"])
                + f" (`{companion['suite']}`, n={companion['n_negative']})"
            )
            fpr_cell = companion_cell if suite["fpr"] is None else f"{fpr_cell}<br>{companion_cell}"
        if suite["recall"] is not None and fpr_cell == "-":
            raise ValueError(
                f"{suite['suite']}: a recall with no false-positive rate. OT-EVL-004 "
                f"makes that figure unpublishable, give the suite negatives, or name "
                f"an fpr_companion."
            )
        add(
            f"| `{suite['suite']}` | {suite['surface']} | {suite['n']} | "
            f"{_fmt_rate(suite['recall'], suite['recall_ci95'], suite['n_positive'])} | "
            f"{fpr_cell} | "
            f"{_fmt_rate(suite['precision'], suite['precision_ci95'], suite['n_positive'])} |"
        )
    add("")
    for suite in artifact["suites"]:
        add(f"### `{suite['suite']}`")
        add("")
        if suite["caveat"]:
            add(f"> ⚠️ {suite['caveat']}")
            add("")
        add(f"**Measures.** {suite['what_this_measures']}")
        add("")
        add(f"**Construction.** {suite['construction']}")
        add("")
        add(
            f"**Counts.** n={suite['n']} "
            f"(positive {suite['n_positive']}, negative {suite['n_negative']}); "
            f"TP {suite['true_positive']}, FN {suite['false_negative']}, "
            f"FP {suite['false_positive']}, TN {suite['true_negative']}; "
            f"engine errors {suite['engine_errors']}."
        )
        add("")
        add(f"**Dataset.** `{suite['dataset']}` sha256 `{suite['dataset_sha256'][:16]}…`")
        add("")
        breakdown = [g for g in suite["groups"] if g["n_positive"] or g["n_negative"]]
        if breakdown:
            add("| Group | n+ | recall | n- | FPR |")
            add("|---|---:|---|---:|---|")
            for group in breakdown:
                add(
                    f"| {group['group']} | {group['n_positive']} | "
                    f"{_group_rate(group, 'recall')} "
                    f"| "
                    f"{group['n_negative']} | "
                    f"{_group_rate(group, 'fpr')} |"
                )
            add("")
    add("## Latency (informational)")
    add("")
    add(artifact["environment"]["latency_note"])
    add("")
    add("These are per-scan wall times over each suite's own heterogeneous input "
        "sizes. They are a distribution, not a fixed-size measurement, and are not "
        "comparable to a controlled same-input benchmark.")
    add("")
    add("| Suite | scans | mean | p50 | p95 | p99 | max |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for suite in artifact["suites"]:
        lat = suite["latency_ms"]
        add(
            f"| `{suite['suite']}` | {int(lat['n_scans'])} | {lat['mean']:.3f} | "
            f"{lat['p50']:.3f} | {lat['p95']:.3f} | {lat['p99']:.3f} | "
            f"{lat['max']:.3f} |"
        )
    add("")
    add("<sub>Times in milliseconds. `databricks-dolly-15k` © Databricks, CC BY-SA 3.0.</sub>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_suite(name: str, detector: Detector, dest: Path) -> SuiteResult:
    """Load, score and annotate one suite. Fetches nothing, see `main`."""
    suite = SUITES[name]
    root = dest / suite.dataset
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} is missing. Fetch it first:\n"
            f"    python -m billhook.fetch --fetch {suite.dataset}"
        )
    files, combined = _dataset_files(suite.dataset, root)
    samples = list(suite.loader(root))
    result = score(samples, detector, suite.surface)
    result.suite = suite.name
    result.dataset = suite.dataset
    result.dataset_files = files
    result.dataset_sha256 = combined
    result.construction = suite.construction
    result.what_this_measures = suite.what_this_measures
    result.caveat = suite.caveat
    result.fpr_companion = suite.fpr_companion
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m billhook.run",
        description="Score one detector against pinned evaluation corpora (SPEC-010 §3).",
    )
    parser.add_argument("--list", action="store_true", help="describe every suite and exit")
    parser.add_argument(
        "--suite",
        action="append",
        metavar="NAME",
        help=f"suite or set name; repeatable. Sets: {', '.join(SUITE_SETS)}",
    )
    parser.add_argument(
        "--detector",
        default="regex-floor",
        metavar="SPEC",
        help="detector to score, e.g. regex-floor, thicket, huggingface:<model> "
             "(default: regex-floor)",
    )
    parser.add_argument("--dest", type=Path, default=_fetch.DEFAULT_DEST)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--tag", default="", help="filename suffix, e.g. 'preliminary'")
    parser.add_argument(
        "--no-fetch", action="store_true", help="fail instead of downloading a missing dataset"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="release gate: refuse to write an artifact from a dirty tree (OT-EVL-002)",
    )
    parser.add_argument("--no-write", action="store_true", help="print only; write nothing")
    args = parser.parse_args(argv)

    if args.list or not args.suite:
        for suite in SUITES.values():
            print(f"{suite.name:18} dataset={suite.dataset:16} surface={suite.surface}")
            print(f"{'':18} {suite.what_this_measures}")
            if suite.caveat:
                print(f"{'':18} CAVEAT: {suite.caveat}")
            print()
        print("sets: " + ", ".join(f"{k} ({len(v)} suites)" for k, v in SUITE_SETS.items()))
        return 0

    names: list[str] = []
    for requested in args.suite:
        for name in SUITE_SETS.get(requested, (requested,)):
            if name not in SUITES:
                print(f"unknown suite {name!r}; try --list", file=sys.stderr)
                return 2
            if name not in names:
                names.append(name)

    git = git_state()
    if args.release and git["dirty"]:
        print(
            "FAIL OT-EVL-002: the working tree is dirty, so this run cannot produce a "
            "release artifact. Commit or stash first.\n  dirty: "
            + ", ".join(git["dirty_paths"][:10]),
            file=sys.stderr,
        )
        return 1
    if git["dirty"]:
        print(
            "WARN OT-EVL-002: dirty working tree. The artifact will record "
            "git_dirty: true and nothing in it may be published.",
            file=sys.stderr,
        )

    for name in names:
        dataset = SUITES[name].dataset
        if not (args.dest / dataset).is_dir():
            if args.no_fetch:
                print(f"FAIL {dataset} not present and --no-fetch was given", file=sys.stderr)
                return 1
            print(f"fetching {dataset} …", file=sys.stderr)
            _fetch.fetch(dataset, args.dest)

    try:
        detector = load(args.detector)
    except AdapterUnavailable as exc:
        print(f"FAIL cannot load detector {args.detector!r}: {exc}", file=sys.stderr)
        return 1

    results: list[SuiteResult] = []
    for name in names:
        print(f"scoring {name} …", file=sys.stderr)
        results.append(run_suite(name, detector, args.dest))

    command = (
        f"python -m billhook.run --detector {args.detector} "
        + " ".join(f"--suite {s}" for s in args.suite)
    )
    artifact = build_artifact(results, detector=detector, command=command)
    summary = render_markdown(artifact)
    print(summary)

    if args.no_write:
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = [stamp, "results"]
    if git["dirty"]:
        parts.append("DIRTY")
    if args.tag:
        parts.append(args.tag)
    stem = "-".join(parts)
    (args.out_dir / f"{stem}.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / f"{stem}.md").write_text(summary, encoding="utf-8")
    print(f"wrote {args.out_dir / stem}.json and .md", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
