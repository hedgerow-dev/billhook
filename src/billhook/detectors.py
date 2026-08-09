"""Detector adapters for the comparison harness (SPEC-011).

The contract is deliberately tiny, three attributes and one method, so that
anyone can wrap their own tool in an afternoon and get a comparable number.

    class MyDetector:
        name = "my-tool"; version = "1.2.3"
        kind = "classifier"; requires = "network+key"
        def scan(self, text: str, surface: str) -> Judgement: ...

Adapters that need torch, transformers, or llm_guard import them lazily inside
`load()`, never at module scope (OT-BCH-009).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

PROMPT = "prompt"
TOOL_RESULT = "tool_result"


@dataclass(frozen=True, slots=True)
class Judgement:
    """One detector's opinion about one piece of text.

    `score` is what makes cross-tool comparison possible: holding the
    false-positive rate fixed across detectors requires a continuous risk
    signal, not each vendor's chosen cut-off (OT-BCH-001). A detector that
    cannot supply one is still runnable, but only at its own operating point.
    """

    detected: bool
    score: float | None = None
    error: str | None = None


@runtime_checkable
class Detector(Protocol):
    name: str
    version: str
    kind: str      # deterministic | classifier | hosted | hybrid
    requires: str  # offline | model-download | network+key

    def scan(self, text: str, surface: str) -> Judgement: ...


class AdapterUnavailable(RuntimeError):
    """A detector could not be loaded, missing dep, model, or credential.

    Raised by `load()` so the runner can report the detector as "not run"
    with a reason. It must never be swallowed into a non-detection, which
    would silently score the absent tool as perfectly permissive
    (OT-BCH-006).
    """


# --------------------------------------------------------------------------
# thicket, an optional adapter for open-thicket's deterministic engine
# --------------------------------------------------------------------------

class ThicketDetector:
    kind = "deterministic"
    requires = "offline"

    def __init__(self) -> None:
        try:
            from thicket import Engine, __version__
            from thicket.rules import RuleSet
        except ImportError as exc:
            raise AdapterUnavailable(
                "the thicket adapter needs open-thicket: pip install open-thicket"
            ) from exc

        self._engine = Engine()
        rules = RuleSet.bundled()
        self.name = "thicket"
        self.version = f"{__version__} ({len(rules)} rules, {rules.hash[:14]})"

    def scan(self, text: str, surface: str) -> Judgement:
        scan = (self._engine.scan_tool_result if surface == TOOL_RESULT
                else self._engine.scan_prompt)
        v = scan(text)
        return Judgement(detected=v.detected, score=v.score, error=v.error)


# --------------------------------------------------------------------------
# regex-floor, the honesty baseline (OT-BCH-005)
# --------------------------------------------------------------------------

_FLOOR_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|preceding)\b",
    r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\b",
    r"\bforget\s+(?:everything|all)\b",
    r"\byou\s+are\s+now\b",
    r"\bnew\s+(?:task|instruction)s?\s*:",
    r"\bsystem\s+prompt\b",
    r"\bDAN\b",
    r"\bdeveloper\s+mode\b",
    r"<\s*\|?\s*im_(?:start|end)\s*\|?\s*>",
    r"\boverride\s+(?:previous|prior|your)\b",
]


class RegexFloorDetector:
    """Ten obvious patterns. The floor every other tool must clear.

    This exists because a benchmark without a naive baseline cannot tell you
    whether a tool is good or whether the corpus is easy. If a commercial
    product does not clearly beat this, that is the single most useful thing
    a comparison can report.
    """

    name = "regex-floor"
    version = f"{len(_FLOOR_PATTERNS)} patterns"
    kind = "deterministic"
    requires = "offline"

    def __init__(self) -> None:
        self._res = [re.compile(p, re.IGNORECASE) for p in _FLOOR_PATTERNS]

    def scan(self, text: str, surface: str) -> Judgement:
        hits = sum(1 for r in self._res if r.search(text))
        # A count is a usable ordering signal; saturate at 3 so one weak
        # pattern does not outrank a genuine multi-signal match.
        return Judgement(detected=hits > 0, score=min(hits / 3.0, 1.0))


# --------------------------------------------------------------------------
# huggingface:<model>, any sequence-classification classifier
# --------------------------------------------------------------------------

class HuggingFaceDetector:
    kind = "classifier"
    requires = "model-download"

    def __init__(self, model_id: str, threshold: float = 0.5) -> None:
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - env dependent
            raise AdapterUnavailable(
                f"{model_id} needs torch + transformers: pip install 'billhook[hf]'"
            ) from exc

        try:
            self._tok = AutoTokenizer.from_pretrained(model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_id)
        except Exception as exc:
            raise AdapterUnavailable(
                f"could not load {model_id}: {exc}. Gated models need "
                f"`huggingface-cli login` and accepted terms."
            ) from exc

        self._torch = torch
        self._model.eval()
        labels = self._model.config.id2label
        self._idx = next(
            (i for i, lab in labels.items() if "INJECT" in str(lab).upper()),
            1 if len(labels) > 1 else 0,
        )
        self._threshold = threshold
        self.name = f"huggingface:{model_id}"
        self.version = f"{model_id} (label={labels[self._idx]})"

    def scan(self, text: str, surface: str) -> Judgement:
        try:
            enc = self._tok(text, truncation=True, max_length=512,
                            return_tensors="pt")
            with self._torch.no_grad():
                logits = self._model(**enc).logits
            p = float(self._torch.softmax(logits, dim=-1)[0, self._idx])
        except Exception as exc:
            return Judgement(detected=False, score=None, error=str(exc))
        return Judgement(detected=p >= self._threshold, score=p)


# --------------------------------------------------------------------------
# normalize:<inner>, measure preprocessing lift separately from detection
# --------------------------------------------------------------------------

class NormalizedDetector:
    """Wrap any detector in open-thicket's normalization.

    Reports max(score(raw), score(normalized)). This isolates what
    deobfuscation contributes from what the detector contributes, the two are
    routinely conflated in vendor material, and separating them is one of the
    few genuinely novel things this harness can show.

    The normalizer is open-thicket's rather than one of our own, deliberately:
    billhook does not ship a deobfuscation pipeline and should not grow one,
    because a referee that supplies the preprocessing is no longer only
    measuring it. This adapter is therefore optional and absent by default.
    """

    def __init__(self, inner: Detector) -> None:
        try:
            from thicket.normalize import normalize
        except ImportError as exc:
            raise AdapterUnavailable(
                "normalize+ needs open-thicket's normalizer: pip install open-thicket"
            ) from exc

        self._inner = inner
        self._normalize = normalize
        self.name = f"normalize+{inner.name}"
        self.version = inner.version
        self.kind = inner.kind
        self.requires = inner.requires

    def scan(self, text: str, surface: str) -> Judgement:
        a = self._inner.scan(text, surface)
        try:
            normalized = self._normalize(text)
        except Exception as exc:
            return Judgement(a.detected, a.score, error=f"normalize: {exc}")
        if normalized.strip() == text.strip():
            return a
        b = self._inner.scan(normalized, surface)
        scores = [s for s in (a.score, b.score) if s is not None]
        return Judgement(
            detected=a.detected or b.detected,
            score=max(scores) if scores else None,
            error=a.error or b.error,
        )


# --------------------------------------------------------------------------
# llm-guard, the upstream package only
# --------------------------------------------------------------------------

class LLMGuardDetector:
    kind = "classifier"
    requires = "model-download"

    def __init__(self, scanner: str = "PromptInjection") -> None:
        self._scanner_name = scanner
        # Upstream only. Scoring open-thicket's compatibility shim and labelling
        # the row "llm-guard" would report our reimplementation as the thing it
        # imitates, which is the one substitution a comparison must never make.
        try:
            mod = __import__(
                "llm_guard.input_scanners", fromlist=[scanner]
            )
            self._impl = getattr(mod, scanner)()
        except Exception as exc:
            raise AdapterUnavailable(
                f"llm-guard does not provide {scanner}: {exc}. "
                f"Install it with: pip install llm-guard"
            ) from exc
        self.name = f"llm-guard:{scanner}"
        self.version = "llm-guard (upstream, archived 2026-07-09)"

    def scan(self, text: str, surface: str) -> Judgement:
        try:
            _sanitized, is_valid, risk = self._impl.scan(text)
        except Exception as exc:
            return Judgement(detected=False, score=None, error=str(exc))
        return Judgement(detected=not is_valid, score=float(risk))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

def load(spec: str) -> Detector:
    """Build a detector from a CLI spec string.

    Accepted forms::

        thicket
        regex-floor
        huggingface:protectai/deberta-v3-base-prompt-injection-v2
        llm-guard:PromptInjection
        normalize+<any of the above>

    Raises AdapterUnavailable with an actionable message rather than
    degrading to a detector that silently never fires.
    """
    spec = spec.strip()
    if spec.startswith("normalize+"):
        return NormalizedDetector(load(spec[len("normalize+"):]))
    if spec == "thicket":
        return ThicketDetector()
    if spec in ("regex-floor", "floor"):
        return RegexFloorDetector()
    if spec.startswith("huggingface:"):
        return HuggingFaceDetector(spec.split(":", 1)[1])
    if spec.startswith("llm-guard:"):
        return LLMGuardDetector(spec.split(":", 1)[1])
    raise AdapterUnavailable(
        f"unknown detector {spec!r}. Known: thicket, regex-floor, "
        f"huggingface:<model>, llm-guard:<scanner>, normalize+<spec>"
    )


#: What runs when no --detector is given. Only the floor, because it is the one
#: detector that is always available and never needs installing. Every other
#: adapter, open-thicket's included, is opt-in.
DEFAULT_COMPARISON = ("regex-floor",)
