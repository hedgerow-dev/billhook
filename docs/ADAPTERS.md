# Writing a detector adapter

The contract is deliberately tiny, three attributes and one method, so that
wrapping a tool is an afternoon's work. The whole harness rests on that being
true. Everything below is the detail that keeps your number comparable to
everyone else's.

## The contract

From `src/billhook/detectors.py`:

```python
class MyDetector:
    name = "my-tool"
    version = "1.2.3"
    kind = "classifier"          # deterministic | classifier | hosted | hybrid
    requires = "network+key"     # offline | model-download | network+key

    def scan(self, text: str, surface: str) -> Judgement: ...
```

`surface` is `"prompt"` or `"tool_result"` (the module constants `PROMPT` and
`TOOL_RESULT`). A detector may legitimately behave differently on each, and
several do, so the surface is part of the measured condition, not an
implementation detail. Do not ignore it and do not collapse the two into one
code path unless your tool genuinely cannot tell them apart.

`kind` and `requires` appear in the published table. A number without its cost
is not a comparison (OT-BCH-004), so set them honestly: if you call a network
API behind a key, that is `network+key`, however fast it feels.

## Judgement

```python
@dataclass(frozen=True, slots=True)
class Judgement:
    detected: bool               # your tool's own default decision
    score: float | None = None   # continuous risk in [0, 1]
    error: str | None = None
```

`score` is the field that matters most. Matched-FPR comparison (OT-BCH-001)
takes each detector's continuous risk, picks a threshold that yields a fixed
FPR on a benign corpus, and reads recall there. Without a score none of that is
possible: a boolean-only detector is reported at its shipped operating point
and marked, never silently compared as though threshold-matched (OT-BCH-008).
Most commercial APIs return a boolean or a coarse band, which is precisely why
cross-tool comparison has not happened, so if your API exposes anything
continuous, even a band index, map it into `[0, 1]` and return it.

Failures split two ways, and both must be visible:

- **Load-time failure** (missing package, model, or credential): raise
  `AdapterUnavailable` from `__init__`. The runner reports the detector in the
  "Not run" table with your reason (OT-BCH-006). Never catch your own
  `AdapterUnavailable`, and never fall back to an object that returns
  `detected=False` for everything, which would score the absent tool as
  perfectly permissive.
- **Per-call failure** (a timeout, a 500, a tensor that will not fit): return
  `Judgement(detected=False, score=None, error=str(exc))`. The harness counts
  these, treats the items as non-detections, and prints the count beside the
  table. Never swallow the exception into a clean non-detection, because the
  count is the difference between "the tool scored 40% recall" and "the tool
  scored 40% recall while erroring on a fifth of the corpus".

## Registration

`load(spec)` in `detectors.py` parses the CLI spec string. The existing forms:

```
thicket
regex-floor
huggingface:<model-id>
llm-guard:<scanner-name>
normalize+<any of the above>
```

To add yours, write the class, then add one branch to `load()`. Keep the
lazy-import pattern: third-party packages are imported inside `__init__` (or
inside `load()`), never at module scope. `import billhook` must pull nothing
(OT-BCH-009), and module scope in `detectors.py` currently imports only `re`,
`dataclasses`, and `typing`. A hard import at the top of the file breaks
`pip install billhook` for everyone who does not have your stack.

## A complete example

A hypothetical hosted vendor API, annotated:

```python
class AcmeGuardDetector:
    kind = "hosted"
    requires = "network+key"

    def __init__(self, model: str = "guard-v3") -> None:
        import os
        try:
            import acme  # lazy: module scope must stay import-light (OT-BCH-009)
        except ImportError as exc:
            raise AdapterUnavailable(
                "acme-guard needs the vendor SDK: pip install 'billhook[acme]'"
            ) from exc
        key = os.environ.get("ACME_API_KEY")
        if not key:
            # Load-time failure: the "Not run" table, never a silent zero
            # (OT-BCH-006).
            raise AdapterUnavailable(
                "acme-guard needs ACME_API_KEY in the environment"
            )
        self._client = acme.Client(api_key=key)
        self._model = model
        self.name = "acme-guard"
        self.version = f"guard-v3 (sdk {acme.__version__})"

    def scan(self, text: str, surface: str) -> Judgement:
        # The vendor scores prompts and retrieved content with different
        # policies; honour the surface rather than collapsing it away.
        channel = "rag" if surface == TOOL_RESULT else "chat"
        try:
            verdict = self._client.moderate(text, model=self._model,
                                            channel=channel)
        except Exception as exc:
            # Per-call failure: counted, printed, scored as a non-detection.
            return Judgement(detected=False, score=None, error=str(exc))
        risk = float(verdict.injection_risk)  # continuous, already in [0, 1]
        return Judgement(detected=verdict.flagged, score=risk)
```

Registration, one branch in `load()`:

```python
    if spec.startswith("acme-guard"):
        model = spec.split(":", 1)[1] if ":" in spec else "guard-v3"
        return AcmeGuardDetector(model)
```

Roughly forty lines, which is the point.

## Testing

Look at `tests/` for the house style. Mock the network and the model: no test
may need an API key, a download, or a live service. Cover at least the three
failure shapes above, the happy path, the missing-dependency path
(`AdapterUnavailable`), and the per-call error path.

One packaging rule is absolute: an adapter PR must not add a hard dependency to
`[project.dependencies]` in `pyproject.toml`. Optional extras only, under
`[project.optional-dependencies]`, each lazy-imported. The comment in that
file is load-bearing.

## After submission

Maintainers run the adapter on the pinned suites and publish the full table
with your row in it. Every detector that ran appears (OT-BCH-003), there is no
curated subset, and a poor score is not grounds for removal, including ours.
If you later believe the published row misrepresents your tool, that is what
[docs/CORRECTIONS.md](CORRECTIONS.md) is for.
