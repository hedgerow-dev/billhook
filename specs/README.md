# Specs

`specs/` is the product. Code that contradicts a spec is a bug in the code.

| Spec | Title | Status |
|---|---|---|
| [010-evaluation.md](010-evaluation.md) | Evaluation and claim honesty | Approved |
| [011-benchmark-harness.md](011-benchmark-harness.md) | The comparison harness | Approved |

## Why the IDs say `OT-`

billhook was extracted from [open-thicket](https://github.com/hedgerow-dev/open-thicket),
where the prefix stood for that project. The IDs were kept rather than renumbered,
because they are cited in commit messages, published artifacts, and result
documents that already exist, and a renumber would silently break every one of
those references while looking like a tidy-up.

An ID means the same thing in both repositories. It is never redefined.

## Where each cited ID is defined

Requirements this codebase cites, and which spec owns them:

| Prefix | Owner | Spec |
|---|---|---|
| `OT-BCH-*` | billhook | [011-benchmark-harness.md](011-benchmark-harness.md) |
| `OT-EVL-001..006`, `008..011` | shared, restated here | [010-evaluation.md](010-evaluation.md) |
| `OT-EVL-007` | open-thicket | SPEC-010 §2 there. Binds detector authors, not this harness. |
| `OT-RUL-006`, `OT-RUL-007` | open-thicket | SPEC-005. Rule provenance and the never-commit-corpora rule. |
| `OT-ENG-019` | open-thicket | SPEC-003 §6.1. Cited where this harness mirrors its reasoning about unpublishable latency. |
| `OT-ARC-006` | open-thicket | SPEC-002. Fail-open: an internal error is a non-detection, never an exception. |

The external ones are cited, not restated. If open-thicket changes one, the
citation here is what goes stale, so treat a surprising cross-reference as a
reason to go read the other repository rather than to guess.
