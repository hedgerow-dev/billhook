# Correcting a published result

This document operationalises OT-BCH-011 (`specs/011-benchmark-harness.md`):

```
OT-BCH-011 MUST offer any vendor whose tool appears in a published table the
  same reproduction path, and MUST correct a result shown to be a harness
  error rather than defending it.
```

## The commitment

If the harness got your tool wrong, the maintainers fix the harness and
re-publish. Wrong means a concrete harness defect: a bad adapter, a wrong
label index, an out-of-distribution corpus passed off as in-distribution, a
threshold mis-set by the matching code. The published number is the harness's
number, and an error in it is a bug here, not a disagreement about taste.

The response to being shown wrong is what decides whether this project is a
referee or a marketing asset, so the procedure is public and mechanical.

## What to file

Open a GitHub issue containing:

1. **The artifact JSON** for the result you dispute (committed under
   `evals/results/`). It records the corpus checksums, detector versions, and
   git SHA, and it is the thing being argued about.
2. **The exact reproduce command** the artifact carries. Run it first if you
   can; the whole point of OT-BCH-010 is that you do not need our cooperation
   to check our work.
3. **What you believe is wrong**, stated against the harness, not the
   conclusion. "Your adapter reads `verdict.safe` instead of
   `verdict.injection_risk`" is actionable. "Your benchmark is unfair to us"
   is not.
4. **The smallest evidence you have.** Five inputs where the shipped tool and
   the harness row disagree will usually settle the question. Fifty pages of
   marketing material will not.

## What happens

The issue is acknowledged, reproduced, and then either fixed or argued, in the
open, in that issue. Both outcomes are public. If the harness is wrong, the fix
lands in the adapter or the harness, the suite is re-run, and the corrected
result is published **as a new artifact**, never by editing the old one.
Artifacts are immutable once committed: they are pinned to a git SHA, and the
release gate (OT-BCH-007, OT-EVL-002) refuses to write a release artifact from
a dirty tree, so an artifact whose underlying code could be quietly patched
after the fact would be asserting nothing. The old artifact stays in the
repository, visibly superseded, because a referee that airbrushes its own
history has no standing to measure anyone else's.

## What is not a correction

Disagreement with the corpus choice is a methodology discussion. It is welcome,
and the choice of negative corpus is documented as the single most flattering
or damning decision a benchmark makes (`docs/BENCHMARK.md` has the story), but
it is an issue against the **spec**, not a bug against the harness. Arguing
that BIPIA is the wrong measure belongs in a discussion of
`specs/010-evaluation.md` §4, argued in the open, and if it persuades, the spec
changes and every tool's row moves together.

The distinction that matters: "your harness measured my tool incorrectly" is a
correction; "your harness measured every tool on the wrong thing" is
methodology. Both are worth filing. Only the first one re-runs a table.
