"""billhook, an independent harness for comparing prompt-injection defenses.

Nothing here detects anything. This package fetches pinned public suites, runs
any detector that satisfies the `Detector` protocol over them, and reports
recall at a matched false-positive rate with the cost of each tool in the table.

The detectors it scores are all optional. `regex-floor` ships in-tree because a
comparison without a naive baseline cannot tell you whether a tool is good or
whether the corpus is easy; everything else, including open-thicket's engine, is
an adapter that reports itself unavailable until you install it.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
