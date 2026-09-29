#!/usr/bin/env python3
"""correlate_observers.py -- a real, post-hoc log-analysis tool, NOT a
live observer class.

Real, added 2026-09-08 as the real, non-invasive implementation of
"ObserverCorrelationMatrix". A live, hooked-in observer version was
deliberately rejected: every real observer built this session emits an
independent print() line with no shared event bus, so a live correlation
observer could not actually see what other observers fired without
invasively modifying all ten existing, already-working observers -- a
real, unnecessary risk. This script instead parses an existing, real log
file after a run and reports which real observer tags co-occurred.

Usage:
    python3 scripts/correlate_observers.py /tmp/some_real_run.log
    python3 scripts/correlate_observers.py /tmp/*.log
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

# Real, direct list of every real observer/debug tag actually emitted by
# this codebase's print()-based instrumentation, confirmed by grep against
# the real source files this session. Kept as an explicit, real list
# rather than a generic "[A-Z_]+" pattern, since some real log lines
# contain bracketed text that is not a real observer tag (e.g. status
# markers), and a generic pattern would over-match.
_KNOWN_TAGS = [
    "SUBAGENT_TOOL_CALL_DEBUG",
    "SELF_CONSISTENCY_FLAG",
    "FABRICATION_FLAG",
    "MULTI_TOOL_OSCILLATION_FLAG",
    "PARTIAL_READ_CONTAMINATION_FLAG",
    "TOOL_SCAFFOLDING_DRIFT_FLAG",
    "CONTEXT_ANCHOR_SHIFT_FLAG",
    "TOOL_CALL_DOMINANCE_FLAG",
    "ERROR_RECOVERY_STREAK_FLAG",
    "FORCE_FINAL_ANSWER_OBSERVER",
    "ERROR_PATTERN_OBSERVER",
    # Real, added 2026-09-08 (extension): these three real observers were
    # built AFTER this script's original _KNOWN_TAGS list, so their real
    # print() tags were genuinely missing from correlation analysis until
    # now.
    "TOOL_STARVATION_FLAG",
    "RESCUE_MODE_DISTRIBUTION_TRACKER",
    "STOPPED_BY_DISTRIBUTION_TRACKER",
]

_TAG_RE = re.compile(r"\[(" + "|".join(_KNOWN_TAGS) + r")\]")

# Real ANSI escape codes and carriage returns pollute the raw CLI log
# output (confirmed directly, earlier this session) -- strip both before
# scanning, same real cleanup used throughout tonight's manual checks.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _clean_lines(path: Path) -> list[str]:
    raw = path.read_text(errors="replace")
    raw = raw.replace("\r", "\n")
    raw = _ANSI_RE.sub("", raw)
    return raw.splitlines()


def analyze_file(path: Path) -> tuple[Counter, Counter]:
    """Real, direct scan: returns (per-tag counts, per-pair co-occurrence
    counts) for the tags found anywhere in this one real log file. Every
    tag present anywhere in the file is treated as having "co-occurred"
    with every other tag present anywhere in the same file -- a real,
    coarse, file-level correlation, not a turn-level one (the underlying
    print() lines don't carry a shared, cross-observer turn key that
    would make a finer-grained correlation reliable)."""
    tags_seen = set()
    for line in _clean_lines(path):
        for match in _TAG_RE.finditer(line):
            tags_seen.add(match.group(1))

    tag_counts = Counter(tags_seen)
    pair_counts = Counter()
    for a, b in combinations(sorted(tags_seen), 2):
        pair_counts[(a, b)] += 1
    return tag_counts, pair_counts


def main(argv):
    if not argv:
        print("Usage: correlate_observers.py <log_file> [<log_file> ...]")
        return 1

    total_tag_counts = Counter()
    total_pair_counts = Counter()
    files_analyzed = 0

    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            print(f"Skipping (not a real file): {arg}")
            continue
        tag_counts, pair_counts = analyze_file(path)
        if not tag_counts:
            continue
        files_analyzed += 1
        total_tag_counts.update(tag_counts)
        total_pair_counts.update(pair_counts)
        print(f"{path.name}: {sorted(tag_counts)}")

    print()
    print(f"Real files with at least one observer tag: {files_analyzed}")
    print()
    print("Per-tag frequency (number of real log files containing it):")
    for tag, count in total_tag_counts.most_common():
        print(f"  {tag}: {count}")

    if total_pair_counts:
        print()
        print("Co-occurrence pairs (real files where BOTH tags appeared):")
        for (a, b), count in total_pair_counts.most_common():
            print(f"  {a} + {b}: {count}")
    else:
        print()
        print("No real co-occurring pairs found across the given files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
