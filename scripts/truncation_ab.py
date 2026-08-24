"""Offline A/B for the tool-result preview shape: head-only vs head+tail.

Runs the SHIPPED code path (``plugins.tools._overflow``) over a corpus of tool
outputs whose load-bearing lines are labelled by position, and reports how much
of what the model called the tool for actually survives inline.

The question this answers is narrow and deterministic — no model, no network:
*given a cap, which of the lines that decide the next action are still visible?*
A head-only cut scores ~0 on tail needles by construction; the point of running
it is to size the gap on realistic output and to prove the switch works on the
real code rather than on a re-implementation of it.

    uv run python scripts/truncation_ab.py
    uv run python scripts/truncation_ab.py --caps 2000,8000 --json out.json

Exit code is 1 if the ``middle`` arm does not beat ``head`` on tail recall,
which makes it usable as a regression gate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class Case:
    """One tool output plus the lines that decide what the agent does next."""

    name: str
    body: str
    # needle -> where it sits in the body: head | middle | tail
    needles: dict[str, str] = field(default_factory=dict)
    # Relevance-ordered output, so its "head" needles are the good hits and are
    # scored on their own axis (``ranked_top_recall``).
    ranked: bool = False


def _lines(prefix: str, count: int, width: int = 60) -> str:
    return "".join(f"{prefix} {i:05d} " + "x" * width + "\n" for i in range(count))


def corpus() -> list[Case]:
    """Eight shapes, drawn from what these tools actually return.

    Every case is >> the caps under test, so every case truncates.
    """
    pytest_body = (
        "collected 1893 items\n\n"
        + _lines("tests/test_mod.py::case PASSED", 450)
        + "tests/test_ledger.py::test_rounding SKIPPED (needs decimal>=1.4)\n"
        + _lines("tests/test_mod.py::case PASSED", 450)
        + "=========================== short test summary info ===========================\n"
        "FAILED tests/test_billing.py::test_proration - AssertionError: 1199 != 1200\n"
        "1 failed, 1892 passed in 41.03s\n"
    )
    build_body = (
        "> app@1.0.0 build\n> webpack --mode production\n\n"
        + _lines("asset chunk", 700)
        + "ERROR in ./src/api/client.ts:88:12\n"
        "TS2345: Argument of type 'string' is not assignable to parameter of type 'URL'.\n"
        "webpack compiled with 1 error\n"
    )
    docker_body = (
        "#1 [internal] load build definition\n"
        + _lines("#7 12.34 Collecting package", 600)
        + "#9 ERROR: process \"/bin/sh -c pip install -r requirements.txt\" "
        "did not complete successfully: exit code: 1\n"
    )
    traceback_body = (
        "starting ingest run 2f19\n"
        + _lines("[info] ingested record", 800)
        + 'Traceback (most recent call last):\n'
        '  File "/app/ingest.py", line 210, in flush\n'
        "    self._conn.commit()\n"
        "psycopg2.errors.UniqueViolation: duplicate key value violates "
        'constraint "orders_pkey"\n'
    )
    grep_body = (
        "src/auth/session.py:12:def issue_token(user):\n"
        + _lines("src/generated/schema.py:1:  field", 450)
        + "src/auth/refresh.py:77:    return issue_token(user)\n"
        + _lines("src/generated/schema.py:1:  field", 450)
        + "tests/test_session.py:410:    assert issue_token(u).expires_in == 3600\n"
    )
    json_body = (
        '{"status":"ok","items":['
        + ",".join(f'{{"id":{i},"sku":"SKU-{i:05d}"}}' for i in range(3000))
        + '],"next_cursor":"eyJvZmZzZXQiOjMwMDB9","total":58211}'
    )
    page_body = (
        "# Transformer-XL: Attentive Language Models\n\n"
        "## Abstract\nWe propose a novel neural architecture...\n\n"
        + _lines("Body paragraph", 800)
        + "## References\n[41] Vaswani et al. Attention Is All You Need. NeurIPS 2017.\n"
    )
    # Relevance-ordered, and — as real search results are — the useful hits are
    # not all at rank 1. Good hits sit at ranks 1, 9 and 12: a contiguous head
    # reaches them, a head-plus-tail split stops around rank 6 and spends the
    # rest of its budget on rank 590+.
    _GOOD = {
        1: ("Stripe docs: proration behaviour",
            "https://stripe.com/docs/billing/subscriptions/prorations",
            "Proration is computed in the smallest currency unit, rounded half-up."),
        9: ("Invoice total off by one cent",
            "https://github.com/stripe/stripe-node/issues/1204",
            "We saw 1199 where 1200 was expected; fix was to round the sum, not each line."),
        12: ("Billing rounding RFC",
             "https://internal.example/rfc/billing-rounding",
             "Decision: round once at invoice level, not per line item."),
    }

    def _hit(rank: int) -> tuple[str, str, str]:
        return _GOOD.get(rank) or (
            f"Unrelated forum thread {rank}",
            f"https://forum.example/t/{rank}",
            "Has anyone else seen weird invoice numbers? bump. bump. still bumping.",
        )

    ranked_body = (
        "Search results for: proration rounding bug stripe invoice\n\n"
        + "".join(
            f"{rank}. {t}\n   {u}\n   {sn}\n\n"
            for rank in range(1, 601)
            for t, u, sn in [_hit(rank)]
        )
    )
    find_body = (
        "./src\n./src/auth\n"
        + _lines("./node_modules/.cache/entry", 900)
        + "./migrations/0042_add_orders_pkey.sql\n"
    )
    return [
        Case("pytest", pytest_body, {
            "collected 1893 items": "head",
            "test_rounding SKIPPED": "middle",
            "FAILED tests/test_billing.py::test_proration": "tail",
            "1 failed, 1892 passed": "tail",
        }),
        Case("webpack", build_body, {
            "webpack --mode production": "head",
            "TS2345: Argument of type": "tail",
            "webpack compiled with 1 error": "tail",
        }),
        Case("docker-build", docker_body, {
            "load build definition": "head",
            "did not complete successfully: exit code: 1": "tail",
        }),
        Case("python-traceback", traceback_body, {
            "starting ingest run 2f19": "head",
            "psycopg2.errors.UniqueViolation": "tail",
        }),
        Case("grep", grep_body, {
            "src/auth/session.py:12:def issue_token": "head",
            "src/auth/refresh.py:77": "middle",
            "tests/test_session.py:410": "tail",
        }),
        Case("json-api", json_body, {
            '"status":"ok"': "head",
            '"next_cursor":"eyJvZmZzZXQiOjMwMDB9"': "tail",
            '"total":58211': "tail",
        }),
        Case("web-page", page_body, {
            "## Abstract": "head",
            "[41] Vaswani et al.": "tail",
        }),
        Case("web_search", ranked_body, ranked=True, needles={
            # A ranked list inverts the usual rule: what decides the next action
            # is the TOP of the list, and its tail is the dross — so only the
            # good hits are scored, on their own axis.
            "stripe.com/docs/billing/subscriptions/prorations": "head",
            "stripe-node/issues/1204": "head",
            "internal.example/rfc/billing-rounding": "head",
        }),
        Case("find", find_body, {
            "./src/auth": "head",
            "0042_add_orders_pkey.sql": "tail",
        }),
    ]


def _run_arm(mode: str, cases: list[Case], caps: list[int]) -> dict:
    """Score one arm through the real ``maybe_overflow`` path."""
    from frontier_agent.infra.config import get_config
    from plugins.tools import _overflow, meta

    get_config().tool_result_truncation = mode
    per_cap: dict[str, dict] = {}
    for cap in caps:
        # Drive the real entry point: cap the fake tools the way ToolMeta would.
        # Two probes, because ``auto`` dispatches on ToolMeta.result_is_ranked.
        meta.TOOL_META["ab_probe"] = meta.ToolMeta(
            category="compute", max_result_chars=cap,
        )
        meta.TOOL_META["ab_probe_ranked"] = meta.ToolMeta(
            category="web", max_result_chars=cap, result_is_ranked=True,
        )
        hits = {"head": [0, 0], "middle": [0, 0], "tail": [0, 0]}
        ranked_top = [0, 0]
        over_cap = 0
        pointer = 0
        rows = []
        for case in cases:
            probe = "ab_probe_ranked" if case.name == "web_search" else "ab_probe"
            out = _overflow.maybe_overflow(probe, case.body)
            over_cap += len(out) > cap
            pointer += "/.spill/" in out or "not readable from this backend" in out
            found = []
            for needle, where in case.needles.items():
                # A ranked list's good hits decide the next action; they are
                # counted on their own axis. "Did you keep the worst results" is
                # not a quality metric, so a ranked case has no other needles.
                bucket = ranked_top if case.ranked else hits[where]
                bucket[1] += 1
                if needle in out:
                    bucket[0] += 1
                    found.append(needle)
            rows.append({
                "case": case.name,
                "body_chars": len(case.body),
                "inline_chars": len(out),
                "needles_found": len(found),
                "needles_total": len(case.needles),
            })
        per_cap[str(cap)] = {
            "recall": {
                where: (n / d if d else 1.0) for where, (n, d) in hits.items()
            },
            "ranked_top_recall": ranked_top[0] / ranked_top[1] if ranked_top[1] else 1.0,
            "results_over_cap": over_cap,
            "results_with_pointer": pointer,
            "cases": rows,
        }
    return per_cap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--caps", default="2000,8000,32000",
        help="Comma-separated inline caps to test (default: 2000,8000,32000)",
    )
    ap.add_argument("--json", default="", help="Also write the full report here")
    args = ap.parse_args()
    caps = [int(c) for c in args.caps.split(",") if c.strip()]

    with tempfile.TemporaryDirectory(prefix="trunc-ab-") as tmp:
        # An isolated, agent-visible spill store: the arms must not inherit each
        # other's files, and the host workspace must not collect any.
        workspace = Path(tmp) / "ws"
        workspace.mkdir()
        saved_env = {
            key: os.environ.get(key)
            for key in ("SANDBOX_BACKEND", "FRONTIER_AGENT_WORKSPACE_DIR")
        }
        os.environ["SANDBOX_BACKEND"] = "container"
        os.environ["FRONTIER_AGENT_WORKSPACE_DIR"] = str(workspace)

        from frontier_agent.core.execution_context import (
            ExecutionScope,
            reset_current_execution_scope,
            set_current_execution_scope,
        )

        # Every global touched here is restored: this script is importable, and a
        # leaked workspace resolver or truncation mode would silently reshape
        # whatever runs next in the same process.
        from frontier_agent.infra.config import get_config
        # No resolver patch is needed: with no task sandbox bound,
        # ``current_local_workspace()`` is empty and ``_overflow_dir`` falls
        # through to the mount dir named by FRONTIER_AGENT_WORKSPACE_DIR above.
        saved_mode = get_config().tool_result_truncation
        token = set_current_execution_scope(
            ExecutionScope(task_id="truncation-ab", metadata={"llm_session_id": "ab"}),
        )
        cases = corpus()
        try:
            report = {
                mode: _run_arm(mode, cases, caps)
                for mode in ("head", "middle", "auto")
            }
            spilled = len(list((workspace / ".spill").rglob("*.md")))
        finally:
            reset_current_execution_scope(token)
            get_config().tool_result_truncation = saved_mode
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            from plugins.tools import meta as tool_meta

            tool_meta.TOOL_META.pop("ab_probe", None)
            tool_meta.TOOL_META.pop("ab_probe_ranked", None)

    _print(report, caps, cases, spilled)
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"\nfull report -> {args.json}")

    failures = []
    for cap in caps:
        c = str(cap)
        if report["middle"][c]["recall"]["tail"] <= report["head"][c]["recall"]["tail"]:
            failures.append(f"cap={cap}: middle does not beat head on tail recall")
        # ``auto`` must take the better shape for BOTH kinds of output: the
        # verdict at the end of sequential output, and the top of a ranked list.
        if report["auto"][c]["recall"]["tail"] < report["middle"][c]["recall"]["tail"]:
            failures.append(f"cap={cap}: auto lost sequential tail recall")
        if report["auto"][c]["ranked_top_recall"] < report["head"][c]["ranked_top_recall"]:
            failures.append(f"cap={cap}: auto lost ranked-list top recall")
        if report["auto"][c]["results_over_cap"]:
            failures.append(f"cap={cap}: auto returned a result over its cap")
    if failures:
        print("\nFAIL:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


def _print(report: dict, caps: list[int], cases: list[Case], spilled: int) -> None:
    needles = sum(len(c.needles) for c in cases)
    print(
        f"corpus: {len(cases)} tool outputs, {needles} load-bearing lines "
        f"(head/middle/tail labelled)",
    )
    print(f"spill files written across both arms: {spilled} "
          f"(content-hash naming makes the arms share identical bodies)\n")
    header = (
        f"{'cap':>7}  {'arm':<7}  {'head':>6}  {'middle':>6}  {'tail':>6}  "
        f"{'ranked':>7}  {'over cap':>8}  {'pointer':>7}"
    )
    print(header)
    print("-" * len(header))
    for cap in caps:
        for mode in ("head", "middle", "auto"):
            row = report[mode][str(cap)]
            print(
                f"{cap:>7}  {mode:<7}  "
                f"{row['recall']['head']:>6.0%}  {row['recall']['middle']:>6.0%}  "
                f"{row['recall']['tail']:>6.0%}  {row['ranked_top_recall']:>7.0%}  "
                f"{row['results_over_cap']:>8}  {row['results_with_pointer']:>7}",
            )
        print("-" * len(header))
    print(
        "\nhead:    lines near the start — both arms should hold this at 100%\n"
        "middle:  what BOTH arms give up; reachable only through the pointer\n"
        "tail:    the verdict lines — FAILED test, TS error, exit code, cursor\n"
        "ranked:  top hits of a relevance-ordered search result, where a\n"
        "         contiguous head beats a head+tail split\n"
        "overcap: results longer than the cap they advertise (must be 0)\n"
        "pointer: results carrying a recovery path (should equal the corpus size)",
    )


if __name__ == "__main__":
    raise SystemExit(main())
