"""Configuration, security and error-mapping units for the HF Space demo."""

from __future__ import annotations

import logging

import pytest

from deploy.huggingface.config import (
    DEFAULT_ALLOWED_TOOLS,
    load_config,
    preflight,
    redact_url,
)
from deploy.huggingface.errors import classify_error, classify_error_name
from deploy.huggingface.security import (
    HARD_DENIED_TOOLS,
    REDACTED,
    DownloadDenied,
    RedactingLogFilter,
    Redactor,
    demo_safe_tool_names,
    demo_safe_tool_policy,
    list_output_files,
    resolve_download,
)

_GOOD_ENV = {
    "OPENAI_BASE_URL": "https://endpoint.example.com/v1",
    "OPENAI_API_KEY": "sk-test-0123456789abcdef",
    "OPENAI_MODEL": "Apodex-1.1-mini",
    "SANDBOX_BACKEND": "native",
    "SERPER_API_KEY": "serper-test-key",
    "HOME": "/tmp",
}


def _env(**overrides: str) -> dict[str, str]:
    return {**_GOOD_ENV, **overrides}


# ── configuration ────────────────────────────────────────────────────────


def test_default_config_passes_preflight() -> None:
    config = load_config(_env())
    checks = preflight(config)
    assert checks.ok, checks.format()
    assert config.workflow == "react"
    assert config.max_concurrency == 1


def test_model_and_endpoint_come_only_from_the_environment() -> None:
    """Swapping model/endpoint must never require a code change."""
    config = load_config(_env(
        OPENAI_MODEL="some-other-model",
        OPENAI_BASE_URL="https://other.example.org/v1",
    ))
    assert config.openai_model == "some-other-model"
    assert config.openai_base_url == "https://other.example.org/v1"
    # …and the values reach the runtime through the profile-override seam.
    overrides = config.profile_overrides()
    assert overrides["llm"]["model"] == "some-other-model"
    assert overrides["llm"]["base_url"] == "https://other.example.org/v1"


@pytest.mark.parametrize("page", [
    "https://huggingface.co/apodex/Apodex-1.1-mini",
    "https://www.huggingface.co/apodex/Apodex-1.1-mini",
    "https://hf.co/apodex/Apodex-1.1-mini",
])
def test_model_repository_page_is_rejected_as_an_endpoint(page: str) -> None:
    """The HF model page is not an inference API; say so before the first call."""
    checks = preflight(load_config(_env(OPENAI_BASE_URL=page)))
    assert not checks.ok
    issue = next(i for i in checks.errors if i.field == "OPENAI_BASE_URL")
    assert "model repository web page" in issue.message
    assert "/v1" in issue.message


def test_hf_inference_router_is_accepted() -> None:
    """``router.huggingface.co`` *is* OpenAI-compatible — don't over-reject."""
    checks = preflight(load_config(_env(
        OPENAI_BASE_URL="https://router.huggingface.co/v1",
    )))
    assert checks.ok, checks.format()


def test_full_chat_completions_url_is_rejected() -> None:
    checks = preflight(load_config(_env(
        OPENAI_BASE_URL="https://e.example.com/v1/chat/completions",
    )))
    assert not checks.ok
    assert any("base" in i.message for i in checks.errors)


@pytest.mark.parametrize("missing", ["OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL"])
def test_missing_required_settings_are_reported(missing: str) -> None:
    checks = preflight(load_config(_env(**{missing: ""})))
    assert not checks.ok
    assert any(issue.field == missing for issue in checks.errors)


def test_web_search_without_a_serper_key_warns_rather_than_failing_silently() -> None:
    """No key means empty results, which looks like a broken agent."""
    checks = preflight(load_config(_env(SERPER_API_KEY="")))
    assert checks.ok, checks.format()
    assert any(i.field == "SERPER_API_KEY" for i in checks.warnings)


def test_unknown_workflow_is_rejected() -> None:
    checks = preflight(load_config(_env(DEMO_WORKFLOW="agent_team")))
    assert not checks.ok
    assert any(i.field == "DEMO_WORKFLOW" for i in checks.errors)


def test_unattested_sandbox_backend_is_rejected() -> None:
    checks = preflight(load_config(_env(SANDBOX_BACKEND="")))
    assert not checks.ok
    assert any(i.field == "SANDBOX_BACKEND" for i in checks.errors)


def test_requesting_concurrency_above_one_warns_and_is_ignored() -> None:
    from deploy.huggingface.adapter import FrontierAgentAdapter

    config = load_config(_env(DEMO_MAX_CONCURRENCY="8"))
    checks = preflight(config)
    assert checks.ok, checks.format()
    assert any(i.field == "DEMO_MAX_CONCURRENCY" for i in checks.warnings)
    assert FrontierAgentAdapter(config).effective_concurrency == 1


def test_public_mode_refuses_a_shell_in_the_allowlist() -> None:
    checks = preflight(load_config(_env(DEMO_ALLOWED_TOOLS="web_search,bash")))
    assert not checks.ok
    assert any("bash" in i.message for i in checks.errors)


def test_limits_are_parsed_and_derived_budgets_fit_the_wall() -> None:
    config = load_config(_env(DEMO_TASK_TIMEOUT_SECONDS="300", DEMO_MAX_TURNS="9"))
    assert config.max_turns == 9
    assert config.task_timeout_s == 300
    # A tool starting just before the research deadline must still land inside
    # the hard wall, or the run dies with no answer.
    assert config.research_wall_time_s + config.tool_timeout_s <= config.hard_wall_time_s


def test_runaway_guards_are_rescaled_from_the_task_wall() -> None:
    """The profile's guardrails are sized for a 9000s TUI session, not a demo.

    Left alone, ``logical_call_timeout_s`` (900) outlives a default demo run,
    so one wedged call would be cut off by the adapter's ``wait_for`` backstop
    instead of the loop landing with a partial answer; and
    ``reasoning_only_max_tokens`` (16384) is unreachable when the whole output
    budget is 4096, leaving only the wall-clock half of a guard whose own
    profile comment says the token half is the one to trust.
    """
    config = load_config(_env(
        DEMO_TASK_TIMEOUT_SECONDS="600", DEMO_MAX_OUTPUT_TOKENS="4096",
    ))
    agent = config.profile_overrides()["agent"]

    # Present at all: a profile-side rename must fail here, not in production.
    for key in (
        "logical_call_timeout_s",
        "reasoning_only_timeout_s",
        "reasoning_only_max_tokens",
    ):
        assert key in agent, key

    # One logical call can never outlive the run that contains it, and never
    # undercuts a single attempt either.
    assert agent["llm_timeout_s"] <= agent["logical_call_timeout_s"]
    assert agent["logical_call_timeout_s"] <= config.task_timeout_s
    # The token guard has to be reachable inside the demo's output budget.
    assert 0 < agent["reasoning_only_max_tokens"] < config.max_output_tokens
    # And the reasoning window cannot exceed the per-call ceiling it sits in.
    assert agent["reasoning_only_timeout_s"] <= agent["llm_timeout_s"]


@pytest.mark.parametrize("wall", ["30", "120", "600", "1800", "3600"])
def test_derived_budgets_stay_ordered_at_any_wall(wall: str) -> None:
    """The relationships above must hold across the whole configurable range,
    not only at the default — a short wall is where the clamps collide."""
    config = load_config(_env(DEMO_TASK_TIMEOUT_SECONDS=wall))
    agent = config.profile_overrides()["agent"]
    assert agent["llm_timeout_s"] <= agent["logical_call_timeout_s"]
    # ``max`` rather than the wall alone: ``llm_timeout_s`` has a 60s floor
    # while the wall accepts 30s, so below 60s a single attempt already
    # outlives the run and no logical-call ceiling can fix that. Preflight
    # warns about the wall instead of this clamp pretending to.
    assert agent["logical_call_timeout_s"] <= max(
        config.task_timeout_s, agent["llm_timeout_s"],
    )
    assert agent["reasoning_only_timeout_s"] <= agent["llm_timeout_s"]
    assert 0 < agent["reasoning_only_max_tokens"] < config.max_output_tokens


@pytest.mark.parametrize("output_budget", ["256", "512", "4096", "16384"])
def test_reasoning_token_guard_fits_every_allowed_output_budget(
    output_budget: str,
) -> None:
    config = load_config(_env(DEMO_MAX_OUTPUT_TOKENS=output_budget))

    assert 0 < config.reasoning_only_max_tokens < config.max_output_tokens


def test_a_wall_shorter_than_one_llm_call_is_reported() -> None:
    """The floors make any wall under 60s incoherent; say so at startup."""
    checks = preflight(load_config(_env(DEMO_TASK_TIMEOUT_SECONDS="30")))
    assert checks.ok, "an incoherent-but-runnable wall is a warning, not an error"
    assert any(
        i.field == "DEMO_TASK_TIMEOUT_SECONDS" and "single model call" in i.message
        for i in checks.warnings
    )


def test_public_summary_never_leaks_the_key() -> None:
    config = load_config(_env(
        OPENAI_BASE_URL="https://user:pw@e.example.com/v1?key=sk-secret-value-1234",
    ))
    rendered = " ".join(config.public_summary().values())
    assert "sk-test-0123456789abcdef" not in rendered
    assert "sk-secret-value-1234" not in rendered
    assert "pw" not in rendered


def test_redact_url_strips_userinfo_and_query() -> None:
    assert redact_url("https://u:p@h.example/v1?k=abc") == "https://h.example/v1"


# ── secret redaction ─────────────────────────────────────────────────────


def test_redactor_masks_configured_secrets() -> None:
    redactor = Redactor.for_secrets(["sk-my-real-key-abcdef123456", "hf_tokenvalue1234567"])
    text = "calling with sk-my-real-key-abcdef123456 and hf_tokenvalue1234567"
    out = redactor.redact(text)
    assert "sk-my-real-key-abcdef123456" not in out
    assert "hf_tokenvalue1234567" not in out
    assert REDACTED in out


def test_redactor_masks_credential_shapes_it_was_never_told_about() -> None:
    out = Redactor.for_secrets([]).redact(
        'upstream said {"api_key": "abcdef1234567890"} Bearer zyxwvu9876543210abc',
    )
    assert "abcdef1234567890" not in out
    assert "zyxwvu9876543210abc" not in out


def test_redactor_ignores_short_values_that_would_destroy_the_logs() -> None:
    assert Redactor.for_secrets(["ab"]).redact("a stable message") == "a stable message"


def test_log_filter_redacts_message_and_args() -> None:
    secret = "sk-log-leak-key-9876543210"
    redactor = Redactor.for_secrets([secret])
    record = logging.LogRecord(
        "t", logging.WARNING, __file__, 1, "auth failed for %s", (secret,), None,
    )
    assert RedactingLogFilter(redactor).filter(record)
    assert secret not in record.getMessage()


# ── tool policy ──────────────────────────────────────────────────────────


def test_default_toolset_excludes_every_command_executing_tool() -> None:
    assert not set(DEFAULT_ALLOWED_TOOLS) & HARD_DENIED_TOOLS
    for denied in ("bash", "run_python_code", "download_file"):
        assert denied not in DEFAULT_ALLOWED_TOOLS


def test_policy_fails_closed_against_an_allowlist_that_asks_for_a_shell() -> None:
    names = demo_safe_tool_names(["web_search", "bash", "run_python_code"])
    assert names == ("web_search",)
    policy = demo_safe_tool_policy(["web_search", "bash"])
    assert policy.allows("web_search")
    assert not policy.allows("bash")
    assert not policy.allows("run_python_code")  # not even mentioned, still denied


def test_policy_denies_tools_outside_the_allowlist() -> None:
    policy = demo_safe_tool_policy(["web_search"])
    assert not policy.allows("write_file")


def test_non_public_mode_still_only_allows_the_named_tools() -> None:
    policy = demo_safe_tool_policy(["web_search", "bash"], public_mode=False)
    assert policy.allows("bash")          # an operator explicitly opted in
    assert not policy.allows("read_file")  # but the allowlist still binds


# ── download containment ─────────────────────────────────────────────────


def test_download_resolves_a_real_file(tmp_path) -> None:
    (tmp_path / "report.md").write_text("hello", encoding="utf-8")
    assert resolve_download(tmp_path, "report.md").name == "report.md"


@pytest.mark.parametrize("attempt", [
    "../../etc/passwd",
    "..%2f..%2fetc/passwd",
    "/etc/passwd",
    "sub/../../outside.txt",
    "",
])
def test_download_refuses_anything_outside_outputs(tmp_path, attempt: str) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path.parent / "outside.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(DownloadDenied):
        resolve_download(tmp_path, attempt)


def test_download_refuses_a_symlink_escaping_the_tree(tmp_path) -> None:
    target = tmp_path.parent / "escape.txt"
    target.write_text("secret", encoding="utf-8")
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "link.txt").symlink_to(target)
    with pytest.raises(DownloadDenied):
        resolve_download(outputs, "link.txt")
    assert list_output_files(outputs) == []


@pytest.mark.parametrize("name", [
    ".env", "api_secret.txt", "server.pem", "auth_token.json", ".hidden",
])
def test_sensitive_filenames_are_never_offered(tmp_path, name: str) -> None:
    (tmp_path / name).write_text("x", encoding="utf-8")
    assert list_output_files(tmp_path) == []
    with pytest.raises(DownloadDenied):
        resolve_download(tmp_path, name)


# ── upstream error mapping ───────────────────────────────────────────────


class _HTTPStatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"upstream returned {status}")
        self.status_code = status


@pytest.mark.parametrize(("status", "slug"), [
    (400, "upstream_bad_request"),
    (401, "upstream_unauthorized"),
    (403, "upstream_forbidden"),
    (404, "upstream_not_found"),
    (422, "upstream_unprocessable"),
    (429, "upstream_rate_limited"),
    (500, "upstream_server_error"),
    (503, "upstream_unavailable"),
])
def test_http_statuses_map_to_actionable_reasons(status: int, slug: str) -> None:
    reason, message = classify_error(_HTTPStatusError(status))
    assert reason == slug
    assert message and not message.startswith("Traceback")


def test_wrapped_provider_error_is_unwrapped_for_classification() -> None:
    from frontier_agent.core.errors import LLMCallExhausted

    reason, _ = classify_error(LLMCallExhausted(_HTTPStatusError(429), "retries"))
    assert reason == "upstream_rate_limited"


def test_timeout_and_connection_failures_are_distinguished() -> None:
    assert classify_error(TimeoutError("slow"))[0] == "upstream_timeout"
    assert classify_error(ConnectionRefusedError("nope"))[0] == "upstream_unreachable"


def test_404_message_points_at_the_two_variables_that_cause_it() -> None:
    _, message = classify_error(_HTTPStatusError(404))
    assert "OPENAI_BASE_URL" in message
    assert "OPENAI_MODEL" in message


def test_loop_watchdog_failures_are_not_blamed_on_the_endpoint() -> None:
    """A runaway or a stalled stream is not "the endpoint could not complete".

    Both are raised by the loop's own watchdogs while the provider is still
    responding, so the generic upstream sentence would send an operator to
    inspect a healthy service. ``LLMStreamStalled`` also subclasses
    ``TimeoutError``, which the timeout branch would otherwise absorb.
    """
    from frontier_agent.core.errors import (
        LLMCallExhausted,
        LLMReasoningRunaway,
        LLMStreamStalled,
    )

    runaway = LLMReasoningRunaway(
        elapsed_s=121.0, estimated_tokens=2500,
        trigger="reasoning_only_timeout_s", partial_response=None,
    )
    stalled = LLMStreamStalled(180.0, 12, 200.0)

    assert classify_error(runaway)[0] == "model_reasoning_runaway"
    assert classify_error(stalled)[0] == "upstream_stream_stalled"
    # Also when the retry wrapper is what surfaces, which is the usual case.
    assert classify_error(
        LLMCallExhausted(stalled, "stream_stalled"),
    )[0] == "upstream_stream_stalled"

    # And by class name alone: once the workflow has absorbed the failure into
    # a best-effort answer, the name in ``LLMAttemptContext.error_type`` is all
    # the demo is left with.
    assert classify_error_name("LLMReasoningRunaway")[0] == "model_reasoning_runaway"
    assert classify_error_name("LLMStreamStalled")[0] == "upstream_stream_stalled"
    assert "healthy" in classify_error_name("LLMReasoningRunaway")[1]


# ── filesystem containment ───────────────────────────────────────────────


def _containment(tmp_path, monkeypatch):
    """A containment observer over a session tree, with the mounts pointed at it.

    Mirrors what ``adapter._runtime_env`` exports for a real run: the aliases
    the tools teach resolve to *this* session's directories.
    """
    from deploy.huggingface.containment import PathContainmentObserver

    workspace = tmp_path / "workspace"
    outputs = workspace / "outputs"
    inputs = tmp_path / "inputs"
    for path in (workspace, outputs, inputs):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FRONTIER_AGENT_WORKSPACE_DIR", str(workspace))
    monkeypatch.setenv("FRONTIER_AGENT_OUTPUTS_DIR", str(outputs))
    monkeypatch.setenv("FRONTIER_AGENT_INPUTS_DIR", str(inputs))
    return PathContainmentObserver(workspace=workspace, read_roots=(inputs,))


@pytest.mark.parametrize(("tool", "args"), [
    # Every file tool's own description teaches these aliases, and native mode
    # resolves them (``plugins/tools/_sandbox.resolve_runtime_path``). Refusing
    # them would cost a turn on a call that was never an escape attempt.
    ("read_file", {"path": "/inputs/data.csv"}),
    ("write_file", {"path": "/outputs/report.md"}),
    ("write_file", {"path": "/workspace/scratch.txt"}),
    ("write_file", {"path": "outputs/report.md"}),
])
async def test_containment_accepts_the_aliases_the_tools_teach(
    tmp_path, monkeypatch, tool: str, args: dict,
) -> None:
    observer = _containment(tmp_path, monkeypatch)
    assert await observer.on_tool_call(None, {"name": tool, "args": args}) is None


@pytest.mark.parametrize(("tool", "args", "expected"), [
    # An alias is not a blanket pass: the traversal is refused on the raw
    # argument, before mapping, because the mapping normalises as it rewrites.
    ("write_file", {"path": "/outputs/../../etc/passwd"}, "traversal"),
    # A sibling of an alias is a different directory and is not rewritten.
    ("write_file", {"path": "/outputs-old/x"}, "outside"),
    ("write_file", {"path": "/etc/passwd"}, "outside"),
    # inputs/ stays readable but immutable, alias or not.
    ("write_file", {"path": "/inputs/evil.txt"}, "read-only"),
])
async def test_containment_refuses_an_alias_that_is_not_the_session_tree(
    tmp_path, monkeypatch, tool: str, args: dict, expected: str,
) -> None:
    observer = _containment(tmp_path, monkeypatch)
    intervention = await observer.on_tool_call(None, {"name": tool, "args": args})
    assert intervention is not None
    assert expected in str(intervention.skip_with_result)
