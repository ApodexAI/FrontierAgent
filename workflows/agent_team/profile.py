"""Load agent-team YAML profiles and build their LLM clients."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from frontier_agent.core.runtime.loop.model_profile import ThinkingFormat

from frontier_agent.core.llm import LLMClient
from frontier_agent.infra.providers import ProviderNotFound
from frontier_agent.infra.session_context import (
    eas_session_headers,
    provider_needs_session_header,
)

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Retired profile names kept resolvable so saved command lines, CI jobs and
# ``report_profile`` metadata written before the rename don't hard-fail with
# FileNotFoundError. ``agent_team_report`` is still matched by name in
# ``resolve_reporter_enabled`` below, so it keeps its reporter-on default.
_PROFILE_ALIASES = {
    "default": "benchmark",
    "agent_team_report": "benchmark",
    "local": "benchmark",
    "Apodex1.1-discover": "tui",
}

REPORTER_BACKEND_HEAVY = "heavy"
REPORTER_BACKEND_FAST = "fast"
_REPORTER_BACKEND_ALIASES = {
    "heavy": REPORTER_BACKEND_HEAVY,
    "heavy_reporter": REPORTER_BACKEND_HEAVY,
    "heavy_reporter_v3": REPORTER_BACKEND_HEAVY,
    "fast": REPORTER_BACKEND_FAST,
    "fast_report": REPORTER_BACKEND_FAST,
    "fast_reporter": REPORTER_BACKEND_FAST,
    # Input-only compatibility with the feature branch's original public name.
    "fast_reporter_v1": REPORTER_BACKEND_FAST,
}


def resolve_reporter_enabled(
    profile: dict[str, Any] | None,
    *,
    pipeline_id: str,
    profile_name: str | None,
) -> bool:
    """Return the effective agent-team reporter switch.

    Both agent-team specs use the profile flag.  The report-named pipeline and
    profile retain their historical default-on behaviour only when the merged
    profile does not explicitly contain ``agent.reporter``.
    """
    agent_cfg = (profile or {}).get("agent") or {}
    reporter_default = (
        pipeline_id in ("agent_team_report", "agent-team-report")
        or profile_name == "agent_team_report"
    )
    return bool(agent_cfg.get("reporter", reporter_default))


def resolve_reporter_backend(profile: dict[str, Any] | None) -> str:
    """Resolve the backend name while keeping the future-backend seam stable.

    Fast is the only shipped backend and remains the safe default. Compatibility
    aliases resolve here so the dispatcher can return an actionable error.
    """
    agent_cfg = (profile or {}).get("agent") or {}
    raw = str(agent_cfg.get("reporter_backend") or REPORTER_BACKEND_FAST)
    normalized = raw.strip().lower().replace("-", "_")
    backend = _REPORTER_BACKEND_ALIASES.get(normalized)
    if backend is not None:
        return backend
    logger.warning(
        "Unknown agent.reporter_backend=%r; using %s",
        raw,
        REPORTER_BACKEND_FAST,
    )
    return REPORTER_BACKEND_FAST


def deep_merge_overrides(
    base: dict[str, Any], overrides: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge ``overrides`` onto ``base`` without mutating either.

    Semantics chosen to match SDK caller intent (override one nested
    knob, leave siblings alone):

    - Nested dicts merge recursively (sibling keys preserved).
    - Lists are replaced wholesale — partial list merging is almost
      never what callers want (e.g. ``fallback_api_keys: [new_key]``
      should *replace* the existing key list, not append).
    - ``None`` in the override **removes** the key, so callers can
      disable a defaulted-on feature.
    - Scalar override replacing a dict in base wins — caller asked
      for that shape, however unusual.
    """
    if not overrides:
        # Returning a deep copy preserves the contract that the caller
        # can mutate the result without leaking back into ``base``.
        return _deep_copy(base)

    out: dict[str, Any] = _deep_copy(base)
    for key, value in overrides.items():
        if value is None:
            out.pop(key, None)
            continue
        base_value = out.get(key)
        if isinstance(base_value, dict) and isinstance(value, dict):
            out[key] = deep_merge_overrides(base_value, value)
        else:
            out[key] = _deep_copy(value)
    return out


def _deep_copy(value: Any) -> Any:
    """Plain dict/list deep copy (json-safe shapes only)."""
    if isinstance(value, dict):
        return {k: _deep_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy(v) for v in value]
    return value


def load_swarm_profile(
    name: str,
    *,
    overrides: dict[str, Any] | None = None,
    inline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an agent_team profile with env-var resolution + optional overrides.

    Resolution order:

    1. If ``inline`` is given, it is used as the base profile dict and the
       YAML file lookup is skipped entirely. This is how SDK / docker
       callers ship a full profile via the API without baking a YAML into
       the image — pass the dict as the API request's
       ``config["profile_inline"]``.
    2. Otherwise, ``workflows/agent_team/profiles/{name}.yaml`` is read.

    Then ``_resolve_env_vars`` replaces ``${VAR}`` placeholders so inline
    callers benefit from the same env-var substitution as YAML profiles.

    Finally, ``overrides`` (if any) is deep-merged on top so callers can
    tweak ``agent.main_max_turns`` and similar settings without
    editing the YAML inside the docker image. See
    :func:`deep_merge_overrides` for merge semantics.
    """
    import yaml
    from dotenv import load_dotenv

    from frontier_agent.infra.config import _resolve_env_vars

    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    if inline is not None:
        raw: Any = inline
    else:
        resolved_name = _PROFILE_ALIASES.get(name, name)
        path = _PROFILES_DIR / f"{resolved_name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Swarm profile not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    resolved = _resolve_env_vars(raw)
    _inject_provider_creds(resolved)
    if overrides:
        resolved = deep_merge_overrides(resolved, overrides)
    _merge_blacklist_file(resolved)
    return resolved


def _merge_blacklist_file(profile: dict[str, Any]) -> None:
    """Merge a shared blocklist file into ``web_domain_blacklist``.

    ``web_domain_blacklist_file:`` names a YAML file (path relative to
    the project root, e.g. ``config/web_blacklist.yaml``) whose
    ``blocked:`` list is appended to the profile's inline
    ``web_domain_blacklist:`` entries — so the long, shared list lives
    in one file instead of being duplicated across profiles, and a
    profile can still add its own inline entries on top.

    Runs AFTER ``overrides`` merging so an override replacing the inline
    list cannot silently drop the file's entries. A missing file raises
    — silently shipping a deploy without its blocklist is worse than
    failing fast.
    """
    import yaml

    file_ref = profile.get("web_domain_blacklist_file")
    if not file_ref or not isinstance(file_ref, str):
        return
    path = Path(file_ref)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"web_domain_blacklist_file not found: {path} "
            f"(referenced by profile)"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("blocked") or []
    if not isinstance(entries, list):
        raise ValueError(
            f"web_domain_blacklist_file {path}: ``blocked:`` must be a list"
        )
    inline = profile.get("web_domain_blacklist") or []
    if isinstance(inline, str):
        inline = [inline]
    profile["web_domain_blacklist"] = [
        str(e) for e in inline
    ] + [str(e) for e in entries]


def _inject_provider_creds(profile: dict[str, Any]) -> None:
    """For every LLM-shaped block that declares a ``provider`` name,
    look it up in the registry and inject ``api_key`` / ``base_url``.

    Heavy_mode profiles carry multiple LLM blocks at the top level
    (``llm``, ``dag_llm``, ``decision_llm``, ``outline_llm``,
    ``report_llm``, ``synth_llm``) — each can declare its own provider
    so a single profile can mix vendors (e.g. qwen for the main agent
    + claude-via-openrouter for the writer).

    An inline ``api_key`` / ``base_url`` on the block still wins
    (``setdefault`` semantics) so legacy profiles with explicit creds
    keep working unchanged. The resolved provider label is stamped
    onto ``<block>['_provider_label']`` so the LLM factory can pass it
    to ``with_provider_stamp`` (replacing the previous hardcoded
    ``"openai"``). The leading underscore signals "internal, don't
    write back out".

    Mismatch detection: when an inline ``base_url`` is set but differs
    from the registered provider's ``base_url``, usage events would be
    attributed to the wrong provider (the wire actually targets the
    inline endpoint, but ``_provider_label`` says the registered name).
    A warning is logged and the block's ``provider_label:`` override
    (if present) becomes the stamped label so callers can accurately
    attribute usage — without that override the registered ``provider``
    name still wins.
    """
    from frontier_agent.infra.providers import resolve_provider

    def _inject(block: dict[str, Any]) -> None:
        name = block.get("provider")
        if not name or not isinstance(name, str):
            return
        try:
            prov = resolve_provider(name)
        except ProviderNotFound as exc:
            # ``provider:`` is commonly an env placeholder now, so an
            # unregistered name is user input, not a packaging bug. Raise the
            # ValueError that profile loading already contracts for, so the
            # caller reports it instead of dying on an unhandled KeyError.
            raise ValueError(
                f"{exc.args[0]} (check OPENAI_PROVIDER or the profile's "
                "``provider:`` field)"
            ) from exc

        inline_base_url = block.get("base_url")
        registry_base_url = prov.get("base_url", "")

        # Mismatch warn: only when *both* sides are non-empty and they
        # differ. Empty registry value means the provider declared no
        # default (caller must supply one); empty inline means we just
        # accept the registry default. Either case is intentional.
        explicit_label = block.get("provider_label")
        if (
            inline_base_url
            and registry_base_url
            and inline_base_url != registry_base_url
            and not explicit_label
        ):
            logger.warning(
                "profile: block declares provider=%r but inline base_url "
                "(%s) differs from the registered endpoint (%s). Usage "
                "events will be attributed to %r even though the wire "
                "targets a different backend. Add ``provider_label: "
                "<true-backend>`` to the block to override the label, "
                "or drop the inline base_url to use the registered one.",
                name, inline_base_url, registry_base_url, name,
            )

        # Surface env-var misconfigs at config-load time. When the
        # registry's ``${X_API_KEY}`` placeholder resolved to empty (env
        # unset or typoed) AND the block has no inline override, the
        # downstream LLM call would normally produce a cryptic 401
        # "Missing Authentication header" because ``build_aux_llm`` falls
        # back to the literal ``"dummy"`` Bearer. A loud WARN here lets
        # operators connect that 401 to the actual root cause (env var)
        # without digging through the stack. Logged once per block —
        # idempotent setdefault below keeps behaviour identical.
        registry_api_key = prov.get("api_key", "")
        if not registry_api_key and not block.get("api_key"):
            env_var = f"{name.upper()}_API_KEY"
            logger.warning(
                "profile: provider %r resolved to an EMPTY api_key — "
                "set $%s in your environment (or .env). Without it, "
                "build_aux_llm will fall back to sending the literal "
                "string 'dummy' as the Bearer token, which surfaces "
                "downstream as '401 Missing Authentication header'.",
                name, env_var,
            )
        block.setdefault("api_key", registry_api_key)
        block.setdefault("base_url", registry_base_url)
        # ``provider_label:`` opt-out: when the caller knowingly routes
        # to a different backend via inline creds, they can name the
        # *true* backend here so usage accounting reflects reality.
        # Strip the field from the block so it doesn't leak into the
        # downstream LLM kwargs.
        if explicit_label and isinstance(explicit_label, str):
            block["_provider_label"] = explicit_label
            block.pop("provider_label", None)
        else:
            block["_provider_label"] = name

    for value in profile.values():
        if not isinstance(value, dict):
            continue
        _inject(value)
        # ``llm.fallback:`` declares a secondary provider for the
        # native ``LLMFallbackChain`` wrapper (see ``create_swarm_llm``).
        # Inject creds for it too so create_swarm_llm can build the
        # secondary ``OpenAIClient`` leg without re-reading the registry.
        fallback = value.get("fallback")
        if isinstance(fallback, dict):
            _inject(fallback)


def _resolve_thinking_format(profile: dict[str, Any]) -> ThinkingFormat:
    """Pick the thinking_format for this profile.

    Order: explicit ``agent.thinking_format`` in YAML → inferred from the
    model id (see :func:`frontier_agent.core.runtime.loop.model_profile.
    infer_thinking_format`) → ``tag`` (the safe fallback — the parser
    no-ops on responses without ``<think>`` blocks).
    """
    from frontier_agent.core.runtime.loop.model_profile import (
        infer_thinking_format,
        is_thinking_format,
    )
    from frontier_agent.infra.protocol_client import (
        protocol_of,
        thinking_format_for_protocol,
    )
    explicit = (profile.get("agent") or {}).get("thinking_format")
    if explicit:
        if is_thinking_format(explicit):
            return explicit
        logger.warning(
            "profile agent.thinking_format=%r is not a known format — ignoring",
            explicit,
        )
    # Native Anthropic / OpenAI Responses return content as a typed block list
    # → content_block so the parser keeps the verbatim blocks.
    by_protocol = thinking_format_for_protocol(protocol_of(profile.get("llm") or {}))
    if by_protocol:
        return by_protocol
    model_id = (profile.get("llm") or {}).get("model")
    return infer_thinking_format(model_id, default="tag")


def _build_leg(cfg: dict[str, Any]) -> LLMClient:
    """Build one native ``OpenAIClient`` leg from an ``llm:``-shaped dict.

    Factored out of ``create_swarm_llm`` so the same wiring (extra_body
    + ``top_p`` / ``repetition_penalty`` folding, sticky session headers,
    provider stamp) builds the primary and the ``llm.fallback:`` secondary
    uniformly.

    Reasoning round-trip is handled below the workflow: native
    :class:`~frontier_agent.infra.openai_client.OpenAIClient` already rescues
    inbound ``reasoning_content`` (SGLang / Qwen / DeepSeek / Doubao Seed),
    and the kernel ``NativeMessageNormalizer`` owns the OUTBOUND
    tag / ``reasoning_content`` wire round-trip per the profile's
    ``thinking_format``, so the workflow never reimplements it.
    """
    from frontier_agent.infra.llm import with_provider_stamp
    from frontier_agent.infra.openai_client import OpenAIClient
    from frontier_agent.infra.protocol_client import build_protocol_client

    # ``protocol: anthropic|responses`` → native reasoning-capturing client
    # (session headers / extra_body sampling knobs below are Chat-Completions
    # specific and don't apply). chat_completions falls through unchanged.
    protocol_client = build_protocol_client(cfg, title="FrontierAgent-Swarm")
    if protocol_client is not None:
        provider = cfg.get("_provider_label") or cfg.get("provider") or "openai"
        return with_provider_stamp(protocol_client, provider)

    extra_body: dict[str, Any] = dict(cfg.get("extra_body") or {})
    # Fold vLLM / SGLang sampling extensions into extra_body.
    if (top_p := cfg.get("top_p")) is not None:
        extra_body.setdefault("top_p", top_p)
    repetition_penalty = cfg.get("repetition_penalty")
    if (
        repetition_penalty is not None
        and repetition_penalty != 1.0
        and "repetition_penalty" not in extra_body
    ):
        extra_body["repetition_penalty"] = repetition_penalty

    default_headers: dict[str, str] = {
        "HTTP-Referer": "frontier_agent",
        "X-Title": "FrontierAgent-Swarm",
    }
    # Backends that require session affinity receive the task-scoped headers
    # configured by ``frontier_agent/infra/session_context.py``. Other providers
    # and calls outside a task context receive no additional headers.
    provider_for_session = (
        cfg.get("_provider_label") or cfg.get("provider") or ""
    )
    if provider_needs_session_header(provider_for_session):
        default_headers.update(eas_session_headers())

    provider = (
        cfg.get("_provider_label")
        or cfg.get("provider")
        or "openai"
    )

    client = OpenAIClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url"),
        temperature=cfg.get("temperature", 0.0),
        # int(): env-var substitution in the profile YAML always yields a
        # string, and this value goes straight into the request body.
        max_completion_tokens=int(cfg.get("max_tokens") or 65536),
        default_headers=default_headers,
        extra_body=extra_body or None,
    )
    return with_provider_stamp(client, provider)


def create_swarm_llm(profile: dict[str, Any]) -> LLMClient:
    """Create an OpenAI-compatible ``LLMClient`` from an agent_team profile.

    ``llm.extra_body`` forwards provider-specific params (e.g.
    ``reasoning_effort``, ``top_p``, ``repetition_penalty``) into every
    request.

    When ``llm.fallback:`` is present, the result is a native
    :class:`~frontier_agent.infra.llm.LLMFallbackChain` that tries the
    primary first and falls over to the secondary on quota / overload /
    timeout failures (``rate_limit`` covers 429 ``insufficient_quota``,
    ``5xx`` covers overload, ``timeout`` covers stalls). Other errors
    propagate immediately. Both legs share the profile's
    ``thinking_format`` (resolved below the workflow by the kernel
    normalizer) so the wire shape stays consistent on rollover.
    """
    cfg = profile["llm"]
    primary = _build_leg(cfg)

    fallback_cfg = cfg.get("fallback")
    if not isinstance(fallback_cfg, dict) or not fallback_cfg.get("model"):
        return primary

    from frontier_agent.infra.llm import (
        FallbackEntry,
        LLMFallbackChain,
    )

    fallback = _build_leg(fallback_cfg)
    primary_provider = (
        cfg.get("_provider_label") or cfg.get("provider") or "openai"
    )
    fallback_provider = (
        fallback_cfg.get("_provider_label")
        or fallback_cfg.get("provider")
        or "openai"
    )
    triggers = ("rate_limit", "5xx", "timeout")
    return LLMFallbackChain(
        entries=[
            FallbackEntry(
                model=primary, triggers=triggers, provider=primary_provider,
            ),
            # Last entry: the chain re-raises on it regardless of triggers,
            # so the fallback leg's own errors propagate to the caller.
            FallbackEntry(
                model=fallback, triggers=triggers, provider=fallback_provider,
            ),
        ],
    )


def build_swarm_model_profile(profile: dict[str, Any]) -> Any:
    """Build a ModelProfile from an agent_team profile dict."""
    from frontier_agent.core.runtime.loop.model_profile import ModelProfile
    from frontier_agent.infra.protocol_client import protocol_of
    return ModelProfile(
        model_id=profile["llm"]["model"],
        provider="openai",
        thinking_format=_resolve_thinking_format(profile),
        protocol=protocol_of(profile.get("llm") or {}),
    )
