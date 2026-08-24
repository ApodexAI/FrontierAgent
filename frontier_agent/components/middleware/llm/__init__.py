"""LLM middleware base types + lazy-loaded concrete impls.

Every name in ``_IMPL_MAP`` must resolve — the map is this package's public
contract, and a stale entry raises ``ModuleNotFoundError`` only when someone
first reaches for that attribute. ``tools/check_lazy_exports.py`` proves the
map, because a static symbol check cannot: it treats ``__all__`` as the
contract for a lazy package, and so believes an entry whose module is absent.
"""

from typing import TYPE_CHECKING, Any

from frontier_agent.components.middleware.llm.base import (
    _RETRYABLE_KEYWORDS,
    LLMCallContext,
    LLMMiddleware,
    LLMMiddlewareChain,
    _is_retryable,
    unwrap_runnable_binding,
)
from frontier_agent.components.middleware.llm.proxy import LLMProxy

if TYPE_CHECKING:
    from frontier_agent.components.middleware.llm.skill_injection import (
        SkillInjectionMiddleware,
    )
    from frontier_agent.components.middleware.llm.stream_repetition import (
        StreamRepetitionDetectorMiddleware,
    )
    from frontier_agent.components.middleware.llm.summarization import (
        SummarizationMiddleware,
    )

_IMPL_MAP: dict[str, tuple[str, str]] = {
    "SkillInjectionMiddleware": (
        "frontier_agent.components.middleware.llm.skill_injection",
        "SkillInjectionMiddleware",
    ),
    "StreamRepetitionDetectorMiddleware": (
        "frontier_agent.components.middleware.llm.stream_repetition",
        "StreamRepetitionDetectorMiddleware",
    ),
    "SummarizationMiddleware": (
        "frontier_agent.components.middleware.llm.summarization",
        "SummarizationMiddleware",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _IMPL_MAP:
        import importlib
        module_path, attr_name = _IMPL_MAP[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "_RETRYABLE_KEYWORDS",
    "LLMCallContext",
    "LLMMiddleware",
    "LLMMiddlewareChain",
    "LLMProxy",
    "SkillInjectionMiddleware",
    "StreamRepetitionDetectorMiddleware",
    "SummarizationMiddleware",
    "_is_retryable",
    "unwrap_runnable_binding",
]
