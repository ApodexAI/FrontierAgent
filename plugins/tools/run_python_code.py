"""Tool: run_python_code — execute Python code in an isolated sandbox."""
from __future__ import annotations

import ast
import asyncio
import logging
import re
import uuid
from pathlib import Path

from frontier_agent.core.tool import tool
from frontier_agent.infra.usage_meter import record_api_request
from plugins.tools._code_sanitize import sanitize_code
from plugins.tools._net_guard import ensure_guard_file, guard_env_prefix
from plugins.tools._overflow import maybe_overflow
from plugins.tools._sandbox import (
    aget_sandbox,
    arun_sandbox_cmd,
    is_e2b_sandbox,
    remote_exec_prefix,
)
from plugins.tools.bash import BASH_STDERR_SEPARATOR

logger = logging.getLogger(__name__)

_MAX_OUTPUT = 10_000

# Tail of captured stdout surfaced when a run is killed at the timeout, so a
# batched crawl that printed progress leaves salvageable output instead of a
# total loss (observed: 2×600s OpenAlex pagination crawls, zero output kept).
_PARTIAL_TAIL = 3_000

# Detect-only (2026-06-04): log when agent code does raw HTTP so we can size
# how often crawls bypass the governed web tools (proxy cache, retries,
# metering) before deciding whether to clamp their timeout. See the network
# discipline section of the sub-agent research prompt.
_NET_LIB_RE = re.compile(
    r"^\s*(?:import|from)\s+(requests|httpx|aiohttp|urllib3|urllib|socket)\b",
    re.MULTILINE,
)

_ML_MODULES = frozenset({
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "datasets",
    "huggingface_hub",
    "sentence_transformers",
    "diffusers",
    "accelerate",
    "tensorflow",
    "keras",
})
_ML_INSTALL_PACKAGES = _ML_MODULES | {
    "huggingface-hub",
    "sentence-transformers",
}
_ML_DOWNLOAD_ATTRS = frozenset({
    "from_pretrained",
    "snapshot_download",
    "hf_hub_download",
    "load_state_dict_from_url",
})
_SUBPROCESS_CALLS = frozenset({
    "subprocess.run",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.Popen",
})

# Hard-deny heavyweight ML/model-download paths in the generic research
# sandbox. These libraries are not part of run_python_code's advertised
# package set, and their convenience APIs can pull multi-GB weights into RAM or
# disk cache before the Python process sees a clean MemoryError.
_ML_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+("
    r"torch|torchvision|torchaudio|transformers|datasets|huggingface_hub|"
    r"sentence_transformers|diffusers|accelerate|tensorflow|keras"
    r")(?:\b|\.)",
    re.MULTILINE,
)
_ML_DOWNLOAD_API_RE = re.compile(
    r"\b("
    r"torch\.hub\.(?:load|download_url_to_file)|"
    r"load_state_dict_from_url|"
    r"(?:from_pretrained|snapshot_download|hf_hub_download)\s*\("
    r")"
)
_ML_INSTALL_RE = re.compile(
    r"\b(?:pip|python\s+-m\s+pip|uv\s+pip)\s+install\b[^\n;]*\b("
    r"torch|torchvision|torchaudio|transformers|datasets|huggingface-hub|"
    r"sentence-transformers|diffusers|accelerate|tensorflow|keras"
    r")\b",
    re.IGNORECASE,
)

_ML_BLOCK_MESSAGE = (
    "Error: run_python_code blocks PyTorch/HuggingFace/transformers-style "
    "model loading and downloads in the generic research sandbox. These "
    "paths can fetch multi-GB weights or metadata caches and OOM the worker. "
    "Use lightweight structured APIs, aggregate/count endpoints, or the "
    "governed web_search/web_fetch tools instead."
)

_OFFLINE_DOWNLOAD_ENV = (
    "HF_HUB_OFFLINE=1 "
    "TRANSFORMERS_OFFLINE=1 "
    "HF_DATASETS_OFFLINE=1 "
    "HF_HUB_DISABLE_TELEMETRY=1 "
    "TORCH_HOME=/tmp/frontier_agent_no_torch_cache "
    "HF_HOME=/tmp/frontier_agent_no_hf_cache "
    "TRANSFORMERS_CACHE=/tmp/frontier_agent_no_hf_cache "
)

# Streaming recipe shared by the OOM (-1) and MemoryError messages. Precise on
# purpose: a vague "use chunked processing" hint led models to chunk only the
# pandas parse while still buffering the whole download via
# ``requests.get(url).content`` — the retry then OOM'd identically.
_MEM_RECIPE = (
    "- pd.read_csv(url, chunksize=10000) streams the download AND the parse; "
    "filter each chunk, keep only needed columns/rows.\n"
    "- Or requests.get(url, stream=True) + iterate lines; never touch "
    "response.content / .text on a large body (it buffers everything, and "
    ".decode() doubles it).\n"
    "- Write the filtered subset to a file in the current working directory "
    "first, then analyze that small file in a second run."
)


def _default_timeout() -> int:
    """Per-exec wall-clock default from config (run_python_timeout_s)."""
    try:
        from frontier_agent.infra.config import get_config
        return int(get_config().run_python_timeout_s)
    except Exception:
        return 300


def _max_timeout() -> int:
    """Hard ceiling on an agent-supplied timeout (run_python_max_timeout_s)."""
    try:
        from frontier_agent.infra.config import get_config
        return int(get_config().run_python_max_timeout_s)
    except Exception:
        return 300


def _root_module(name: str) -> str:
    return name.split(".", 1)[0]


def _literal_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _iter_literal_strings(node: ast.AST) -> list[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return []
    out: list[str] = []
    for elt in node.elts:
        text = _literal_str(elt)
        if text is not None:
            out.append(text)
    return out


def _is_blocked_pip_install_args(args: list[str]) -> bool:
    lowered = [arg.lower() for arg in args]
    if "install" not in lowered:
        return False
    installer = lowered[:3]
    if not (
        any(arg.endswith("pip") or arg in {"pip", "pip3"} for arg in installer)
        or (len(installer) >= 3 and installer[1:3] == ["-m", "pip"])
        or installer[:2] == ["uv", "pip"]
    ):
        return False
    install_at = lowered.index("install")
    packages = {
        arg.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0]
        for arg in lowered[install_at + 1 :]
        if arg and not arg.startswith("-")
    }
    return any(pkg in _ML_INSTALL_PACKAGES for pkg in packages)


class _MLDownloadBlockVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.blocked = False

    def visit_Import(self, node: ast.Import) -> None:
        if any(_root_module(alias.name) in _ML_MODULES for alias in node.names):
            self.blocked = True
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and _root_module(node.module) in _ML_MODULES:
            self.blocked = True
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)
        if name == "__import__" and node.args:
            mod = _literal_str(node.args[0])
            if mod and _root_module(mod) in _ML_MODULES:
                self.blocked = True
                return
        if name == "importlib.import_module" and node.args:
            mod = _literal_str(node.args[0])
            if mod and _root_module(mod) in _ML_MODULES:
                self.blocked = True
                return
        if (
            name in {"torch.hub.load", "torch.hub.download_url_to_file"}
            or name.rsplit(".", 1)[-1] in _ML_DOWNLOAD_ATTRS
        ):
            self.blocked = True
            return
        if name == "getattr" and len(node.args) >= 2:
            attr = _literal_str(node.args[1])
            if attr in _ML_DOWNLOAD_ATTRS:
                self.blocked = True
                return
        if (
            name in _SUBPROCESS_CALLS
            and node.args
            and _is_blocked_pip_install_args(_iter_literal_strings(node.args[0]))
        ):
            self.blocked = True
            return
        self.generic_visit(node)


def _blocked_ml_download_ast(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    visitor = _MLDownloadBlockVisitor()
    visitor.visit(tree)
    return visitor.blocked


def _blocked_ml_download_reason(code: str) -> str | None:
    """Return a user-facing block reason for heavyweight ML download paths."""
    if _ML_IMPORT_RE.search(code):
        return _ML_BLOCK_MESSAGE
    if _ML_DOWNLOAD_API_RE.search(code):
        return _ML_BLOCK_MESSAGE
    if _ML_INSTALL_RE.search(code):
        return _ML_BLOCK_MESSAGE
    if _blocked_ml_download_ast(code):
        return _ML_BLOCK_MESSAGE
    return None


@tool
async def run_python_code(code: str, timeout: int = 0) -> str:
    """Execute Python code in an isolated sandbox environment.

    Pre-installed packages: numpy, pandas, scipy, sympy, mpmath, networkx,
    plotly, Pillow, openpyxl, tabulate.

    The code MUST terminate on its own within seconds. NEVER submit code that
    runs forever: no unbounded ``while True:`` refresh loops, no live curses /
    TUI dashboards, no servers, daemons, or ``input()`` waits. If you are
    testing a long-running program (e.g. a monitoring dashboard that samples
    repeatedly), drive it with a BOUNDED test harness instead — call its
    collect/render function 2-3 times with a short (<=1s) sleep, or run it with
    an explicit ``--iterations N`` / ``--once`` flag — so it exits quickly. A
    non-terminating program is killed only at the timeout, wasting the whole
    budget; repeated kills can stall the entire task.

    Network rules: to read the CONTENT of a web page or PDF use ``web_fetch``,
    not requests/httpx. Calling a structured API (OpenAlex, Crossref, EDGAR,
    ...) from code is allowed ONLY for aggregate/count queries (e.g. OpenAlex
    ``group_by``, ``meta.count``) — a handful of requests that each return a
    small statistical answer. Bulk-paginating a full result set (downloading
    every record's metadata page by page) is FORBIDDEN: a corpus-scale crawl
    cannot be verified item-by-item anyway, so sample instead — get the total
    via an aggregate endpoint, pull ≤2 pages as a representative sample, and
    reason from count + sample. Set a short per-request timeout and write any
    intermediate data to files in the current working directory, not stdout.
    If the plan seems to need more than ~10 requests, narrow the query
    server-side.

    Scratch files: use RELATIVE paths. The working directory is private to
    this agent and persists across calls, so a follow-up run sees what this
    one wrote. Absolute ``/tmp`` and ``/workspace`` paths are shared with
    every other agent on this task — a fixed absolute name can be overwritten
    by a concurrent agent, and you would read back their data as if it were
    yours.

    Output rules: NEVER print a full dataset or raw API responses to stdout —
    print counts, aggregates, and at most ~20 sample rows. Anything larger
    belongs in a file in the working directory (analyze it in a follow-up
    run). Stdout is capped; a full-corpus dump gets truncated AND bloats every
    downstream consumer of this conversation.

    Memory rules: the sandbox has LIMITED RAM (~512MB) and an out-of-memory
    kill loses the whole run. Any file/dataset over ~20MB MUST be streamed,
    never buffered: use ``pd.read_csv(url, chunksize=10000)`` (streams both
    download and parse; filter each chunk) or ``requests.get(url,
    stream=True)`` + line iteration. NEVER call ``response.content`` /
    ``response.text`` on a large body — it buffers the full payload and
    ``.decode()`` doubles it; loading the result into a DataFrame multiplies
    it again. Filter early, keep only needed columns, write the reduced
    subset to a file in the working directory and analyze that instead.

    Args:
        code: Python source code to execute. Must self-terminate.
        timeout: Maximum execution time in seconds. 0 (default) uses the
            server-configured default (run_python_timeout_s). Hard-capped at
            run_python_max_timeout_s (300s) — a larger request is clamped, so
            a single exec can't pin a scarce sandbox slot for many minutes.

    Returns:
        stdout + stderr from the execution, or an error message.
    """
    if not code or not code.strip():
        return "Error: empty code provided."

    if timeout <= 0:
        timeout = _default_timeout()
    # Clamp an agent-supplied timeout to the hard ceiling regardless of what
    # the model asked for (it can ask for less, never more). Stops a runaway
    # data-collection exec from burning many minutes on one sandbox slot.
    max_timeout = _max_timeout()
    if timeout > max_timeout:
        timeout = max_timeout

    # Normalise Unicode math symbols copied from problem statements
    # (``∫ Σ π ≤`` → ``integral sum pi <=``) before executing — LLMs
    # frequently echo these and Python rejects them with SyntaxError.
    code = sanitize_code(code)

    blocked = _blocked_ml_download_reason(code)
    if blocked:
        return blocked

    net_libs = sorted(set(_NET_LIB_RE.findall(code)))
    if net_libs:
        logger.info(
            "run_python_code: raw HTTP libs in agent code: %s "
            "(timeout=%ss, code_len=%d)",
            ",".join(net_libs), timeout, len(code),
        )

    try:
        sandbox = await aget_sandbox()
    except RuntimeError as exc:
        return f"Error: sandbox unavailable — {exc}"

    filename = f"/tmp/exec_{uuid.uuid4().hex[:8]}.py"
    # E2B / Docker ``commands.run(timeout=...)`` only times out the SDK client
    # wait — it does NOT kill the in-container process, so a brute-force
    # enumeration keeps burning a scarce sandbox slot well past ``timeout``
    # (issue #221: observed up to 600s on a 120s budget). Wrap remote execs in
    # a coreutils ``timeout -s KILL`` for an OS-level hard kill that even a
    # numpy/C loop can't ignore.
    # Socket-level download cap (sitecustomize injection): bounds how many
    # bytes any python process in this exec tree — including pip children —
    # can receive per connection, whether buffered or streamed to disk. See
    # ``plugins/tools/_net_guard.py`` (smoke-memwt-003: GB-scale dataset
    # downloads inside the sandbox).
    await ensure_guard_file(sandbox)
    net_guard_env = guard_env_prefix()
    base_exec_cmd = f"python3 {filename}"
    # Add the per-exec memory cap + single-thread math-lib env so a buffered
    # parse fails inside the sandbox instead of OOM-killing its environment.
    exec_cmd = (
        f"{remote_exec_prefix()}{net_guard_env}{_OFFLINE_DOWNLOAD_ENV}"
        f"timeout -s KILL {timeout}s {base_exec_cmd}"
    )
    cmd_timeout = timeout + 30  # let the inner OS timeout fire first
    try:
        if hasattr(sandbox, "files"):
            await asyncio.to_thread(sandbox.files.write, filename, code)
        else:
            await asyncio.to_thread(
                Path(filename).write_text, code, encoding="utf-8",
            )

        # Count one E2B execution (lifetime is metered at the
        # sandbox create/kill sites in ``_sandbox.py``). Bwrap/Current/Docker
        # facades bill nobody — they all carry a ``sandbox_id``
        # too, so discriminate by implementing module instead.
        if is_e2b_sandbox(sandbox):
            record_api_request("e2b")
        result = await arun_sandbox_cmd(
            sandbox, exec_cmd, timeout=cmd_timeout,
            # Match ``bash``: both run model-authored code, so denying the
            # network here while allowing it there only means the same snippet
            # succeeds under ``bash -c 'python3 …'`` and fails via this tool —
            # an asymmetry with no security value and a hard-to-place error.
            # The bound is the socket cap injected above, not the namespace;
            # on E2B/CurrentSandbox this path already had network anyway, so
            # only the bwrap backend changes.
            allow_net=True,
        )
    except TimeoutError:
        return f"Error: execution timed out after {timeout}s."
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"

    # coreutils ``timeout`` exit codes: 124 = TERM expired, 137 = 128+SIGKILL.
    # Surface a consistent timeout message so the LLM
    # gets a consistent signal instead of a bare "[exit code 137]". Append the
    # tail of whatever stdout the process streamed before the kill — a batched
    # crawl that printed progress / checkpointed partial results can be resumed
    # or narrowed instead of being a total loss.
    if result.exit_code in (124, 137):
        msg = f"Error: execution timed out after {timeout}s."
        partial = (result.stdout or "").strip()
        if partial:
            msg += (
                "\nPartial stdout before the kill (salvage it: resume from the "
                "last checkpoint, or narrow the query / use an aggregate "
                f"endpoint instead of re-running as-is):\n{partial[-_PARTIAL_TAIL:]}"
            )
        return msg

    # E2B reports a process that died without a normal exit status (OOM-killed,
    # envd-side failure) as exit code -1, usually with an empty stderr
    # (observed: swarm_gv 2026-06-03, repeated -1 bursts on one sandbox). Give
    # the model an actionable signal instead of a bare "[exit code -1]".
    if result.exit_code == -1:
        detail = (result.stderr or result.stdout or "").strip()
        suffix = f"\n{detail}" if detail else ""
        return (
            "Error: sandbox process died unexpectedly (exit code -1, likely "
            "out-of-memory or a sandbox-side failure). The sandbox has very "
            "limited RAM — do NOT retry the same code. To process a large "
            "file/dataset, stream it end-to-end instead of buffering it:\n"
            f"{_MEM_RECIPE}{suffix}"
        )

    output = result.stdout or ""
    if result.stderr:
        output += f"{BASH_STDERR_SEPARATOR}{result.stderr}"
    if result.exit_code != 0:
        output = f"[exit code {result.exit_code}]\n{output}"
        # The per-exec ``ulimit -v`` cap converts a would-be VM OOM kill into
        # a clean MemoryError traceback — steer the retry toward streaming
        # instead of letting the model shrink the workload and re-buffer.
        if "MemoryError" in (result.stderr or ""):
            output += (
                "\n[hint] The process hit the per-exec memory cap. Do NOT "
                "retry the same approach with a smaller slice — stream "
                f"end-to-end instead of buffering:\n{_MEM_RECIPE}"
            )

    return maybe_overflow("run_python_code", output)
