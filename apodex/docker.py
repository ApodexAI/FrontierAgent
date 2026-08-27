"""Run the CLI inside a container — the supported macOS path.

macOS has no bubblewrap, so instead of inventing a second isolation mechanism
the whole CLI is re-executed inside the repo's own image. The container is the
boundary; inside it :mod:`apodex.sandbox` resolves to ``container`` and runs
commands directly, exactly as the production benchmark path does.

What crosses the boundary, and nothing else:

- your working directory, read-write at ``/workspace``;
- a dedicated ``.apodex/runs/<session-id>/outputs`` directory, read-write at
  ``/outputs``;
- ``~/.apodex`` (session history, traces), so ``--resume`` works across runs;
- ``.env`` from the repo, for model and search credentials.

The image is built on first use and reused after that. It is the same
``Dockerfile`` the benchmark runner uses, so there is one image to maintain.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

_DEFAULT_IMAGE = "apodex:local"
IMAGE = os.environ.get("APODEX_IMAGE", _DEFAULT_IMAGE)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def terminal_env(environ: Mapping[str, str]) -> list[str]:
    """Return ``-e`` args carrying the host terminal contract inward.

    Docker sets ``TERM=xterm`` inside a ``-it`` container and forwards no
    ``COLORTERM`` at all. Rich and Textual read exactly those two variables, and
    a bare ``xterm`` means *8 colours* — at which point every theme is destroyed
    rather than merely approximated: gruvbox's ``#ebdbb2`` quantises to pure
    white, its muted tan to ``#aaaaaa`` grey, and its orange and red both to
    ``#ff5555``, so two different states become the same colour.

    The container's ``TERM`` is a Docker default, not a measurement — the host
    terminal is what actually paints the pixels. So forward the host's values,
    and where the host says nothing, assume the 256-colour floor every terminal
    emulator capable of running this UI has cleared for years. Anything that
    genuinely cannot do colour still has ``--no-color`` / ``NO_COLOR`` /
    ``--theme mono``, which are checked before we get here.
    """
    term = environ.get("TERM", "").strip()
    color_term = environ.get("COLORTERM", "").strip()
    if term.lower() == "dumb":  # honour it rather than upgrading it away
        return ["-e", "TERM=dumb"]
    args = ["-e", f"TERM={term or 'xterm-256color'}"]
    if color_term:
        args += ["-e", f"COLORTERM={color_term}"]
    elif not term.endswith(("-256color", "-truecolor", "-direct")):
        args += ["-e", "COLORTERM=truecolor"]
    # The default macOS launch re-executes inside Docker.  Keep terminal
    # identity available for compatibility decisions there, and preserve the
    # Textual keyboard override chosen by the host CLI or supplied by the user.
    for name in (
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "TEXTUAL_DISABLE_KITTY_KEY",
    ):
        if name in environ:
            args += ["-e", f"{name}={environ[name]}"]
    return args


def _host_identity_env() -> list[str]:
    """``-e`` args naming the invoking host user, when there is one to name.

    Empty on platforms without POSIX uids: the entrypoint requires numeric
    values and ignores anything else, so sending nothing is the same answer
    stated honestly.
    """
    if not hasattr(os, "getuid"):
        return []
    return [
        "-e", f"APODEX_HOST_UID={os.getuid()}",
        "-e", f"APODEX_HOST_GID={os.getgid()}",
    ]


def _without_cwd_arg(argv: list[str]) -> list[str]:
    """Drop host-only path flags before entering the container."""
    result: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
        elif arg in {"--cwd", "--input"}:
            skip_next = True
        elif not arg.startswith(("--cwd=", "--input=")):
            result.append(arg)
    return result


def docker_available() -> tuple[bool, str]:
    """Whether a usable docker CLI and daemon are present."""
    if shutil.which("docker") is None:
        return False, "the `docker` command is not on PATH"
    probe = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=60,
    )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip().splitlines()
        hint = detail[-1] if detail else "docker info failed"
        return False, f"the docker daemon is not reachable ({hint})"
    return True, "docker available"


def image_exists(image: str = IMAGE) -> bool:
    probe = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, text=True,
    )
    return probe.returncode == 0


def build_image(image: str = IMAGE, *, quiet: bool = False) -> None:
    """Build the image from the repo Dockerfile.

    Streams the build output: it takes minutes the first time (LibreOffice and
    the document readers are large), and a silent multi-minute wait reads as a
    hang.
    """
    print(f"apodex: building {image} (first run only, this takes a few minutes)…",
          file=sys.stderr)
    cmd = ["docker", "build", "-t", image, str(_REPO_ROOT)]
    if quiet:
        cmd.insert(2, "--quiet")
    subprocess.run(cmd, check=True)


def pull_image(image: str) -> bool:
    """Fetch *image* from its registry, returning whether it arrived.

    Streams the pull for the same reason ``build_image`` streams the build: a
    multi-gigabyte transfer with no output reads as a hang.
    """
    print(f"apodex: pulling {image} (first run only)…", file=sys.stderr)
    return subprocess.run(["docker", "pull", image]).returncode == 0


def run_in_container(
    argv: list[str], *, cwd: str | None = None, image: str = IMAGE,
) -> int:
    """Re-exec ``apodex argv`` inside the container. Returns its exit code."""
    ok, why = docker_available()
    if not ok:
        print(
            f"apodex: cannot use the Docker path — {why}.\n"
            "        Install Docker and start it, or rerun with --native "
            "to use the workspace-local host runtime.",
            file=sys.stderr,
        )
        return 1

    if not image_exists(image):
        # Only the built-in tag names this checkout's Dockerfile. Building
        # something else under an explicitly requested tag would hand the user
        # an image that is not the one they asked for, while looking like it is.
        if image == _DEFAULT_IMAGE:
            try:
                build_image(image)
            except subprocess.CalledProcessError as exc:
                print(f"apodex: image build failed (exit {exc.returncode}).",
                      file=sys.stderr)
                return exc.returncode or 1
        elif not pull_image(image):
            print(
                f"apodex: cannot use {image} — it is not present locally and "
                "the pull failed.\n"
                "        Pull or build that tag yourself, or unset "
                f"APODEX_IMAGE to build {_DEFAULT_IMAGE} from this checkout.",
                file=sys.stderr,
            )
            return 1

    workspace = Path(cwd or os.getcwd()).expanduser().resolve()
    session_id = _session_id_for_run(argv)
    home_state = Path.home() / ".apodex"
    home_state.mkdir(parents=True, exist_ok=True)
    home_config = Path.home() / ".config" / "apodex"
    home_config.mkdir(parents=True, exist_ok=True)
    runs_root = workspace / ".apodex" / "runs"
    run_workspace = runs_root / session_id / "workspace"
    outputs = runs_root / session_id / "outputs"
    run_workspace.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    # Deliberately outside both the writable workspace and the separately
    # mounted ~/.apodex state tree, so /inputs:ro has no writable alias.
    inputs = Path.home() / ".apodex-inputs" / session_id
    inputs.mkdir(parents=True, exist_ok=True)
    from apodex.attachments import AttachmentError, AttachmentManager
    old_staging = os.environ.get("APODEX_INPUT_STAGING_DIR")
    old_agent_dir = os.environ.get("FRONTIER_AGENT_INPUTS_DIR")
    os.environ["APODEX_INPUT_STAGING_DIR"] = str(inputs)
    os.environ["FRONTIER_AGENT_INPUTS_DIR"] = str(inputs)
    try:
        manager = AttachmentManager(str(workspace), session_id)
        requested_inputs: list[str] = []
        skip_next = False
        for index, arg in enumerate(argv):
            if skip_next:
                skip_next = False
                continue
            if arg == "--input" and index + 1 < len(argv):
                requested_inputs.append(argv[index + 1])
                skip_next = True
            elif arg.startswith("--input="):
                requested_inputs.append(arg.split("=", 1)[1])
        manager.attach_many(requested_inputs)
    except AttachmentError as exc:
        print(f"apodex: cannot attach input — {exc}", file=sys.stderr)
        return 2
    finally:
        if old_staging is None:
            os.environ.pop("APODEX_INPUT_STAGING_DIR", None)
        else:
            os.environ["APODEX_INPUT_STAGING_DIR"] = old_staging
        if old_agent_dir is None:
            os.environ.pop("FRONTIER_AGENT_INPUTS_DIR", None)
        else:
            os.environ["FRONTIER_AGENT_INPUTS_DIR"] = old_agent_dir
    clipboard_broker = None
    clipboard_env: list[str] = []
    if sys.platform == "darwin" and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from apodex.clipboard import ClipboardBroker

            clipboard_broker = ClipboardBroker(manager)
            clipboard_broker.start()
            clipboard_env = [
                "-e", f"APODEX_CLIPBOARD_BROKER_URL=http://host.docker.internal:{clipboard_broker.port}",
                "-e", f"APODEX_CLIPBOARD_BROKER_TOKEN={clipboard_broker.token}",
            ]
        except Exception as exc:
            print(
                f"warning: macOS clipboard bridge is unavailable ({exc})",
                file=sys.stderr,
            )
    docker_cmd = [
        "docker", "run", "--rm",
        # Enter through the image entrypoint with ``apodex`` as its command,
        # rather than replacing it with ``apodex``. The entrypoint is what
        # remaps the unprivileged tool account onto APODEX_HOST_UID/GID below
        # and exports APODEX_TOOL_HOST_IDENTITY; without it the tool layer
        # cannot hand the mounts to a host identity and falls back to widening
        # their mode to 0777 (plugins/tools/_sandbox.py), which on a Linux bind
        # mount lands on the user's own checkout. It still cannot be omitted
        # entirely: a bare task string ("fix the crash") does not start with
        # ``-``, and the entrypoint would exec it as a command.
        "--entrypoint", "/app/docker/entrypoint.sh",
        # A TTY only when we have one: without this an `apodex -p "…"` in a
        # pipeline fails on "the input device is not a TTY".
        *(["-it"] if sys.stdin.isatty() and sys.stdout.isatty() else []),
        # Keep the user's project distinct from run-private scratch. The
        # session creates /workspace as a stable symlink into
        # /apodex-runs/<session-id>/workspace before any tool can run.
        "-v", f"{workspace}:/project",
        # Mount the whole session-output namespace. TerminalSession maintains
        # /outputs as a symlink to the active session, so an in-app /resume can
        # switch directories without restarting or remounting the container.
        "-v", f"{runs_root}:/apodex-runs",
        "-v", f"{inputs}:/apodex-input-staging",
        "-v", f"{inputs}:/inputs:ro",
        "-w", "/project",
        "-v", f"{home_state}:/root/.apodex",
        "-v", f"{home_config}:/root/.config/apodex",
        "-e", "APODEX_IN_CONTAINER=1",
        "-e", "HOME=/root",
        # Same contract as the Compose launchers: files written by the dropped-
        # privilege tool process keep useful host ownership. The entrypoint
        # validates and rejects a root identity, so this is inert when Docker is
        # driven by root and the 0777 fallback still applies there.
        *_host_identity_env(),
        # Colour capability travels with the host terminal, not the container.
        *terminal_env(os.environ),
        # Native Stateful ReAct / Agent Team workflows use the shared sandbox
        # backend. This outer container is already the boundary, so do not try
        # to nest bubblewrap (Docker Desktop blocks its user namespaces by
        # default).
        "-e", "SANDBOX_BACKEND=container",
        "-e", f"APODEX_LOCAL_UTC_OFFSET={dt.datetime.now().astimezone():%z}",
        "-e", "FRONTIER_AGENT_WORKSPACE_DIR=/workspace",
        "-e", "APODEX_SESSION_WORKSPACES_ROOT=/apodex-runs",
        "-e", "APODEX_WORKSPACE_LINK=/workspace",
        "-e", "FRONTIER_AGENT_OUTPUTS_DIR=/outputs",
        "-e", "APODEX_RUNS_ROOT=/apodex-runs",
        "-e", "APODEX_RUNS_ROOT_PINNED=1",
        # The *host* path, not the mount point: this is what the TUI and the
        # follow-up prompts show the user, so it has to exist on their machine.
        "-e", f"APODEX_HOST_RUNS_ROOT={runs_root}",
        "-e", f"APODEX_HOST_WORKSPACE_ROOT={runs_root}",
        "-e", f"APODEX_HOST_WORKSPACE_DIR={run_workspace}",
        "-e", "APODEX_SESSION_OUTPUTS_ROOT=/apodex-runs",
        "-e", "APODEX_OUTPUTS_LINK=/outputs",
        "-e", "FRONTIER_AGENT_INPUTS_DIR=/inputs",
        "-e", "APODEX_INPUT_STAGING_DIR=/apodex-input-staging",
        "-e", f"APODEX_SESSION_ID={session_id}",
        "-e", f"APODEX_HOST_OUTPUTS_DIR={outputs}",
        "-e", f"APODEX_HOST_OUTPUTS_ROOT={runs_root}",
        "-e", f"APODEX_HOST_INPUTS_DIR={inputs}",
        *clipboard_env,
    ]
    env_file = _REPO_ROOT / ".env"
    if env_file.is_file():
        docker_cmd += ["--env-file", str(env_file)]
    docker_cmd += [image, "apodex", *_without_cwd_arg(argv)]

    try:
        proc = subprocess.run(docker_cmd)
        return proc.returncode
    finally:
        if clipboard_broker is not None:
            clipboard_broker.close()


def _session_id_for_run(argv: list[str]) -> str:
    """Choose the session/output directory name before Docker starts."""
    from apodex.session import new_session_id

    mode = "react"
    resume_id = ""
    for index, arg in enumerate(argv):
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1] or mode
        elif arg == "--mode" and index + 1 < len(argv):
            mode = argv[index + 1]
        elif arg.startswith("--resume="):
            resume_id = arg.split("=", 1)[1]
        elif arg == "--resume" and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            resume_id = argv[index + 1]
    raw = resume_id or new_session_id(mode)
    safe = re.sub(r"[^A-Za-z0-9._+-]+", "-", raw).strip(".-")
    return safe or new_session_id(mode)
