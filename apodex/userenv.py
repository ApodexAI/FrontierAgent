"""Environment resolution for the CLI, including the optional user env file.

A checkout keeps its endpoint in ``<checkout>/.env``. An installation that
lives outside any checkout (``uv tool install``) has no such file to fall back
on, so credentials may also come from one user-level file:

    ``$XDG_CONFIG_HOME/apodex/env``  (default ``~/.config/apodex/env``)

Precedence, highest first:

1. explicit CLI options (``--model``, ``--max-tokens`` …);
2. the exported environment;
3. ``.env`` in the launch directory or the nearest ancestor that has one
   (the pre-existing behaviour, unchanged);
4. the user env file.

The file is plain ``KEY=value`` lines. It is read literally: no ``${VAR}``
interpolation, so a value can never resolve against whatever happens to be
exported. Blank values are ignored. Nothing here prints or logs a value —
every note names variables and files only.

Credential and endpoint pairs (``<PREFIX>_API_KEY`` / ``<PREFIX>_BASE_URL``)
defined together in the user file are applied together. If a higher source
already fixes one half to a different value, the other half is withheld and
reported: a key that was written down next to one endpoint must not be sent to
another one just because a project ``.env`` overrode the URL.
"""

from __future__ import annotations

import contextlib
import os
import stat
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

USER_ENV_FILE_VAR = "APODEX_ENV_FILE"
_API_KEY_SUFFIX = "_API_KEY"
_BASE_URL_SUFFIX = "_BASE_URL"

# Names the Docker launcher forwards from the resolved host environment when
# they are set, in addition to whatever the loaded env files define. These are
# the runtime variables ``.env.example`` documents plus the ones the shipped
# profiles interpolate, so an exported value reaches the container the same
# way it reaches a native run.
FORWARDED_RUNTIME_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "OPENAI_PROVIDER",
    "OPENAI_MAX_TOKENS",
    "OPENAI_CONTEXT_WINDOW",
    "APODEX_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "BEDROCK_API_KEY",
    "BEDROCK_BASE_URL",
    "JUDGE_API_KEY",
    "JUDGE_BASE_URL",
    "JUDGE_MODEL",
    "HF_TOKEN",
    "OFFICEQA_DOC_MODE",
    "FRONTIER_AGENT_DATASETS_DIR",
    "SERPER_API_KEY",
    "SERPER_BASE_URL",
    "JINA_API_KEY",
    "JINA_BASE_URL",
    "SUMMARY_LLM_API_KEY",
    "SUMMARY_LLM_BASE_URL",
    "SUMMARY_LLM_MODEL_NAME",
    "READDOC_VISION_URL",
    "READDOC_VISION_MODEL",
    "READDOC_VISION_KEY",
    "READDOC_OCR_URL",
    "READDOC_OCR_KEY",
    "REACT_NO_WEB",
    "SWARM_NO_WEB",
)


@dataclass(frozen=True)
class EnvResolution:
    """What the CLI's environment load did, without any of the values."""

    #: The user env file that was read, or ``None`` when there is none.
    user_env_path: Path | None
    #: ``.env`` files loaded from the launch directory or its ancestors.
    dotenv_paths: tuple[Path, ...]
    #: Variables the user env file supplied (they were not set before).
    applied: tuple[str, ...]
    #: Variables the user env file defined but the pair guard withheld.
    withheld: tuple[str, ...]
    #: Human-facing, secret-free notes for stderr / the TUI transcript.
    notes: tuple[str, ...]
    #: Every variable any loaded file defined, for the Docker launcher.
    file_names: tuple[str, ...]

    @classmethod
    def empty(cls) -> EnvResolution:
        """The resolution of a run that loaded nothing (tests stub this in)."""
        return cls(None, (), (), (), (), ())

    def forwarded_names(self, environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """Names worth carrying into a container, restricted to what is set.

        Only names, never values: the launcher passes them as ``-e NAME`` so
        Docker reads each value from the process environment rather than
        from a command line that ``ps`` could show.
        """
        env = os.environ if environ is None else environ
        candidates = dict.fromkeys((*self.file_names, *FORWARDED_RUNTIME_VARS))
        return tuple(name for name in candidates if name in env)


def user_env_path(environ: Mapping[str, str] | None = None) -> Path:
    """Where the user env file lives for this process.

    ``APODEX_ENV_FILE`` names it outright. Otherwise it is the ``apodex``
    directory under ``XDG_CONFIG_HOME``, falling back to ``~/.config``, which
    is the same namespace the CLI already uses for ``settings.json``.
    """
    env = os.environ if environ is None else environ
    override = (env.get(USER_ENV_FILE_VAR) or "").strip()
    if override:
        return Path(override).expanduser()
    xdg = (env.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        base = Path(xdg).expanduser()
    else:
        home = (env.get("HOME") or "").strip()
        base = (Path(home).expanduser() if home else Path.home()) / ".config"
    return base / "apodex" / "env"


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse ``path`` literally: no interpolation, blank values dropped."""
    from dotenv import dotenv_values

    values: dict[str, str] = {}
    for name, value in dotenv_values(path, interpolate=False).items():
        if not name or value is None:
            continue
        stripped = value.strip()
        if stripped:
            values[str(name)] = stripped
    return values


def _same_endpoint(a: str, b: str) -> bool:
    return a.strip().rstrip("/") == b.strip().rstrip("/")


def _pairs(names: set[str]) -> list[tuple[str, str]]:
    """``(<P>_API_KEY, <P>_BASE_URL)`` for every prefix that has both."""
    out: list[tuple[str, str]] = []
    for name in sorted(names):
        if not name.endswith(_API_KEY_SUFFIX):
            continue
        prefix = name[: -len(_API_KEY_SUFFIX)]
        url_name = f"{prefix}{_BASE_URL_SUFFIX}"
        if url_name in names:
            out.append((name, url_name))
    return out


def _permission_note(path: Path) -> str | None:
    """A warning when other users could read the file; POSIX only."""
    try:
        mode = path.stat().st_mode
    except OSError:
        return None
    if os.name != "posix":
        return None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return (
            f"{path} is readable by other users on this machine; restrict it with: chmod 600 {path}"
        )
    return None


def apply_user_env(
    environ: MutableMapping[str, str],
    *,
    path: Path | None = None,
) -> tuple[Path | None, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Layer the user env file under ``environ`` (never over it).

    Returns ``(path_or_None, applied, withheld, notes, defined_names)``. A
    missing file is not an error: it is simply the checkout-only setup.
    """
    target = path if path is not None else user_env_path(environ)
    if not target.is_file():
        return None, (), (), (), ()

    try:
        values = _read_env_file(target)
    except (OSError, UnicodeDecodeError) as exc:
        # Ignored rather than fatal, but said out loud: a run that silently
        # fell back to "no credentials" would fail one step later with a
        # message that never mentions the file.
        return (
            target,
            (),
            (),
            (f"could not read {target} ({exc.__class__.__name__}: {exc}); ignoring it",),
            (),
        )

    notes: list[str] = []
    withheld: set[str] = set()
    for key_name, url_name in _pairs(set(values)):
        key_fixed = key_name in environ
        url_fixed = url_name in environ
        if key_fixed and url_fixed:
            continue  # the pair is fully decided elsewhere; nothing to apply
        if url_fixed and not _same_endpoint(environ[url_name], values[url_name]):
            withheld.add(key_name)
            notes.append(
                f"{key_name} from {target} was not applied: {url_name} is set "
                "to a different endpoint by the environment or a .env file. "
                f"Set {key_name} alongside that {url_name}, or remove the "
                "override, so a key is only sent to the endpoint it was "
                "written next to."
            )
        elif key_fixed and environ[key_name] != values[key_name]:
            withheld.add(url_name)
            notes.append(
                f"{url_name} from {target} was not applied: {key_name} is "
                "already set to a different value by the environment or a "
                f".env file. Set {url_name} alongside that {key_name} if the "
                "two belong together."
            )

    applied: list[str] = []
    for name, value in values.items():
        if name in withheld or name in environ:
            continue
        environ[name] = value
        applied.append(name)

    permission = _permission_note(target)
    if permission:
        notes.append(permission)
    return (
        target,
        tuple(applied),
        tuple(sorted(withheld)),
        tuple(notes),
        tuple(values),
    )


def load_environment() -> EnvResolution:
    """Resolve ``.env`` files and the user env file into ``os.environ``.

    Deliberately bound to the real process: ``load_dotenv`` only writes to
    ``os.environ`` and ``find_dotenv`` only walks up from the process cwd, so
    accepting a mapping or a directory here would promise an isolation the
    call cannot keep. Tests set the environment and ``chdir`` instead.

    Runs before any ``--cwd`` chdir and before native mode rewrites ``HOME``
    and the XDG directories, so both the launch directory's ``.env`` and the
    user's real config directory are what get read. ``override=False``
    throughout: an exported variable always wins over every file.
    """
    dotenv_paths: list[Path] = []
    file_names: dict[str, None] = {}
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        load_dotenv = None  # type: ignore[assignment]
        find_dotenv = None  # type: ignore[assignment]

    if load_dotenv is not None and find_dotenv is not None:
        local = Path.cwd() / ".env"
        candidates: list[Path] = []
        if local.is_file():
            candidates.append(local)
        found = find_dotenv(usecwd=True)
        if found:
            found_path = Path(found)
            if found_path.resolve() not in {c.resolve() for c in candidates}:
                candidates.append(found_path)
        for candidate in candidates:
            # Same call the CLI has always made: dotenv semantics, including
            # interpolation, are unchanged for a project's own .env.
            load_dotenv(candidate, override=False)
            dotenv_paths.append(candidate)
            # Names only, for the Docker launcher. ``load_dotenv`` already
            # tolerated an unparsable line; recording the names must not be
            # stricter than the load itself was.
            with contextlib.suppress(OSError, UnicodeDecodeError):
                file_names.update(dict.fromkeys(_read_env_file(candidate)))

    path, applied, withheld, notes, defined = apply_user_env(os.environ)
    file_names.update(dict.fromkeys(defined))
    return EnvResolution(
        user_env_path=path,
        dotenv_paths=tuple(dotenv_paths),
        applied=applied,
        withheld=withheld,
        notes=notes,
        file_names=tuple(file_names),
    )


__all__ = [
    "FORWARDED_RUNTIME_VARS",
    "USER_ENV_FILE_VAR",
    "EnvResolution",
    "apply_user_env",
    "load_environment",
    "user_env_path",
]
