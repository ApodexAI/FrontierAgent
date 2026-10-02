"""The documented container quick start must work without registry credentials.

`ghcr.io/apodexai/frontieragent` is private by org policy — see the comment in
`.github/workflows/docker-publish.yml` and the note in
`docs/install/global-install.md`. An anonymous pull of it fails, so the
user-facing quick start cannot promise a build-free `docker compose run`.

`README.md`, `docs/install/docker.md` and the chooser table in
`docs/install/README.md` all did promise exactly that, which sent anyone
outside the org to an `unauthorized` error on their first command.

These assertions fail on the pre-fix tree and pass after it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Every file that walks a reader through starting the container.
_QUICKSTART_DOCS = (
    "README.md",
    "docs/install/docker.md",
    "docs/install/README.md",
)

# Claims that cannot hold while the published image is private.
_UNREACHABLE_CLAIMS = (
    "no local build needed",
    "does not build the repository locally",
    "Pull and launch the pre-built container",
)


@pytest.mark.parametrize("rel_path", _QUICKSTART_DOCS)
def test_quickstart_docs_say_the_image_is_private(rel_path: str) -> None:
    """Each quick start has to acknowledge the private package."""
    text = (_REPO_ROOT / rel_path).read_text(encoding="utf-8")
    assert re.search(r"private", text, re.IGNORECASE), (
        f"{rel_path} documents a container quick start but never says the "
        "ghcr.io/apodexai/frontieragent package is private, so a reader "
        "outside the org hits `unauthorized` on the first command"
    )


@pytest.mark.parametrize("rel_path", _QUICKSTART_DOCS)
def test_quickstart_docs_drop_unreachable_claims(rel_path: str) -> None:
    """The pre-fix wording promised a pull that cannot succeed anonymously."""
    lowered = (_REPO_ROOT / rel_path).read_text(encoding="utf-8").lower()
    for claim in _UNREACHABLE_CLAIMS:
        assert claim not in lowered, (
            f"{rel_path} still claims {claim!r}, which is false while the "
            "published image is private"
        )


def _run_commands(text: str) -> list[str]:
    """Every `docker compose ... run ...` invocation, joined across line wraps.

    Two things make a naive per-line scan miss these: the compose file flags
    sit between `compose` and `run`, and long invocations wrap onto the next
    line with a trailing backslash.
    """
    joined = text.replace("\\\n", " ")
    return [
        line.strip()
        for line in joined.splitlines()
        if re.match(r"\s*docker compose\b.*\brun\b", line)
    ]


def test_docker_quickstart_offers_a_path_that_needs_no_registry() -> None:
    """The local-build escape hatch has to be spelled out.

    `compose.yaml` sets `pull_policy: always`, so the documented build path is
    only usable if the commands keep the `compose.dev.yaml` override — its
    `pull_policy: build` is what keeps the local image instead of retrying the
    private registry.
    """
    text = (_REPO_ROOT / "docs/install/docker.md").read_text(encoding="utf-8")
    assert "compose.dev.yaml" in text, (
        "docs/install/docker.md offers no local-build path, but the published "
        "image cannot be pulled anonymously"
    )

    build_blocks = re.findall(r"```bash\n(.*?)```", text, re.DOTALL)
    offenders = [
        command
        for block in build_blocks
        if "compose.dev.yaml build" in block
        for command in _run_commands(block)
        if "compose.dev.yaml" not in command and "docker login" not in block
    ]
    assert not offenders, (
        "a block that builds locally then runs without compose.dev.yaml, so "
        "compose.yaml's pull_policy: always re-fetches the private image: " + "; ".join(offenders)
    )


def test_documented_local_tag_opts_out_of_the_registry() -> None:
    """A locally built tag has to say `--pull never` to be usable.

    `compose.yaml` sets `pull_policy: always`, so `FRONTIER_AGENT_IMAGE` pointed at a
    tag that exists only on this machine still makes Compose resolve it against a
    registry and fail.
    """
    text = (_REPO_ROOT / "docs/install/docker.md").read_text(encoding="utf-8")
    # The local-tag example is prefixed with FRONTIER_AGENT_IMAGE=..., which `_run_commands`
    # only matches when `docker compose` starts the line.
    joined = text.replace("\\\n", " ")
    local_runs = [
        line.strip()
        for line in joined.splitlines()
        if re.match(r"\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)?docker compose\b.*\brun\b", line)
        and "frontier-agent:local" in line
    ]
    assert local_runs, (
        "docs/install/docker.md documents a locally built tag but never runs it "
        "through Compose, so there is no local-only example to check"
    )
    offenders = [command for command in local_runs if "--pull never" not in command]
    assert not offenders, (
        "a locally built tag without --pull never is re-fetched from the registry "
        "by compose.yaml's pull_policy: always: " + "; ".join(offenders)
    )


def test_readme_quickstart_offers_a_path_that_needs_no_registry() -> None:
    """The README snippet is the most-read entry point for the container path."""
    text = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("## Containers and local models", 1)[1].split("\n## ", 1)[0]
    runs = _run_commands(section)
    assert runs, "README no longer shows a container run command"
    assert all("compose.dev.yaml" in command for command in runs), (
        "README runs the container without the compose.dev.yaml override, so "
        "the private image is pulled: " + "; ".join(runs)
    )
