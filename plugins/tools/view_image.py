"""Image analysis tool using Vision API."""

from __future__ import annotations

import logging
import shlex
import unicodedata
from pathlib import Path

import httpx

from frontier_agent.core.tool import tool
from frontier_agent.infra.config import get_config

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


async def _call_vision_api(
    image_content: list[dict],
    question: str,
) -> str:
    """Call an OpenAI-compatible vision API.

    Uses the `vision_*` config section (falls back to `openai_*` when
    vision-specific values are empty). This lets deployments target a
    cheaper multimodal model (e.g. gemini-flash) for OCR/image QA while
    keeping the primary chat LLM configured independently.
    """
    config = get_config()

    messages = [
        {
            "role": "user",
            "content": [
                *image_content,
                {"type": "text", "text": question},
            ],
        }
    ]

    headers = {
        "Authorization": f"Bearer {config.effective_vision_api_key}",
        "Content-Type": "application/json",
    }

    model = config.effective_vision_model
    base_url = config.effective_vision_base_url

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.2,
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                if resp.status_code == 429:
                    import asyncio
                    wait = 2 ** attempt
                    logger.warning("Vision API 429, retrying in %ds", wait)
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            logger.warning("Vision API timeout (attempt %d)", attempt + 1)
            if attempt < 2:
                continue
            return "Error: Vision API timed out."
        except Exception as e:
            logger.error("Vision API error: %s", e)
            return f"Error: {e}"

    return "Error: Vision API failed after 3 attempts."


def _rejected_image_path(file_path: str) -> str | None:
    """Why *file_path* cannot be used, or ``None`` if it is fine.

    ``shlex.quote`` in the sandbox branch is what makes the path safe as a shell
    word, and it is complete for that job: spaces, quotes, brackets, ``;``,
    backticks and ``$`` all survive quoting harmlessly. So ordinary punctuation
    is NOT rejected here — ``photo (1).png``, ``John's scan.jpg`` and
    ``chart[final].webp`` are all normal upload names, and the inventory
    advertises exactly such paths for this tool.

    What quoting cannot repair is a character that ends the argument itself, so
    control characters and line separators are still refused. The extension
    check does the remaining narrowing: ``x; id`` carries no supported suffix and
    never reaches the command at all.
    """
    if _unsafe_shell_word(file_path):
        return (
            "Error: image path contains control or line-separator characters, "
            "which is never a real filename here."
        )
    if Path(file_path).suffix.lower() not in _SUPPORTED_EXTENSIONS:
        return (
            f"Error: unsupported image type {Path(file_path).suffix!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )
    return None


def _unsafe_shell_word(text: str) -> bool:
    """Control characters and line separators, by Unicode category.

    Not an ASCII range check: U+0085 sits in the C1 block that ``ord(ch) < 0x20``
    misses, and U+2028 / U+2029 are separators outside it entirely.
    """
    return any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in text)


@tool
async def view_image(
    image_path_or_url: str,
    question: str = "Describe this image in detail. Extract any text visible in the image.",
) -> str:
    """Analyze an image using vision AI. Supports local files and URLs.

    Performs OCR text extraction and visual question answering.

    Args:
        image_path_or_url: Local file path or URL to the image.
        question: What to analyze in the image (default: describe + OCR).

    Returns:
        Detailed analysis of the image content.
    """
    if not image_path_or_url or not image_path_or_url.strip():
        return "Error: image path or URL is required."

    # Determine if it's a URL or local file
    if image_path_or_url.startswith(("http://", "https://")):
        image_content = [
            {
                "type": "image_url",
                "image_url": {"url": image_path_or_url},
            }
        ]
    else:
        # Validated before it reaches the sandbox command: that command is a
        # string, and the roles holding this tool have no shell of their own, so
        # an unchecked path here would be their one route to arbitrary commands
        # in the task container.
        rejected = _rejected_image_path(image_path_or_url)
        if rejected:
            return rejected

        # ALWAYS read through the sandbox — never from the harness process.
        #
        # There used to be a host fast path that authorized the resolved target
        # and then read the original pathname. Those are two separate
        # resolutions, so an atomically swapped symlink between them turned a
        # model-supplied path into a root-read primitive whose bytes were then
        # shipped to an external Vision API. Binding the check to the read (open
        # once, authorize /proc/self/fd, read that fd) would close the race, but
        # the read still runs as the harness uid, which in container mode is
        # root; the only way for that class of bug to be absent rather than
        # narrowly avoided is not to read as the harness at all.
        #
        # In container mode the sandbox is CurrentSandbox, so the file is read as
        # the unprivileged tool user, and any swap can at most substitute
        # something that uid could already read. Trade-off: where no sandbox
        # exists (local dev on ``auto`` with neither an E2B key nor bwrap) a
        # local image is no longer readable at all. Container mode always
        # reports available, so no deployment that mounts /inputs is affected.
        try:
            from plugins.tools._sandbox import get_sandbox, sandbox_available
            if not sandbox_available():
                return (
                    f"Error: cannot read {image_path_or_url} — no sandbox is "
                    "available, and this tool does not read the filesystem as "
                    "the harness process. Configure a sandbox backend "
                    "(SANDBOX_BACKEND=container inside a task container, or "
                    "bwrap/E2B locally), or pass an http(s) URL."
                )
            sandbox = get_sandbox()
            # Quoted as well as pre-validated: the check above is the boundary,
            # this keeps an ordinary space or bracket in a real filename from
            # breaking the command.
            result_cmd = sandbox.commands.run(
                f"base64 -w 0 {shlex.quote(image_path_or_url)}", timeout=15
            )
            if result_cmd.exit_code != 0:
                return f"Error: Cannot read image from sandbox: {result_cmd.stderr}"
            b64 = result_cmd.stdout.strip()
            mime = _MIME_MAP.get(Path(image_path_or_url).suffix.lower(), "image/png")
            image_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            ]
        except Exception as e:
            return f"Error loading image from sandbox: {e}"

    result = await _call_vision_api(image_content, question)

    # Graceful fallback for vision service outages — return metadata
    # instead of a hard error so the agent can continue with partial info.
    if _is_vision_service_unavailable(result):
        return _vision_unavailable_fallback(image_path_or_url, result)

    return result


def _is_vision_service_unavailable(result: str) -> bool:
    """Check if the result indicates a vision API service outage (5xx)."""
    if not result.startswith("Error:"):
        return False
    return any(
        code in result
        for code in ("503 Service Unavailable", "502 Bad Gateway", "500 Internal Server Error")
    )


def _vision_unavailable_fallback(image_path: str, error_msg: str) -> str:
    """Return a best-effort fallback when vision API is unavailable."""
    import re

    status = "unknown"
    m = re.search(r"(\d{3})\s+\w+", error_msg)
    if m:
        status = m.group(1)

    path = Path(image_path)
    size_info = ""
    if path.is_file():
        size_kb = path.stat().st_size / 1024
        size_info = f" ({size_kb:.0f} KB)"

    return (
        f"Vision API unavailable HTTP {status} — image was loaded successfully"
        f" from {image_path}{size_info} but could not be analyzed."
        f" The image file exists and is readable."
        f" Proceed with text-based reasoning."
    )
