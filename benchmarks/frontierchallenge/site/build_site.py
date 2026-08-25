"""Package the dependency-free static site as a minimal Worker module."""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).parent
DIST = ROOT / "dist"
MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
}


def discover_assets() -> dict[str, tuple[str, str]]:
    """Return every public static file and extensionless HTML alias."""
    assets: dict[str, tuple[str, str]] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or DIST in path.parents or ".openai" in path.parts:
            continue
        content_type = MIME_TYPES.get(path.suffix.lower())
        if content_type is None:
            continue
        relative = path.relative_to(ROOT).as_posix()
        assets[f"/{relative}"] = (relative, content_type)
        if path.suffix.lower() == ".html":
            alias = "/" if relative == "index.html" else f"/{relative.removesuffix('.html')}"
            assets[alias] = (relative, content_type)
    return assets


def main() -> None:
    shutil.rmtree(DIST, ignore_errors=True)
    (DIST / "server").mkdir(parents=True)
    assets = discover_assets()
    packaged = {
        route: {"body": base64.b64encode((ROOT / filename).read_bytes()).decode("ascii"), "type": content_type}
        for route, (filename, content_type) in assets.items()
    }
    module = """const assets = %s;

function decodeBase64(value) {
  const binary = atob(value);
  return Uint8Array.from(binary, character => character.charCodeAt(0));
}

export default {
  fetch(request) {
    const url = new URL(request.url);
    const requestPath = url.pathname;
    if (requestPath.length > 1 && requestPath.endsWith("/")) {
      url.pathname = requestPath.slice(0, -1);
      return Response.redirect(url, 308);
    }
    const path = requestPath;
    const asset = assets[path];
    if (!asset) return new Response("Not found", { status: 404 });
    const isHtml = asset.type.startsWith("text/html");
    return new Response(decodeBase64(asset.body), {
      headers: {
        "content-type": asset.type,
        "cache-control": isHtml ? "no-cache" : "public, max-age=3600"
      }
    });
  }
};
""" % json.dumps(packaged, ensure_ascii=False)
    (DIST / "server" / "index.js").write_text(module)


if __name__ == "__main__":
    main()
