"""Download + standardize benchmark datasets from HuggingFace into."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load repo-root .env so HF_TOKEN (for gated officeqa/apex) is picked up without
# needing to export it in the shell.
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")

# Must resolve exactly like benchmarks.public.core.registry._DATA_ROOT, or a download
# lands somewhere the runner does not look.
_DATASETS = Path(
    os.environ.get("FRONTIER_AGENT_DATASETS_DIR", "").strip()
    or _REPO_ROOT / "benchmarks" / "public" / "datasets"
)

_REPOS: dict[str, dict] = {
    "officeqa": {
        "repo_id": "databricks/officeqa",
        "local_key": "OfficeQA",
        "allow_patterns": [
            "officeqa_pro.csv",
            "officeqa_full.csv",
            "treasury_bulletins_parsed/transformed/*",
        ],
        "gated": True,
        "standardize": ["officeqa", "officeqa_full"],
    },
    "gdpval": {
        "repo_id": "openai/gdpval",
        "local_key": "GDPval",
        "allow_patterns": ["data/*", "reference_files/*"],
        "gated": False,
        "standardize": ["gdpval"],
    },
    "onemillion": {
        "repo_id": "OneMillionBench/OneMillion-Bench",
        "local_key": "OneMillion-Bench",
        "allow_patterns": ["*/test.json", "economic_value.csv"],
        "gated": False,
        "standardize": ["onemillion"],
    },
    "apex": {
        "repo_id": "mercor/apex-agents",
        "local_key": "APEX",
        "allow_patterns": [
            "tasks_and_rubrics.json",
            "world_descriptions.json",
            "world_files_zipped/*",
            "task_files/*",
        ],
        "gated": True,
        "standardize": ["apex"],
    },
}


def _download(which: str, *, raw_pdfs: bool = False) -> None:
    from huggingface_hub import snapshot_download

    spec = _REPOS[which]
    patterns = list(spec["allow_patterns"])
    if which == "officeqa" and raw_pdfs:
        patterns.append("treasury_bulletin_pdfs/*")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if spec["gated"] and not token:
        print(
            f"  WARNING: {spec['repo_id']} is gated — set HF_TOKEN (and accept "
            f"the terms at https://huggingface.co/datasets/{spec['repo_id']}) "
            "or the download will 401.",
            file=sys.stderr,
        )

    local_dir = _DATASETS / spec["local_key"]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"downloading {spec['repo_id']} -> {local_dir}  (patterns: {patterns})")
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=patterns,
        token=token,
    )

    for std in spec.get("standardize", []):
        print(f"re-standardizing {std} from downloaded source ...")
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("standardize_file_benchmarks.py")), std],
            check=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "which", nargs="*", default=["all"],
        help="officeqa | gdpval | onemillion | apex | all (default: all)",
    )
    parser.add_argument(
        "--raw-pdfs", action="store_true",
        help="OfficeQA: also fetch the original Treasury PDFs (OFFICEQA_DOC_MODE=raw)",
    )
    args = parser.parse_args()

    targets = list(_REPOS) if "all" in args.which else args.which
    unknown = [t for t in targets if t not in _REPOS]
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}; choose from {list(_REPOS)}")

    for t in targets:
        _download(t, raw_pdfs=args.raw_pdfs)
    print("done.")


if __name__ == "__main__":
    main()
