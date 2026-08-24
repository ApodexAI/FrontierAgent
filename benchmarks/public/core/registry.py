"""Dataset registry and loader for benchmarks."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Importing the config module loads ``.env`` into ``os.environ`` as a side
# effect. Without it, a root set in ``.env`` is silently ignored here and the
# default is used — the loader only populates the settings object, and this
# module reads the environment directly.
import frontier_agent.infra.config  # noqa: F401  (side effect: loads .env)
from benchmarks.public.core.question import BenchmarkQuestion

logger = logging.getLogger(__name__)

# Single root for all eval inputs. Run artefacts land in the per-run
# directory passed via ``--out`` (the bundled scripts default to
# ``./results/<benchmark>``). Each registered benchmark resolves to
# ``_DATA_ROOT / <key>/``.
#
# ``FRONTIER_AGENT_DATASETS_DIR`` moves that root off the repository, which is
# what you want once the corpora are real: OfficeQA and GDPval are several GB
# each, and keeping them inside the checkout puts them in reach of git status,
# the Docker build context, and anything that copies the tree.
_PUBLIC_ROOT = Path(__file__).resolve().parents[1]
_DATA_ROOT = Path(
    os.environ.get("FRONTIER_AGENT_DATASETS_DIR", "").strip()
    or _PUBLIC_ROOT / "datasets"
)


# ── Registry ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetConfig:
    """Declarative description of one benchmark dataset."""

    name: str               # Human-readable ("HLE", "GAIA")
    key: str                # Directory name under data_root/
    jsonl: str = "standardized_data.jsonl"
    extra_metadata_fields: tuple[str, ...] = ()
    # Which JSONL fields hold attachment paths
    image_field: str = "image_path"
    file_field: str = ""          # e.g. "file_path" for GAIA
    file_name_field: str = ""     # e.g. "file_name" for GAIA
    default_pipeline: str = "stateful-react-agent"
    # Override the global ``_DATA_ROOT`` (e.g. workflow-local data/).
    data_root: str = ""
    # Field-name overrides for non-standard column names.
    id_field: str = "id"
    question_field: str = "question"
    answer_field: str = "answer"
    # Storage/scoring modes. Most datasets are JSONL rows that are scored
    # immediately after each task. Some bundled benchmarks publish a JSON
    # question bank and require a separate cross-query scorer instead.
    source_format: str = "jsonl"       # "jsonl" | "json"
    scoring_mode: str = "inline"       # "inline" | "external"
    default_answer_type: str = "exactMatch"


# REGISTRY is built by per-family modules under benchmarks/public/families/.
# Imported lazily here to break the circular dependency:
# families/<x>.py needs DatasetConfig from this module, so this module
# must finish defining DatasetConfig before triggering families load.

def _load_registry() -> dict[str, DatasetConfig]:
    from benchmarks.public.families import REGISTRY as _r
    return _r


REGISTRY: dict[str, DatasetConfig] = _load_registry()


def dataset_root_for(dataset: str) -> Path:
    """Return the directory containing a registered dataset's JSONL/files."""
    cfg = get_config(dataset)
    root = Path(cfg.data_root) if cfg.data_root else _DATA_ROOT
    return root / cfg.key


# ── Loader ───────────────────────────────────────────────────────────────


def _ground_truth(
    row: dict, cfg: DatasetConfig, dataset: str, row_index: int
) -> str:
    """Read a row's answer field as text.

    An empty ``answer_field`` means the dataset carries no ground truth at all
    (external batch scorers hold it). Otherwise the field is mandatory: a
    renamed or missing answer column has to fail here, because the alternative
    is a full run in which every question is judged against an empty target and
    the report reads 0% — the same silent-zero outcome the judge preflight in
    ``run_subprocess`` exists to prevent.

    Non-string values are serialised rather than ``str()``-ed so structured
    ground truth (WideSearch's eval_spec) still round-trips through
    ``json.loads``, and falsy scalars (``0``, ``false``) keep their text form
    instead of collapsing to an empty target.
    """
    if not cfg.answer_field:
        return ""
    if cfg.answer_field not in row:
        raise KeyError(
            f"Dataset {dataset!r} row {row_index + 1} has no {cfg.answer_field!r} "
            f"field. Every inline-scored row needs ground truth; check that the "
            f"standardized file matches the dataset's answer_field."
        )
    raw = row[cfg.answer_field]
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, ensure_ascii=False)


def load_questions(
    dataset: str,
    *,
    limit: int | None = None,
    offset: int = 0,
    answer_type: str | None = None,
    category: str | None = None,
    subset: str | None = None,
) -> list[BenchmarkQuestion]:
    """Load questions from a registered dataset.

    Args:
        dataset: Registry key ("browsecomp", "widesearch", ...).
        limit: Max questions to return (None = all).
        offset: Skip this many questions at the start.
        answer_type: Filter by answer_type field.
        category: Filter by category field.
        subset: Override JSONL path (for custom splits).

    Returns:
        List of BenchmarkQuestion.
    """
    cfg = REGISTRY.get(dataset)
    if cfg is None:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {available}"
        )

    root = Path(cfg.data_root) if cfg.data_root else _DATA_ROOT
    jsonl_path = Path(subset) if subset else root / cfg.key / cfg.jsonl
    if not jsonl_path.exists():
        hint = f"Place the JSONL at {jsonl_path}."
        if dataset == "widesearch":
            base = jsonl_path.parent / "standardized_data.jsonl"
            if base.exists():
                hint = (
                    f"Found {base.name} but not {jsonl_path.name}. "
                    f"WideSearch needs the gold_table baked into the JSONL."
                )
        raise FileNotFoundError(f"Dataset not found at {jsonl_path}. {hint}")

    if cfg.source_format == "jsonl":
        def _jsonl_rows():
            with open(jsonl_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

        rows = _jsonl_rows()
    elif cfg.source_format == "json":
        rows = json.loads(jsonl_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(
                f"Dataset {dataset!r} must contain a top-level JSON list: {jsonl_path}"
            )
    else:
        raise ValueError(
            f"Unsupported source_format {cfg.source_format!r} for dataset {dataset!r}"
        )

    questions: list[BenchmarkQuestion] = []
    for row_index, row in enumerate(rows):
        if row_index < offset:
            continue
        if not isinstance(row, dict):
            raise ValueError(
                f"Dataset {dataset!r} row {row_index + 1} is not an object"
            )

        row_answer_type = row.get("answer_type", cfg.default_answer_type)
        if answer_type and row_answer_type != answer_type:
            continue
        if category and row.get("category", "") != category:
            continue

        metadata = {"category": row.get("category", "")}
        for field_name in cfg.extra_metadata_fields:
            metadata[field_name] = row.get(field_name, "")

        # Attachment paths
        img = row.get(cfg.image_field)
        img = img if img and img != "None" else None

        fp = row.get(cfg.file_field) if cfg.file_field else None
        fp = fp if fp else None

        fn = row.get(cfg.file_name_field) if cfg.file_name_field else None
        fn = fn if fn else None

        questions.append(BenchmarkQuestion(
            id=str(row[cfg.id_field]),
            question=row[cfg.question_field],
            ground_truth=_ground_truth(row, cfg, dataset, row_index),
            answer_type=row_answer_type,
            image_path=img,
            file_path=fp,
            file_name=fn,
            metadata=metadata,
        ))

        if limit is not None and len(questions) >= limit:
            break

    logger.info(
        "Loaded %d %s questions (offset=%d, limit=%s)",
        len(questions), cfg.name, offset, limit,
    )
    return questions


def get_config(dataset: str) -> DatasetConfig:
    """Get dataset config by key. Raises ValueError if not found."""
    cfg = REGISTRY.get(dataset)
    if cfg is None:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(
            f"Unknown dataset {dataset!r}. Available: {available}"
        )
    return cfg
