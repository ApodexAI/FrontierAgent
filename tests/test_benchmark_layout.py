import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARKS = _ROOT / "benchmarks"


def test_benchmark_tree_has_two_top_level_areas():
    assert (_BENCHMARKS / "public" / "README.md").is_file()
    assert (_BENCHMARKS / "frontier_search_bench" / "README.md").is_file()

    legacy_public_dirs = {
        "core",
        "families",
        "harbor_agent",
        "judges",
        "runner",
        "scripts",
    }
    # Assert the *sources* moved rather than that the directories are absent: a
    # checkout that ran the eval layer before the rename keeps untracked
    # benchmarks/<name>/__pycache__ around, and an exists() check would then fail
    # for a reason that has nothing to do with the layout.
    legacy_sources = [
        source
        for name in legacy_public_dirs
        for source in (_BENCHMARKS / name).rglob("*.py")
        if "__pycache__" not in source.parts
    ]
    assert not legacy_sources


def test_frontier_search_source_stays_outside_public_package():
    assert not (_BENCHMARKS / "public" / "frontier_search_bench").exists()
    assert (
        _BENCHMARKS
        / "frontier_search_bench"
        / "queries"
        / "verifiable.json"
    ).is_file()


def test_dataset_root_override_is_shared_by_runtime_and_tooling(tmp_path):
    env = dict(os.environ)
    env["FRONTIER_AGENT_DATASETS_DIR"] = str(tmp_path)
    probe = """
from benchmarks.public.core.registry import _DATA_ROOT
from benchmarks.public.scripts.download_datasets import _DATASETS as download_root
from benchmarks.public.scripts.standardize_file_benchmarks import _DATASETS as standardize_root
print(_DATA_ROOT.resolve())
print(download_root.resolve())
print(standardize_root.resolve())
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == [str(tmp_path.resolve())] * 3


def test_dataset_tooling_reads_the_repo_root_dotenv(tmp_path):
    """FRONTIER_AGENT_DATASETS_DIR set only in .env must still be honoured.

    The runtime picks it up because registry.py imports
    ``frontier_agent.infra.config`` for its dotenv side effect. The dataset
    scripts have no such import, so they have to call ``load_dotenv``
    themselves -- otherwise ``standardize_file_benchmarks.py`` writes into the
    in-repo ``benchmarks/public/datasets/`` while the runner loads from the
    override, and the run silently sees stale data.
    """
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "dotenv.py").write_text(
        "import os, pathlib\n"
        "def load_dotenv(path=None, *a, **k):\n"
        "    expected = pathlib.Path(os.environ['TEST_DOTENV_PATH']).resolve()\n"
        "    assert pathlib.Path(path).resolve() == expected\n"
        "    os.environ['FRONTIER_AGENT_DATASETS_DIR'] = "
        "os.environ['TEST_DATASETS_ROOT']\n"
        "    return True\n"
        "def dotenv_values(*a, **k):\n"
        "    return {}\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.pop("FRONTIER_AGENT_DATASETS_DIR", None)
    expected_root = tmp_path / "datasets-from-dotenv"
    env["TEST_DOTENV_PATH"] = str(_ROOT / ".env")
    env["TEST_DATASETS_ROOT"] = str(expected_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    for module in (
        "benchmarks.public.scripts.download_datasets",
        "benchmarks.public.scripts.standardize_file_benchmarks",
    ):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from {module} import _DATASETS; print(_DATASETS.resolve())",
            ],
            cwd=_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == str(expected_root.resolve()), module
