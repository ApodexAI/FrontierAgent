from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_eval.sh"


def test_orca_preflight_uses_declared_environment_not_instruction_text():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'EFFECTIVE_TASK_ENVS[$index]' in text
    assert "grep -qil 'orca'" not in text


def test_harbor_receives_exact_effective_task_ids():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'for task_id in "${EFFECTIVE_TASK_IDS[@]}"' in text
    assert 'INCLUDE_ARGS+=("--include-task-name" "$task_id")' in text


def test_staging_dereferences_hugging_face_cache_symlinks():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'cp -aL "$task_dir" "$dest"' in text


def test_verifier_work_is_limited_to_effective_task_directories():
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.count('for task_dir in "${EFFECTIVE_TASK_DIRS[@]}"') == 2
