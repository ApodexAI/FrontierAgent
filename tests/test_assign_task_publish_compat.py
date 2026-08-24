import pytest

from plugins.tools.assign_task import (
    AgentTeamAssignmentSpec,
    _fold_top_level_publish_metadata,
    agent_team_assign_task,
)


def test_top_level_false_publish_applies_to_research_batch() -> None:
    tasks = [
        {"agent": "one", "prompt": "inspect"},
        {"agent": "two", "prompt": "verify"},
    ]

    folded = _fold_top_level_publish_metadata(tasks, publish="false")

    assert [item["publish"] for item in folded] == [False, False]


def test_top_level_publisher_metadata_folds_into_single_task() -> None:
    tasks = [{"agent": "writer", "prompt": "publish"}]

    folded = _fold_top_level_publish_metadata(
        tasks,
        publish="true",
        output_paths='["/outputs/report.md"]',
    )

    assert folded == [
        {
            "agent": "writer",
            "prompt": "publish",
            "publish": True,
            "output_paths": '["/outputs/report.md"]',
        }
    ]


def test_top_level_true_publish_never_expands_to_a_batch() -> None:
    tasks = [
        {"agent": "one", "prompt": "inspect"},
        {"agent": "two", "prompt": "verify"},
    ]

    assert _fold_top_level_publish_metadata(tasks, publish=True) is tasks


def test_per_task_metadata_survives_an_echoed_top_level_value() -> None:
    """A call that is already correct must pass through untouched.

    Overwriting demoted the real publisher, and its surviving ``output_paths``
    then failed the contract with a misleading "requires publish=true".
    """
    tasks = [
        {"agent": "researcher", "prompt": "inspect", "publish": False},
        {
            "agent": "writer",
            "prompt": "write",
            "publish": True,
            "output_paths": ["/outputs/report.md"],
        },
    ]

    folded = _fold_top_level_publish_metadata(tasks, publish="false")

    assert [item["publish"] for item in folded] == [False, True]


def test_a_non_mapping_task_entry_is_left_for_normal_validation() -> None:
    """``dict("writer")`` raised ValueError out of the tool call."""
    tasks = ["writer", "reviewer"]

    assert _fold_top_level_publish_metadata(tasks, publish="false") is tasks


@pytest.mark.parametrize("spelling", [True, "true", "TRUE", "True"])
async def test_every_true_spelling_rejects_a_publishing_batch(spelling) -> None:
    """The guard must read ``publish`` the same way the fold does.

    A literal membership test accepted only three spellings, so ``"True"``
    neither folded nor errored: the batch ran with the publish decision and its
    manifest silently dropped, leaving no publisher and nothing to act on.
    """
    result = await agent_team_assign_task.func(
        tasks=[
            {"agent": "one", "prompt": "inspect"},
            {"agent": "two", "prompt": "verify"},
        ],
        publish=spelling,
        output_paths=["/outputs/report.md"],
    )

    assert "accepted only for one task" in result


def test_a_manifest_alone_authorizes_publishing() -> None:
    """``output_paths`` is the grant; the boolean is not required to state it.

    Before, an omitted ``publish`` defaulted to false and collided with the
    manifest, so a coordinator that named the deliverable but skipped the
    boolean got "output_paths requires publish=true" instead of a publisher.
    It skipped the boolean on roughly a quarter of its assignments.
    """
    spec = AgentTeamAssignmentSpec(
        agent="writer",
        prompt="write",
        output_paths=["/outputs/answer.md"],
    )

    assert spec.publish is None
    assert spec.can_publish is True
    assert spec.output_paths == ["/outputs/answer.md"]


def test_no_manifest_is_workspace_only() -> None:
    spec = AgentTeamAssignmentSpec(agent="researcher", prompt="inspect")

    assert spec.can_publish is False


def test_an_explicit_false_still_dispatches_workspace_only() -> None:
    """Callers that keep sending the boolean must not start failing."""
    spec = AgentTeamAssignmentSpec(
        agent="researcher", prompt="inspect", publish=False,
    )

    assert spec.can_publish is False


def test_an_explicit_true_beside_a_manifest_still_publishes() -> None:
    spec = AgentTeamAssignmentSpec(
        agent="writer",
        prompt="write",
        publish=True,
        output_paths=["/outputs/answer.md"],
    )

    assert spec.can_publish is True


def test_false_contradicting_a_manifest_is_rejected_not_resolved() -> None:
    """Neither side of the contradiction may win silently.

    Honouring the flag drops a manifest the coordinator asked for, which is the
    shape that loses the deliverable; honouring the manifest widens authority on
    a call that says not to.
    """
    with pytest.raises(ValueError, match="contradicts output_paths"):
        AgentTeamAssignmentSpec(
            agent="writer",
            prompt="write",
            publish=False,
            output_paths=["/outputs/answer.md"],
        )


def test_true_without_a_manifest_is_still_rejected() -> None:
    with pytest.raises(ValueError, match="requires at least one exact absolute"):
        AgentTeamAssignmentSpec(agent="writer", prompt="write", publish=True)


def test_replace_manifest_without_a_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="replace_manifest requires output_paths"):
        AgentTeamAssignmentSpec(
            agent="writer", prompt="write", replace_manifest=True,
        )


def test_a_top_level_manifest_never_expands_to_a_batch() -> None:
    """The manifest is now the grant, so the fold must gate on it too.

    The old gate keyed on ``publish is True`` only. That was safe while every
    item's ``publish: false`` collided with a folded manifest and the contract
    rejected the call; with the boolean optional there is nothing left to
    collide with, and folding would authorize every researcher in the batch.
    """
    tasks = [
        {"agent": "one", "prompt": "inspect"},
        {"agent": "two", "prompt": "verify"},
    ]

    assert (
        _fold_top_level_publish_metadata(
            tasks, output_paths=["/outputs/answer.md"],
        )
        is tasks
    )


async def test_a_top_level_manifest_on_a_batch_reports_the_error() -> None:
    """Declining to fold must not be silent -- see the sibling fold test."""
    result = await agent_team_assign_task.func(
        tasks=[
            {"agent": "one", "prompt": "inspect"},
            {"agent": "two", "prompt": "verify"},
        ],
        output_paths=["/outputs/answer.md"],
    )

    assert "accepted only for one task" in result


@pytest.mark.parametrize("nested_output_paths", [[], None])
async def test_an_empty_item_manifest_cannot_hide_top_level_authority(
    nested_output_paths,
) -> None:
    """A present-but-empty item key must not silently defeat the top-level grant."""
    result = await agent_team_assign_task.func(
        tasks=[
            {
                "agent": "writer",
                "prompt": "write",
                "output_paths": nested_output_paths,
            },
        ],
        output_paths=["/outputs/answer.md"],
    )

    assert "conflicts with the single tasks[] item" in result
    assert "output_paths manifest inside that task" in result


async def test_item_false_cannot_hide_top_level_publish_true() -> None:
    """The legacy boolean compatibility path needs the same post-fold guard."""
    result = await agent_team_assign_task.func(
        tasks=[
            {"agent": "writer", "prompt": "write", "publish": False},
        ],
        publish=True,
    )

    assert "conflicts with the single tasks[] item" in result


async def test_model_defaults_cannot_hide_top_level_manifest() -> None:
    """BaseModel.model_dump includes default empty fields, so guard them too."""
    result = await agent_team_assign_task.func(
        tasks=[AgentTeamAssignmentSpec(agent="writer", prompt="write")],
        output_paths=["/outputs/answer.md"],
    )

    assert "conflicts with the single tasks[] item" in result


def test_a_top_level_manifest_folds_into_a_single_task() -> None:
    tasks = [{"agent": "writer", "prompt": "write"}]

    folded = _fold_top_level_publish_metadata(
        tasks, output_paths='["/outputs/answer.md"]',
    )

    assert folded == [
        {
            "agent": "writer",
            "prompt": "write",
            "output_paths": '["/outputs/answer.md"]',
        }
    ]
