# Mock support incidents for TUI review

> Demo asset only. These fictional reports intentionally mix possible defects,
> expectation problems, and documentation problems. Investigators must verify
> them against current code and tests.

## INC-101 — “My command disappeared during a run”

A developer entered `/config` while an Agent Team task was still running. They
expected the command to execute automatically after the task, but only noticed
a short busy notification and did not see configuration output.

## INC-102 — “The intervention did not stop the researchers”

During a five-worker investigation, a user entered: “Stop the current research
and focus only on tests.” Existing worker rows continued to run for several
minutes. The final report did reflect the new emphasis, but the user believed
the steering feature had failed because they expected immediate cancellation.

## INC-103 — “Queued and running looked the same”

A user reported that a worker waiting behind another assignment appeared to be
actively consuming model capacity. The report does not include a screenshot and
may refer to an older build.

## INC-104 — “Space did nothing in Plan”

A user selected the Plan tab and pressed `Space`, expecting a task-detail modal.
They concluded that keyboard navigation was broken. The onboarding material they
followed described Space as a universal preview shortcut.

## INC-105 — “The right pane vanished over SSH”

After resizing an SSH terminal to 88 columns, the sidebar disappeared. The user
could still type and read the transcript but did not know whether the run had
lost its task state or how to restore the pane.

## INC-106 — “The generated archive could not be previewed”

A task produced `analysis.zip`. The file appeared in Files, but the user says
the preview was blank. It is unclear whether the archive was corrupt, empty, or
larger than the preview limits.

## INC-107 — “Interrupted means fully resumable”

A user pressed `Ctrl-C` during a long tool call and expected `/resume` to
continue from the middle of that exact command. After resuming, the session only
contained work completed before the interruption.

## INC-108 — “Approval was accepted accidentally”

A user claims that pressing Enter immediately after an approval dialog appeared
executed a file write. No terminal recording or trace was supplied. Verify the
default selection and whether dangerous actions use a stronger confirmation.
