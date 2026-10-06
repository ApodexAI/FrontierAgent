"""Execution/data pairs across evaluators, wrappers and input redirections.

Assert the hard-deny reason in permissive tiers too: an unrelated allowlist or
nested-shell rejection must not camouflage failure to inspect the payload.
Dangerous examples are ONLY passed to the policy, never to a real shell.
"""
from __future__ import annotations

import shlex

import pytest

from plugins.tools._bash_policy import assess_bash_command

_EVALUATORS = [
    "watch {code}",
    "watch -n 5 {code}",
    "watch -n5 -- {code}",
    "watch --interval=5 {code}",
    "watch --interval 5 {code}",
    "watch --equexit 5 {code}",
    "watch -x -n 5 bash -c {code}",
    "tmux new -d {code}",
    "tmux -L session new-session -d -s session {code}",
    "tmux new-window -n window {code}",
    "tmux split-window -h -t session {code}",
    "tmux respawn-pane -k -t session {code}",
    "tmux new -d bash -c {code}",
    "screen -dmS session bash -c {code}",
    "script -qc {code} /dev/null",
    "script --quiet --command={code} /dev/null",
]
_PREFIXES = ["", "env -u UNUSED ", "timeout --signal TERM 5 ", "> /tmp/log "]
_DANGEROUS_PAYLOADS = [
    ("halt", "Refuses host shutdown/reboot commands."),
    ("bash -c 'halt'", "Refuses host shutdown/reboot commands."),
    ("systemctl start 'reboot.target'", "Refuses host shutdown/reboot commands."),
    ("bash -lc 'mkfs.ext4 /dev/sda'", "Refuses filesystem formatting commands."),
    ("> /tmp/log halt", "Refuses host shutdown/reboot commands."),
]


@pytest.mark.parametrize("template", _EVALUATORS)
@pytest.mark.parametrize("prefix", _PREFIXES)
@pytest.mark.parametrize("payload,reason", _DANGEROUS_PAYLOADS)
@pytest.mark.parametrize("mode", ["off", "warn", "enforce"])
def test_evaluator_compositions_inspect_the_real_code(template, prefix, payload, reason, mode):
    command = prefix + template.format(code=shlex.quote(payload))
    result = assess_bash_command(command, mode=mode)
    assert (result.level, result.reason) == ("deny", reason), command


@pytest.mark.parametrize("template", _EVALUATORS)
@pytest.mark.parametrize("prefix", _PREFIXES)
def test_evaluator_compositions_preserve_document_data(template, prefix):
    command = prefix + template.format(code=shlex.quote("echo 'halt reboot mkfs'"))
    result = assess_bash_command(command, mode="off")
    assert result.level in {"allow", "audit"}, (command, result)


@pytest.mark.parametrize("redirection", ["<<< 'halt'", "<<<'halt'", "0<<< 'halt'", "0<<<'halt'"])
@pytest.mark.parametrize("shell", ["bash", "bash --", "sh -s", "bash -o errexit"])
@pytest.mark.parametrize("prefix", _PREFIXES)
@pytest.mark.parametrize("mode", ["off", "warn", "enforce"])
def test_here_string_variants_are_code(redirection, shell, prefix, mode):
    command = f"{prefix}{shell} {redirection}"
    result = assess_bash_command(command, mode=mode)
    assert (result.level, result.reason) == (
        "deny", "Refuses host shutdown/reboot commands.",
    ), command


@pytest.mark.parametrize("redirection", ["<<< 'halt'", "<<<'halt'", "0<<< 'halt'", "0<<<'halt'"])
@pytest.mark.parametrize("consumer", ["cat", "bash /workspace/run.sh", "python3 /workspace/run.py"])
@pytest.mark.parametrize("prefix", _PREFIXES)
def test_here_string_to_data_consumer_remains_data(redirection, consumer, prefix):
    result = assess_bash_command(f"{prefix}{consumer} {redirection}", mode="off")
    assert result.level in {"allow", "audit"}, result


@pytest.mark.parametrize("command", [
    "watch --unknown-option 5 \"bash -c 'halt'\"",
    "tmux unsupported-command \"bash -c 'halt'\"",
    "screen -unknown-option \"bash -c 'halt'\"",
    "su -c \"bash -c 'halt'\" user",
    "runuser -u user -- bash -c 'halt'",
    "sg group -c \"bash -c 'halt'\"",
    "ssh -p 22 host \"bash -c 'halt'\"",
    "ssh -o StrictHostKeyChecking=no host \"systemctl start 'reboot.target'\"",
])
def test_unsupported_forms_keep_raw_protection_and_remote_payloads_are_checked(command):
    result = assess_bash_command(command, mode="off")
    assert (result.level, result.reason) == (
        "deny", "Refuses host shutdown/reboot commands.",
    )


@pytest.mark.parametrize("command", [
    "tmux -L 'halt' new-session -s 'reboot' \"echo 'mkfs'\"",
    "screen -dmS 'reboot' echo 'halt'",
    "watch -n 5 \"echo 'halt'\" > /tmp/log",
    "script -qc \"echo 'halt'\" /tmp/log",
])
def test_evaluator_metadata_is_not_executable_code(command):
    result = assess_bash_command(command, mode="off")
    assert result.level in {"allow", "audit"}, result


# ── write-then-execute within one command ───────────────────────────────
#
# Text written into a file is data until the same command runs that file as a
# shell script. Interpreters (python3 gen.py) stay data: that is the #628 case.

_WRITERS = [
    "echo {code} > {path}",
    "printf '%s\\n' {code} >> {path}",
    "echo {code} &> {path}",
    "echo {code} >{path}",
    "echo {code} | tee {path} >/dev/null",
    "echo {code} | tee -a {path}",
    "cat > {path} <<'EOF'\n{raw}\nEOF\n",
    "cat > {path} <<EOF\n{raw}\nEOF\n",
]
_RUNNERS = [
    "bash {path}",
    "sh {path}",
    "bash -e {path}",
    "bash -o errexit -- {path}",
    "env bash {path}",
    "timeout 5 bash {path}",
    "source {path}",
    ". {path}",
    "chmod +x {path} && {path}",
]
_SCRIPT_PAYLOADS = [
    ("halt", "Refuses host shutdown/reboot commands."),
    ("mkfs.ext4 /dev/sda", "Refuses filesystem formatting commands."),
]


def _write_then(writer: str, runner: str, code: str, path: str) -> str:
    write = writer.format(code=shlex.quote(code), raw=code, path=path)
    separator = "" if write.endswith("\n") else " && "
    return write + separator + runner.format(path=path)


@pytest.mark.parametrize("writer", _WRITERS)
@pytest.mark.parametrize("runner", _RUNNERS)
@pytest.mark.parametrize("payload,reason", _SCRIPT_PAYLOADS)
def test_written_then_executed_script_is_code(writer, runner, payload, reason):
    command = _write_then(writer, runner, payload, "/workspace/x.sh")
    result = assess_bash_command(command, mode="off")
    assert (result.level, result.reason) == ("deny", reason), command


@pytest.mark.parametrize("command", [
    # relative spelling after cd, and ./ path execution
    "cd /workspace && echo 'halt' > x.sh && bash x.sh",
    "cat > x.sh <<'EOF'\nreboot\nEOF\nchmod +x x.sh && ./x.sh",
    "echo 'halt' >s.sh; . ./s.sh",
])
def test_written_then_executed_script_spellings(command):
    result = assess_bash_command(command, mode="off")
    assert (result.level, result.reason) == (
        "deny", "Refuses host shutdown/reboot commands.",
    ), command


@pytest.mark.parametrize("command", [
    # the #628 incident shape: prose written, then an interpreter runs a script
    "cat > /workspace/gen.py <<'PYEOF'\nprint('exchange halt')\nPYEOF\npython3 /workspace/gen.py",
    # a different file is executed
    "echo 'exchange halt' > /workspace/a.md && bash /workspace/build.sh",
    # the written file is only read
    "cat > /workspace/a.md <<'MD'\nhalt\nMD\ncat /workspace/a.md",
    "echo 'halt' > x.txt && wc -l x.txt",
    # the written script is harmless
    "echo 'ls' > x.sh && bash x.sh",
])
def test_written_file_that_is_not_run_as_shell_stays_data(command):
    result = assess_bash_command(command, mode="off")
    assert result.level in {"allow", "audit"}, (command, result)


def test_parallel_keeps_raw_protection():
    result = assess_bash_command("parallel ::: 'halt'", mode="off")
    assert (result.level, result.reason) == ("deny", "Refuses host shutdown/reboot commands.")


# ── default to the whole-text screen for anything not known to be data ───
#
# Each case was denied on deploy-1.2 and allowed by an earlier revision of this
# PR: data turned back into code through a channel nobody had enumerated.
# Blanking is now opt-in per known data consumer, so these fall back.

@pytest.mark.parametrize("command", [
    # variable indirection and dynamic command names
    "x='halt'; $x",
    "read v <<< 'halt'; $v",
    "printf -v c '%s' 'halt'; $c",
    "declare c='reboot'; $c",
    "export F='halt'; bash -c \"$F\"",
    "x='halt'; eval \"$x\"",
    "$(echo 'halt')",
    "`printf 'halt'`",
    "eval \"$(echo 'halt')\"",
    # commands that execute strings or argv they are handed
    "echo 'halt' | xargs",
    "echo 'halt' | xargs -I{} {}",
    "echo 'halt' | xargs env",
    "trap 'halt' EXIT",
    "alias x='halt'; x",
    "coproc 'halt'",
    "systemd-run 'halt'",
    "chroot / 'halt'",
    "unshare 'reboot'",
    "nsenter -t 1 -m 'halt'",
    "taskset 1 'halt'",
    "strace 'halt'",
    "docker exec c sh -c 'halt'",
    "kubectl exec p -- sh -c 'halt'",
    "expect -c 'spawn halt'",
    "perl -e 'system(\"halt\")'",
    "awk 'BEGIN{system(\"halt\")}'",
    "echo x | sed 's/x/halt/e'",
    "git -c alias.x='!halt' x",
    "echo '* * * * * halt' | crontab -",
    # "script files" that are really stdin or a process substitution
    "source <(echo 'halt')",
    "echo 'halt' | source /dev/stdin",
    "echo 'halt' | . /dev/stdin",
    "bash /dev/stdin <<< 'halt'",
    "echo 'halt' | bash /proc/self/fd/0",
])
def test_unknown_data_to_code_channels_keep_whole_text_screen(command):
    result = assess_bash_command(command, mode="off")
    assert (result.level, result.reason) == (
        "deny", "Refuses host shutdown/reboot commands.",
    ), command


_PROSE = "cat > /workspace/r.md <<'MD'\nWhen an exchange halt applies, reboot the model.\nMD\n"


@pytest.mark.parametrize("follow_up", [
    "pandoc /workspace/r.md -o /workspace/r.docx",
    "soffice --headless --convert-to pdf /workspace/r.md",
    "git add r.md && git commit -m 'add halt policy doc'",
    "awk '/halt/ {print NR}' /workspace/r.md",
    "sed -i 's/halt/pause/' /workspace/r.md",
    "grep -n 'halt' /workspace/r.md | head",
    "python3 /workspace/build.py",
    "ls -la /workspace && wc -l /workspace/r.md",
    "mkdir -p /outputs && cp /workspace/r.md /outputs/",
    "cd /workspace && zip r.zip r.md",
    "systemctl status 'halt'",
])
def test_document_workflows_with_known_consumers_stay_data(follow_up):
    result = assess_bash_command(_PROSE + follow_up, mode="off")
    assert result.level in {"allow", "audit"}, (follow_up, result)
