"""Known-data exemptions must be invariant to paths and option spelling."""
import shlex

import pytest

from plugins.tools._bash_policy import assess_bash_command


@pytest.mark.parametrize('executable', ['perl', '/usr/bin/perl', './perl'])
@pytest.mark.parametrize('prefix', ['', 'env -u UNUSED ', 'timeout 5 '])
def test_unknown_executable_path_does_not_opt_into_data(executable, prefix):
    result = assess_bash_command(
        prefix + executable + " -e 'system(\"halt\")'", mode='off',
    )
    assert (result.level, result.reason) == ('deny', 'Refuses host shutdown/reboot commands.')


@pytest.mark.parametrize('program', [
    's/old/halt/e', 's!old!halt!e', 's#old#halt#ge', 's|old|halt|e',
    '1e halt', '1,2e halt', '$e halt', '/old/e halt', 'e halt',
    's/old/new/;e halt', 's!old!halt!e; p',
])
@pytest.mark.parametrize('invocation', ['sed {code}', '/usr/bin/sed -n {code}', 'sed -ne {code}'])
def test_sed_execution_forms_do_not_opt_into_data(program, invocation):
    command = invocation.format(code=shlex.quote(program)) + ' /tmp/input'
    result = assess_bash_command(command, mode='off')
    assert (result.level, result.reason) == ('deny', 'Refuses host shutdown/reboot commands.')


@pytest.mark.parametrize('option', [
    "--checkpoint-action='exec=halt'", "--checkpoint-act='exec=halt'",
    "--to-command='halt'", "--use-compress-program='halt'", "-I 'halt'",
    "--rsh-command='halt'", "--unknown='halt'",
])
@pytest.mark.parametrize('template', [
    'tar -cf /tmp/out.tar --checkpoint=1 {option} /tmp/input',
    '/usr/bin/tar -cf /tmp/out.tar /tmp/input {option}',
])
def test_tar_code_options_keep_the_raw_guard(option, template):
    result = assess_bash_command(template.format(option=option), mode='off')
    assert (result.level, result.reason) == ('deny', 'Refuses host shutdown/reboot commands.')


@pytest.mark.parametrize('followup', [
    "sed -i 's/halt/pause/' /workspace/report.md",
    "sed -n 's!halt!pause!gp' /workspace/report.md",
    "sed -ne 's#halt#pause#g' /workspace/report.md",
    "sed --expression='1,2s/halt/pause/' /workspace/report.md",
    "sed -e 's/halt/pause/' -e 's/reboot/restart/' /workspace/report.md",
    "sed -n '1,2p' /workspace/report.md",
    "tar -czf /tmp/report.tar.gz /workspace/report.md",
    "tar czf /tmp/report.tar.gz /workspace/report.md",
    "tar --create --file=/tmp/report.tar --directory /workspace report.md",
    "tar -cf /tmp/report.tar /workspace/report.md --exclude='halt'",
    "/usr/bin/python3 -c \"print('halt')\"",
    "bash /workspace/build.sh",
])
def test_known_document_operations_still_skip_prose(followup):
    command = "cat <<'MD' > /workspace/report.md\nexchange halt and reboot\nMD\n" + followup
    result = assess_bash_command(command, mode='off')
    assert result.level in {'allow', 'audit'}, result


@pytest.mark.parametrize('command', [
    '# halt\nmake all', 'make all # reboot',
    '# mkfs\n/usr/bin/custom-tool',
    '# halt\nbash -c "make all"',
    '# halt\nbash',
    '# halt\necho ls > /tmp/x.sh; bash /tmp/x.sh',
])
def test_fallback_does_not_revive_shell_comments(command):
    result = assess_bash_command(command, mode='off')
    assert result.level in {'allow', 'audit'}, result


@pytest.mark.parametrize('command', [
    "# harmless\nmake 'halt'", "# harmless\n/usr/bin/perl -e 'system(\"halt\")'",
    "# harmless\necho 'halt' > /tmp/x.sh; bash /tmp/x.sh",
    "# harmless\ncat <<'SH' | bash\nhalt\nSH",
])
def test_comments_do_not_hide_actual_code(command):
    result = assess_bash_command(command, mode='off')
    assert (result.level, result.reason) == ('deny', 'Refuses host shutdown/reboot commands.')


# Known data consumers still run programs through specific options,
# subcommands or environment variables. Each case was denied on deploy-1.2.
@pytest.mark.parametrize('command', [
    "pandoc --filter='halt' a.md", "pandoc -F 'halt' a.md",
    "pandoc --pdf-engine='halt' a.md -o a.pdf",
    "pdflatex -shell-escape '\\immediate\\write18{halt}'",
    "latexmk -e '$pdflatex=q/halt/' a.tex",
    "rg --pre 'halt' x .", "sort --compress-program='halt' big.txt",
    "zip -T -TT 'halt' a.zip a.md", "zip -T --unzip-command='halt' a.zip a.md",
    "wget --use-askpass='halt' https://example.invalid",
    "soffice 'macro:///Standard.Module1.halt'",
    "echo 'halt' | xmllint --shell a.xml",
    "uv run 'halt'", "uv run -- sh -c 'halt'", "uv tool run 'halt'",
    "git rebase --exec 'halt' HEAD~1", "git rebase -x 'halt' HEAD~1",
    "git bisect run 'halt'", "git submodule foreach 'halt'",
    "git filter-branch --tree-filter 'halt'", "git difftool -x 'halt'",
    "git grep -O'halt' x", "git fetch --upload-pack='halt' origin",
    "git clone -u 'halt' x y", "git config alias.x '!halt' && git x",
    "git -C /workspace -c core.pager='halt' log",
    "GIT_SSH_COMMAND='halt' git fetch", "export GIT_SSH_COMMAND='halt'; git fetch",
    "PAGER='halt' git log", "GIT_EXTERNAL_DIFF='halt' git diff",
    "EDITOR='halt' git commit", "env GIT_SSH_COMMAND='halt' git fetch",
    # written under another name, then run
    "echo 'halt' > a.txt && cp a.txt x.sh && bash x.sh",
    "echo 'halt' > a && mv a x.sh && sh x.sh",
    "echo 'halt' > a && ln -s a x && . ./x",
])
def test_consumer_code_paths_keep_the_raw_guard(command):
    result = assess_bash_command(command, mode='off')
    assert (result.level, result.reason) == ('deny', 'Refuses host shutdown/reboot commands.')


@pytest.mark.parametrize('followup', [
    "pandoc /workspace/report.md -o /workspace/report.docx",
    "pandoc --toc -s /workspace/report.md -o /workspace/report.html",
    "xelatex -interaction=nonstopmode /workspace/report.tex",
    "rg -n 'halt' /workspace", "sort -u /workspace/report.md",
    "zip -r /outputs/report.zip /workspace/report.md",
    "wget -q -O /workspace/data.csv https://example.invalid/data.csv",
    "soffice --headless --convert-to pdf /workspace/report.md",
    "uv pip install pandas", "uv sync",
    "git add report.md && git commit -m 'document the halt rule'",
    "git -C /workspace log --oneline -5", "git diff --stat",
    "cp /workspace/report.md /outputs/report.md",
])
def test_known_document_operations_with_guarded_tools_skip_prose(followup):
    command = "cat <<'MD' > /workspace/report.md\nexchange halt and reboot\nMD\n" + followup
    result = assess_bash_command(command, mode='off')
    assert result.level in {'allow', 'audit'}, (followup, result)
