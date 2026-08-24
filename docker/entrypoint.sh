#!/usr/bin/env bash
set -euo pipefail

align_tool_identity() {
  local requested_uid="${APODEX_HOST_UID:-}"
  local requested_gid="${APODEX_HOST_GID:-}"

  # Only a root harness can remap the deliberately unprivileged tool account.
  # Reject root/nonnumeric host identities rather than weakening that boundary.
  if [ "$(id -u)" -ne 0 ] \
    || ! [[ "$requested_uid" =~ ^[1-9][0-9]*$ ]] \
    || ! [[ "$requested_gid" =~ ^[1-9][0-9]*$ ]]; then
    return
  fi

  current_uid="$(id -u agent-tool)"
  current_gid="$(id -g agent-tool)"
  if [ "$requested_gid" != "$current_gid" ]; then
    groupmod --non-unique --gid "$requested_gid" "$(id -gn agent-tool)"
  fi
  if [ "$requested_uid" != "$current_uid" ]; then
    usermod --non-unique --uid "$requested_uid" agent-tool
  fi
  if [ "$requested_uid" != "$current_uid" ] || [ "$requested_gid" != "$current_gid" ]; then
    # Only the files the remap orphaned need rewriting, and this is best effort
    # under `set -e`: a read-only layer or a bind mount over either path must
    # not abort startup. The tool account still works, it just may not own its
    # own cache.
    chown -R agent-tool:"$(id -gn agent-tool)" /home/agent-tool /opt/tool-venv || true
  fi
  export APODEX_TOOL_HOST_IDENTITY=1
}

align_tool_identity

# No arguments starts the interactive FrontierAgent CLI.
if [ "$#" -eq 0 ]; then
  exec frontier-agent
fi

# If the first argument is a CLI flag (starts with '-'), pass to frontier-agent
if [ "${1:0:1}" = '-' ]; then
  exec frontier-agent "$@"
fi

# Otherwise execute the custom command (e.g. python -m ..., bash, etc.)
exec "$@"
