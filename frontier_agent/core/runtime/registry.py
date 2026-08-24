"""Back-compat re-export. Canonical location: registries.services."""

from frontier_agent.core.runtime.registries.services import (
    _services as _services,
)
from frontier_agent.core.runtime.registries.services import (
    clear as clear,
)
from frontier_agent.core.runtime.registries.services import (
    get as get,
)
from frontier_agent.core.runtime.registries.services import (
    get_optional as get_optional,
)
from frontier_agent.core.runtime.registries.services import (
    get_optional_by_type_name as get_optional_by_type_name,
)
from frontier_agent.core.runtime.registries.services import (
    is_registered as is_registered,
)
from frontier_agent.core.runtime.registries.services import (
    register as register,
)
from frontier_agent.core.runtime.registries.services import (
    restore as restore,
)
from frontier_agent.core.runtime.registries.services import (
    snapshot as snapshot,
)

__all__ = [
    "clear",
    "get",
    "get_optional",
    "get_optional_by_type_name",
    "is_registered",
    "register",
    "restore",
    "snapshot",
]
