"""ensoul — Give your AI a soul.

Define digital employees in Markdown with identity, memory, and
negotiation capabilities.  Load them into Claude Code, Cursor, or
any AI IDE via the Model Context Protocol (MCP).
"""

__version__ = "0.1.0"

from ensoul.models import Employee, DiscoveryResult, PipelineResult, StepResult  # noqa: E402, F401
from ensoul.memory import MemoryEntry  # noqa: E402, F401
from ensoul.exceptions import *  # noqa: E402, F401, F403
