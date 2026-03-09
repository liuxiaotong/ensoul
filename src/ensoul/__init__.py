"""ensoul — Give your AI a soul.

Define digital employees in Markdown with identity, memory, and
negotiation capabilities.  Load them into Claude Code, Cursor, or
any AI IDE via the Model Context Protocol (MCP).
"""

__version__ = "0.1.0"

from ensoul.models import Employee, DiscoveryResult, PipelineResult, StepResult  # noqa: E402, F401
from ensoul.memory import MemoryEntry  # noqa: E402, F401
from ensoul.exceptions import *  # noqa: E402, F401, F403

# Wave 2 modules — scoring, cost tracking, trajectory, delivery, etc.
from ensoul.scoring import score_trajectory, check_behavior_match  # noqa: E402, F401
from ensoul.cost import estimate_cost, enrich_result_with_cost  # noqa: E402, F401
from ensoul.trajectory import TrajectoryCollector  # noqa: E402, F401
from ensoul.trajectory_export import TrajectoryExporter  # noqa: E402, F401
from ensoul.delivery import deliver, DeliveryTarget, DeliveryResult  # noqa: E402, F401
from ensoul.task_registry import TaskRegistry, TaskRecord  # noqa: E402, F401
from ensoul.metrics import MetricsCollector, get_collector  # noqa: E402, F401
from ensoul.permission_request import PermissionManager  # noqa: E402, F401
from ensoul.classification import get_effective_clearance  # noqa: E402, F401
from ensoul.output_sanitizer import strip_internal_tags  # noqa: E402, F401
