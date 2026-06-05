"""
Orchestration subpackage for workflow scheduling and execution.

What it does:
    Turns a validated WorkflowSchema into a staged execution plan (via
    topological sort with synchronization constraints) and then runs each
    stage in parallel, wiring up context and execution harnesses per node,
    invoking CoreAgent or AgentGroup, enforcing guardrails, and recording
    the full run as a RunRecord.

Entities in it:
    - scheduler: ExecutionScheduler, ExecutionPlan.
    - executor: WorkflowExecutor, RunRecord.

How used by other modules:
    The API runs router calls ``WorkflowExecutor.execute_workflow`` to start
    runs and streams events back to the client.  The scheduler is an internal
    detail consumed only by the executor.
"""

from backend.orchestration.scheduler import ExecutionPlan, ExecutionScheduler
from backend.orchestration.executor import RunRecord, WorkflowExecutor
