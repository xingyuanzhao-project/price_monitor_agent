"""
Orchestration subpackage for workflow scheduling and execution.

What it does:
    Turns a validated WorkflowSchema into a staged, loop-aware execution
    plan (SCC condensation with synchronization constraints; drawn
    cycles become explicit loop units) and then runs each stage's units
    in parallel, wiring up context and execution harnesses per node,
    driving each agent's round-trips (call → result → answer) through
    AgentLoop, coordinating sub-agents through AgentGroup, enforcing
    guardrails, and recording the full run as a RunRecord.

Entities in it:
    - scheduler: ExecutionScheduler, ExecutionPlan.
    - agent_loop: AgentLoop (the agentic round-trip driver) and
      AgentExecutionError.
    - group: AgentGroup multi-agent coordinator.
    - executor: WorkflowExecutor, RunRecord.

How used by other modules:
    The API runs router calls ``WorkflowExecutor.execute_workflow`` to start
    runs and streams events back to the client.  The scheduler, agent loop,
    and group are internal details consumed by the executor.
"""

from backend.orchestration.scheduler import ExecutionPlan, ExecutionScheduler
from backend.orchestration.agent_loop import AgentExecutionError, AgentLoop
from backend.orchestration.group import AgentGroup
from backend.orchestration.executor import RunRecord, WorkflowExecutor
