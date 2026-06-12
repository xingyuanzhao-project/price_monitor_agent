"""
Centralized state management for workflow execution.

What it does:
    Defines the three levels of state used during workflow execution:

    - OrchestrationState: Workflow-wide state shared across all nodes.
      Holds per-node status, the workflow-level status, and accumulated
      outputs.  Individual nodes access it through a configurable read-pass.

    - NodeState: Per-node state that tracks the agent's task, conversation
      history, tool call records, and iteration progress through the
      agentic loop.  Owned by the AgentLoop during execution.

    - GroupState: Per-group state shared among sub-agents within an
      AgentGroup.  Sub-agents read from it and write their outputs back
      to it after completion.  Access is configurable via the group
      node's settings.

    These are the ONLY state objects in the system.  Context harnesses,
    execution harnesses, and LLM providers are not state — they are
    configuration and infrastructure.

Entities in it:
    - NodeStatus: Enum of lifecycle statuses for a node.
    - OrchestrationState: Workflow-level state.
    - NodeState: Per-node state.
    - GroupState: Per-group state.

How used by other modules:
    - The orchestration executor creates one OrchestrationState per run
      and one NodeState per node.
    - AgentLoop (backend.orchestration.agent_loop) reads and writes the
      NodeState while driving the agentic loop.
    - AgentGroup creates a GroupState from its config and registers one
      NodeState per sub-agent in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    """Lifecycle status of a single node within a workflow run."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Node-level state
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    """Mutable state for a single agent node during execution.

    Owned by the AgentLoop.  The loop reads and writes this throughout
    the agentic cycle.  When the node finishes, the orchestration layer
    copies the final status into OrchestrationState.

    Attributes:
        node_id: Unique identifier for this node.
        task: The task description assigned to this node.
        status: Current lifecycle status.
        conversation_history: Full ordered list of messages exchanged
            between the agent and the LLM (system, user, assistant,
            tool-result messages).
        tool_calls_record: Structured log of all tool calls made,
            one entry per dispatched call with its result content.
        iteration: Current iteration number (1-indexed).
        max_iterations: Configured ceiling.
        started_at: UTC timestamp when the node began execution.
        completed_at: UTC timestamp when the node finished.
        result_summary: First 500 chars of the final content on success,
            or error message on failure.
    """

    node_id: str
    task: str
    status: NodeStatus = NodeStatus.PENDING
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_record: list[dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 10
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_summary: str = ""

    def start(self) -> None:
        """Transition to IN_PROGRESS."""
        self.status = NodeStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)

    def complete(self, summary: str = "") -> None:
        """Transition to COMPLETED with an optional result summary."""
        self.status = NodeStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.result_summary = summary[:500]

    def fail(self, error: str = "") -> None:
        """Transition to FAILED with an error summary."""
        self.status = NodeStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.result_summary = error[:500]

    def check_completion(self, latest_result_messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Mechanical check: did the latest tool calls produce usable results?

        Reads from ``tool_calls_record`` (accumulated across all iterations)
        and ``latest_result_messages`` (from this iteration only).

        Returns a structured verdict dict with:
            - ``complete`` (bool): Whether the task is mechanically done.
            - ``has_successful_result`` (bool): At least one tool returned
              non-empty, non-error content.
            - ``total_tool_calls`` (int): Total calls across all iterations.
            - ``successful_count`` (int): Number of successful results
              in the latest batch.
            - ``failed_count`` (int): Number of error results.

        Args:
            latest_result_messages: Tool result messages from the current
                iteration (role=tool dicts with content).

        Returns:
            Verdict dict consumed by the agent and traced.
        """
        substantive_count = 0
        trivial_count = 0
        for msg in latest_result_messages:
            content = msg.get("content", "")
            if len(content) > 2:
                substantive_count += 1
            else:
                trivial_count += 1

        has_substantive_result = substantive_count > 0

        return {
            "complete": has_substantive_result,
            "has_successful_result": has_substantive_result,
            "total_tool_calls": len(self.tool_calls_record),
            "successful_count": substantive_count,
            "failed_count": trivial_count,
        }

    @staticmethod
    def evaluate_termination_conditions(
        conditions: list[str], output_text: str
    ) -> dict[str, Any]:
        """Declarative completion layer: are the user's conditions satisfied?

        Each condition in ``conditions`` is a user-authored completion
        criterion. A condition counts as met when its text appears
        (case-insensitive) in ``output_text`` — the agent's accumulated
        output and tool results so far. The task is declaratively done
        when every listed condition is met.

        When ``conditions`` is empty this layer is inactive: it never
        forces or blocks termination, deferring to the mechanical check
        and the agent's own judgement.

        Args:
            conditions: User-specified completion criteria from NodeConfig.
            output_text: Accumulated agent output and tool results.

        Returns:
            Verdict dict with ``active``, ``satisfied``, ``met``, ``unmet``.
        """
        if not conditions:
            return {"active": False, "satisfied": False, "met": [], "unmet": []}
        haystack = output_text.lower()
        met = [c for c in conditions if c.lower() in haystack]
        unmet = [c for c in conditions if c.lower() not in haystack]
        return {
            "active": True,
            "satisfied": not unmet,
            "met": met,
            "unmet": unmet,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "task": self.task,
            "status": self.status.value,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "tool_calls_count": len(self.tool_calls_record),
            "conversation_length": len(self.conversation_history),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result_summary": self.result_summary,
        }


# ---------------------------------------------------------------------------
# Orchestration-level state
# ---------------------------------------------------------------------------

@dataclass
class OrchestrationState:
    """Workflow-wide state shared across all nodes in a run.

    One instance per workflow run.  The executor creates it at run start
    and each node can read it (subject to access configuration).  Only
    the executor writes node statuses; agents read them.

    Attributes:
        run_id: Unique identifier for this run.
        workflow_status: Overall workflow lifecycle status.
        node_states: Per-node state objects keyed by node_id.
        node_outputs: Final output of each completed node, keyed by
            node_id.  Used for data-flow between nodes.
    """

    run_id: str
    workflow_status: str = "running"
    node_states: dict[str, NodeState] = field(default_factory=dict)
    node_outputs: dict[str, Any] = field(default_factory=dict)

    def register_node(self, node_id: str, task: str, max_iterations: int = 10) -> NodeState:
        """Create and register a NodeState for a new node.

        Args:
            node_id: Unique node identifier.
            task: Task description for this node.
            max_iterations: Configured max iterations for the agent loop.

        Returns:
            The newly created NodeState.
        """
        ns = NodeState(
            node_id=node_id,
            task=task,
            max_iterations=max_iterations,
        )
        self.node_states[node_id] = ns
        return ns

    def get_node_state(self, node_id: str) -> NodeState | None:
        """Retrieve a node's state by ID, or None if not registered."""
        return self.node_states.get(node_id)

    def get_readable_snapshot(self, requesting_node_id: str, allowed_node_ids: set[str] | None = None) -> dict[str, Any]:
        """Build a read-only snapshot of orchestration state visible to a node.

        Args:
            requesting_node_id: The node requesting the snapshot.
            allowed_node_ids: If provided, only these node_ids are
                visible.  If None, all nodes are visible.

        Returns:
            A dict with ``workflow_status`` and per-node summaries.
        """
        snapshot: dict[str, Any] = {
            "workflow_status": self.workflow_status,
            "nodes": {},
        }
        for nid, ns in self.node_states.items():
            if nid == requesting_node_id:
                continue
            if allowed_node_ids is not None and nid not in allowed_node_ids:
                continue
            snapshot["nodes"][nid] = ns.to_dict()
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_status": self.workflow_status,
            "node_states": {
                nid: ns.to_dict() for nid, ns in self.node_states.items()
            },
        }


# ---------------------------------------------------------------------------
# Group-level state
# ---------------------------------------------------------------------------

@dataclass
class GroupState:
    """State shared among sub-agents within an AgentGroup.

    Created from the group node's ``shared_context`` config at group
    start.  Sub-agents read from it (subject to access configuration)
    and write their outputs back after completion.

    Attributes:
        group_node_id: The parent group node's ID.
        context: Mutable dict of context data shared across sub-agents.
        sub_agent_states: Per-sub-agent NodeState objects.
    """

    group_node_id: str
    context: dict[str, Any] = field(default_factory=dict)
    sub_agent_states: dict[str, NodeState] = field(default_factory=dict)

    def register_sub_agent(self, agent_id: str, task: str, max_iterations: int = 10) -> NodeState:
        """Create and register a NodeState for a sub-agent."""
        ns = NodeState(
            node_id=agent_id,
            task=task,
            max_iterations=max_iterations,
        )
        self.sub_agent_states[agent_id] = ns
        return ns

    def write_output(self, agent_id: str, content: str) -> None:
        """Write a sub-agent's output into the group context."""
        self.context[agent_id] = content

    def get_readable_snapshot(self, requesting_agent_id: str) -> dict[str, Any]:
        """Build a read-only snapshot visible to a sub-agent.

        Args:
            requesting_agent_id: The sub-agent requesting the snapshot.

        Returns:
            A dict with group context and sibling agent summaries.
        """
        return {
            "context": dict(self.context),
            "siblings": {
                aid: ns.to_dict()
                for aid, ns in self.sub_agent_states.items()
                if aid != requesting_agent_id
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_node_id": self.group_node_id,
            "context": dict(self.context),
            "sub_agent_states": {
                aid: ns.to_dict() for aid, ns in self.sub_agent_states.items()
            },
        }
