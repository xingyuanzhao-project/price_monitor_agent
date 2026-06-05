"""
Workflow execution scheduler with topological ordering.

What it does:
    Analyses a WorkflowSchema's edge set to produce an ExecutionPlan.
    DATA_FLOW edges determine dependency order (Kahn's algorithm),
    SYNCHRONIZATION edges force node pairs into the same stage (via
    union-find merging before the sort), and TOOL_CALL edges define
    which tools are bound to each node.

Entities in it:
    - ExecutionPlan: immutable plan consumed by the executor.
    - ExecutionScheduler: stateless builder that converts a schema into a plan.

How used by other modules:
    WorkflowExecutor calls ``ExecutionScheduler.build_execution_plan`` at the
    start of every run.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from backend.schema.models import EdgeType, WorkflowSchema


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable execution plan produced by the scheduler.

    Attributes:
        stages: Ordered list of stages; each stage is a list of node IDs
            that may execute in parallel.
        tool_bindings: Mapping of node ID → list of tool names bound via
            TOOL_CALL edges.
        data_flow_sources: Mapping of node ID → list of upstream node IDs
            connected via DATA_FLOW edges.
        synchronization_pairs: List of ``(node_a, node_b)`` pairs that must
            reside in the same stage.
    """

    stages: list[list[str]]
    tool_bindings: dict[str, list[str]] = field(default_factory=dict)
    data_flow_sources: dict[str, list[str]] = field(default_factory=dict)
    synchronization_pairs: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class ExecutionScheduler:
    """Stateless builder that converts a WorkflowSchema into an ExecutionPlan.

    Methods:
        build_execution_plan: Full pipeline — extract edges, sort, return plan.
        extract_tool_bindings: Pull TOOL_CALL edges into a node→tools mapping.
        extract_data_flow_sources: Pull DATA_FLOW edges into a node→sources mapping.
    """

    def build_execution_plan(self, schema: WorkflowSchema) -> ExecutionPlan:
        """Build a complete execution plan from *schema*.

        Args:
            schema: A validated WorkflowSchema.

        Returns:
            An ExecutionPlan with topologically sorted stages, tool bindings,
            data-flow sources, and synchronization pairs.

        Raises:
            ValueError: If the DATA_FLOW graph contains a cycle (after
                merging synchronization groups).
        """
        tool_bindings = self.extract_tool_bindings(schema)
        data_flow_sources = self.extract_data_flow_sources(schema)
        synchronization_pairs = self._extract_synchronization_pairs(schema)
        stages = self._topological_sort(schema, synchronization_pairs)
        return ExecutionPlan(
            stages=stages,
            tool_bindings=tool_bindings,
            data_flow_sources=data_flow_sources,
            synchronization_pairs=synchronization_pairs,
        )

    def extract_tool_bindings(
        self,
        schema: WorkflowSchema,
    ) -> dict[str, list[str]]:
        """Extract tool bindings from TOOL_CALL edges.

        Args:
            schema: The workflow schema to analyse.

        Returns:
            Mapping of source-node ID → list of target-node (tool) IDs.
        """
        bindings: dict[str, list[str]] = defaultdict(list)
        for edge in schema.edges:
            if edge.edge_type == EdgeType.TOOL_CALL:
                bindings[edge.source_node_id].append(edge.target_node_id)
        return dict(bindings)

    def extract_data_flow_sources(
        self,
        schema: WorkflowSchema,
    ) -> dict[str, list[str]]:
        """Extract data-flow sources from DATA_FLOW edges.

        Args:
            schema: The workflow schema to analyse.

        Returns:
            Mapping of target-node ID → list of upstream source-node IDs.
        """
        sources: dict[str, list[str]] = defaultdict(list)
        for edge in schema.edges:
            if edge.edge_type == EdgeType.DATA_FLOW:
                sources[edge.target_node_id].append(edge.source_node_id)
        return dict(sources)

    # -- private helpers ----------------------------------------------------

    def _extract_synchronization_pairs(
        self,
        schema: WorkflowSchema,
    ) -> list[tuple[str, str]]:
        """Pull SYNCHRONIZATION edges into ``(source, target)`` tuples.

        Args:
            schema: The workflow schema.

        Returns:
            List of synchronization pairs.
        """
        return [
            (edge.source_node_id, edge.target_node_id)
            for edge in schema.edges
            if edge.edge_type == EdgeType.SYNCHRONIZATION
        ]

    def _topological_sort(
        self,
        schema: WorkflowSchema,
        synchronization_pairs: list[tuple[str, str]],
    ) -> list[list[str]]:
        """Kahn's topological sort with synchronization-group merging.

        Synchronized nodes are merged into super-nodes via union-find *before*
        the sort so they naturally land in the same stage without violating
        any DATA_FLOW dependency.

        Args:
            schema: The workflow schema.
            synchronization_pairs: ``(node_a, node_b)`` pairs to co-locate.

        Returns:
            Ordered list of stages (each a list of individual node IDs).

        Raises:
            ValueError: If a cycle is detected.
        """
        node_ids = {node.node_id for node in schema.nodes}

        # ---- union-find for sync groups -----------------------------------
        parent: dict[str, str] = {nid: nid for nid in node_ids}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        for node_a, node_b in synchronization_pairs:
            if node_a in node_ids and node_b in node_ids:
                union(node_a, node_b)

        groups: dict[str, list[str]] = defaultdict(list)
        for nid in node_ids:
            groups[find(nid)].append(nid)

        # ---- build group-level adjacency ----------------------------------
        group_ids = set(groups.keys())
        group_edges: set[tuple[str, str]] = set()

        for edge in schema.edges:
            if edge.edge_type == EdgeType.DATA_FLOW:
                source_group = find(edge.source_node_id)
                target_group = find(edge.target_node_id)
                if source_group != target_group:
                    group_edges.add((source_group, target_group))

        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {gid: 0 for gid in group_ids}
        for src_g, tgt_g in group_edges:
            adjacency[src_g].append(tgt_g)
            in_degree[tgt_g] += 1

        # ---- Kahn's algorithm on groups -----------------------------------
        queue: deque[str] = deque(
            gid for gid in group_ids if in_degree[gid] == 0
        )
        group_stages: list[list[str]] = []

        while queue:
            stage = list(queue)
            queue.clear()
            group_stages.append(stage)
            for gid in stage:
                for neighbour in adjacency[gid]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        queue.append(neighbour)

        processed = sum(len(s) for s in group_stages)
        if processed < len(group_ids):
            unprocessed = group_ids - {
                gid for stage in group_stages for gid in stage
            }
            raise ValueError(
                f"Cycle detected in workflow graph. "
                f"Unprocessable groups: {unprocessed}"
            )

        # ---- expand groups back to individual nodes -----------------------
        stages: list[list[str]] = []
        for group_stage in group_stages:
            node_stage: list[str] = []
            for gid in group_stage:
                node_stage.extend(groups[gid])
            if node_stage:
                stages.append(node_stage)

        return stages
