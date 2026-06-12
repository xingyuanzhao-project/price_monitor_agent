"""
Workflow execution scheduler with loop-aware topological ordering.

What it does:
    Analyses a WorkflowSchema's edge set to produce an ExecutionPlan.
    Edge direction is taken exactly as drawn (source → target).
    DATA_FLOW and TOOL_CALL edges form the dependency graph,
    SYNCHRONIZATION edges force node pairs into the same unit, and
    TOOL_CALL edges additionally bind tools to their calling node.

    Explicit loops are first-class: any cycle in the dependency graph
    (a self-loop edge or a multi-node cycle) condenses into a single
    ExecutionUnit with ``is_loop=True``, which the executor repeats for
    the workflow's configured number of loop rounds.  The condensation
    is acyclic by construction, so stage ordering never fails.

Entities in it:
    - ExecutionUnit: one schedulable unit (plain node, synchronized
      group, or explicit loop) with ordered internal steps.
    - ExecutionPlan: immutable plan consumed by the executor.
    - ExecutionScheduler: stateless builder.

How used by other modules:
    WorkflowExecutor calls ``ExecutionScheduler.build_execution_plan``
    at the start of every run.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from backend.graph import condense
from backend.schema.models import EdgeType, WorkflowSchema


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecutionUnit:
    """One schedulable unit within a stage.

    Attributes:
        steps: Ordered internal steps; each step is a list of node IDs
            that may run in parallel.  Non-loop units have one step.
            For loop units the executor repeats all steps every round.
        is_loop: True when the unit is an explicit loop drawn in the
            editor (self-loop edge or multi-node cycle).
    """

    steps: list[list[str]]
    is_loop: bool

    @property
    def node_ids(self) -> list[str]:
        """All node IDs in this unit, flattened across steps."""
        return [node_id for step in self.steps for node_id in step]


@dataclass(frozen=True)
class ExecutionPlan:
    """Immutable execution plan produced by the scheduler.

    Attributes:
        stages: Ordered stages; each stage is a list of ExecutionUnits
            that may execute in parallel.
        tool_bindings: caller node ID → TOOL-node IDs (TOOL_CALL edges,
            direction as drawn: the agent is the source).
        data_flow_sources: consumer node ID → producer node IDs
            (DATA_FLOW edges, direction as drawn; includes self-loops
            and TOOL-node producers).
        synchronization_pairs: ``(node_a, node_b)`` pairs that must
            share a unit.
    """

    stages: list[list[ExecutionUnit]]
    tool_bindings: dict[str, list[str]] = field(default_factory=dict)
    data_flow_sources: dict[str, list[str]] = field(default_factory=dict)
    synchronization_pairs: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class ExecutionScheduler:
    """Stateless builder that converts a WorkflowSchema into an ExecutionPlan."""

    def build_execution_plan(self, schema: WorkflowSchema) -> ExecutionPlan:
        """Build the complete plan: bindings, sources, loop-aware stages."""
        synchronization_pairs = [
            (edge.source_node_id, edge.target_node_id)
            for edge in schema.edges
            if edge.edge_type == EdgeType.SYNCHRONIZATION
        ]
        return ExecutionPlan(
            stages=self._build_stages(schema, synchronization_pairs),
            tool_bindings=self.extract_tool_bindings(schema),
            data_flow_sources=self.extract_data_flow_sources(schema),
            synchronization_pairs=synchronization_pairs,
        )

    def extract_tool_bindings(self, schema: WorkflowSchema) -> dict[str, list[str]]:
        """caller node ID → TOOL-node IDs, from TOOL_CALL edges as drawn."""
        bindings: dict[str, list[str]] = defaultdict(list)
        for edge in schema.edges:
            if edge.edge_type == EdgeType.TOOL_CALL:
                bindings[edge.source_node_id].append(edge.target_node_id)
        return dict(bindings)

    def extract_data_flow_sources(self, schema: WorkflowSchema) -> dict[str, list[str]]:
        """consumer node ID → producer node IDs, from DATA_FLOW edges as drawn.

        Self-loops appear here: the node lists itself as a producer and
        receives its own previous-round output on every round after the
        first.
        """
        sources: dict[str, list[str]] = defaultdict(list)
        for edge in schema.edges:
            if edge.edge_type == EdgeType.DATA_FLOW:
                sources[edge.target_node_id].append(edge.source_node_id)
        return dict(sources)

    # -- stage construction ---------------------------------------------------

    def _build_stages(
        self,
        schema: WorkflowSchema,
        synchronization_pairs: list[tuple[str, str]],
    ) -> list[list[ExecutionUnit]]:
        """Condense the dependency graph into loop-aware stages.

        1. Union-find merges synchronization pairs into groups.
        2. DATA_FLOW + TOOL_CALL edges become group-level dependency
           edges; a self-loop marks its group as looping.
        3. SCC condensation turns every cycle into one loop unit.
        4. Kahn's algorithm over the acyclic condensation yields stages.
        """
        order = {node.node_id: i for i, node in enumerate(schema.nodes)}
        if not order:
            return []

        parent = {nid: nid for nid in order}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for node_a, node_b in synchronization_pairs:
            if node_a in parent and node_b in parent:
                root_a, root_b = find(node_a), find(node_b)
                if root_a != root_b:
                    parent[root_a] = root_b

        members: dict[str, list[str]] = defaultdict(list)
        for nid in order:
            members[find(nid)].append(nid)
        for nodes in members.values():
            nodes.sort(key=order.__getitem__)

        def group_order(gid: str) -> int:
            return order[members[gid][0]]

        group_ids = sorted(members, key=group_order)

        # Group-level dependency edges, direction exactly as drawn.
        dependency_types = {EdgeType.DATA_FLOW, EdgeType.TOOL_CALL}
        group_edges: set[tuple[str, str]] = set()
        self_looping: set[str] = set()
        for edge in schema.edges:
            if edge.edge_type not in dependency_types:
                continue
            if edge.source_node_id not in parent or edge.target_node_id not in parent:
                continue
            source_group = find(edge.source_node_id)
            target_group = find(edge.target_node_id)
            if source_group == target_group:
                self_looping.add(source_group)
            else:
                group_edges.add((source_group, target_group))

        sorted_edges = sorted(group_edges)
        components, component_of, component_edges = condense(group_ids, sorted_edges)

        units = []
        for component_index, component in enumerate(components):
            is_loop = len(component) > 1 or component[0] in self_looping
            if len(component) == 1:
                steps = [list(members[component[0]])]
            else:
                steps = self._loop_steps(
                    component, members, sorted_edges, component_of,
                    component_index, group_order,
                )
            units.append(ExecutionUnit(steps=steps, is_loop=is_loop))

        # Kahn over the condensation.
        adjacency: dict[int, list[int]] = defaultdict(list)
        in_degree = {i: 0 for i in range(len(units))}
        for source_component, target_component in sorted(component_edges):
            adjacency[source_component].append(target_component)
            in_degree[target_component] += 1

        ready = deque(i for i in in_degree if in_degree[i] == 0)
        stages: list[list[ExecutionUnit]] = []
        while ready:
            wave = sorted(ready, key=lambda i: order[units[i].steps[0][0]])
            ready.clear()
            stages.append([units[i] for i in wave])
            for unit_index in wave:
                for neighbour in adjacency[unit_index]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        ready.append(neighbour)
        return stages

    def _loop_steps(
        self,
        component: list[str],
        members: dict[str, list[str]],
        group_edges: list[tuple[str, str]],
        component_of: dict[str, int],
        component_index: int,
        group_order,
    ) -> list[list[str]]:
        """Order the groups inside a multi-group loop into steps.

        Entry groups are those fed from outside the loop (or the
        earliest-defined group when the loop has no external input).
        Edges leading back into entries are the loop's back edges;
        removing them makes the interior sortable.  Groups stranded by
        nested sub-cycles are appended in definition order.
        """
        member_set = set(component)
        entries = {
            target for source, target in group_edges
            if target in member_set and component_of.get(source) != component_index
        }
        if not entries:
            entries = {min(component, key=group_order)}

        adjacency: dict[str, list[str]] = defaultdict(list)
        in_degree = {gid: 0 for gid in component}
        for source, target in group_edges:
            if source in member_set and target in member_set and target not in entries:
                adjacency[source].append(target)
                in_degree[target] += 1

        ready = deque(sorted(
            (gid for gid in component if in_degree[gid] == 0), key=group_order,
        ))
        steps: list[list[str]] = []
        placed: set[str] = set()
        while ready:
            level = list(ready)
            ready.clear()
            steps.append([nid for gid in level for nid in members[gid]])
            placed.update(level)
            unlocked = []
            for gid in level:
                for neighbour in adjacency[gid]:
                    in_degree[neighbour] -= 1
                    if in_degree[neighbour] == 0:
                        unlocked.append(neighbour)
            ready.extend(sorted(unlocked, key=group_order))

        for gid in sorted((g for g in component if g not in placed), key=group_order):
            steps.append(list(members[gid]))
        return steps
