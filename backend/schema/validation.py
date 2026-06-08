"""
Structural validation for workflow schemas.

What it does:
    Validates a WorkflowSchema for structural correctness including node ID
    uniqueness, edge endpoint validity, edge type compatibility with node types,
    cycle detection via topological sort, unreachable node detection, and
    required configuration presence.

Entities in it:
    - SchemaValidationError: Exception carrying a list of validation error messages.
    - SchemaValidator: Stateless validator class with a validate() method.

How used by other modules:
    - The orchestration engine validates schemas before execution to ensure
      the graph is well-formed and executable.
    - The frontend calls validate() to provide real-time feedback when users
      edit workflow schemas in the visual editor.
"""

from collections import deque

from backend.schema.models import EdgeType, NodeType, WorkflowSchema


class SchemaValidationError(Exception):
    """
    Raised when a workflow schema fails validation.

    Description:
        Carries a list of all validation errors found during schema analysis,
        allowing callers to present all issues at once rather than one at a time.

    Attributes:
        errors: List of human-readable error description strings.
    """

    def __init__(self, errors: list[str]) -> None:
        """
        Initialize with the list of validation error messages.

        Description:
            Stores the error list and constructs a combined message string.

        Params:
            errors (list[str]): All validation errors found.

        Returns:
            None
        """
        self.errors = errors
        combined_message = f"Schema validation failed with {len(errors)} error(s): " + "; ".join(errors)
        super().__init__(combined_message)


class SchemaValidator:
    """
    Validates WorkflowSchema instances for structural correctness.

    Description:
        Performs a comprehensive set of checks on a workflow schema to ensure
        it represents a valid, executable workflow graph. All checks are run
        and all errors are collected before raising.

    Attributes:
        None (stateless validator).

    Methods:
        validate: Run all validation checks on a WorkflowSchema.
    """

    def validate(self, schema: WorkflowSchema) -> None:
        """
        Validate a workflow schema for structural correctness.

        Description:
            Runs all validation checks and raises SchemaValidationError if
            any issues are found. Checks include: node ID uniqueness, edge
            endpoint validity, edge type compatibility, tool node constraints,
            cycle detection, unreachable nodes, group config requirements,
            missing tool bindings, and data type mismatches.

        Params:
            schema (WorkflowSchema): The workflow schema to validate.

        Returns:
            None

        Raises:
            SchemaValidationError: If any validation checks fail.
        """
        errors: list[str] = []

        node_map = self._check_node_id_uniqueness(schema, errors)
        self._check_edge_endpoint_validity(schema, node_map, errors)
        self._check_edge_type_compatibility(schema, node_map, errors)
        self._check_tool_node_outgoing_data_flow(schema, node_map, errors)
        self._check_cycles(schema, node_map, errors)
        self._check_unreachable_nodes(schema, node_map, errors)
        self._check_agent_group_config(schema, errors)
        self._check_tool_bindings(schema, node_map, errors)
        self._check_data_type_mismatches(schema, node_map, errors)

        if errors:
            raise SchemaValidationError(errors)

    def _check_node_id_uniqueness(
        self, schema: WorkflowSchema, errors: list[str]
    ) -> dict[str, "NodeType"]:
        """
        Check that all node IDs in the schema are unique.

        Description:
            Builds a mapping of node_id to NodeType, detecting duplicates.

        Params:
            schema (WorkflowSchema): The schema being validated.
            errors (list[str]): Accumulator for error messages.

        Returns:
            dict[str, NodeType]: Mapping from node_id to its NodeType.
        """
        node_map: dict[str, NodeType] = {}
        seen_ids: set[str] = set()

        for node in schema.nodes:
            if node.node_id in seen_ids:
                errors.append(f"Duplicate node_id: '{node.node_id}'")
            else:
                seen_ids.add(node.node_id)
                node_map[node.node_id] = node.node_type

        return node_map

    def _check_edge_endpoint_validity(
        self, schema: WorkflowSchema, node_map: dict[str, NodeType], errors: list[str]
    ) -> None:
        """
        Check that all edge source and target node IDs reference existing nodes.

        Description:
            Verifies each edge's source_node_id and target_node_id exist in
            the schema's node definitions.

        Params:
            schema (WorkflowSchema): The schema being validated.
            node_map (dict[str, NodeType]): Known node IDs and their types.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        for edge in schema.edges:
            if edge.source_node_id not in node_map:
                errors.append(
                    f"Edge '{edge.edge_id}' references non-existent source node: "
                    f"'{edge.source_node_id}'"
                )
            if edge.target_node_id not in node_map:
                errors.append(
                    f"Edge '{edge.edge_id}' references non-existent target node: "
                    f"'{edge.target_node_id}'"
                )

    def _check_edge_type_compatibility(
        self, schema: WorkflowSchema, node_map: dict[str, NodeType], errors: list[str]
    ) -> None:
        """
        Check that edge types are compatible with the connected node types.

        Description:
            Enforces rules: TOOL_CALL edges must target TOOL nodes only,
            DATA_FLOW edges must connect AGENT or AGENT_GROUP nodes only,
            SYNCHRONIZATION edges must connect AGENT or AGENT_GROUP nodes only.

        Params:
            schema (WorkflowSchema): The schema being validated.
            node_map (dict[str, NodeType]): Known node IDs and their types.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        agent_or_group = {NodeType.AGENT, NodeType.AGENT_GROUP}

        for edge in schema.edges:
            source_type = node_map.get(edge.source_node_id)
            target_type = node_map.get(edge.target_node_id)

            if source_type is None or target_type is None:
                continue

            if edge.edge_type == EdgeType.TOOL_CALL:
                if target_type != NodeType.TOOL:
                    errors.append(
                        f"Edge '{edge.edge_id}' is TOOL_CALL but target node "
                        f"'{edge.target_node_id}' is not a TOOL node (is {target_type.value})"
                    )
                if source_type == NodeType.TOOL:
                    errors.append(
                        f"Edge '{edge.edge_id}' is TOOL_CALL but source node "
                        f"'{edge.source_node_id}' is a TOOL node (tools cannot call tools)"
                    )

            elif edge.edge_type == EdgeType.DATA_FLOW:
                if source_type not in agent_or_group:
                    errors.append(
                        f"Edge '{edge.edge_id}' is DATA_FLOW but source node "
                        f"'{edge.source_node_id}' is not AGENT or AGENT_GROUP (is {source_type.value})"
                    )
                if target_type not in agent_or_group:
                    errors.append(
                        f"Edge '{edge.edge_id}' is DATA_FLOW but target node "
                        f"'{edge.target_node_id}' is not AGENT or AGENT_GROUP (is {target_type.value})"
                    )

            elif edge.edge_type == EdgeType.SYNCHRONIZATION:
                if source_type not in agent_or_group:
                    errors.append(
                        f"Edge '{edge.edge_id}' is SYNCHRONIZATION but source node "
                        f"'{edge.source_node_id}' is not AGENT or AGENT_GROUP (is {source_type.value})"
                    )
                if target_type not in agent_or_group:
                    errors.append(
                        f"Edge '{edge.edge_id}' is SYNCHRONIZATION but target node "
                        f"'{edge.target_node_id}' is not AGENT or AGENT_GROUP (is {target_type.value})"
                    )

    def _check_tool_node_outgoing_data_flow(
        self, schema: WorkflowSchema, node_map: dict[str, NodeType], errors: list[str]
    ) -> None:
        """
        Check that TOOL nodes do not have outgoing DATA_FLOW edges.

        Description:
            Tool nodes should only receive calls, not emit data flow to other nodes.

        Params:
            schema (WorkflowSchema): The schema being validated.
            node_map (dict[str, NodeType]): Known node IDs and their types.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        for edge in schema.edges:
            source_type = node_map.get(edge.source_node_id)
            if source_type == NodeType.TOOL and edge.edge_type == EdgeType.DATA_FLOW:
                errors.append(
                    f"TOOL node '{edge.source_node_id}' has outgoing DATA_FLOW edge "
                    f"'{edge.edge_id}', which is not allowed"
                )

    def _check_cycles(
        self, schema: WorkflowSchema, node_map: dict[str, NodeType], errors: list[str]
    ) -> None:
        """
        Detect cycles in the workflow graph using topological sort (Kahn's algorithm).

        Description:
            Builds an adjacency list and in-degree map, then performs BFS-based
            topological sort. If not all nodes are visited, the graph has a cycle.

        Params:
            schema (WorkflowSchema): The schema being validated.
            node_map (dict[str, NodeType]): Known node IDs and their types.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        if not node_map:
            return

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_map}
        in_degree: dict[str, int] = {node_id: 0 for node_id in node_map}

        for edge in schema.edges:
            if edge.source_node_id in node_map and edge.target_node_id in node_map:
                adjacency[edge.source_node_id].append(edge.target_node_id)
                in_degree[edge.target_node_id] += 1

        queue: deque[str] = deque()
        for node_id, degree in in_degree.items():
            if degree == 0:
                queue.append(node_id)

        visited_count = 0
        while queue:
            current_node_id = queue.popleft()
            visited_count += 1
            for neighbor_id in adjacency[current_node_id]:
                in_degree[neighbor_id] -= 1
                if in_degree[neighbor_id] == 0:
                    queue.append(neighbor_id)

        if visited_count < len(node_map):
            cycle_node_ids = [
                node_id for node_id, degree in in_degree.items() if degree > 0
            ]
            errors.append(
                f"Cycle detected in workflow graph involving nodes: {cycle_node_ids}"
            )

    def _check_unreachable_nodes(
        self, schema: WorkflowSchema, node_map: dict[str, NodeType], errors: list[str]
    ) -> None:
        """
        Detect nodes that are unreachable from any root (zero in-degree) node.

        Description:
            Identifies root nodes (no incoming edges) and performs BFS to find
            all reachable nodes. Any node not reached is reported as unreachable.

        Params:
            schema (WorkflowSchema): The schema being validated.
            node_map (dict[str, NodeType]): Known node IDs and their types.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        if not node_map:
            return

        in_degree: dict[str, int] = {node_id: 0 for node_id in node_map}
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_map}

        for edge in schema.edges:
            if edge.source_node_id in node_map and edge.target_node_id in node_map:
                adjacency[edge.source_node_id].append(edge.target_node_id)
                in_degree[edge.target_node_id] += 1

        root_nodes = [node_id for node_id, degree in in_degree.items() if degree == 0]

        if not root_nodes and node_map:
            return

        reachable: set[str] = set()
        queue: deque[str] = deque(root_nodes)

        while queue:
            current_node_id = queue.popleft()
            if current_node_id in reachable:
                continue
            reachable.add(current_node_id)
            for neighbor_id in adjacency[current_node_id]:
                if neighbor_id not in reachable:
                    queue.append(neighbor_id)

        unreachable = set(node_map.keys()) - reachable
        if unreachable:
            errors.append(
                f"Unreachable nodes detected (not reachable from any root): "
                f"{sorted(unreachable)}"
            )

    def _check_agent_group_config(
        self, schema: WorkflowSchema, errors: list[str]
    ) -> None:
        """
        Check that AGENT_GROUP nodes have group_config defined.

        Description:
            Ensures every node with type AGENT_GROUP includes a non-None
            group_config field, which is required for group execution.

        Params:
            schema (WorkflowSchema): The schema being validated.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        for node in schema.nodes:
            if node.node_type == NodeType.AGENT_GROUP and node.group_config is None:
                errors.append(
                    f"AGENT_GROUP node '{node.node_id}' is missing required group_config"
                )

    def _check_tool_bindings(
        self, schema: WorkflowSchema, node_map: dict[str, "NodeType"], errors: list[str]
    ) -> None:
        """
        Check that every tool name referenced in node configs has a TOOL_CALL edge.

        Description:
            For each AGENT or AGENT_GROUP node, checks that every tool name
            listed in node.config.tools has a corresponding TOOL_CALL edge
            targeting a TOOL node whose label matches the tool name.

        Params:
            schema (WorkflowSchema): The schema being validated.
            node_map (dict[str, NodeType]): Known node IDs and their types.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        # Build a map from agent node_id → set of tool names reachable via TOOL_CALL edges.
        agent_to_tool_labels: dict[str, set[str]] = {
            node.node_id: set()
            for node in schema.nodes
            if node.node_type in {NodeType.AGENT, NodeType.AGENT_GROUP}
        }
        tool_node_labels: dict[str, str] = {
            node.node_id: node.label
            for node in schema.nodes
            if node.node_type == NodeType.TOOL
        }

        for edge in schema.edges:
            if edge.edge_type == EdgeType.TOOL_CALL:
                if (
                    edge.source_node_id in agent_to_tool_labels
                    and edge.target_node_id in tool_node_labels
                ):
                    agent_to_tool_labels[edge.source_node_id].add(
                        tool_node_labels[edge.target_node_id]
                    )

        for node in schema.nodes:
            if node.node_type not in {NodeType.AGENT, NodeType.AGENT_GROUP}:
                continue
            for required_tool_name in node.config.tools:
                if required_tool_name not in agent_to_tool_labels.get(node.node_id, set()):
                    errors.append(
                        f"Node '{node.node_id}' references tool '{required_tool_name}' "
                        f"in its config but has no TOOL_CALL edge to a TOOL node "
                        f"with that label"
                    )

    def _check_data_type_mismatches(
        self, schema: WorkflowSchema, node_map: dict[str, "NodeType"], errors: list[str]
    ) -> None:
        """
        Detect data type mismatches on DATA_FLOW edges using response_format hints.

        Description:
            When a source node specifies a response_format and a target node
            specifies expected input via instruction, checks whether the declared
            output type of the source is compatible with what the target expects.
            Reports a warning-level error when both nodes declare incompatible
            response_format types.

        Params:
            schema (WorkflowSchema): The schema being validated.
            node_map (dict[str, NodeType]): Known node IDs and their types.
            errors (list[str]): Accumulator for error messages.

        Returns:
            None
        """
        node_by_id = {node.node_id: node for node in schema.nodes}

        for edge in schema.edges:
            if edge.edge_type != EdgeType.DATA_FLOW:
                continue
            source_node = node_by_id.get(edge.source_node_id)
            target_node = node_by_id.get(edge.target_node_id)
            if source_node is None or target_node is None:
                continue

            source_format = source_node.config.response_format
            target_format = target_node.config.response_format

            if source_format is None or target_format is None:
                continue

            source_type = source_format.get("type")
            target_type = target_format.get("type")
            if source_type and target_type and source_type != target_type:
                errors.append(
                    f"DATA_FLOW edge '{edge.edge_id}': source node "
                    f"'{edge.source_node_id}' outputs response_format type "
                    f"'{source_type}' but target node '{edge.target_node_id}' "
                    f"expects '{target_type}' — potential data type mismatch"
                )
