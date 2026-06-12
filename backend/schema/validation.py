"""
Structural validation for workflow schemas.

What it does:
    Validates a WorkflowSchema against the runtime's execution
    capabilities.  Edge direction is meaningful exactly as drawn
    (source → target), and each edge type has semantic requirements:

    - TOOL_CALL is initiated by an LLM loop, so its source must be an
      AGENT or AGENT_GROUP and its target a TOOL.  One tool-call edge
      represents the full call-and-return round trip.
    - DATA_FLOW delivers a producer's output to a consumer's input.
      Consumers must be nodes that take upstream input (AGENT or
      AGENT_GROUP; tools receive input only through calls).  Producers
      must be nodes that can ever produce output: agents and groups
      always can, a TOOL produces output only when at least one agent
      binds it via a TOOL_CALL edge.
    - SYNCHRONIZATION co-stages two executable (AGENT/AGENT_GROUP) nodes.

    Cycles are NOT errors: a cycle in the dependency graph (including a
    node connected to itself) is an explicit loop and is executed as
    one by the orchestrator.

Entities in it:
    - SchemaValidationError: Exception carrying all validation errors.
    - SchemaValidator: Stateless validator with a validate() method.

How used by other modules:
    - The orchestration engine validates schemas before execution.
    - The schemas API exposes validate() for editor feedback.
"""

from backend.schema.models import EdgeType, NodeType, WorkflowSchema

_EXECUTABLE_TYPES = {NodeType.AGENT, NodeType.AGENT_GROUP}


class SchemaValidationError(Exception):
    """Raised when a workflow schema fails validation.

    Attributes:
        errors: List of human-readable error description strings.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        combined_message = (
            f"Schema validation failed with {len(errors)} error(s): "
            + "; ".join(errors)
        )
        super().__init__(combined_message)


class SchemaValidator:
    """Validates WorkflowSchema instances against runtime semantics.

    All checks run and all errors are collected before raising.

    Methods:
        validate: Run all validation checks on a WorkflowSchema.
    """

    def validate(self, schema: WorkflowSchema) -> None:
        """Validate *schema*; raise SchemaValidationError listing every issue.

        Checks: node ID uniqueness, edge endpoint existence, edge
        semantics (direction-aware capability rules per edge type),
        group config presence, and declared data-type compatibility on
        data-flow edges.
        """
        errors: list[str] = []

        node_map = self._check_node_id_uniqueness(schema, errors)
        self._check_edge_endpoints(schema, node_map, errors)
        self._check_edge_semantics(schema, node_map, errors)
        self._check_agent_group_config(schema, errors)
        self._check_data_type_mismatches(schema, errors)

        if errors:
            raise SchemaValidationError(errors)

    def _check_node_id_uniqueness(
        self, schema: WorkflowSchema, errors: list[str]
    ) -> dict[str, NodeType]:
        """Detect duplicate node IDs; return node_id → NodeType for valid ones."""
        node_map: dict[str, NodeType] = {}
        for node in schema.nodes:
            if node.node_id in node_map:
                errors.append(f"Duplicate node_id: '{node.node_id}'")
            else:
                node_map[node.node_id] = node.node_type
        return node_map

    def _check_edge_endpoints(
        self, schema: WorkflowSchema, node_map: dict[str, NodeType], errors: list[str]
    ) -> None:
        """Every edge endpoint must reference an existing node."""
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

    def _check_edge_semantics(
        self, schema: WorkflowSchema, node_map: dict[str, NodeType], errors: list[str]
    ) -> None:
        """Direction-aware capability rules for each edge type.

        The rules mirror the executor exactly: who can initiate a call,
        who can consume upstream input, and who can ever produce output.
        Loops of any shape are legal and intentionally not checked here.
        """
        tools_with_callers = {
            edge.target_node_id
            for edge in schema.edges
            if edge.edge_type == EdgeType.TOOL_CALL
            and node_map.get(edge.source_node_id) in _EXECUTABLE_TYPES
        }

        for edge in schema.edges:
            source_type = node_map.get(edge.source_node_id)
            target_type = node_map.get(edge.target_node_id)
            if source_type is None or target_type is None:
                continue

            if edge.edge_type == EdgeType.TOOL_CALL:
                if source_type not in _EXECUTABLE_TYPES:
                    errors.append(
                        f"Edge '{edge.edge_id}' is TOOL_CALL but its source "
                        f"'{edge.source_node_id}' is {source_type.value}: tool calls "
                        f"are initiated by an LLM loop, so the caller must be an "
                        f"agent or agent group"
                    )
                if target_type != NodeType.TOOL:
                    errors.append(
                        f"Edge '{edge.edge_id}' is TOOL_CALL but its target "
                        f"'{edge.target_node_id}' is {target_type.value}: only TOOL "
                        f"nodes can be called"
                    )

            elif edge.edge_type == EdgeType.DATA_FLOW:
                if target_type not in _EXECUTABLE_TYPES:
                    errors.append(
                        f"Edge '{edge.edge_id}' is DATA_FLOW into "
                        f"'{edge.target_node_id}' ({target_type.value}): tools "
                        f"receive input only through tool calls, so data flow "
                        f"cannot target a TOOL node"
                    )
                if (
                    source_type == NodeType.TOOL
                    and edge.source_node_id not in tools_with_callers
                ):
                    errors.append(
                        f"Edge '{edge.edge_id}' takes data from TOOL node "
                        f"'{edge.source_node_id}', but no agent binds that tool "
                        f"via a tool-call edge, so it can never produce output"
                    )

            elif edge.edge_type == EdgeType.SYNCHRONIZATION:
                for end_id, end_type in (
                    (edge.source_node_id, source_type),
                    (edge.target_node_id, target_type),
                ):
                    if end_type not in _EXECUTABLE_TYPES:
                        errors.append(
                            f"Edge '{edge.edge_id}' is SYNCHRONIZATION but "
                            f"'{end_id}' is {end_type.value}: only executable "
                            f"nodes (agent or agent group) can be synchronized"
                        )

    def _check_agent_group_config(
        self, schema: WorkflowSchema, errors: list[str]
    ) -> None:
        """AGENT_GROUP nodes require a group_config."""
        for node in schema.nodes:
            if node.node_type == NodeType.AGENT_GROUP and node.group_config is None:
                errors.append(
                    f"AGENT_GROUP node '{node.node_id}' is missing required group_config"
                )

    def _check_data_type_mismatches(
        self, schema: WorkflowSchema, errors: list[str]
    ) -> None:
        """Flag DATA_FLOW edges whose endpoints declare incompatible response_format types."""
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
