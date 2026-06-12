"""
Pydantic data models for workflow schema definitions.

What it does:
    Defines all structured data types used to represent a workflow schema,
    including node types, edge types, configuration objects, positional data,
    and the top-level WorkflowSchema container.

Entities in it:
    - NodeType: Enumeration of node kinds (AGENT, AGENT_GROUP, TOOL).
    - EdgeType: Enumeration of edge kinds (DATA_FLOW, TOOL_CALL, SYNCHRONIZATION).
    - LoggingLevel: Enumeration of logging verbosity levels.
    - GroupStructure: Enumeration of agent group execution structures.
    - WorkflowConfig: Global workflow settings (timeout, logging, tracing).
    - NodeConfig: Per-node LLM and execution configuration.
    - AgentGroupConfig: Configuration for agent group nodes.
    - NodePosition: X/Y coordinates for visual layout.
    - NodeDefinition: Complete node specification in a workflow.
    - EdgeDefinition: Connection between two nodes in a workflow.
    - WorkflowSchema: Top-level container for an entire workflow definition.

How used by other modules:
    - schema.persistence serializes/deserializes these models to/from YAML.
    - schema.validation inspects these models for structural correctness.
    - agent.core reads NodeConfig to configure LLM calls and execution behavior.
    - agent.group reads AgentGroupConfig to determine execution structure.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class NodeType(str, Enum):
    """
    Enumeration of node types in a workflow graph.

    Description:
        Categorizes nodes by their role in the workflow execution.

    Attributes:
        AGENT: A single LLM-powered agent node.
        AGENT_GROUP: A group of coordinated agents.
        TOOL: A tool invocation node.
    """

    AGENT = "agent"
    AGENT_GROUP = "agent_group"
    TOOL = "tool"


class EdgeType(str, Enum):
    """
    Enumeration of edge types connecting workflow nodes.

    Description:
        Categorizes edges by the nature of the connection they represent.

    Attributes:
        DATA_FLOW: Data passing between agent or group nodes.
        TOOL_CALL: An agent invoking a tool node.
        SYNCHRONIZATION: Coordination signal between agent or group nodes.
    """

    DATA_FLOW = "data_flow"
    TOOL_CALL = "tool_call"
    SYNCHRONIZATION = "synchronization"


class LoggingLevel(str, Enum):
    """
    Enumeration of logging verbosity levels for workflow execution.

    Description:
        Controls the granularity of log output during workflow runs.

    Attributes:
        NONE: No logging output.
        ALL: Everything — all events regardless of trace_enabled.
        INFO: Informational messages and above (respects trace_enabled).
        CRITICAL_INFO: Only critical informational messages.
        ERRORS: Only error-level messages.
    """

    NONE = "none"
    ALL = "all"
    INFO = "info"
    CRITICAL_INFO = "critical_info"
    ERRORS = "errors"


class GroupStructure(str, Enum):
    """
    Enumeration of execution structures for agent groups.

    Description:
        Determines how sub-agents within a group are orchestrated.

    Attributes:
        PARALLEL: All sub-agents execute concurrently.
        SEQUENTIAL: Sub-agents execute one after another in order.
        PYRAMID: A lead agent delegates to parallel workers.
        DEFAULT: Falls back to parallel execution.
    """

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    PYRAMID = "pyramid"
    DEFAULT = "default"


class WorkflowConfig(BaseModel):
    """
    Global configuration for a workflow execution.

    Description:
        Holds top-level settings that apply to the entire workflow run,
        including timeout, logging, tracing, and the loop-round bound.

    Attributes:
        total_timeout: Maximum seconds before workflow times out.
        logging_level: Verbosity level for execution logs.
        trace_enabled: Whether to emit execution trace events.
        max_loop_rounds: How many rounds each explicit graph loop
            (self-loop edge or multi-node cycle) executes before moving on.
        max_iterations: Ceiling on the agentic loop inside a single agent
            execution (LLM turn → tool round-trip → LLM turn). Owned by the
            workflow and threaded through orchestration, not per node.
        iteration_sleep: Seconds to sleep between agentic iterations,
            applied uniformly across the workflow.
    """

    total_timeout: int = Field(default=300, description="Maximum execution time in seconds")
    logging_level: LoggingLevel = Field(default=LoggingLevel.INFO, description="Logging verbosity level")
    trace_enabled: bool = Field(default=True, description="Whether execution tracing is active")
    max_loop_rounds: int = Field(default=3, ge=1, description="Rounds each explicit graph loop executes")
    max_iterations: int = Field(default=10, ge=1, description="Max agentic iterations per agent execution")
    iteration_sleep: float = Field(default=0, ge=0, description="Seconds to sleep between agentic iterations")


class NodeConfig(BaseModel):
    """
    Configuration for an individual agent node's LLM and execution behavior.

    Description:
        Specifies the model, generation parameters, retry behavior, and
        termination conditions for a single agent node.

    Attributes:
        model_id: Identifier of the LLM model to use.
        temperature: Sampling temperature for generation.
        max_tokens: Optional maximum token limit for responses.
        response_format: Optional structured output format specification.
        instruction: User-created instruction prompts for the agent.
        retries: Number of retry attempts on failure.
        retry_waiting_time: Base seconds for exponential backoff between retries.
        termination_conditions: Declarative completion criteria for this
            agent. Each is matched against the agent's accumulated output;
            when all are present the agentic loop terminates (the third
            completion layer alongside the mechanical check and the agent's
            own judgement). Loop bound and sleep are workflow-level.
        token_budget: Maximum token budget for assembled context.
        scope_window: Number of few-shot examples to retain.
        tools: List of tool names authorized for this node.
        call_budget: Maximum number of tool calls permitted per execution.
        rate_limit_per_minute: Maximum tool calls per 60-second window.
        few_shot_examples: In-context learning examples for the agent.
    """

    provider: str = Field(default="openrouter", description="LLM provider identifier (e.g. openrouter, openai, ollama)")
    model_id: str = Field(description="LLM model identifier")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum response tokens")
    response_format: Optional[dict] = Field(default=None, description="Structured output format")
    instruction: list[str] = Field(default_factory=list, description="User-created instruction prompts for the agent")
    retries: int = Field(default=2, description="Number of retry attempts")
    retry_waiting_time: float = Field(default=1.5, description="Base seconds for exponential backoff between retries")
    termination_conditions: list[str] = Field(default_factory=list, description="Declarative completion criteria")
    token_budget: int = Field(default=32768, description="Maximum token budget for context assembly")
    scope_window: int = Field(default=5, description="Number of few-shot examples to retain")
    tools: list[str] = Field(default_factory=list, description="Authorized tool names for this node")
    tool_strict: bool = Field(default=True, description="Enforce strict schema conformance on tool call arguments")
    tool_choice: str = Field(default="auto", description="Tool calling mode: auto, required, or none")
    parallel_tool_calls: bool = Field(default=True, description="Whether the LLM may call multiple tools in one turn")
    call_budget: int = Field(default=50, description="Maximum tool calls per execution")
    rate_limit_per_minute: int = Field(default=30, description="Tool calls per 60-second window")
    few_shot_examples: list[dict[str, str]] = Field(default_factory=list, description="In-context learning examples")
    read_upstream_state: bool = Field(default=True, description="Whether this node can read upstream nodes' state")
    expose_downstream_state: bool = Field(default=True, description="Whether this node's state is visible to downstream nodes")
    read_orchestration_state: bool = Field(default=False, description="Whether this node can read the full orchestration state (all nodes)")


class AgentGroupConfig(BaseModel):
    """
    Configuration for an agent group node's orchestration behavior.

    Description:
        Specifies concurrency limits, agent count bounds, execution structure,
        shared state, and tool authorization for a group of agents.

    Attributes:
        max_parallel_agents: Maximum agents running concurrently.
        min_agents: Minimum number of sub-agents required.
        max_agents: Maximum number of sub-agents allowed.
        group_structure: Execution structure for orchestrating sub-agents.
        shared_context: Initial context dictionary shared across sub-agents.
        tool_authorization: List of tool names the group is authorized to use.
        sub_agent_read_group_state: Whether sub-agents can read
            the group state (sibling statuses and shared context).

    Methods:
        validate_agent_bounds: Ensures max_agents >= min_agents.
    """

    max_parallel_agents: int = Field(default=5, description="Maximum concurrent agents")
    min_agents: int = Field(default=2, description="Minimum sub-agent count")
    max_agents: int = Field(default=10, description="Maximum sub-agent count")
    group_structure: GroupStructure = Field(default=GroupStructure.DEFAULT, description="Execution structure")
    shared_context: dict = Field(default_factory=dict, description="Initial context shared across sub-agents")
    tool_authorization: list[str] = Field(default_factory=list, description="Authorized tool names")
    sub_agent_read_group_state: bool = Field(default=True, description="Whether sub-agents can read the group state")

    @model_validator(mode="after")
    def validate_agent_bounds(self) -> "AgentGroupConfig":
        """
        Validate that max_agents is not less than min_agents.

        Description:
            Ensures the agent count bounds form a valid range.

        Params:
            self: The AgentGroupConfig instance being validated.

        Returns:
            AgentGroupConfig: The validated instance.

        Raises:
            ValueError: If max_agents < min_agents.
        """
        if self.max_agents < self.min_agents:
            raise ValueError(
                f"max_agents ({self.max_agents}) must be greater than or equal to "
                f"min_agents ({self.min_agents})"
            )
        return self


class NodePosition(BaseModel):
    """
    Visual position of a node in the workflow editor canvas.

    Description:
        Stores x/y coordinates for rendering nodes in the visual editor.

    Attributes:
        x: Horizontal position coordinate.
        y: Vertical position coordinate.
    """

    x: float = Field(description="Horizontal position")
    y: float = Field(description="Vertical position")


class NodeDefinition(BaseModel):
    """
    Complete definition of a single node in a workflow graph.

    Description:
        Combines identity, type, configuration, and visual position into
        a full node specification for the workflow schema.

    Attributes:
        node_id: Unique identifier for this node within the workflow.
        node_type: The category of this node (AGENT, AGENT_GROUP, TOOL).
        label: Human-readable display name for this node.
        config: LLM and execution configuration for this node.
        group_config: Optional group configuration (required for AGENT_GROUP nodes).
        position: Visual position in the workflow editor.
    """

    node_id: str = Field(description="Unique node identifier")
    node_type: NodeType = Field(description="Node category")
    label: str = Field(description="Display name")
    config: NodeConfig = Field(description="Node execution configuration")
    group_config: Optional[AgentGroupConfig] = Field(default=None, description="Group configuration")
    position: NodePosition = Field(description="Visual editor position")


class EdgeDefinition(BaseModel):
    """
    Definition of a directed edge connecting two nodes in a workflow.

    Description:
        Represents a connection between a source and target node with a
        specific semantic type.

    Attributes:
        edge_id: Unique identifier for this edge within the workflow.
        edge_type: The semantic type of this connection.
        source_node_id: Identifier of the node this edge originates from.
        target_node_id: Identifier of the node this edge points to.
    """

    edge_id: str = Field(description="Unique edge identifier")
    edge_type: EdgeType = Field(description="Edge semantic type")
    source_node_id: str = Field(description="Source node identifier")
    target_node_id: str = Field(description="Target node identifier")


class WorkflowSchema(BaseModel):
    """
    Top-level container for a complete workflow definition.

    Description:
        Aggregates all nodes, edges, and configuration into a single
        serializable schema that fully describes a workflow.

    Attributes:
        schema_id: Unique identifier for this workflow schema.
        name: Human-readable name for this workflow.
        description: Detailed description of what this workflow does.
        nodes: List of all node definitions in the workflow.
        edges: List of all edge definitions connecting the nodes.
        config: Global workflow configuration settings.
    """

    schema_id: str = Field(description="Unique schema identifier")
    name: str = Field(description="Workflow name")
    description: str = Field(description="Workflow description")
    nodes: list[NodeDefinition] = Field(description="All nodes in the workflow")
    edges: list[EdgeDefinition] = Field(description="All edges in the workflow")
    config: WorkflowConfig = Field(default_factory=WorkflowConfig, description="Global workflow config")
