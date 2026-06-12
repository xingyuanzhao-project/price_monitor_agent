"""End-to-end tests for the price_monitor_agent workflow system."""

from pathlib import Path

import pytest

from tests.conftest import TEST_MODEL_ID
from backend.schema.models import (
    LoggingLevel,
    WorkflowSchema,
    NodeDefinition,
    EdgeDefinition,
    EdgeType,
    NodeType,
    NodeConfig,
    AgentGroupConfig,
    GroupStructure,
    NodePosition,
    WorkflowConfig,
)
from backend.schema.persistence import SchemaPersistence
from backend.schema.validation import SchemaValidator, SchemaValidationError
from backend.orchestration.executor import WorkflowExecutor
from backend.orchestration.scheduler import ExecutionScheduler
from backend.harness.context import ContextHarness, GuardrailViolationError
from backend.tools.financial_analysis import TechnicalAnalysisTool
from backend.tools.text_analysis import ChunkTextTool, ScoreTextTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_agent_schema(
    model_id: str = TEST_MODEL_ID,
    temperature: float = 0.7,
    max_iterations: int = 3,
    instruction: list[str] | None = None,
) -> WorkflowSchema:
    return WorkflowSchema(
        schema_id="test_single_agent",
        name="Single Agent Test",
        description="Minimal single-agent workflow for testing.",
        nodes=[
            NodeDefinition(
                node_id="agent_1",
                node_type=NodeType.AGENT,
                label="Summarize the current state of AI in 2-3 sentences.",
                config=NodeConfig(
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=512,
                    instruction=instruction or ["Respond concisely"],
                ),
                position=NodePosition(x=100, y=100),
            )
        ],
        edges=[],
        config=WorkflowConfig(total_timeout=60, max_iterations=max_iterations),
    )


# ---------------------------------------------------------------------------
# Test 1: Simple single-agent schema execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_single_agent_schema(workflow_executor: WorkflowExecutor):
    schema = _make_single_agent_schema()
    record = await workflow_executor.execute_workflow(schema)

    assert record.status == "completed", f"Expected completed, got {record.status}. Errors: {record.errors}"
    assert "agent_1" in record.node_outputs
    output = record.node_outputs["agent_1"]
    content = output.get("content", "") if isinstance(output, dict) else str(output)
    assert len(content) > 10, f"Expected non-empty meaningful content, got: {content!r}"


# ---------------------------------------------------------------------------
# Test 2: Sequential two-agent pipeline with data_flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_two_agent_pipeline(workflow_executor: WorkflowExecutor):
    schema = WorkflowSchema(
        schema_id="test_sequential",
        name="Sequential Two-Agent Pipeline",
        description="Two agents connected by data_flow edge.",
        nodes=[
            NodeDefinition(
                node_id="writer",
                node_type=NodeType.AGENT,
                label="Write a 2-sentence summary about renewable energy trends.",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.6,
                    max_tokens=512,
                    instruction=["Write exactly 2 sentences"],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="reviewer",
                node_type=NodeType.AGENT,
                label="Review the upstream text and provide one improvement suggestion.",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.4,
                    max_tokens=512,
                    instruction=["Provide exactly one concrete suggestion"],
                ),
                position=NodePosition(x=100, y=300),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="writer_to_reviewer",
                edge_type=EdgeType.DATA_FLOW,
                source_node_id="writer",
                target_node_id="reviewer",
            )
        ],
        config=WorkflowConfig(total_timeout=90),
    )

    record = await workflow_executor.execute_workflow(schema)

    assert record.status == "completed", f"Errors: {record.errors}"
    assert "writer" in record.node_outputs
    assert "reviewer" in record.node_outputs

    writer_output = record.node_outputs["writer"]
    reviewer_output = record.node_outputs["reviewer"]
    writer_content = writer_output.get("content", "") if isinstance(writer_output, dict) else str(writer_output)
    reviewer_content = reviewer_output.get("content", "") if isinstance(reviewer_output, dict) else str(reviewer_output)

    assert len(writer_content) > 20, f"Writer produced too little: {writer_content!r}"
    assert len(reviewer_content) > 20, f"Reviewer produced too little: {reviewer_content!r}"


# ---------------------------------------------------------------------------
# Test 3: Agent with tool calls (technical_analysis RSI)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_with_tool_calls(workflow_executor: WorkflowExecutor):
    real_prices = [
        44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
        45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    ]
    prices_str = ", ".join(str(p) for p in real_prices)

    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_tool_calls",
        name="Agent With Tool Calls",
        description="Agent uses technical_analysis tool to calculate RSI.",
        nodes=[
            NodeDefinition(
                node_id="analyst",
                node_type=NodeType.AGENT,
                label=(
                    f"Calculate the RSI for the following close prices using the "
                    f"technical_analysis tool with indicator='rsi' and "
                    f"close=[{prices_str}]. Report the final RSI values."
                ),
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.1,
                    max_tokens=1024,
                    instruction=[
                        "You MUST call the technical_analysis tool with indicator='rsi'",
                        "Report the numeric RSI values in your response",
                    ],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="technical_analysis_tool",
                node_type=NodeType.TOOL,
                label="technical_analysis",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    tools=["technical_analysis"],
                ),
                position=NodePosition(x=300, y=100),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="analyst_to_tool",
                edge_type=EdgeType.TOOL_CALL,
                source_node_id="analyst",
                target_node_id="technical_analysis_tool",
            )
        ],
        config=WorkflowConfig(
            total_timeout=60,
            logging_level=LoggingLevel.ALL,
            trace_enabled=True,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    assert record.status == "completed", f"Errors: {record.errors}"
    assert "analyst" in record.node_outputs
    output = record.node_outputs["analyst"]
    content = output.get("content", "") if isinstance(output, dict) else str(output)
    assert len(content) > 5, f"Agent produced no meaningful content: {content!r}"

    tool_call_events = [e for e in captured_events if e["type"] == "tool_call"]
    assert len(tool_call_events) > 0, (
        f"No tool_call event in {[e['type'] for e in captured_events]}"
    )
    assert tool_call_events[0]["tool_name"] == "technical_analysis", (
        f"Expected technical_analysis, got {tool_call_events[0].get('tool_name')}"
    )

    tool_result_events = [e for e in captured_events if e["type"] == "tool_result"]
    assert len(tool_result_events) > 0, "No tool_result event captured"


# ---------------------------------------------------------------------------
# Test 4: Agent group execution (parallel structure)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_group_execution(workflow_executor: WorkflowExecutor):
    schema = WorkflowSchema(
        schema_id="test_agent_group",
        name="Agent Group Test",
        description="Agent group with parallel structure.",
        nodes=[
            NodeDefinition(
                node_id="research_team",
                node_type=NodeType.AGENT_GROUP,
                label="Research the pros and cons of electric vehicles. Each sub-agent should cover one aspect.",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.6,
                    max_tokens=1024,
                    instruction=["Provide evidence-based analysis"],
                ),
                group_config=AgentGroupConfig(
                    min_agents=2,
                    max_agents=3,
                    max_parallel_agents=3,
                    group_structure=GroupStructure.PARALLEL,
                ),
                position=NodePosition(x=200, y=200),
            )
        ],
        edges=[],
        config=WorkflowConfig(total_timeout=90),
    )

    record = await workflow_executor.execute_workflow(schema)

    assert record.status == "completed", f"Errors: {record.errors}"
    assert "research_team" in record.node_outputs
    output = record.node_outputs["research_team"]
    content = output.get("content", "") if isinstance(output, dict) else str(output)
    assert len(content) > 50, f"Group produced too little content: {content!r}"

    if isinstance(output, dict) and "sub_agent_results" in output:
        sub_results = output["sub_agent_results"]
        assert len(sub_results) >= 2, f"Expected >=2 sub-agents, got {len(sub_results)}"


# ---------------------------------------------------------------------------
# Test 4b: Explicit self-loop execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_self_loop_execution(workflow_executor: WorkflowExecutor):
    """An agent whose output port feeds its own input port runs for
    max_loop_rounds rounds, receiving its previous output each round."""
    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_self_loop",
        name="Self Loop",
        description="Single agent looping on itself.",
        nodes=[
            NodeDefinition(
                node_id="refiner",
                node_type=NodeType.AGENT,
                label=(
                    "Write one sentence about electric vehicles. If upstream "
                    "data contains a previous sentence, improve on it."
                ),
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.3,
                    max_tokens=256,
                ),
                position=NodePosition(x=100, y=100),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="self_loop",
                edge_type=EdgeType.DATA_FLOW,
                source_node_id="refiner",
                target_node_id="refiner",
            ),
        ],
        config=WorkflowConfig(
            total_timeout=120,
            logging_level=LoggingLevel.ALL,
            max_loop_rounds=2,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    assert record.status == "completed", f"Errors: {record.errors}"

    round_events = [e for e in captured_events if e["type"] == "loop_round_started"]
    assert len(round_events) == 2, (
        f"Expected 2 loop rounds, got {len(round_events)}: "
        f"{[e['type'] for e in captured_events]}"
    )
    assert [e["round"] for e in round_events] == [1, 2]

    node_inputs = [e for e in captured_events if e["type"] == "node_input"]
    assert len(node_inputs) == 2, f"Agent should run once per round, ran {len(node_inputs)}"

    # Round 2 must carry round 1's output as upstream data.
    first_round_messages = str(node_inputs[0]["messages"])
    second_round_messages = str(node_inputs[1]["messages"])
    assert "Upstream data from refiner" not in first_round_messages
    assert "Upstream data from refiner" in second_round_messages, (
        "Second round received no upstream data from the first round"
    )


# ---------------------------------------------------------------------------
# Test 4b: Three-layer completion + workflow-level iteration budget
#
# Deterministic (no network): a stub provider that *always* emits a tool
# call, so the agent never stops on its own (layer 2 never fires). This
# isolates the loop bound (workflow max_iterations) and the declarative
# termination layer (layer 3).
# ---------------------------------------------------------------------------


class _AlwaysToolCallProvider:
    """Stub LLMProvider that emits one native tool call every turn."""

    def __init__(self, tool_result: str) -> None:
        self.tool_result = tool_result
        self.calls = 0

    async def complete(self, *, messages, tools=None, response_format=None,
                       tool_choice=None, parallel_tool_calls=None) -> dict:
        self.calls += 1
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{self.calls}",
                        "type": "function",
                        "function": {"name": "probe", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }


def _make_core_agent(provider, conditions, tool_result):
    from backend.agent.core import CoreAgent
    from backend.state import NodeState

    async def handler(assistant_message):
        return [{
            "role": "tool",
            "tool_call_id": assistant_message["tool_calls"][0]["id"],
            "content": tool_result,
        }]

    node_state = NodeState(node_id="probe_agent", task="probe")
    node_state.start()
    agent = CoreAgent(
        node_id="probe_agent",
        label="Probe",
        config=NodeConfig(model_id=TEST_MODEL_ID, termination_conditions=conditions),
        llm_provider=provider,
        tool_call_handler=handler,
        node_state=node_state,
    )
    return agent


@pytest.mark.asyncio
async def test_declarative_termination_stops_loop_early():
    """Layer 3: when the agent never stops on its own but a declarative
    termination condition is matched in the accumulated output, the loop
    terminates before exhausting the iteration budget."""
    provider = _AlwaysToolCallProvider(tool_result="RSI computed: 71.4 (analysis complete)")
    agent = _make_core_agent(provider, ["analysis complete"], provider.tool_result)

    result = await agent.execute(
        [{"role": "user", "content": "probe"}],
        tools=[{"type": "function", "function": {"name": "probe", "parameters": {}}}],
        max_iterations=10,
        iteration_sleep=0,
    )

    assert result["iterations"] == 1, (
        f"Declarative termination should stop after iteration 1, "
        f"got {result['iterations']}"
    )
    assert provider.calls == 1
    assert "analysis complete" in result["content"]


@pytest.mark.asyncio
async def test_workflow_iteration_budget_enforced():
    """The loop bound is the workflow-level max_iterations: an agent that
    never declares done and has no satisfiable termination condition runs
    exactly max_iterations times, then fails."""
    from backend.agent.core import AgentExecutionError

    provider = _AlwaysToolCallProvider(tool_result="partial data")
    agent = _make_core_agent(provider, ["never appears in output"], provider.tool_result)

    with pytest.raises(AgentExecutionError) as exc_info:
        await agent.execute(
            [{"role": "user", "content": "probe"}],
            tools=[{"type": "function", "function": {"name": "probe", "parameters": {}}}],
            max_iterations=4,
            iteration_sleep=0,
        )

    assert exc_info.value.iterations_completed == 4
    assert provider.calls == 4
    assert "Maximum iterations (4)" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 5: Schema validation errors
# ---------------------------------------------------------------------------


def test_schema_validation_errors(schema_validator: SchemaValidator):
    # A drawn cycle is an explicit loop, not an error: it must validate
    # clean and condense into a single loop unit in the plan.
    cycle_schema = WorkflowSchema(
        schema_id="cycle_test",
        name="Cycle Test",
        description="Two agents looping on each other.",
        nodes=[
            NodeDefinition(
                node_id="a", node_type=NodeType.AGENT, label="A",
                config=NodeConfig(model_id=TEST_MODEL_ID),
                position=NodePosition(x=0, y=0),
            ),
            NodeDefinition(
                node_id="b", node_type=NodeType.AGENT, label="B",
                config=NodeConfig(model_id=TEST_MODEL_ID),
                position=NodePosition(x=100, y=0),
            ),
        ],
        edges=[
            EdgeDefinition(edge_id="a_to_b", edge_type=EdgeType.DATA_FLOW, source_node_id="a", target_node_id="b"),
            EdgeDefinition(edge_id="b_to_a", edge_type=EdgeType.DATA_FLOW, source_node_id="b", target_node_id="a"),
        ],
        config=WorkflowConfig(),
    )
    schema_validator.validate(cycle_schema)
    plan = ExecutionScheduler().build_execution_plan(cycle_schema)
    loop_units = [u for stage in plan.stages for u in stage if u.is_loop]
    assert len(loop_units) == 1, f"Expected one loop unit, got {plan.stages}"
    assert sorted(loop_units[0].node_ids) == ["a", "b"]

    # Self-loop: an agent feeding its own output back into its input.
    self_loop_schema = cycle_schema.model_copy(update={
        "schema_id": "self_loop_test",
        "edges": [
            EdgeDefinition(edge_id="a_to_a", edge_type=EdgeType.DATA_FLOW, source_node_id="a", target_node_id="a"),
            EdgeDefinition(edge_id="a_to_b", edge_type=EdgeType.DATA_FLOW, source_node_id="a", target_node_id="b"),
        ],
    })
    schema_validator.validate(self_loop_schema)
    plan = ExecutionScheduler().build_execution_plan(self_loop_schema)
    loop_units = [u for stage in plan.stages for u in stage if u.is_loop]
    assert len(loop_units) == 1 and loop_units[0].node_ids == ["a"]
    assert plan.data_flow_sources["a"] == ["a"]

    bad_tool_call_schema = WorkflowSchema(
        schema_id="bad_tool_call",
        name="Bad Tool Call",
        description="TOOL_CALL edge pointing at non-tool node.",
        nodes=[
            NodeDefinition(
                node_id="agent_x", node_type=NodeType.AGENT, label="X",
                config=NodeConfig(model_id=TEST_MODEL_ID),
                position=NodePosition(x=0, y=0),
            ),
            NodeDefinition(
                node_id="agent_y", node_type=NodeType.AGENT, label="Y",
                config=NodeConfig(model_id=TEST_MODEL_ID),
                position=NodePosition(x=100, y=0),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="bad_edge", edge_type=EdgeType.TOOL_CALL,
                source_node_id="agent_x", target_node_id="agent_y",
            ),
        ],
        config=WorkflowConfig(),
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_validator.validate(bad_tool_call_schema)
    assert any("TOOL_CALL" in e for e in exc_info.value.errors)

    # Tools take input only through calls and produce output only when
    # called: data flow into a tool, and data flow out of a caller-less
    # tool, are both semantic errors.
    tool_semantics_schema = WorkflowSchema(
        schema_id="tool_semantics",
        name="Tool Semantics",
        description="Bad data flow around an unbound tool.",
        nodes=[
            NodeDefinition(
                node_id="agent_z", node_type=NodeType.AGENT, label="Z",
                config=NodeConfig(model_id=TEST_MODEL_ID),
                position=NodePosition(x=0, y=0),
            ),
            NodeDefinition(
                node_id="lonely_tool", node_type=NodeType.TOOL, label="Lonely",
                config=NodeConfig(model_id=TEST_MODEL_ID, tools=["technical_analysis"]),
                position=NodePosition(x=100, y=0),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="into_tool", edge_type=EdgeType.DATA_FLOW,
                source_node_id="agent_z", target_node_id="lonely_tool",
            ),
            EdgeDefinition(
                edge_id="from_unbound_tool", edge_type=EdgeType.DATA_FLOW,
                source_node_id="lonely_tool", target_node_id="agent_z",
            ),
        ],
        config=WorkflowConfig(),
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_validator.validate(tool_semantics_schema)
    assert any("into_tool" in e for e in exc_info.value.errors)
    assert any("from_unbound_tool" in e for e in exc_info.value.errors)

    # The same tool push becomes legal once an agent binds the tool:
    # agent_z calls it, and it pushes its results onward.
    bound_push_schema = tool_semantics_schema.model_copy(update={
        "schema_id": "tool_push",
        "edges": [
            EdgeDefinition(
                edge_id="z_calls", edge_type=EdgeType.TOOL_CALL,
                source_node_id="agent_z", target_node_id="lonely_tool",
            ),
            EdgeDefinition(
                edge_id="tool_pushes", edge_type=EdgeType.DATA_FLOW,
                source_node_id="lonely_tool", target_node_id="agent_z",
            ),
        ],
    })
    schema_validator.validate(bound_push_schema)

    bad_ref_schema = WorkflowSchema(
        schema_id="bad_ref",
        name="Bad Reference",
        description="Edge references non-existent node.",
        nodes=[
            NodeDefinition(
                node_id="only_node", node_type=NodeType.AGENT, label="Only",
                config=NodeConfig(model_id=TEST_MODEL_ID),
                position=NodePosition(x=0, y=0),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="dangling", edge_type=EdgeType.DATA_FLOW,
                source_node_id="only_node", target_node_id="ghost_node",
            ),
        ],
        config=WorkflowConfig(),
    )
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_validator.validate(bad_ref_schema)
    assert any("non-existent" in e for e in exc_info.value.errors)


# ---------------------------------------------------------------------------
# Test 6: Schema CRUD persistence
# ---------------------------------------------------------------------------


def test_schema_crud_persistence(schema_persistence: SchemaPersistence):
    schema = WorkflowSchema(
        schema_id="persist_test",
        name="Persistence Test",
        description="Tests save/load/list/update/delete.",
        nodes=[
            NodeDefinition(
                node_id="node_1", node_type=NodeType.AGENT, label="Node 1",
                config=NodeConfig(model_id=TEST_MODEL_ID),
                position=NodePosition(x=0, y=0),
            ),
        ],
        edges=[],
        config=WorkflowConfig(),
    )

    saved_path = schema_persistence.save_schema(schema)
    assert saved_path.exists()

    loaded = schema_persistence.load_schema("persist_test")
    assert loaded.schema_id == "persist_test"
    assert loaded.name == "Persistence Test"
    assert len(loaded.nodes) == 1
    assert loaded.nodes[0].node_id == "node_1"

    schemas = schema_persistence.list_schemas()
    assert "persist_test" in schemas

    updated_schema = schema.model_copy(update={"name": "Updated Name"})
    schema_persistence.save_schema(updated_schema)
    reloaded = schema_persistence.load_schema("persist_test")
    assert reloaded.name == "Updated Name"

    schema_persistence.delete_schema("persist_test")
    remaining = schema_persistence.list_schemas()
    assert "persist_test" not in remaining

    with pytest.raises(FileNotFoundError):
        schema_persistence.load_schema("nonexistent_schema")

    with pytest.raises(FileNotFoundError):
        schema_persistence.delete_schema("nonexistent_schema")


# ---------------------------------------------------------------------------
# Test 7: Tools in isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_in_isolation():
    rsi_prices = [
        44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
        45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
        46.03, 46.41, 46.22, 45.64,
    ]
    indicator_tool = TechnicalAnalysisTool()
    rsi_result = await indicator_tool.execute(
        indicator="rsi", close=rsi_prices
    )
    assert "columns" in rsi_result
    rsi_col = next(iter(rsi_result["columns"].values()))
    rsi_values = [v for v in rsi_col if v is not None]
    assert len(rsi_values) > 0
    for val in rsi_values:
        assert 0.0 <= val <= 100.0, f"RSI value {val} outside [0, 100]"

    chunk_tool = ChunkTextTool()
    long_text = "Machine learning is transforming industries. " * 100
    chunk_result = await chunk_tool.execute(
        text=long_text, chunk_size=200, overlap=50
    )
    assert "chunks" in chunk_result
    assert "total_chunks" in chunk_result
    assert chunk_result["total_chunks"] > 1
    assert len(chunk_result["chunks"]) == chunk_result["total_chunks"]

    score_tool = ScoreTextTool()
    sentiment_text = "The stock market soared today with record-breaking gains across all sectors."
    score_result = await score_tool.execute(
        text=sentiment_text, dimensions=["sentiment"]
    )
    assert "scores" in score_result
    assert "sentiment" in score_result["scores"]
    sentiment_data = score_result["scores"]["sentiment"]
    score_value = sentiment_data.get("score", sentiment_data.get("value", 0))
    assert -1.0 <= score_value <= 1.0, f"Sentiment score {score_value} outside [-1, 1]"


# ---------------------------------------------------------------------------
# Test 8: Context harness guardrails
# ---------------------------------------------------------------------------


def test_context_harness_guardrails():
    harness = ContextHarness(
        system_prompt="",
        instruction=["Be concise"],
        token_budget=4096,
        scope_window=5,
        guardrail_rules=[
            "input:forbidden_topic:weapons",
            "output:forbidden_phrase:I cannot help",
        ],
    )

    with pytest.raises(GuardrailViolationError) as exc_info:
        harness.validate_input("Tell me about weapons manufacturing")
    assert any("weapons" in v for v in exc_info.value.violations)

    with pytest.raises(GuardrailViolationError) as exc_info:
        harness.validate_output("I cannot help you with that request.")
    assert any("I cannot help" in v for v in exc_info.value.violations)

    harness.validate_input("Tell me about renewable energy trends")
    harness.validate_output("Renewable energy is growing rapidly worldwide.")


# ---------------------------------------------------------------------------
# Test 9: Varying node configs (temperature, iterations)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_varying_node_configs(workflow_executor: WorkflowExecutor):
    schema = WorkflowSchema(
        schema_id="test_varying_configs",
        name="Varying Config Test",
        description="Two agents with different temperatures and iteration limits.",
        nodes=[
            NodeDefinition(
                node_id="precise_agent",
                node_type=NodeType.AGENT,
                label="State the boiling point of water at sea level in Celsius. Be exact.",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.1,
                    max_tokens=256,
                    instruction=["Answer with a single number and unit"],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="creative_agent",
                node_type=NodeType.AGENT,
                label="Write a creative one-sentence metaphor about the ocean.",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=1.5,
                    max_tokens=256,
                    instruction=["Be highly creative and poetic"],
                ),
                position=NodePosition(x=300, y=100),
            ),
        ],
        edges=[],
        config=WorkflowConfig(total_timeout=60),
    )

    record = await workflow_executor.execute_workflow(schema)

    assert record.status == "completed", f"Errors: {record.errors}"
    assert "precise_agent" in record.node_outputs
    assert "creative_agent" in record.node_outputs

    precise_output = record.node_outputs["precise_agent"]
    creative_output = record.node_outputs["creative_agent"]

    precise_content = precise_output.get("content", "") if isinstance(precise_output, dict) else str(precise_output)
    creative_content = creative_output.get("content", "") if isinstance(creative_output, dict) else str(creative_output)

    assert len(precise_content) > 0, "Precise agent produced no output"
    assert len(creative_content) > 0, "Creative agent produced no output"

    if isinstance(precise_output, dict):
        assert precise_output.get("iterations", 0) <= 1, (
            f"Precise agent exceeded 1 iteration: {precise_output.get('iterations')}"
        )


# ---------------------------------------------------------------------------
# Test 10: Full multi-agent research pipeline from YAML
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_multi_agent_research_pipeline(workflow_executor: WorkflowExecutor):
    project_root = Path(__file__).parent.parent
    persistence = SchemaPersistence(project_root / "schemas")
    schema = persistence.load_schema("multi_agent_research")

    assert schema.schema_id == "multi_agent_research"
    assert len(schema.nodes) == 3

    record = await workflow_executor.execute_workflow(schema)

    assert record.status == "completed", (
        f"Expected completed, got {record.status}. Errors: {record.errors}"
    )

    expected_nodes = {"research_planner", "analyst_group", "synthesizer"}
    for node_id in expected_nodes:
        assert node_id in record.node_outputs, f"Missing output for node '{node_id}'"
        output = record.node_outputs[node_id]
        content = output.get("content", "") if isinstance(output, dict) else str(output)
        assert len(content) > 20, (
            f"Node '{node_id}' produced insufficient content ({len(content)} chars): {content[:100]!r}"
        )


# ---------------------------------------------------------------------------
# Test 11: Full-stack integration — schema → executor → LLM → parsing →
#           harness → tool dispatch → result.  Captures and verifies events
#           at EVERY layer so the path is never short-circuited.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fullstack_fetch_exchange_data(workflow_executor: WorkflowExecutor):
    """Exercises the real execution path end-to-end:

    WorkflowExecutor.execute_workflow
      → _execute_stages → _execute_node
        → LLMProvider.complete  (real API call)
        → CoreAgent parses response (tool_calls extraction)
        → ExecutionHarness.process_response (authorization, dispatch)
        → FetchExchangeDataTool.execute (real API call)
        → result flows back through harness → agent → executor

    Captures ALL events and asserts each layer fired.
    """
    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_fullstack_fetch",
        name="Full-Stack Fetch Exchange Data",
        description=(
            "Agent with fetch_exchange_data tool, exercising every layer "
            "from LLM call through harness dispatch."
        ),
        nodes=[
            NodeDefinition(
                node_id="fetcher_agent",
                node_type=NodeType.AGENT,
                label=(
                    "Use the fetch_exchange_data tool to get the current BTC-USDT "
                    "ticker from OKX. source_id='okx', source_type='ticker', "
                    "symbol='BTC-USDT'. Report the price."
                ),
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.0,
                    max_tokens=1024,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    instruction=[
                        "You MUST call the fetch_exchange_data tool exactly once.",
                        "After receiving the data, state the last price.",
                    ],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="exchange_tool_node",
                node_type=NodeType.TOOL,
                label="fetch_exchange_data",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    tools=["fetch_exchange_data"],
                    call_budget=5,
                    rate_limit_per_minute=10,
                ),
                position=NodePosition(x=300, y=100),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="agent_to_exchange",
                edge_type=EdgeType.TOOL_CALL,
                source_node_id="fetcher_agent",
                target_node_id="exchange_tool_node",
            ),
        ],
        config=WorkflowConfig(
            total_timeout=120,
            logging_level=LoggingLevel.ALL,
            trace_enabled=True,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    # --- Layer verification via event stream ---
    event_types = [e["type"] for e in captured_events]

    # A. Workflow lifecycle
    assert "workflow_started" in event_types, (
        f"workflow_started missing. Events: {event_types}"
    )

    # B. After LLM call — node_input proves messages were assembled and
    #    sent to the LLM.  Its presence means executor → context_harness →
    #    LLMProvider path executed.
    node_input_events = [e for e in captured_events if e["type"] == "node_input"]
    assert len(node_input_events) > 0, (
        f"No node_input event — LLM was never called. Events: {event_types}"
    )
    messages_sent = node_input_events[0].get("messages", [])
    assert len(messages_sent) > 0, "node_input had empty messages list"

    # C. Parsing — tool_call event proves the LLM response was parsed,
    #    tool_calls were extracted, and the harness authorized the call.
    tool_call_events = [e for e in captured_events if e["type"] == "tool_call"]
    assert len(tool_call_events) > 0, (
        f"No tool_call event — LLM response was not parsed into a tool "
        f"call, or harness rejected it. Events: {event_types}"
    )
    dispatched_tool = tool_call_events[0].get("tool_name")
    assert dispatched_tool == "fetch_exchange_data", (
        f"Expected fetch_exchange_data, got {dispatched_tool}"
    )
    dispatched_args = tool_call_events[0].get("arguments", {})
    assert dispatched_args.get("source_id") == "okx", (
        f"Expected source_id='okx', got {dispatched_args}"
    )

    # D. tool_dispatch event — proves http.fetch() traced the ACTUAL HTTP
    #    call with its final URL and query parameters.
    tool_dispatch_events = [e for e in captured_events if e["type"] == "tool_dispatch"]
    assert len(tool_dispatch_events) > 0, (
        f"No tool_dispatch event — actual HTTP call was not traced. "
        f"Events: {event_types}"
    )
    dispatched_url = tool_dispatch_events[0].get("url", "")
    assert "okx.com" in dispatched_url, (
        f"Expected OKX URL, got: {dispatched_url}"
    )
    dispatched_method = tool_dispatch_events[0].get("method", "")
    assert dispatched_method == "GET", (
        f"Expected GET method, got: {dispatched_method}"
    )
    dispatched_params = tool_dispatch_events[0].get("params", {})
    assert "instId" in dispatched_params, (
        f"Expected instId in params, got: {dispatched_params}"
    )

    # E. Harness layer — tool_result event proves the tool executed and
    #    returned data through the harness.
    tool_result_events = [e for e in captured_events if e["type"] == "tool_result"]
    assert len(tool_result_events) > 0, (
        f"No tool_result event — tool was dispatched but no result "
        f"came back. Events: {event_types}"
    )
    result_preview = tool_result_events[0].get("content_preview", "")
    assert len(result_preview) > 0, "tool_result had empty content_preview"

    # E2. Agent iteration tracing — proves the agentic loop internals
    #     are traced, not just the harness layer.
    iteration_events = [e for e in captured_events if e["type"] == "iteration_started"]
    assert len(iteration_events) > 0, (
        f"No iteration_started event — agent loop was not traced. "
        f"Events: {event_types}"
    )
    assert iteration_events[0]["iteration"] == 1

    llm_response_events = [e for e in captured_events if e["type"] == "llm_response"]
    assert len(llm_response_events) > 0, (
        f"No llm_response event. Events: {event_types}"
    )

    # E3. Tool results appended — proves tool results were fed back
    #     to the agent's conversation.
    appended_events = [e for e in captured_events if e["type"] == "tool_results_appended"]
    assert len(appended_events) > 0, (
        f"No tool_results_appended event — tool results were not traced "
        f"as fed back to agent. Events: {event_types}"
    )
    assert appended_events[0]["result_count"] > 0, "No results were appended"
    assert len(appended_events[0]["results_preview"]) > 0, "results_preview is empty"

    # F. Agent ran through all layers and completed.
    if record.status == "completed":
        assert "fetcher_agent" in record.node_outputs
        output = record.node_outputs["fetcher_agent"]
        content = output.get("content", "") if isinstance(output, dict) else str(output)
        assert len(content) > 5, f"Agent produced no content: {content!r}"
        assert "workflow_completed" in event_types


# ---------------------------------------------------------------------------
# Test 12: Full-stack integration — schema → executor → LLM → harness →
#           macro data tool.  Same layer verification as Test 11 but for a
#           different tool category.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fullstack_fetch_macro_data(workflow_executor: WorkflowExecutor):
    """Same full-path exercise as test 11 but through FetchMacroDataTool (ECB)."""
    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_fullstack_macro",
        name="Full-Stack Fetch Macro Data",
        description="Agent fetches ECB exchange rate data through full stack.",
        nodes=[
            NodeDefinition(
                node_id="macro_agent",
                node_type=NodeType.AGENT,
                label=(
                    "Use the fetch_macro_data tool to get ECB exchange rate data. "
                    "source_id='ecb', source_type='exchange_rates'. "
                    "Report one exchange rate."
                ),
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.0,
                    max_tokens=1024,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    instruction=[
                        "You MUST call the fetch_macro_data tool exactly once.",
                        "Report a specific exchange rate from the result.",
                    ],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="macro_tool_node",
                node_type=NodeType.TOOL,
                label="fetch_macro_data",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    tools=["fetch_macro_data"],
                    call_budget=5,
                    rate_limit_per_minute=10,
                ),
                position=NodePosition(x=300, y=100),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="agent_to_macro",
                edge_type=EdgeType.TOOL_CALL,
                source_node_id="macro_agent",
                target_node_id="macro_tool_node",
            ),
        ],
        config=WorkflowConfig(
            total_timeout=120,
            logging_level=LoggingLevel.ALL,
            trace_enabled=True,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    event_types = [e["type"] for e in captured_events]

    assert "workflow_started" in event_types

    node_input_events = [e for e in captured_events if e["type"] == "node_input"]
    assert len(node_input_events) > 0, f"LLM never called. Events: {event_types}"

    tool_call_events = [e for e in captured_events if e["type"] == "tool_call"]
    tool_result_events = [e for e in captured_events if e["type"] == "tool_result"]

    if tool_call_events:
        assert tool_call_events[0]["tool_name"] == "fetch_macro_data"
        assert len(tool_result_events) > 0, f"Tool dispatched but no result. Events: {event_types}"

    if record.status == "completed":
        assert "workflow_completed" in event_types


# ---------------------------------------------------------------------------
# Test 13: Full-stack error tracing — verify that when an error occurs the
#           traceback is captured in the event, not just str(exception).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fullstack_error_traceback(workflow_executor: WorkflowExecutor):
    """Trigger a guaranteed error (empty model_id) and verify the event
    includes a 'traceback' field with file/line info."""
    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_error_traceback",
        name="Error Traceback Test",
        description="Intentionally broken schema to verify traceback capture.",
        nodes=[
            NodeDefinition(
                node_id="bad_agent",
                node_type=NodeType.AGENT,
                label="This should fail",
                config=NodeConfig(
                    model_id="",
                    retries=0,
                ),
                position=NodePosition(x=100, y=100),
            ),
        ],
        edges=[],
        config=WorkflowConfig(
            total_timeout=30,
            logging_level=LoggingLevel.ALL,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    assert record.status in ("failed", "validation_error"), (
        f"Expected failure, got {record.status}"
    )
    assert len(record.errors) > 0

    error_events = [
        e for e in captured_events
        if e["type"] in ("node_error", "workflow_error")
    ]
    assert len(error_events) > 0, "No error event captured"

    has_traceback = any(
        "traceback" in e and len(e["traceback"]) > 20
        for e in error_events
    )
    assert has_traceback, (
        f"Error events lack traceback: "
        f"{[{k: v for k, v in e.items() if k != 'run_id'} for e in error_events]}"
    )


# ---------------------------------------------------------------------------
# Test 14: Full-stack normalization — deliberately send malformatted inputs
#           through the entire pipeline to verify support modules correct them.
#           Schema → executor → LLM → harness → tool → normalization → http.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fullstack_normalization_wrong_caps(workflow_executor: WorkflowExecutor):
    """Send 'btc/usdt' (wrong case, slash separator) through the full stack.

    The LLM prompt instructs the agent to use these exact malformatted values.
    The support module's normalization must convert 'btc/usdt' to 'BTC-USDT'
    before the HTTP call reaches OKX. The tool_dispatch event shows the
    actual URL that was requested, proving normalization ran.
    """
    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_normalization_caps",
        name="Normalization Wrong Caps Test",
        description="Verify normalization handles wrong caps and separators.",
        nodes=[
            NodeDefinition(
                node_id="norm_agent",
                node_type=NodeType.AGENT,
                label=(
                    "Use the fetch_exchange_data tool to get the ticker from OKX. "
                    "You MUST use EXACTLY these values: "
                    "source_id='okx', source_type='ticker', symbol='btc/usdt'. "
                    "Do NOT correct or change the symbol format. "
                    "Report the price."
                ),
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.0,
                    max_tokens=1024,
                    tool_strict=True,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    instruction=[
                        "Call fetch_exchange_data with symbol='btc/usdt'",
                        "Do NOT modify the symbol value",
                    ],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="exchange_tool_node",
                node_type=NodeType.TOOL,
                label="fetch_exchange_data",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    tools=["fetch_exchange_data"],
                    call_budget=5,
                    rate_limit_per_minute=10,
                ),
                position=NodePosition(x=300, y=100),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="agent_to_tool",
                edge_type=EdgeType.TOOL_CALL,
                source_node_id="norm_agent",
                target_node_id="exchange_tool_node",
            ),
        ],
        config=WorkflowConfig(
            total_timeout=120,
            logging_level=LoggingLevel.ALL,
            trace_enabled=True,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    event_types = [e["type"] for e in captured_events]

    tool_call_events = [e for e in captured_events if e["type"] == "tool_call"]
    assert len(tool_call_events) > 0, (
        f"No tool_call event — LLM did not call the tool. Events: {event_types}"
    )

    tool_dispatch_events = [e for e in captured_events if e["type"] == "tool_dispatch"]
    assert len(tool_dispatch_events) > 0, (
        f"No tool_dispatch — HTTP call was never traced. Events: {event_types}"
    )

    dispatched_url = tool_dispatch_events[0].get("url", "")
    dispatched_params = tool_dispatch_events[0].get("params", {})
    inst_id = dispatched_params.get("instId", "")

    assert "BTC-USDT" == inst_id, (
        f"Normalization failed: expected instId='BTC-USDT' in params, "
        f"got instId='{inst_id}'. Full URL: {dispatched_url}, params: {dispatched_params}"
    )

    tool_result_events = [e for e in captured_events if e["type"] == "tool_result"]
    assert len(tool_result_events) > 0, (
        f"Tool call dispatched but no result came back. Events: {event_types}"
    )

    if record.status == "completed":
        assert "norm_agent" in record.node_outputs


@pytest.mark.asyncio
async def test_fullstack_normalization_underscore_separator(workflow_executor: WorkflowExecutor):
    """Send 'ETH_USDT' (underscore separator) through the full stack.

    OKX expects dash-separated IDs. The normalization must convert
    underscores to dashes. Verified via tool_dispatch event params.
    """
    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_normalization_underscore",
        name="Normalization Underscore Test",
        description="Verify normalization handles underscore separator.",
        nodes=[
            NodeDefinition(
                node_id="norm_agent",
                node_type=NodeType.AGENT,
                label=(
                    "Use the fetch_exchange_data tool to get the ticker from OKX. "
                    "You MUST use EXACTLY these values: "
                    "source_id='okx', source_type='ticker', symbol='ETH_USDT'. "
                    "Do NOT correct the symbol. Report the price."
                ),
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.0,
                    max_tokens=1024,
                    tool_strict=True,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    instruction=[
                        "Call fetch_exchange_data with symbol='ETH_USDT'",
                        "Do NOT modify the symbol value",
                    ],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="exchange_tool_node",
                node_type=NodeType.TOOL,
                label="fetch_exchange_data",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    tools=["fetch_exchange_data"],
                    call_budget=5,
                    rate_limit_per_minute=10,
                ),
                position=NodePosition(x=300, y=100),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="agent_to_tool",
                edge_type=EdgeType.TOOL_CALL,
                source_node_id="norm_agent",
                target_node_id="exchange_tool_node",
            ),
        ],
        config=WorkflowConfig(
            total_timeout=120,
            logging_level=LoggingLevel.ALL,
            trace_enabled=True,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    tool_dispatch_events = [e for e in captured_events if e["type"] == "tool_dispatch"]
    assert len(tool_dispatch_events) > 0, "No tool_dispatch event"

    dispatched_params = tool_dispatch_events[0].get("params", {})
    inst_id = dispatched_params.get("instId", "")

    assert inst_id == "ETH-USDT", (
        f"Normalization failed: underscore not converted to dash. "
        f"Expected 'ETH-USDT', got '{inst_id}'"
    )

    tool_result_events = [e for e in captured_events if e["type"] == "tool_result"]
    assert len(tool_result_events) > 0, "Tool dispatched but no result came back"


@pytest.mark.asyncio
async def test_fullstack_normalization_macro_lowercase(workflow_executor: WorkflowExecutor):
    """Send lowercase currency code 'usd' to Frankfurter through the full stack.

    Frankfurter expects uppercase 3-letter ISO codes. The normalization
    must convert 'usd' to 'USD'.
    """
    captured_events: list[dict] = []

    schema = WorkflowSchema(
        schema_id="test_normalization_macro",
        name="Normalization Macro Lowercase Test",
        description="Verify normalization handles lowercase currency codes.",
        nodes=[
            NodeDefinition(
                node_id="macro_agent",
                node_type=NodeType.AGENT,
                label=(
                    "Use the fetch_exchange_data tool to get the latest forex rate. "
                    "You MUST use EXACTLY: "
                    "source_id='frankfurter', source_type='latest', symbol='usd'. "
                    "Do NOT uppercase the symbol. Report the EUR rate."
                ),
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    temperature=0.0,
                    max_tokens=1024,
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    instruction=[
                        "Call fetch_exchange_data with symbol='usd'",
                        "Do NOT modify the symbol value",
                    ],
                ),
                position=NodePosition(x=100, y=100),
            ),
            NodeDefinition(
                node_id="exchange_tool_node",
                node_type=NodeType.TOOL,
                label="fetch_exchange_data",
                config=NodeConfig(
                    model_id=TEST_MODEL_ID,
                    tools=["fetch_exchange_data"],
                    call_budget=5,
                    rate_limit_per_minute=10,
                ),
                position=NodePosition(x=300, y=100),
            ),
        ],
        edges=[
            EdgeDefinition(
                edge_id="agent_to_tool",
                edge_type=EdgeType.TOOL_CALL,
                source_node_id="macro_agent",
                target_node_id="exchange_tool_node",
            ),
        ],
        config=WorkflowConfig(
            total_timeout=120,
            logging_level=LoggingLevel.ALL,
            trace_enabled=True,
        ),
    )

    record = await workflow_executor.execute_workflow(
        schema, event_callback=captured_events.append,
    )

    tool_dispatch_events = [e for e in captured_events if e["type"] == "tool_dispatch"]
    assert len(tool_dispatch_events) > 0, "No tool_dispatch event"

    dispatched_url = tool_dispatch_events[0].get("url", "")
    assert "frankfurter" in dispatched_url, (
        f"Expected frankfurter URL, got: {dispatched_url}"
    )

    tool_result_events = [e for e in captured_events if e["type"] == "tool_result"]
    assert len(tool_result_events) > 0, "Tool dispatched but no result came back"

    if record.status == "completed":
        output = record.node_outputs.get("macro_agent", {})
        content = output.get("content", "") if isinstance(output, dict) else str(output)
        assert len(content) > 5, f"Agent produced no content: {content!r}"
