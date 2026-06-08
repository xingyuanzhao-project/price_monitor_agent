"""End-to-end tests for the price_monitor_agent workflow system."""

from pathlib import Path

import pytest

from tests.conftest import TEST_MODEL_ID
from backend.schema.models import (
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
                    max_iterations=max_iterations,
                    instruction=instruction or ["Respond concisely"],
                ),
                position=NodePosition(x=100, y=100),
            )
        ],
        edges=[],
        config=WorkflowConfig(total_timeout=60),
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
                    max_iterations=2,
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
                    max_iterations=2,
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
                    max_iterations=5,
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
                config=NodeConfig(model_id=TEST_MODEL_ID),
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
        config=WorkflowConfig(total_timeout=60),
    )

    record = await workflow_executor.execute_workflow(schema)

    assert record.status == "completed", f"Errors: {record.errors}"
    assert "analyst" in record.node_outputs
    output = record.node_outputs["analyst"]
    content = output.get("content", "") if isinstance(output, dict) else str(output)
    assert len(content) > 5, f"Agent produced no meaningful content: {content!r}"

    tool_calls_made = output.get("tool_calls_made", []) if isinstance(output, dict) else []
    assert len(tool_calls_made) > 0, "Agent did not make any tool calls"

    found_rsi_call = any(tc.get("tool_name") == "technical_analysis" for tc in tool_calls_made)
    assert found_rsi_call, f"No technical_analysis call found in: {tool_calls_made}"


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
                    max_iterations=3,
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
# Test 5: Schema validation errors
# ---------------------------------------------------------------------------


def test_schema_validation_errors(schema_validator: SchemaValidator):
    cycle_schema = WorkflowSchema(
        schema_id="cycle_test",
        name="Cycle Test",
        description="Schema with a cycle.",
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
    with pytest.raises(SchemaValidationError) as exc_info:
        schema_validator.validate(cycle_schema)
    assert any("ycle" in e for e in exc_info.value.errors)

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
                    max_iterations=1,
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
                    max_iterations=3,
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
