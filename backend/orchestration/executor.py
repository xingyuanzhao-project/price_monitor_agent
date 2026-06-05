"""
Workflow executor for running multi-agent workflows end-to-end.

What it does:
    Takes a validated WorkflowSchema, builds an execution plan via the
    scheduler, then walks through each stage in order — executing nodes
    within a stage in parallel via ``asyncio.gather``.  Per node it
    resolves an LLM provider, builds context and execution harnesses
    using NodeConfig fields with global_defaults as fallback, assembles
    messages, enforces input/output guardrails, injects upstream state,
    and invokes either a CoreAgent or an AgentGroup.  The full run is
    wrapped in a total-timeout guard (``asyncio.wait_for``) and every
    significant event is streamed via an optional callback.

    WorkflowConfig fields are all applied at runtime:
        - total_timeout: enforced via asyncio.wait_for
        - logging_level: gates which events are emitted
        - trace_enabled: when False, node_output events are suppressed
        - dead_loop_detection: when True, detects repeated node visits

Entities in it:
    - RunRecord: dataclass capturing the full state of a single run.
    - WorkflowExecutor: stateful executor that owns run history.

How used by other modules:
    The API runs router calls ``execute_workflow`` to kick off runs and
    ``list_run_records`` / ``get_run_record`` to query history.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from backend.agent.core import CoreAgent, AgentExecutionError
from backend.agent.group import AgentGroup
from backend.agent.llm_provider import LLMProvider
from backend.harness.context import ContextHarness
from backend.harness.execution import ExecutionHarness
from backend.orchestration.scheduler import ExecutionPlan, ExecutionScheduler
from backend.schema.models import LoggingLevel, NodeDefinition, NodeType, WorkflowSchema
from backend.schema.validation import SchemaValidator, SchemaValidationError
from backend.settings.models import UserSettings, resolve_provider_api_key
from backend.tools.registry import ToolRegistry

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Mutable record for a single workflow execution run.

    Attributes:
        run_id: Unique identifier for this run.
        schema_id: ID of the workflow schema being executed.
        schema_name: Human-readable name of the workflow schema.
        status: Current lifecycle status (``running``, ``completed``,
            ``failed``, ``validation_error``, ``timeout``).
        started_at: UTC timestamp of run start.
        completed_at: UTC timestamp of run completion (``None`` while running).
        node_outputs: Mapping of node ID → execution output.
        events: Ordered list of event dicts emitted during the run.
        errors: Ordered list of human-readable error messages.
    """

    run_id: str
    schema_id: str
    schema_name: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    node_outputs: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class WorkflowExecutor:
    """Stateful executor that runs workflow schemas and stores RunRecords.

    Attributes:
        tool_registry: Central tool registry shared across all nodes.
        user_settings: User-level settings (LLM providers, credentials).

    Methods:
        execute_workflow: Run a schema end-to-end with optional event streaming.
        get_run_record: Retrieve a single RunRecord by its run_id.
        list_run_records: Return all stored RunRecords.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        user_settings: UserSettings,
    ) -> None:
        """Initialise the executor.

        Args:
            tool_registry: Registry of all available tool instances.
            user_settings: User settings for LLM providers and API credentials.
        """
        self.tool_registry = tool_registry
        self.user_settings = user_settings
        self._scheduler = ExecutionScheduler()
        self._validator = SchemaValidator()
        self._run_records: dict[str, RunRecord] = {}

    # -- public API ---------------------------------------------------------

    async def execute_workflow(
        self,
        schema: WorkflowSchema,
        event_callback: Callable[[dict[str, Any]], Any] | None = None,
    ) -> RunRecord:
        """Execute *schema* as a full workflow run.

        Steps:
            1. Validate schema (→ ``validation_error`` on failure).
            2. Build execution plan.
            3. Walk stages; within each stage execute nodes in parallel.
            4. Apply ``total_timeout`` via ``asyncio.wait_for``.
            5. Stream events via *event_callback*.

        WorkflowConfig fields applied:
            - total_timeout: enforced as asyncio.wait_for timeout.
            - logging_level: gates event emission verbosity.
            - trace_enabled: suppresses node_output events when False.
            - dead_loop_detection: aborts on repeated node visits.

        Args:
            schema: The workflow schema to run.
            event_callback: Optional callable (sync or async) receiving event
                dicts.  Called for every significant lifecycle transition.

        Returns:
            A RunRecord capturing the full run state.
        """
        run_id = str(uuid.uuid4())
        record = RunRecord(
            run_id=run_id,
            schema_id=schema.schema_id,
            schema_name=schema.name,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        self._run_records[run_id] = record

        workflow_config = schema.config
        logging_level = workflow_config.logging_level
        trace_enabled = workflow_config.trace_enabled
        dead_loop_detection = workflow_config.dead_loop_detection

        async def emit(event: dict[str, Any]) -> None:
            """Append *event* to the record and relay to the callback."""
            event_type = event.get("type", "")

            if logging_level == LoggingLevel.NONE:
                return
            if logging_level == LoggingLevel.ERRORS and event_type not in {
                "workflow_error", "node_error", "workflow_timeout"
            }:
                return
            if logging_level == LoggingLevel.CRITICAL_INFO and event_type not in {
                "workflow_started", "workflow_completed", "workflow_error",
                "workflow_timeout", "node_error",
            }:
                return

            if not trace_enabled and event_type == "node_output":
                return

            event["run_id"] = run_id
            record.events.append(event)
            if event_callback is not None:
                result = event_callback(event)
                if asyncio.iscoroutine(result):
                    await result

        try:
            # 1. Validate schema
            try:
                self._validator.validate(schema)
            except SchemaValidationError as validation_error:
                record.status = "validation_error"
                record.errors = list(validation_error.errors)
                record.completed_at = datetime.now(timezone.utc)
                await emit({"type": "workflow_error", "error": str(validation_error)})
                return record

            # 2. Build execution plan
            plan = self._scheduler.build_execution_plan(schema)
            await emit({
                "type": "workflow_started",
                "stages": len(plan.stages),
            })

            # 3. Execute stages under total_timeout
            total_timeout = workflow_config.total_timeout
            try:
                await asyncio.wait_for(
                    self._execute_stages(
                        schema, plan, record, emit, dead_loop_detection
                    ),
                    timeout=total_timeout,
                )
            except asyncio.TimeoutError:
                record.status = "timeout"
                record.errors.append(
                    f"Workflow timed out after {total_timeout} seconds"
                )
                await emit({
                    "type": "workflow_timeout",
                    "timeout": total_timeout,
                })
                record.completed_at = datetime.now(timezone.utc)
                return record

            record.status = "completed"
            await emit({"type": "workflow_completed"})

        except Exception as unexpected_error:
            record.status = "failed"
            record.errors.append(str(unexpected_error))
            await emit({"type": "workflow_error", "error": str(unexpected_error)})

        record.completed_at = datetime.now(timezone.utc)
        return record

    def get_run_record(self, run_id: str) -> RunRecord:
        """Retrieve a single RunRecord.

        Args:
            run_id: Unique run identifier.

        Returns:
            The matching RunRecord.

        Raises:
            KeyError: If *run_id* is not found.
        """
        if run_id not in self._run_records:
            raise KeyError(f"Run record '{run_id}' not found")
        return self._run_records[run_id]

    def list_run_records(self) -> list[RunRecord]:
        """Return all stored RunRecords (newest first).

        Returns:
            List of RunRecord instances ordered by ``started_at`` descending.
        """
        return sorted(
            self._run_records.values(),
            key=lambda record: record.started_at,
            reverse=True,
        )

    # -- private helpers ----------------------------------------------------

    async def _execute_stages(
        self,
        schema: WorkflowSchema,
        plan: ExecutionPlan,
        record: RunRecord,
        emit: Callable[[dict[str, Any]], Any],
        dead_loop_detection: bool,
    ) -> None:
        """Walk stages sequentially, running nodes within each stage in parallel.

        Args:
            schema: The workflow schema.
            plan: The execution plan with stages and bindings.
            record: The mutable run record.
            emit: Event emitter coroutine.
            dead_loop_detection: Whether to detect repeated node visits.
        """
        node_map = {node.node_id: node for node in schema.nodes}
        visited_node_ids: set[str] = set()

        for stage_index, stage_node_ids in enumerate(plan.stages):
            executable_node_ids = [
                node_id for node_id in stage_node_ids
                if node_map[node_id].node_type != NodeType.TOOL
            ]

            if dead_loop_detection:
                repeated_node_ids = [
                    node_id for node_id in executable_node_ids
                    if node_id in visited_node_ids
                ]
                if repeated_node_ids:
                    raise AgentExecutionError(
                        f"Dead loop detected: nodes {repeated_node_ids} "
                        "visited more than once during execution",
                        node_id=repeated_node_ids[0],
                        iterations_completed=stage_index,
                    )
            visited_node_ids.update(executable_node_ids)

            await emit({
                "type": "stage_started",
                "stage": stage_index,
                "nodes": executable_node_ids,
            })

            if not executable_node_ids:
                await emit({"type": "stage_completed", "stage": stage_index})
                continue

            tasks = [
                self._execute_node(
                    node=node_map[node_id],
                    plan=plan,
                    node_outputs=record.node_outputs,
                    node_map=node_map,
                    emit=emit,
                )
                for node_id in executable_node_ids
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for node_id, result in zip(executable_node_ids, results):
                if isinstance(result, BaseException):
                    record.errors.append(f"Node '{node_id}': {result}")
                    await emit({
                        "type": "node_error",
                        "node_id": node_id,
                        "error": str(result),
                    })
                    raise result
                record.node_outputs[node_id] = result

            await emit({"type": "stage_completed", "stage": stage_index})

    async def _execute_node(
        self,
        node: NodeDefinition,
        plan: ExecutionPlan,
        node_outputs: dict[str, Any],
        node_map: dict[str, NodeDefinition],
        emit: Callable[[dict[str, Any]], Any],
    ) -> Any:
        """Execute a single node (AGENT, AGENT_GROUP, or TOOL).

        All NodeConfig fields are consumed directly:
            - system_prompt → ContextHarness system prompt
            - few_shot_examples → ContextHarness few-shot examples
            - token_budget → ContextHarness token budget (global_defaults fallback)
            - scope_window → ContextHarness scope window (global_defaults fallback)
            - call_budget → ExecutionHarness call budget (global_defaults fallback)
            - rate_limit_per_minute → ExecutionHarness rate limit
            - tools → additional explicitly authorised tool names

        Upstream node outputs are collected into state and merged into the
        ContextHarness via merge_upstream_state so downstream nodes can
        access them through the harness's state dict.

        Args:
            node: The node definition from the schema.
            plan: The execution plan (for data-flow / tool-binding look-ups).
            node_outputs: Already-computed outputs keyed by node ID.
            node_map: All schema nodes keyed by node_id.
            emit: Event emitter coroutine.

        Returns:
            The node's execution output.

        Raises:
            AgentExecutionError: If LLM provider resolution fails.
            GuardrailViolationError: If input or output validation fails.
        """
        await emit({"type": "node_started", "node_id": node.node_id})

        global_defaults = self.user_settings.global_defaults

        # -- resolve upstream data (data_flow sources) ----------------------
        upstream_data: dict[str, str] = {}
        for source_node_id in plan.data_flow_sources.get(node.node_id, []):
            if source_node_id in node_outputs:
                upstream_data[source_node_id] = str(node_outputs[source_node_id])

        # -- build upstream state from all completed node outputs -----------
        upstream_state: dict[str, Any] = {
            source_node_id: node_outputs[source_node_id]
            for source_node_id in plan.data_flow_sources.get(node.node_id, [])
            if source_node_id in node_outputs
        }

        # -- separate guardrail rules from behavioural agent rules ----------
        guardrail_rules: list[str] = []
        agent_rules: list[str] = []
        for rule in node.config.agent_rules:
            if rule.startswith("GUARDRAIL:"):
                guardrail_rules.append(rule.removeprefix("GUARDRAIL:").strip())
            else:
                agent_rules.append(rule)

        # -- authorised tool names from TOOL_CALL edges + node.config.tools -
        tool_node_ids = plan.tool_bindings.get(node.node_id, [])
        authorized_tool_names: set[str] = set()
        for tool_node_id in tool_node_ids:
            if tool_node_id in node_map:
                authorized_tool_names.add(node_map[tool_node_id].label)
        for explicit_tool_name in node.config.tools:
            authorized_tool_names.add(explicit_tool_name)

        if node.node_type == NodeType.TOOL:
            authorized_tool_names.add(node.label)

        # -- resolve per-node config with global_defaults fallback ----------
        node_token_budget = (
            node.config.token_budget
            if node.config.token_budget != 32768
            else global_defaults.get("max_tokens", 32768)
        )
        node_scope_window = node.config.scope_window
        node_call_budget = node.config.call_budget
        node_rate_limit = node.config.rate_limit_per_minute

        # -- build context harness with all NodeConfig HOW fields -----------
        context_harness = ContextHarness(
            system_prompt=node.config.system_prompt,
            agent_rules=agent_rules,
            token_budget=node_token_budget,
            scope_window=node_scope_window,
            guardrail_rules=guardrail_rules,
            state=upstream_state,
            few_shot_examples=node.config.few_shot_examples,
        )
        context_harness.merge_upstream_state(upstream_state)

        # -- build execution harness with NodeConfig execution constraints --
        execution_harness = ExecutionHarness(
            tool_registry=self.tool_registry,
            user_settings=self.user_settings,
            authorized_tools=authorized_tool_names,
            call_budget=node_call_budget,
            rate_limit_per_minute=node_rate_limit,
        )

        # -- assemble messages for the LLM ----------------------------------
        user_task = node.label
        messages = context_harness.assemble_messages(
            user_task=user_task,
            upstream_data=upstream_data,
            tool_results=[],
        )

        # -- validate input against guardrails (MUST raise on violation) ----
        context_harness.validate_input(user_task)

        # -- execute the node -----------------------------------------------
        if node.node_type == NodeType.TOOL:
            result = await self._execute_tool_node(
                node, execution_harness, upstream_data
            )
            output_text = str(result)
        else:
            llm_provider = self._resolve_llm_provider(
                model_id=node.config.model_id,
                temperature=node.config.temperature,
                max_tokens=node.config.max_tokens,
            )

            tool_definitions = (
                execution_harness.get_tool_definitions(
                    list(authorized_tool_names)
                )
                if authorized_tool_names
                else None
            )

            async def node_stream_callback(chunk: dict[str, Any]) -> None:
                """Forward streaming chunks as node_output events."""
                await emit({
                    "type": "node_output",
                    "node_id": node.node_id,
                    "chunk": chunk,
                })

            if node.node_type == NodeType.AGENT_GROUP:
                group = AgentGroup(
                    node_id=node.node_id,
                    label=node.label,
                    config=node.config,
                    group_config=node.group_config,
                    llm_provider=llm_provider,
                    tool_call_handler=execution_harness.handle_tool_call,
                    stream_callback=node_stream_callback,
                )
                agent_result = await group.execute(
                    messages, tools=tool_definitions
                )
            else:
                agent = CoreAgent(
                    node_id=node.node_id,
                    label=node.label,
                    config=node.config,
                    llm_provider=llm_provider,
                    tool_call_handler=execution_harness.handle_tool_call,
                    stream_callback=node_stream_callback,
                )
                agent_result = await agent.execute(
                    messages, tools=tool_definitions
                )

            output_text = agent_result.get("content", str(agent_result))
            result = agent_result

        # -- validate output against guardrails (MUST raise on violation) ---
        context_harness.validate_output(output_text)

        await emit({
            "type": "node_completed",
            "node_id": node.node_id,
            "output_length": len(output_text),
        })
        return result

    async def _execute_tool_node(
        self,
        node: NodeDefinition,
        execution_harness: ExecutionHarness,
        upstream_data: dict[str, str],
    ) -> Any:
        """Directly execute a TOOL-type node without LLM involvement.

        Args:
            node: The TOOL node definition.
            execution_harness: Pre-configured execution harness for this node.
            upstream_data: Data from upstream nodes.

        Returns:
            The tool's return value.
        """
        return await execution_harness.handle_tool_call(
            node.label, upstream_data
        )

    def _resolve_llm_provider(
        self,
        model_id: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMProvider:
        """Find the LLM provider whose available_models contains *model_id*.

        Args:
            model_id: The model identifier to look up.
            temperature: Sampling temperature for the provider.
            max_tokens: Maximum response tokens for the provider.

        Returns:
            An LLMProvider instance configured for *model_id*.

        Raises:
            AgentExecutionError: If no provider offers *model_id*.
        """
        for provider_config in self.user_settings.llm_providers:
            if model_id in provider_config.available_models:
                api_key = resolve_provider_api_key(provider_config)
                if not api_key:
                    raise AgentExecutionError(
                        f"Environment variable '{provider_config.api_key_env}' "
                        f"is not set for provider '{provider_config.provider_name}'. "
                        f"Populate it in the .env file or export it before running.",
                        node_id="executor",
                        iterations_completed=0,
                    )
                return LLMProvider(
                    api_key=api_key,
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    base_url=provider_config.base_url,
                )
        raise AgentExecutionError(
            f"No LLM provider found containing model '{model_id}'. "
            f"Available providers: "
            f"{[p.provider_name for p in self.user_settings.llm_providers]}",
            node_id="executor",
            iterations_completed=0,
        )
