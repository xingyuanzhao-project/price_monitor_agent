"""
Workflow executor for running multi-agent workflows end-to-end.

What it does:
    Takes a validated WorkflowSchema, builds a loop-aware execution plan
    via the scheduler, then walks through each stage in order.  A stage
    holds ExecutionUnits that run in parallel; a unit is a plain node, a
    synchronized group, or an explicit loop (self-loop edge or
    multi-node cycle) whose internal steps are repeated for
    ``max_loop_rounds`` rounds, with each round's outputs feeding the
    next through the ordinary upstream-data path.

    TOOL nodes never execute on their own: a TOOL node is bound to its
    callers via TOOL_CALL edges, and its output is the record of
    results it returned during the callers' LLM loops.  That output
    then flows through DATA_FLOW edges like any other node's.

    Per executable node the executor resolves an LLM provider, builds
    context and execution harnesses from NodeConfig fields with
    global_defaults as fallback, assembles messages, enforces
    input/output guardrails, injects upstream data, and then drives the
    node: an AGENT node runs as a CoreAgent inside an AgentLoop, an
    AGENT_GROUP node runs as an AgentGroup (whose sub-agents each run
    inside their own AgentLoop).  The full run is wrapped in a
    total-timeout guard and every significant event is streamed via an
    optional callback.

    WorkflowConfig fields are all applied at runtime:
        - total_timeout: enforced via asyncio.wait_for
        - logging_level: gates which events are emitted
        - trace_enabled: when False, node_output events are suppressed
        - max_loop_rounds: bounds every explicit loop's round count
        - max_iterations: agentic-loop ceiling passed to every AgentLoop
        - iteration_sleep: inter-iteration pacing passed to every AgentLoop

Entities in it:
    - RunRecord: dataclass capturing the full state of a single run.
    - WorkflowExecutor: stateful executor that owns run history.

How used by other modules:
    The API runs router calls ``execute_workflow`` to kick off runs and
    ``list_run_records`` / ``get_run_record`` to query history.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from backend.agent.core import CoreAgent
from backend.agent.llm_provider import LLMProvider
from backend.harness.context import ContextHarness
from backend.harness.execution import ExecutionHarness
from backend.orchestration.agent_loop import AgentExecutionError, AgentLoop
from backend.orchestration.group import AgentGroup
from backend.orchestration.scheduler import ExecutionPlan, ExecutionScheduler, ExecutionUnit
from backend.schema.models import LoggingLevel, NodeDefinition, NodeType, WorkflowSchema
from backend.schema.validation import SchemaValidator, SchemaValidationError
from backend.settings.models import (
    CLOUD_PROVIDER_API_BASE,
    LOCAL_PROVIDERS,
    UserSettings,
    resolve_provider_api_key,
    resolve_provider_base_url,
)
from backend.agent.localhost_resolver import resolve_localhost_url
from backend.state import NodeStatus, OrchestrationState, NodeState
from backend.tools.registry import ToolRegistry

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """Mutable record for a single workflow execution run.

    Contains two distinct concerns:
        - Record (log): ``events``, ``errors`` — append-only transcript.
        - State: ``state`` — live OrchestrationState tracking node statuses,
          outputs, and workflow lifecycle.  Agents can read (not write)
          this during execution.

    Attributes:
        run_id: Unique identifier for this run.
        schema_id: ID of the workflow schema being executed.
        schema_name: Human-readable name of the workflow schema.
        status: Current lifecycle status (``running``, ``completed``,
            ``failed``, ``validation_error``, ``timeout``).
        started_at: UTC timestamp of run start.
        completed_at: UTC timestamp of run completion (``None`` while running).
        state: Live orchestration state — per-node statuses and outputs.
        events: Ordered list of event dicts emitted during the run (record).
        errors: Ordered list of human-readable error messages (record).
    """

    run_id: str
    schema_id: str
    schema_name: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    state: OrchestrationState | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def node_outputs(self) -> dict[str, Any]:
        """Accessor for node outputs stored in the orchestration state."""
        if self.state is not None:
            return self.state.node_outputs
        return {}


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
        run_id: str | None = None,
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
            - max_loop_rounds: bounds every explicit loop's round count.

        Args:
            schema: The workflow schema to run.
            event_callback: Optional callable (sync or async) receiving event
                dicts.  Called for every significant lifecycle transition.
            run_id: Optional run identifier supplied by the caller. If None,
                a new UUID is generated.

        Returns:
            A RunRecord capturing the full run state.
        """
        if run_id is None:
            run_id = str(uuid.uuid4())
        orch_state = OrchestrationState(run_id=run_id)
        record = RunRecord(
            run_id=run_id,
            schema_id=schema.schema_id,
            schema_name=schema.name,
            status="running",
            started_at=datetime.now(timezone.utc),
            state=orch_state,
        )
        self._run_records[run_id] = record

        workflow_config = schema.config
        logging_level = workflow_config.logging_level
        trace_enabled = workflow_config.trace_enabled
        max_loop_rounds = workflow_config.max_loop_rounds
        max_iterations = workflow_config.max_iterations
        iteration_sleep = workflow_config.iteration_sleep

        async def emit(event: dict[str, Any]) -> None:
            """Append *event* to the record and relay to the callback."""
            event_type = event.get("type", "")

            if logging_level == LoggingLevel.NONE:
                return
            if logging_level != LoggingLevel.ALL:
                if logging_level == LoggingLevel.ERRORS and event_type not in {
                    "workflow_error", "node_error", "workflow_timeout"
                }:
                    return
                if logging_level == LoggingLevel.CRITICAL_INFO and event_type not in {
                    "workflow_started", "workflow_completed", "workflow_error",
                    "workflow_timeout", "node_error",
                }:
                    return

                if not trace_enabled and event_type in {
                    "node_output", "node_input", "tool_call", "tool_dispatch",
                    "tool_result", "iteration_started", "llm_response",
                    "tool_results_appended", "state_change",
                    "completion_check", "llm_retry",
                }:
                    return

            event["run_id"] = run_id
            record.events.append(event)
            if event_callback is not None:
                result = event_callback(event)
                if asyncio.iscoroutine(result):
                    await result

        try:
            # 1. Validate schema
            _LOGGER.info("Validating schema '%s' (%d nodes, %d edges)",
                         schema.name, len(schema.nodes), len(schema.edges))
            try:
                self._validator.validate(schema)
            except SchemaValidationError as validation_error:
                _LOGGER.error("Schema validation failed: %s", validation_error)
                record.status = "validation_error"
                record.errors = list(validation_error.errors)
                record.completed_at = datetime.now(timezone.utc)
                await emit({"type": "workflow_error", "error": str(validation_error)})
                return record

            # 2. Build execution plan
            plan = self._scheduler.build_execution_plan(schema)
            _LOGGER.info("Execution plan built: %d stages", len(plan.stages))
            await emit({
                "type": "workflow_started",
                "stages": len(plan.stages),
            })

            # 3. Execute stages under total_timeout
            total_timeout = workflow_config.total_timeout
            try:
                await asyncio.wait_for(
                    self._execute_stages(
                        schema, plan, record, emit,
                        max_loop_rounds, max_iterations, iteration_sleep,
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
            _LOGGER.info("Workflow completed successfully")
            await emit({"type": "workflow_completed"})

        except Exception as unexpected_error:
            tb = traceback.format_exc()
            _LOGGER.error("Workflow failed: %s\n%s", unexpected_error, tb)
            record.status = "failed"
            record.errors.append(str(unexpected_error))
            await emit({"type": "workflow_error", "error": str(unexpected_error), "traceback": tb})

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
        max_loop_rounds: int,
        max_iterations: int,
        iteration_sleep: float,
    ) -> None:
        """Walk stages sequentially, running each stage's units in parallel.

        Args:
            schema: The workflow schema.
            plan: The loop-aware execution plan.
            record: The mutable run record.
            emit: Event emitter coroutine.
            max_loop_rounds: Round bound for explicit loop units.
            max_iterations: Agentic-loop ceiling, from the workflow config.
            iteration_sleep: Inter-iteration sleep, from the workflow config.
        """
        node_map = {node.node_id: node for node in schema.nodes}

        for stage_index, units in enumerate(plan.stages):
            executable_node_ids = [
                node_id
                for unit in units
                for node_id in unit.node_ids
                if node_map[node_id].node_type != NodeType.TOOL
            ]
            await emit({
                "type": "stage_started",
                "stage": stage_index,
                "nodes": executable_node_ids,
            })

            await asyncio.gather(*[
                self._execute_unit(
                    unit, node_map, plan, record, emit,
                    max_loop_rounds, max_iterations, iteration_sleep,
                )
                for unit in units
            ])

            await emit({"type": "stage_completed", "stage": stage_index})

    async def _execute_unit(
        self,
        unit: ExecutionUnit,
        node_map: dict[str, NodeDefinition],
        plan: ExecutionPlan,
        record: RunRecord,
        emit: Callable[[dict[str, Any]], Any],
        max_loop_rounds: int,
        max_iterations: int,
        iteration_sleep: float,
    ) -> None:
        """Execute one unit: its steps once, or repeatedly if it is a loop.

        Each round re-executes every step; outputs land in
        ``node_outputs``, so the next round's upstream resolution picks
        up the previous round's results (including back edges and
        self-loops) through the ordinary data-flow path.

        Args:
            unit: The unit to execute.
            node_map: All schema nodes keyed by node_id.
            plan: The execution plan.
            record: The mutable run record.
            emit: Event emitter coroutine.
            max_loop_rounds: Round bound when ``unit.is_loop``.
            max_iterations: Agentic-loop ceiling, from the workflow config.
            iteration_sleep: Inter-iteration sleep, from the workflow config.
        """
        rounds = max_loop_rounds if unit.is_loop else 1

        for round_index in range(rounds):
            if unit.is_loop:
                await emit({
                    "type": "loop_round_started",
                    "round": round_index + 1,
                    "total_rounds": rounds,
                    "nodes": unit.node_ids,
                })

            for step in unit.steps:
                executable = [
                    node_id for node_id in step
                    if node_map[node_id].node_type != NodeType.TOOL
                ]
                if not executable:
                    continue

                results = await asyncio.gather(*[
                    self._execute_node(
                        node=node_map[node_id],
                        plan=plan,
                        record=record,
                        node_map=node_map,
                        emit=emit,
                        max_iterations=max_iterations,
                        iteration_sleep=iteration_sleep,
                    )
                    for node_id in executable
                ], return_exceptions=True)

                for node_id, result in zip(executable, results):
                    if isinstance(result, BaseException):
                        tb = "".join(traceback.format_exception(
                            type(result), result, result.__traceback__))
                        record.errors.append(f"Node '{node_id}': {result}")
                        await emit({
                            "type": "node_error",
                            "node_id": node_id,
                            "error": str(result),
                            "traceback": tb,
                        })
                        raise result
                    record.state.node_outputs[node_id] = result

    async def _execute_node(
        self,
        node: NodeDefinition,
        plan: ExecutionPlan,
        record: RunRecord,
        node_map: dict[str, NodeDefinition],
        emit: Callable[[dict[str, Any]], Any],
        max_iterations: int,
        iteration_sleep: float,
    ) -> Any:
        """Execute a single node (AGENT, AGENT_GROUP, or TOOL).

        All NodeConfig fields are consumed directly:
            - user input prompt → agent receives instruction from user
            - few_shot_examples → ContextHarness few-shot examples
            - token_budget → ContextHarness token budget (global_defaults fallback)
            - scope_window → ContextHarness scope window (global_defaults fallback)
            - call_budget → ExecutionHarness call budget (global_defaults fallback)
            - rate_limit_per_minute → ExecutionHarness rate limit
            - tools → additional explicitly authorised tool names

        Upstream node outputs are collected as upstream_data and injected
        into the LLM prompt via the context harness's assemble_messages.

        Args:
            node: The node definition from the schema.
            plan: The execution plan (for data-flow / tool-binding look-ups).
            record: The mutable RunRecord (for node_outputs and run_id).
            node_map: All schema nodes keyed by node_id.
            emit: Event emitter coroutine.
            max_iterations: Agentic-loop ceiling, from the workflow config.
            iteration_sleep: Inter-iteration sleep, from the workflow config.

        Returns:
            The node's execution output.

        Raises:
            AgentExecutionError: If LLM provider resolution fails.
            GuardrailViolationError: If input or output validation fails.
        """
        orch_state = record.state
        node_outputs = orch_state.node_outputs
        run_id = record.run_id

        node_state = orch_state.register_node(
            node_id=node.node_id,
            task=node.label,
            max_iterations=max_iterations,
        )
        node_state.start()

        await emit({"type": "node_started", "node_id": node.node_id})
        await emit({
            "type": "state_change",
            "node_id": node.node_id,
            "new_status": "in_progress",
            "task": node.label,
        })
        _LOGGER.info("Node '%s' (%s) started — model=%s",
                     node.label, node.node_id, node.config.model_id)

        global_defaults = self.user_settings.global_defaults

        # -- resolve upstream data (data_flow sources) ----------------------
        upstream_data: dict[str, str] = {}
        for source_node_id in plan.data_flow_sources.get(node.node_id, []):
            if source_node_id in node_outputs:
                upstream_data[source_node_id] = str(node_outputs[source_node_id])

        # -- separate guardrail rules from user instruction ------------------
        guardrail_rules: list[str] = []
        instruction: list[str] = []
        for item in node.config.instruction:
            if item.startswith("GUARDRAIL:"):
                guardrail_rules.append(item.removeprefix("GUARDRAIL:").strip())
            else:
                instruction.append(item)

        # -- authorised tool names from TOOL_CALL edges only ----------------
        tool_node_ids = plan.tool_bindings.get(node.node_id, [])
        authorized_tool_names: set[str] = set()
        total_call_budget = 0
        strictest_rate_limit = 0
        tool_strict = True
        for tool_node_id in tool_node_ids:
            tool_node = node_map[tool_node_id]
            authorized_tool_names.update(tool_node.config.tools)
            total_call_budget += tool_node.config.call_budget
            strictest_rate_limit = (
                tool_node.config.rate_limit_per_minute
                if strictest_rate_limit == 0
                else min(strictest_rate_limit, tool_node.config.rate_limit_per_minute)
            )
            if not tool_node.config.tool_strict:
                tool_strict = False

        # -- resolve per-node config with global_defaults fallback ----------
        node_token_budget = (
            node.config.token_budget
            if node.config.token_budget != 32768
            else global_defaults.get("max_tokens", 32768)
        )
        node_scope_window = node.config.scope_window
        node_call_budget = total_call_budget if tool_node_ids else node.config.call_budget
        node_rate_limit = strictest_rate_limit if tool_node_ids else node.config.rate_limit_per_minute

        # -- build context harness with all NodeConfig HOW fields -----------
        context_harness = ContextHarness(
            system_prompt="",
            instruction=instruction,
            token_budget=node_token_budget,
            scope_window=node_scope_window,
            guardrail_rules=guardrail_rules,
            few_shot_examples=node.config.few_shot_examples,
        )

        # -- build execution harness with NodeConfig execution constraints --
        async def _harness_event_cb(event: dict[str, Any]) -> None:
            event["node_id"] = node.node_id
            await emit(event)

        execution_harness = ExecutionHarness(
            tool_registry=self.tool_registry,
            user_settings=self.user_settings,
            authorized_tools=authorized_tool_names,
            call_budget=node_call_budget,
            rate_limit_per_minute=node_rate_limit,
            run_id=run_id,
            event_callback=_harness_event_cb,
        )

        # -- assemble messages for the LLM ----------------------------------
        user_task = node.label
        messages = context_harness.assemble_messages(
            user_task=user_task,
            upstream_data=upstream_data,
            tool_results=[],
        )

        await emit({
            "type": "node_input",
            "node_id": node.node_id,
            "messages": messages,
        })

        # -- validate input against guardrails (MUST raise on violation) ----
        context_harness.validate_input(user_task)

        # -- execute the node -----------------------------------------------
        llm_provider = self._resolve_llm_provider(
            model_id=node.config.model_id,
            provider_name=node.config.provider,
            temperature=node.config.temperature,
            max_tokens=node.config.max_tokens,
        )

        tool_definitions = (
            execution_harness.get_tool_definitions(
                list(authorized_tool_names), strict=tool_strict,
            )
            if authorized_tool_names
            else None
        )

        async def agent_emit_event(event: dict[str, Any]) -> None:
            """Forward agent-layer events into the unified trace stream."""
            event.setdefault("node_id", node.node_id)
            await emit(event)

        if node.node_type == NodeType.AGENT_GROUP:
            group = AgentGroup(
                node_id=node.node_id,
                label=node.label,
                config=node.config,
                group_config=node.group_config,
                llm_provider=llm_provider,
                tool_call_handler=execution_harness.process_response,
                max_iterations=max_iterations,
                iteration_sleep=iteration_sleep,
                emit_event=agent_emit_event,
            )
            agent_result = await group.execute(messages, tools=tool_definitions)
        else:
            agent = CoreAgent(
                node_id=node.node_id,
                label=node.label,
                config=node.config,
                llm_provider=llm_provider,
                emit_event=agent_emit_event,
            )
            agent_loop = AgentLoop(
                agent=agent,
                tool_call_handler=execution_harness.process_response,
                node_state=node_state,
                termination_conditions=node.config.termination_conditions,
                max_iterations=max_iterations,
                iteration_sleep=iteration_sleep,
                emit_event=agent_emit_event,
            )
            agent_result = await agent_loop.execute(messages, tools=tool_definitions)

        output_text = agent_result.get("content", str(agent_result))
        result = agent_result

        # -- route tool results to bound TOOL nodes --------------------------
        # A TOOL node's output is the record of results it returned during
        # this caller's LLM loop; downstream nodes consume it through the
        # ordinary data-flow path.
        await self._publish_tool_node_outputs(
            tool_node_ids, node_map, execution_harness.calls_log,
            node_outputs, emit,
        )

        # -- validate output against guardrails (MUST raise on violation) ---
        context_harness.validate_output(output_text)

        if node_state.status != NodeStatus.COMPLETED:
            node_state.complete(summary=output_text[:500])

        _LOGGER.info("Node '%s' completed — output length: %d chars",
                     node.label, len(output_text))
        await emit({
            "type": "node_completed",
            "node_id": node.node_id,
            "output_length": len(output_text),
        })
        return result

    async def _publish_tool_node_outputs(
        self,
        tool_node_ids: list[str],
        node_map: dict[str, NodeDefinition],
        calls_log: list[dict[str, Any]],
        node_outputs: dict[str, Any],
        emit: Callable[[dict[str, Any]], Any],
    ) -> None:
        """Publish each bound TOOL node's output from the caller's call log.

        Every dispatched call is matched to the TOOL node(s) whose
        config binds that tool name.  The matched calls become the TOOL
        node's output in ``node_outputs``, ready for downstream
        DATA_FLOW consumers.

        Args:
            tool_node_ids: TOOL nodes bound to the caller via TOOL_CALL edges.
            node_map: All schema nodes keyed by node_id.
            calls_log: The caller harness's normalized call records.
            node_outputs: Live output map of the orchestration state.
            emit: Event emitter coroutine.
        """
        for tool_node_id in tool_node_ids:
            # Endpoints were validated before the plan was built; a miss
            # here is a structural bug and must raise.
            bound_names = set(node_map[tool_node_id].config.tools)
            entries = [c for c in calls_log if c["tool_name"] in bound_names]
            if not entries:
                continue
            node_outputs[tool_node_id] = "\n\n".join(
                f"{entry['tool_name']}({json.dumps(entry['arguments'], default=str)}):\n"
                f"{entry['content']}"
                for entry in entries
            )
            await emit({
                "type": "tool_node_output",
                "node_id": tool_node_id,
                "call_count": len(entries),
            })

    def _resolve_llm_provider(
        self,
        model_id: str,
        provider_name: str = "",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> LLMProvider:
        """Resolve an LLM provider by declared provider name, falling back to model lookup.

        Routes by provider identity first (the general path). If provider_name
        is empty, falls back to scanning available_models lists for backward
        compatibility with schemas that only declare model_id.

        Args:
            model_id: The model identifier to use.
            provider_name: Declared provider name from the node config.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.

        Returns:
            An LLMProvider instance configured for the provider and model.

        Raises:
            AgentExecutionError: If provider cannot be resolved or key is missing.
        """
        provider_config = None

        if provider_name:
            try:
                provider_config = self.user_settings.get_provider_by_name(provider_name)
            except KeyError:
                raise AgentExecutionError(
                    f"Provider '{provider_name}' is not configured in settings. "
                    f"Available: {[p.provider_name for p in self.user_settings.llm_providers]}",
                    node_id="executor",
                    iterations_completed=0,
                )
        else:
            for candidate in self.user_settings.llm_providers:
                if model_id in candidate.available_models:
                    provider_config = candidate
                    break

        if provider_config is None:
            raise AgentExecutionError(
                f"No LLM provider found for model '{model_id}'. "
                f"Available providers: "
                f"{[p.provider_name for p in self.user_settings.llm_providers]}",
                node_id="executor",
                iterations_completed=0,
            )

        api_key = resolve_provider_api_key(provider_config)
        if not api_key and not provider_config.is_local:
            raise AgentExecutionError(
                f"API key not configured for provider '{provider_config.provider_name}'. "
                f"Set the key via the Settings page or add it to the .env file.",
                node_id="executor",
                iterations_completed=0,
            )

        base_url = resolve_provider_base_url(provider_config)
        if not base_url:
            raise AgentExecutionError(
                f"No base URL configured for provider '{provider_config.provider_name}'. "
                f"Set the endpoint URL in the Settings page.",
                node_id="executor",
                iterations_completed=0,
            )

        if provider_config.is_local:
            base_url = resolve_localhost_url(base_url)

        return LLMProvider(
            api_key=api_key or "dummy",
            model_id=model_id,
            provider=provider_config.provider_name,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
        )
