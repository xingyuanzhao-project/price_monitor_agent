"""
Agent group orchestration for coordinating multiple sub-agents.

What it does:
    Implements the AgentGroup class which uses an LLM planner phase to
    decompose a task into sub-agent assignments, then executes those
    sub-agents according to the configured group structure.

    Four distinct execution structures:
        PARALLEL — all sub-agents execute concurrently (semaphore-bounded).
        SEQUENTIAL — sub-agents execute one after another, chaining context.
        PYRAMID — one lead agent executes first, workers run in parallel
                  using the lead's output as additional context.
        DEFAULT — the planner itself decides the structure; the planner
                  response includes a ``structure`` key choosing between
                  parallel, sequential, or pyramid.

    Tool authorization is enforced: sub-agents only receive tool definitions
    for tools listed in ``AgentGroupConfig.tool_authorization``.

    Group context is read at startup (injected into each sub-agent's prompt)
    and written after each sub-agent completes (updated with its output).

    Each sub-agent receives its own ContextHarness built from the planner's
    per-agent context instructions.

Entities in it:
    - AgentGroup: Orchestrator that plans and executes a group of sub-agents.

How used by other modules:
    - The orchestration engine creates AgentGroup instances for AGENT_GROUP
      nodes in the workflow schema.
    - Each sub-agent is instantiated as a CoreAgent during execution.
    - Tool calls from sub-agents are dispatched via the shared tool_call_handler.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Optional

from backend.agent.core import AgentExecutionError, CoreAgent

_LOGGER = logging.getLogger(__name__)
from backend.agent.llm_provider import LLMProvider
from backend.harness.context import ContextHarness
from backend.prompts import load_prompt_template
from backend.schema.models import AgentGroupConfig, GroupStructure, NodeConfig
from backend.state import GroupState


class AgentGroup:
    """Orchestrates a group of sub-agents executing a decomposed task.

    Description:
        Uses an internal LLM planner phase to break the task into sub-agent
        assignments, then executes according to ``group_config.group_structure``.
        Tool authorization is gated by ``group_config.tool_authorization``.
        Group context from ``group_config.shared_context`` is injected into each
        sub-agent and updated after each sub-agent completes.

    Attributes:
        node_id: Unique identifier for this group node.
        label: Human-readable label for this group.
        config: NodeConfig with LLM and execution settings for the planner.
        group_config: AgentGroupConfig with structure, concurrency, and
            authorization settings.
        llm_provider: LLMProvider for the planner and sub-agent LLM calls.
        tool_call_handler: Async callable for executing tool calls.
        emit_event: Optional async-compatible callback for trace events.

    Methods:
        execute: Plan and execute the agent group.
    """

    def __init__(
        self,
        node_id: str,
        label: str,
        config: NodeConfig,
        group_config: AgentGroupConfig,
        llm_provider: LLMProvider,
        tool_call_handler: Callable,
        emit_event: Optional[Callable] = None,
    ) -> None:
        """Initialize the AgentGroup with its configuration and dependencies.

        Description:
            Stores all parameters needed for planning and orchestrating
            sub-agents.

        Params:
            node_id (str): Unique group node identifier.
            label (str): Human-readable group label.
            config (NodeConfig): Configuration for the planner LLM calls.
            group_config (AgentGroupConfig): Group orchestration configuration.
            llm_provider (LLMProvider): Provider for LLM API calls.
            tool_call_handler (Callable): Async function to execute tool calls.
                Signature: async (assistant_message: dict) -> list[dict]
            emit_event (Optional[Callable]): Optional async-compatible callback
                for trace events.

        Returns:
            None
        """
        self.node_id = node_id
        self.label = label
        self.config = config
        self.group_config = group_config
        self.llm_provider = llm_provider
        self.tool_call_handler = tool_call_handler
        self.emit_event = emit_event
        # Loop bound and pacing for sub-agents, supplied by orchestration
        # in execute() from the workflow config (not per-node).
        self._max_iterations: int = 0
        self._iteration_sleep: float = 0.0

    async def execute(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        *,
        max_iterations: int,
        iteration_sleep: float,
    ) -> dict:
        """Plan and execute the agent group.

        Description:
            Runs the planner phase to decompose the task into sub-agent
            assignments, validates the plan, then executes sub-agents
            according to the configured group structure.  When the structure
            is DEFAULT the planner also decides the execution structure.

        Params:
            messages (list[dict]): Input conversation messages for the group.
            tools (Optional[list[dict]]): Full tool definitions available to
                sub-agents before authorization filtering.
            max_iterations (int): Agentic-loop ceiling for each sub-agent,
                from the workflow config.
            iteration_sleep (float): Inter-iteration sleep for each
                sub-agent, from the workflow config.

        Returns:
            dict: Result dictionary with keys:
                - content (str): Aggregated final content from sub-agents.
                - sub_agent_results (list[dict]): Individual results.
                - structure (str): The execution structure used.
                - agent_count (int): Number of sub-agents executed.

        Raises:
            AgentExecutionError: If planner returns invalid JSON, agent count
                violates bounds, or sub-agent execution fails.
        """
        self._max_iterations = max_iterations
        self._iteration_sleep = iteration_sleep
        authorized_tool_definitions = self._filter_authorized_tools(tools)

        group_context: dict[str, Any] = dict(self.group_config.shared_context)
        self._group_state = GroupState(
            group_node_id=self.node_id,
            context=dict(group_context),
        )

        structure = self.group_config.group_structure
        _LOGGER.info("[%s] AgentGroup starting — structure=%s", self.node_id, structure.value)
        if structure == GroupStructure.DEFAULT:
            sub_agent_plans, chosen_structure = await self._run_default_planner_phase(
                messages, group_context
            )
        else:
            sub_agent_plans = await self._run_planner_phase(
                messages, group_context, structure
            )
            chosen_structure = structure

        _LOGGER.info("[%s] Planner produced %d sub-agents (structure=%s)",
                     self.node_id, len(sub_agent_plans), chosen_structure.value)
        self._validate_agent_count(len(sub_agent_plans))

        if chosen_structure == GroupStructure.PARALLEL:
            sub_agent_results = await self._execute_parallel(
                sub_agent_plans, messages, authorized_tool_definitions, group_context
            )
        elif chosen_structure == GroupStructure.SEQUENTIAL:
            sub_agent_results = await self._execute_sequential(
                sub_agent_plans, messages, authorized_tool_definitions, group_context
            )
        elif chosen_structure == GroupStructure.PYRAMID:
            sub_agent_results = await self._execute_pyramid(
                sub_agent_plans, messages, authorized_tool_definitions, group_context
            )
        else:
            sub_agent_results = await self._execute_parallel(
                sub_agent_plans, messages, authorized_tool_definitions, group_context
            )

        aggregated_content = self._aggregate_results(sub_agent_results)

        return {
            "content": aggregated_content,
            "sub_agent_results": sub_agent_results,
            "structure": chosen_structure.value if hasattr(chosen_structure, "value") else str(chosen_structure),
            "agent_count": len(sub_agent_plans),
        }

    # -- planner phases -----------------------------------------------------

    async def _run_planner_phase(
        self,
        messages: list[dict],
        group_context: dict[str, Any],
        structure: GroupStructure,
    ) -> list[dict]:
        """Run the planner LLM call to decompose the task into sub-agent assignments.

        Description:
            Sends a planning prompt to the LLM requesting a JSON response with
            a sub_agents array.  Each entry contains agent_id, task, focus,
            and context_instructions for the sub-agent's ContextHarness.

        Params:
            messages (list[dict]): Input conversation context for planning.
            group_context (dict[str, Any]): Group context to inform planning.
            structure (GroupStructure): The execution structure to plan for.

        Returns:
            list[dict]: List of sub-agent plan dictionaries from the LLM.

        Raises:
            AgentExecutionError: If the LLM returns invalid JSON or missing structure.
        """
        context_summary = json.dumps(group_context, default=str) if group_context else "none"
        template = load_prompt_template("group_planner_structured.json")
        prompt_content = template.format(
            structure=structure.value,
            shared_state=context_summary,
            min_agents=self.group_config.min_agents,
            max_agents=self.group_config.max_agents,
        )
        planning_prompt = {"role": "system", "content": prompt_content}

        planner_messages = [planning_prompt] + list(messages)

        try:
            response = await self.llm_provider.complete(
                messages=planner_messages,
                tools=None,
                response_format={"type": "json_object"},
            )
        except Exception as planner_error:
            raise AgentExecutionError(
                f"Planner LLM call failed: {planner_error}",
                node_id=self.node_id,
                iterations_completed=0,
            ) from planner_error

        response_content = response["choices"][0]["message"].get("content", "")
        return self._parse_planner_response(response_content)

    async def _run_default_planner_phase(
        self,
        messages: list[dict],
        group_context: dict[str, Any],
    ) -> tuple[list[dict], GroupStructure]:
        """Run the DEFAULT planner that also decides the execution structure.

        Description:
            When group_structure is DEFAULT the planner decides not only the
            sub-agent assignments but also which execution structure to use
            (parallel, sequential, or pyramid).  The LLM returns a JSON object
            with both ``structure`` and ``sub_agents`` keys.

        Params:
            messages (list[dict]): Input conversation context for planning.
            group_context (dict[str, Any]): Group context to inform planning.

        Returns:
            tuple[list[dict], GroupStructure]: Sub-agent plans and the chosen
                execution structure.

        Raises:
            AgentExecutionError: If the LLM returns invalid JSON or invalid structure.
        """
        context_summary = json.dumps(group_context, default=str) if group_context else "none"
        template = load_prompt_template("group_planner_default.json")
        prompt_content = template.format(
            shared_state=context_summary,
            min_agents=self.group_config.min_agents,
            max_agents=self.group_config.max_agents,
        )
        planning_prompt = {"role": "system", "content": prompt_content}

        planner_messages = [planning_prompt] + list(messages)

        try:
            response = await self.llm_provider.complete(
                messages=planner_messages,
                tools=None,
                response_format={"type": "json_object"},
            )
        except Exception as planner_error:
            raise AgentExecutionError(
                f"DEFAULT planner LLM call failed: {planner_error}",
                node_id=self.node_id,
                iterations_completed=0,
            ) from planner_error

        response_content = response["choices"][0]["message"].get("content", "")

        try:
            parsed_plan = json.loads(response_content)
        except json.JSONDecodeError as parse_error:
            raise AgentExecutionError(
                f"DEFAULT planner returned invalid JSON: {parse_error}. "
                f"Raw response: {response_content[:500]}",
                node_id=self.node_id,
                iterations_completed=0,
            ) from parse_error

        structure_value = parsed_plan.get("structure", "parallel")
        structure_map = {
            "parallel": GroupStructure.PARALLEL,
            "sequential": GroupStructure.SEQUENTIAL,
            "pyramid": GroupStructure.PYRAMID,
        }
        if structure_value not in structure_map:
            raise AgentExecutionError(
                f"DEFAULT planner returned invalid structure '{structure_value}'. "
                f"Must be one of: {list(structure_map.keys())}",
                node_id=self.node_id,
                iterations_completed=0,
            )
        chosen_structure = structure_map[structure_value]

        if "sub_agents" not in parsed_plan:
            raise AgentExecutionError(
                f"DEFAULT planner response missing 'sub_agents' key. "
                f"Got keys: {list(parsed_plan.keys())}",
                node_id=self.node_id,
                iterations_completed=0,
            )

        sub_agents = parsed_plan["sub_agents"]
        if not isinstance(sub_agents, list):
            raise AgentExecutionError(
                f"DEFAULT planner 'sub_agents' must be a list, got {type(sub_agents).__name__}",
                node_id=self.node_id,
                iterations_completed=0,
            )

        return sub_agents, chosen_structure

    def _parse_planner_response(self, response_content: str) -> list[dict]:
        """Parse the planner LLM response into a list of sub-agent plan dicts.

        Description:
            Validates the JSON structure returned by the planner and extracts
            the sub_agents list.

        Params:
            response_content (str): Raw JSON string from the planner LLM call.

        Returns:
            list[dict]: Validated list of sub-agent plan dictionaries.

        Raises:
            AgentExecutionError: If JSON is invalid or structure is missing.
        """
        try:
            parsed_plan = json.loads(response_content)
        except json.JSONDecodeError as parse_error:
            raise AgentExecutionError(
                f"Planner returned invalid JSON: {parse_error}. "
                f"Raw response: {response_content[:500]}",
                node_id=self.node_id,
                iterations_completed=0,
            ) from parse_error

        if "sub_agents" not in parsed_plan:
            raise AgentExecutionError(
                f"Planner response missing 'sub_agents' key. "
                f"Got keys: {list(parsed_plan.keys())}",
                node_id=self.node_id,
                iterations_completed=0,
            )

        sub_agents = parsed_plan["sub_agents"]
        if not isinstance(sub_agents, list):
            raise AgentExecutionError(
                f"Planner 'sub_agents' must be a list, got {type(sub_agents).__name__}",
                node_id=self.node_id,
                iterations_completed=0,
            )

        return sub_agents

    # -- validation ---------------------------------------------------------

    def _validate_agent_count(self, agent_count: int) -> None:
        """Validate that the planned agent count is within configured bounds.

        Description:
            Checks that the number of planned sub-agents falls within
            [min_agents, max_agents] from the group configuration.

        Params:
            agent_count (int): Number of sub-agents planned.

        Returns:
            None

        Raises:
            AgentExecutionError: If agent count is outside the allowed range.
        """
        if agent_count < self.group_config.min_agents:
            raise AgentExecutionError(
                f"Planner created {agent_count} sub-agents but minimum is "
                f"{self.group_config.min_agents}",
                node_id=self.node_id,
                iterations_completed=0,
            )
        if agent_count > self.group_config.max_agents:
            raise AgentExecutionError(
                f"Planner created {agent_count} sub-agents but maximum is "
                f"{self.group_config.max_agents}",
                node_id=self.node_id,
                iterations_completed=0,
            )

    # -- execution strategies -----------------------------------------------

    async def _execute_parallel(
        self,
        sub_agent_plans: list[dict],
        messages: list[dict],
        authorized_tools: Optional[list[dict]],
        group_context: dict[str, Any],
    ) -> list[dict]:
        """Execute all sub-agents concurrently using asyncio.gather.

        Description:
            Creates and runs all sub-agents in parallel, bounded by
            max_parallel_agents using a semaphore.  Each sub-agent's output
            is written back to group_context upon completion.

        Params:
            sub_agent_plans (list[dict]): Plans for each sub-agent.
            messages (list[dict]): Base conversation context.
            authorized_tools (Optional[list[dict]]): Filtered tool definitions.
            group_context (dict[str, Any]): Mutable group context.

        Returns:
            list[dict]: Results from all sub-agents.
        """
        semaphore = asyncio.Semaphore(self.group_config.max_parallel_agents)

        async def run_with_semaphore(plan: dict, agent_index: int) -> dict:
            async with semaphore:
                result = await self._execute_single_sub_agent(
                    plan, agent_index, messages, authorized_tools, group_context
                )
                agent_id = plan.get("agent_id", f"{self.node_id}_sub_{agent_index}")
                group_context[agent_id] = result.get("content", "")
                return result

        tasks = [
            run_with_semaphore(plan, index)
            for index, plan in enumerate(sub_agent_plans)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _execute_sequential(
        self,
        sub_agent_plans: list[dict],
        messages: list[dict],
        authorized_tools: Optional[list[dict]],
        group_context: dict[str, Any],
    ) -> list[dict]:
        """Execute sub-agents one after another, chaining context.

        Description:
            Runs each sub-agent sequentially, passing the previous agent's
            output as additional context to the next agent.  Group context is
            updated after each agent completes.

        Params:
            sub_agent_plans (list[dict]): Plans for each sub-agent.
            messages (list[dict]): Base conversation context.
            authorized_tools (Optional[list[dict]]): Filtered tool definitions.
            group_context (dict[str, Any]): Mutable group context.

        Returns:
            list[dict]: Results from all sub-agents in execution order.
        """
        results: list[dict] = []
        accumulated_context = list(messages)

        for index, plan in enumerate(sub_agent_plans):
            result = await self._execute_single_sub_agent(
                plan, index, accumulated_context, authorized_tools, group_context
            )
            results.append(result)

            agent_id = plan.get("agent_id", f"{self.node_id}_sub_{index}")
            group_context[agent_id] = result.get("content", "")

            accumulated_context.append({
                "role": "assistant",
                "content": result.get("content", ""),
            })

        return results

    async def _execute_pyramid(
        self,
        sub_agent_plans: list[dict],
        messages: list[dict],
        authorized_tools: Optional[list[dict]],
        group_context: dict[str, Any],
    ) -> list[dict]:
        """Execute with a lead agent followed by parallel workers.

        Description:
            The first sub-agent acts as the lead (executed first). Its output
            is added to group context, then remaining agents execute in
            parallel using the enriched context.

        Params:
            sub_agent_plans (list[dict]): Plans for each sub-agent.
            messages (list[dict]): Base conversation context.
            authorized_tools (Optional[list[dict]]): Filtered tool definitions.
            group_context (dict[str, Any]): Mutable group context.

        Returns:
            list[dict]: Results from lead agent followed by parallel workers.
        """
        lead_plan = sub_agent_plans[0]
        worker_plans = sub_agent_plans[1:]

        lead_result = await self._execute_single_sub_agent(
            lead_plan, 0, messages, authorized_tools, group_context
        )
        lead_agent_id = lead_plan.get("agent_id", f"{self.node_id}_sub_0")
        group_context[lead_agent_id] = lead_result.get("content", "")

        enriched_messages = list(messages) + [{
            "role": "assistant",
            "content": f"Lead agent output: {lead_result.get('content', '')}",
        }]

        if worker_plans:
            semaphore = asyncio.Semaphore(self.group_config.max_parallel_agents)

            async def run_worker(plan: dict, worker_index: int) -> dict:
                async with semaphore:
                    result = await self._execute_single_sub_agent(
                        plan, worker_index + 1, enriched_messages, authorized_tools, group_context
                    )
                    worker_agent_id = plan.get("agent_id", f"{self.node_id}_sub_{worker_index + 1}")
                    group_context[worker_agent_id] = result.get("content", "")
                    return result

            worker_tasks = [
                run_worker(plan, index)
                for index, plan in enumerate(worker_plans)
            ]
            worker_results = await asyncio.gather(*worker_tasks)
            return [lead_result] + list(worker_results)

        return [lead_result]

    # -- sub-agent construction ---------------------------------------------

    async def _execute_single_sub_agent(
        self,
        plan: dict,
        agent_index: int,
        messages: list[dict],
        authorized_tools: Optional[list[dict]],
        group_context: dict[str, Any],
    ) -> dict:
        """Create and execute a single sub-agent from its plan.

        Description:
            Constructs a ContextHarness from the plan's context_instructions
            and group_context, then creates a CoreAgent and runs it.  The
            ContextHarness gives the sub-agent scoped context built by the
            planner rather than raw messages.

        Params:
            plan (dict): Sub-agent plan with 'agent_id', 'task', 'focus',
                and 'context_instructions'.
            agent_index (int): Index of this agent in the group.
            messages (list[dict]): Conversation context for this agent.
            authorized_tools (Optional[list[dict]]): Filtered tool definitions.
            group_context (dict[str, Any]): Current group context.

        Returns:
            dict: The sub-agent's execution result.
        """
        agent_id = plan.get("agent_id", f"{self.node_id}_sub_{agent_index}")
        task_description = plan.get("task", "")
        focus_area = plan.get("focus", "")
        context_instructions = plan.get(
            "context_instructions",
            f"Your assigned task: {task_description}. Focus: {focus_area}.",
        )

        sub_agent_harness = ContextHarness(
            system_prompt=context_instructions,
            instruction=self.config.instruction,
            token_budget=self.config.token_budget,
            scope_window=self.config.scope_window,
            guardrail_rules=[],
            few_shot_examples=self.config.few_shot_examples,
        )

        upstream_data: dict[str, str] = {
            source_id: content
            for source_id, content in group_context.items()
            if isinstance(content, str)
        }
        sub_agent_messages = sub_agent_harness.assemble_messages(
            user_task=task_description,
            upstream_data=upstream_data,
            tool_results=[],
        )

        sub_node_state = self._group_state.register_sub_agent(
            agent_id=agent_id,
            task=task_description,
            max_iterations=self._max_iterations,
        )
        sub_node_state.start()

        sub_agent = CoreAgent(
            node_id=agent_id,
            label=f"{self.label} — Sub-agent {agent_index}",
            config=self.config,
            llm_provider=self.llm_provider,
            tool_call_handler=self.tool_call_handler,
            emit_event=self.emit_event,
            node_state=sub_node_state,
        )

        result = await sub_agent.execute(
            sub_agent_messages, authorized_tools,
            max_iterations=self._max_iterations,
            iteration_sleep=self._iteration_sleep,
        )
        result["agent_id"] = agent_id
        result["task"] = task_description
        return result

    # -- authorization & aggregation ----------------------------------------

    def _filter_authorized_tools(
        self,
        tool_definitions: Optional[list[dict]],
    ) -> Optional[list[dict]]:
        """Filter tool definitions to only those in group_config.tool_authorization.

        Description:
            Only tools whose name appears in the authorization list are
            forwarded to sub-agents.  An empty authorization list means
            no tools are permitted (deny-all default).  To grant all tools,
            the caller must explicitly list them.

        Params:
            tool_definitions (Optional[list[dict]]): Full tool definition list
                from the executor.

        Returns:
            Optional[list[dict]]: Filtered tool definitions, or None if none
                remain after filtering or if no tools were supplied.
        """
        if not tool_definitions:
            return None

        authorized_names = self.group_config.tool_authorization
        if not authorized_names:
            return None

        filtered = [
            tool_def for tool_def in tool_definitions
            if tool_def.get("function", {}).get("name") in authorized_names
        ]
        return filtered if filtered else None

    def _aggregate_results(self, sub_agent_results: list[dict]) -> str:
        """Aggregate results from all sub-agents into a combined content string.

        Description:
            Joins the content from all sub-agent results with clear
            section separators indicating the source agent.

        Params:
            sub_agent_results (list[dict]): Results from all executed sub-agents.

        Returns:
            str: Combined content string from all sub-agents.
        """
        content_parts: list[str] = []
        for result in sub_agent_results:
            agent_id = result.get("agent_id", "unknown")
            content = result.get("content", "")
            if content:
                content_parts.append(f"[{agent_id}]: {content}")

        return "\n\n".join(content_parts)
