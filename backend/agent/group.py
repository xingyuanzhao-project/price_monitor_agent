"""
Agent group orchestration for coordinating multiple agents.

What it does:
    Implements the AgentGroup class which uses an LLM planner phase to decompose
    a task into sub-agent assignments, then executes those sub-agents according
    to the configured group structure (PARALLEL, SEQUENTIAL, PYRAMID, DEFAULT).

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
from typing import Any, Callable, Optional

from backend.agent.core import AgentExecutionError, CoreAgent
from backend.agent.llm_provider import LLMProvider
from backend.schema.models import AgentGroupConfig, GroupStructure, NodeConfig


class AgentGroup:
    """
    Orchestrates a group of sub-agents executing a decomposed task.

    Description:
        Uses a planner phase to break the task into sub-agent assignments via
        an LLM call that returns JSON. Then executes the sub-agents according
        to the group_config's structure: PARALLEL (concurrent gather), SEQUENTIAL
        (chained one after another), PYRAMID (lead agent + parallel workers),
        or DEFAULT (falls back to parallel).

    Attributes:
        node_id: Unique identifier for this group node.
        label: Human-readable label for this group.
        config: NodeConfig with LLM and execution settings for the planner.
        group_config: AgentGroupConfig with structure and concurrency settings.
        llm_provider: LLMProvider for the planner and sub-agent LLM calls.
        tool_call_handler: Async callable for executing tool calls.
        stream_callback: Optional async callable for streaming events.

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
        stream_callback: Optional[Callable] = None,
    ) -> None:
        """
        Initialize the AgentGroup with its configuration and dependencies.

        Description:
            Stores all parameters needed for planning and orchestrating sub-agents.

        Params:
            node_id (str): Unique group node identifier.
            label (str): Human-readable group label.
            config (NodeConfig): Configuration for the planner LLM calls.
            group_config (AgentGroupConfig): Group orchestration configuration.
            llm_provider (LLMProvider): Provider for LLM API calls.
            tool_call_handler (Callable): Async function to execute tool calls.
                Signature: async (tool_name: str, tool_args: dict) -> Any
            stream_callback (Optional[Callable]): Optional streaming callback.

        Returns:
            None
        """
        self.node_id = node_id
        self.label = label
        self.config = config
        self.group_config = group_config
        self.llm_provider = llm_provider
        self.tool_call_handler = tool_call_handler
        self.stream_callback = stream_callback

    async def execute(self, messages: list[dict], tools: Optional[list[dict]] = None) -> dict:
        """
        Plan and execute the agent group.

        Description:
            Runs the planner phase to decompose the task into sub-agent
            assignments, validates the plan, then executes sub-agents
            according to the configured group structure.

        Params:
            messages (list[dict]): Input conversation messages for the group.
            tools (Optional[list[dict]]): Tool definitions available to sub-agents.

        Returns:
            dict: Result dictionary with keys:
                - content (str): Aggregated final content from sub-agents.
                - sub_agent_results (list[dict]): Individual results from each sub-agent.
                - structure (str): The execution structure used.
                - agent_count (int): Number of sub-agents executed.

        Raises:
            AgentExecutionError: If planner returns invalid JSON, agent count
                violates bounds, or sub-agent execution fails.
        """
        sub_agent_plans = await self._run_planner_phase(messages)
        self._validate_agent_count(len(sub_agent_plans))

        structure = self.group_config.group_structure
        if structure == GroupStructure.PARALLEL or structure == GroupStructure.DEFAULT:
            sub_agent_results = await self._execute_parallel(sub_agent_plans, messages, tools)
        elif structure == GroupStructure.SEQUENTIAL:
            sub_agent_results = await self._execute_sequential(sub_agent_plans, messages, tools)
        elif structure == GroupStructure.PYRAMID:
            sub_agent_results = await self._execute_pyramid(sub_agent_plans, messages, tools)
        else:
            sub_agent_results = await self._execute_parallel(sub_agent_plans, messages, tools)

        aggregated_content = self._aggregate_results(sub_agent_results)

        return {
            "content": aggregated_content,
            "sub_agent_results": sub_agent_results,
            "structure": structure.value,
            "agent_count": len(sub_agent_plans),
        }

    async def _run_planner_phase(self, messages: list[dict]) -> list[dict]:
        """
        Run the planner LLM call to decompose the task into sub-agent assignments.

        Description:
            Sends a planning prompt to the LLM requesting a JSON response with
            sub_agents array. Parses and validates the response structure.

        Params:
            messages (list[dict]): Input conversation context for planning.

        Returns:
            list[dict]: List of sub-agent plan dictionaries from the LLM.

        Raises:
            AgentExecutionError: If the LLM returns invalid JSON or missing structure.
        """
        planning_prompt = {
            "role": "system",
            "content": (
                "You are a task planner. Decompose the given task into sub-agent assignments. "
                "Respond with ONLY valid JSON in this format: "
                '{"sub_agents": [{"agent_id": "unique_id", "task": "description", '
                '"focus": "specific focus area"}]}. '
                f"Create between {self.group_config.min_agents} and "
                f"{self.group_config.max_agents} sub-agents."
            ),
        }

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

    def _validate_agent_count(self, agent_count: int) -> None:
        """
        Validate that the planned agent count is within configured bounds.

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

    async def _execute_parallel(
        self, sub_agent_plans: list[dict], messages: list[dict], tools: Optional[list[dict]]
    ) -> list[dict]:
        """
        Execute all sub-agents concurrently using asyncio.gather.

        Description:
            Creates and runs all sub-agents in parallel, bounded by
            max_parallel_agents using a semaphore.

        Params:
            sub_agent_plans (list[dict]): Plans for each sub-agent.
            messages (list[dict]): Base conversation context.
            tools (Optional[list[dict]]): Tool definitions for sub-agents.

        Returns:
            list[dict]: Results from all sub-agents.
        """
        semaphore = asyncio.Semaphore(self.group_config.max_parallel_agents)

        async def run_with_semaphore(plan: dict, agent_index: int) -> dict:
            async with semaphore:
                return await self._execute_single_sub_agent(plan, agent_index, messages, tools)

        tasks = [
            run_with_semaphore(plan, index)
            for index, plan in enumerate(sub_agent_plans)
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _execute_sequential(
        self, sub_agent_plans: list[dict], messages: list[dict], tools: Optional[list[dict]]
    ) -> list[dict]:
        """
        Execute sub-agents one after another, chaining context.

        Description:
            Runs each sub-agent sequentially, passing the previous agent's
            output as additional context to the next agent.

        Params:
            sub_agent_plans (list[dict]): Plans for each sub-agent.
            messages (list[dict]): Base conversation context.
            tools (Optional[list[dict]]): Tool definitions for sub-agents.

        Returns:
            list[dict]: Results from all sub-agents in execution order.
        """
        results: list[dict] = []
        accumulated_context = list(messages)

        for index, plan in enumerate(sub_agent_plans):
            result = await self._execute_single_sub_agent(plan, index, accumulated_context, tools)
            results.append(result)

            accumulated_context.append({
                "role": "assistant",
                "content": result.get("content", ""),
            })

        return results

    async def _execute_pyramid(
        self, sub_agent_plans: list[dict], messages: list[dict], tools: Optional[list[dict]]
    ) -> list[dict]:
        """
        Execute with a lead agent followed by parallel workers.

        Description:
            The first sub-agent acts as the lead (executed first). Its output
            is added to context, then remaining agents execute in parallel.

        Params:
            sub_agent_plans (list[dict]): Plans for each sub-agent.
            messages (list[dict]): Base conversation context.
            tools (Optional[list[dict]]): Tool definitions for sub-agents.

        Returns:
            list[dict]: Results from lead agent followed by parallel workers.
        """
        lead_plan = sub_agent_plans[0]
        worker_plans = sub_agent_plans[1:]

        lead_result = await self._execute_single_sub_agent(lead_plan, 0, messages, tools)

        enriched_messages = list(messages) + [{
            "role": "assistant",
            "content": f"Lead agent output: {lead_result.get('content', '')}",
        }]

        if worker_plans:
            semaphore = asyncio.Semaphore(self.group_config.max_parallel_agents)

            async def run_worker(plan: dict, worker_index: int) -> dict:
                async with semaphore:
                    return await self._execute_single_sub_agent(
                        plan, worker_index + 1, enriched_messages, tools
                    )

            worker_tasks = [
                run_worker(plan, index)
                for index, plan in enumerate(worker_plans)
            ]
            worker_results = await asyncio.gather(*worker_tasks)
            return [lead_result] + list(worker_results)

        return [lead_result]

    async def _execute_single_sub_agent(
        self,
        plan: dict,
        agent_index: int,
        messages: list[dict],
        tools: Optional[list[dict]],
    ) -> dict:
        """
        Create and execute a single sub-agent from its plan.

        Description:
            Constructs a CoreAgent with the group's config and runs it with
            task-specific instructions from the plan.

        Params:
            plan (dict): Sub-agent plan with 'agent_id', 'task', 'focus'.
            agent_index (int): Index of this agent in the group.
            messages (list[dict]): Conversation context for this agent.
            tools (Optional[list[dict]]): Tool definitions.

        Returns:
            dict: The sub-agent's execution result.
        """
        agent_id = plan.get("agent_id", f"{self.node_id}_sub_{agent_index}")
        task_description = plan.get("task", "")
        focus_area = plan.get("focus", "")

        sub_agent_messages = list(messages) + [{
            "role": "system",
            "content": f"Your assigned task: {task_description}. Focus area: {focus_area}.",
        }]

        sub_agent = CoreAgent(
            node_id=agent_id,
            label=f"{self.label} - Sub-agent {agent_index}",
            config=self.config,
            llm_provider=self.llm_provider,
            tool_call_handler=self.tool_call_handler,
            stream_callback=self.stream_callback,
        )

        result = await sub_agent.execute(sub_agent_messages, tools)
        result["agent_id"] = agent_id
        result["task"] = task_description
        return result

    def _aggregate_results(self, sub_agent_results: list[dict]) -> str:
        """
        Aggregate results from all sub-agents into a combined content string.

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
