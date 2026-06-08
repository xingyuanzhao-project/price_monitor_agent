"""
Agent subpackage for LLM-powered agent execution and orchestration.

What it does:
    Provides the LLM provider abstraction for making API calls, the core agent
    execution loop with tool calling and termination logic, and the agent group
    orchestrator for coordinating multiple agents in parallel, sequential, or
    pyramid structures.

Entities in it:
    - llm_provider: LLMProvider class for async LLM API communication.
    - core: CoreAgent class implementing the agentic loop with retries and fallback.
    - group: AgentGroup class orchestrating multiple agents per GroupStructure.

How used by other modules:
    - The orchestration engine instantiates LLMProvider with settings from
      backend.settings and passes it to CoreAgent/AgentGroup constructors.
    - CoreAgent and AgentGroup are created from NodeDefinition configurations
      loaded from backend.schema.
    - Tool calls during execution are resolved via backend.tools.registry.
"""

from backend.agent.llm_provider import LLMProvider, LLMProviderError
from backend.agent.format_adapter import normalize_request_kwargs
from backend.agent.localhost_resolver import resolve_localhost_url
from backend.agent.core import CoreAgent, AgentExecutionError
from backend.agent.group import AgentGroup
