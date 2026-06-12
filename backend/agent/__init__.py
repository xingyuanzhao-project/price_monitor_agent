"""
Agent subpackage for core LLM agent execution.

What it does:
    Provides the LLM provider abstraction for making API calls and the
    core agent, which performs exactly one LLM turn with retries.  The
    agentic loop around the core agent (round-trips, tool dispatch,
    completion, pacing) is orchestration and lives in
    backend.orchestration.agent_loop — not here.

Entities in it:
    - llm_provider: LLMProvider class for async LLM API communication.
    - core: CoreAgent class performing a single LLM turn with retries.
    - format_adapter: provider-specific request kwarg normalization.
    - localhost_resolver: container-aware localhost URL rewriting.

How used by other modules:
    - backend.orchestration.executor instantiates LLMProvider from user
      settings and CoreAgent from NodeDefinition configurations, then
      drives the agent through backend.orchestration.agent_loop.AgentLoop.
    - backend.orchestration.group builds one CoreAgent per sub-agent.
    - Tool calls during execution are resolved via backend.tools.registry
      by the execution harness, never by the agent itself.
"""

from backend.agent.llm_provider import LLMProvider, LLMProviderError
from backend.agent.format_adapter import normalize_request_kwargs
from backend.agent.localhost_resolver import resolve_localhost_url
from backend.agent.core import CoreAgent
