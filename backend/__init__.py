"""
Backend core package for the Price Monitor Agent system.

What it does:
    Provides the foundational layers for building, configuring, and executing
    multi-agent workflows that monitor financial markets, analyze data, and
    dispatch alerts based on configurable schemas.

Entities in it:
    - schema: Workflow schema definition, validation, and persistence (YAML-based).
    - settings: User configuration, API credentials, and LLM provider management.
    - tools: Extensible tool system for data acquisition, technical analysis,
      text analysis, alert dispatch, and output writing.
    - agent: LLM provider abstraction, core agent execution with agentic loops,
      and agent group orchestration (parallel, sequential, pyramid structures).

How used by other modules:
    The frontend and orchestration layers import from this package to define
    workflow schemas via the schema subpackage, configure providers and credentials
    via the settings subpackage, register and invoke tools via the tools subpackage,
    and execute agent workflows via the agent subpackage.
"""
