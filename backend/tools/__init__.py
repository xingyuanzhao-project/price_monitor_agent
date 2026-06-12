"""
Tools subpackage for extensible tool execution in agent workflows.

What it does:
    Provides a base class for tools, a registry for tool discovery, and concrete
    tool implementations covering data acquisition, technical analysis, text
    analysis, backtesting, alert dispatch, and output writing.

Entities in it:
    - base: Abstract BaseTool class and ToolExecutionError exception.
    - registry: ToolRegistry for registering, retrieving, and listing tools.
    - data_acquisition: FetchExchangeDataTool, FetchMacroDataTool,
      FetchNewsDataTool, FetchSocialMediaDataTool for domain-specific data retrieval.
    - financial_analysis: TechnicalAnalysisTool, QuantitativeAnalysisTool,
      SignalAnalysisTool, DiagnosticAnalysisTool.
    - backtest: DetectRegimeTool, EstimateParametersTool, SimulateProcessTool,
      RunMonteCarloTool.
    - text_analysis: ChunkTextTool, SemanticSearchTool, ExtractEntitiesTool,
      ClassifyTextTool, ScoreTextTool, SummarizeTextTool, CrossModalAlignmentTool.
    - alert_dispatch: SendWebhookTool, SendEmailTool, SendTelegramTool.
    - write_output: WriteOutputTool for persisting results to files.

How used by other modules:
    - The agent subpackage uses the registry to resolve tool names from LLM
      tool_calls and invokes them via the BaseTool.execute() interface.
    - The schema subpackage references tool names in NodeDefinition for TOOL nodes.
    - The orchestration engine injects credentials into tools before execution.
"""

from backend.tools.base import BaseTool, ToolExecutionError
from backend.tools.registry import ToolRegistry
