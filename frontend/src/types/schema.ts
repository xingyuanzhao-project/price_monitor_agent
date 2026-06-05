/**
 * TypeScript interfaces matching backend Pydantic models.
 *
 * Every type here mirrors a corresponding model in backend/schema/models.py
 * plus additional types for user settings, run records, and run events.
 */

export enum NodeType {
  AGENT = "agent",
  AGENT_GROUP = "agent_group",
  TOOL = "tool",
}

export enum EdgeType {
  DATA_FLOW = "data_flow",
  TOOL_CALL = "tool_call",
  SYNCHRONIZATION = "synchronization",
}

export enum LoggingLevel {
  NONE = "none",
  ERRORS = "errors",
  INFO = "info",
  CRITICAL_INFO = "critical_info",
}

export enum GroupStructure {
  PARALLEL = "parallel",
  SEQUENTIAL = "sequential",
  PYRAMID = "pyramid",
  DEFAULT = "default",
}

export interface WorkflowConfig {
  total_timeout: number;
  logging_level: LoggingLevel;
  trace_enabled: boolean;
  dead_loop_detection: boolean;
}

export interface NodeConfig {
  model_id: string;
  temperature: number;
  max_tokens: number | null;
  response_format: Record<string, unknown> | null;
  agent_rules: string[];
  retries: number;
  backoff_multiplier: number;
  fallback_model_id: string | null;
  termination_conditions: string[];
  max_iterations: number;
}

export interface AgentGroupConfig {
  max_parallel_agents: number;
  min_agents: number;
  max_agents: number;
  group_structure: GroupStructure;
  shared_state: Record<string, unknown>;
  tool_authorization: string[];
}

export interface NodePosition {
  x: number;
  y: number;
}

export interface NodeDefinition {
  node_id: string;
  node_type: NodeType;
  label: string;
  config: NodeConfig;
  group_config: AgentGroupConfig | null;
  position: NodePosition;
}

export interface EdgeDefinition {
  edge_id: string;
  edge_type: EdgeType;
  source_node_id: string;
  target_node_id: string;
}

export interface WorkflowSchema {
  schema_id: string;
  name: string;
  description: string;
  nodes: NodeDefinition[];
  edges: EdgeDefinition[];
  config: WorkflowConfig;
}

export interface LLMProviderConfig {
  provider_name: string;
  base_url: string;
  api_key: string;
  available_models: string[];
}

export interface APICredential {
  credential_name: string;
  credential_type: string;
  fields: Record<string, string>;
}

export interface UserSettings {
  llm_providers: LLMProviderConfig[];
  api_credentials: APICredential[];
  global_defaults: {
    temperature: number;
    max_tokens: number;
    rate_limit_rpm: number;
  };
}

export interface RunRecord {
  run_id: string;
  schema_id: string;
  schema_name: string;
  status: "pending" | "running" | "completed" | "failed" | "cancelled";
  started_at: string;
  finished_at: string | null;
  error_message: string | null;
}

export interface RunEvent {
  event_id: string;
  run_id: string;
  node_id: string;
  event_type: "node_start" | "node_output" | "node_error" | "node_complete" | "tool_call" | "tool_result" | "run_complete" | "run_error";
  timestamp: string;
  data: Record<string, unknown>;
}
