/**
 * Right sidebar: per-node configuration editor.
 *
 * Displays when a node is selected on the canvas. Provides controls for
 * model selection, temperature, max tokens, output schema (structured
 * field editor), instruction (labeled "Prompts"), retries,
 * retry waiting time, termination conditions, max iterations, token budget,
 * scope window, tools, call budget, rate limit, and few-shot examples.
 * For AGENT_GROUP nodes: group structure, min/max agents, max parallel,
 * tool authorization, and shared state editor.
 */

import { useState, useCallback, useMemo, useEffect } from "react";
import { useWorkflowStore } from "../../store/workflowStore";
import { Combobox } from "../ui/Combobox";
import type { ComboboxOption } from "../ui/Combobox";
import { settingsApi } from "../../api/client";
import { NodeType, GroupStructure } from "../../types/schema";
import type {
  NodeConfig,
  AgentGroupConfig,
  AnyProviderName,
  ProviderStatusResponse,
  OutputFieldRow,
  OutputFieldDataType,
} from "../../types/schema";

const PROVIDER_LABELS: Record<string, string> = {
  openrouter: "OpenRouter",
  openai: "OpenAI",
  anthropic: "Anthropic",
  google: "Google",
  ollama: "Ollama",
  vllm: "vLLM",
  llama_cpp: "llama.cpp",
};

const DATA_TYPE_OPTIONS: { value: OutputFieldDataType; label: string }[] = [
  { value: "string", label: "String" },
  { value: "binary", label: "Binary" },
  { value: "category", label: "Category" },
  { value: "numeric", label: "Numeric" },
  { value: "integer", label: "Integer" },
];

function parseDataType(raw: unknown): OutputFieldDataType {
  if (
    typeof raw === "string" &&
    DATA_TYPE_OPTIONS.some((dt) => dt.value === raw)
  ) {
    return raw as OutputFieldDataType;
  }
  return "string";
}

function readOutputRows(responseFormat: Record<string, unknown> | null): OutputFieldRow[] {
  if (!responseFormat) return [];
  const jsonSchema = responseFormat.json_schema as Record<string, unknown> | undefined;
  if (!jsonSchema) return [];
  const schema = jsonSchema.schema as Record<string, unknown> | undefined;
  if (!schema) return [];
  const properties = schema.properties as Record<string, Record<string, unknown>> | undefined;
  if (!properties) return [];
  const requiredList = (schema.required ?? []) as string[];
  const requiredSet = new Set(requiredList);

  return Object.entries(properties).map(([fieldName, fieldSpec]) => ({
    field_name: fieldName,
    data_type: parseDataType(fieldSpec.data_type ?? fieldSpec.type ?? "string"),
    required: requiredSet.has(fieldName),
    options: Array.isArray(fieldSpec.options) ? fieldSpec.options.map(String) : [],
    range_start: String(fieldSpec.range_start ?? ""),
    range_end: String(fieldSpec.range_end ?? ""),
    step: String(fieldSpec.step ?? ""),
  }));
}

function rowsToResponseFormat(rows: OutputFieldRow[]): Record<string, unknown> | null {
  if (rows.length === 0) return null;
  const DATA_TYPE_TO_JSON_TYPE: Record<string, string> = {
    string: "string",
    binary: "string",
    category: "string",
    numeric: "number",
    integer: "integer",
  };
  const properties: Record<string, Record<string, unknown>> = {};
  const required: string[] = [];
  for (const row of rows) {
    if (!row.field_name.trim()) continue;
    const entry: Record<string, unknown> = {
      type: DATA_TYPE_TO_JSON_TYPE[row.data_type] ?? "string",
      data_type: row.data_type,
    };
    if (row.data_type === "category" && row.options.length > 0) {
      entry.options = row.options;
    }
    if ((row.data_type === "numeric" || row.data_type === "integer") && row.range_start) {
      entry.range_start = row.range_start;
    }
    if ((row.data_type === "numeric" || row.data_type === "integer") && row.range_end) {
      entry.range_end = row.range_end;
    }
    if (row.data_type === "integer" && row.step) {
      entry.step = row.step;
    }
    properties[row.field_name] = entry;
    if (row.required) required.push(row.field_name);
  }
  return {
    type: "json_schema",
    json_schema: {
      name: "output",
      schema: { type: "object", properties, required },
    },
  };
}

export default function NodeConfigPanel() {
  const selectedNodeId = useWorkflowStore((s) => s.selectedNodeId);
  const nodes = useWorkflowStore((s) => s.nodes);
  const availableTools = useWorkflowStore((s) => s.availableTools);
  const updateNode = useWorkflowStore((s) => s.updateNode);
  const updateNodeConfig = useWorkflowStore((s) => s.updateNodeConfig);
  const updateGroupConfig = useWorkflowStore((s) => s.updateGroupConfig);
  const removeNode = useWorkflowStore((s) => s.removeNode);
  const setSelectedNodeId = useWorkflowStore((s) => s.setSelectedNodeId);

  const [newRule, setNewRule] = useState("");
  const [newCondition, setNewCondition] = useState("");

  const node = nodes.find((n) => n.id === selectedNodeId);
  if (!node) return null;

  const config = node.data.config as NodeConfig;
  const nodeType = node.data.nodeType as NodeType;
  const groupConfig = node.data.groupConfig as AgentGroupConfig | null;
  const label = node.data.label as string;

  if (nodeType === NodeType.TOOL) {
    return (
      <ToolNodeConfigPanel
        nodeId={node.id}
        label={label}
        config={config}
        availableTools={availableTools}
        updateNode={updateNode}
        updateNodeConfig={updateNodeConfig}
        removeNode={removeNode}
        setSelectedNodeId={setSelectedNodeId}
      />
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">Node Configuration</div>
      <div className="panel-body">
        <div className="form-group">
          <label className="form-label">Label</label>
          <input
            className="form-input"
            value={label}
            onChange={(e) => updateNode(node.id, { label: e.target.value })}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Node Type</label>
          <div className="text-sm text-muted" style={{ textTransform: "uppercase" }}>
            {nodeType}
          </div>
        </div>

        <hr className="divider" />

        <ProviderModelSelector
          provider={config.provider}
          modelId={config.model_id}
          onProviderChange={(provider) =>
            updateNodeConfig(node.id, { provider, model_id: "" })
          }
          onModelChange={(modelId) => updateNodeConfig(node.id, { model_id: modelId })}
        />

        <div className="form-group">
          <label className="form-label">
            Temperature: {config.temperature.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="2"
            step="0.01"
            value={config.temperature}
            onChange={(e) =>
              updateNodeConfig(node.id, { temperature: parseFloat(e.target.value) })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Max Tokens (blank = model max)</label>
          <input
            type="number"
            className="form-input"
            value={config.max_tokens ?? ""}
            placeholder="Default (model max)"
            onChange={(e) =>
              updateNodeConfig(node.id, {
                max_tokens: e.target.value ? parseInt(e.target.value) : null,
              })
            }
          />
        </div>

        <hr className="divider" />

        <OutputSchemaEditor
          responseFormat={config.response_format}
          onChange={(rf) => updateNodeConfig(node.id, { response_format: rf })}
        />

        <hr className="divider" />

        <TagListEditor
          label="Prompts"
          items={config.instruction}
          newValue={newRule}
          onNewValueChange={setNewRule}
          onAdd={() => {
            if (!newRule.trim()) return;
            updateNodeConfig(node.id, {
              instruction: [...config.instruction, newRule.trim()],
            });
            setNewRule("");
          }}
          onRemove={(index) =>
            updateNodeConfig(node.id, {
              instruction: config.instruction.filter((_, i) => i !== index),
            })
          }
        />

        <TagListEditor
          label="Termination Conditions"
          items={config.termination_conditions}
          newValue={newCondition}
          onNewValueChange={setNewCondition}
          onAdd={() => {
            if (!newCondition.trim()) return;
            updateNodeConfig(node.id, {
              termination_conditions: [
                ...config.termination_conditions,
                newCondition.trim(),
              ],
            });
            setNewCondition("");
          }}
          onRemove={(index) =>
            updateNodeConfig(node.id, {
              termination_conditions: config.termination_conditions.filter(
                (_, i) => i !== index
              ),
            })
          }
        />

        <hr className="divider" />

        <div className="form-group">
          <label className="form-label">Retries</label>
          <input
            type="number"
            className="form-input"
            min="0"
            value={config.retries}
            onChange={(e) =>
              updateNodeConfig(node.id, { retries: parseInt(e.target.value) || 0 })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Retry Waiting Time (seconds)</label>
          <input
            type="number"
            className="form-input"
            min="0.1"
            step="0.1"
            value={config.retry_waiting_time}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                retry_waiting_time: parseFloat(e.target.value) || 1,
              })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Max Iterations</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={config.max_iterations}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                max_iterations: parseInt(e.target.value) || 1,
              })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Iteration Sleep (seconds)</label>
          <input
            type="number"
            className="form-input"
            min="0"
            step="0.1"
            value={config.iteration_sleep}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                iteration_sleep: parseFloat(e.target.value) || 0,
              })
            }
          />
        </div>

        <hr className="divider" />

        <div className="form-group">
          <label className="form-label">Token Budget</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={config.token_budget}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                token_budget: parseInt(e.target.value) || 32768,
              })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Scope Window (few-shot count)</label>
          <input
            type="number"
            className="form-input"
            min="0"
            value={config.scope_window}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                scope_window: parseInt(e.target.value) || 0,
              })
            }
          />
        </div>

        <hr className="divider" />

        <ToolDropdownEditor
          label="Authorized Tools"
          selectedTools={config.tools}
          availableTools={availableTools}
          onAdd={(toolName) =>
            updateNodeConfig(node.id, {
              tools: [...config.tools, toolName],
            })
          }
          onRemove={(index) =>
            updateNodeConfig(node.id, {
              tools: config.tools.filter((_, i) => i !== index),
            })
          }
        />

        <div className="form-group">
          <label className="form-label">Call Budget (max tool calls)</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={config.call_budget}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                call_budget: parseInt(e.target.value) || 50,
              })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Rate Limit (calls/min)</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={config.rate_limit_per_minute}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                rate_limit_per_minute: parseInt(e.target.value) || 30,
              })
            }
          />
        </div>

        {nodeType === NodeType.AGENT_GROUP && groupConfig && (
          <AgentGroupConfigEditor
            groupConfig={groupConfig}
            nodeId={node.id}
            availableTools={availableTools}
            updateGroupConfig={updateGroupConfig}
          />
        )}

        <hr className="divider" />
        <button
          className="btn btn-danger"
          onClick={() => {
            removeNode(node.id);
            setSelectedNodeId(null);
          }}
        >
          Delete Node
        </button>
      </div>
    </div>
  );
}

function OutputSchemaEditor({
  responseFormat,
  onChange,
}: {
  responseFormat: Record<string, unknown> | null;
  onChange: (rf: Record<string, unknown> | null) => void;
}) {
  const rows = useMemo(() => readOutputRows(responseFormat), [responseFormat]);

  const writeRows = useCallback(
    (nextRows: OutputFieldRow[]) => {
      onChange(rowsToResponseFormat(nextRows));
    },
    [onChange]
  );

  const onRowChange = useCallback(
    (rowIndex: number, patch: Partial<OutputFieldRow>) => {
      const next = rows.map((row, i) => (i === rowIndex ? { ...row, ...patch } : row));
      writeRows(next);
    },
    [rows, writeRows]
  );

  const onRowRemove = useCallback(
    (rowIndex: number) => {
      writeRows(rows.filter((_, i) => i !== rowIndex));
    },
    [rows, writeRows]
  );

  const onAddRow = useCallback(() => {
    writeRows([
      ...rows,
      {
        field_name: `field_${rows.length + 1}`,
        data_type: "string",
        required: false,
        options: [],
        range_start: "",
        range_end: "",
        step: "",
      },
    ]);
  }, [rows, writeRows]);

  return (
    <div className="form-group">
      <label className="form-label">Output Schema</label>
      <span className="text-sm text-muted">
        Define the fields the LLM should return.
      </span>

      <div className="output-schema-rows">
        {rows.map((row, rowIndex) => (
          <div key={rowIndex} className="output-schema-row">
            <div className="output-schema-row-header">
              <label className="output-field-label">
                <span className="text-sm">Field</span>
                <input
                  className="form-input"
                  value={row.field_name}
                  onChange={(e) =>
                    onRowChange(rowIndex, { field_name: e.target.value })
                  }
                />
              </label>
              <label className="output-field-required">
                <span className="text-sm">Required</span>
                <input
                  type="checkbox"
                  checked={row.required}
                  onChange={(e) =>
                    onRowChange(rowIndex, { required: e.target.checked })
                  }
                />
              </label>
              <button
                className="btn-link text-danger"
                onClick={() => onRowRemove(rowIndex)}
              >
                Remove
              </button>
            </div>
            <label className="output-field-label">
              <span className="text-sm">Type</span>
              <select
                className="form-select"
                value={row.data_type}
                onChange={(e) =>
                  onRowChange(rowIndex, {
                    data_type: e.target.value as OutputFieldDataType,
                    options: [],
                    range_start: "",
                    range_end: "",
                    step: "",
                  })
                }
              >
                {DATA_TYPE_OPTIONS.map((dt) => (
                  <option key={dt.value} value={dt.value}>
                    {dt.label}
                  </option>
                ))}
              </select>
            </label>
            {row.data_type === "category" && (
              <label className="output-field-label">
                <span className="text-sm">Options (comma-separated)</span>
                <input
                  className="form-input"
                  value={row.options.join(", ")}
                  onChange={(e) =>
                    onRowChange(rowIndex, {
                      options: e.target.value
                        .split(",")
                        .map((v) => v.trim())
                        .filter((v) => v.length > 0),
                    })
                  }
                />
              </label>
            )}
            {(row.data_type === "numeric" || row.data_type === "integer") && (
              <div className="output-schema-range">
                <label className="output-field-label">
                  <span className="text-sm">Range start</span>
                  <input
                    className="form-input"
                    placeholder="e.g. 0"
                    value={row.range_start}
                    onChange={(e) =>
                      onRowChange(rowIndex, { range_start: e.target.value })
                    }
                  />
                </label>
                <label className="output-field-label">
                  <span className="text-sm">Range end</span>
                  <input
                    className="form-input"
                    placeholder="e.g. 100"
                    value={row.range_end}
                    onChange={(e) =>
                      onRowChange(rowIndex, { range_end: e.target.value })
                    }
                  />
                </label>
                {row.data_type === "integer" && (
                  <label className="output-field-label">
                    <span className="text-sm">Step</span>
                    <input
                      className="form-input"
                      placeholder="e.g. 1"
                      value={row.step}
                      onChange={(e) =>
                        onRowChange(rowIndex, { step: e.target.value })
                      }
                    />
                  </label>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <button className="btn btn-sm" onClick={onAddRow}>
        Add field
      </button>
    </div>
  );
}

function ToolNodeConfigPanel({
  nodeId,
  label,
  config,
  availableTools,
  updateNode,
  updateNodeConfig,
  removeNode,
  setSelectedNodeId,
}: {
  nodeId: string;
  label: string;
  config: NodeConfig;
  availableTools: string[];
  updateNode: (nodeId: string, updates: Partial<{ label: string }>) => void;
  updateNodeConfig: (nodeId: string, updates: Partial<NodeConfig>) => void;
  removeNode: (nodeId: string) => void;
  setSelectedNodeId: (nodeId: string | null) => void;
}) {
  const selectedTool = config.tools.length > 0 ? config.tools[0] : "";

  return (
    <div className="panel">
      <div className="panel-header">Tool Configuration</div>
      <div className="panel-body">
        <div className="form-group">
          <label className="form-label">Label</label>
          <input
            className="form-input"
            value={label}
            onChange={(e) => updateNode(nodeId, { label: e.target.value })}
          />
        </div>

        <div className="form-group">
          <label className="form-label">Node Type</label>
          <div className="text-sm text-muted" style={{ textTransform: "uppercase" }}>
            TOOL
          </div>
        </div>

        <hr className="divider" />

        <div className="form-group">
          <label className="form-label">Tool</label>
          <select
            className="form-select"
            value={selectedTool}
            onChange={(e) => {
              const toolName = e.target.value;
              updateNodeConfig(nodeId, { tools: toolName ? [toolName] : [] });
              if (toolName && (label === "New Tool" || availableTools.includes(label))) {
                updateNode(nodeId, { label: toolName });
              }
            }}
          >
            <option value="">Select a tool…</option>
            {availableTools.map((tool) => (
              <option key={tool} value={tool}>
                {tool}
              </option>
            ))}
          </select>
        </div>

        <hr className="divider" />

        <div className="form-group">
          <label className="form-label">Retries</label>
          <input
            type="number"
            className="form-input"
            min="0"
            value={config.retries}
            onChange={(e) =>
              updateNodeConfig(nodeId, { retries: parseInt(e.target.value) || 0 })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Retry Waiting Time (seconds)</label>
          <input
            type="number"
            className="form-input"
            min="0.1"
            step="0.1"
            value={config.retry_waiting_time}
            onChange={(e) =>
              updateNodeConfig(nodeId, {
                retry_waiting_time: parseFloat(e.target.value) || 1,
              })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Call Budget (max invocations)</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={config.call_budget}
            onChange={(e) =>
              updateNodeConfig(nodeId, {
                call_budget: parseInt(e.target.value) || 50,
              })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Rate Limit (calls/min)</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={config.rate_limit_per_minute}
            onChange={(e) =>
              updateNodeConfig(nodeId, {
                rate_limit_per_minute: parseInt(e.target.value) || 30,
              })
            }
          />
        </div>

        <hr className="divider" />
        <button
          className="btn btn-danger"
          onClick={() => {
            removeNode(nodeId);
            setSelectedNodeId(null);
          }}
        >
          Delete Node
        </button>
      </div>
    </div>
  );
}

function ToolDropdownEditor({
  label,
  selectedTools,
  availableTools,
  onAdd,
  onRemove,
}: {
  label: string;
  selectedTools: string[];
  availableTools: string[];
  onAdd: (toolName: string) => void;
  onRemove: (index: number) => void;
}) {
  const [pendingTool, setPendingTool] = useState("");
  const unselected = availableTools.filter((t) => !selectedTools.includes(t));

  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      {selectedTools.length > 0 && (
        <div className="tag-list">
          {selectedTools.map((tool, index) => (
            <span key={index} className="tag-item">
              {tool}
              <span className="tag-remove" onClick={() => onRemove(index)}>
                &times;
              </span>
            </span>
          ))}
        </div>
      )}
      <div className="tag-input-row">
        <select
          className="form-select"
          value={pendingTool}
          onChange={(e) => setPendingTool(e.target.value)}
          style={{ flex: 1 }}
        >
          <option value="">Select tool…</option>
          {unselected.map((tool) => (
            <option key={tool} value={tool}>
              {tool}
            </option>
          ))}
        </select>
        <button
          className="btn btn-sm"
          onClick={() => {
            if (!pendingTool) return;
            onAdd(pendingTool);
            setPendingTool("");
          }}
        >
          Add
        </button>
      </div>
    </div>
  );
}

interface ModelCacheEntry {
  models: ComboboxOption[];
  fetchedAt: number;
}

const MODEL_CACHE_STALE_MS = 5 * 60_000;
const modelCache: Record<string, ModelCacheEntry> = {};

function ProviderModelSelector({
  provider,
  modelId,
  onProviderChange,
  onModelChange,
}: {
  provider: AnyProviderName;
  modelId: string;
  onProviderChange: (provider: AnyProviderName) => void;
  onModelChange: (modelId: string) => void;
}) {
  const [loading, setLoading] = useState(false);
  const [errorText, setErrorText] = useState<string | undefined>(undefined);
  const [models, setModels] = useState<ComboboxOption[]>([]);
  const [providerStatus, setProviderStatus] = useState<ProviderStatusResponse | null>(null);

  useEffect(() => {
    settingsApi.providerStatus().then(setProviderStatus).catch(() => {});
  }, []);

  useEffect(() => {
    if (!provider) return;
    let cancelled = false;

    const cached = modelCache[provider];
    if (cached && Date.now() - cached.fetchedAt < MODEL_CACHE_STALE_MS) {
      setModels(cached.models);
      setErrorText(undefined);
      setLoading(false);
      return;
    }

    setLoading(true);
    setErrorText(undefined);

    fetch(`/api/models/${encodeURIComponent(provider)}`)
      .then((res) => res.json())
      .then((data: { models?: { id: string; label: string }[]; error?: string | null }) => {
        if (cancelled) return;
        if (data.error) {
          setErrorText(data.error);
          setModels([]);
          return;
        }
        const options: ComboboxOption[] = (data.models ?? []).map((m) => ({
          value: m.id,
          label: m.label,
        }));
        modelCache[provider] = { models: options, fetchedAt: Date.now() };
        setModels(options);
      })
      .catch(() => {
        if (!cancelled) setErrorText("Could not load model catalogue.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => { cancelled = true; };
  }, [provider]);

  const configuredProviders = useMemo(() => {
    if (!providerStatus) return [];
    const items: { value: string; label: string }[] = [];
    for (const cp of providerStatus.cloud_providers) {
      if (cp.configured) {
        items.push({
          value: cp.provider_name,
          label: PROVIDER_LABELS[cp.provider_name] ?? cp.provider_name,
        });
      }
    }
    for (const lp of providerStatus.local_endpoints) {
      if (lp.configured) {
        items.push({
          value: lp.provider_name,
          label: PROVIDER_LABELS[lp.provider_name] ?? lp.provider_name,
        });
      }
    }
    return items;
  }, [providerStatus]);

  const hintText = useMemo(() => {
    if (!providerStatus) return null;
    const cloud = providerStatus.cloud_providers.find(
      (p) => p.provider_name === provider,
    );
    if (cloud) return `Key: ${cloud.api_key_env}`;
    const local = providerStatus.local_endpoints.find(
      (p) => p.provider_name === provider,
    );
    if (local && local.api_base) return `Endpoint: ${local.api_base}`;
    return null;
  }, [providerStatus, provider]);

  return (
    <>
      <div className="form-group">
        <label className="form-label">Provider</label>
        <select
          className="form-select"
          value={provider}
          onChange={(e) => onProviderChange(e.target.value as AnyProviderName)}
        >
          {configuredProviders.length > 0 ? (
            configuredProviders.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))
          ) : (
            Object.entries(PROVIDER_LABELS).map(([val, lbl]) => (
              <option key={val} value={val}>
                {lbl}
              </option>
            ))
          )}
        </select>
        {hintText && (
          <span className="text-xs text-muted" style={{ marginTop: "0.25rem", display: "block" }}>
            {hintText}
          </span>
        )}
      </div>
      <div className="form-group">
        <label className="form-label">Model</label>
        <Combobox
          options={models}
          value={modelId}
          onChange={onModelChange}
          placeholder="openrouter/auto"
          loading={loading}
          loadingText="Loading model catalogue…"
          errorText={errorText}
        />
      </div>
    </>
  );
}

function TagListEditor({
  label,
  items,
  newValue,
  onNewValueChange,
  onAdd,
  onRemove,
}: {
  label: string;
  items: string[];
  newValue: string;
  onNewValueChange: (value: string) => void;
  onAdd: () => void;
  onRemove: (index: number) => void;
}) {
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") {
        e.preventDefault();
        onAdd();
      }
    },
    [onAdd]
  );

  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      {items.length > 0 && (
        <div className="tag-list">
          {items.map((item, index) => (
            <span key={index} className="tag-item">
              {item}
              <span className="tag-remove" onClick={() => onRemove(index)}>
                &times;
              </span>
            </span>
          ))}
        </div>
      )}
      <div className="tag-input-row">
        <input
          className="form-input"
          value={newValue}
          onChange={(e) => onNewValueChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={`Add ${label.toLowerCase()}...`}
        />
        <button className="btn btn-sm" onClick={onAdd}>
          Add
        </button>
      </div>
    </div>
  );
}

function AgentGroupConfigEditor({
  groupConfig,
  nodeId,
  availableTools,
  updateGroupConfig,
}: {
  groupConfig: AgentGroupConfig;
  nodeId: string;
  availableTools: string[];
  updateGroupConfig: (nodeId: string, updates: Partial<AgentGroupConfig>) => void;
}) {
  const [newStateKey, setNewStateKey] = useState("");
  const [newStateValue, setNewStateValue] = useState("");

  return (
    <>
      <hr className="divider" />
      <div className="panel-header" style={{ padding: "12px 0 8px", border: "none" }}>
        Agent Group Config
      </div>

      <div className="form-group">
        <label className="form-label">Group Structure</label>
        <select
          className="form-select"
          value={groupConfig.group_structure}
          onChange={(e) =>
            updateGroupConfig(nodeId, {
              group_structure: e.target.value as GroupStructure,
            })
          }
        >
          {Object.values(GroupStructure).map((structure) => (
            <option key={structure} value={structure}>
              {structure.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      <div className="inline-fields">
        <div className="form-group">
          <label className="form-label">Min Agents</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={groupConfig.min_agents}
            onChange={(e) =>
              updateGroupConfig(nodeId, {
                min_agents: parseInt(e.target.value) || 1,
              })
            }
          />
        </div>
        <div className="form-group">
          <label className="form-label">Max Agents</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={groupConfig.max_agents}
            onChange={(e) =>
              updateGroupConfig(nodeId, {
                max_agents: parseInt(e.target.value) || 1,
              })
            }
          />
        </div>
      </div>

      <div className="form-group">
        <label className="form-label">Max Parallel Agents</label>
        <input
          type="number"
          className="form-input"
          min="1"
          value={groupConfig.max_parallel_agents}
          onChange={(e) =>
            updateGroupConfig(nodeId, {
              max_parallel_agents: parseInt(e.target.value) || 1,
            })
          }
        />
      </div>

      <ToolDropdownEditor
        label="Tool Authorization"
        selectedTools={groupConfig.tool_authorization}
        availableTools={availableTools}
        onAdd={(toolName) =>
          updateGroupConfig(nodeId, {
            tool_authorization: [...groupConfig.tool_authorization, toolName],
          })
        }
        onRemove={(index) =>
          updateGroupConfig(nodeId, {
            tool_authorization: groupConfig.tool_authorization.filter(
              (_, i) => i !== index
            ),
          })
        }
      />

      <div className="form-group">
        <label className="form-label">Shared State</label>
        {Object.keys(groupConfig.shared_state).length > 0 && (
          <div className="tag-list">
            {Object.entries(groupConfig.shared_state).map(([key, value]) => (
              <span key={key} className="tag-item">
                {key}: {JSON.stringify(value)}
                <span
                  className="tag-remove"
                  onClick={() => {
                    const next = { ...groupConfig.shared_state };
                    delete next[key];
                    updateGroupConfig(nodeId, { shared_state: next });
                  }}
                >
                  &times;
                </span>
              </span>
            ))}
          </div>
        )}
        <div className="tag-input-row">
          <input
            className="form-input"
            value={newStateKey}
            onChange={(e) => setNewStateKey(e.target.value)}
            placeholder="Key"
            style={{ flex: 1 }}
          />
          <input
            className="form-input"
            value={newStateValue}
            onChange={(e) => setNewStateValue(e.target.value)}
            placeholder="Value"
            style={{ flex: 1 }}
          />
          <button
            className="btn btn-sm"
            onClick={() => {
              if (!newStateKey.trim()) return;
              let parsedValue: unknown = newStateValue;
              try {
                parsedValue = JSON.parse(newStateValue);
              } catch {
                /* use raw string */
              }
              updateGroupConfig(nodeId, {
                shared_state: {
                  ...groupConfig.shared_state,
                  [newStateKey.trim()]: parsedValue,
                },
              });
              setNewStateKey("");
              setNewStateValue("");
            }}
          >
            Add
          </button>
        </div>
      </div>
    </>
  );
}
