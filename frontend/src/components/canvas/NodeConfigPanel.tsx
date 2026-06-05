/**
 * Right sidebar: per-node configuration editor.
 *
 * Displays when a node is selected on the canvas. Provides controls for
 * model selection (dropdown from API), temperature slider, max tokens,
 * response format, agent rules list, retries, termination conditions,
 * max iterations. For AGENT_GROUP nodes: group structure, min/max agents,
 * max parallel, and tool authorization.
 */

import { useState, useCallback } from "react";
import { useWorkflowStore } from "../../store/workflowStore";
import { NodeType, GroupStructure } from "../../types/schema";
import type { NodeConfig, AgentGroupConfig } from "../../types/schema";

export default function NodeConfigPanel() {
  const selectedNodeId = useWorkflowStore((storeState) => storeState.selectedNodeId);
  const nodes = useWorkflowStore((storeState) => storeState.nodes);
  const availableModels = useWorkflowStore((storeState) => storeState.availableModels);
  const availableTools = useWorkflowStore((storeState) => storeState.availableTools);
  const updateNode = useWorkflowStore((storeState) => storeState.updateNode);
  const updateNodeConfig = useWorkflowStore((storeState) => storeState.updateNodeConfig);
  const updateGroupConfig = useWorkflowStore((storeState) => storeState.updateGroupConfig);
  const removeNode = useWorkflowStore((storeState) => storeState.removeNode);
  const setSelectedNodeId = useWorkflowStore((storeState) => storeState.setSelectedNodeId);

  const [newRule, setNewRule] = useState("");
  const [newCondition, setNewCondition] = useState("");
  const [newTool, setNewTool] = useState("");

  const node = nodes.find((n) => n.id === selectedNodeId);
  if (!node) return null;

  const config = node.data.config as NodeConfig;
  const nodeType = node.data.nodeType as NodeType;
  const groupConfig = node.data.groupConfig as AgentGroupConfig | null;
  const label = node.data.label as string;

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

        <ModelSelector
          modelId={config.model_id}
          availableModels={availableModels}
          onChange={(modelId) => updateNodeConfig(node.id, { model_id: modelId })}
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

        <div className="form-group">
          <label className="form-label">Response Format (JSON)</label>
          <textarea
            className="form-textarea"
            value={
              config.response_format
                ? JSON.stringify(config.response_format, null, 2)
                : ""
            }
            placeholder='{"type": "json_object"}'
            onChange={(e) => {
              if (!e.target.value.trim()) {
                updateNodeConfig(node.id, { response_format: null });
                return;
              }
              try {
                const parsed = JSON.parse(e.target.value);
                updateNodeConfig(node.id, { response_format: parsed });
              } catch {
                /* Let user keep typing until valid JSON */
              }
            }}
          />
        </div>

        <hr className="divider" />

        <TagListEditor
          label="Agent Rules"
          items={config.agent_rules}
          newValue={newRule}
          onNewValueChange={setNewRule}
          onAdd={() => {
            if (!newRule.trim()) return;
            updateNodeConfig(node.id, {
              agent_rules: [...config.agent_rules, newRule.trim()],
            });
            setNewRule("");
          }}
          onRemove={(index) =>
            updateNodeConfig(node.id, {
              agent_rules: config.agent_rules.filter((_, i) => i !== index),
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
          <label className="form-label">Backoff Multiplier</label>
          <input
            type="number"
            className="form-input"
            min="1"
            step="0.1"
            value={config.backoff_multiplier}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                backoff_multiplier: parseFloat(e.target.value) || 1,
              })
            }
          />
        </div>

        <div className="form-group">
          <label className="form-label">Fallback Model</label>
          <select
            className="form-select"
            value={config.fallback_model_id ?? ""}
            onChange={(e) =>
              updateNodeConfig(node.id, {
                fallback_model_id: e.target.value || null,
              })
            }
          >
            <option value="">None</option>
            {availableModels.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
          </select>
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

        {nodeType === NodeType.AGENT_GROUP && groupConfig && (
          <AgentGroupConfigEditor
            groupConfig={groupConfig}
            nodeId={node.id}
            availableTools={availableTools}
            updateGroupConfig={updateGroupConfig}
            newTool={newTool}
            setNewTool={setNewTool}
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

function ModelSelector({
  modelId,
  availableModels,
  onChange,
}: {
  modelId: string;
  availableModels: string[];
  onChange: (modelId: string) => void;
}) {
  return (
    <div className="form-group">
      <label className="form-label">Model</label>
      {availableModels.length > 0 ? (
        <select
          className="form-select"
          value={modelId}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">Select a model</option>
          {availableModels.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </select>
      ) : (
        <input
          className="form-input"
          value={modelId}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Model identifier"
        />
      )}
    </div>
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
  newTool,
  setNewTool,
}: {
  groupConfig: AgentGroupConfig;
  nodeId: string;
  availableTools: string[];
  updateGroupConfig: (nodeId: string, updates: Partial<AgentGroupConfig>) => void;
  newTool: string;
  setNewTool: (value: string) => void;
}) {
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

      <div className="form-group">
        <label className="form-label">Tool Authorization</label>
        {groupConfig.tool_authorization.length > 0 && (
          <div className="tag-list">
            {groupConfig.tool_authorization.map((tool, index) => (
              <span key={index} className="tag-item">
                {tool}
                <span
                  className="tag-remove"
                  onClick={() =>
                    updateGroupConfig(nodeId, {
                      tool_authorization: groupConfig.tool_authorization.filter(
                        (_, i) => i !== index
                      ),
                    })
                  }
                >
                  &times;
                </span>
              </span>
            ))}
          </div>
        )}
        <div className="tag-input-row">
          {availableTools.length > 0 ? (
            <select
              className="form-select"
              value={newTool}
              onChange={(e) => setNewTool(e.target.value)}
              style={{ flex: 1 }}
            >
              <option value="">Select tool...</option>
              {availableTools
                .filter((t) => !groupConfig.tool_authorization.includes(t))
                .map((tool) => (
                  <option key={tool} value={tool}>
                    {tool}
                  </option>
                ))}
            </select>
          ) : (
            <input
              className="form-input"
              value={newTool}
              onChange={(e) => setNewTool(e.target.value)}
              placeholder="Tool name"
            />
          )}
          <button
            className="btn btn-sm"
            onClick={() => {
              if (!newTool.trim()) return;
              updateGroupConfig(nodeId, {
                tool_authorization: [
                  ...groupConfig.tool_authorization,
                  newTool.trim(),
                ],
              });
              setNewTool("");
            }}
          >
            Add
          </button>
        </div>
      </div>
    </>
  );
}
