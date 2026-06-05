/**
 * Custom ReactFlow node for agent, agent_group, and tool nodes.
 *
 * Renders a color-coded card with a type dot, label, and model_id badge.
 * Provides source and target handles for edge connections.
 */

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { NodeType, NodeConfig } from "../../types/schema";

interface AgentNodeData extends Record<string, unknown> {
  label: string;
  nodeType: NodeType;
  config: NodeConfig;
}

function AgentNodeComponent({ data, selected }: NodeProps) {
  const { label, nodeType, config } = data as unknown as AgentNodeData;

  return (
    <div
      className={`agent-node type-${nodeType} ${selected ? "selected" : ""}`}
    >
      <Handle type="target" position={Position.Top} />

      <div className="agent-node-header">
        <span className="agent-node-type-dot" />
        <span>{label}</span>
      </div>

      {config?.model_id && (
        <div className="agent-node-badge">{config.model_id}</div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export default memo(AgentNodeComponent);
