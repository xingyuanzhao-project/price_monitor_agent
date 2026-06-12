/**
 * Custom ReactFlow node for agent, agent_group, and tool nodes.
 *
 * Ports are explicit and directional:
 * - top    "in"   (target) — receives data flow            (agents/groups)
 * - bottom "out"  (source) — emits data flow               (all node types)
 * - right  "tool" (source) — initiates tool calls (green)  (agents/groups)
 * - left   "call" (target) — receives tool calls (green)   (tools)
 *
 * Tools have no top input: they receive input only through calls.
 * An agent's "out" dragged onto its own "in" declares a self-loop.
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
  const isTool = nodeType === "tool";

  return (
    <div
      className={`agent-node type-${nodeType} ${selected ? "selected" : ""}`}
    >
      {isTool ? (
        <>
          <Handle type="target" position={Position.Left} id="call" className="port-tool" />
          <span className="port-label port-label-left">calls</span>
        </>
      ) : (
        <>
          <Handle type="target" position={Position.Top} id="in" />
          <span className="port-label port-label-top">input</span>
          <Handle type="source" position={Position.Right} id="tool" className="port-tool" />
          <span className="port-label port-label-right">tools</span>
        </>
      )}

      <div className="agent-node-header">
        <span className="agent-node-type-dot" />
        <span>{label}</span>
      </div>

      {isTool
        ? config?.tools?.[0] && (
            <div className="agent-node-badge">{config.tools[0]}</div>
          )
        : config?.model_id && (
            <div className="agent-node-badge">{config.model_id}</div>
          )}

      <Handle type="source" position={Position.Bottom} id="out" />
      <span className="port-label port-label-bottom">output</span>
    </div>
  );
}

export default memo(AgentNodeComponent);
