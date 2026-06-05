/**
 * Central ReactFlow canvas for composing workflow graphs.
 *
 * Renders custom agent nodes and typed edges. Provides a toolbar to add
 * new nodes (agent, agent_group, tool). Handles node selection, edge
 * connections, and position changes.
 */

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  addEdge,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import AgentNode from "./AgentNode";
import { useWorkflowStore } from "../../store/workflowStore";
import { NodeType, EdgeType, GroupStructure } from "../../types/schema";
import type { NodeConfig } from "../../types/schema";

const DEFAULT_NODE_CONFIG: NodeConfig = {
  model_id: "",
  temperature: 0.7,
  max_tokens: null,
  response_format: null,
  agent_rules: [],
  retries: 2,
  backoff_multiplier: 1.5,
  fallback_model_id: null,
  termination_conditions: [],
  max_iterations: 10,
};

const NODE_TYPE_LABELS: Record<NodeType, string> = {
  [NodeType.AGENT]: "Agent",
  [NodeType.AGENT_GROUP]: "Agent Group",
  [NodeType.TOOL]: "Tool",
};

export default function WorkflowCanvas() {
  const nodes = useWorkflowStore((storeState) => storeState.nodes);
  const edges = useWorkflowStore((storeState) => storeState.edges);
  const setNodes = useWorkflowStore((storeState) => storeState.setNodes);
  const setEdges = useWorkflowStore((storeState) => storeState.setEdges);
  const addEdgeAction = useWorkflowStore((storeState) => storeState.addEdge);
  const setSelectedNodeId = useWorkflowStore((storeState) => storeState.setSelectedNodeId);
  const schemaId = useWorkflowStore((storeState) => storeState.schemaId);
  const addNode = useWorkflowStore((storeState) => storeState.addNode);

  const nodeTypes = useMemo(() => ({ agentNode: AgentNode }), []);

  const onNodesChange: OnNodesChange = useCallback(
    (changes) => {
      setNodes(applyNodeChanges(changes, nodes));
    },
    [nodes, setNodes]
  );

  const onEdgesChange: OnEdgesChange = useCallback(
    (changes) => {
      setEdges(applyEdgeChanges(changes, edges));
    },
    [edges, setEdges]
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      const edgeId = `edge-${crypto.randomUUID().slice(0, 8)}`;
      const sourceNode = nodes.find((node) => node.id === connection.source);
      const targetNode = nodes.find((node) => node.id === connection.target);

      let edgeType = EdgeType.DATA_FLOW;
      if (
        sourceNode?.data.nodeType === NodeType.TOOL ||
        targetNode?.data.nodeType === NodeType.TOOL
      ) {
        edgeType = EdgeType.TOOL_CALL;
      }

      addEdgeAction({
        edge_id: edgeId,
        edge_type: edgeType,
        source_node_id: connection.source!,
        target_node_id: connection.target!,
      });

      setEdges(
        addEdge(
          {
            ...connection,
            id: edgeId,
            type: "smoothstep",
            style: edgeType === EdgeType.TOOL_CALL
              ? { stroke: "#4ade80" }
              : { stroke: "#4a9eff" },
            data: { edgeType },
          },
          edges
        )
      );
    },
    [nodes, edges, setEdges, addEdgeAction]
  );

  const onNodeClick = useCallback(
    (_event: React.MouseEvent, node: Node) => {
      setSelectedNodeId(node.id);
    },
    [setSelectedNodeId]
  );

  const onPaneClick = useCallback(() => {
    setSelectedNodeId(null);
  }, [setSelectedNodeId]);

  const handleAddNode = useCallback(
    (nodeType: NodeType) => {
      const nodeId = `${nodeType}-${crypto.randomUUID().slice(0, 8)}`;
      const groupConfig =
        nodeType === NodeType.AGENT_GROUP
          ? {
              max_parallel_agents: 5,
              min_agents: 2,
              max_agents: 10,
              group_structure: GroupStructure.DEFAULT,
              shared_state: {},
              tool_authorization: [],
            }
          : null;

      addNode({
        node_id: nodeId,
        node_type: nodeType,
        label: `New ${NODE_TYPE_LABELS[nodeType]}`,
        config: { ...DEFAULT_NODE_CONFIG },
        group_config: groupConfig,
        position: { x: 250 + Math.random() * 200, y: 150 + Math.random() * 200 },
      });
    },
    [addNode]
  );

  if (!schemaId) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
        <div className="empty-state">
          Select or create a schema to begin editing.
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div className="canvas-toolbar">
        {Object.values(NodeType).map((type) => (
          <button key={type} className="btn btn-sm" onClick={() => handleAddNode(type)}>
            + {NODE_TYPE_LABELS[type]}
          </button>
        ))}
      </div>

      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={20} size={1} color="#2a3a5c" />
        <Controls />
        <MiniMap
          nodeColor={(node) => {
            const typeColors: Record<string, string> = {
              agent: "#4a9eff",
              agent_group: "#a78bfa",
              tool: "#4ade80",
            };
            return typeColors[node.data?.nodeType as string] ?? "#64748b";
          }}
          maskColor="rgba(26, 26, 46, 0.8)"
        />
      </ReactFlow>
    </div>
  );
}
