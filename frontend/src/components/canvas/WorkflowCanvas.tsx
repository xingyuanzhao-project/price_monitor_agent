/**
 * Central ReactFlow canvas for composing workflow graphs.
 *
 * Renders custom agent nodes and typed edges. Provides a toolbar to add
 * new nodes (agent, agent_group, tool). Handles node selection, edge
 * connections, and position changes.
 *
 * Edge direction and type are structural: a connection is only valid
 * from an output port to a matching input port, and the edge type is
 * read from the ports involved (EDGE_TYPE_PORTS) — never guessed from
 * node types.
 */

import { useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type OnConnect,
  type OnNodesChange,
  type OnEdgesChange,
  type IsValidConnection,
  applyNodeChanges,
  applyEdgeChanges,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import AgentNode from "./AgentNode";
import { DataFlowEdge, ToolCallEdge } from "./edges";
import { useWorkflowStore, EDGE_TYPE_PORTS } from "../../store/workflowStore";
import { NodeType, EdgeType, GroupStructure } from "../../types/schema";
import type { NodeConfig } from "../../types/schema";

/**
 * Resolve the edge type declared by a pair of ports, or null when the
 * ports do not form a valid connection. Synchronization edges share the
 * data ports but are not drawable, so they are skipped.
 */
function edgeTypeForPorts(
  sourceHandle: string | null | undefined,
  targetHandle: string | null | undefined,
): EdgeType | null {
  for (const [edgeType, ports] of Object.entries(EDGE_TYPE_PORTS)) {
    if (edgeType === EdgeType.SYNCHRONIZATION) continue;
    if (ports.sourceHandle === sourceHandle && ports.targetHandle === targetHandle) {
      return edgeType as EdgeType;
    }
  }
  return null;
}

const DEFAULT_NODE_CONFIG: NodeConfig = {
  provider: "openrouter",
  model_id: "",
  temperature: 0.7,
  max_tokens: null,
  response_format: null,
  instruction: [],
  retries: 2,
  retry_waiting_time: 1.5,
  termination_conditions: [],
  token_budget: 32768,
  scope_window: 5,
  tools: [],
  tool_strict: true,
  tool_choice: "auto",
  parallel_tool_calls: true,
  call_budget: 50,
  rate_limit_per_minute: 30,
  few_shot_examples: [],
  read_upstream_state: true,
  expose_downstream_state: true,
  read_orchestration_state: false,
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
  const edgeTypes = useMemo(() => ({ dataFlow: DataFlowEdge, toolCall: ToolCallEdge }), []);

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

  /** Only output→input or tool→call port pairs may connect. */
  const isValidConnection: IsValidConnection = useCallback(
    (connection) =>
      edgeTypeForPorts(connection.sourceHandle, connection.targetHandle) !== null,
    []
  );

  const onConnect: OnConnect = useCallback(
    (connection) => {
      const edgeType = edgeTypeForPorts(connection.sourceHandle, connection.targetHandle);
      if (edgeType === null) return;

      // The store is the single edge-construction path: it derives the
      // ReactFlow edge (handles, component, style, marker) from the
      // definition, exactly as it does when loading a saved schema.
      addEdgeAction({
        edge_id: `edge-${crypto.randomUUID().slice(0, 8)}`,
        edge_type: edgeType,
        source_node_id: connection.source!,
        target_node_id: connection.target!,
      });
    },
    [addEdgeAction]
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
              shared_context: {},
              tool_authorization: [],
              sub_agent_read_group_state: true,
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
        isValidConnection={isValidConnection}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
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
