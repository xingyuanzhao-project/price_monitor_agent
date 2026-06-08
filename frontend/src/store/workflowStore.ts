/**
 * Zustand store for workflow editor state.
 *
 * Manages the currently loaded schema, canvas nodes/edges, available models,
 * schema list, and selection state. All mutations are exposed as named actions
 * so components can dispatch changes without direct state access.
 */

import { create } from "zustand";
import type { Node, Edge } from "@xyflow/react";
import type {
  WorkflowSchema,
  WorkflowConfig,
  NodeDefinition,
  EdgeDefinition,
  NodeConfig,
  AgentGroupConfig,
  LoggingLevel,
  NodeType,
  EdgeType,
} from "../types/schema";

interface WorkflowState {
  schemaId: string | null;
  schemaName: string;
  schemaDescription: string;
  nodes: Node[];
  edges: Edge[];
  workflowConfig: WorkflowConfig;
  selectedNodeId: string | null;
  schemas: WorkflowSchema[];
  availableModels: string[];
  availableTools: string[];
  modelsByProvider: Record<string, { id: string; label: string }[]>;

  setSchemas: (schemas: WorkflowSchema[]) => void;
  setAvailableModels: (models: string[]) => void;
  setAvailableTools: (tools: string[]) => void;
  setModelsForProvider: (provider: string, models: { id: string; label: string }[]) => void;
  fetchModelsForProvider: (provider: string) => Promise<{ id: string; label: string }[]>;

  setSchema: (schema: WorkflowSchema) => void;
  clearSchema: () => void;

  addNode: (nodeDefinition: NodeDefinition) => void;
  removeNode: (nodeId: string) => void;
  updateNode: (nodeId: string, updates: Partial<NodeDefinition>) => void;
  updateNodeConfig: (nodeId: string, configUpdates: Partial<NodeConfig>) => void;
  updateGroupConfig: (nodeId: string, groupUpdates: Partial<AgentGroupConfig>) => void;

  addEdge: (edgeDefinition: EdgeDefinition) => void;
  removeEdge: (edgeId: string) => void;

  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;

  setSelectedNodeId: (nodeId: string | null) => void;
  setWorkflowConfig: (config: Partial<WorkflowConfig>) => void;
  setSchemaName: (name: string) => void;
  setSchemaDescription: (description: string) => void;

  toWorkflowSchema: () => WorkflowSchema;
}

const EDGE_TYPE_STYLES: Record<EdgeType, { stroke: string; strokeDasharray?: string }> = {
  data_flow: { stroke: "#4a9eff" },
  tool_call: { stroke: "#4ade80" },
  synchronization: { stroke: "#f59e0b", strokeDasharray: "6 3" },
};

function nodeDefinitionToReactFlowNode(definition: NodeDefinition): Node {
  return {
    id: definition.node_id,
    type: "agentNode",
    position: { x: definition.position.x, y: definition.position.y },
    data: {
      label: definition.label,
      nodeType: definition.node_type,
      config: definition.config,
      groupConfig: definition.group_config,
    },
  };
}

function edgeDefinitionToReactFlowEdge(definition: EdgeDefinition): Edge {
  const style = EDGE_TYPE_STYLES[definition.edge_type];
  return {
    id: definition.edge_id,
    source: definition.source_node_id,
    target: definition.target_node_id,
    type: definition.edge_type === "synchronization" ? "straight" : "smoothstep",
    animated: definition.edge_type === "synchronization",
    style,
    data: { edgeType: definition.edge_type },
  };
}

function reactFlowNodeToDefinition(node: Node): NodeDefinition {
  return {
    node_id: node.id,
    node_type: node.data.nodeType as NodeType,
    label: node.data.label as string,
    config: node.data.config as NodeConfig,
    group_config: (node.data.groupConfig as AgentGroupConfig) ?? null,
    position: { x: node.position.x, y: node.position.y },
  };
}

function reactFlowEdgeToDefinition(edge: Edge): EdgeDefinition {
  return {
    edge_id: edge.id,
    edge_type: (edge.data?.edgeType as EdgeType) ?? "data_flow",
    source_node_id: edge.source,
    target_node_id: edge.target,
  };
}

const DEFAULT_WORKFLOW_CONFIG: WorkflowConfig = {
  total_timeout: 300,
  logging_level: "info" as LoggingLevel,
  trace_enabled: true,
  dead_loop_detection: true,
};

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  schemaId: null,
  schemaName: "",
  schemaDescription: "",
  nodes: [],
  edges: [],
  workflowConfig: { ...DEFAULT_WORKFLOW_CONFIG },
  selectedNodeId: null,
  schemas: [],
  availableModels: [],
  availableTools: [],
  modelsByProvider: {},

  setSchemas: (schemas) => set({ schemas }),
  setAvailableModels: (models) => set({ availableModels: models }),
  setAvailableTools: (tools) => set({ availableTools: tools }),
  setModelsForProvider: (provider, models) =>
    set((state) => ({
      modelsByProvider: { ...state.modelsByProvider, [provider]: models },
    })),
  fetchModelsForProvider: async (provider: string): Promise<{ id: string; label: string }[]> => {
    const cached = get().modelsByProvider[provider];
    if (cached) return cached;
    try {
      const response = await fetch(`/api/models/${encodeURIComponent(provider)}`);
      if (!response.ok) return [];
      const data = await response.json();
      const raw = data.models ?? [];
      const models: { id: string; label: string }[] = raw.map(
        (m: { id: string; label: string } | string) =>
          typeof m === "string" ? { id: m, label: m } : m,
      );
      set((state) => ({
        modelsByProvider: { ...state.modelsByProvider, [provider]: models },
      }));
      return models;
    } catch {
      return [];
    }
  },

  setSchema: (schema) =>
    set({
      schemaId: schema.schema_id,
      schemaName: schema.name,
      schemaDescription: schema.description,
      nodes: schema.nodes.map(nodeDefinitionToReactFlowNode),
      edges: schema.edges.map(edgeDefinitionToReactFlowEdge),
      workflowConfig: schema.config,
      selectedNodeId: null,
    }),

  clearSchema: () =>
    set({
      schemaId: null,
      schemaName: "",
      schemaDescription: "",
      nodes: [],
      edges: [],
      workflowConfig: { ...DEFAULT_WORKFLOW_CONFIG },
      selectedNodeId: null,
    }),

  addNode: (definition) =>
    set((state) => ({
      nodes: [...state.nodes, nodeDefinitionToReactFlowNode(definition)],
    })),

  removeNode: (nodeId) =>
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== nodeId),
      edges: state.edges.filter(
        (e) => e.source !== nodeId && e.target !== nodeId
      ),
      selectedNodeId:
        state.selectedNodeId === nodeId ? null : state.selectedNodeId,
    })),

  updateNode: (nodeId, updates) =>
    set((state) => ({
      nodes: state.nodes.map((node) => {
        if (node.id !== nodeId) return node;
        return {
          ...node,
          data: {
            ...node.data,
            ...(updates.label !== undefined && { label: updates.label }),
            ...(updates.node_type !== undefined && { nodeType: updates.node_type }),
            ...(updates.config !== undefined && { config: updates.config }),
            ...(updates.group_config !== undefined && { groupConfig: updates.group_config }),
          },
          ...(updates.position !== undefined && {
            position: { x: updates.position.x, y: updates.position.y },
          }),
        };
      }),
    })),

  updateNodeConfig: (nodeId, configUpdates) =>
    set((state) => ({
      nodes: state.nodes.map((node) => {
        if (node.id !== nodeId) return node;
        return {
          ...node,
          data: {
            ...node.data,
            config: { ...(node.data.config as NodeConfig), ...configUpdates },
          },
        };
      }),
    })),

  updateGroupConfig: (nodeId, groupUpdates) =>
    set((state) => ({
      nodes: state.nodes.map((node) => {
        if (node.id !== nodeId) return node;
        const existing = (node.data.groupConfig as AgentGroupConfig) ?? {
          max_parallel_agents: 5,
          min_agents: 2,
          max_agents: 10,
          group_structure: "default",
          shared_state: {},
          tool_authorization: [],
        };
        return {
          ...node,
          data: {
            ...node.data,
            groupConfig: { ...existing, ...groupUpdates },
          },
        };
      }),
    })),

  addEdge: (definition) =>
    set((state) => ({
      edges: [...state.edges, edgeDefinitionToReactFlowEdge(definition)],
    })),

  removeEdge: (edgeId) =>
    set((state) => ({
      edges: state.edges.filter((e) => e.id !== edgeId),
    })),

  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  setSelectedNodeId: (nodeId) => set({ selectedNodeId: nodeId }),
  setWorkflowConfig: (config) =>
    set((state) => ({
      workflowConfig: { ...state.workflowConfig, ...config },
    })),
  setSchemaName: (name) => set({ schemaName: name }),
  setSchemaDescription: (description) => set({ schemaDescription: description }),

  toWorkflowSchema: (): WorkflowSchema => {
    const state = get();
    return {
      schema_id: state.schemaId ?? crypto.randomUUID(),
      name: state.schemaName,
      description: state.schemaDescription,
      nodes: state.nodes.map(reactFlowNodeToDefinition),
      edges: state.edges.map(reactFlowEdgeToDefinition),
      config: state.workflowConfig,
    };
  },
}));
