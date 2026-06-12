/**
 * Root application component with tab navigation.
 *
 * Renders three tabs — Canvas (workflow editor), Runs (execution history),
 * and Settings (LLM providers, API credentials, global defaults).
 * On mount, fetches schemas and available models from the backend.
 */

import { useEffect, useState, useCallback } from "react";
import SchemaManager from "./components/canvas/SchemaManager";
import WorkflowCanvas from "./components/canvas/WorkflowCanvas";
import NodeConfigPanel from "./components/canvas/NodeConfigPanel";
import RunViewer from "./components/runs/RunViewer";
import SettingsPanel from "./components/settings/SettingsPanel";
import { useWorkflowStore } from "./store/workflowStore";
import { schemasApi, modelsApi } from "./api/client";

type TabId = "canvas" | "runs" | "settings";

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>("canvas");
  const selectedNodeId = useWorkflowStore((storeState) => storeState.selectedNodeId);
  const setSchemas = useWorkflowStore((storeState) => storeState.setSchemas);
  const setAvailableModels = useWorkflowStore((storeState) => storeState.setAvailableModels);
  const setAvailableTools = useWorkflowStore((storeState) => storeState.setAvailableTools);
  const setToolHierarchy = useWorkflowStore((storeState) => storeState.setToolHierarchy);

  const loadInitialData = useCallback(async () => {
    const [schemasResult, modelsResult, toolsResult] = await Promise.allSettled([
      schemasApi.list(),
      modelsApi.list(),
      modelsApi.listTools(),
    ]);
    if (schemasResult.status === "fulfilled") setSchemas(schemasResult.value);
    if (modelsResult.status === "fulfilled") setAvailableModels(modelsResult.value);
    if (toolsResult.status === "fulfilled") {
      setAvailableTools(toolsResult.value.tools);
      setToolHierarchy(toolsResult.value.hierarchy);
    }
  }, [setSchemas, setAvailableModels, setAvailableTools, setToolHierarchy]);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  return (
    <>
      <nav className="nav-bar">
        <span className="nav-brand">Price Monitor Agent</span>
        {(["canvas", "runs", "settings"] as TabId[]).map((tab) => (
          <button
            key={tab}
            className={`nav-tab ${activeTab === tab ? "active" : ""}`}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </nav>

      <div className="app-content">
        {activeTab === "canvas" && (
          <div
            className={`canvas-layout ${selectedNodeId ? "" : "no-selection"}`}
          >
            <SchemaManager onRunStart={() => setActiveTab("runs")} />
            <WorkflowCanvas />
            {selectedNodeId && <NodeConfigPanel />}
          </div>
        )}

        {activeTab === "runs" && <RunViewer />}
        {activeTab === "settings" && <SettingsPanel />}
      </div>
    </>
  );
}
