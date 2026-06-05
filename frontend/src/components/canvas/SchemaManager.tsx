/**
 * Left sidebar: schema list with CRUD operations and workflow config editing.
 *
 * Displays all saved schemas, allows creating, loading, saving, and deleting
 * schemas. When a schema is loaded, shows editable workflow-level configuration
 * (timeout, logging, tracing, dead loop detection) and a Run button that
 * starts execution and navigates to the Runs tab.
 */

import { useState, useCallback } from "react";
import { useWorkflowStore } from "../../store/workflowStore";
import { schemasApi, runsApi } from "../../api/client";
import { LoggingLevel } from "../../types/schema";

interface SchemaManagerProps {
  onRunStart: () => void;
}

export default function SchemaManager({ onRunStart }: SchemaManagerProps) {
  const schemas = useWorkflowStore((storeState) => storeState.schemas);
  const schemaId = useWorkflowStore((storeState) => storeState.schemaId);
  const schemaName = useWorkflowStore((storeState) => storeState.schemaName);
  const schemaDescription = useWorkflowStore((storeState) => storeState.schemaDescription);
  const workflowConfig = useWorkflowStore((storeState) => storeState.workflowConfig);
  const setSchema = useWorkflowStore((storeState) => storeState.setSchema);
  const clearSchema = useWorkflowStore((storeState) => storeState.clearSchema);
  const setSchemas = useWorkflowStore((storeState) => storeState.setSchemas);
  const setSchemaName = useWorkflowStore((storeState) => storeState.setSchemaName);
  const setSchemaDescription = useWorkflowStore((storeState) => storeState.setSchemaDescription);
  const setWorkflowConfig = useWorkflowStore((storeState) => storeState.setWorkflowConfig);
  const toWorkflowSchema = useWorkflowStore((storeState) => storeState.toWorkflowSchema);

  const [createMode, setCreateMode] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [launching, setLaunching] = useState(false);

  const refreshSchemas = useCallback(async () => {
    const list = await schemasApi.list();
    setSchemas(list);
  }, [setSchemas]);

  const handleLoad = useCallback(
    async (targetSchemaId: string) => {
      const schema = await schemasApi.get(targetSchemaId);
      setSchema(schema);
    },
    [setSchema]
  );

  const handleCreate = useCallback(async () => {
    if (!newName.trim()) return;
    const schema = {
      schema_id: crypto.randomUUID(),
      name: newName.trim(),
      description: newDescription.trim(),
      nodes: [],
      edges: [],
      config: {
        total_timeout: 300,
        logging_level: LoggingLevel.INFO,
        trace_enabled: true,
        dead_loop_detection: true,
      },
    };
    const created = await schemasApi.create(schema);
    setSchema(created);
    await refreshSchemas();
    setCreateMode(false);
    setNewName("");
    setNewDescription("");
  }, [newName, newDescription, setSchema, refreshSchemas]);

  const handleSave = useCallback(async () => {
    if (!schemaId) return;
    setSaving(true);
    try {
      const workflowSchema = toWorkflowSchema();
      await schemasApi.update(schemaId, workflowSchema);
      await refreshSchemas();
    } finally {
      setSaving(false);
    }
  }, [schemaId, toWorkflowSchema, refreshSchemas]);

  const handleDelete = useCallback(
    async (targetSchemaId: string) => {
      await schemasApi.delete(targetSchemaId);
      if (schemaId === targetSchemaId) clearSchema();
      await refreshSchemas();
    },
    [schemaId, clearSchema, refreshSchemas]
  );

  const handleRun = useCallback(async () => {
    if (!schemaId) return;
    setLaunching(true);
    try {
      await runsApi.start(schemaId);
      onRunStart();
    } finally {
      setLaunching(false);
    }
  }, [schemaId, onRunStart]);

  return (
    <div className="panel">
      <div className="panel-header">Schemas</div>
      <div className="panel-body">
        <div className="btn-row mb-12">
          <button className="btn btn-primary btn-sm" onClick={() => setCreateMode(!createMode)}>
            {createMode ? "Cancel" : "New Schema"}
          </button>
          {schemaId && (
            <button className="btn btn-success btn-sm" onClick={handleSave} disabled={saving}>
              {saving ? "Saving..." : "Save"}
            </button>
          )}
        </div>

        {createMode && (
          <div className="mb-12">
            <div className="form-group">
              <label className="form-label">Name</label>
              <input
                className="form-input"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Workflow name"
              />
            </div>
            <div className="form-group">
              <label className="form-label">Description</label>
              <textarea
                className="form-textarea"
                value={newDescription}
                onChange={(e) => setNewDescription(e.target.value)}
                placeholder="What does this workflow do?"
              />
            </div>
            <button className="btn btn-primary btn-sm" onClick={handleCreate}>
              Create
            </button>
          </div>
        )}

        {schemas.length === 0 && !createMode && (
          <div className="empty-state">No schemas yet. Create one to begin.</div>
        )}

        {schemas.map((schema) => (
          <div
            key={schema.schema_id}
            className={`schema-list-item ${schemaId === schema.schema_id ? "active" : ""}`}
            onClick={() => handleLoad(schema.schema_id)}
          >
            <div className="schema-list-item-name">{schema.name}</div>
            <div className="schema-list-item-desc">{schema.description}</div>
            {schemaId === schema.schema_id && (
              <button
                className="btn btn-danger btn-sm mt-8"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(schema.schema_id);
                }}
              >
                Delete
              </button>
            )}
          </div>
        ))}

        {schemaId && (
          <>
            <hr className="divider" />
            <div className="panel-header" style={{ padding: "12px 0 8px", border: "none" }}>
              Workflow Config
            </div>

            <div className="form-group">
              <label className="form-label">Schema Name</label>
              <input
                className="form-input"
                value={schemaName}
                onChange={(e) => setSchemaName(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Description</label>
              <textarea
                className="form-textarea"
                value={schemaDescription}
                onChange={(e) => setSchemaDescription(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Total Timeout (seconds)</label>
              <input
                type="number"
                className="form-input"
                value={workflowConfig.total_timeout}
                onChange={(e) =>
                  setWorkflowConfig({ total_timeout: parseInt(e.target.value) || 0 })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Logging Level</label>
              <select
                className="form-select"
                value={workflowConfig.logging_level}
                onChange={(e) =>
                  setWorkflowConfig({
                    logging_level: e.target.value as LoggingLevel,
                  })
                }
              >
                {Object.values(LoggingLevel).map((level) => (
                  <option key={level} value={level}>
                    {level.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="form-checkbox">
                <input
                  type="checkbox"
                  checked={workflowConfig.trace_enabled}
                  onChange={(e) =>
                    setWorkflowConfig({ trace_enabled: e.target.checked })
                  }
                />
                Trace Enabled
              </label>
            </div>

            <div className="form-group">
              <label className="form-checkbox">
                <input
                  type="checkbox"
                  checked={workflowConfig.dead_loop_detection}
                  onChange={(e) =>
                    setWorkflowConfig({ dead_loop_detection: e.target.checked })
                  }
                />
                Dead Loop Detection
              </label>
            </div>

            <hr className="divider" />
            <button
              className="btn btn-run"
              onClick={handleRun}
              disabled={launching}
              style={{ width: "100%" }}
            >
              {launching ? "Launching..." : "Run Workflow"}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
