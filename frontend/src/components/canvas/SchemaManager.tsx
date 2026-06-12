/**
 * Left sidebar: schema list with CRUD operations and workflow config editing.
 *
 * Displays all saved schemas, allows creating, loading, saving, and deleting
 * schemas. When a schema is loaded, shows editable workflow-level configuration
 * (timeout, logging, tracing, dead loop detection) and a Run button that
 * starts execution and navigates to the Runs tab.
 *
 * New Schema flow:
 *   1. Click "New Schema" → setSchema with empty nodes/edges → canvas clears.
 *   2. User edits canvas (adds nodes/edges) and config (name, desc, timeout…).
 *   3. Click "Save" → toWorkflowSchema captures current store → persists.
 *
 * This is the same general path used when loading and re-saving an existing
 * schema. No separate create form or create handler exists.
 */

import { useState, useCallback } from "react";
import { useWorkflowStore, DEFAULT_WORKFLOW_CONFIG } from "../../store/workflowStore";
import { schemasApi, runsApi } from "../../api/client";
import { LoggingLevel } from "../../types/schema";

interface SchemaManagerProps {
  onRunStart: () => void;
}

export default function SchemaManager({ onRunStart }: SchemaManagerProps) {
  const schemas = useWorkflowStore((s) => s.schemas);
  const schemaId = useWorkflowStore((s) => s.schemaId);
  const schemaName = useWorkflowStore((s) => s.schemaName);
  const schemaDescription = useWorkflowStore((s) => s.schemaDescription);
  const workflowConfig = useWorkflowStore((s) => s.workflowConfig);
  const setSchema = useWorkflowStore((s) => s.setSchema);
  const clearSchema = useWorkflowStore((s) => s.clearSchema);
  const setSchemas = useWorkflowStore((s) => s.setSchemas);
  const setSchemaName = useWorkflowStore((s) => s.setSchemaName);
  const setSchemaDescription = useWorkflowStore((s) => s.setSchemaDescription);
  const setWorkflowConfig = useWorkflowStore((s) => s.setWorkflowConfig);
  const toWorkflowSchema = useWorkflowStore((s) => s.toWorkflowSchema);

  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [launching, setLaunching] = useState(false);

  const refreshSchemas = useCallback(async () => {
    const list = await schemasApi.list();
    setSchemas(list);
  }, [setSchemas]);

  const handleLoad = useCallback(
    async (targetSchemaId: string) => {
      const schema = await schemasApi.get(targetSchemaId);
      setSchema(schema);
      setSaveError(null);
    },
    [setSchema]
  );

  const handleNewSchema = useCallback(() => {
    setSchema({
      schema_id: crypto.randomUUID(),
      name: "Untitled",
      description: "",
      nodes: [],
      edges: [],
      config: { ...DEFAULT_WORKFLOW_CONFIG },
    });
    setSaveError(null);
  }, [setSchema]);

  const handleSave = useCallback(async () => {
    if (!schemaId) return;
    setSaving(true);
    setSaveError(null);
    try {
      const workflowSchema = toWorkflowSchema();
      const isNew = !schemas.some((s) => s.schema_id === schemaId);
      if (isNew) {
        await schemasApi.create(workflowSchema);
      } else {
        await schemasApi.update(schemaId, workflowSchema);
      }
      await refreshSchemas();
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setSaveError(message);
      throw err;
    } finally {
      setSaving(false);
    }
  }, [schemaId, schemas, toWorkflowSchema, refreshSchemas]);

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
    <div className="panel schema-manager-panel">
      <div className="panel-header">Schemas</div>
      <div className="panel-body">
        <div className="btn-row mb-12">
          <button className="btn btn-primary btn-sm" onClick={handleNewSchema}>
            New Schema
          </button>
          <button
            className="btn btn-success btn-sm"
            onClick={handleSave}
            disabled={saving || !schemaId}
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>

        {saveError && (
          <div className="save-error-banner">{saveError}</div>
        )}

        <div className="schema-list-scroll">
          {schemas.length === 0 && (
            <div className="empty-state">No schemas yet. Create one to begin.</div>
          )}

          {schemas.map((schema) => (
            <div
              key={schema.schema_id}
              className={`schema-list-item ${schemaId === schema.schema_id ? "active" : ""}`}
              onClick={() => handleLoad(schema.schema_id)}
            >
              <div className="schema-list-item-row">
                <div className="schema-list-item-name">{schema.name}</div>
                {schemaId === schema.schema_id && (
                  <button
                    className="btn btn-danger btn-xs"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(schema.schema_id);
                    }}
                  >
                    Delete
                  </button>
                )}
              </div>
              <div className="schema-list-item-desc">{schema.description}</div>
              {schemaId === schema.schema_id && (
                <button
                  className="btn btn-run btn-sm mt-8"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRun();
                  }}
                  disabled={launching}
                  style={{ width: "100%" }}
                >
                  {launching ? "Launching..." : "Run Workflow"}
                </button>
              )}
            </div>
          ))}
        </div>

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
              <label className="form-label">Max Iterations</label>
              <input
                type="number"
                className="form-input"
                min={1}
                value={workflowConfig.max_iterations}
                onChange={(e) =>
                  setWorkflowConfig({ max_iterations: Math.max(1, parseInt(e.target.value) || 1) })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Iteration Sleep (seconds)</label>
              <input
                type="number"
                className="form-input"
                min={0}
                step={0.1}
                value={workflowConfig.iteration_sleep}
                onChange={(e) =>
                  setWorkflowConfig({ iteration_sleep: Math.max(0, parseFloat(e.target.value) || 0) })
                }
              />
            </div>

            <div className="form-group">
              <label className="form-label">Max Loop Rounds</label>
              <input
                type="number"
                className="form-input"
                min={1}
                value={workflowConfig.max_loop_rounds}
                onChange={(e) =>
                  setWorkflowConfig({ max_loop_rounds: Math.max(1, parseInt(e.target.value) || 1) })
                }
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
