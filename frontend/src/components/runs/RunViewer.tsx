/**
 * Run history table with live SSE event streaming.
 *
 * Lists past and current runs from the backend. When a run is selected,
 * opens an SSE connection to stream events in real time and renders each
 * event with type-colored labels, timestamps, node IDs, and data payloads.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { runsApi } from "../../api/client";
import { useWorkflowStore } from "../../store/workflowStore";
import type { RunRecord, RunEvent } from "../../types/schema";

export default function RunViewer() {
  const schemaId = useWorkflowStore((s) => s.schemaId);
  const [records, setRecords] = useState<RunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [streaming, setStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const streamEndRef = useRef<HTMLDivElement>(null);

  const loadRecords = useCallback(async () => {
    try {
      const list = await runsApi.listRecords();
      setRecords(list);
    } catch {
      /* Backend may be offline */
    }
  }, []);

  useEffect(() => {
    loadRecords();
  }, [loadRecords]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  useEffect(() => {
    streamEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const handleStartRun = useCallback(async () => {
    if (!schemaId) return;
    try {
      const record = await runsApi.start(schemaId);
      setRecords((prev) => [record, ...prev]);
      setSelectedRunId(record.run_id);
      setEvents([]);
      setStreaming(true);

      eventSourceRef.current?.close();
      eventSourceRef.current = runsApi.streamEvents(
        record.run_id,
        (event) => setEvents((prev) => [...prev, event]),
        () => setStreaming(false)
      );
    } catch (error) {
      throw new Error(
        `Failed to start run: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }, [schemaId]);

  const handleSelectRun = useCallback(
    (runId: string) => {
      eventSourceRef.current?.close();
      setSelectedRunId(runId);
      setEvents([]);
      setStreaming(false);

      const record = records.find((r) => r.run_id === runId);
      if (record?.status === "running") {
        setStreaming(true);
        eventSourceRef.current = runsApi.streamEvents(
          runId,
          (event) => setEvents((prev) => [...prev, event]),
          () => setStreaming(false)
        );
      }
    },
    [records]
  );

  const formatTimestamp = (iso: string) => {
    const date = new Date(iso);
    return date.toLocaleString();
  };

  const formatEventTime = (iso: string) => {
    const date = new Date(iso);
    return date.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  return (
    <div className="run-viewer">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700 }}>Run History</h2>
        <div className="btn-row">
          <button className="btn btn-sm" onClick={loadRecords}>
            Refresh
          </button>
          <button
            className="btn btn-primary btn-sm"
            onClick={handleStartRun}
            disabled={!schemaId}
          >
            Start Run
          </button>
        </div>
      </div>

      {records.length === 0 ? (
        <div className="empty-state">
          No runs yet. Load a schema and click Start Run.
        </div>
      ) : (
        <table className="run-table">
          <thead>
            <tr>
              <th>Run ID</th>
              <th>Schema</th>
              <th>Status</th>
              <th>Started</th>
              <th>Finished</th>
            </tr>
          </thead>
          <tbody>
            {records.map((record) => (
              <tr
                key={record.run_id}
                onClick={() => handleSelectRun(record.run_id)}
                style={{
                  cursor: "pointer",
                  background:
                    selectedRunId === record.run_id
                      ? "var(--color-bg-surface)"
                      : undefined,
                }}
              >
                <td style={{ fontFamily: "monospace", fontSize: 11 }}>
                  {record.run_id.slice(0, 8)}
                </td>
                <td>{record.schema_name}</td>
                <td>
                  <span className={`run-status status-${record.status}`}>
                    <span className="run-status-dot" />
                    {record.status}
                  </span>
                </td>
                <td className="text-muted text-sm">
                  {formatTimestamp(record.started_at)}
                </td>
                <td className="text-muted text-sm">
                  {record.finished_at
                    ? formatTimestamp(record.finished_at)
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedRunId && (
        <div className="event-stream">
          <div
            style={{
              padding: "8px 12px",
              borderBottom: "1px solid var(--color-border)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span className="text-sm" style={{ fontWeight: 600 }}>
              Events — {selectedRunId.slice(0, 8)}
            </span>
            {streaming && (
              <span className="run-status status-running">
                <span className="run-status-dot" />
                Streaming
              </span>
            )}
          </div>

          {events.length === 0 && !streaming && (
            <div className="empty-state">No events for this run.</div>
          )}

          {events.map((event) => (
            <div key={event.event_id} className="event-item">
              <span className="event-timestamp">
                {formatEventTime(event.timestamp)}
              </span>
              <span className={`event-type type-${event.event_type}`}>
                {event.event_type}
              </span>
              {event.node_id && (
                <span className="event-node-id">{event.node_id}</span>
              )}
              {event.data && Object.keys(event.data).length > 0 && (
                <div className="event-data">
                  {JSON.stringify(event.data, null, 2)}
                </div>
              )}
            </div>
          ))}
          <div ref={streamEndRef} />
        </div>
      )}
    </div>
  );
}
