/**
 * Runs viewer: left panel of run list, right panel of live event stream,
 * bottom terminal showing raw event payloads for the selected run.
 *
 * Run list shows all runs (active and historical) in a single scrollable list
 * with status badges. Selecting a run opens its event stream. If the run is
 * still active, an SSE connection is established to receive events in real time.
 * The bottom terminal renders raw JSON payloads in chronological order.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { runsApi } from "../../api/client";
import type { RunRecord, RunEvent } from "../../types/schema";

export default function RunViewer() {
  const [records, setRecords] = useState<RunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [streaming, setStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);
  const terminalEndRef = useRef<HTMLDivElement>(null);

  const loadRecords = useCallback(async () => {
    const list = await runsApi.listRecords();
    setRecords(list);
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
    eventsEndRef.current?.scrollIntoView({ behavior: "smooth" });
    terminalEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  const openEventStream = useCallback((runId: string) => {
    eventSourceRef.current?.close();
    setStreaming(true);
    eventSourceRef.current = runsApi.streamEvents(
      runId,
      (event) => setEvents((previous) => [...previous, event]),
      () => setStreaming(false)
    );
  }, []);

  const handleSelectRun = useCallback(
    (runId: string) => {
      eventSourceRef.current?.close();
      setSelectedRunId(runId);
      setEvents([]);
      setStreaming(false);

      const record = records.find((record) => record.run_id === runId);
      if (record?.status === "running") {
        openEventStream(runId);
      }
    },
    [records, openEventStream]
  );

  const formatTimestamp = (iso: string) => {
    return new Date(iso).toLocaleString();
  };

  const formatEventTime = (iso: string) => {
    return new Date(iso).toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  const selectedRecord = records.find((record) => record.run_id === selectedRunId);

  return (
    <div className="run-viewer-layout">

      {/* ── Left panel: run list ────────────────────────────────────── */}
      <div className="runs-list-panel">
        <div className="runs-list-header">
          <span>Runs</span>
          <button className="btn btn-sm" onClick={loadRecords}>
            Refresh
          </button>
        </div>

        {records.length === 0 ? (
          <div className="empty-state" style={{ padding: "24px 12px" }}>
            No runs yet. Load a schema and click Run Workflow.
          </div>
        ) : (
          records.map((record) => (
            <div
              key={record.run_id}
              className={`run-list-item ${selectedRunId === record.run_id ? "active" : ""}`}
              onClick={() => handleSelectRun(record.run_id)}
            >
              <div className="run-list-item-top">
                <span className={`run-status-badge status-${record.status}`}>
                  <span className="run-status-dot" />
                  {record.status}
                </span>
                <span className="run-list-item-id">{record.run_id.slice(0, 8)}</span>
              </div>
              <div className="run-list-item-schema">{record.schema_name}</div>
              <div className="run-list-item-time">{formatTimestamp(record.started_at)}</div>
            </div>
          ))
        )}
      </div>

      {/* ── Right panel: event stream ────────────────────────────────── */}
      <div className="run-events-panel">
        {selectedRunId ? (
          <>
            <div className="run-events-header">
              <div className="run-events-title">
                {selectedRecord?.schema_name ?? selectedRunId.slice(0, 8)}
                <span style={{ marginLeft: 8, fontFamily: "monospace", fontSize: 11, color: "var(--color-text-muted)" }}>
                  {selectedRunId.slice(0, 8)}
                </span>
              </div>
              {streaming && (
                <span className="run-status-badge status-running">
                  <span className="run-status-dot" />
                  Live
                </span>
              )}
            </div>

            <div className="run-events-body">
              {events.length === 0 && !streaming && (
                <div className="empty-state">No events recorded for this run.</div>
              )}
              {events.map((event) => (
                <div key={event.event_id} className="event-item">
                  <span className="event-timestamp">{formatEventTime(event.timestamp)}</span>
                  <span className={`event-type type-${event.event_type}`}>{event.event_type}</span>
                  {event.node_id && (
                    <span className="event-node-id">{event.node_id}</span>
                  )}
                  {event.data && Object.keys(event.data).length > 0 && (
                    <div className="event-data">{JSON.stringify(event.data, null, 2)}</div>
                  )}
                </div>
              ))}
              <div ref={eventsEndRef} />
            </div>
          </>
        ) : (
          <div className="empty-state" style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
            Select a run to view its events.
          </div>
        )}
      </div>

      {/* ── Bottom terminal ──────────────────────────────────────────── */}
      <div className="run-terminal-panel">
        <div className="run-terminal-bar">Terminal</div>
        <div className="run-terminal-body">
          {events.length === 0 ? (
            <span style={{ color: "var(--color-text-muted)" }}>No output.</span>
          ) : (
            events.map((event) => (
              <div key={`term-${event.event_id}`} className="terminal-line">
                <span className="terminal-prompt">&gt;</span>
                <span className={`terminal-event-type type-${event.event_type}`}>
                  [{event.event_type}]
                </span>
                {event.node_id && (
                  <span className="terminal-node-id">{event.node_id}</span>
                )}
                <span className="terminal-payload">
                  {JSON.stringify(event.data)}
                </span>
              </div>
            ))
          )}
          <div ref={terminalEndRef} />
        </div>
      </div>

    </div>
  );
}
