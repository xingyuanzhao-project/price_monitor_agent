/**
 * Runs viewer: left panel of run list, right panel of live event stream,
 * bottom panel with real log terminal powered by the backend's Python
 * logging pipeline streamed via SSE.
 */

import { useState, useEffect, useCallback, useRef } from "react";
import { runsApi } from "../../api/client";
import { useRunLogStream } from "../../hooks/useRunLogStream";
import LogStreamViewer from "./LogStreamViewer";
import type { RunRecord, RunEvent } from "../../types/schema";

const RUN_LIST_POLL_MS = 2000;

export default function RunViewer() {
  const [records, setRecords] = useState<RunRecord[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [streaming, setStreaming] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { lines, terminalStatus, connectionError } =
    useRunLogStream(selectedRunId);

  const loadRecords = useCallback(async () => {
    const list = await runsApi.listRecords();
    setRecords(list);
  }, []);

  useEffect(() => {
    loadRecords();
  }, [loadRecords]);

  // Auto-poll while any run is in "running" state
  useEffect(() => {
    const hasRunning = records.some((r) => r.status === "running");
    if (hasRunning && !pollRef.current) {
      pollRef.current = setInterval(loadRecords, RUN_LIST_POLL_MS);
    } else if (!hasRunning && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [records, loadRecords]);

  useEffect(() => {
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: "smooth" });
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
      openEventStream(runId);
    },
    [openEventStream]
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

  const selectedRecord = records.find((r) => r.run_id === selectedRunId);

  return (
    <div className="run-viewer-layout">

      {/* Left panel: run list */}
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

      {/* Right panel: event stream + log terminal */}
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

      {/* Bottom panel: real log terminal */}
      <div className="run-terminal-panel">
        <LogStreamViewer
          lines={lines}
          terminalStatus={terminalStatus}
          connectionError={connectionError}
        />
      </div>

    </div>
  );
}
