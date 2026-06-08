/**
 * Hook that subscribes to GET /api/runs/:run_id/logs/stream SSE endpoint
 * and feeds a bounded in-memory buffer of log lines.
 *
 * Adopts the same pattern as nocode-workflow's use_run_log_stream.
 */

import { useEffect, useState } from "react";
import { openRunLogStream } from "../api/client";

export interface RunLogLine {
  sequenceNumber: number;
  payload: string;
}

interface UseRunLogStreamResult {
  lines: RunLogLine[];
  terminalStatus: string | null;
  connectionError: string | null;
}

const MAX_BUFFERED_LINES = 2000;

export function useRunLogStream(
  runId: string | null
): UseRunLogStreamResult {
  const [lines, setLines] = useState<RunLogLine[]>([]);
  const [terminalStatus, setTerminalStatus] = useState<string | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  useEffect(() => {
    if (runId === null) {
      return;
    }
    setLines([]);
    setTerminalStatus(null);
    setConnectionError(null);
    let sequenceCounter = 0;

    const eventSource = openRunLogStream(runId);

    const handleLogEvent = (event: MessageEvent) => {
      sequenceCounter += 1;
      const newLine: RunLogLine = {
        sequenceNumber: sequenceCounter,
        payload: event.data,
      };
      setLines((prev) => {
        const next = [...prev, newLine];
        if (next.length > MAX_BUFFERED_LINES) {
          return next.slice(-MAX_BUFFERED_LINES);
        }
        return next;
      });
    };

    const handleStatusEvent = (event: MessageEvent) => {
      setTerminalStatus(event.data);
      eventSource.close();
    };

    const handleError = () => {
      setConnectionError("Log stream connection lost.");
      eventSource.close();
    };

    eventSource.addEventListener("log", handleLogEvent as EventListener);
    eventSource.addEventListener("status", handleStatusEvent as EventListener);
    eventSource.addEventListener("error", handleError);

    return () => {
      eventSource.removeEventListener("log", handleLogEvent as EventListener);
      eventSource.removeEventListener("status", handleStatusEvent as EventListener);
      eventSource.removeEventListener("error", handleError);
      eventSource.close();
    };
  }, [runId]);

  return { lines, terminalStatus, connectionError };
}
