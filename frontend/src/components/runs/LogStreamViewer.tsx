/**
 * Scrolling log viewer — renders formatted log lines from the backend's
 * logging pipeline. Each line is parsed as JSON to extract timestamp,
 * level, and message. Falls back to raw text if not valid JSON.
 *
 * Adopts the same rendering approach as nocode-workflow's LogStreamViewer.
 */

import { useEffect, useMemo, useRef } from "react";
import type { RunLogLine } from "../../hooks/useRunLogStream";

export interface LogStreamViewerProps {
  lines: RunLogLine[];
  terminalStatus: string | null;
  connectionError: string | null;
}

interface ParsedLogRecord {
  timestamp: string | null;
  level: string | null;
  message: string;
}

function parseLogRecord(payload: string): ParsedLogRecord {
  try {
    const parsed = JSON.parse(payload);
    if (parsed && typeof parsed === "object") {
      return {
        timestamp:
          typeof parsed.timestamp === "string" ? parsed.timestamp : null,
        level: typeof parsed.level === "string" ? parsed.level : null,
        message:
          typeof parsed.message === "string" ? parsed.message : payload,
      };
    }
  } catch {
    // not JSON — use raw
  }
  return { timestamp: null, level: null, message: payload };
}

function levelClass(level: string | null): string {
  switch (level) {
    case "ERROR":
    case "CRITICAL":
      return "log-level-error";
    case "WARNING":
      return "log-level-warning";
    case "INFO":
      return "log-level-info";
    case "DEBUG":
      return "log-level-debug";
    default:
      return "";
  }
}

export default function LogStreamViewer({
  lines,
  terminalStatus,
  connectionError,
}: LogStreamViewerProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = scrollRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [lines, terminalStatus]);

  const parsedLines = useMemo(
    () =>
      lines.map((line) => ({
        sequenceNumber: line.sequenceNumber,
        ...parseLogRecord(line.payload),
      })),
    [lines]
  );

  return (
    <div className="log-stream-viewer">
      <div className="log-stream-header">
        <span className="log-stream-title">Live log</span>
        <span className="log-stream-status">
          {terminalStatus
            ? `Finished (${terminalStatus})`
            : connectionError
              ? `Disconnected: ${connectionError}`
              : `Streaming — ${lines.length} lines`}
        </span>
      </div>
      <div ref={scrollRef} className="log-stream-body">
        {parsedLines.length === 0 ? (
          <div className="log-stream-empty">Waiting for first log line...</div>
        ) : (
          parsedLines.map((line) => (
            <div
              key={line.sequenceNumber}
              className={`log-line ${levelClass(line.level)}`}
            >
              {line.timestamp && (
                <span className="log-timestamp">{line.timestamp}</span>
              )}
              {line.level && (
                <span className="log-level">{line.level}</span>
              )}
              <span className="log-message">{line.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
