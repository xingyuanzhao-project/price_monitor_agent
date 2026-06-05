/**
 * API client for communicating with the price_monitor_agent backend.
 *
 * Provides typed functions for schema CRUD, run management, settings,
 * and model discovery. Uses fetch with JSON serialization and SSE for
 * streaming run events.
 */

import type {
  WorkflowSchema,
  UserSettings,
  RunRecord,
  RunEvent,
} from "../types/schema";

const API_BASE = "/api";

async function request<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(
      `API ${options.method ?? "GET"} ${path} failed (${response.status}): ${errorBody}`
    );
  }
  return response.json();
}

export const schemasApi = {
  list(): Promise<WorkflowSchema[]> {
    return request("/schemas");
  },

  get(schemaId: string): Promise<WorkflowSchema> {
    return request(`/schemas/${encodeURIComponent(schemaId)}`);
  },

  create(schema: WorkflowSchema): Promise<WorkflowSchema> {
    return request("/schemas", {
      method: "POST",
      body: JSON.stringify(schema),
    });
  },

  update(schemaId: string, schema: WorkflowSchema): Promise<WorkflowSchema> {
    return request(`/schemas/${encodeURIComponent(schemaId)}`, {
      method: "PUT",
      body: JSON.stringify(schema),
    });
  },

  delete(schemaId: string): Promise<void> {
    return request(`/schemas/${encodeURIComponent(schemaId)}`, {
      method: "DELETE",
    });
  },

  validate(schema: WorkflowSchema): Promise<{ valid: boolean; errors: string[] }> {
    return request("/schemas/validate", {
      method: "POST",
      body: JSON.stringify(schema),
    });
  },
};

export const runsApi = {
  start(schemaId: string): Promise<RunRecord> {
    return request("/runs", {
      method: "POST",
      body: JSON.stringify({ schema_id: schemaId }),
    });
  },

  streamEvents(
    runId: string,
    onEvent: (event: RunEvent) => void,
    onError: (error: Error) => void
  ): EventSource {
    const source = new EventSource(
      `${API_BASE}/runs/${encodeURIComponent(runId)}/events`
    );
    source.onmessage = (message) => {
      const event: RunEvent = JSON.parse(message.data);
      onEvent(event);
    };
    source.onerror = () => {
      onError(new Error(`SSE connection lost for run ${runId}`));
      source.close();
    };
    return source;
  },

  listRecords(): Promise<RunRecord[]> {
    return request("/runs");
  },
};

export const settingsApi = {
  get(): Promise<UserSettings> {
    return request("/settings");
  },

  save(settings: UserSettings): Promise<UserSettings> {
    return request("/settings", {
      method: "PUT",
      body: JSON.stringify(settings),
    });
  },
};

export const modelsApi = {
  async list(): Promise<string[]> {
    const data = await request<{
      providers: Array<{ provider_name: string; models: string[]; error: string | null }>;
    }>("/models");
    return data.providers.flatMap((provider) => provider.models ?? []);
  },

  listTools(): Promise<string[]> {
    return request("/models/tools");
  },
};
