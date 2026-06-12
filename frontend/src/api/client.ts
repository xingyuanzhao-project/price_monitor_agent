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
  ProviderStatusResponse,
  ApiKeyTestResponse,
  LocalEndpointTestResponse,
  DataSourcesResponse,
  RunRecord,
  RunEvent,
  ToolCategory,
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
    onDone: () => void
  ): EventSource {
    const source = new EventSource(
      `${API_BASE}/runs/${encodeURIComponent(runId)}/events`
    );
    source.onmessage = (message) => {
      const event: RunEvent = JSON.parse(message.data);
      onEvent(event);
    };
    source.onerror = () => {
      source.close();
      onDone();
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

  providerStatus(): Promise<ProviderStatusResponse> {
    return request("/settings/provider-status");
  },

  setApiKey(providerName: string, apiKey: string): Promise<ProviderStatusResponse> {
    return request("/settings/api-key", {
      method: "POST",
      body: JSON.stringify({ provider_name: providerName, api_key: apiKey }),
    });
  },

  testApiKey(providerName: string, apiKey: string): Promise<ApiKeyTestResponse> {
    return request("/settings/api-key/test", {
      method: "POST",
      body: JSON.stringify({ provider_name: providerName, api_key: apiKey }),
    });
  },

  setLocalEndpoint(providerName: string, apiBase: string): Promise<ProviderStatusResponse> {
    return request("/settings/local-endpoint", {
      method: "POST",
      body: JSON.stringify({ provider_name: providerName, api_base: apiBase }),
    });
  },

  testLocalEndpoint(apiBase: string): Promise<LocalEndpointTestResponse> {
    return request("/settings/local-endpoint/test", {
      method: "POST",
      body: JSON.stringify({ api_base: apiBase }),
    });
  },

  getDataSources(): Promise<DataSourcesResponse> {
    return request("/settings/data-sources");
  },

  togglePublicSource(sourceId: string, enabled: boolean): Promise<void> {
    return request("/settings/data-sources/public/toggle", {
      method: "POST",
      body: JSON.stringify({ source_id: sourceId, enabled }),
    });
  },

  togglePublicSourceBatch(sourceIds: string[], enabled: boolean): Promise<void> {
    return request("/settings/data-sources/public/toggle-batch", {
      method: "POST",
      body: JSON.stringify({ source_ids: sourceIds, enabled }),
    });
  },

  addAdditionalApi(sourceId: string, apiKey: string, baseUrl: string = ""): Promise<void> {
    return request("/settings/data-sources/additional", {
      method: "POST",
      body: JSON.stringify({ source_id: sourceId, api_key: apiKey, base_url: baseUrl }),
    });
  },

  removeAdditionalApi(sourceId: string): Promise<void> {
    return request(`/settings/data-sources/additional/${encodeURIComponent(sourceId)}`, {
      method: "DELETE",
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

  async listForProvider(providerName: string): Promise<string[]> {
    const data = await request<{
      provider_name: string;
      models: string[];
      error: string | null;
    }>(`/models/${encodeURIComponent(providerName)}`);
    return data.models ?? [];
  },

  async listTools(): Promise<{ tools: string[]; hierarchy: ToolCategory[] }> {
    const data = await request<{
      tools: Array<{ type: string; function: { name: string } }>;
      hierarchy: ToolCategory[];
    }>("/models/tools");
    return {
      tools: data.tools.map((t) => t.function.name),
      hierarchy: data.hierarchy ?? [],
    };
  },
};


export function openRunLogStream(runId: string): EventSource {
  return new EventSource(`${API_BASE}/runs/${encodeURIComponent(runId)}/logs/stream`);
}
