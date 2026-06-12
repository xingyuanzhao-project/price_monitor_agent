/**
 * Settings page — cloud key management, local endpoint config, and data sources.
 *
 * Cloud providers (OpenRouter, OpenAI, Anthropic, Google): set API keys via
 * masked input with Test + Save.
 * Local providers (Ollama, vLLM, llama.cpp): set base URL and test
 * connectivity.
 * Public Data Tools: enable/disable free public data sources.
 * Additional Tool APIs: manage third-party data source credentials.
 *
 * Ported from nocode-workflow/gui/src/pages/SettingsPage.tsx.
 */

import { useState, useCallback, useEffect } from "react";
import { settingsApi } from "../../api/client";
import type {
  ProviderKeyStatus,
  LocalEndpointStatus,
  ProviderStatusResponse,
  ApiKeyTestResponse,
  LocalEndpointTestResponse,
  DataSourcesResponse,
  PublicDataSourceEntry,
  AdditionalApiSourceEntry,
  ConfiguredAdditionalApi,
} from "../../types/schema";

const CLOUD_PROVIDER_LABELS: Record<string, string> = {
  openrouter: "OpenRouter",
  openai: "OpenAI",
  anthropic: "Anthropic",
  google: "Google",
};

const DATA_CATEGORY_LABELS: Record<string, string> = {
  macro: "Macro",
  exchange: "Exchange",
  news: "News",
  social: "Social Media",
};

const LOCAL_PROVIDER_LABELS: Record<string, string> = {
  ollama: "Ollama",
  vllm: "vLLM",
  llama_cpp: "llama.cpp",
};

const LOCAL_PROVIDER_PLACEHOLDER: Record<string, string> = {
  ollama: "http://localhost:11434/v1",
  vllm: "http://localhost:8000/v1",
  llama_cpp: "http://localhost:8080/v1",
};

export default function SettingsPanel() {
  const [status, setStatus] = useState<ProviderStatusResponse | null>(null);
  const [dataSources, setDataSources] = useState<DataSourcesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = useCallback(async () => {
    try {
      const data = await settingsApi.providerStatus();
      setStatus(data);
      setError(null);
    } catch (err) {
      setError("Failed to load provider status.");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadDataSources = useCallback(async () => {
    try {
      const data = await settingsApi.getDataSources();
      setDataSources(data);
    } catch {
      // non-fatal: data sources panel won't render
    }
  }, []);

  useEffect(() => {
    loadStatus();
    loadDataSources();
  }, [loadStatus, loadDataSources]);

  if (loading) {
    return (
      <div className="settings-page">
        <div className="empty-state">Loading provider status...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="settings-page">
        <div className="error-banner">{error}</div>
      </div>
    );
  }

  return (
    <div className="settings-page">
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700 }}>API Keys</h2>
        <p className="text-sm text-muted" style={{ marginTop: 4 }}>
          Manage API keys and endpoints for LLM providers. Keys are stored in
          the server&apos;s .env file.
        </p>
      </div>

      <section className="settings-section">
        <div className="settings-section-title">Cloud Providers</div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {status?.cloud_providers.map((item) => (
            <CloudProviderRow
              key={item.provider_name}
              item={item}
              onSaved={loadStatus}
            />
          ))}
        </div>
      </section>

      <section className="settings-section">
        <div className="settings-section-title">Local Servers</div>
        <p className="text-sm text-muted" style={{ marginBottom: 12 }}>
          Connect to Ollama, vLLM, or llama.cpp servers running on your
          machine. All use the OpenAI-compatible API format.
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {status?.local_endpoints.map((item) => (
            <LocalEndpointRow
              key={item.provider_name}
              item={item}
              onSaved={loadStatus}
            />
          ))}
        </div>
      </section>

      {dataSources && (
        <PublicDataToolsSection
          dataSources={dataSources}
          onChanged={loadDataSources}
        />
      )}

      {dataSources && (
        <AdditionalToolApisSection
          dataSources={dataSources}
          onChanged={loadDataSources}
        />
      )}
    </div>
  );
}

function CloudProviderRow({
  item,
  onSaved,
}: {
  item: ProviderKeyStatus;
  onSaved: () => void;
}) {
  const [keyInput, setKeyInput] = useState("");
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<ApiKeyTestResponse | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const label = CLOUD_PROVIDER_LABELS[item.provider_name] ?? item.provider_name;

  const handleTest = useCallback(async () => {
    if (!keyInput.trim()) return;
    setTesting(true);
    setTestResult(null);
    try {
      const result = await settingsApi.testApiKey(item.provider_name, keyInput.trim());
      setTestResult(result);
    } catch {
      setTestResult({ provider_name: item.provider_name, valid: false, message: "Test request failed." });
    } finally {
      setTesting(false);
    }
  }, [keyInput, item.provider_name]);

  const handleSave = useCallback(async () => {
    if (!keyInput.trim()) return;
    setSaving(true);
    setSaveError(null);
    try {
      await settingsApi.setApiKey(item.provider_name, keyInput.trim());
      setKeyInput("");
      setTestResult(null);
      onSaved();
    } catch {
      setSaveError("Failed to save key.");
    } finally {
      setSaving(false);
    }
  }, [keyInput, item.provider_name, onSaved]);

  return (
    <div className="provider-card">
      <div className="provider-card-header">
        <span className="provider-card-title">{label}</span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {item.configured && (
            <span className="status-badge status-badge-ok">Configured</span>
          )}
          <span className="text-xs text-muted">env: {item.api_key_env}</span>
        </div>
      </div>

      <div className="tag-input-row" style={{ marginTop: 12 }}>
        <input
          className="form-input"
          type="password"
          value={keyInput}
          onChange={(e) => {
            setKeyInput(e.target.value);
            setTestResult(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSave();
          }}
          placeholder={`Enter ${label} API key`}
        />
        <button
          className="btn btn-sm"
          onClick={handleTest}
          disabled={!keyInput.trim() || testing}
        >
          {testing ? "Testing..." : "Test"}
        </button>
        <button
          className="btn btn-primary btn-sm"
          onClick={handleSave}
          disabled={!keyInput.trim() || saving}
        >
          {saving ? "Saving..." : "Save"}
        </button>
      </div>

      {testResult && (
        <div
          className={`feedback-banner ${testResult.valid ? "feedback-ok" : "feedback-error"}`}
        >
          {testResult.message}
        </div>
      )}
      {saveError && (
        <div className="feedback-banner feedback-error">{saveError}</div>
      )}
    </div>
  );
}

function LocalEndpointRow({
  item,
  onSaved,
}: {
  item: LocalEndpointStatus;
  onSaved: () => void;
}) {
  const [urlInput, setUrlInput] = useState(item.configured ? item.api_base : "");
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<LocalEndpointTestResponse | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);

  const label = LOCAL_PROVIDER_LABELS[item.provider_name] ?? item.provider_name;
  const placeholder = LOCAL_PROVIDER_PLACEHOLDER[item.provider_name] ?? "http://localhost:8000/v1";

  const handleTest = useCallback(async () => {
    if (!urlInput.trim()) return;
    setTesting(true);
    setTestResult(null);
    try {
      const result = await settingsApi.testLocalEndpoint(urlInput.trim());
      setTestResult(result);
    } catch {
      setTestResult({ reachable: false, models: [], message: "Test request failed." });
    } finally {
      setTesting(false);
    }
  }, [urlInput]);

  const handleSave = useCallback(async () => {
    if (!urlInput.trim()) return;
    setSaving(true);
    setSaveError(null);
    try {
      await settingsApi.setLocalEndpoint(item.provider_name, urlInput.trim());
      setTestResult(null);
      onSaved();
    } catch {
      setSaveError("Failed to save endpoint.");
    } finally {
      setSaving(false);
    }
  }, [urlInput, item.provider_name, onSaved]);

  return (
    <div className="provider-card">
      <div className="provider-card-header">
        <span className="provider-card-title">{label}</span>
        {item.configured && (
          <span className="status-badge status-badge-ok">Configured</span>
        )}
      </div>

      <div className="tag-input-row" style={{ marginTop: 12 }}>
        <input
          className="form-input"
          type="text"
          style={{ fontFamily: "monospace" }}
          value={urlInput}
          onChange={(e) => {
            setUrlInput(e.target.value);
            setTestResult(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleSave();
          }}
          placeholder={placeholder}
        />
        <button
          className="btn btn-sm"
          onClick={handleTest}
          disabled={!urlInput.trim() || testing}
        >
          {testing ? "Testing..." : "Test"}
        </button>
        <button
          className="btn btn-primary btn-sm"
          onClick={handleSave}
          disabled={!urlInput.trim() || saving}
        >
          {saving ? "Saving..." : "Save"}
        </button>
      </div>

      {testResult && (
        <div
          className={`feedback-banner ${testResult.reachable ? "feedback-ok" : "feedback-error"}`}
        >
          {testResult.message}
        </div>
      )}
      {saveError && (
        <div className="feedback-banner feedback-error">{saveError}</div>
      )}
    </div>
  );
}


function PublicDataToolsSection({
  dataSources,
  onChanged,
}: {
  dataSources: DataSourcesResponse;
  onChanged: () => void;
}) {
  const handleToggle = useCallback(
    async (sourceId: string, enabled: boolean) => {
      await settingsApi.togglePublicSource(sourceId, enabled);
      onChanged();
    },
    [onChanged]
  );

  const handleToggleBatch = useCallback(
    async (sourceIds: string[], enabled: boolean) => {
      await settingsApi.togglePublicSourceBatch(sourceIds, enabled);
      onChanged();
    },
    [onChanged]
  );

  const categories = Object.keys(dataSources.public_sources);
  const enabledSet = new Set(dataSources.enabled_public);
  const allSourceIds = categories.flatMap((cat) =>
    dataSources.public_sources[cat].map((s) => s.source_id)
  );
  const allEnabled = allSourceIds.length > 0 && allSourceIds.every((id) => enabledSet.has(id));
  const someEnabled = allSourceIds.some((id) => enabledSet.has(id));

  return (
    <section className="settings-section">
      <div className="settings-section-title">Public Data Tools</div>
      <p className="text-sm text-muted" style={{ marginBottom: 12 }}>
        Free public data sources. Enable the ones you want agents to use.
      </p>

      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
        <input
          type="checkbox"
          checked={allEnabled}
          ref={(el) => { if (el) el.indeterminate = someEnabled && !allEnabled; }}
          onChange={() => handleToggleBatch(allSourceIds, !allEnabled)}
          style={{ width: 16, height: 16, cursor: "pointer" }}
        />
        <span style={{ fontWeight: 700, fontSize: 13 }}>Select All</span>
      </div>

      {categories.map((category) => {
        const catSources = dataSources.public_sources[category];
        const catIds = catSources.map((s) => s.source_id);
        const catAllEnabled = catIds.length > 0 && catIds.every((id) => enabledSet.has(id));
        const catSomeEnabled = catIds.some((id) => enabledSet.has(id));
        return (
          <div key={category} style={{ marginBottom: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
              <input
                type="checkbox"
                checked={catAllEnabled}
                ref={(el) => { if (el) el.indeterminate = catSomeEnabled && !catAllEnabled; }}
                onChange={() => handleToggleBatch(catIds, !catAllEnabled)}
                style={{ width: 16, height: 16, cursor: "pointer" }}
              />
              <span style={{ fontWeight: 600, fontSize: 13 }}>
                {DATA_CATEGORY_LABELS[category] ?? category}
              </span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8, paddingLeft: 24 }}>
              {catSources.map((source) => (
                <PublicSourceRow
                  key={source.source_id}
                  source={source}
                  enabled={enabledSet.has(source.source_id)}
                  onToggle={handleToggle}
                />
              ))}
            </div>
          </div>
        );
      })}
    </section>
  );
}


function PublicSourceRow({
  source,
  enabled,
  onToggle,
}: {
  source: PublicDataSourceEntry;
  enabled: boolean;
  onToggle: (sourceId: string, enabled: boolean) => void;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
      <input
        type="checkbox"
        checked={enabled}
        onChange={(e) => onToggle(source.source_id, e.target.checked)}
        style={{ width: 16, height: 16, cursor: "pointer" }}
      />
      <span style={{ fontWeight: 500, minWidth: 100 }}>{source.name}</span>
      <input
        className="form-input"
        type="text"
        value={source.base_url}
        readOnly
        disabled
        style={{ flex: 1, fontFamily: "monospace", fontSize: 12, opacity: 0.6 }}
      />
    </div>
  );
}


function AdditionalToolApisSection({
  dataSources,
  onChanged,
}: {
  dataSources: DataSourcesResponse;
  onChanged: () => void;
}) {
  const categories = Object.keys(dataSources.additional_sources);

  return (
    <section className="settings-section">
      <div className="settings-section-title">Additional Tool APIs</div>
      <p className="text-sm text-muted" style={{ marginBottom: 12 }}>
        Third-party data sources that require an API key.
      </p>
      {categories.map((category) => (
        <div key={category} style={{ marginBottom: 16 }}>
          <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 8 }}>
            {DATA_CATEGORY_LABELS[category] ?? category}
          </div>
          <AdditionalApiCategoryBlock
            sources={dataSources.additional_sources[category]}
            configured={dataSources.configured_additional}
            onChanged={onChanged}
          />
        </div>
      ))}
    </section>
  );
}


function AdditionalApiCategoryBlock({
  sources,
  configured,
  onChanged,
}: {
  sources: AdditionalApiSourceEntry[];
  configured: ConfiguredAdditionalApi[];
  onChanged: () => void;
}) {
  const [addingSourceId, setAddingSourceId] = useState("");
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [saving, setSaving] = useState(false);

  const configuredIds = configured.map((c) => c.source_id);
  const availableSources = sources.filter((s) => !configuredIds.includes(s.source_id));

  const handleAdd = useCallback(async () => {
    if (!addingSourceId) return;
    setSaving(true);
    try {
      await settingsApi.addAdditionalApi(addingSourceId, apiKeyInput);
      setAddingSourceId("");
      setApiKeyInput("");
      onChanged();
    } finally {
      setSaving(false);
    }
  }, [addingSourceId, apiKeyInput, onChanged]);

  const handleRemove = useCallback(
    async (sourceId: string) => {
      await settingsApi.removeAdditionalApi(sourceId);
      onChanged();
    },
    [onChanged]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {configured
        .filter((c) => sources.some((s) => s.source_id === c.source_id))
        .map((entry) => {
          const src = sources.find((s) => s.source_id === entry.source_id)!;
          return (
            <div key={entry.source_id} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontWeight: 500, minWidth: 120 }}>{src.name}</span>
              <input
                className="form-input"
                type="text"
                value={entry.api_key || "(configured)"}
                readOnly
                disabled
                style={{ flex: 1, fontFamily: "monospace", fontSize: 12, opacity: 0.6 }}
              />
              <button
                className="btn btn-sm"
                onClick={() => handleRemove(entry.source_id)}
                style={{ color: "var(--error)" }}
              >
                Remove
              </button>
            </div>
          );
        })}

      {availableSources.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 8 }}>
          <select
            className="form-input"
            value={addingSourceId}
            onChange={(e) => setAddingSourceId(e.target.value)}
            style={{ minWidth: 140 }}
          >
            <option value="">Select source...</option>
            {availableSources.map((s) => (
              <option key={s.source_id} value={s.source_id}>
                {s.name}
              </option>
            ))}
          </select>
          <input
            className="form-input"
            type="password"
            value={apiKeyInput}
            onChange={(e) => setApiKeyInput(e.target.value)}
            placeholder="API key"
            style={{ flex: 1 }}
          />
          <button
            className="btn btn-primary btn-sm"
            onClick={handleAdd}
            disabled={!addingSourceId || saving}
          >
            {saving ? "Adding..." : "Add"}
          </button>
        </div>
      )}
    </div>
  );
}
