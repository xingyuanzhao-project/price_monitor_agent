/**
 * Settings page: LLM providers, API credentials, and global defaults.
 *
 * Loads user settings from the backend on mount. Provides CRUD forms for
 * LLM provider configs (name, base URL, masked API key, models list) and
 * API credentials (name, type, key-value fields). Global defaults section
 * covers temperature, max tokens, and rate limit RPM.
 */

import { useState, useEffect, useCallback } from "react";
import { settingsApi } from "../../api/client";
import type {
  UserSettings,
  LLMProviderConfig,
  APICredential,
} from "../../types/schema";

const EMPTY_SETTINGS: UserSettings = {
  llm_providers: [],
  api_credentials: [],
  global_defaults: {
    temperature: 0.7,
    max_tokens: 4096,
    rate_limit_rpm: 60,
  },
};

export default function SettingsPanel() {
  const [settings, setSettings] = useState<UserSettings>(EMPTY_SETTINGS);
  const [saving, setSaving] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const loadSettings = useCallback(async () => {
    const data = await settingsApi.get();
    setSettings(data);
    setLoaded(true);
  }, []);

  useEffect(() => {
    loadSettings();
  }, [loadSettings]);

  const handleSave = useCallback(async () => {
    setSaving(true);
    try {
      const saved = await settingsApi.save(settings);
      setSettings(saved);
    } catch (error) {
      throw new Error(
        `Failed to save settings: ${error instanceof Error ? error.message : String(error)}`
      );
    } finally {
      setSaving(false);
    }
  }, [settings]);

  const updateProvider = useCallback(
    (index: number, updates: Partial<LLMProviderConfig>) => {
      setSettings((prev) => ({
        ...prev,
        llm_providers: prev.llm_providers.map((provider, i) =>
          i === index ? { ...provider, ...updates } : provider
        ),
      }));
    },
    []
  );

  const addProvider = useCallback(() => {
    setSettings((prev) => ({
      ...prev,
      llm_providers: [
        ...prev.llm_providers,
        { provider_name: "", base_url: "", api_key: "", available_models: [] },
      ],
    }));
  }, []);

  const removeProvider = useCallback((index: number) => {
    setSettings((prev) => ({
      ...prev,
      llm_providers: prev.llm_providers.filter((_, i) => i !== index),
    }));
  }, []);

  const updateCredential = useCallback(
    (index: number, updates: Partial<APICredential>) => {
      setSettings((prev) => ({
        ...prev,
        api_credentials: prev.api_credentials.map((cred, i) =>
          i === index ? { ...cred, ...updates } : cred
        ),
      }));
    },
    []
  );

  const addCredential = useCallback(() => {
    setSettings((prev) => ({
      ...prev,
      api_credentials: [
        ...prev.api_credentials,
        { credential_name: "", credential_type: "", fields: {} },
      ],
    }));
  }, []);

  const removeCredential = useCallback((index: number) => {
    setSettings((prev) => ({
      ...prev,
      api_credentials: prev.api_credentials.filter((_, i) => i !== index),
    }));
  }, []);

  if (!loaded) {
    return (
      <div className="settings-page">
        <div className="empty-state">Loading settings...</div>
      </div>
    );
  }

  return (
    <div className="settings-page">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700 }}>Settings</h2>
        <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
          {saving ? "Saving..." : "Save Settings"}
        </button>
      </div>

      <section className="settings-section">
        <div className="settings-section-title">LLM Providers</div>
        {settings.llm_providers.map((provider, index) => (
          <ProviderCard
            key={index}
            provider={provider}
            onChange={(updates) => updateProvider(index, updates)}
            onRemove={() => removeProvider(index)}
          />
        ))}
        <button className="btn btn-sm" onClick={addProvider}>
          + Add Provider
        </button>
      </section>

      <section className="settings-section">
        <div className="settings-section-title">API Credentials</div>
        {settings.api_credentials.map((credential, index) => (
          <CredentialCard
            key={index}
            credential={credential}
            onChange={(updates) => updateCredential(index, updates)}
            onRemove={() => removeCredential(index)}
          />
        ))}
        <button className="btn btn-sm" onClick={addCredential}>
          + Add Credential
        </button>
      </section>

      <section className="settings-section">
        <div className="settings-section-title">Global Defaults</div>
        <div className="inline-fields">
          <div className="form-group">
            <label className="form-label">Temperature</label>
            <input
              type="number"
              className="form-input"
              min="0"
              max="2"
              step="0.1"
              value={settings.global_defaults.temperature}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  global_defaults: {
                    ...prev.global_defaults,
                    temperature: parseFloat(e.target.value) || 0,
                  },
                }))
              }
            />
          </div>
          <div className="form-group">
            <label className="form-label">Max Tokens</label>
            <input
              type="number"
              className="form-input"
              min="1"
              value={settings.global_defaults.max_tokens}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  global_defaults: {
                    ...prev.global_defaults,
                    max_tokens: parseInt(e.target.value) || 1,
                  },
                }))
              }
            />
          </div>
        </div>
        <div className="form-group">
          <label className="form-label">Rate Limit (RPM)</label>
          <input
            type="number"
            className="form-input"
            min="1"
            value={settings.global_defaults.rate_limit_rpm}
            onChange={(e) =>
              setSettings((prev) => ({
                ...prev,
                global_defaults: {
                  ...prev.global_defaults,
                  rate_limit_rpm: parseInt(e.target.value) || 1,
                },
              }))
            }
          />
        </div>
      </section>
    </div>
  );
}

function ProviderCard({
  provider,
  onChange,
  onRemove,
}: {
  provider: LLMProviderConfig;
  onChange: (updates: Partial<LLMProviderConfig>) => void;
  onRemove: () => void;
}) {
  const [newModel, setNewModel] = useState("");

  const maskedKey = provider.api_key
    ? provider.api_key.slice(0, 4) + "••••" + provider.api_key.slice(-4)
    : "";

  return (
    <div className="provider-card">
      <div className="provider-card-header">
        <span className="provider-card-title">
          {provider.provider_name || "New Provider"}
        </span>
        <button className="btn btn-danger btn-sm" onClick={onRemove}>
          Remove
        </button>
      </div>

      <div className="inline-fields">
        <div className="form-group">
          <label className="form-label">Provider Name</label>
          <input
            className="form-input"
            value={provider.provider_name}
            onChange={(e) => onChange({ provider_name: e.target.value })}
            placeholder="e.g. OpenRouter"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Base URL</label>
          <input
            className="form-input"
            value={provider.base_url}
            onChange={(e) => onChange({ base_url: e.target.value })}
            placeholder="https://api.example.com/v1"
          />
        </div>
      </div>

      <div className="form-group">
        <label className="form-label">
          API Key {maskedKey && <span className="text-muted">({maskedKey})</span>}
        </label>
        <input
          className="form-input"
          type="password"
          value={provider.api_key}
          onChange={(e) => onChange({ api_key: e.target.value })}
          placeholder="sk-..."
        />
      </div>

      <div className="form-group">
        <label className="form-label">Models</label>
        {provider.available_models.length > 0 && (
          <div className="tag-list">
            {provider.available_models.map((model: string, modelIndex: number) => (
              <span key={modelIndex} className="tag-item">
                {model}
                <span
                  className="tag-remove"
                  onClick={() =>
                    onChange({
                      available_models: provider.available_models.filter(
                        (_: string, filterIndex: number) => filterIndex !== modelIndex
                      ),
                    })
                  }
                >
                  &times;
                </span>
              </span>
            ))}
          </div>
        )}
        <div className="tag-input-row">
          <input
            className="form-input"
            value={newModel}
            onChange={(e) => setNewModel(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && newModel.trim()) {
                onChange({ available_models: [...provider.available_models, newModel.trim()] });
                setNewModel("");
              }
            }}
            placeholder="Add model name..."
          />
          <button
            className="btn btn-sm"
            onClick={() => {
              if (!newModel.trim()) return;
              onChange({ available_models: [...provider.available_models, newModel.trim()] });
              setNewModel("");
            }}
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
}

function CredentialCard({
  credential,
  onChange,
  onRemove,
}: {
  credential: APICredential;
  onChange: (updates: Partial<APICredential>) => void;
  onRemove: () => void;
}) {
  const [newFieldKey, setNewFieldKey] = useState("");
  const [newFieldValue, setNewFieldValue] = useState("");

  return (
    <div className="credential-card">
      <div className="credential-card-header">
        <span className="credential-card-title">
          {credential.credential_name || "New Credential"}
        </span>
        <button className="btn btn-danger btn-sm" onClick={onRemove}>
          Remove
        </button>
      </div>

      <div className="inline-fields">
        <div className="form-group">
          <label className="form-label">Credential Name</label>
          <input
            className="form-input"
            value={credential.credential_name}
            onChange={(e) => onChange({ credential_name: e.target.value })}
            placeholder="e.g. Binance API"
          />
        </div>
        <div className="form-group">
          <label className="form-label">Type</label>
          <input
            className="form-input"
            value={credential.credential_type}
            onChange={(e) => onChange({ credential_type: e.target.value })}
            placeholder="e.g. exchange, social_media"
          />
        </div>
      </div>

      <div className="form-group">
        <label className="form-label">Fields</label>
        {Object.entries(credential.fields).length > 0 && (
          <div className="tag-list">
            {Object.entries(credential.fields).map(([key, value]) => (
              <span key={key} className="tag-item">
                {key}: {value.slice(0, 8)}••••
                <span
                  className="tag-remove"
                  onClick={() => {
                    const updated = { ...credential.fields };
                    delete updated[key];
                    onChange({ fields: updated });
                  }}
                >
                  &times;
                </span>
              </span>
            ))}
          </div>
        )}
        <div className="tag-input-row">
          <input
            className="form-input"
            value={newFieldKey}
            onChange={(e) => setNewFieldKey(e.target.value)}
            placeholder="Key"
            style={{ flex: 1 }}
          />
          <input
            className="form-input"
            value={newFieldValue}
            onChange={(e) => setNewFieldValue(e.target.value)}
            placeholder="Value"
            style={{ flex: 1 }}
          />
          <button
            className="btn btn-sm"
            onClick={() => {
              if (!newFieldKey.trim() || !newFieldValue.trim()) return;
              onChange({
                fields: {
                  ...credential.fields,
                  [newFieldKey.trim()]: newFieldValue.trim(),
                },
              });
              setNewFieldKey("");
              setNewFieldValue("");
            }}
          >
            Add
          </button>
        </div>
      </div>
    </div>
  );
}
