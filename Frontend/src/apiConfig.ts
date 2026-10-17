// Single source of truth for the backend address.
//
// Every fetch in the app goes through apiBase(). The values are persisted in
// localStorage by the Settings dialog (StatusBar gear) so a chosen backend
// survives reloads; without a stored setting the defaults below apply.

const STORAGE_KEY = 'treeflow.api'
// Keys used under the project's earlier names (earth2orf, before that TCT),
// newest first. Read once as a fallback and moved to the current key, so a
// saved backend address survives the renames.
const LEGACY_STORAGE_KEYS = ['earth2orf.api', 'tct.api']

export const DEFAULT_HOST = 'localhost'
export const DEFAULT_PORT = 8000

export type ApiConfig = { host: string; port: number }

function load(): ApiConfig {
  try {
    let raw = localStorage.getItem(STORAGE_KEY)
    for (const legacy of LEGACY_STORAGE_KEYS) {
      if (raw !== null) break
      raw = localStorage.getItem(legacy)
      if (raw !== null) localStorage.setItem(STORAGE_KEY, raw)
    }
    // Old keys are dead after the move — clear them all.
    for (const legacy of LEGACY_STORAGE_KEYS) localStorage.removeItem(legacy)
    if (raw) {
      const parsed = JSON.parse(raw)
      const host = typeof parsed.host === 'string' && parsed.host.trim() ? parsed.host.trim() : DEFAULT_HOST
      const port = Number.isInteger(parsed.port) && parsed.port > 0 && parsed.port < 65536
        ? parsed.port : DEFAULT_PORT
      return { host, port }
    }
  } catch {
    // corrupt/unavailable storage → defaults
  }
  return { host: DEFAULT_HOST, port: DEFAULT_PORT }
}

let current: ApiConfig = load()

export function getApiConfig(): ApiConfig {
  return current
}

/** Backend origin, e.g. "http://localhost:8000" — no trailing slash. */
export function apiBase(): string {
  return `http://${current.host}:${current.port}`
}

/** Persist a new backend address. Callers reload the app so every cached
    atom/fetch is re-issued against the new backend. */
export function setApiConfig(cfg: ApiConfig): void {
  current = cfg
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(cfg))
  } catch {
    // storage unavailable → setting applies for this session only
  }
}
