import { useState } from 'react'
import { createPortal } from 'react-dom'
import { DEFAULT_HOST, DEFAULT_PORT, getApiConfig, setApiConfig } from '../../../../apiConfig'
import { HEADING_FONT } from '../../../../theme'

// Settings popup (StatusBar gear): the backend address the frontend talks to.
// Applying persists the address and reloads the app, so every cached atom and
// in-flight fetch is re-issued against the new backend — a half-switched state
// (old forest data, new backend) can't occur.

const LABEL = { fontSize: 11, fontWeight: 600, color: '#666', marginBottom: 4 } as const
const INPUT = {
  width: '100%', boxSizing: 'border-box' as const,
  fontSize: 13, padding: '6px 8px',
  border: '1px solid #ddd', borderRadius: 4, color: '#333', outline: 'none',
}

export default function SettingsDialog({ onClose }: { onClose: () => void }) {
  const initial = getApiConfig()
  const [host, setHost] = useState(initial.host)
  const [port, setPort] = useState(String(initial.port))
  const [error, setError] = useState<string | null>(null)

  const apply = () => {
    const h = host.trim()
    const p = Number(port)
    if (!h) { setError('Host must not be empty.'); return }
    if (!Number.isInteger(p) || p < 1 || p > 65535) { setError('Port must be 1–65535.'); return }
    setApiConfig({ host: h, port: p })
    // Reload so all cached data is re-fetched from the new backend.
    window.location.reload()
  }

  const isDefault = host.trim() === DEFAULT_HOST && Number(port) === DEFAULT_PORT

  const modal = (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        zIndex: 2100,   // above the maximized-chart modals (2000)
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: '#fff', borderRadius: 10, width: 320,
          padding: '16px 20px 18px',
          boxShadow: '0 8px 36px rgba(0,0,0,0.2)',
          fontFamily: 'inherit',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: 14 }}>
          <span style={{ fontFamily: HEADING_FONT, fontWeight: 700, fontSize: 15, color: '#333', flex: 1 }}>Settings</span>
          <button
            onClick={onClose}
            title="Close"
            style={{
              border: 'none', background: 'none', fontSize: 20, lineHeight: 1,
              padding: '0 2px', cursor: 'pointer', color: '#aaa',
            }}
          >
            ×
          </button>
        </div>

        <div style={{ fontSize: 10, fontWeight: 700, color: '#aaa', fontFamily: HEADING_FONT, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
          Backend connection
        </div>

        <div style={{ marginBottom: 10 }}>
          <div style={LABEL}>IP / Host</div>
          <input
            value={host}
            onChange={e => { setHost(e.target.value); setError(null) }}
            onKeyDown={e => { if (e.key === 'Enter') apply() }}
            placeholder={DEFAULT_HOST}
            style={INPUT}
          />
        </div>

        <div style={{ marginBottom: 12 }}>
          <div style={LABEL}>Port</div>
          <input
            value={port}
            onChange={e => { setPort(e.target.value); setError(null) }}
            onKeyDown={e => { if (e.key === 'Enter') apply() }}
            placeholder={String(DEFAULT_PORT)}
            inputMode="numeric"
            style={INPUT}
          />
        </div>

        <div style={{ fontSize: 10, color: '#bbb', marginBottom: 12, lineHeight: 1.5 }}>
          Connecting to <span style={{ color: '#888', fontWeight: 600 }}>
            http://{host.trim() || DEFAULT_HOST}:{port || DEFAULT_PORT}
          </span>
          {isDefault && ' (default)'}
          <br />Applying reloads the app.
        </div>

        {error && (
          <div style={{
            fontSize: 10.5, color: '#c0392b', background: '#fff0f0',
            border: '1px solid #f5c6c6', borderRadius: 4,
            padding: '4px 7px', marginBottom: 10,
          }}>
            {error}
          </div>
        )}

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            onClick={() => { setHost(DEFAULT_HOST); setPort(String(DEFAULT_PORT)); setError(null) }}
            style={{
              fontSize: 11.5, padding: '6px 11px', border: '1px solid #ddd',
              borderRadius: 4, background: '#fff', color: '#888', cursor: 'pointer',
            }}
          >
            Reset
          </button>
          <button
            onClick={apply}
            style={{
              fontSize: 11.5, fontWeight: 700, padding: '6px 16px',
              border: '1px solid #3A8C3A', borderRadius: 4,
              background: '#3A8C3A', color: '#fff', cursor: 'pointer',
            }}
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  )

  return createPortal(modal, document.body)
}
