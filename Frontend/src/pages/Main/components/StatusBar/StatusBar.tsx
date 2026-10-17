import { apiBase } from '../../../../apiConfig'
import { useEffect, useState } from 'react'
import AppBar from '@mui/material/AppBar'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import Box from '@mui/material/Box'
import IconButton from '@mui/material/IconButton'
import SettingsIcon from '@mui/icons-material/Settings'
import { keyframes } from '@emotion/react'
import SettingsDialog from './SettingsDialog'

const POLL_INTERVAL_MS = 5000
const TIMEOUT_MS = 3000

type ConnectionState = 'connecting' | 'online' | 'offline'

interface BackendStatus {
  state: ConnectionState
  backendState: string
  latencyMs: number | null
}

const pulse = keyframes`
  0%   { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7); }
  70%  { box-shadow: 0 0 0 6px rgba(76, 175, 80, 0); }
  100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
`

const DOT_COLOR: Record<ConnectionState, string> = {
  connecting: '#FFA726',
  online: '#4CAF50',
  offline: '#F44336',
}

const STATUS_LABEL: Record<ConnectionState, string> = {
  connecting: 'Connecting',
  online: 'Online',
  offline: 'Offline',
}

const StatusBar = () => {
  const [status, setStatus] = useState<BackendStatus>({
    state: 'connecting',
    backendState: '',
    latencyMs: null,
  })
  const [settingsOpen, setSettingsOpen] = useState(false)

  useEffect(() => {
    const check = async () => {
      const t0 = performance.now()
      try {
        const res = await fetch(`${apiBase()}/`, {
          signal: AbortSignal.timeout(TIMEOUT_MS),
        })
        const latencyMs = Math.round(performance.now() - t0)
        if (res.ok) {
          const data = await res.json()
          setStatus({ state: 'online', backendState: data.State ?? '', latencyMs })
        } else {
          setStatus({ state: 'offline', backendState: '', latencyMs: null })
        }
      } catch {
        setStatus({ state: 'offline', backendState: '', latencyMs: null })
      }
    }

    check()
    const id = setInterval(check, POLL_INTERVAL_MS)
    return () => clearInterval(id)
  }, [])

  const { state, backendState, latencyMs } = status
  const dotColor = DOT_COLOR[state]
  const isOnline = state === 'online'

  return (
    <AppBar position="sticky" color="primary">
      <Toolbar sx={{ gap: 2 }}>
        <Typography variant="h6" sx={{ flexGrow: 0, mr: 3 }}>
          TreeFlow
        </Typography>

        {/* Connection indicator */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 10,
              height: 10,
              borderRadius: '50%',
              bgcolor: dotColor,
              animation: isOnline ? `${pulse} 2s ease-in-out infinite` : 'none',
              flexShrink: 0,
            }}
          />
          <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
            {STATUS_LABEL[state]}
          </Typography>
          {isOnline && latencyMs !== null && (
            <Typography variant="body2" sx={{ opacity: 0.6, fontSize: '0.75rem' }}>
              {latencyMs} ms
            </Typography>
          )}
        </Box>

        {/* Backend state message */}
        {isOnline && backendState && (
          <Typography
            variant="body2"
            sx={{
              opacity: 0.7,
              fontSize: '0.75rem',
              borderLeft: '1px solid rgba(255,255,255,0.3)',
              pl: 2,
            }}
          >
            {backendState}
          </Typography>
        )}

        {/* Settings (backend address) — pinned to the right edge */}
        <Box sx={{ flexGrow: 1 }} />
        <IconButton
          onClick={() => setSettingsOpen(true)}
          title="Settings"
          size="small"
          sx={{ color: 'inherit', opacity: 0.85, '&:hover': { opacity: 1 } }}
        >
          <SettingsIcon fontSize="small" />
        </IconButton>
      </Toolbar>

      {settingsOpen && <SettingsDialog onClose={() => setSettingsOpen(false)} />}
    </AppBar>
  )
}

export default StatusBar
