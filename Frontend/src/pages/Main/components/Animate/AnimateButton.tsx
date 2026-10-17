import { useState } from 'react'
import AnimateModal from './AnimateModal'

type Props = { iterations: number[] }

export default function AnimateButton({ iterations }: Props) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        disabled={iterations.length === 0}
        title="Animate a tree across all iterations"
        style={{
          flexShrink: 0,
          padding: '4px 9px',
          fontSize: 10,
          fontWeight: 600,
          border: '1px solid #ccc',
          borderRadius: 4,
          background: iterations.length === 0 ? '#f5f5f5' : '#fff',
          color: iterations.length === 0 ? '#aaa' : '#555',
          cursor: iterations.length === 0 ? 'default' : 'pointer',
          whiteSpace: 'nowrap',
        }}
      >
        ▶ Animate
      </button>
      {open && <AnimateModal iterations={iterations} onClose={() => setOpen(false)} />}
    </>
  )
}
