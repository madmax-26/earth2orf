import { useAtom } from 'jotai'
import { selectedIterationAtom } from './IterationAtoms'

type IterationProps = {
  index: number
}

export default function Iteration({ index }: IterationProps) {
  const [selected, setSelected] = useAtom(selectedIterationAtom)
  const isSelected = selected === index

  return (
    <button
      onClick={() => setSelected(isSelected ? null : index)}
      title={`Iteration ${index}`}
      style={{
        padding: '4px 16px',
        borderRadius: 20,
        border: isSelected ? 'none' : '1px solid #ccc',
        background: isSelected ? '#3A8C3A' : '#fff',
        color: isSelected ? '#fff' : '#555',
        fontWeight: isSelected ? 600 : 400,
        fontSize: 13,
        cursor: 'pointer',
        flexShrink: 0,
        transition: 'background 0.15s, color 0.15s',
        userSelect: 'none',
        lineHeight: '22px',
      }}
    >
      {index}
    </button>
  )
}
