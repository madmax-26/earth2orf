import { atom } from 'jotai'

export const selectedTreeIdAtom = atom<string | null>(null)
export const forestViewModeAtom = atom<'list' | 'map'>('list')
