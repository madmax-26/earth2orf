import { atom } from "jotai"
import * as React from 'react'
export const treeRenderElementAtom = atom<React.Ref<HTMLDivElement>>()