import { createContext, useContext } from 'react'

export const SHEET_MEDIA = '(max-width: 1023px), (pointer: coarse) and (max-height: 520px)'
export const SHEET_MQ = `@media ${SHEET_MEDIA}`
export const SHEET_LANDSCAPE_MEDIA =
  '(pointer: coarse) and (max-height: 520px) and (orientation: landscape)'

export type SheetMode = 'phone' | 'tablet' | 'land'
export type SheetState = 'peek' | 'half' | 'full'

export const SHEET = {
  bg: 'rgb(22,25,30)',
  border: '#2e3138',
  text: '#d0d0d0',
  dim: '#8f95a0',
  accent: '#1dbd8f',
  ease: 'cubic-bezier(.2,.8,.2,1)',
  ms: 220,
  maxTabletWidth: 600,
  landWidth: 380,
  searchClearance: 68,
}

export const SheetContext = createContext(false)

export function useTone(desktopDim: string) {
  const sheet = useContext(SheetContext)
  return {
    sheet,
    dim: sheet ? SHEET.dim : desktopDim,
    fs: (px: number) => (sheet ? Math.max(12, px) : px),
  }
}

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
