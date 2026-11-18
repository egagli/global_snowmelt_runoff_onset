import React, { ReactNode, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Box, Global } from 'theme-ui'
import { useStore } from '../lib/store'
import {
  SHEET,
  SHEET_MEDIA,
  SHEET_LANDSCAPE_MEDIA,
  SHEET_MQ,
  SheetContext,
  prefersReducedMotion,
  type SheetMode,
  type SheetState,
} from '../lib/sheet'

const useMediaMode = (): SheetMode | null => {
  const [mode, setMode] = useState<SheetMode | null>(null)
  useEffect(() => {
    const sheetMq = window.matchMedia(SHEET_MEDIA)
    const landMq = window.matchMedia(SHEET_LANDSCAPE_MEDIA)
    const update = () =>
      setMode(
        !sheetMq.matches ? null : landMq.matches ? 'land' : window.innerWidth >= 640 ? 'tablet' : 'phone'
      )
    update()
    sheetMq.addEventListener('change', update)
    landMq.addEventListener('change', update)
    window.addEventListener('resize', update)
    return () => {
      sheetMq.removeEventListener('change', update)
      landMq.removeEventListener('change', update)
      window.removeEventListener('resize', update)
    }
  }, [])
  return mode
}

const useViewportHeight = () => {
  const [h, setH] = useState(800)
  useEffect(() => {
    const update = () => setH(window.innerHeight)
    update()
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [])
  return h
}

const Chevron = ({ down }: { down: boolean }) => (
  <svg
    width='18'
    height='18'
    viewBox='0 0 24 24'
    fill='none'
    stroke='currentColor'
    strokeWidth='2.4'
    strokeLinecap='round'
    strokeLinejoin='round'
    style={{ transform: down ? 'rotate(180deg)' : 'none', transition: `transform ${SHEET.ms}ms ${SHEET.ease}` }}
  >
    <polyline points='6 15 12 9 18 15' />
  </svg>
)

export const MobileSheet = ({
  title,
  subtitle,
  legend,
  queryKey,
  children,
}: {
  title: string
  subtitle: ReactNode
  legend: ReactNode
  queryKey: string
  children: ReactNode
}) => {
  const mode = useMediaMode()
  const vh = useViewportHeight()
  const setSheet = useStore((s) => s.setSheet)

  const sheetRef = useRef<HTMLDivElement>(null)
  const headRef = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<SheetState>('peek')
  const [peekH, setPeekH] = useState(120)
  const [drag, setDrag] = useState<number | null>(null)
  const [width, setWidth] = useState(0)

  useLayoutEffect(() => {
    const head = headRef.current, sheet = sheetRef.current
    if (!head || !sheet) return
    const ro = new ResizeObserver(() => {
      setPeekH(head.offsetHeight)
      setWidth(sheet.offsetWidth)
    })
    ro.observe(head)
    ro.observe(sheet)
    return () => ro.disconnect()
  }, [])

  const states: SheetState[] = mode === 'land' ? ['peek', 'full'] : ['peek', 'half', 'full']
  const full = mode === 'land' ? vh - 24 : mode === 'tablet' ? Math.round(vh * 0.88) - 12 : Math.round(vh * 0.88)
  const heightFor = useCallback(
    (s: SheetState) =>
      s === 'peek' ? peekH : s === 'half' ? Math.min(full, Math.max(Math.round(vh * 0.5), peekH + 140)) : full,
    [peekH, full, vh]
  )
  const settled = states.includes(state) ? state : 'full'
  const visible = Math.max(peekH, Math.min(full, drag ?? heightFor(settled)))
  const next = states[(states.indexOf(settled) + 1) % states.length]

  const covered = mode === 'land' ? 0 : visible + (mode === 'tablet' ? 12 : 0)
  useEffect(() => {
    document.documentElement.style.setProperty('--sheet-covered', `${covered}px`)
  }, [covered])
  useEffect(() => {
    if (drag !== null) return
    if (!mode) return setSheet(null, { bottom: 0, left: 0 })
    if (mode === 'land')
      setSheet(mode, settled === 'full' ? { bottom: 0, left: width + 12 } : { bottom: peekH + 12, left: 0 })
    else setSheet(mode, { bottom: covered, left: 0 })
  }, [mode, settled, covered, width, peekH, drag, setSheet])

  const goTo = useCallback((s: SheetState) => {
    setState(s)
    if (s === 'peek') sheetRef.current?.scrollTo({ top: 0 })
  }, [])

  const lastQueryKey = useRef(queryKey)
  useEffect(() => {
    if (!mode || !queryKey || queryKey === lastQueryKey.current) {
      lastQueryKey.current = queryKey
      return
    }
    lastQueryKey.current = queryKey
    setState((s) => (s === 'peek' ? (mode === 'land' ? 'full' : 'half') : s))
    sheetRef.current?.scrollTo({ top: 0, behavior: prefersReducedMotion() ? 'auto' : 'smooth' })
  }, [queryKey, mode])

  const gesture = useRef<{ id: number; y0: number; v0: number; t0: number; moved: boolean } | null>(null)
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button) return
    gesture.current = { id: e.pointerId, y0: e.clientY, v0: visible, t0: performance.now(), moved: false }
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g || g.id !== e.pointerId) return
    const dy = e.clientY - g.y0
    if (!g.moved && Math.abs(dy) < 6) return
    g.moved = true
    setDrag(g.v0 - dy)
  }
  const onPointerUp = (e: React.PointerEvent) => {
    const g = gesture.current
    if (!g || g.id !== e.pointerId) return
    gesture.current = null
    if (!g.moved) return goTo(next)
    const v = (g.y0 - e.clientY) / Math.max(1, performance.now() - g.t0)
    const pos = visible
    const order = states.map((s) => [s, heightFor(s)] as const)
    let pick: SheetState
    if (Math.abs(v) > 0.5) {
      pick = v > 0
        ? (order.find(([, h]) => h > pos + 4) ?? order[order.length - 1])[0]
        : ([...order].reverse().find(([, h]) => h < pos - 4) ?? order[0])[0]
    } else {
      pick = order.reduce((a, b) => (Math.abs(b[1] - pos) < Math.abs(a[1] - pos) ? b : a))[0]
    }
    setDrag(null)
    goTo(pick)
  }

  const onHeadKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      goTo(next)
    }
  }
  const onSheetKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && settled !== 'peek') {
      goTo('peek')
      headRef.current?.focus()
    }
  }

  const animate = drag === null && !prefersReducedMotion()
  const offset = full - visible

  return (
    <SheetContext.Provider value={true}>
      <Global
        styles={{
          [SHEET_MQ]: {
            'html, body': { overflow: 'hidden', overscrollBehavior: 'none' },
            '.maplibregl-ctrl-bottom-right': {
              transform: 'translateY(calc(-1 * var(--sheet-covered, 0px)))',
              transition: `transform ${SHEET.ms}ms ${SHEET.ease}`,
            },
            '.maplibregl-ctrl-bottom-right .maplibregl-ctrl': { margin: '0 12px 12px 0' },
            '.maplibregl-ctrl-attrib.maplibregl-compact': {
              minHeight: 32,
              boxSizing: 'border-box',
              borderRadius: 16,
              padding: '6px 36px 6px 12px',
              maxWidth: 'calc(100vw - 24px)',
            },
            '.maplibregl-ctrl-attrib.maplibregl-compact:not(.maplibregl-compact-show)': {
              width: 32,
              height: 32,
              padding: 0,
            },
            '.maplibregl-ctrl-attrib.maplibregl-compact .maplibregl-ctrl-attrib-button': {
              width: 32,
              height: 32,
              top: 0,
              right: 0,
              left: 'auto',
              backgroundPosition: 'center',
            },
            '.maplibregl-ctrl-attrib-button::after': { content: '""', position: 'absolute', inset: '-6px' },
            '.maplibregl-ctrl-attrib-inner': { fontSize: 12, lineHeight: '20px' },
          },
          '@media (prefers-reduced-motion: reduce)': {
            '.maplibregl-ctrl-bottom-right': { transition: 'none' },
          },
        }}
      />
      <Box
        ref={sheetRef}
        role='region'
        aria-label='Map legend, controls and information'
        onKeyDown={onSheetKey}
        style={
          mode
            ? {
                height: full,
                transform: `translateY(${offset}px)`,
                transition: animate ? `transform ${SHEET.ms}ms ${SHEET.ease}` : 'none',
                paddingBottom: offset + 24,
                overflowY: settled === 'peek' && drag === null ? 'hidden' : 'auto',
              }
            : undefined
        }
        sx={{
          display: 'none',
          [SHEET_MQ]: {
            display: 'flex',
            flexDirection: 'column',
            gap: '16px',
            position: 'absolute',
            bottom: 0,
            left: 'env(safe-area-inset-left)',
            right: 'env(safe-area-inset-right)',
            zIndex: 1000,
            bg: SHEET.bg,
            color: SHEET.text,
            fontSize: 13,
            border: `1px solid ${SHEET.border}`,
            borderBottom: 0,
            borderRadius: '16px 16px 0 0',
            boxShadow: '0 -4px 16px rgba(0,0,0,.25)',
            overflowX: 'hidden',
            overscrollBehavior: 'contain',
            willChange: 'transform',
            ...(mode === 'tablet' && {
              left: `max(12px, calc((100% - ${SHEET.maxTabletWidth}px) / 2))`,
              right: `max(12px, calc((100% - ${SHEET.maxTabletWidth}px) / 2))`,
              bottom: '12px',
              borderBottom: `1px solid ${SHEET.border}`,
              borderRadius: 16,
            }),
            ...(mode === 'land' && {
              left: 'calc(12px + env(safe-area-inset-left))',
              right: 'auto',
              bottom: '12px',
              width: `min(${SHEET.landWidth}px, 50%)`,
              borderBottom: `1px solid ${SHEET.border}`,
              borderRadius: 16,
              boxShadow: '0 4px 16px rgba(0,0,0,.25)',
            }),
          },
          '@media (prefers-reduced-motion: reduce)': { transition: 'none !important' },
          '& > :not([data-sheet-head])': { mx: '16px', flexShrink: 0 },
          '& :focus-visible': { outline: `2px solid ${SHEET.accent}`, outlineOffset: '2px' },
        }}
      >
        <div
          ref={headRef}
          data-sheet-head=''
          role='button'
          tabIndex={0}
          aria-expanded={settled !== 'peek'}
          aria-label={`${next === 'peek' ? 'Collapse' : 'Expand'} panel: ${title}`}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onKeyDown={onHeadKey}
          style={{
            position: 'sticky',
            top: 0,
            zIndex: 100,
            flexShrink: 0,
            background: SHEET.bg,
            padding: '0 16px 12px',
            minHeight: 44,
            borderBottom: `1px solid ${settled === 'peek' ? 'transparent' : SHEET.border}`,
            touchAction: 'none',
            cursor: 'grab',
            userSelect: 'none',
            WebkitUserSelect: 'none',
            outline: 'none',
          }}
        >
          <div
            aria-hidden
            style={{ width: 36, height: 4, borderRadius: 2, background: '#4a4f58', margin: '8px auto 4px' }}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, minHeight: 44 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  fontSize: 17, fontWeight: 600, color: '#fff', lineHeight: '22px',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}
              >
                {title}
              </div>
              <div
                style={{
                  fontSize: 13, color: SHEET.dim, lineHeight: '18px',
                  whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                }}
              >
                {subtitle}
              </div>
            </div>
            <span
              aria-hidden
              style={{
                width: 44, height: 44, flexShrink: 0, marginRight: -8,
                display: 'grid', placeItems: 'center', color: SHEET.text,
              }}
            >
              <Chevron down={next === 'peek'} />
            </span>
          </div>
          <div aria-hidden style={{ marginTop: 4 }}>{legend}</div>
        </div>
        {children}
      </Box>
    </SheetContext.Provider>
  )
}
