import React, { CSSProperties } from 'react'
import {
  useStore,
  WATER_YEARS,
  VARIABLE_CONFIGS,
  type Variable,
  type Basemap,
} from '../lib/store'

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const TEXT = '#d0d0d0'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'
const CARD_WIDTH = 340

const cardStyle: CSSProperties = {
  position: 'absolute',
  background: BG,
  border: `1px solid ${BORDER}`,
  backdropFilter: 'blur(6px)',
  borderRadius: 8,
  color: TEXT,
  fontSize: 12,
  zIndex: 10,
  padding: '14px 16px',
  width: CARD_WIDTH,
}

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.06em',
  color: DIM,
  marginBottom: 6,
  fontWeight: 600,
}

function chip(active: boolean): CSSProperties {
  return {
    flex: 1,
    padding: '5px 0',
    borderRadius: 4,
    border: `1px solid ${active ? ACCENT : BORDER}`,
    cursor: 'pointer',
    fontSize: 12,
    textAlign: 'center' as const,
    background: active ? ACCENT : 'transparent',
    color: active ? '#fff' : TEXT,
    transition: 'background 0.15s, color 0.15s',
  }
}

// DOWY → calendar date, hemisphere-aware.
function dowyToDate(dowy: number, waterYear: number, southernHemisphere: boolean): string {
  const start = southernHemisphere
    ? new Date(waterYear, 3, 1)
    : new Date(waterYear - 1, 9, 1)
  const target = new Date(start.getTime() + (Math.round(dowy) - 1) * 86400000)
  return target.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

function formatValue(
  value: number | null,
  variable: Variable,
  waterYear: number,
  southernHemisphere: boolean,
): string {
  if (value === null) return 'No data'
  const cfg = VARIABLE_CONFIGS[variable]
  if (cfg.units === 'days') {
    return `${value.toFixed(1)} days`
  }
  // DOWY variables: the yearly one maps to a calendar date; the multi-year
  // median has no single year to anchor a date to.
  const dowy = Math.round(value)
  if (variable === 'runoff_onset') {
    return `${dowy} (${dowyToDate(dowy, waterYear, southernHemisphere)})`
  }
  return `DOWY ${dowy}`
}

// ---------------------------------------------------------------------------
// ZoomDisplay — live zoom level in bottom-right corner
// ---------------------------------------------------------------------------

const ZoomDisplay = ({ right, bottom }: { right: number; bottom: number }) => {
  const zoomLevel = useStore((s) => s.zoomLevel)
  return (
    <div style={{
      position: 'absolute', right, bottom,
      color: TEXT, fontSize: 13, fontFamily: 'monospace',
      pointerEvents: 'none', userSelect: 'none',
    }}>
      {'zoom level: '}
      <span style={{ display: 'inline-block', width: '4ch', textAlign: 'right' }}>
        {zoomLevel.toFixed(1)}
      </span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// TopRightCard — Basemap + Projection
// ---------------------------------------------------------------------------

const TopRightCard = ({ right, top }: { right: number; top: number }) => {
  const globeProjection = useStore((s) => s.globeProjection)
  const basemap = useStore((s) => s.basemap)
  const setGlobeProjection = useStore((s) => s.setGlobeProjection)
  const setBasemap = useStore((s) => s.setBasemap)

  const BASEMAP_OPTS: { label: string; value: Basemap }[] = [
    { label: 'dark',        value: 'dark'        },
    { label: 'satellite',   value: 'satellite'   },
    { label: 'topography',  value: 'topography'  },
  ]

  return (
    <div style={{ ...cardStyle, top, right }}>
      <div style={sectionLabelStyle}>basemap</div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
        {BASEMAP_OPTS.map((opt) => (
          <button key={opt.value} onClick={() => setBasemap(opt.value)} style={chip(basemap === opt.value)}>
            {opt.label}
          </button>
        ))}
      </div>
      <div style={sectionLabelStyle}>projection</div>
      <div style={{ display: 'flex', gap: 6 }}>
        {[{ label: 'globe', value: true }, { label: 'mercator', value: false }].map((opt) => (
          <button key={String(opt.value)} onClick={() => setGlobeProjection(opt.value)} style={chip(globeProjection === opt.value)}>
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// PointQueryCard — exact level-0 values at the clicked point
// ---------------------------------------------------------------------------

const QUERY_ROWS: { key: Variable; label: string }[] = [
  { key: 'runoff_onset',               label: 'onset' },
  { key: 'runoff_onset_median',        label: 'onset median' },
  { key: 'runoff_onset_mad',           label: 'onset MAD' },
  { key: 'temporal_resolution',        label: 'temporal res.' },
  { key: 'temporal_resolution_median', label: 'TR median' },
]

const PointQueryCard = ({ right, top }: { right: number; top: number }) => {
  const clickInfo = useStore((s) => s.clickInfo)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const waterYear = WATER_YEARS[waterYearIndex]
  const southernHemisphere = (clickInfo?.lat ?? 0) < 0

  return (
    <div style={{
      ...cardStyle, top, right,
      maxHeight: 'calc(100vh - 160px)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      <div style={{ ...sectionLabelStyle, flexShrink: 0 }}>point query</div>

      <div style={{ overflowY: 'auto', flex: 1 }}>
        {clickInfo ? (
          <>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 7 }}>
              <span style={{ color: DIM, fontSize: 14 }}>latitude</span>
              <span style={{ fontFamily: 'monospace', fontSize: 15 }}>{clickInfo.lat.toFixed(4)}°</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 7 }}>
              <span style={{ color: DIM, fontSize: 14 }}>longitude</span>
              <span style={{ fontFamily: 'monospace', fontSize: 15 }}>{clickInfo.lng.toFixed(4)}°</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 7 }}>
              <span style={{ color: DIM, fontSize: 14 }}>water year</span>
              <span style={{ fontFamily: 'monospace', fontSize: 15 }}>{waterYear}</span>
            </div>
            <div style={{ borderTop: `1px solid ${BORDER}`, margin: '10px 0' }} />
            {clickInfo.status === 'querying' ? (
              <div style={{ color: DIM, fontSize: 14 }}>querying…</div>
            ) : (
              QUERY_ROWS.map(({ key, label }) => (
                <div key={key} style={{
                  display: 'flex', justifyContent: 'space-between',
                  alignItems: 'baseline', gap: 8, marginBottom: 6,
                }}>
                  <span style={{ color: DIM, fontSize: 14, flexShrink: 0 }}>{label}</span>
                  <span style={{
                    fontFamily: 'monospace', fontSize: 16, fontWeight: 700,
                    textAlign: 'right',
                    color: clickInfo.values[key] === null ? DIM : ACCENT,
                  }}>
                    {formatValue(clickInfo.values[key], key, waterYear, southernHemisphere)}
                  </span>
                </div>
              ))
            )}
            <div style={{ color: DIM, fontSize: 10, marginTop: 8, lineHeight: 1.5 }}>
              Values read from the full-resolution (~80 m) data. Yearly variables use the
              selected water year; median/MAD span all years.
            </div>
          </>
        ) : (
          <div style={{ color: DIM, fontStyle: 'italic', fontSize: 14 }}>
            Click the map to query
          </div>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// FloatingCards — always rendered (desktop only via parent Box in index.tsx)
// ---------------------------------------------------------------------------

export const FloatingCards = () => {
  const CARD_RIGHT = 16
  const TOP_RIGHT_TOP = 16
  const TOP_CARD_HEIGHT = 138
  const QUERY_CARD_TOP = TOP_RIGHT_TOP + TOP_CARD_HEIGHT + 4

  return (
    <>
      <TopRightCard right={CARD_RIGHT} top={TOP_RIGHT_TOP} />
      <PointQueryCard right={CARD_RIGHT} top={QUERY_CARD_TOP} />
      <ZoomDisplay right={CARD_RIGHT} bottom={8} />
    </>
  )
}
