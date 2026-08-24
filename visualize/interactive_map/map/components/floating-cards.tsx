import React, { CSSProperties, useEffect, useRef, useState } from 'react'
import {
  useStore,
  VERSION_CONFIGS,
  VARIABLE_CONFIGS,
  COMPOSITE_VARIABLES,
  ANNUAL_VARIABLES,
  SNOW_CLASS_INFO,
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

/** Legend for the categorical snow-class basemap — the colors the shader
 *  paints, in class-code order. Classes 8 (Ocean) and 9 (Fill) are omitted:
 *  they carry no class and the shader discards them. */
const SnowClassLegend = () => (
  <div style={{ marginTop: 12 }}>
    <div style={sectionLabelStyle}>snow classes (Sturm &amp; Liston 2021)</div>
    {Object.entries(SNOW_CLASS_INFO).map(([code, { name, color }]) => (
      <div
        key={code}
        style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 3 }}
      >
        <span
          aria-hidden
          style={{
            width: 11,
            height: 11,
            flexShrink: 0,
            borderRadius: 2,
            background: color,
            display: 'inline-block',
          }}
        />
        <span style={{ fontSize: 11 }}>{name}</span>
      </div>
    ))}
    <div style={{ fontSize: 10, color: DIM, marginTop: 5, lineHeight: 1.5 }}>
      Ocean and unclassified areas fall through to the dark basemap. Coarse
      zooms decimate (nearest), so small patches can drop out.
    </div>
  </div>
)

const TopRightCard = ({
  right,
  top,
  innerRef,
}: {
  right: number
  top: number
  innerRef?: React.Ref<HTMLDivElement>
}) => {
  const globeProjection = useStore((s) => s.globeProjection)
  const basemap = useStore((s) => s.basemap)
  const setGlobeProjection = useStore((s) => s.setGlobeProjection)
  const setBasemap = useStore((s) => s.setBasemap)

  const BASEMAP_OPTS: { label: string; value: Basemap }[] = [
    { label: 'dark',        value: 'dark'        },
    { label: 'satellite',   value: 'satellite'   },
    { label: 'topography',  value: 'topography'  },
    { label: 'snow class',  value: 'snowclass'   },
  ]

  return (
    <div ref={innerRef} style={{ ...cardStyle, top, right }}>
      <div style={sectionLabelStyle}>basemap</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
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
      {basemap === 'snowclass' && <SnowClassLegend />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// PointQueryCard — exact level-0 values at the clicked point
// ---------------------------------------------------------------------------

// Composites first, then annual -- same names as the variable selector; the
// annual rows get the active water year prepended at render time.
const QUERY_ROWS: Variable[] = [...COMPOSITE_VARIABLES, ...ANNUAL_VARIABLES]

const PointQueryCard = ({ right, top }: { right: number; top: number }) => {
  const clickInfo = useStore((s) => s.clickInfo)
  const version = useStore((s) => s.version)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const activeVariable = useStore((s) => s.variable)
  const waterYear = VERSION_CONFIGS[version].waterYears[waterYearIndex]
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
            {/* Sturm & Liston (2021) class at the point — shown regardless of
                the seasonal-only display filter. */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 7 }}>
              <span style={{ color: DIM, fontSize: 14 }}>
                snow class <span style={{ fontSize: 11 }}>(Sturm &amp; Liston 2021)</span>
              </span>
              {clickInfo.status === 'querying' ? (
                <span style={{ color: DIM, fontSize: 14 }}>…</span>
              ) : clickInfo.snowClass !== null && SNOW_CLASS_INFO[clickInfo.snowClass] ? (
                <span style={{ fontSize: 14, display: 'inline-flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                  <span
                    aria-hidden
                    style={{
                      width: 9,
                      height: 9,
                      borderRadius: 2,
                      background: SNOW_CLASS_INFO[clickInfo.snowClass].color,
                      display: 'inline-block',
                    }}
                  />
                  {SNOW_CLASS_INFO[clickInfo.snowClass].name}
                </span>
              ) : (
                <span style={{ color: DIM, fontSize: 14 }}>—</span>
              )}
            </div>
            <div style={{ borderTop: `1px solid ${BORDER}`, margin: '10px 0' }} />
            {clickInfo.status === 'querying' ? (
              <div style={{ color: DIM, fontSize: 14 }}>querying…</div>
            ) : (
              QUERY_ROWS.map((key) => {
                const cfg = VARIABLE_CONFIGS[key]
                const label = cfg.hasWaterYear ? `WY${waterYear} ${cfg.label}` : cfg.label
                const isActive = key === activeVariable
                return (
                  <div key={key} style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'baseline', gap: 8, marginBottom: 4,
                    padding: '2px 6px', borderRadius: 4,
                    background: isActive ? 'rgba(29,189,143,0.15)' : 'transparent',
                    border: `1px solid ${isActive ? ACCENT : 'transparent'}`,
                  }}>
                    <span style={{
                      color: isActive ? ACCENT : DIM,
                      fontSize: 13, flexShrink: 0,
                      fontWeight: isActive ? 700 : 400,
                    }}>{label}</span>
                    <span style={{
                      fontFamily: 'monospace', fontSize: 15, fontWeight: 700,
                      textAlign: 'right',
                      color: clickInfo.values[key] === null ? DIM : ACCENT,
                    }}>
                      {formatValue(clickInfo.values[key], key, waterYear, southernHemisphere)}
                    </span>
                  </div>
                )
              })
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
// WarningToast — region-specific caution notification (issue #13; same
// pattern as the MODIS_snow_phenology map). Set from WARNING_ZONES in
// map.tsx on every moveend/zoomend; dismissible per zone.
// ---------------------------------------------------------------------------

const WarningToast = () => {
  const activeWarning = useStore((s) => s.activeWarning)
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const [dismissed, setDismissed] = useState(false)

  // A new zone re-arms the toast; re-entering the same zone after dismissing
  // it does not (the name only changes when the zone does).
  useEffect(() => { setDismissed(false) }, [activeWarning?.name])

  if (!activeWarning || dismissed) return null

  const left = (sidebarWidth ?? 0) + 16
  const right = CARD_WIDTH + 32 // leave room for the floating cards

  return (
    <div style={{
      position: 'absolute',
      top: 12,
      left,
      right,
      margin: '0 auto',
      maxWidth: 520,
      background: BG,
      border: '1px solid rgba(251,191,36,0.55)',
      backdropFilter: 'blur(6px)',
      borderRadius: 8,
      padding: '10px 14px',
      color: TEXT,
      fontSize: 12,
      zIndex: 20,
      display: 'flex',
      gap: 10,
      alignItems: 'flex-start',
    }}>
      <span style={{ fontSize: 16, lineHeight: 1.4, flexShrink: 0 }}>⚠</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 3, fontSize: 12 }}>
          {activeWarning.name}
        </div>
        <div style={{ color: DIM, lineHeight: 1.5 }}>{activeWarning.message}</div>
      </div>
      <button
        onClick={() => setDismissed(true)}
        style={{
          background: 'none', border: 'none', color: DIM,
          cursor: 'pointer', fontSize: 16, lineHeight: 1,
          padding: 0, flexShrink: 0,
        }}
      >
        ×
      </button>
    </div>
  )
}

// ---------------------------------------------------------------------------
// GmbaHoverCard — details of the hovered GMBA range (issue #13). Fed by the
// mousemove handler on the overlay's fill layer in map.tsx; pointerEvents
// none so it never steals the hover it displays.
// ---------------------------------------------------------------------------

const GmbaHoverCard = () => {
  const gmbaHover = useStore((s) => s.gmbaHover)
  const sidebarWidth = useStore((s) => s.sidebarWidth)

  if (!gmbaHover) return null

  const fmtInt = (v: number | null) =>
    v === null ? '—' : Math.round(v).toLocaleString('en-US')

  return (
    <div style={{
      position: 'absolute',
      left: (sidebarWidth ?? 0) + 16,
      bottom: 24,
      maxWidth: 360,
      background: BG,
      border: `1px solid ${BORDER}`,
      backdropFilter: 'blur(6px)',
      borderRadius: 8,
      padding: '10px 14px',
      color: TEXT,
      fontSize: 12,
      zIndex: 15,
      pointerEvents: 'none',
    }}>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2 }}>
        {gmbaHover.name}
      </div>
      <div style={{ color: DIM, fontSize: 11, marginBottom: 6, lineHeight: 1.4 }}>
        {gmbaHover.feature}
        {gmbaHover.countries ? ` · ${gmbaHover.countries}` : ''}
      </div>
      <div style={{ display: 'flex', gap: 16, fontFamily: 'monospace', fontSize: 12 }}>
        <span>{fmtInt(gmbaHover.areaKm2)} km²</span>
        <span>{fmtInt(gmbaHover.elevLow)}–{fmtInt(gmbaHover.elevHigh)} m</span>
      </div>
      <div style={{ color: DIM, fontSize: 9.5, marginTop: 6 }}>
        GMBA Mountain Inventory v2.0 (Snethlage et al., 2022)
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

  // The top card's height is content-dependent (the snow-class legend adds a
  // block when that basemap is selected), so measure it instead of assuming —
  // a stale constant would let the two cards overlap.
  const topCardRef = useRef<HTMLDivElement>(null)
  const [topCardHeight, setTopCardHeight] = useState(138)
  useEffect(() => {
    const el = topCardRef.current
    if (!el) return
    const observer = new ResizeObserver(() => setTopCardHeight(el.offsetHeight))
    observer.observe(el)
    setTopCardHeight(el.offsetHeight)
    return () => observer.disconnect()
  }, [])

  const QUERY_CARD_TOP = TOP_RIGHT_TOP + topCardHeight + 4

  return (
    <>
      <TopRightCard right={CARD_RIGHT} top={TOP_RIGHT_TOP} innerRef={topCardRef} />
      <PointQueryCard right={CARD_RIGHT} top={QUERY_CARD_TOP} />
      <ZoomDisplay right={CARD_RIGHT} bottom={8} />
      <WarningToast />
      <GmbaHoverCard />
    </>
  )
}
