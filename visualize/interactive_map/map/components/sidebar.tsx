import React, { useEffect, useState, CSSProperties } from 'react'
import { Box, Flex } from 'theme-ui'
import { Slider, Input } from '@carbonplan/components'
import {
  useStore,
  VARIABLE_CONFIGS,
  WATER_YEARS,
  COMPOSITE_VARIABLES,
  ANNUAL_VARIABLES,
  type Variable,
} from '../lib/store'
import { COLORMAP_ARRAYS, MONTH_STARTS_DOWY } from '../lib/colormaps'

export const SIDEBAR_WIDTH = 380
const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const TEXT = '#d0d0d0'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'

const sectionLabelStyle: CSSProperties = {
  fontSize: 10,
  letterSpacing: '0.06em',
  color: DIM,
  fontWeight: 600,
  marginBottom: 8,
}

const variableButtonStyle = (active: boolean): CSSProperties => ({
  display: 'block',
  width: '100%',
  textAlign: 'left',
  padding: '4px 8px',
  marginBottom: 4,
  borderRadius: 4,
  border: `1px solid ${active ? ACCENT : BORDER}`,
  background: active ? 'rgba(29,189,143,0.15)' : 'transparent',
  color: active ? ACCENT : TEXT,
  fontSize: 12,
  cursor: 'pointer',
  transition: 'background 0.15s, border-color 0.15s, color 0.15s',
})

/** Horizontal gradient bar from a colormap ramp; DOWY variables get month
 *  boundary ticks with BOTH hemispheres' month labels (NH row + italic SH
 *  row, mirroring plot_utils.create_month_colorbar(hemisphere='both')). */
const FixedColorbar = ({
  colormap,
  clim,
  showMonths,
}: {
  colormap: string[]
  clim: [number, number]
  showMonths: boolean
}) => {
  const gradient = `linear-gradient(to right, ${colormap.join(', ')})`
  const [lo, hi] = clim
  const span = hi - lo

  // month boundaries falling inside the clim range, as bar fractions
  const ticks =
    showMonths && span > 0
      ? MONTH_STARTS_DOWY.filter(({ dowy }) => dowy > lo && dowy < hi).map(
          ({ month, shMonth, dowy }) => ({ month, shMonth, frac: (dowy - lo) / span })
        )
      : []

  return (
    <div>
      <div
        style={{
          position: 'relative',
          height: 14,
          borderRadius: 3,
          background: gradient,
          overflow: 'hidden',
        }}
      >
        {ticks.map(({ frac }, i) => (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: `${frac * 100}%`,
              top: 0,
              bottom: 0,
              width: 1,
              background: 'rgba(255,255,255,0.85)',
            }}
          />
        ))}
      </div>
      {showMonths && (
        <>
          {/* northern-hemisphere months (WY starts Oct 1) */}
          <div style={{ position: 'relative', height: 12, marginTop: 2 }}>
            {ticks.map(({ month, frac }, i) => (
              <span
                key={i}
                style={{
                  position: 'absolute',
                  left: `${frac * 100}%`,
                  transform: 'translateX(2px)',
                  fontSize: 9,
                  color: TEXT,
                  whiteSpace: 'nowrap',
                }}
              >
                {month}
              </span>
            ))}
          </div>
          {/* southern-hemisphere months (WY starts Apr 1; same DOWY, +6 months) */}
          <div style={{ position: 'relative', height: 12 }}>
            {ticks.map(({ shMonth, frac }, i) => (
              <span
                key={i}
                style={{
                  position: 'absolute',
                  left: `${frac * 100}%`,
                  transform: 'translateX(2px)',
                  fontSize: 8.5,
                  fontStyle: 'italic',
                  color: DIM,
                  whiteSpace: 'nowrap',
                }}
              >
                ({shMonth})
              </span>
            ))}
          </div>
          <div style={{ fontSize: 8.5, color: DIM, marginTop: 2 }}>
            N. hemisphere month <span style={{ fontStyle: 'italic' }}>(S. hemisphere month)</span>
          </div>
        </>
      )}
    </div>
  )
}

const SidebarContent = () => {
  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const opacity = useStore((s) => s.opacity)
  const clim = useStore((s) => s.clim)
  const seasonalOnly = useStore((s) => s.seasonalOnly)
  const seasonalMaskAvailable = useStore((s) => s.seasonalMaskAvailable)
  const setVariable = useStore((s) => s.setVariable)
  const setWaterYearIndex = useStore((s) => s.setWaterYearIndex)
  const setOpacity = useStore((s) => s.setOpacity)
  const setClim = useStore((s) => s.setClim)
  const setSeasonalOnly = useStore((s) => s.setSeasonalOnly)

  const vCfg = VARIABLE_CONFIGS[variable]
  const colormap = COLORMAP_ARRAYS[vCfg.colormap]
  const isDowy = vCfg.units === 'day of water year'
  const annualActive = vCfg.hasWaterYear

  const [climInputs, setClimInputs] = useState<[string, string]>([
    String(clim[0]),
    String(clim[1]),
  ])

  useEffect(() => {
    setClimInputs([String(clim[0]), String(clim[1])])
  }, [clim])

  const commitClim = (index: 0 | 1, value?: string) => {
    const val = parseFloat(value ?? climInputs[index])
    if (Number.isFinite(val)) {
      setClim(index === 0 ? [val, clim[1]] : [clim[0], val])
    } else {
      setClimInputs([String(clim[0]), String(clim[1])])
    }
  }

  const handleClimInput = (index: 0 | 1, newValue: string) => {
    const newNum = parseFloat(newValue)
    const isArrow = Number.isFinite(newNum) && Math.abs(newNum - clim[index]) <= 1.01
    if (isArrow) {
      commitClim(index, newValue)
    } else {
      setClimInputs(index === 0 ? [newValue, climInputs[1]] : [climInputs[0], newValue])
    }
  }

  const renderVariableColumn = (header: string, vars: Variable[]) => (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div style={{ fontSize: 10, color: DIM, lineHeight: 1.4, marginBottom: 6, minHeight: 28 }}>
        {header}
      </div>
      {vars.map((v) => (
        <button key={v} onClick={() => setVariable(v)} style={variableButtonStyle(v === variable)}>
          {VARIABLE_CONFIGS[v].label}
        </button>
      ))}
    </div>
  )

  return (
    <div style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 16, color: TEXT, fontSize: 13 }}>

      {/* Header */}
      <div>
        <Box
          as='h1'
          sx={{
            fontSize: [2, 2, 3, 3],
            fontFamily: 'heading',
            letterSpacing: 'heading',
            lineHeight: 'heading',
            color: 'primary',
            m: 0,
            mb: '4px',
          }}
        >
          Global snowmelt runoff onset
        </Box>
        <div style={{ fontSize: 11, color: DIM, lineHeight: 1.6, marginBottom: 4 }}>
          A global, ~80 meter resolution snowmelt runoff onset dataset derived from Sentinel-1
          synthetic aperture radar backscatter time series, covering water years 2015 to 2025.
          Per water year, each pixel stores the estimated day of snowmelt runoff onset [DOWY]
          (the C-band backscatter minimum within a MODIS-derived snow season search window) and
          the effective temporal resolution of that estimate [days]. Multi-year median and
          median-absolute-deviation composites summarize the full record. For methods, code,
          and data access, see the GitHub repo linked below.
        </div>
        <a
          href='https://github.com/egagli/global_snowmelt_runoff_onset'
          target='_blank'
          rel='noopener noreferrer'
          style={{ fontSize: 11, color: ACCENT, textDecoration: 'none' }}
        >
          egagli/global_snowmelt_runoff_onset ↗
        </a>
      </div>

      <div style={{ borderTop: `1px solid ${BORDER}` }} />

      {/* Variable — two columns: composites | annual */}
      <div>
        <div style={sectionLabelStyle}>variable</div>
        <div style={{ display: 'flex', gap: 10 }}>
          {renderVariableColumn(
            `multi-year composite products (WY${WATER_YEARS[0]}–WY${WATER_YEARS[WATER_YEARS.length - 1]})`,
            COMPOSITE_VARIABLES
          )}
          {renderVariableColumn('annual products', ANNUAL_VARIABLES)}
        </div>
        <div style={{ fontSize: 11, color: DIM, marginTop: 4 }}>
          {vCfg.label} [{vCfg.units}]
        </div>
      </div>

      {/* Water Year — disabled while a composite is selected */}
      <div
        style={{
          opacity: annualActive ? 1 : 0.35,
          pointerEvents: annualActive ? 'auto' : 'none',
        }}
        aria-disabled={!annualActive}
      >
        <div style={{ ...sectionLabelStyle, marginBottom: 4 }}>
          water year —{' '}
          <span style={{ color: TEXT, fontWeight: 700 }}>{WATER_YEARS[waterYearIndex]}</span>
          {!annualActive && (
            <span style={{ fontWeight: 400 }}> (composite spans all years)</span>
          )}
        </div>
        <Slider
          min={0}
          max={WATER_YEARS.length - 1}
          step={1}
          value={waterYearIndex}
          disabled={!annualActive}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setWaterYearIndex(parseInt(e.target.value))
          }
        />
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          {WATER_YEARS.map((year, i) => (
            <button
              key={year}
              onClick={() => setWaterYearIndex(i)}
              disabled={!annualActive}
              style={{
                background: 'none', border: 'none', padding: 0, margin: 0,
                fontSize: 9, lineHeight: 1, cursor: annualActive ? 'pointer' : 'default',
                fontFamily: 'monospace',
                color: i === waterYearIndex ? ACCENT : DIM,
              }}
            >
              {year}
            </button>
          ))}
        </div>
      </div>

      {/* Range — fixed colormap; user adjusts vmin/vmax only */}
      <div>
        <div style={sectionLabelStyle}>
          range{isDowy ? ' [day of water year]' : ' [days]'}
        </div>
        <Flex sx={{ gap: 2, alignItems: 'flex-start' }}>
          <Input
            size='xs'
            type='number'
            value={climInputs[0]}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(0, e.target.value)}
            onBlur={() => commitClim(0)}
            onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(0) }}
            sx={{ width: `${Math.max(2, climInputs[0].length + 2)}ch` }}
          />
          <Box sx={{ flex: 1, pt: '2px' }}>
            <FixedColorbar colormap={colormap} clim={clim} showMonths={isDowy} />
          </Box>
          <Input
            size='xs'
            type='number'
            value={climInputs[1]}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => handleClimInput(1, e.target.value)}
            onBlur={() => commitClim(1)}
            onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') commitClim(1) }}
            sx={{ width: `${Math.max(2, climInputs[1].length + 2)}ch` }}
          />
        </Flex>
      </div>

      {/* Opacity */}
      <div>
        <div style={sectionLabelStyle}>opacity</div>
        <Slider
          min={0}
          max={1}
          step={0.01}
          value={opacity}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
            setOpacity(parseFloat(e.target.value))
          }
        />
      </div>

      {/* Seasonal-snow display filter (issue #9) — applies to whichever
          variable/layer is active; a shader-side discard, no refetches. */}
      <div
        style={{
          opacity: seasonalMaskAvailable ? 1 : 0.35,
          pointerEvents: seasonalMaskAvailable ? 'auto' : 'none',
        }}
        aria-disabled={!seasonalMaskAvailable}
      >
        <div style={sectionLabelStyle}>display filter</div>
        <button
          onClick={() => setSeasonalOnly(!seasonalOnly)}
          disabled={!seasonalMaskAvailable}
          role='checkbox'
          aria-checked={seasonalOnly}
          style={{
            ...variableButtonStyle(seasonalOnly),
            display: 'flex',
            alignItems: 'center',
            gap: 8,
          }}
        >
          <span
            aria-hidden
            style={{
              width: 12,
              height: 12,
              flexShrink: 0,
              borderRadius: 3,
              border: `1px solid ${seasonalOnly ? ACCENT : DIM}`,
              background: seasonalOnly ? ACCENT : 'transparent',
              color: '#fff',
              fontSize: 10,
              lineHeight: '12px',
              textAlign: 'center',
            }}
          >
            {seasonalOnly ? '✓' : ''}
          </span>
          limit to seasonal snow (Sturm &amp; Liston 2021)
        </button>
        <div style={{ fontSize: 10, color: DIM, lineHeight: 1.5 }}>
          {seasonalMaskAvailable ? (
            <>
              Hides pixels outside seasonal snow classes in the Sturm &amp;
              Liston (2021) classification (zoomed out, cells less than half
              seasonal snow). Note this also hides valid estimates in
              ephemeral-snow regions (e.g. UK, Denmark, lowland Japan, New
              Zealand) — check the snow class in the point query, or switch the
              basemap to “snow class” to see the classification itself.
            </>
          ) : (
            <>Seasonal snow mask not yet available in this pyramid.</>
          )}
        </div>
      </div>

    </div>
  )
}

export const Sidebar = () => {
  const setSidebarWidth = useStore((s) => s.setSidebarWidth)

  useEffect(() => {
    setSidebarWidth(SIDEBAR_WIDTH + 32)
    return () => setSidebarWidth(0)
  }, [setSidebarWidth])

  return (
    <>
      {/* Desktop: floating card (overlays map, no left offset on map) */}
      <Box
        sx={{
          display: ['none', 'none', 'block'],
          position: 'absolute',
          top: 16,
          left: 16,
          width: SIDEBAR_WIDTH,
          maxHeight: 'calc(100vh - 32px)',
          bg: BG,
          border: `1px solid ${BORDER}`,
          borderRadius: 8,
          backdropFilter: 'blur(6px)',
          overflowY: 'auto',
          zIndex: 10,
        }}
      >
        <SidebarContent />
      </Box>

      {/* Mobile: bottom panel */}
      <Box
        sx={{
          display: ['block', 'block', 'none'],
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '50vh',
          bg: 'background',
          overflowY: 'auto',
          zIndex: 1000,
          px: [4, 5],
          py: [3],
          borderTop: '1px solid',
          borderColor: 'muted',
        }}
      >
        <SidebarContent />
      </Box>
    </>
  )
}
