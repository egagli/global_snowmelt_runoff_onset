import React, { CSSProperties, useEffect, useMemo, useRef, useState } from 'react'
import { Box } from 'theme-ui'
import { useStore, loadGmbaIndex, type GmbaInfo } from '../lib/store'

// Search box (issue #36): one input, three kinds of result —
//  - coordinates ('46.5, 8.0', '46.5N 8.0E') → fly there and run the point query
//  - GMBA ranges, matched client-side against the lazy-loaded index (name,
//    then hierarchy path, so 'alps' also finds the Western/Eastern Alps) →
//    turn the overlay on, pin the range, fit its bounds
//  - places from Photon (komoot's OSM geocoder, which permits as-you-type
//    use, unlike Nominatim) → fit the place's extent, or fly + point query
//    for point features like peaks

const BG = 'rgba(22,25,30,0.96)'
const BORDER = '#2e3138'
const TEXT = '#d0d0d0'
const DIM = '#6b7280'
const ACCENT = '#1dbd8f'

const PHOTON_URL = 'https://photon.komoot.io/api/'
const PHOTON_DEBOUNCE_MS = 350
const MAX_RANGES = 6
const MAX_PLACES = 5

type Result =
  | { kind: 'coords'; lat: number; lng: number }
  | { kind: 'range'; range: GmbaInfo }
  | {
      kind: 'place'
      name: string
      detail: string
      lat: number
      lng: number
      extent: [number, number, number, number] | null // [w, s, e, n]
    }

/** 'lat, lon' with optional degree signs and N/S/E/W suffixes. */
export function parseCoords(q: string): { lat: number; lng: number } | null {
  const m = q
    .trim()
    .match(/^(-?\d+(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,;\s]\s*(-?\d+(?:\.\d+)?)\s*°?\s*([EWew])?$/)
  if (!m) return null
  let lat = parseFloat(m[1])
  let lng = parseFloat(m[3])
  if (m[2] && /s/i.test(m[2])) lat = -Math.abs(lat)
  if (m[4] && /w/i.test(m[4])) lng = -Math.abs(lng)
  if (Math.abs(lat) > 90 || Math.abs(lng) > 180) return null
  return { lat, lng }
}

const fold = (s: string) =>
  s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase()

function matchRanges(index: GmbaInfo[], q: string): GmbaInfo[] {
  const needle = fold(q.trim())
  if (needle.length < 2) return []
  const scored: [number, GmbaInfo][] = []
  for (const r of index) {
    const name = fold(r.name)
    let score = -1
    if (name.startsWith(needle)) score = 0
    else if (name.split(/[\s-]+/).some((w) => w.startsWith(needle))) score = 1
    else if (name.includes(needle)) score = 2
    else if (fold(r.path).includes(needle)) score = 3
    if (score >= 0) scored.push([score, r])
  }
  scored.sort((a, b) => a[0] - b[0] || (b[1].area_km2 ?? 0) - (a[1].area_km2 ?? 0))
  return scored.slice(0, MAX_RANGES).map(([, r]) => r)
}

async function searchPlaces(q: string, signal: AbortSignal): Promise<Result[]> {
  const url = `${PHOTON_URL}?q=${encodeURIComponent(q)}&limit=${MAX_PLACES}&lang=en`
  const res = await fetch(url, { signal })
  if (!res.ok) return []
  const json = await res.json()
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return (json.features ?? []).map((f: any): Result => {
    const p = f.properties ?? {}
    const [lng, lat] = f.geometry.coordinates
    // Photon extent order is [west, north, east, south]
    const ext = Array.isArray(p.extent) && p.extent.length === 4
      ? ([p.extent[0], p.extent[3], p.extent[2], p.extent[1]] as [number, number, number, number])
      : null
    const type = String(p.osm_value ?? '').replace(/_/g, ' ')
    const where = [p.state, p.country].filter(Boolean).join(', ')
    return {
      kind: 'place',
      name: p.name || where || `${lat.toFixed(3)}, ${lng.toFixed(3)}`,
      detail: [type, where].filter(Boolean).join(' · '),
      lat,
      lng,
      extent: ext,
    }
  })
}

const rowStyle = (active: boolean): CSSProperties => ({
  padding: '6px 10px',
  cursor: 'pointer',
  background: active ? 'rgba(29,189,143,0.15)' : 'transparent',
  borderLeft: `2px solid ${active ? ACCENT : 'transparent'}`,
})

const groupLabelStyle: CSSProperties = {
  fontSize: 9.5,
  letterSpacing: '0.06em',
  color: DIM,
  fontWeight: 600,
  padding: '7px 10px 3px',
}

export const SearchBox = () => {
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const gmbaIndex = useStore((s) => s.gmbaIndex)
  const requestNav = useStore((s) => s.requestNav)
  const setGmbaOn = useStore((s) => s.setGmbaOn)
  const setGmbaPinnedId = useStore((s) => s.setGmbaPinnedId)

  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const [places, setPlaces] = useState<Result[]>([])
  const [placesLoading, setPlacesLoading] = useState(false)
  const boxRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const coords = useMemo(() => parseCoords(query), [query])
  const ranges = useMemo(
    () => (coords || !gmbaIndex ? [] : matchRanges(gmbaIndex, query)),
    [coords, gmbaIndex, query]
  )

  // Debounced place search; skipped for coordinate input.
  useEffect(() => {
    setPlaces([])
    const q = query.trim()
    if (coords || q.length < 3) {
      setPlacesLoading(false)
      return
    }
    setPlacesLoading(true)
    const ctrl = new AbortController()
    const timer = setTimeout(() => {
      searchPlaces(q, ctrl.signal)
        .then(setPlaces)
        .catch(() => {})
        .finally(() => { if (!ctrl.signal.aborted) setPlacesLoading(false) })
    }, PHOTON_DEBOUNCE_MS)
    return () => { clearTimeout(timer); ctrl.abort() }
  }, [query, coords])

  // Close on outside click.
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [])

  const results: Result[] = [
    ...(coords ? [{ kind: 'coords' as const, ...coords }] : []),
    ...ranges.map((range) => ({ kind: 'range' as const, range })),
    ...places,
  ]
  useEffect(() => setActive(0), [query])

  const select = (r: Result) => {
    if (r.kind === 'coords') {
      requestNav({ kind: 'point', lat: r.lat, lng: r.lng, fly: true })
    } else if (r.kind === 'range') {
      setGmbaOn(true)
      setGmbaPinnedId(r.range.id)
      requestNav({ kind: 'bounds', bbox: r.range.bbox })
    } else if (r.extent && r.extent[2] - r.extent[0] > 0.05) {
      requestNav({ kind: 'bounds', bbox: r.extent })
    } else {
      requestNav({ kind: 'point', lat: r.lat, lng: r.lng, fly: true })
    }
    setOpen(false)
    inputRef.current?.blur()
  }

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setOpen(true)
      setActive((a) => Math.min(a + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((a) => Math.max(a - 1, 0))
    } else if (e.key === 'Enter') {
      if (results[active]) select(results[active])
    } else if (e.key === 'Escape') {
      setOpen(false)
      inputRef.current?.blur()
    }
  }

  const showDropdown = open && query.trim().length > 0
  const firstRange = coords ? 1 : 0
  const firstPlace = firstRange + ranges.length

  return (
    <Box
      ref={boxRef}
      sx={{
        position: 'absolute',
        // px strings: bare numbers index the theme's space scale
        top: ['10px', '10px', '16px'],
        left: ['10px', '10px', `${(sidebarWidth ?? 0) + 16}px`],
        width: ['calc(100vw - 20px)', 340, 340],
        zIndex: 25,
        fontSize: 12,
        color: TEXT,
      }}
    >
      <input
        ref={inputRef}
        type='search'
        value={query}
        placeholder='Search a place, mountain range, or lat, lon'
        aria-label='Search a place, mountain range, or coordinates'
        onChange={(e) => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => { setOpen(true); loadGmbaIndex() }}
        onKeyDown={onKeyDown}
        style={{
          width: '100%',
          boxSizing: 'border-box',
          padding: '8px 12px',
          borderRadius: showDropdown ? '8px 8px 0 0' : 8,
          border: `1px solid ${BORDER}`,
          background: BG,
          color: TEXT,
          fontSize: 13,
          outline: 'none',
          backdropFilter: 'blur(6px)',
        }}
      />
      {showDropdown && (
        <div
          role='listbox'
          style={{
            background: BG,
            border: `1px solid ${BORDER}`,
            borderTop: 'none',
            borderRadius: '0 0 8px 8px',
            maxHeight: '60vh',
            overflowY: 'auto',
            paddingBottom: 4,
          }}
        >
          {results.map((r, i) => (
            <React.Fragment key={i}>
              {i === firstRange && r.kind === 'range' && (
                <div style={groupLabelStyle}>MOUNTAIN RANGES (GMBA v2.0)</div>
              )}
              {i === firstPlace && r.kind === 'place' && (
                <div style={groupLabelStyle}>PLACES</div>
              )}
              <div
                role='option'
                aria-selected={i === active}
                onMouseEnter={() => setActive(i)}
                onMouseDown={(e) => { e.preventDefault(); select(r) }}
                style={rowStyle(i === active)}
              >
                {r.kind === 'coords' && (
                  <>
                    <div>
                      Go to <span style={{ fontFamily: 'monospace' }}>
                        {r.lat.toFixed(4)}°, {r.lng.toFixed(4)}°
                      </span>
                    </div>
                    <div style={{ color: DIM, fontSize: 10.5 }}>and query this point</div>
                  </>
                )}
                {r.kind === 'range' && (
                  <>
                    <div>{r.range.name}</div>
                    <div style={{
                      color: DIM, fontSize: 10.5,
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {r.range.path.split(' > ').slice(-2, -1)[0] ?? ''}
                      {r.range.countries ? ` · ${r.range.countries}` : ''}
                    </div>
                  </>
                )}
                {r.kind === 'place' && (
                  <>
                    <div>{r.name}</div>
                    <div style={{ color: DIM, fontSize: 10.5 }}>{r.detail}</div>
                  </>
                )}
              </div>
            </React.Fragment>
          ))}
          {results.length === 0 && !placesLoading && (
            <div style={{ ...groupLabelStyle, fontWeight: 400, letterSpacing: 0, fontSize: 11 }}>
              {gmbaIndex || query.trim().length >= 3 ? 'No matches' : 'Searching…'}
            </div>
          )}
          {placesLoading && (
            <div style={{ ...groupLabelStyle, fontWeight: 400, letterSpacing: 0, fontSize: 11 }}>
              Searching places…
            </div>
          )}
          {!coords && query.trim().length >= 3 && (
            <div style={{ color: DIM, fontSize: 9.5, padding: '4px 10px 2px' }}>
              Places by{' '}
              <a href='https://photon.komoot.io' target='_blank' rel='noopener noreferrer' style={{ color: DIM }}>
                Photon
              </a>{' '}
              · ©{' '}
              <a href='https://www.openstreetmap.org/copyright' target='_blank' rel='noopener noreferrer' style={{ color: DIM }}>
                OpenStreetMap
              </a>
            </div>
          )}
        </div>
      )}
    </Box>
  )
}
