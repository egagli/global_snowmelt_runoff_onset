import {
  useStore,
  VARIABLE_CONFIGS,
  VERSION_CONFIGS,
  ALL_VERSIONS,
  ALL_VARIABLES,
  type Basemap,
  type DatasetVersion,
  type Variable,
} from './store'

// Shareable view state in the URL hash (issue #36):
//   #map=zoom/lat/lon&v=v10&var=runoff_onset&wy=2023&pt=46.5,8.0&gmba=1&range=12345&bm=satellite
//    &clim=100,250&op=0.7&seas=0&proj=mercator   (each only when it differs from the default)
// maplibre owns `map=` (its named-hash option, see map.tsx) and preserves the
// other params when it rewrites; this module owns the rest and preserves
// `map=`. Values are written raw (not URL-encoded) because maplibre splits
// its own value on '/', which encodeURIComponent would escape.

const OWN_KEYS = ['v', 'var', 'wy', 'pt', 'gmba', 'range', 'bm', 'clim', 'op', 'seas', 'proj'] as const

// "Share this view" links point at this map's page on egagli.github.io, which embeds
// the app and passes its own query and hash through to it. When embedded there, the app
// also posts its state up after each change so that page's address bar stays shareable.
export const SHARE_BASE = 'https://egagli.github.io/maps/global-snowmelt-runoff-onset.html'
const SITE_ORIGIN = 'https://egagli.github.io'

export function shareUrl(): string {
  return SHARE_BASE + window.location.search + window.location.hash
}

let notifyTimer: ReturnType<typeof setTimeout> | undefined
/** Tell the embedding page (if any) the current search + hash. Debounced past
 *  maplibre's own throttled `map=` hash update (300 ms) so the camera is current. */
export function notifyParent() {
  if (typeof window === 'undefined' || window.parent === window) return
  clearTimeout(notifyTimer)
  notifyTimer = setTimeout(() => {
    try {
      window.parent.postMessage(
        { type: 'egagli-map-state', search: window.location.search, hash: window.location.hash },
        SITE_ORIGIN,
      )
    } catch {}
  }, 400)
}
type Camera = { center: [number, number]; zoom: number; bearing: number; pitch: number }
const BASEMAPS: Basemap[] = ['dark', 'satellite', 'topography', 'snowclass']

function readParams(): Map<string, string> {
  const params = new Map<string, string>()
  window.location.hash
    .replace(/^#/, '')
    .split('&')
    .filter(Boolean)
    .forEach((part) => {
      const i = part.indexOf('=')
      if (i > 0) params.set(part.slice(0, i), part.slice(i + 1))
    })
  return params
}

let initialCameraCache: Camera | null | undefined

/** The camera in the page's ORIGINAL `map=zoom/lat/lon[/bearing[/pitch]]`,
 *  read once and cached. map.tsx passes it as the constructor's center/zoom:
 *  maplibre reads the hash itself too, but its Hash.remove() strips `map=`
 *  from the URL, so after React strict mode's dev-only unmount/remount the
 *  second map would otherwise start at the default view. */
export function initialCamera(): Camera | null {
  if (initialCameraCache === undefined) {
    const v = readParams().get('map')?.split('/').map(Number) ?? []
    initialCameraCache =
      v.length >= 3 && v.every(Number.isFinite)
        ? { zoom: v[0], center: [v[2], v[1]], bearing: v[3] ?? 0, pitch: v[4] ?? 0 }
        : null
  }
  return initialCameraCache
}

function writeParams(own: Record<string, string | null>) {
  const params = readParams()
  OWN_KEYS.forEach((k) => {
    const v = own[k]
    if (v === null || v === undefined) params.delete(k)
    else params.set(k, v)
  })
  const hash = Array.from(params, ([k, v]) => `${k}=${v}`).join('&')
  const url = `${window.location.pathname}${window.location.search}${hash ? `#${hash}` : ''}`
  window.history.replaceState(window.history.state, '', url)
  notifyParent()
}

/** Apply the hash to the store (call once on mount, before the map exists),
 *  then keep the hash in sync with the store. Returns the unsubscribe. */
export function syncUrlHash(): () => void {
  const params = readParams()
  const s = useStore.getState()
  initialCamera() // cache before maplibre can touch `map=`

  const v = params.get('v') as DatasetVersion | undefined
  if (v && ALL_VERSIONS.includes(v)) s.setVersion(v)
  const variable = params.get('var') as Variable | undefined
  if (variable && ALL_VARIABLES.includes(variable)) s.setVariable(variable)
  const wy = Number(params.get('wy'))
  const yearIdx = VERSION_CONFIGS[useStore.getState().version].waterYears.indexOf(wy)
  if (yearIdx >= 0) s.setWaterYearIndex(yearIdx)
  const bm = params.get('bm') as Basemap | undefined
  if (bm && BASEMAPS.includes(bm)) s.setBasemap(bm)
  // after setVariable, which resets clim to the variable's default
  const clim = params.get('clim')?.split(',').map(Number)
  if (clim && clim.length === 2 && clim.every(Number.isFinite) && clim[0] < clim[1]) {
    s.setClim([clim[0], clim[1]])
  }
  const op = Number(params.get('op'))
  if (params.has('op') && Number.isFinite(op) && op >= 0 && op <= 1) s.setOpacity(op)
  if (params.get('seas') === '0') s.setSeasonalOnly(false)
  if (params.get('proj') === 'mercator') s.setGlobeProjection(false)
  if (params.get('gmba') === '1') {
    s.setGmbaOn(true)
    const range = Number(params.get('range'))
    if (Number.isInteger(range) && range > 0) s.setGmbaPinnedId(range)
  }
  const pt = params.get('pt')?.split(',').map(Number)
  if (pt && pt.length === 2 && pt.every(Number.isFinite)) {
    s.requestNav({ kind: 'point', lat: pt[0], lng: pt[1], fly: false })
  }

  const write = () => {
    const st = useStore.getState()
    writeParams({
      v: st.version,
      var: st.variable,
      wy: String(VERSION_CONFIGS[st.version].waterYears[st.waterYearIndex]),
      pt: st.clickInfo
        ? `${st.clickInfo.lat.toFixed(5)},${st.clickInfo.lng.toFixed(5)}`
        : null,
      gmba: st.gmbaOn ? '1' : null,
      range: st.gmbaOn && st.gmbaPinnedId !== null ? String(st.gmbaPinnedId) : null,
      bm: st.basemap === 'dark' ? null : st.basemap,
      clim:
        st.clim[0] === VARIABLE_CONFIGS[st.variable].clim[0] &&
        st.clim[1] === VARIABLE_CONFIGS[st.variable].clim[1]
          ? null
          : `${st.clim[0]},${st.clim[1]}`,
      op: st.opacity === 1 ? null : String(Math.round(st.opacity * 100) / 100),
      seas: st.seasonalOnly ? null : '0',
      proj: st.globeProjection ? null : 'mercator',
    })
  }
  // No initial write: the store already mirrors the hash, and writing now
  // would drop `pt=` until the restored point's query fills clickInfo.
  notifyParent()
  return useStore.subscribe((st, prev) => {
    if (
      st.version !== prev.version ||
      st.variable !== prev.variable ||
      st.waterYearIndex !== prev.waterYearIndex ||
      st.gmbaOn !== prev.gmbaOn ||
      st.gmbaPinnedId !== prev.gmbaPinnedId ||
      st.basemap !== prev.basemap ||
      st.clim !== prev.clim ||
      st.opacity !== prev.opacity ||
      st.seasonalOnly !== prev.seasonalOnly ||
      st.globeProjection !== prev.globeProjection ||
      st.clickInfo?.lat !== prev.clickInfo?.lat ||
      st.clickInfo?.lng !== prev.clickInfo?.lng
    ) {
      write()
    }
  })
}
