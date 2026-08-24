import React, { useEffect, useRef, useState } from 'react'
import { Box, Spinner } from 'theme-ui'
import { ZarrLayer, ZarrLayerOptions } from '@carbonplan/zarr-layer'
import FetchStore from '@zarrita/storage/fetch'
import { open, get } from 'zarrita'
import maplibregl from 'maplibre-gl'
import { layers, namedFlavor } from '@protomaps/basemaps'
import { Protocol } from 'pmtiles'
import {
  useStore,
  VERSION_CONFIGS,
  ALL_VERSIONS,
  GMBA_URL,
  SNOW_CLASS_URL,
  SNOW_CLASS_VARIABLE,
  SNOW_CLASS_INFO,
  SEASONAL_MASK_VARIABLE,
  SEASONAL_MASK_THRESHOLD,
  VARIABLE_CONFIGS,
  ALL_VARIABLES,
  type DatasetVersion,
  type Variable,
  type ClickInfo,
} from '../lib/store'
import { COLORMAP_ARRAYS } from '../lib/colormaps'

const ACCENT = '#1dbd8f'
const FILL_VALUE = -9999
const SNOW_CLASS_FILL = 9
const SNOW_CLASS_LAYER_ID = 'zarr-snow-class'
const GMBA_SOURCE_ID = 'gmba'
const GMBA_FILL_ID = 'gmba-fill'
const GMBA_LINE_ID = 'gmba-line'

// Regions where estimates are known to be absent or need caution — checked
// against the viewport on every moveend/zoomend and surfaced as a dismissible
// toast (issue #13; same pattern as the MODIS_snow_phenology map). The MODIS
// zones matter here too because that product defines this dataset's
// melt-search window, so its false-snow artifacts propagate.
const WARNING_ZONES: {
  name: string
  message: string
  bbox: [number, number, number, number] // [west, south, east, north]
  minZoom: number
}[] = [
  {
    name: 'Greenland — no data (HH polarization)',
    bbox: [-74, 59, -10, 84],
    minZoom: 3,
    message:
      'Sentinel-1 RTC scenes over Greenland are acquired in HH/HV polarization. ' +
      'This dataset requires VV backscatter, so Greenland has no runoff onset ' +
      'estimates. HH support is a candidate for a future version.',
  },
  {
    name: 'Canadian Arctic Archipelago — no data (HH polarization)',
    bbox: [-128, 66, -60, 84],
    minZoom: 3,
    message:
      'Sentinel-1 RTC scenes over the Canadian Arctic Archipelago are acquired ' +
      'in HH/HV polarization. This dataset requires VV backscatter, so the ' +
      'archipelago has no runoff onset estimates. HH support is a candidate ' +
      'for a future version.',
  },
  {
    name: 'Equator — hemisphere seam (Volcán Cayambe)',
    bbox: [-79.5, -1.5, -76.5, 1.5],
    minZoom: 7,
    message:
      'Tiles are processed per hemisphere with different water-year ' +
      'conventions, which meet at the equator: a ~160 m no-data strip sits ' +
      'just south of it, and the MODIS-derived snow-season window blends both ' +
      'conventions within ~500 m of it. Volcán Cayambe is the only ' +
      'seasonal-snow area on the equator — interpret estimates here with caution.',
  },
  {
    name: 'Salt Flats — Atacama / Altiplano',
    bbox: [-73, -28, -60, -14],
    minZoom: 6,
    message:
      'Salt flats in this region (e.g., Salar de Uyuni, Salar de Coipasa) cause ' +
      'false-positive snow detections in the MODIS snow product that defines ' +
      'this dataset’s melt-search window. Runoff onset estimates on or near ' +
      'salares may reflect a spurious snow season — interpret with caution.',
  },
  {
    name: 'Tibetan Plateau — Turbid & Shallow Lakes',
    bbox: [78, 27, 105, 38],
    minZoom: 6,
    message:
      'Turbid and shallow water bodies, especially on the Tibetan Plateau, can ' +
      'trigger false-positive snow presence in the MODIS snow product that ' +
      'defines this dataset’s melt-search window. Interpret estimates on or ' +
      'near lakes with caution.',
  },
  {
    name: 'Eastern Tropical Andes — Cloud Cover',
    bbox: [-82, -22, -68, 8],
    minZoom: 6,
    message:
      'Near-permanent cloud cover on the eastern slopes of the tropical Andes ' +
      'causes cloud–snow misclassification in the MODIS snow product that ' +
      'defines this dataset’s melt-search window. Interpret estimates in this ' +
      'region with caution.',
  },
]

/** '#rrggbb' → a GLSL vec3 literal in 0-1 space. */
function hexToVec3(hex: string): string {
  const n = parseInt(hex.slice(1), 16)
  const c = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) =>
    (v / 255).toFixed(4)
  )
  return `vec3(${c.join(', ')})`
}

// Categorical shader for the snow-class basemap: class code → flat color.
// Codes 8 (Ocean) and 9 (Fill, → NaN) carry no class, so they discard and the
// basemap beneath shows through. A colormap texture can't express this — it
// interpolates, which would invent colors between class codes.
const SNOW_CLASS_FRAG = `
  if (${SNOW_CLASS_VARIABLE} != ${SNOW_CLASS_VARIABLE}) { discard; }
  int cls = int(${SNOW_CLASS_VARIABLE} + 0.5);
  vec3 c;
${Object.entries(SNOW_CLASS_INFO)
  .map(
    ([code, { color }], i) =>
      `  ${i === 0 ? 'if' : 'else if'} (cls == ${code}) { c = ${hexToVec3(color)}; }`
  )
  .join('\n')}
  else { discard; }
  fragColor = vec4(c * opacity, opacity);
`

// Inject keyframes for the pulsing marker once at module load.
if (typeof document !== 'undefined' && !document.getElementById('pulsing-marker-style')) {
  const s = document.createElement('style')
  s.id = 'pulsing-marker-style'
  s.textContent = `
    @keyframes markerPulse {
      0%   { transform: translate(-50%,-50%) scale(1); opacity: 0.55; }
      100% { transform: translate(-50%,-50%) scale(3.5); opacity: 0; }
    }
  `
  document.head.appendChild(s)
}

function createPulsingMarkerElement(): HTMLElement {
  const wrap = document.createElement('div')
  wrap.style.cssText = 'position:relative;width:14px;height:14px;'

  const ring = document.createElement('div')
  ring.style.cssText = `
    position:absolute;top:50%;left:50%;
    width:14px;height:14px;border-radius:50%;
    background:${ACCENT};
    animation:markerPulse 3.0s ease-out infinite;
  `

  const dot = document.createElement('div')
  dot.style.cssText = `
    position:absolute;top:50%;left:50%;
    transform:translate(-50%,-50%);
    width:8px;height:8px;border-radius:50%;
    background:${ACCENT};border:2px solid #fff;
  `

  wrap.appendChild(ring)
  wrap.appendChild(dot)
  return wrap
}

// Level-0 grid of a version's pyramid, for zoom-independent point queries.
// Parsed at startup from the store's own spatial:transform / spatial:shape
// root attrs (pixel registration: x = a*col + c, y = e*row + f give the
// pixel's top-left corner), so per-version grids are never hardcoded — a new
// version's pyramid self-describes. The v10 constants below are only the
// offline fallback if that fetch fails.
type GridSpec = {
  xOrigin: number
  yOrigin: number
  xRes: number
  yRes: number
  nRows: number
  nCols: number
}

const V10_FALLBACK_GRID: GridSpec = {
  xOrigin: -179.99945999946,
  yOrigin: 84.04856404856403,
  xRes: 0.0007200007200083292,
  yRes: 0.0007200007199941183,
  nRows: 204800,
  nCols: 499998,
}

/** Fetch + parse a pyramid's level-0 grid from its root zarr.json. Returns
 *  null when the store doesn't exist (probe-disabled version) or the attrs
 *  don't parse. */
async function fetchVersionGrid(zarrUrl: string): Promise<GridSpec | null> {
  try {
    const res = await fetch(`${zarrUrl}/zarr.json`)
    if (!res.ok) return null
    const attrs = (await res.json())?.attributes ?? {}
    const t = attrs['spatial:transform']
    const shape = attrs['spatial:shape']
    if (!Array.isArray(t) || t.length !== 6 || !Array.isArray(shape) || shape.length !== 2) {
      return null
    }
    return {
      xRes: t[0],
      xOrigin: t[2],
      yRes: -t[4], // stored as the (negative) row step
      yOrigin: t[5],
      nRows: shape[0],
      nCols: shape[1],
    }
  } catch {
    return null
  }
}

/** Convert WGS84 (lat, lon in degrees) → level-0 [row, col] on a version's
 *  grid. Returns null if the point is outside the grid. */
function latlonToRowCol(lat: number, lon: number, grid: GridSpec): [number, number] | null {
  const col = Math.floor((lon - grid.xOrigin) / grid.xRes)
  const row = Math.floor((grid.yOrigin - lat) / grid.yRes)
  if (row < 0 || row >= grid.nRows || col < 0 || col >= grid.nCols) return null
  return [row, col]
}

// Standalone 300 m (10 arcsec) Sturm & Liston classification grid — its own
// affine, independent of the pyramid's (full globe, GeoTIFF-derived; see
// visualize/interactive_map/build_snow_class_store.py).
const SC_Y_ORIGIN = 89.99999999994958
const SC_X_ORIGIN = -180.0
const SC_RES = 0.0027777777777770003
const SC_N_ROWS = 64800
const SC_N_COLS = 129600

function latlonToSnowClassRowCol(lat: number, lon: number): [number, number] | null {
  const row = Math.floor((SC_Y_ORIGIN - lat) / SC_RES)
  const col = Math.floor((lon - SC_X_ORIGIN) / SC_RES)
  if (row < 0 || row >= SC_N_ROWS || col < 0 || col >= SC_N_COLS) return null
  return [row, col]
}

const backgroundColor = '#1b1e23'
const mapTheme = {
  ...namedFlavor('black'),
  background: backgroundColor,
  earth: backgroundColor,
  park_a: backgroundColor,
  park_b: backgroundColor,
  golf_course: backgroundColor,
  aerodrome: backgroundColor,
  industrial: backgroundColor,
  university: backgroundColor,
  school: backgroundColor,
  zoo: backgroundColor,
  farmland: backgroundColor,
  wood_a: backgroundColor,
  wood_b: backgroundColor,
  residential: backgroundColor,
  protected_area: backgroundColor,
  scrub_a: backgroundColor,
  scrub_b: backgroundColor,
  landcover: {
    barren: backgroundColor,
    farmland: backgroundColor,
    forest: backgroundColor,
    glacier: backgroundColor,
    grassland: backgroundColor,
    scrub: backgroundColor,
    urban_area: backgroundColor,
  },
  regular: 'Relative Pro Book',
  bold: 'Relative Pro Book',
  italic: 'Relative Pro Book',
}

let pmtilesRegistered = false

const OWN_LAYER_IDS = new Set([
  'zarr-layer', 'esri-imagery', 'topo', SNOW_CLASS_LAYER_ID,
  GMBA_FILL_ID, GMBA_LINE_ID,
])

function setBasemapFillVisibility(map: maplibregl.Map, visible: boolean) {
  const vis = visible ? 'visible' : 'none'
  map.getStyle()?.layers.forEach((layer) => {
    if (OWN_LAYER_IDS.has(layer.id)) return
    if (layer.type === 'fill' || layer.type === 'background') {
      try { map.setLayoutProperty(layer.id, 'visibility', vis) } catch {}
    }
  })
}

function setBasemapSymbolVisibility(map: maplibregl.Map, visible: boolean) {
  const vis = visible ? 'visible' : 'none'
  map.getStyle()?.layers.forEach((layer) => {
    if (OWN_LAYER_IDS.has(layer.id)) return
    if (layer.type === 'symbol') {
      try { map.setLayoutProperty(layer.id, 'visibility', vis) } catch {}
    }
  })
}

function isValidValue(raw: unknown): raw is number {
  return typeof raw === 'number' && !isNaN(raw) && raw !== FILL_VALUE && raw > -100
}

const EMPTY_VALUES = Object.fromEntries(
  ALL_VARIABLES.map((v) => [v, null])
) as ClickInfo['values']

/** Probe whether a version's pyramid carries the seasonal-snow mask variable.
 *  The map can deploy before the mask job has written it — the toggle stays
 *  disabled and no layer references the missing array (which would fail level
 *  loads). */
async function probeSeasonalMask(zarrUrl: string): Promise<boolean> {
  try {
    const res = await fetch(`${zarrUrl}/0/${SEASONAL_MASK_VARIABLE}/zarr.json`)
    return res.ok
  } catch {
    return false
  }
}

export const Map = () => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const zarrLayersRef = useRef<Partial<Record<Variable, InstanceType<typeof ZarrLayer>>>>({})
  const snowClassLayerRef = useRef<InstanceType<typeof ZarrLayer> | null>(null)
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const lastClickRef = useRef<{ lng: number; lat: number } | null>(null)
  // Level-0 zarrita arrays for the ACTIVE version — reopened on version switch
  // and used for zoom-independent point queries.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const l0ArraysRef = useRef<Partial<Record<Variable, any>>>({})
  // Promise that resolves when all arrays are open — awaited in requeryAllVariables
  // so every query round reads from a consistent data source.
  const l0ArraysPromiseRef = useRef<Promise<void> | null>(null)
  // The standalone 300 m snow-classification array (query card class row).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const snowClassArrayRef = useRef<any>(null)
  // Resolves to whether the active version's pyramid has the seasonal-snow
  // mask — awaited by the layer-creation effect so the shader only samples an
  // existing array.
  const seasonalMaskProbeRef = useRef<Promise<boolean> | null>(null)
  // Per-version level-0 grids, parsed from each pyramid's root attrs at mount.
  const gridSpecsRef = useRef<Partial<Record<DatasetVersion, GridSpec>>>({})
  const gridPromisesRef = useRef<Partial<Record<DatasetVersion, Promise<GridSpec | null>>>>({})
  // Feature id currently hovered on the GMBA overlay (maplibre feature-state).
  const hoveredGmbaIdRef = useRef<number | string | null>(null)

  const setSeasonalMaskAvailable = useStore((s) => s.setSeasonalMaskAvailable)
  const setVersionAvailable = useStore((s) => s.setVersionAvailable)
  const setGmbaHover = useStore((s) => s.setGmbaHover)
  const setActiveWarning = useStore((s) => s.setActiveWarning)

  const [isMapLoaded, setIsMapLoaded] = useState(false)

  const version = useStore((s) => s.version)
  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const opacity = useStore((s) => s.opacity)
  const clim = useStore((s) => s.clim)
  const globeProjection = useStore((s) => s.globeProjection)
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const loadingState = useStore((s) => s.loadingState)
  const basemap = useStore((s) => s.basemap)
  const gmbaOn = useStore((s) => s.gmbaOn)
  const setLoadingState = useStore((s) => s.setLoadingState)
  const setClickInfo = useStore((s) => s.setClickInfo)
  const setZoomLevel = useStore((s) => s.setZoomLevel)

  // fixed matplotlib ramp per variable (no user colormap selection)
  const colormapArray = COLORMAP_ARRAYS[VARIABLE_CONFIGS[variable].colormap]

  // Probe every version's pyramid at mount: availability drives the sidebar
  // dropdown; the parsed grid drives point queries. v10 falls back to the
  // baked-in grid so the app still works if the metadata fetch fails.
  useEffect(() => {
    ALL_VERSIONS.forEach((v) => {
      const promise = fetchVersionGrid(VERSION_CONFIGS[v].zarrUrl).then((grid) => {
        if (!grid && v === 'v10') grid = V10_FALLBACK_GRID
        if (grid) gridSpecsRef.current[v] = grid
        setVersionAvailable(v, !!grid)
        return grid
      })
      gridPromisesRef.current[v] = promise
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // The standalone snow-class array is version-independent — open once at mount.
  useEffect(() => {
    ;(async () => {
      try {
        // level 0 of the class pyramid = the native 300 m grid
        const arrayStore = new FetchStore(
          `${SNOW_CLASS_URL}/0/${SNOW_CLASS_VARIABLE}`
        )
        snowClassArrayRef.current = await open(arrayStore, { kind: 'array' })
      } catch (e) {
        console.warn('Snow classification store unavailable:', e)
      }
    })()
  }, [])

  // Open level-0 arrays for all variables of the ACTIVE version (re-runs on
  // version switch). Each variable gets its own FetchStore pointing at its
  // sub-URL so zarrita opens the array at the correct path.
  useEffect(() => {
    const cancelled = { val: false }
    const zarrUrl = VERSION_CONFIGS[version].zarrUrl
    l0ArraysRef.current = {}
    setSeasonalMaskAvailable(false)
    l0ArraysPromiseRef.current = Promise.all(
      ALL_VARIABLES.map(async (varName) => {
        try {
          const arrayStore = new FetchStore(`${zarrUrl}/0/${varName}`)
          const arr = await open(arrayStore, { kind: 'array' })
          if (!cancelled.val) l0ArraysRef.current[varName] = arr
        } catch (e) {
          console.warn(`Failed to open level-0 array for ${varName}:`, e)
        }
      })
    ).then(() => undefined)

    seasonalMaskProbeRef.current = probeSeasonalMask(zarrUrl).then((ok) => {
      if (!cancelled.val) setSeasonalMaskAvailable(ok)
      return ok
    })

    // Re-query a pinned point against the new version's data.
    if (lastClickRef.current) requeryAllVariables(cancelled)

    return () => { cancelled.val = true }
  }, [version]) // eslint-disable-line react-hooks/exhaustive-deps

  // Map initialization — runs once
  useEffect(() => {
    if (!mapContainer.current || mapRef.current) return

    if (!pmtilesRegistered) {
      const protocol = new Protocol()
      maplibregl.addProtocol('pmtiles', protocol.tile)
      pmtilesRegistered = true
    }

    const pmLayers = layers('protomaps', mapTheme as any, { lang: 'en' })
    const satLayer = {
      id: 'esri-imagery',
      type: 'raster' as const,
      source: 'esri-imagery',
      layout: { visibility: 'none' as const },
    }
    const topoLayer = {
      id: 'topo',
      type: 'raster' as const,
      source: 'topo',
      layout: { visibility: 'none' as const },
    }
    const styleLayers = [pmLayers[0], satLayer, topoLayer, ...pmLayers.slice(1)]

    const map = new maplibregl.Map({
      container: mapContainer.current,
      attributionControl: false,
      style: {
        projection: { type: 'globe' } as any,
        version: 8,
        glyphs:
          'https://carbonplan-maps.s3.us-west-2.amazonaws.com/basemaps/fonts/{fontstack}/{range}.pbf',
        sources: {
          protomaps: {
            type: 'vector',
            url: 'pmtiles://https://carbonplan-maps.s3.us-west-2.amazonaws.com/basemaps/pmtiles/global.pmtiles',
            attribution:
              '<a href="https://overturemaps.org/">Overture Maps</a>, <a href="https://protomaps.com">Protomaps</a>, © <a href="https://openstreetmap.org">OpenStreetMap</a>',
          },
          'esri-imagery': {
            type: 'raster',
            tiles: [
              'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            ],
            tileSize: 256,
            maxzoom: 19,
            attribution:
              'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
          },
          topo: {
            type: 'raster',
            tiles: [
              'https://tile.opentopomap.org/{z}/{x}/{y}.png',
            ],
            tileSize: 256,
            maxzoom: 17,
            attribution:
              'Map data: &copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">OpenTopoMap</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)',
          },
        },
        layers: styleLayers,
      },
      // Data spans ~81°N–60°S but is concentrated in NH mountains
      center: [0, 40],
      zoom: window.innerWidth < 640 ? 1.2 : 2.4,
    })

    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left')
    mapRef.current = map

    map.on('load', () => {
      setIsMapLoaded(true)
    })

    return () => {
      markerRef.current?.remove()
      markerRef.current = null
      map.remove()
      mapRef.current = null
      setIsMapLoaded(false)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Zoom display + warning zones — update on every moveend/zoomend
  useEffect(() => {
    const map = mapRef.current
    if (!map || !isMapLoaded) return
    const check = () => {
      const zoom = map.getZoom()
      setZoomLevel(zoom)
      const b = map.getBounds()
      const w = b.getWest(), e = b.getEast(), s = b.getSouth(), n = b.getNorth()
      for (const zone of WARNING_ZONES) {
        if (zoom < zone.minZoom) continue
        const [zw, zs, ze, zn] = zone.bbox
        if (w < ze && e > zw && s < zn && n > zs) {
          setActiveWarning({ name: zone.name, message: zone.message })
          return
        }
      }
      setActiveWarning(null)
    }
    map.on('moveend', check)
    map.on('zoomend', check)
    check()
    return () => { map.off('moveend', check); map.off('zoomend', check) }
  }, [isMapLoaded, setZoomLevel, setActiveWarning])

  // Projection toggle
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    ;(mapRef.current as any).setProjection(
      globeProjection ? { type: 'globe' } : { type: 'mercator' }
    )
  }, [globeProjection, isMapLoaded])

  // Basemap toggle — satellite/topo hide the vector basemap fills to avoid
  // masking. The snow-class basemap keeps them: classes 8/9 (ocean, fill)
  // discard, so the dark map still supplies ocean and unclassified ground.
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    map.setLayoutProperty('esri-imagery', 'visibility', basemap === 'satellite' ? 'visible' : 'none')
    map.setLayoutProperty('topo', 'visibility', basemap === 'topography' ? 'visible' : 'none')
    setBasemapFillVisibility(map, basemap === 'dark' || basemap === 'snowclass')
    setBasemapSymbolVisibility(map, basemap !== 'topography')
    // Same park/unpark as the data layers: only fetch chunks when selected.
    const snowLayer = snowClassLayerRef.current
    if (snowLayer) {
      const active = basemap === 'snowclass'
      ;(snowLayer as any).maxZoom = active ? Infinity : -1
      snowLayer.setOpacity(active ? 1 : 0)
    }
  }, [basemap, isMapLoaded])

  // GMBA mountain-range overlay (issue #13). The GeoJSON source is added
  // lazily on first enable (maplibre fetches GMBA_URL itself — ~2 MB gzip);
  // afterwards the toggle is a pure visibility flip. Hover drives a
  // feature-state highlight plus the info card in floating-cards.tsx.
  useEffect(() => {
    const map = mapRef.current
    if (!map || !isMapLoaded) return

    if (gmbaOn && !map.getSource(GMBA_SOURCE_ID)) {
      // generateId gives numeric feature ids, required for setFeatureState.
      map.addSource(GMBA_SOURCE_ID, {
        type: 'geojson',
        data: GMBA_URL,
        generateId: true,
        attribution:
          '<a href="https://doi.org/10.1038/s41597-022-01256-y">GMBA Mountain Inventory v2.0</a>',
      })
      // Keep place labels above the overlay, same convention as the data layers.
      let beforeId: string | undefined
      try { if (map.getLayer('address_label')) beforeId = 'address_label' } catch {}
      map.addLayer({
        id: GMBA_FILL_ID,
        type: 'fill',
        source: GMBA_SOURCE_ID,
        paint: {
          'fill-color': ACCENT,
          // Near-invisible base fill keeps the whole polygon hoverable; the
          // hovered range gets a visible tint.
          'fill-opacity': [
            'case', ['boolean', ['feature-state', 'hover'], false], 0.14, 0.02,
          ],
        },
      }, beforeId)
      map.addLayer({
        id: GMBA_LINE_ID,
        type: 'line',
        source: GMBA_SOURCE_ID,
        paint: {
          'line-color': [
            'case', ['boolean', ['feature-state', 'hover'], false],
            ACCENT, 'rgba(208,208,208,0.6)',
          ],
          'line-width': [
            'case', ['boolean', ['feature-state', 'hover'], false], 2, 0.8,
          ],
        },
      }, beforeId)

      map.on('mousemove', GMBA_FILL_ID, (e) => {
        const f = e.features?.[0]
        if (!f || f.id === undefined) return
        if (hoveredGmbaIdRef.current !== null && hoveredGmbaIdRef.current !== f.id) {
          map.setFeatureState(
            { source: GMBA_SOURCE_ID, id: hoveredGmbaIdRef.current },
            { hover: false }
          )
        }
        hoveredGmbaIdRef.current = f.id
        map.setFeatureState({ source: GMBA_SOURCE_ID, id: f.id }, { hover: true })
        const p = f.properties ?? {}
        setGmbaHover({
          name: p.name ?? '—',
          feature: p.feature ?? '',
          countries: p.countries ?? '',
          areaKm2: typeof p.area_km2 === 'number' ? p.area_km2 : null,
          elevLow: typeof p.elev_low === 'number' ? p.elev_low : null,
          elevHigh: typeof p.elev_high === 'number' ? p.elev_high : null,
        })
      })
      map.on('mouseleave', GMBA_FILL_ID, () => {
        if (hoveredGmbaIdRef.current !== null) {
          map.setFeatureState(
            { source: GMBA_SOURCE_ID, id: hoveredGmbaIdRef.current },
            { hover: false }
          )
          hoveredGmbaIdRef.current = null
        }
        setGmbaHover(null)
      })
    }

    for (const id of [GMBA_FILL_ID, GMBA_LINE_ID]) {
      if (map.getLayer(id)) {
        map.setLayoutProperty(id, 'visibility', gmbaOn ? 'visible' : 'none')
      }
    }
    if (!gmbaOn) {
      if (hoveredGmbaIdRef.current !== null && map.getSource(GMBA_SOURCE_ID)) {
        map.setFeatureState(
          { source: GMBA_SOURCE_ID, id: hoveredGmbaIdRef.current },
          { hover: false }
        )
        hoveredGmbaIdRef.current = null
      }
      setGmbaHover(null)
    }
  }, [gmbaOn, isMapLoaded, setGmbaHover])

  // Query all variables at the currently clicked point. Always reads from the
  // active version's level-0 (finest) zarr arrays so values are
  // zoom-independent, exact, and consistent across variables. Yearly variables
  // read the selected water year; composites are year-less.
  const requeryAllVariables = (cancelled: { val: boolean }) => {
    const coords = lastClickRef.current
    if (!coords) return
    setClickInfo({ lng: coords.lng, lat: coords.lat, status: 'querying', values: EMPTY_VALUES, snowClass: null })

    const runQuery = async () => {
      // Wait for the level-0 arrays to finish opening (metadata only, < 1 s)
      if (l0ArraysPromiseRef.current) await l0ArraysPromiseRef.current
      if (cancelled.val) return

      const state = useStore.getState()
      const waterYearIdx = state.waterYearIndex
      const grid =
        (await (gridPromisesRef.current[state.version] ?? Promise.resolve(null))) ??
        gridSpecsRef.current[state.version] ??
        null
      if (cancelled.val) return
      const rowCol = grid ? latlonToRowCol(coords.lat, coords.lng, grid) : null

      // Sturm & Liston class at the point (own 300 m grid, own store).
      // Codes 8 (Ocean) / 9 (Fill) are non-classes -> null.
      const querySnowClass = async (): Promise<number | null> => {
        const arr = snowClassArrayRef.current
        const scRowCol = latlonToSnowClassRowCol(coords.lat, coords.lng)
        if (!arr || !scRowCol) return null
        try {
          const result = await get(arr, scRowCol)
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const raw = (result as any)?.data?.[0] ?? result
          if (typeof raw !== 'number' || raw < 1 || raw > 7) return null
          return raw
        } catch {
          return null
        }
      }

      const snowClassPromise = querySnowClass()
      const entries = await Promise.all(
        ALL_VARIABLES.map(async (varName): Promise<[Variable, number | null]> => {
          const arr = l0ArraysRef.current[varName]
          if (arr && rowCol) {
            try {
              const [row, col] = rowCol
              const sel = VARIABLE_CONFIGS[varName].hasWaterYear
                ? [waterYearIdx, row, col]
                : [row, col]
              const result = await get(arr, sel)
              // zarrita returns a scalar or a 0-d ndarray; unwrap either form
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const raw = (result as any)?.data?.[0] ?? result
              if (!isValidValue(raw)) return [varName, null]
              // decode CF scale_factor (raw is the stored int16)
              return [varName, (raw as number) * VARIABLE_CONFIGS[varName].scale]
            } catch {
              return [varName, null]
            }
          }
          return [varName, null]
        })
      )

      const snowClass = await snowClassPromise

      if (cancelled.val) return
      const values = Object.fromEntries(entries) as ClickInfo['values']
      setClickInfo({ lng: coords.lng, lat: coords.lat, status: 'done', values, snowClass })
    }

    runQuery()
  }

  // Create all five ZarrLayers for the active version when the map loads (and
  // again on every version switch — the cleanup below removes the old
  // version's layers first); attach a single click handler.
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    const cancelled = { val: false }
    const zarrUrl = VERSION_CONFIGS[version].zarrUrl

    const createLayers = async () => {
      // Only reference the mask variable when it exists in the pyramid — a
      // layer with a missing aux array fails its level loads and draws nothing.
      const maskAvailable = (await seasonalMaskProbeRef.current) ?? false
      if (cancelled.val) return
      const state = useStore.getState()

      let beforeIdBase: string | undefined
      try { if (map.getLayer('address_label')) beforeIdBase = 'address_label' } catch {}

      // Snow-class basemap first, so the data layers added below land on top
      // of it (maplibre inserts each at the same beforeId, in call order).
      const snowClassLayer = new ZarrLayer({
        id: SNOW_CLASS_LAYER_ID,
        source: SNOW_CLASS_URL,
        variable: SNOW_CLASS_VARIABLE,
        // clim/colormap are unused by the categorical shader but required by
        // the constructor; the class palette doubles as the colormap array.
        clim: [1, 7],
        colormap: Object.values(SNOW_CLASS_INFO).map((c) => c.color),
        opacity: state.basemap === 'snowclass' ? 1 : 0,
        zarrVersion: 3,
        fillValue: SNOW_CLASS_FILL,
        customFrag: SNOW_CLASS_FRAG,
        maxzoom: state.basemap === 'snowclass' ? Infinity : -1,
      })
      map.addLayer(snowClassLayer, beforeIdBase)
      snowClassLayerRef.current = snowClassLayer

      ALL_VARIABLES.forEach((varName) => {
      const isActive = varName === state.variable
      const vCfg = VARIABLE_CONFIGS[varName]
      // The band value in customFrag is ALREADY CF-decoded: zarr-layer >= 0.8
      // reads scale_factor/add_offset from the array attrs and applies them in
      // the band alias (verified in dist: `<band> = raw * u_scaleFactor + ...`),
      // so clim stays in physical units with no scaling here. Two discard arms,
      // both deliberate:
      //   x!=x       portable NaN test (zarr-layer masks the *Zarr* fill_value;
      //              this catches anything it converted to NaN).
      //   x < -100.0 catches fill that reaches the shader unmasked (raw -9999,
      //              or -999.9 after a 0.1 scale) — insurance against a store
      //              written without the explicit zarr-level fill_value (the
      //              bug class that bit the v10 icechunk store init).
      //
      // Seasonal-snow arm (issue #9): seasonal_snow_pct is an aux band (fork
      // auxVariables), CF-decoded to 0-100 with NaN fill. `< threshold` is
      // false for NaN, so mask-fill pixels (ocean/off-grid, plus 300 m
      // coastline mismatch slivers) are NEVER hidden by the toggle — the main
      // variable's own fill discard already handles pixels without data.
      const seasonalArm = maskAvailable
        ? `if (u_seasonal_only > 0.5 && ${SEASONAL_MASK_VARIABLE} < u_seasonal_threshold) { discard; }\n        `
        : ''
      const customFrag = `
        ${seasonalArm}if (${varName} != ${varName} || ${varName} < -100.0) { discard; }
        float rescaled = (${varName} - clim.x) / (clim.y - clim.x);
        vec4 c = texture(colormap, vec2(rescaled, 0.5));
        fragColor = vec4(c.rgb, opacity);
        fragColor.rgb *= fragColor.a;
      `
      const options: ZarrLayerOptions = {
        id: `zarr-${varName}`,
        source: zarrUrl,
        variable: varName,
        clim: isActive ? state.clim : vCfg.clim,
        colormap: COLORMAP_ARRAYS[vCfg.colormap],
        opacity: isActive ? state.opacity : 0,
        // CRS/extent/orientation come from the store's proj:/spatial: attrs
        // (zarr-layer >= 0.8 self-describing stores) — no georeferencing here.
        ...(vCfg.hasWaterYear
          ? { selector: { water_year: { selected: state.waterYearIndex, type: 'index' as const } } }
          : {}),
        ...(maskAvailable
          ? {
              auxVariables: [SEASONAL_MASK_VARIABLE],
              uniforms: {
                u_seasonal_only: state.seasonalOnly ? 1 : 0,
                u_seasonal_threshold: SEASONAL_MASK_THRESHOLD,
              },
            }
          : {}),
        zarrVersion: 3,
        fillValue: FILL_VALUE,
        onLoadingStateChange: (ls) => {
          if (useStore.getState().variable === varName) setLoadingState(ls)
        },
        customFrag,
        // Inactive layers are suppressed from the render/chunk-loading loop via
        // maxzoom:-1 so only the active variable fetches tile data.
        maxzoom: isActive ? Infinity : -1,
      }
      if (!cancelled.val) {
        const layer = new ZarrLayer(options)
        map.addLayer(layer, beforeIdBase)
        zarrLayersRef.current[varName] = layer
      }
      })
    }
    createLayers()

    const clickHandler = (event: maplibregl.MapMouseEvent) => {
      const { lng, lat } = event.lngLat
      lastClickRef.current = { lng, lat }
      markerRef.current?.remove()
      markerRef.current = new maplibregl.Marker({ element: createPulsingMarkerElement(), anchor: 'center' }).setLngLat([lng, lat]).addTo(map)
      requeryAllVariables({ val: false })
    }
    map.on('click', clickHandler)

    return () => {
      cancelled.val = true
      map.off('click', clickHandler)
      ALL_VARIABLES.forEach((varName) => {
        try { if (map.getLayer(`zarr-${varName}`)) map.removeLayer(`zarr-${varName}`) } catch {}
      })
      try { if (map.getLayer(SNOW_CLASS_LAYER_ID)) map.removeLayer(SNOW_CLASS_LAYER_ID) } catch {}
      zarrLayersRef.current = {}
      snowClassLayerRef.current = null
    }
  }, [isMapLoaded, version, setLoadingState, setClickInfo]) // eslint-disable-line react-hooks/exhaustive-deps

  // Switch which layer is visible when variable changes; re-query selected point
  useEffect(() => {
    if (!isMapLoaded) return
    ALL_VARIABLES.forEach((v) => {
      const layer = zarrLayersRef.current[v]
      if (!layer) return
      const active = v === variable
      // Toggle chunk-loading: inactive layers sit at maxzoom:-1 so prerender
      // never calls mode.update(). Active layer uses Infinity (library default).
      ;(layer as any).maxZoom = active ? Infinity : -1
      layer.setOpacity(active ? opacity : 0)
    })
    if (lastClickRef.current) requeryAllVariables({ val: false })
  }, [variable, isMapLoaded]) // eslint-disable-line react-hooks/exhaustive-deps

  // Opacity + clim + colormap updates for the active layer
  useEffect(() => {
    const layer = zarrLayersRef.current[variable]
    if (!layer) return
    layer.setOpacity(opacity)
    layer.setClim(clim)
    layer.setColormap(colormapArray)
  }, [variable, opacity, clim, colormapArray])  // colormapArray changes with variable

  // Water year selector update on the yearly layers + re-query selected point
  useEffect(() => {
    ALL_VARIABLES.forEach((v) => {
      if (!VARIABLE_CONFIGS[v].hasWaterYear) return
      zarrLayersRef.current[v]?.setSelector({ water_year: { selected: waterYearIndex, type: 'index' } })
    })
    if (lastClickRef.current) requeryAllVariables({ val: false })
  }, [waterYearIndex]) // eslint-disable-line react-hooks/exhaustive-deps

  // Seasonal-snow display filter: a pure uniform flip on every layer — no
  // layer rebuilds, no refetches (the mask band is already resident). No-op
  // (with a console warning from zarr-layer) if the mask probe failed, but the
  // sidebar disables the toggle in that case anyway.
  const seasonalOnly = useStore((s) => s.seasonalOnly)
  useEffect(() => {
    ALL_VARIABLES.forEach((v) => {
      zarrLayersRef.current[v]?.setUniforms({ u_seasonal_only: seasonalOnly ? 1 : 0 })
    })
  }, [seasonalOnly])

  // Resize map when sidebar width changes
  useEffect(() => {
    if (mapRef.current) mapRef.current.resize()
  }, [sidebarWidth])

  return (
    <>
      <Box
        ref={mapContainer}
        sx={{
          position: 'absolute',
          top: 0,
          right: 0,
          bottom: ['50vh', '50vh', 0],
          left: 0,
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          top: ['56px', '56px', '8px'],
          left: (sidebarWidth ?? 0) + 10,
          pointerEvents: 'none',
        }}
      >
        {loadingState.loading && <Spinner size={40} />}
      </Box>
    </>
  )
}
