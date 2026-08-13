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
  ZARR_URL,
  SNOW_CLASS_URL,
  SEASONAL_MASK_VARIABLE,
  SEASONAL_MASK_THRESHOLD,
  VARIABLE_CONFIGS,
  ALL_VARIABLES,
  type Variable,
  type ClickInfo,
} from '../lib/store'
import { COLORMAP_ARRAYS } from '../lib/colormaps'

const ACCENT = '#1dbd8f'
const FILL_VALUE = -9999

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

// Level-0 spatial constants (from the pyramid root zarr.json spatial:transform,
// pixel registration: x = a*col + c, y = e*row + f give the pixel's top-left
// corner). Used for point queries so they always hit the full-resolution data
// regardless of map zoom.
const L0_X_ORIGIN = -179.99945999946
const L0_Y_ORIGIN = 84.04856404856403
const L0_X_RES = 0.0007200007200083292
const L0_Y_RES = 0.0007200007199941183
const L0_N_ROWS = 204800
const L0_N_COLS = 499998

/** Convert WGS84 (lat, lon in degrees) → level-0 [row, col].
 *  Returns null if the point is outside the grid. */
function latlonToL0RowCol(lat: number, lon: number): [number, number] | null {
  const col = Math.floor((lon - L0_X_ORIGIN) / L0_X_RES)
  const row = Math.floor((L0_Y_ORIGIN - lat) / L0_Y_RES)
  if (row < 0 || row >= L0_N_ROWS || col < 0 || col >= L0_N_COLS) return null
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

const OWN_LAYER_IDS = new Set(['zarr-layer', 'esri-imagery', 'topo'])

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

/** Probe whether the pyramid carries the seasonal-snow mask variable. The map
 *  can deploy before the mask job has written it — the toggle stays disabled
 *  and no layer references the missing array (which would fail level loads). */
async function probeSeasonalMask(): Promise<boolean> {
  try {
    const res = await fetch(`${ZARR_URL}/0/${SEASONAL_MASK_VARIABLE}/zarr.json`)
    return res.ok
  } catch {
    return false
  }
}

export const Map = () => {
  const mapContainer = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const zarrLayersRef = useRef<Partial<Record<Variable, InstanceType<typeof ZarrLayer>>>>({})
  const markerRef = useRef<maplibregl.Marker | null>(null)
  const lastClickRef = useRef<{ lng: number; lat: number } | null>(null)
  // Level-0 zarrita arrays opened once at mount — used for zoom-independent point queries
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const l0ArraysRef = useRef<Partial<Record<Variable, any>>>({})
  // Promise that resolves when all arrays are open — awaited in requeryAllVariables
  // so every query round reads from a consistent data source.
  const l0ArraysPromiseRef = useRef<Promise<void> | null>(null)
  // The standalone 300 m snow-classification array (query card class row).
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const snowClassArrayRef = useRef<any>(null)
  // Resolves to whether the pyramid has the seasonal-snow mask — awaited by
  // the layer-creation effect so the shader only samples an existing array.
  const seasonalMaskProbeRef = useRef<Promise<boolean> | null>(null)

  const setSeasonalMaskAvailable = useStore((s) => s.setSeasonalMaskAvailable)

  // Open level-0 arrays for all variables at mount (runs once). Each variable
  // gets its own FetchStore pointing at its sub-URL so zarrita opens the array
  // at the correct path.
  useEffect(() => {
    l0ArraysPromiseRef.current = Promise.all([
      ...ALL_VARIABLES.map(async (varName) => {
        try {
          const arrayStore = new FetchStore(`${ZARR_URL}/0/${varName}`)
          const arr = await open(arrayStore, { kind: 'array' })
          l0ArraysRef.current[varName] = arr
        } catch (e) {
          console.warn(`Failed to open level-0 array for ${varName}:`, e)
        }
      }),
      (async () => {
        try {
          const arrayStore = new FetchStore(`${SNOW_CLASS_URL}/snow_class`)
          snowClassArrayRef.current = await open(arrayStore, { kind: 'array' })
        } catch (e) {
          console.warn('Snow classification store unavailable:', e)
        }
      })(),
    ]).then(() => undefined)

    seasonalMaskProbeRef.current = probeSeasonalMask().then((ok) => {
      setSeasonalMaskAvailable(ok)
      return ok
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const [isMapLoaded, setIsMapLoaded] = useState(false)

  const variable = useStore((s) => s.variable)
  const waterYearIndex = useStore((s) => s.waterYearIndex)
  const opacity = useStore((s) => s.opacity)
  const clim = useStore((s) => s.clim)
  const globeProjection = useStore((s) => s.globeProjection)
  const sidebarWidth = useStore((s) => s.sidebarWidth)
  const loadingState = useStore((s) => s.loadingState)
  const basemap = useStore((s) => s.basemap)
  const setLoadingState = useStore((s) => s.setLoadingState)
  const setClickInfo = useStore((s) => s.setClickInfo)
  const setZoomLevel = useStore((s) => s.setZoomLevel)

  // fixed matplotlib ramp per variable (no user colormap selection)
  const colormapArray = COLORMAP_ARRAYS[VARIABLE_CONFIGS[variable].colormap]

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

  // Zoom display — update on every moveend/zoomend
  useEffect(() => {
    const map = mapRef.current
    if (!map || !isMapLoaded) return
    const check = () => setZoomLevel(map.getZoom())
    map.on('moveend', check)
    map.on('zoomend', check)
    check()
    return () => { map.off('moveend', check); map.off('zoomend', check) }
  }, [isMapLoaded, setZoomLevel])

  // Projection toggle
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    ;(mapRef.current as any).setProjection(
      globeProjection ? { type: 'globe' } : { type: 'mercator' }
    )
  }, [globeProjection, isMapLoaded])

  // Basemap toggle — satellite/topo hide the vector basemap fills to avoid masking
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    map.setLayoutProperty('esri-imagery', 'visibility', basemap === 'satellite' ? 'visible' : 'none')
    map.setLayoutProperty('topo', 'visibility', basemap === 'topography' ? 'visible' : 'none')
    setBasemapFillVisibility(map, basemap === 'dark')
    setBasemapSymbolVisibility(map, basemap !== 'topography')
  }, [basemap, isMapLoaded])

  // Query all variables at the currently clicked point. Always reads from the
  // level-0 (finest) zarr arrays so values are zoom-independent, exact, and
  // consistent across variables. Yearly variables read the selected water
  // year; composites are year-less.
  const requeryAllVariables = (cancelled: { val: boolean }) => {
    const coords = lastClickRef.current
    if (!coords) return
    setClickInfo({ lng: coords.lng, lat: coords.lat, status: 'querying', values: EMPTY_VALUES, snowClass: null })

    const runQuery = async () => {
      // Wait for the level-0 arrays to finish opening (metadata only, < 1 s)
      if (l0ArraysPromiseRef.current) await l0ArraysPromiseRef.current
      if (cancelled.val) return

      const waterYearIdx = useStore.getState().waterYearIndex
      const rowCol = latlonToL0RowCol(coords.lat, coords.lng)

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

  // Create all five ZarrLayers once when map loads; attach single click handler
  useEffect(() => {
    if (!mapRef.current || !isMapLoaded) return
    const map = mapRef.current
    const cancelled = { val: false }

    const createLayers = async () => {
      // Only reference the mask variable when it exists in the pyramid — a
      // layer with a missing aux array fails its level loads and draws nothing.
      const maskAvailable = (await seasonalMaskProbeRef.current) ?? false
      if (cancelled.val) return
      const state = useStore.getState()

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
        source: ZARR_URL,
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
      let beforeId: string | undefined
      try { if (map.getLayer('address_label')) beforeId = 'address_label' } catch {}
      if (!cancelled.val) {
        const layer = new ZarrLayer(options)
        map.addLayer(layer, beforeId)
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
      zarrLayersRef.current = {}
    }
  }, [isMapLoaded, setLoadingState, setClickInfo]) // eslint-disable-line react-hooks/exhaustive-deps

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
