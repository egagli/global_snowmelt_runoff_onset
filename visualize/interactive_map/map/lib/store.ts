import { create } from 'zustand'
import { track, trackDebounced } from './analytics'
import type { LoadingState } from '@carbonplan/zarr-layer'

export type Variable =
  | 'runoff_onset'
  | 'runoff_onset_median'
  | 'runoff_onset_mad'
  | 'temporal_resolution'
  | 'temporal_resolution_median'

export type Basemap = 'dark' | 'satellite' | 'topography' | 'snowclass'

import type { ColormapName } from './colormaps'

// scale: on-disk int16 -> physical units (CF scale_factor; zarr-layer and
// zarrita both hand us raw stored values, so scaling is applied in the
// fragment shader and in point-query formatting).
// hasWaterYear: variable carries the water_year dimension (year slider applies).
// Colormaps are FIXED per variable (matplotlib ramps matching the manuscript
// figures); the user adjusts vmin/vmax/opacity only.
export const VARIABLE_CONFIGS: Record<
  Variable,
  {
    clim: [number, number]
    colormap: ColormapName
    label: string
    units: string
    scale: number
    hasWaterYear: boolean
  }
> = {
  runoff_onset: {
    clim: [110, 270],
    colormap: 'viridis',
    label: 'runoff onset',
    units: 'day of water year',
    scale: 1,
    hasWaterYear: true,
  },
  runoff_onset_median: {
    clim: [110, 270],
    colormap: 'viridis',
    label: 'runoff onset median',
    units: 'day of water year',
    scale: 1,
    hasWaterYear: false,
  },
  runoff_onset_mad: {
    clim: [0, 30],
    colormap: 'Reds',
    label: 'median absolute deviation',
    units: 'days',
    scale: 0.1,
    hasWaterYear: false,
  },
  temporal_resolution: {
    clim: [2, 14],
    colormap: 'YlGn_r',
    label: 'temporal resolution',
    units: 'days',
    scale: 0.1,
    hasWaterYear: true,
  },
  temporal_resolution_median: {
    clim: [2, 14],
    colormap: 'YlGn_r',
    label: 'temporal resolution median',
    units: 'days',
    scale: 0.1,
    hasWaterYear: false,
  },
}

export const ALL_VARIABLES = Object.keys(VARIABLE_CONFIGS) as Variable[]

// Selector columns + point-query order: composites first, then annual.
export const COMPOSITE_VARIABLES: Variable[] = [
  'runoff_onset_median', 'runoff_onset_mad', 'temporal_resolution_median',
]
export const ANNUAL_VARIABLES: Variable[] = [
  'runoff_onset', 'temporal_resolution',
]

// ---------------------------------------------------------------------------
// Dataset versions (issue #13)
// ---------------------------------------------------------------------------

export type DatasetVersion = 'v10' | 'v9'

// Per-version pyramid URL + water-year coverage. The map probes each
// pyramid's root zarr.json at startup: versions whose pyramid does not exist
// (yet) render disabled in the sidebar dropdown, so this list can safely name
// versions ahead of their pyramid build. Grid/affine metadata is NOT
// hardcoded per version — the probe parses spatial:transform/spatial:shape
// from the store itself (fetchVersionGrid in map.tsx), so publishing a v9
// pyramid at the URL below lights the option up with no code change.
export const VERSION_CONFIGS: Record<
  DatasetVersion,
  { label: string; note: string; zarrUrl: string; waterYears: number[] }
> = {
  v10: {
    label: 'v10 — WY2015–2025 (latest)',
    note: 'Icechunk rebuild: 11 water years, grid extended to 84.05°N–63.41°S.',
    // Injected at build time via NEXT_PUBLIC_ZARR_URL (deploy_map.yml derives
    // it from the config); falls back to the current production pyramid so
    // local `npm run dev` works without setting the env var.
    zarrUrl:
      process.env.NEXT_PUBLIC_ZARR_URL ??
      'https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/global_runoff_onset_v10_multiscale_1',
    waterYears: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
  },
  v9: {
    label: 'v9 — WY2015–2024 (manuscript)',
    note: 'The dataset version described in Gagliano et al. (2026).',
    // No v9 pyramid has been published yet — the dropdown entry stays
    // disabled until one exists at this URL (probe-driven, see above).
    zarrUrl:
      process.env.NEXT_PUBLIC_ZARR_URL_V9 ??
      'https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/global_runoff_onset_v9_multiscale_1',
    waterYears: [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
  },
}

export const ALL_VERSIONS: DatasetVersion[] = ['v10', 'v9']
export const DEFAULT_VERSION: DatasetVersion = 'v10'

// Display-side seasonal-snow mask (issue #9): a sixth pyramid variable holding
// percent seasonal-snow area per cell (0-100, int16, -9999 fill), reclassified
// from Sturm & Liston (2021). Sampled as an aux band by every layer's shader;
// the sidebar toggle discards pixels below SEASONAL_MASK_THRESHOLD when on.
// The map probes for it at startup, so deploying this map before the mask job
// has run just leaves the toggle disabled.
export const SEASONAL_MASK_VARIABLE = 'seasonal_snow_pct'
// Percent-seasonal-snow a cell must reach to survive the filter. Only bites at
// coarse levels: level 0 is exactly 0 or 100, so any 0 < threshold <= 100
// behaves identically at native zoom. 50 = "at least half this cell's area is
// seasonal snow" (ocean counts as 0, so mixed land/water coastal cells can
// drop below it at coarse zooms). Drop toward 0.5 for the permissive "any
// seasonal snow here".
export const SEASONAL_MASK_THRESHOLD = 50

// Standalone 300 m Sturm & Liston (2021) classification pyramid (NSIDC-0768),
// published by visualize/interactive_map/build_snow_class_store.py. Serves two
// consumers: the snow-class row of the point-query card (reads level 0) and the
// "snow class" basemap (a categorical zarr-layer, which needs the coarse
// levels — a single-level 300 m array would read every chunk at world zoom).
export const SNOW_CLASS_URL =
  process.env.NEXT_PUBLIC_SNOW_CLASS_URL ??
  'https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/snow_classification_300m_multiscale_1'

export const SNOW_CLASS_VARIABLE = 'snow_class'

// Class code → name/color (easysnowdata convention; NSIDC-0768 user guide).
// 8 (Ocean) and 9 (Fill) are non-classes for display purposes.
export const SNOW_CLASS_INFO: Record<number, { name: string; color: string }> =
  {
    1: { name: 'Tundra', color: '#a100c8' },
    2: { name: 'Boreal Forest', color: '#00a0fe' },
    3: { name: 'Maritime', color: '#fe0000' },
    4: { name: 'Ephemeral', color: '#e7dc32' },
    5: { name: 'Prairie', color: '#f08328' },
    6: { name: 'Montane Forest', color: '#00dc00' },
    7: { name: 'Ice', color: '#aaaaaa' },
  }

// ---------------------------------------------------------------------------
// GMBA mountain inventory overlay (issue #13)
// ---------------------------------------------------------------------------

// GMBA Mountain Inventory v2.0 standard 300-selection (290 major ranges;
// Snethlage et al. 2022, doi:10.1038/s41597-022-01256-y), published by
// prepare_gmba_overlay.py as two blobs (issue #36):
//  - full-resolution outlines as vector tiles (PMTiles, z0-z10, range-read via
//    the pmtiles:// protocol), so outlines sharpen as you zoom in while the
//    world view stays ~0.3 MB. Features carry only id + name.
//  - a ~20 KB (gzip) index with everything else, one entry per range: feeds
//    the range card (looked up by the clicked feature's id) and the search box.
// Both are lazy-loaded (tiles when the overlay is first enabled; the index
// then or on first search). Immutable-cache convention: bump the _2 suffix
// here and the script's --suffix together.
export const GMBA_PMTILES_URL =
  process.env.NEXT_PUBLIC_GMBA_PMTILES_URL ??
  'https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/gmba_v2_standard_300_2.pmtiles'
export const GMBA_INDEX_URL =
  process.env.NEXT_PUBLIC_GMBA_INDEX_URL ??
  'https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/gmba_v2_standard_300_index_2.json'

// One index entry (write_index in prepare_gmba_overlay.py).
export type GmbaInfo = {
  id: number
  name: string
  path: string // 'Eurasia > Europe > Alps > Western Alps', ending at this range
  countries: string
  feature: string
  level: number | null
  area_km2: number | null
  elev_low: number | null
  elev_high: number | null
  wikidata: string // Q-id, '' when GMBA has none
  bbox: [number, number, number, number] // east may exceed 180 (antimeridian)
  center: [number, number] // label point, always inside the range
}

let gmbaIndexPromise: Promise<GmbaInfo[] | null> | null = null

/** Fetch the GMBA index once (memoized) and put it in the store. */
export function loadGmbaIndex(): Promise<GmbaInfo[] | null> {
  if (!gmbaIndexPromise) {
    gmbaIndexPromise = fetch(GMBA_INDEX_URL)
      .then((r) => (r.ok ? r.json() : null))
      .then((entries: GmbaInfo[] | null) => {
        if (entries) useStore.getState().setGmbaIndex(entries)
        return entries
      })
      .catch((e) => {
        console.warn('GMBA index unavailable:', e)
        gmbaIndexPromise = null // allow a retry on the next request
        return null
      })
  }
  return gmbaIndexPromise
}

// A request from outside the map (search box, URL hash) for the map to move
// and/or run a point query. seq makes repeat requests for the same target
// distinct, so the map's effect fires every time.
export type NavTarget =
  | { kind: 'point'; lng: number; lat: number; fly: boolean }
  | { kind: 'bounds'; bbox: [number, number, number, number] }
export type NavRequest = NavTarget & { seq: number }

export type ClickInfo = {
  lng: number
  lat: number
  status: 'querying' | 'done'
  values: Record<Variable, number | null>
  // Sturm & Liston class code at the point (1-7 displayable; null = ocean/
  // fill/off-grid or the class store is unavailable). Independent of the
  // seasonal-only toggle.
  snowClass: number | null
}

interface AppState {
  version: DatasetVersion
  // null = probe still in flight; false = pyramid missing (option disabled).
  versionAvailable: Record<DatasetVersion, boolean | null>
  variable: Variable
  waterYearIndex: number
  opacity: number
  clim: [number, number]
  globeProjection: boolean
  // Display-side filter: hide pixels outside Sturm & Liston seasonal snow
  // classes (default ON since issue #13; the sidebar copy explains what the
  // filter hides, and the point-query card ignores it).
  seasonalOnly: boolean
  // Whether the pyramid actually has the mask variable (probed at startup).
  seasonalMaskAvailable: boolean
  // GMBA mountain-range outlines overlay (issue #13; default off).
  gmbaOn: boolean
  // Lazy-loaded GMBA index (null until fetched) and the pinned range's id —
  // set by clicking a range or picking one in search; the card reads the
  // index entry, so an id can be pinned before the index has arrived.
  gmbaIndex: GmbaInfo[] | null
  gmbaPinnedId: number | null
  navRequest: NavRequest | null
  // Region-specific caution toast (issue #13; same pattern as the
  // MODIS_snow_phenology map). Set from WARNING_ZONES on map move.
  activeWarning: { name: string; message: string } | null
  loadingState: LoadingState
  sidebarWidth: number | null
  basemap: Basemap
  clickInfo: ClickInfo | null
  zoomLevel: number
  setVersion: (v: DatasetVersion) => void
  setVersionAvailable: (v: DatasetVersion, ok: boolean) => void
  setVariable: (v: Variable) => void
  setWaterYearIndex: (i: number) => void
  setOpacity: (o: number) => void
  setClim: (c: [number, number]) => void
  setGlobeProjection: (g: boolean) => void
  setSeasonalOnly: (s: boolean) => void
  setSeasonalMaskAvailable: (a: boolean) => void
  setGmbaOn: (g: boolean) => void
  setGmbaIndex: (entries: GmbaInfo[]) => void
  setGmbaPinnedId: (id: number | null) => void
  requestNav: (r: NavTarget) => void
  setActiveWarning: (w: { name: string; message: string } | null) => void
  setLoadingState: (s: LoadingState) => void
  setSidebarWidth: (w: number | null) => void
  setBasemap: (b: Basemap) => void
  setClickInfo: (info: ClickInfo | null) => void
  setZoomLevel: (z: number) => void
}

export const useStore = create<AppState>((set) => ({
  version: DEFAULT_VERSION,
  versionAvailable: { v10: null, v9: null },
  variable: 'runoff_onset_median',
  waterYearIndex: VERSION_CONFIGS[DEFAULT_VERSION].waterYears.length - 1,
  opacity: 1,
  clim: VARIABLE_CONFIGS.runoff_onset_median.clim,
  globeProjection: true,
  seasonalOnly: true,
  seasonalMaskAvailable: false,
  gmbaOn: false,
  gmbaIndex: null,
  gmbaPinnedId: null,
  navRequest: null,
  activeWarning: null,
  loadingState: { loading: false, metadata: false, chunks: false },
  sidebarWidth: null,
  basemap: 'dark',
  clickInfo: null,
  zoomLevel: 2.4,
  // Version switch keeps the same water YEAR when the target version has it
  // (v9 lacks 2025), else snaps to the target's latest year.
  setVersion: (version) =>
    set((state) => {
      if (version !== state.version) track('map_version_change', { version })
      const prevYear =
        VERSION_CONFIGS[state.version].waterYears[state.waterYearIndex]
      const years = VERSION_CONFIGS[version].waterYears
      const idx = years.indexOf(prevYear)
      return {
        version,
        waterYearIndex: idx >= 0 ? idx : years.length - 1,
      }
    }),
  setVersionAvailable: (v, ok) =>
    set((state) => ({
      versionAvailable: { ...state.versionAvailable, [v]: ok },
    })),
  setVariable: (variable) =>
    set((state) => {
      if (variable !== state.variable) track('map_layer_change', { layer: variable })
      return { variable, clim: VARIABLE_CONFIGS[variable].clim }
    }),
  setWaterYearIndex: (waterYearIndex) =>
    set((state) => {
      trackDebounced('map_time_change', {
        water_year: VERSION_CONFIGS[state.version].waterYears[waterYearIndex],
      })
      return { waterYearIndex }
    }),
  setOpacity: (opacity) => set({ opacity }),
  setClim: (clim) => set({ clim }),
  setGlobeProjection: (globeProjection) =>
    set((state) => {
      if (globeProjection !== state.globeProjection)
        track('map_projection_change', { projection: globeProjection ? 'globe' : 'mercator' })
      return { globeProjection }
    }),
  setSeasonalOnly: (seasonalOnly) => {
    track('map_filter_toggle', { seasonal_only: seasonalOnly })
    set({ seasonalOnly })
  },
  setSeasonalMaskAvailable: (seasonalMaskAvailable) =>
    set({ seasonalMaskAvailable }),
  // track on change only: search re-enables the overlay on every range pick
  setGmbaOn: (gmbaOn) =>
    set((state) => {
      if (gmbaOn !== state.gmbaOn)
        track('map_overlay_toggle', { overlay: 'gmba', visible: gmbaOn })
      return { gmbaOn }
    }),
  setGmbaIndex: (gmbaIndex) => set({ gmbaIndex }),
  setGmbaPinnedId: (gmbaPinnedId) => set({ gmbaPinnedId }),
  requestNav: (r) =>
    set((state) => ({
      navRequest: { ...r, seq: (state.navRequest?.seq ?? 0) + 1 },
    })),
  setActiveWarning: (activeWarning) => set({ activeWarning }),
  setLoadingState: (loadingState) => set({ loadingState }),
  setSidebarWidth: (sidebarWidth) => set({ sidebarWidth }),
  setBasemap: (basemap) =>
    set((state) => {
      if (basemap !== state.basemap) track('map_basemap_change', { basemap })
      return { basemap }
    }),
  setClickInfo: (clickInfo) =>
    set((state) => {
      if (clickInfo?.status === 'querying') track('map_point_query', { layer: state.variable })
      return { clickInfo }
    }),
  setZoomLevel: (zoomLevel) => set({ zoomLevel }),
}))
