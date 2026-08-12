import { create } from 'zustand'
import type { LoadingState } from '@carbonplan/zarr-layer'

export type Variable =
  | 'runoff_onset'
  | 'runoff_onset_median'
  | 'runoff_onset_mad'
  | 'temporal_resolution'
  | 'temporal_resolution_median'

export type Basemap = 'dark' | 'satellite' | 'topography'

// scale: on-disk int16 -> physical units (CF scale_factor; zarr-layer and
// zarrita both hand us raw stored values, so scaling is applied in the
// fragment shader and in point-query formatting).
// hasWaterYear: variable carries the water_year dimension (year slider applies).
export const VARIABLE_CONFIGS: Record<
  Variable,
  {
    clim: [number, number]
    colormap: string
    label: string
    units: string
    scale: number
    hasWaterYear: boolean
  }
> = {
  runoff_onset: {
    clim: [110, 270],
    colormap: 'rainbow',
    label: 'Snowmelt Runoff Onset',
    units: 'day of water year',
    scale: 1,
    hasWaterYear: true,
  },
  runoff_onset_median: {
    clim: [110, 270],
    colormap: 'rainbow',
    label: 'Median Snowmelt Runoff Onset (all water years)',
    units: 'day of water year',
    scale: 1,
    hasWaterYear: false,
  },
  runoff_onset_mad: {
    clim: [0, 30],
    colormap: 'fire',
    label: 'Runoff Onset Median Absolute Deviation (all water years)',
    units: 'days',
    scale: 0.1,
    hasWaterYear: false,
  },
  temporal_resolution: {
    clim: [0, 24],
    colormap: 'water',
    label: 'Temporal Resolution',
    units: 'days',
    scale: 0.1,
    hasWaterYear: true,
  },
  temporal_resolution_median: {
    clim: [0, 24],
    colormap: 'water',
    label: 'Median Temporal Resolution (all water years)',
    units: 'days',
    scale: 0.1,
    hasWaterYear: false,
  },
}

export const ALL_VARIABLES = Object.keys(VARIABLE_CONFIGS) as Variable[]

export const WATER_YEARS = [
  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025,
] as const

// Injected at build time via NEXT_PUBLIC_ZARR_URL; falls back to the current
// production pyramid so local `npm run dev` works without setting the env var.
export const ZARR_URL =
  process.env.NEXT_PUBLIC_ZARR_URL ??
  'https://uwcryo.blob.core.windows.net/snowmelt/snowmelt_runoff_onset/global_runoff_onset_v10_multiscale_1'

export type ClickInfo = {
  lng: number
  lat: number
  status: 'querying' | 'done'
  values: Record<Variable, number | null>
}

interface AppState {
  variable: Variable
  waterYearIndex: number
  opacity: number
  clim: [number, number]
  colormap: string
  globeProjection: boolean
  loadingState: LoadingState
  sidebarWidth: number | null
  basemap: Basemap
  clickInfo: ClickInfo | null
  zoomLevel: number
  setVariable: (v: Variable) => void
  setWaterYearIndex: (i: number) => void
  setOpacity: (o: number) => void
  setClim: (c: [number, number]) => void
  setColormap: (c: string) => void
  setGlobeProjection: (g: boolean) => void
  setLoadingState: (s: LoadingState) => void
  setSidebarWidth: (w: number | null) => void
  setBasemap: (b: Basemap) => void
  setClickInfo: (info: ClickInfo | null) => void
  setZoomLevel: (z: number) => void
}

export const useStore = create<AppState>((set) => ({
  variable: 'runoff_onset_median',
  waterYearIndex: WATER_YEARS.length - 1,
  opacity: 1,
  clim: VARIABLE_CONFIGS.runoff_onset_median.clim,
  colormap: VARIABLE_CONFIGS.runoff_onset_median.colormap,
  globeProjection: true,
  loadingState: { loading: false, metadata: false, chunks: false },
  sidebarWidth: null,
  basemap: 'dark',
  clickInfo: null,
  zoomLevel: 2.4,
  setVariable: (variable) =>
    set({
      variable,
      clim: VARIABLE_CONFIGS[variable].clim,
      colormap: VARIABLE_CONFIGS[variable].colormap,
    }),
  setWaterYearIndex: (waterYearIndex) => set({ waterYearIndex }),
  setOpacity: (opacity) => set({ opacity }),
  setClim: (clim) => set({ clim }),
  setColormap: (colormap) => set({ colormap }),
  setGlobeProjection: (globeProjection) => set({ globeProjection }),
  setLoadingState: (loadingState) => set({ loadingState }),
  setSidebarWidth: (sidebarWidth) => set({ sidebarWidth }),
  setBasemap: (basemap) => set({ basemap }),
  setClickInfo: (clickInfo) => set({ clickInfo }),
  setZoomLevel: (zoomLevel) => set({ zoomLevel }),
}))
