import { create } from 'zustand'
import type { LoadingState } from '@carbonplan/zarr-layer'

export type Variable =
  | 'runoff_onset'
  | 'runoff_onset_median'
  | 'runoff_onset_mad'
  | 'temporal_resolution'
  | 'temporal_resolution_median'

export type Basemap = 'dark' | 'satellite' | 'topography'

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
    }),
  setWaterYearIndex: (waterYearIndex) => set({ waterYearIndex }),
  setOpacity: (opacity) => set({ opacity }),
  setClim: (clim) => set({ clim }),
  setGlobeProjection: (globeProjection) => set({ globeProjection }),
  setLoadingState: (loadingState) => set({ loadingState }),
  setSidebarWidth: (sidebarWidth) => set({ sidebarWidth }),
  setBasemap: (basemap) => set({ basemap }),
  setClickInfo: (clickInfo) => set({ clickInfo }),
  setZoomLevel: (zoomLevel) => set({ zoomLevel }),
}))
