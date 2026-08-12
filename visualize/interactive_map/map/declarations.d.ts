declare module '@carbonplan/colormaps' {
  type ColormapFormat = 'hex' | 'rgb'
  type ColormapOptions = { format?: ColormapFormat; count?: number }
  export function makeColormap(name: string, options?: ColormapOptions): string[]
  export function useThemedColormap(name: string, options?: ColormapOptions): string[]
}

declare module '@carbonplan/components' {
  import type React from 'react'
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Filter: React.FC<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Slider: React.FC<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Row: React.FC<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Column: React.FC<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Colorbar: React.FC<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Input: React.FC<any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export const Spinner: React.FC<any>
}

declare module '@carbonplan/theme' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const theme: any
  export default theme
}

declare module '@protomaps/basemaps' {
  type Flavor = Record<string, unknown>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export function layers(source: string, theme: Flavor, opts?: { lang?: string }): any[]
  export function namedFlavor(name: string): Flavor
}
