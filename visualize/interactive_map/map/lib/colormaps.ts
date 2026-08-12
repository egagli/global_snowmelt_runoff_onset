// Matplotlib colormap ramps (32 stops each), generated from matplotlib so the
// map exactly matches the manuscript figures (visualize/global/). These are
// fixed per variable -- the user adjusts vmin/vmax/opacity, not the colormap.
export type ColormapName = 'viridis' | 'Reds' | 'YlGn_r'

export const COLORMAP_ARRAYS: Record<ColormapName, string[]> = {
  viridis: [
    '#440154', '#460c5f', '#47186a', '#482273', '#462d7c', '#443781', '#414186', '#3d4a89',
    '#39548b', '#355c8c', '#31648d', '#2e6c8e', '#2a758e', '#277c8e', '#24848d', '#228b8d',
    '#1f948b', '#1e9b89', '#1fa386', '#24aa82', '#2eb27c', '#39b976', '#47c06e', '#57c665',
    '#6bcd59', '#7ed24e', '#92d741', '#a7db33', '#bfdf24', '#d4e11a', '#e9e419', '#fde724',
  ],
  Reds: [
    '#fff5f0', '#feefe8', '#feeae0', '#fee5d9', '#fdded0', '#fdd5c3', '#fcccb7', '#fcc2ab',
    '#fcb89d', '#fcad91', '#fca386', '#fc997a', '#fb8d6d', '#fb8363', '#fb7959', '#fb6f4f',
    '#f96345', '#f6573e', '#f34b36', '#f03f2f', '#e83429', '#df2c25', '#d62321', '#cd1a1e',
    '#c2161b', '#b91319', '#af1117', '#a60f15', '#950b13', '#860711', '#76030f', '#67000c',
  ],
  YlGn_r: [
    '#004529', '#004d2c', '#005630', '#005f33', '#016837', '#0a703a', '#12773d', '#1b7e40',
    '#258644', '#2c904b', '#349a51', '#3ba458', '#46ad5f', '#54b466', '#62bb6e', '#70c275',
    '#7fc97b', '#8ccf81', '#9ad486', '#a7da8b', '#b4e091', '#bfe596', '#cae99c', '#d5eea1',
    '#dff2a7', '#e6f5ad', '#eef8b2', '#f5fbb8', '#f8fcc3', '#fafdce', '#fcfed9', '#ffffe5',
  ],
}

// Month starts in day-of-water-year (non-leap), with BOTH hemisphere
// conventions: the NH water year starts Oct 1, the SH water year Apr 1, so
// the same DOWY is offset by exactly six months between hemispheres (the
// month-length drift between the two sequences is <= 2 days -- invisible at
// colorbar scale, so one set of tick positions serves both label rows).
// Mirrors plot_utils.create_month_colorbar(hemisphere='both') in the figures.
export const MONTH_STARTS_DOWY: { month: string; shMonth: string; dowy: number }[] = [
  { month: 'Oct', shMonth: 'Apr', dowy: 1 },
  { month: 'Nov', shMonth: 'May', dowy: 32 },
  { month: 'Dec', shMonth: 'Jun', dowy: 62 },
  { month: 'Jan', shMonth: 'Jul', dowy: 93 },
  { month: 'Feb', shMonth: 'Aug', dowy: 124 },
  { month: 'Mar', shMonth: 'Sep', dowy: 152 },
  { month: 'Apr', shMonth: 'Oct', dowy: 183 },
  { month: 'May', shMonth: 'Nov', dowy: 213 },
  { month: 'Jun', shMonth: 'Dec', dowy: 244 },
  { month: 'Jul', shMonth: 'Jan', dowy: 274 },
  { month: 'Aug', shMonth: 'Feb', dowy: 305 },
  { month: 'Sep', shMonth: 'Mar', dowy: 336 },
]
