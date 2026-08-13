"""
Shared plotting helpers and the repository's figure conventions.

FIGURE CONVENTIONS
==================
Read this before writing a new figure. Anything below that is expressible as
code lives in this module as a constant or helper, so a figure that imports
from here is consistent by construction rather than by copy-paste.

1. Colormaps and limits are per *variable*, not per figure.
   `VARIABLE_KW` (via `variable_kw()`) is the single source of truth:
   runoff onset viridis 110-270 DOWY, anomaly RdBu +/-30 days, MAD Reds 0-30,
   temporal resolution YlGn_r 2-14 days. Never hand-write a cmap/vmin/vmax for
   these variables -- if a figure needs a different range, pass an override
   (`variable_kw('runoff_onset', vmax=300)`) so the deviation is visible in the
   diff, and change the default here if it becomes the new convention.

2. Runoff onset is a *day of water year*, so it gets a month colorbar.
   `create_month_colorbar()`, not a plain matplotlib colorbar: DOWY 1 = Oct 1
   is meaningless to a reader without month labels. Southern-hemisphere water
   years are offset 6 months, so pass `hemisphere=` deliberately -- 'northern'
   for a northern-hemisphere region, 'both' for a global map.

3. Never bake a water-year count into a label or title.
   Derive it from the data (`len(ds.water_year)`) and interpolate it. The
   dataset grows by a water year annually; a hardcoded "10-year" silently
   becomes wrong. For the same reason no default label in this module contains
   a year count.

4. Maps are projected -- UTM or an equal-area CRS, never plotted in raw
   EPSG:4326 degrees. Lat/lon axes distort area and shape at the latitudes
   this dataset cares about, and make a scalebar meaningless. Regional figures
   use the region's UTM zone (`gdf.estimate_utm_crs()`), which keeps linear
   scale error under ~1% out to roughly +/-800 km from the central meridian --
   fine for a region and its immediate surroundings. Past that (a continental
   context inset, a multi-region map) switch the whole figure to a
   region-centred equal-area CRS, e.g.
   `'+proj=laea +lat_0=<clat> +lon_0=<clon> +datum=WGS84 +units=m'`.
   Global figures are the exception: they use cartopy projections (Robinson,
   NorthPolarStereo) and plot the native 4326 grid with `transform=PlateCarree`,
   letting cartopy handle the antimeridian wrap.

5. A projected map is drawn at equal aspect -- always.
   `ax.set_aspect('equal')`, which `style_map_axes()` does for you. A UTM or
   equal-area CRS only preserves shape and area if one metre of easting is
   drawn the same length as one metre of northing; stretching the axes to fill
   its slot throws that away and silently misrepresents the geometry the
   projection was chosen for. It also invalidates any scalebar in one
   direction, which is why `ScaleBar` warns when the aspect isn't 1. The
   consequence is that a map axes will not fill a non-matching slot -- solve
   that by sizing the figure to the data aspect and using
   `layout='compressed'` (convention 9), never by relaxing the aspect.

6. A projected map gets a scalebar; a global/continental one does not.
   `add_scalebar()`. Place it in an empty corner -- away from the context inset
   (upper left by convention) and away from per-panel annotations (water-year
   labels sit lower right by convention). Bars carry no box, so also pick a
   corner whose background contrasts: `color='black'` over hillshade or light
   data, `color='white'` over ocean or deep shadow. On a small-multiples grid
   where every panel shares one extent, one scalebar on the first panel is
   enough.

7. Hillshade underneath, data on top, `zorder` explicit.
   `HILLSHADE_KW` (gray, 0-400, zorder 0) then the data at zorder 1. The
   hillshade is what makes no-data areas read as terrain instead of holes.
   The 0-400 range is tuned for the regional clips of
   `visualize/data/global_hillshade_robinson.tif`; the global figures coarsen
   the same raster far harder and set their own gray limits.

8. Figure outputs are versioned by dataset version.
   Write to `figures/<config.version>/`, mirroring `results/<version>/`, so a
   v9 render is never silently overwritten by a v10 one.

9. Save at dpi=300 (350 for the large global composites) with
   `bbox_inches='tight'`. Grids of fixed-aspect map panels want
   `layout='compressed'` -- it removes the dead space equal-aspect axes
   otherwise leave inside their grid cells, with no wspace/hspace tuning.

10. An inset is a guest on the map. `add_context_inset()` draws over data the
    reader came for, so keep it small (~25% of the parent) and anchored in its
    corner with equal padding from both adjacent edges -- which is what the
    `_LOC_ANCHOR` pinning in that helper is for.
"""
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.patheffects import withStroke


# ─── Per-variable display styling (convention 1) ────────────────────────────
# Single source of truth for cmap/vmin/vmax. Read through variable_kw(), which
# hands back a copy so a caller mutating its dict can't poison other figures.
VARIABLE_KW = {
    'runoff_onset':               dict(cmap='viridis', vmin=110, vmax=270),
    'runoff_onset_median':        dict(cmap='viridis', vmin=110, vmax=270),
    'runoff_onset_anomaly':       dict(cmap='RdBu',    vmin=-30, vmax=30),
    'runoff_onset_mad':           dict(cmap='Reds',    vmin=0,   vmax=30),
    'temporal_resolution':        dict(cmap='YlGn_r',  vmin=2,   vmax=14),
    'temporal_resolution_median': dict(cmap='YlGn_r',  vmin=2,   vmax=14),
}


def variable_kw(name, **overrides):
    """
    Conventional `.plot`/`.plot.imshow` styling for a dataset variable.

    Parameters
    ----------
    name : str
        Data variable name, e.g. `'runoff_onset_median'`. Must be a key of
        `VARIABLE_KW` — an unknown name is a typo, not a request for defaults.
    **overrides
        Keys merged over the convention (`variable_kw('runoff_onset',
        vmax=300)`). Deviating deliberately is fine; deviating silently is not,
        which is why it has to be spelled out at the call site.

    Returns
    -------
    dict
        A fresh dict, safe to mutate or to `**`-splat alongside other kwargs.
    """
    if name not in VARIABLE_KW:
        raise KeyError(
            f"no display convention for {name!r}; "
            f"known variables: {sorted(VARIABLE_KW)}")
    return {**VARIABLE_KW[name], **overrides}


# Width of a bold character relative to its point size, used by both colorbar
# builders to estimate whether a string fits a given slot before drawing it.
_BOLD_CHAR_WIDTH_EM = 0.58


# ─── Water-year calendar ────────────────────────────────────────────────────
# DOWY 1 = Oct 1  (non-leap year)
_MONTH_STARTS = [1,  32,  62,  93, 124, 152, 183, 213, 244, 274, 305, 336]
_MONTH_ENDS   = [31, 61,  92, 123, 151, 182, 212, 243, 273, 304, 335, 365]

# Northern Hemisphere calendar months (water-year order Oct → Sep)
_NH_LONG  = ['October', 'November', 'December', 'January', 'February',
             'March',   'April',    'May',       'June',    'July',
             'August',  'September']
_NH_SHORT = ['OCT', 'NOV', 'DEC', 'JAN', 'FEB',
             'MAR', 'APR', 'MAY', 'JUN', 'JUL',
             'AUG', 'SEP']

# Southern Hemisphere equivalents (6-month offset in calendar)
_SH_LONG  = ['April',   'May',      'June',     'July',    'August',
             'September','October', 'November', 'December', 'January',
             'February', 'March']
_SH_SHORT = ['APR', 'MAY', 'JUN', 'JUL', 'AUG',
             'SEP', 'OCT', 'NOV', 'DEC', 'JAN',
             'FEB', 'MAR']

_HEMISPHERE_OPTIONS = ('both', 'northern', 'southern')


def create_month_colorbar(dowy_start, dowy_end,
                          hemisphere='both',
                          major_tick_spacing=40, minor_tick_spacing=10,
                          cmap='viridis', figsize=(8, 3), ax=None,
                          label='median runoff onset date [day of water year]',
                          label_fontsize=15, tick_labelsize=15,
                          month_fontsize=None,
                          abbreviate_month_names=True,
                          display_month_if_no_room=True):
    """
    Create a horizontal colorbar for runoff onset date with month labels.

    Parameters
    ----------
    dowy_start : int
        Left edge of the colorbar (Day of Water Year).
    dowy_end   : int
        Right edge of the colorbar (Day of Water Year).
    hemisphere : {'both', 'northern', 'southern'}, default 'both'
        Which hemisphere's month names to display.  'both' shows Northern
        Hemisphere labels (bold, top) and Southern Hemisphere labels
        (italic, parenthesized, bottom).  'northern' / 'southern' show only
        that hemisphere's labels, centered vertically.
    major_tick_spacing : int
        Spacing between labelled major ticks in DOWY units.  Ticks are placed
        at dowy_start, dowy_start + major_tick_spacing, … up to dowy_end,
        so the endpoints are always labelled regardless of the step size.
        Example: dowy_start=110, dowy_end=270, major_tick_spacing=40 gives
        ticks at 110, 150, 190, 230, 270.
    minor_tick_spacing : int
        Spacing between unlabelled minor ticks in DOWY units.
    cmap : str, default 'viridis'
        Colormap name.
    figsize : tuple, default (8, 3)
        Figure size in inches.  Ignored if `ax` is given.
    ax : matplotlib.axes.Axes or None, default None
        If given, draw the colorbar on this axes instead of creating a new
        figure.  Useful for embedding a colorbar within a larger figure
        (e.g. a GridSpec subplot).
    label : str
        Colorbar axis label.  The default deliberately carries **no** year
        count — per convention 3, a caller wanting one interpolates it from
        the data (`f'{len(ds.water_year)}-year median runoff onset date …'`),
        because a hardcoded count goes stale every new water year.
    label_fontsize : int, default 15
        Font size for the colorbar axis label.
    tick_labelsize : int, default 15
        Font size for the colorbar tick labels.
    month_fontsize : int or None, default None
        Font size for the primary month labels.  If None, a font size is
        chosen automatically based on the narrowest fully-visible month
        slot (clamped to [10, 20]).  The secondary label in 'both' mode is
        always 2pt smaller than the primary.
    abbreviate_month_names : bool, default True
        If True, use 3-letter abbreviations (JAN, FEB, …) instead of full
        names.
    display_month_if_no_room : bool, default True
        If False, a month label is silently omitted when its visible slot
        is too narrow to fit the text at the current font size.  If True
        (default), the label is drawn regardless — it may overlap the
        boundary lines on cramped colorbars.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the colorbar (the `ax`'s parent figure if
        `ax` was given).
    """

    if hemisphere not in _HEMISPHERE_OPTIONS:
        raise ValueError(f"hemisphere must be one of {_HEMISPHERE_OPTIONS}, got {hemisphere!r}")

    total_range = dowy_end - dowy_start
    if total_range <= 0:
        raise ValueError("dowy_end must be greater than dowy_start.")

    # ── Find months that overlap with [dowy_start, dowy_end] ─────────────────
    boundaries = []   # internal month-boundary lines that fall inside the range
    visible   = []    # list of dicts with positioning info per visible month

    for ms, me, nh_l, nh_s, sh_l, sh_s in zip(
            _MONTH_STARTS, _MONTH_ENDS,
            _NH_LONG, _NH_SHORT, _SH_LONG, _SH_SHORT):

        if me <= dowy_start or ms >= dowy_end:
            continue

        vis_start = max(ms, dowy_start)
        vis_end   = min(me, dowy_end)
        center    = (vis_start + vis_end) / 2
        width     = vis_end - vis_start

        visible.append(dict(center=center, width=width,
                            nh_long=nh_l, nh_short=nh_s,
                            sh_long=sh_l, sh_short=sh_s))

        # Internal boundary (month start that falls strictly inside the range)
        if dowy_start < ms < dowy_end:
            boundaries.append(ms)

    if not visible:
        raise ValueError(f"No months fall in DOWY range {dowy_start}-{dowy_end}.")

    # ── Font-size heuristic ──────────────────────────────────────────────────
    # Base font size on the narrowest *fully-visible* month slot.  Months
    # that are edge-clipped to fewer than 15 days (e.g. the 2-day stub of
    # December when dowy_start=90) are excluded so they don't drag the font
    # size down to an unreadably small value — those stubs will simply fail
    # the per-label fit check and be silently skipped.
    if month_fontsize is None:
        sizing_widths = [m['width'] for m in visible if m['width'] >= 15]
        if not sizing_widths:                        # all months are tiny stubs
            sizing_widths = [m['width'] for m in visible]
        min_width  = min(sizing_widths)
        proportion = min_width / total_range        # fraction of bar width

        # Font size: reference point → prop=0.15 gives size 14; clamped [10, 20]
        raw_fs = round(95 * proportion)             # 95 = 14 / 0.15
        fs_primary = max(10, min(20, raw_fs))
    else:
        fs_primary = month_fontsize
    fs_secondary = max(9, fs_primary - 2)

    # ── Tick layout ──────────────────────────────────────────────────────────
    major_ticks = list(range(dowy_start, dowy_end + 1, major_tick_spacing))
    if major_ticks[-1] != dowy_end:
        major_ticks.append(dowy_end)

    norm = mpl.colors.Normalize(vmin=dowy_start, vmax=dowy_end)

    # ── Build figure ──────────────────────────────────────────────────────────
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.10, 0.45, 0.80, 0.15])
    else:
        fig = ax.figure

    # ── Per-month fit check ───────────────────────────────────────────────────
    # Estimate whether a string fits inside a slot of a given width (in DOWY
    # units), based on the actual width of `ax` in inches.
    # At fontsize pt, a bold character is ~0.58 × (pt/72) inches wide.
    # We add a small horizontal padding factor to avoid edge crowding.
    fig.canvas.draw()
    BAR_WIDTH_IN  = ax.get_window_extent().width / fig.dpi   # inches
    CHAR_WIDTH_EM = _BOLD_CHAR_WIDTH_EM
    PAD_FACTOR    = 1.05          # extra breathing room

    def _fits(text, slot_width_dowy, fontsize):
        slot_in   = (slot_width_dowy / total_range) * BAR_WIDTH_IN
        text_in   = len(text) * CHAR_WIDTH_EM * (fontsize / 72.0) * PAD_FACTOR
        return text_in <= slot_in

    cb = mpl.colorbar.ColorbarBase(
        ax, orientation='horizontal', cmap=cmap,
        norm=norm, ticks=major_ticks, extend='both')

    cb.set_label(label, fontsize=label_fontsize, weight='normal', labelpad=10)
    cb.ax.tick_params(labelsize=tick_labelsize, length=4, width=1)
    cb.ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))
    cb.ax.tick_params(which='minor', length=2, width=0.5)

    # Month boundary lines
    for b in boundaries:
        ax.axvline(x=b, ymin=0, ymax=1,
                   color='white', linewidth=1.5, linestyle='--', zorder=10)

    # Month labels
    for m in visible:
        c     = m['center']
        width = m['width']

        if hemisphere == 'both':
            primary   = m['nh_short'] if abbreviate_month_names else m['nh_long']
            secondary = m['sh_short'] if abbreviate_month_names else m['sh_long']

            if display_month_if_no_room or _fits(primary, width, fs_primary):
                t = ax.text(c, 0.7, primary, fontsize=fs_primary,
                            ha='center', va='center', color='white',
                            weight='bold', transform=ax.transData, clip_on=False)
                t.set_path_effects([withStroke(linewidth=2, foreground='black')])

            secondary_label = f'({secondary})'
            if display_month_if_no_room or _fits(secondary_label, width, fs_secondary):
                t2 = ax.text(c, 0.24, secondary_label, fontsize=fs_secondary,
                             ha='center', va='center', color='white',
                             weight='bold', style='italic',
                             transform=ax.transData, clip_on=False)
                t2.set_path_effects([withStroke(linewidth=2, foreground='black')])

        else:
            if hemisphere == 'northern':
                primary = m['nh_short'] if abbreviate_month_names else m['nh_long']
            else:
                primary = m['sh_short'] if abbreviate_month_names else m['sh_long']

            if display_month_if_no_room or _fits(primary, width, fs_primary):
                t = ax.text(c, 0.5, primary, fontsize=fs_primary,
                            ha='center', va='center', color='white',
                            weight='bold', transform=ax.transData, clip_on=False)
                t.set_path_effects([withStroke(linewidth=2, foreground='black')])

    return fig


def create_diverging_colorbar(vmin, vmax,
                              cmap='RdBu',
                              label='',
                              ticks=None, minor_tick_spacing=None,
                              left_text='', right_text='',
                              figsize=(8, 3), ax=None,
                              label_fontsize=15, tick_labelsize=15,
                              text_fontsize=18, min_text_fontsize=7):
    """
    Create a horizontal diverging colorbar with optional left/right text labels.

    Parameters
    ----------
    vmin, vmax : float
        Data range of the colorbar.
    cmap : str, default 'RdBu'
        Colormap name.
    label : str
        Colorbar axis label.
    ticks : list of float or None
        Tick locations.  If None, ticks are placed at vmin, vmax, and
        evenly-spaced steps of (vmax - vmin) / 6 between them.
    minor_tick_spacing : float or None
        Spacing between unlabelled minor ticks.  If None, no minor ticks
        are drawn.
    left_text : str
        Text drawn at the left edge of the colorbar (e.g. 'earlier than median').
    right_text : str
        Text drawn at the right edge of the colorbar (e.g. 'later than median').
    figsize : tuple, default (8, 3)
        Figure size in inches.  Ignored if `ax` is given.
    ax : matplotlib.axes.Axes or None, default None
        If given, draw the colorbar on this axes instead of creating a new
        figure.  Useful for embedding a colorbar within a larger figure
        (e.g. a GridSpec subplot).
    label_fontsize : int, default 15
        Font size for the colorbar axis label.
    tick_labelsize : int, default 15
        Font size for the colorbar tick labels.
    text_fontsize : int, default 18
        Upper bound on the font size for left_text / right_text.  The actual
        size is shrunk from here if the two strings would otherwise collide in
        the middle of the bar (they are anchored to opposite ends), so this is
        a ceiling rather than a fixed size — a narrow colorbar axes or a longer
        label degrades gracefully instead of overprinting.
    min_text_fontsize : int, default 7
        Floor for that shrinking.  If the texts still don't fit at this size
        they are drawn anyway and will overlap — the bar is simply too narrow
        for them, and silently dropping a caller's annotation is worse.

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the colorbar (the `ax`'s parent figure if
        `ax` was given).
    """

    if ticks is None:
        step = (vmax - vmin) / 6
        ticks = [vmin + i * step for i in range(7)]

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.10, 0.45, 0.80, 0.15])
    else:
        fig = ax.figure

    cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap,
                                    norm=norm, ticks=ticks, extend='both')

    cb.set_label(label, fontsize=label_fontsize, weight='normal', labelpad=10)
    cb.ax.tick_params(labelsize=tick_labelsize, length=4, width=1)
    if minor_tick_spacing is not None:
        cb.ax.xaxis.set_minor_locator(MultipleLocator(minor_tick_spacing))
        cb.ax.tick_params(which='minor', length=2, width=0.5)

    # ── Fit the two end annotations ──────────────────────────────────────────
    # left_text is anchored at vmin and right_text at vmax, so together they
    # must fit within the bar's width or they collide mid-bar. Shrink the font
    # until they do, using the same character-width estimate as
    # create_month_colorbar (bold char ~ 0.58 * pt wide) plus a gap allowance.
    if left_text and right_text:
        fig.canvas.draw()
        bar_width_in = ax.get_window_extent().width / fig.dpi
        text_em = (len(left_text) + len(right_text)) * _BOLD_CHAR_WIDTH_EM * 1.15
        if text_em > 0:
            fits_at = 72.0 * bar_width_in / text_em      # largest pt that fits
            text_fontsize = max(min_text_fontsize,
                                min(text_fontsize, int(fits_at)))

    if left_text:
        t1 = ax.text(vmin, 0.5, left_text, fontsize=text_fontsize, ha='left', va='center',
                      color='white', weight='bold', transform=ax.transData)
        t1.set_path_effects([withStroke(linewidth=2.5, foreground='black')])

    if right_text:
        t2 = ax.text(vmax, 0.5, right_text, fontsize=text_fontsize, ha='right', va='center',
                      color='white', weight='bold', transform=ax.transData)
        t2.set_path_effects([withStroke(linewidth=2.5, foreground='black')])

    return fig


# Helper function to plot geodataframe geometries efficiently
def plot_geoms(gdf, ax, color="black", linewidth=1, transform=None, **kwargs):
    """Plot geometries efficiently without geopandas overhead"""
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type in ["Polygon", "MultiPolygon"]:
            # Get exterior coordinates
            if geom.geom_type == "Polygon":
                polys = [geom]
            else:
                polys = list(geom.geoms)
            for poly in polys:
                x, y = poly.exterior.xy
                ax.plot(x, y, color=color, linewidth=linewidth, transform=transform, **kwargs)


# ─── Regional map helpers ───────────────────────────────────────────────────
# Small primitives shared by the regional viewers in visualize/regions/.  They
# stay deliberately thin: each does one repetitive thing and hands the axes
# straight back, so a notebook can always drop down to raw matplotlib.

DEFAULT_HILLSHADE_PATH = 'visualize/data/global_hillshade_robinson.tif'
HILLSHADE_KW = dict(cmap='gray', vmin=0, vmax=400, add_colorbar=False, zorder=0)


def _repo_path(path):
    """Resolve a repo-relative path, so notebooks at any depth pass the same string."""
    path = pathlib.Path(path)
    if path.is_absolute():
        return path
    return pathlib.Path(__file__).resolve().parent.parent / path


def load_hillshade(gdf, buffer=50_000, path=DEFAULT_HILLSHADE_PATH,
                   crs=None, resampling=None, chunks='auto'):
    """
    Clip the global hillshade to a GeoDataFrame's (buffered) bounds and reproject it.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Region of interest.  Its CRS is the output CRS unless `crs` is given.
    buffer : float, default 50_000
        Buffer applied to the region's bounds, in units of the output CRS.
        Use a large value (a few 1e5 m) for a zoomed-out context inset.
    path : str or pathlib.Path
        Hillshade raster; relative paths resolve against the repository root.
    crs : any pyproj-parsable CRS or None
        Output CRS.  Defaults to `gdf.crs` (project to UTM first for a
        metric, equal-aspect regional map).
    resampling : rasterio.enums.Resampling or None
        Reprojection resampling, default bilinear.
    chunks : default 'auto'
        Passed to `rioxarray.open_rasterio`.

    Returns
    -------
    xarray.DataArray
        2-D hillshade in `crs`.  Plot with `**plot_utils.HILLSHADE_KW`.
    """
    import rasterio                # local imports keep this module importable
    import rioxarray               # in matplotlib-only contexts

    crs = gdf.crs if crs is None else crs
    if resampling is None:
        resampling = rasterio.enums.Resampling.bilinear
    bounds = gdf.to_crs(crs).buffer(buffer).total_bounds
    return (rioxarray.open_rasterio(_repo_path(path), masked=True, chunks=chunks)
            .rio.clip_box(*bounds, crs=crs)
            .squeeze()
            .rio.reproject(crs, resampling=resampling))


def style_map_axes(ax, bounds=None, title=''):
    """
    Strip an axes down to a bare map panel: no ticks/frame, no title, equal aspect.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    bounds : sequence of 4 floats or None
        `(xmin, ymin, xmax, ymax)` in data coordinates — the GeoPandas
        `total_bounds` order — applied as the axes limits.  None leaves the
        current limits alone.
    title : str, default ''
        Title text (default clears the one xarray's `.plot` sets).

    Returns
    -------
    matplotlib.axes.Axes
        The same axes, so calls can be chained/inlined in a loop.
    """
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    return ax


def add_scalebar(ax, dx=1, units='m', location='lower right',
                 length_fraction=0.25, color='black', frameon=False,
                 scale_loc='bottom', **kwargs):
    """
    Add a frameless scalebar to a projected, equal-aspect map axes (conventions 5-6).

    Only meaningful when the axes' data coordinates are linear metres — a UTM
    or equal-area CRS — which is why this module's regional maps are projected
    before plotting.  Do not put one on a lat/lon or global-projection map:
    the metres-per-degree varies across the frame and any single bar lies.
    `ScaleBar` itself warns if the axes aspect isn't 1, since a bar drawn on a
    stretched axes measures only one direction correctly.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Map axes whose data units are metres (`dx=1`) and whose aspect is equal.
    dx : float, default 1
        Size of one data unit in `units`.  1 for a metre-based CRS; e.g. 1000
        if the axes are in kilometres.
    units : str, default 'm'
        Unit of `dx`.  The bar's own label is chosen automatically (m/km).
    location : str, default 'lower right'
        Corner to place the bar in.  Two things decide it: pick a corner
        nothing else occupies (the context inset takes the upper left and
        per-panel water-year labels the lower right, so a map with both wants
        `'lower left'`), and — since the bar carries no box — one whose
        background contrasts with `color`.
    length_fraction : float, default 0.25
        Bar length as a fraction of the axes width (rounded to a nice number).
    color : default 'black'
        Bar and text colour.  Black reads over the mid-grey hillshade and over
        light data; pass `'white'` where the chosen corner is dark (ocean, deep
        shadow).  There is no halo available — `ScaleBar` builds its text
        inside `draw()`, so path effects can't be attached after the fact —
        which is exactly why the corner has to be chosen deliberately.
    frameon : bool, default False
        No box by convention; the bar sits directly on the map.  `True` (plus
        `box_color`/`box_alpha` via `**kwargs`) restores the boxed look.
    scale_loc : default 'bottom'
        Where the length text sits relative to the bar.
    **kwargs
        Any other `matplotlib_scalebar.scalebar.ScaleBar` argument.

    Returns
    -------
    matplotlib_scalebar.scalebar.ScaleBar
        The artist, already added to `ax`.
    """
    from matplotlib_scalebar.scalebar import ScaleBar

    bar = ScaleBar(dx, units, location=location, length_fraction=length_fraction,
                   color=color, frameon=frameon, scale_loc=scale_loc, **kwargs)
    ax.add_artist(bar)
    return bar


# Which corner of its allotted box an inset keeps when equal aspect shrinks it.
# Without this an aspect-shrunk inset recentres (Axes default anchor 'C'), so a
# corner inset ends up with visibly unequal padding on two of its sides.
_LOC_ANCHOR = {
    'upper left': 'NW',   'upper center': 'N',  'upper right': 'NE',
    'center left': 'W',   'center': 'C',        'center right': 'E',
    'lower left': 'SW',   'lower center': 'S',  'lower right': 'SE',
    'right': 'E',
}


def add_context_inset(ax, data_da, hillshade_da=None, loc='upper left',
                      size='26%', borderpad=0.15, bounds=None,
                      data_kw=None, hillshade_kw=None,
                      show_extent=True, extent_kw=None, frame_kw=None):
    """
    Draw a zoomed-out locator map inside a regional map axes.

    The inset renders the *same* variable with the *same* styling as the main
    map (pass the identical `cmap`/`vmin`/`vmax` via `data_kw`) over the same
    hillshade, and outlines the main axes' current extent — so it reads as
    "here is the map you're looking at, in its surroundings".

    Call this **after** the main map is drawn and its limits are set: the
    extent rectangle is taken from `ax.get_xlim()`/`get_ylim()`.  `data_da`
    (and `hillshade_da`) must already cover the wider area and be in the same
    CRS as `ax` — typically a coarser pyramid level clipped with a large
    buffer, plus `load_hillshade(gdf, buffer=<same>)`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The main map axes.
    data_da : xarray.DataArray
        Zoomed-out field, same variable as the main map.
    hillshade_da : xarray.DataArray or None
        Zoomed-out hillshade drawn underneath.
    loc : str, default 'upper left'
        Inset corner, any matplotlib legend location string.  The inset is
        anchored to that corner (see `_LOC_ANCHOR`) so equal aspect shrinking
        it doesn't pull it off the corner into the middle of the map.
    size : str, float, or (w, h), default '26%'
        Inset size as a percentage string ('26%') or inches (float), relative
        to the parent axes.  A 2-tuple sets width and height separately.
        Keep it small — it sits *on top of* the map it describes, and every
        percent costs data the reader came to look at.
    borderpad : float, default 0.15
        Padding between the inset and the parent axes edge, in font-size
        units.  Small by default so the inset tucks into the corner; 0 puts
        it flush against the map edge.
    bounds : sequence of 4 floats or None
        `(xmin, ymin, xmax, ymax)` limits for the inset; defaults to the full
        extent of `data_da`.  Passing the buffered region box explicitly
        (`gdf.buffer(buffer).total_bounds`) is usually better — it crops the
        NaN corners that reprojecting a lon/lat clip into a metric CRS leaves.
    data_kw, hillshade_kw : dict or None
        Overrides for the two `.plot.imshow` calls.  `hillshade_kw` defaults
        to `HILLSHADE_KW`.
    show_extent : bool, default True
        Draw the rectangle marking the main axes' extent.
    extent_kw : dict or None
        Overrides for that rectangle (default: red, linewidth 1.2, no fill).
    frame_kw : dict or None
        Overrides for the inset frame (default: black, linewidth 0.8) and its
        `facecolor` (default white, so ocean/no-data isn't transparent).

    Returns
    -------
    matplotlib.axes.Axes
        The inset axes, for any further annotation.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    width, height = size if isinstance(size, tuple) else (size, size)
    axins = inset_axes(ax, width=width, height=height, loc=loc,
                       borderpad=borderpad)
    # keep the requested corner when equal aspect shrinks the box, so padding
    # from the two adjacent figure edges stays equal
    if isinstance(loc, str) and loc.lower() in _LOC_ANCHOR:
        axins.set_anchor(_LOC_ANCHOR[loc.lower()])

    # main-map extent captured before anything else touches the figure
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()

    if hillshade_da is not None:
        hillshade_da.plot.imshow(ax=axins, **{**HILLSHADE_KW, **(hillshade_kw or {})})
    data_da.plot.imshow(ax=axins, **{'add_colorbar': False, 'zorder': 1,
                                     **(data_kw or {})})

    if bounds is None:
        left, bottom, right, top = data_da.rio.bounds()
        bounds = (min(left, right), min(bottom, top),
                  max(left, right), max(bottom, top))
    axins.set_title('')
    axins.set_aspect('equal')
    axins.set_xlim(bounds[0], bounds[2])
    axins.set_ylim(bounds[1], bounds[3])

    if show_extent:
        axins.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0, zorder=3,
            **{'facecolor': 'none', 'edgecolor': 'red', 'linewidth': 1.2,
               **(extent_kw or {})}))

    # ticks off but frame deliberately kept, so the inset reads as an inset
    frame_kw = {'edgecolor': 'black', 'linewidth': 0.8, 'facecolor': 'white',
                **(frame_kw or {})}
    axins.patch.set_facecolor(frame_kw['facecolor'])
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xlabel('')
    axins.set_ylabel('')
    for spine in axins.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(frame_kw['edgecolor'])
        spine.set_linewidth(frame_kw['linewidth'])
    return axins


def orthographic_locator_map(gdf, ax=None, figsize=(6, 6), edgecolor='black',
                             facecolor='none', linewidth=0.3,
                             central_longitude=None, central_latitude=None):
    """
    Globe-scale locator: the region outlined on an orthographic land/ocean view.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Region outline.  Reprojected to EPSG:4326 internally, so any CRS works.
    ax : matplotlib.axes.Axes or None
        Existing cartopy axes to draw on.  If None, a new figure is created
        with the globe centred on the region.
    figsize : tuple, default (6, 6)
        Figure size, ignored if `ax` is given.
    edgecolor, facecolor, linewidth : outline styling for the region geometry.
    central_longitude, central_latitude : float or None
        Globe centre; defaults to the centre of the region's bounds.

    Returns
    -------
    matplotlib.axes.Axes
        The (cartopy GeoAxes) axes the locator was drawn on.
    """
    import cartopy.crs as ccrs
    from cartopy import feature as cfeature

    gdf_4326 = gdf.to_crs('EPSG:4326')
    xmin, ymin, xmax, ymax = gdf_4326.total_bounds
    if central_longitude is None:
        central_longitude = (xmin + xmax) / 2
    if central_latitude is None:
        central_latitude = (ymin + ymax) / 2

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=ccrs.Orthographic(
            central_longitude=central_longitude,
            central_latitude=central_latitude))

    ax.add_feature(cfeature.LAND, zorder=0)
    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_geometries(gdf_4326.geometry, crs=ccrs.PlateCarree(),
                      edgecolor=edgecolor, facecolor=facecolor,
                      linewidth=linewidth, zorder=2)
    return ax