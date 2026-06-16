import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patheffects import withStroke


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
                          label='10-year median runoff onset date [day of water year]',
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
        Colorbar axis label.
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
    CHAR_WIDTH_EM = 0.58          # bold-font character width relative to pt size
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
                              text_fontsize=18):
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
        Font size for left_text / right_text.

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