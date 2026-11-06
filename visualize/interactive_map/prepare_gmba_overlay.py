"""Publish the GMBA Mountain Inventory v2.0 overlay for the interactive map.

Downloads the GMBA Mountain Inventory v2.0 "standard" 300-selection (the 291
polygons GMBA curates for ~1:300M cartography; Snethlage et al. 2022,
doi:10.1038/s41597-022-01256-y, https://www.earthenv.org/mountains), drops the
one Ocean feature, and publishes two immutable-cached blobs to the public
Azure container (issue #36):

    .../snowmelt_runoff_onset/gmba_v2_standard_300_2.pmtiles
        Vector tiles (tippecanoe, z0-z10) of the outlines, layer "ranges",
        plus one label point per range, layer "labels". Tiles are simplified
        per zoom, so the world view costs ~0.3 MB while outlines sharpen to
        full resolution as you zoom in -- the v1 single GeoJSON had to be
        simplified to ~2 km everywhere to stay ~2 MB. GMBA ranges are
        multipolygons of hill-sized fragments (25.6k parts; half under
        50 km^2, together ~1% of the area), which at world zoom read as
        speckle, so each range is written in three zoom tiers that drop
        parts and holes below a size floor (GENERALIZATION_TIERS; a range
        always keeps its largest part). --detect-shared-borders simplifies
        neighbouring ranges' common edges identically (the selection is a
        partition: no gaps or slivers). Features carry only id + name;
        everything else lives in the index.
    .../snowmelt_runoff_onset/gmba_v2_standard_300_index_2.json
        ~25 KB gzip-encoded index, one entry per range: name, hierarchy path,
        countries, feature type, area, elevation span, Wikidata id, bbox,
        label point. Feeds the map's range card (looked up by the clicked
        feature's id) and the search box, without fetching any tiles.

The map (map/components/map.tsx; GMBA_PMTILES_URL / GMBA_INDEX_URL in
map/lib/store.ts) reads the PMTiles through the pmtiles:// protocol (HTTP
range requests; Azure CORS already allows Range for the sharded pyramids) the
first time the overlay is toggled on. Immutable-cache convention: to change
the overlay, bump the trailing _2 in --suffix AND in both URLs in store.ts.

tippecanoe is not a project dependency: the script runs it through
`pixi exec` (a throwaway conda-forge env, no pixi.lock change) unless a
tippecanoe binary is already on PATH.

    # rebuild + upload (needs config/sas_token.txt or AZURE_STORAGE_SAS_TOKEN)
    pixi run python visualize/interactive_map/prepare_gmba_overlay.py

    # inspect locally without touching Azure
    pixi run python visualize/interactive_map/prepare_gmba_overlay.py \
        --local-dir /tmp/gmba --skip-upload
"""

import argparse
import gzip
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

GMBA_ZIP_URL = (
    "https://data.earthenv.org/mountains/standard/"
    "GMBA_Inventory_v2.0_standard_300.zip"
)
DEST_PREFIX = "snowmelt_runoff_onset"
SUFFIX_DEFAULT = "2"
CONTAINER = "snowmelt"
CACHE = "public, max-age=31536000, immutable"
TIPPECANOE_SPEC = "tippecanoe>=2.79"
MAX_ZOOM = 10  # ~10 m tile precision; the map overzooms past this

# (minzoom, maxzoom, size floor in km^2) per geometry tier of a range: parts
# and holes smaller than the floor are dropped in that zoom band. A pixel is
# ~40 km at z2, ~10 km at z4, ~2.5 km at z6 (equator), so each floor keeps
# parts at least ~2x2 px at the top of its band.
GENERALIZATION_TIERS = ((0, 2, 5000.0), (3, 4, 1000.0), (5, 6, 100.0),
                        (7, MAX_ZOOM, 0.0))

# Attributes kept for the range card / search index, renamed to compact keys.
FIELD_MAP = {
    "GMBA_V2_ID": "id",
    "MapName": "name",
    "Feature": "feature",
    "Countries": "countries",
    "Area": "area_km2",
    "Elev_Low": "elev_low",
    "Elev_High": "elev_high",
    "Path": "path",
    "Hier_Lvl": "level",
    "WikiDataUR": "wikidata",
}


def label_minzoom(area_km2: float) -> int:
    """Zoom at which a range's label first appears. Big ranges label from the
    world view; small ones wait until they're a few hundred pixels across.
    MapLibre's collision pass (sorted by area) thins whatever still overlaps.
    """
    for threshold, zoom in ((300_000, 1), (100_000, 2), (30_000, 3),
                            (8_000, 4)):
        if area_km2 >= threshold:
            return zoom
    return 5


def label_point(geom) -> tuple[float, float]:
    """Pole of inaccessibility of the largest part: always inside the range
    (a centroid can fall outside a crescent-shaped range)."""
    from shapely.ops import polylabel

    parts = getattr(geom, "geoms", [geom])
    pt = polylabel(max(parts, key=lambda p: p.area), tolerance=0.01)
    return round(pt.x, 4), round(pt.y, 4)


def lon_bbox(geom) -> list[float]:
    """[west, south, east, north], with east > 180 for antimeridian crossers
    (Chukotka, Aleutians), so a fitBounds lands on the range, not the globe."""
    import shapely

    w, s, e, n = geom.bounds
    if e - w <= 180:
        return [w, s, e, n]
    xs = shapely.get_coordinates(geom)[:, 0]
    xs = (xs + 360) % 360  # shift western-hemisphere parts east of 180
    return [float(xs.min()), s, float(xs.max()), n]


def polygons_only(geom):
    """make_valid can return a GeometryCollection; keep its polygonal parts."""
    import shapely
    from shapely.geometry import MultiPolygon

    parts = [p for p in shapely.get_parts(geom)
             if p.geom_type in ("Polygon", "MultiPolygon")]
    return MultiPolygon([q for p in parts
                         for q in getattr(p, "geoms", [p])])


def drop_crumbs(geom, geom_ea, min_km2: float):
    """Drop parts and holes smaller than min_km2 (measured on the equal-area
    copy geom_ea, whose parts/rings are in the same order), always keeping
    the largest part."""
    from shapely.geometry import MultiPolygon, Polygon

    if min_km2 <= 0:
        return geom
    areas = [p.area / 1e6 for p in geom_ea.geoms]
    largest = areas.index(max(areas))
    kept = []
    for i, (part, part_ea) in enumerate(zip(geom.geoms, geom_ea.geoms)):
        if areas[i] < min_km2 and i != largest:
            continue
        holes = [h for h, h_ea in zip(part.interiors, part_ea.interiors)
                 if Polygon(h_ea).area / 1e6 >= min_km2]
        kept.append(Polygon(part.exterior, holes))
    return MultiPolygon(kept)


def load(src: str):
    """Read the shapefile, drop the ocean, repair the 4 invalid polygons,
    rename/trim attributes."""
    import geopandas as gpd

    gdf = gpd.read_file(src)
    gdf = gdf[gdf["Feature"] != "Ocean"]  # one polygon: the Arctic Ocean
    gdf = gdf[list(FIELD_MAP) + ["geometry"]].rename(columns=FIELD_MAP)
    gdf["geometry"] = gdf.geometry.make_valid().apply(polygons_only)
    gdf["area_km2"] = gdf["area_km2"].round(0)
    gdf["wikidata"] = (gdf["wikidata"].fillna("")
                       .str.rsplit("/", n=1).str[-1])  # keep just the Q-id
    for col in ("elev_low", "elev_high"):
        gdf[col] = gdf[col].round(0)
    return gdf.reset_index(drop=True)


def write_inputs(gdf, tmp: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Ranges GeoJSON (one feature per range per generalization tier, id +
    name only) + label points. Written by hand rather than with to_file
    because tippecanoe's per-feature zoom range is a GeoJSON foreign member.
    The two antimeridian crossers are already split at +/-180 in the source.
    """
    from shapely.geometry import mapping

    geom_ea = gdf.geometry.to_crs("+proj=eqearth")  # for part/hole areas
    ranges = tmp / "ranges.geojson"
    with ranges.open("w") as f:
        f.write('{"type":"FeatureCollection","features":[\n')
        first = True
        for row, g_ea in zip(gdf.itertuples(), geom_ea):
            for minzoom, maxzoom, min_km2 in GENERALIZATION_TIERS:
                feature = {
                    "type": "Feature",
                    "tippecanoe": {"minzoom": minzoom, "maxzoom": maxzoom},
                    "properties": {"id": int(row.id), "name": row.name},
                    "geometry": mapping(drop_crumbs(row.geometry, g_ea,
                                                    min_km2)),
                }
                f.write(("" if first else ",\n") + json.dumps(feature))
                first = False
        f.write("\n]}\n")

    features = []
    for row in gdf.itertuples():
        features.append({
            "type": "Feature",
            "tippecanoe": {"minzoom": label_minzoom(row.area_km2)},
            "properties": {"id": int(row.id), "name": row.name,
                           "area_km2": row.area_km2},
            "geometry": {"type": "Point",
                         "coordinates": label_point(row.geometry)},
        })
    labels = tmp / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection",
                                  "features": features}))
    return ranges, labels


def write_index(gdf, out_path: pathlib.Path) -> None:
    """Range card + search index: one entry per range, no geometry."""
    import math

    def num(v):
        return None if v is None or math.isnan(v) else int(v)

    entries = []
    for row in gdf.itertuples():
        entries.append({
            "id": int(row.id),
            "name": row.name,
            "path": row.path,
            "countries": row.countries or "",
            "feature": row.feature,
            "level": num(row.level),
            "area_km2": num(row.area_km2),
            "elev_low": num(row.elev_low),
            "elev_high": num(row.elev_high),
            "wikidata": row.wikidata,
            "bbox": [round(v, 3) for v in lon_bbox(row.geometry)],
            "center": label_point(row.geometry),
        })
    out_path.write_text(json.dumps(entries, ensure_ascii=False,
                                   separators=(",", ":")))


def tippecanoe_cmd() -> list[str]:
    if shutil.which("tippecanoe"):
        return ["tippecanoe"]
    return ["pixi", "exec", "--spec", TIPPECANOE_SPEC, "tippecanoe"]


def build_pmtiles(ranges: pathlib.Path, labels: pathlib.Path,
                  out_path: pathlib.Path) -> None:
    cmd = tippecanoe_cmd() + [
        "-o", str(out_path), "--force", "--quiet",
        "-Z0", f"-z{MAX_ZOOM}",
        "-n", "GMBA Mountain Inventory v2.0 (standard 300-selection)",
        "-A", '<a href="https://doi.org/10.1038/s41597-022-01256-y">'
              'GMBA Mountain Inventory v2.0</a>',
        "-L", json.dumps({"file": str(ranges), "layer": "ranges"}),
        "-L", json.dumps({"file": str(labels), "layer": "labels"}),
        # outlines must never be dropped or thinned; the 290 polygons are
        # small enough per tile once simplified
        "--no-feature-limit", "--no-tile-size-limit",
        "--detect-shared-borders",
        # keep every label point at every zoom (tippecanoe otherwise thins
        # points below maxzoom); label_minzoom already staggers them
        "-r1",
        # numeric GMBA id as the MVT feature id -> maplibre feature-state
        "--use-attribute-for-id=id",
    ]
    subprocess.run(cmd, check=True)


def upload(pmtiles_path: pathlib.Path, index_path: pathlib.Path,
           suffix: str) -> None:
    """Upload both blobs with cache/type headers set in the same call.

    The PMTiles archive goes up as-is (its tiles are already gzipped inside
    and the client range-reads it); the index is gzip-encoded.
    """
    from azure.storage.blob import ContainerClient, ContentSettings

    from global_snowmelt_runoff_onset.config import Config

    config = Config("config/global_config_v10.txt")
    account_url = f"https://{config.azure_storage_account}.blob.core.windows.net"
    client = ContainerClient(account_url, CONTAINER,
                             credential=config.sas_token)

    pm_blob = f"{DEST_PREFIX}/gmba_v2_standard_300_{suffix}.pmtiles"
    client.upload_blob(
        pm_blob, pmtiles_path.read_bytes(), overwrite=True,
        content_settings=ContentSettings(
            content_type="application/vnd.pmtiles", cache_control=CACHE),
    )
    print(f"uploaded {pmtiles_path.stat().st_size / 1e6:.2f} MB -> "
          f"{account_url}/{CONTAINER}/{pm_blob}")

    idx_blob = f"{DEST_PREFIX}/gmba_v2_standard_300_index_{suffix}.json"
    compressed = gzip.compress(index_path.read_bytes(), compresslevel=9)
    client.upload_blob(
        idx_blob, compressed, overwrite=True,
        content_settings=ContentSettings(
            content_type="application/json", content_encoding="gzip",
            cache_control=CACHE),
    )
    print(f"uploaded {len(compressed) / 1e3:.0f} KB (gzip) -> "
          f"{account_url}/{CONTAINER}/{idx_blob}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", default=None,
                        help="Local GMBA shapefile/zip instead of downloading "
                             f"(default: {GMBA_ZIP_URL})")
    parser.add_argument("--suffix", default=SUFFIX_DEFAULT,
                        help="Immutable-cache generation suffix of both blobs; "
                             "bump together with GMBA_PMTILES_URL and "
                             "GMBA_INDEX_URL in map/lib/store.ts")
    parser.add_argument("--local-dir", default=None,
                        help="Also keep the PMTiles + index in this directory")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Build only; do not touch Azure")
    args = parser.parse_args()

    src = args.src or f"zip+{GMBA_ZIP_URL}"
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = pathlib.Path(tmp_str)
        out_dir = pathlib.Path(args.local_dir) if args.local_dir else tmp
        out_dir.mkdir(parents=True, exist_ok=True)
        pmtiles = out_dir / f"gmba_v2_standard_300_{args.suffix}.pmtiles"
        index = out_dir / f"gmba_v2_standard_300_index_{args.suffix}.json"

        gdf = load(src)
        ranges, labels = write_inputs(gdf, tmp)
        build_pmtiles(ranges, labels, pmtiles)
        write_index(gdf, index)
        print(f"{len(gdf)} ranges -> {pmtiles} "
              f"({pmtiles.stat().st_size / 1e6:.2f} MB), {index} "
              f"({index.stat().st_size / 1e3:.0f} KB raw)")
        if not args.skip_upload:
            upload(pmtiles, index, args.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
