import argparse
import warnings

warnings.filterwarnings('ignore', '.*Shapely GEOS.*')

# I had to change cutil because the unknown import was messing with running the file outside the project
from cutil import (
    load_image,
    deg2num,
    nums2degs,
    num2deg
)

# if True:
#     from shadow.cutil import (
#         load_image,
#         deg2num,
#         nums2degs,
#         num2deg
#     )

import dask.dataframe as dd
import dask.array as da

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import cv2
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import pygeos.creation
import pygeos.creation
from geopandas import GeoDataFrame
from pandas import Series
from pyproj import Transformer
from pathlib import Path

from typing import Union


def _max_height(gdf: Union[GeoDataFrame, dgpd.GeoDataFrame, str, Path]) -> tuple[float, dgpd.GeoDataFrame]:
    if isinstance(gdf, GeoDataFrame):
        if isinstance(gdf, GeoDataFrame):
            gdf = dgpd.from_dask_dataframe(gdf)
    elif isinstance(gdf, str):
        if isinstance(gdf, str):
            ext = gdf.rpartition('.')[2]
        else:
            ext = gdf.name.rpartition('.')[2]

        if ext == 'feather':
            gdf = dgpd.read_feather(gdf)
        elif ext == 'parquet':
            gdf = dgpd.read_parquet(gdf)
        else:
            gdf = dgpd.read_file(gdf)
    else:
        raise TypeError

    gdf: dgpd.GeoDataFrame
    return gdf['height'].max().compute(), gdf


def get_tiles(gdf: GeoDataFrame, zoom: int) -> GeoDataFrame:
    pw, ps, pe, pn = gdf.total_bounds

    trans = Transformer.from_crs(gdf.crs, 4326, always_xy=True)
    gw, gn = trans.transform(pw, pn)
    ge, gs = trans.transform(pe, ps)

    tw, tn = deg2num(gw, gn, zoom, always_xy=True)
    te, ts = deg2num(ge, gs, zoom, always_xy=True)

    # Just making sure that the tiles are actually north, west
    tn, ts = min(tn, ts), max(tn, ts)
    tw, te = min(tw, te), max(tw, te)

    # np.ndarray indexing is [row, column], so I am using [north, west] to maintain that convention
    # Convention: repeat rows, tile columns

    # Slippy Tiles
    tn = np.arange(tn, ts, dtype=np.uint)  # xtile goes from n to s
    tw = np.arange(tw, te, dtype=np.uint)  # ytile goes from w to e

    # Geographic
    # Generate from northmost tiles and westmost tiles O(n) instead of all tiles O(n^2)
    _, tgn = nums2degs(np.repeat(tw[0], len(tn)), tn, zoom, always_xy=True)
    tgw, _ = nums2degs(tw, np.repeat(tn[0], len(tw)), zoom, always_xy=True)
    tgs = np.append(
        tgn[1:],
        num2deg(tw[0], ts, zoom, always_xy=True)[1]
    )
    tge = np.append(
        tgw[1:],
        num2deg(te, tn[0], zoom, always_xy=True)[0]
    )

    # Projected
    # Generate from northmost geographic and westmost geographic O(n) instead of all tiles O(n^2)
    trans = Transformer.from_crs(4326, gdf.crs, always_xy=True)
    _, tpn = trans.transform(np.repeat(tgw[0], len(tgn)), tgn)
    tpw, _ = trans.transform(tgw, np.repeat(tgn[0], len(tgw)))
    tps = np.append(
        tpn[1:],
        trans.transform(tgw[0], tgs[-1])[1]
    )
    tpe = np.append(
        tpw[1:],
        trans.transform(tge[-1], tgn[0])[0]
    )

    repeat_rows = len(tw)
    tile_columns = len(tn)
    tn = np.repeat(tn, repeat_rows)
    tw = np.tile(tw, tile_columns)

    tns = tn << 32
    tntw = np.bitwise_or(tns, tw)
    tntw = pd.Index(tntw, name='tntw', dtype=np.uint64)

    tpw = np.tile(tpw, tile_columns)
    tps = np.repeat(tps, repeat_rows)
    tpe = np.tile(tpe, tile_columns)
    tpn = np.repeat(tpn, repeat_rows)
    geometry = pygeos.creation.box(tpw, tps, tpe, tpn)
    h = (tps - tpn)
    w = (tpe - tpw)

    tiles = GeoDataFrame({
        'tn': tn, 'tw': tw,
        'tpn': tpn, 'tpw': tpw,
        # 'tpw': tpw, 'tps': tps, 'tpe': tpe, 'tpn': tpn,
        'h': h, 'w': w,
        # }, geometry=geometry, crs=gdf.crs)
    }, index=tntw, geometry=geometry, crs=gdf.crs)

    itile, igdf = gdf.sindex.query_bulk(tiles.geometry)
    loc = tiles.index[itile].unique()
    tiles: GeoDataFrame = tiles.loc[loc]
    # tiles = tiles.sort_values(['tn', 'tw'], ascending=True)
    tiles = tiles.sort_index(ascending=True)
    return tiles


def get_cells(tiles: GeoDataFrame, cell_length: float = 10.0) -> tuple[dgpd.GeoDataFrame, int]:
    s, w, n, e = tiles.geometry.iloc[0].bounds
    cells_oned = math.ceil(
        abs(s - n) / cell_length
    )
    cells_twod = cells_oned ** 2
    tile_count = len(tiles)

    mb_per_tile = 8 * 8 * cells_twod / 1024 / 1024
    tiles_per_chunk = math.floor(75 / mb_per_tile)
    chunksize = cells_twod * tiles_per_chunk

    dh = tiles['h'].values / cells_oned
    dw = tiles['w'].values / cells_oned
    if cells_oned > 255:
        raise ValueError(
            f"{cells_oned=}>255. This means that the image will be downscaled, and cells require more than"
            f" uint8. Increase zoom level."
        )
    cn = np.repeat(
        np.arange(cells_oned, dtype=np.uint8), cells_oned,
    )
    cw = np.tile(
        np.arange(cells_oned, dtype=np.uint8), cells_oned,
    )
    cs = np.repeat(
        np.arange(1, cells_oned + 1, dtype=np.uint8), cells_oned
    )
    ce = np.tile(
        np.arange(1, cells_oned + 1, dtype=np.uint8), cells_oned
    )
    cnr = np.tile(cn, tile_count)
    cwr = np.tile(cw, tile_count)
    csr = np.tile(cs, tile_count)
    cer = np.tile(ce, tile_count)

    tpnr = da.from_array(
        np.repeat(tiles['tpn'].values, cells_twod),
        name='tpnr',
        chunks=chunksize,
    )
    tpwr = da.from_array(
        np.repeat(tiles['tpw'].values, cells_twod),
        name='tpwr',
        chunks=chunksize,
    )
    dhr = da.from_array(
        np.repeat(dh, cells_twod),
        name='dhr',
        chunks=chunksize,
    )
    dwr = da.from_array(
        np.repeat(dw, cells_twod),
        name='dwr',
        chunks=chunksize,
    )

    cpn = tpnr + (dhr * cnr)
    cps = tpnr + (dhr * csr)
    cpw = tpwr + (dwr * cwr)
    cpe = tpwr + (dwr * cer)

    tntw = dd.from_dask_array(da.from_array(
        tiles.index.values.repeat(cells_twod),
        chunksize,
    ), columns='tntw')
    area = np.abs(dh * dw)
    arear = dd.from_dask_array(da.from_array(
        np.repeat(area, cells_twod),
        chunksize,
    ), columns='area')

    geometry = da.map_blocks(
        pygeos.creation.box, cpw, cps, cpe, cpn,
        dtype=object,
    )
    geometry = dd.from_dask_array(geometry, columns='geometry')
    cn = dd.from_dask_array(da.from_array(
        cnr, chunksize
    ), 'cn')
    cw = dd.from_dask_array(da.from_array(
        cwr, chunksize,
    ), 'cw')
    cells = dd.concat([cn, cw, arear, geometry, tntw], axis=1)
    cells: dgpd.GeoDataFrame = dgpd.from_dask_dataframe(cells)
    cells.crs = tiles.crs

    iloc = list(range(0, tile_count - 1, tiles_per_chunk))
    iloc.append(tile_count - 1)
    divisions = list(tiles.index[iloc])
    cells = cells.set_index('tntw', sorted=True, divisions=divisions)
    return cells, cells_oned


def partition_mapping(cells: GeoDataFrame, directory: str, length_cells: int, zoom: int, ):
    weight: Series = cells.groupby(['tntw', 'cn', 'cw'], sort=False).weight.sum()
    weight: Series = weight.astype(np.uint16)
    groups = weight.groupby('tntw', sort=False).groups
    tntw = np.fromiter(groups.keys(), dtype=np.uint64)
    tn = np.bitwise_and(tntw, (2 ** 64 - (2 ** 32))) >> 32
    tw = np.bitwise_and(tntw, (2 ** 32 - 1))

    paths = [
        os.path.join(directory, f'{zoom}/{tw_}/{tn_}.png')
        for tn_, tw_ in zip(tn, tw)
    ]
    nodirs = (
        dir
        for path in paths
        if not os.path.exists(dir := os.path.dirname(path))
    )
    subaggs: Iterator[Series] = (
        weight.loc[loc]
        for loc in groups.values()
    )
    images = (
        load_image(
            cn=subagg.index.get_level_values('cn').values,
            cw=subagg.index.get_level_values('cw').values,
            weights=subagg.values,
            cell_length=length_cells,
        )
        for subagg in subaggs
    )
    with ThreadPoolExecutor() as te:
        te.map(os.makedirs, nodirs)
    with ThreadPoolExecutor() as te:
        te.map(cv2.imwrite, paths, images)


def run(gdf: GeoDataFrame, zoom: int, max_height: float, outputfolder: str):
    tiles = get_tiles(gdf, zoom)
    cells, length_cells = get_cells(tiles)

    cells = cells.sjoin(gdf)
    cells = cells.merge(
        gdf[['geometry']], how='left', left_on='index_right', right_index=True, suffixes=('_cells', '_gdf'),
    )
    del gdf
    cells: dgpd.GeoDataFrame = dgpd.from_dask_dataframe(cells, geometry='geometry_gdf')

    cells['weight'] = (
            dgpd.GeoSeries.intersection(cells['geometry_gdf'], cells['geometry_cells']).area
            / cells['area']
            * cells['height']
            / max_height
            * (2 ** 16 - 1)
    )

    warnings.filterwarnings('ignore', '.*empty Series.*')
    meta = dd.utils.make_meta((None, None))
    warnings.filterwarnings('ignore', '.*empty Series.*')

    cells.map_partitions(
        partition_mapping,
        directory=outputfolder,
        length_cells=length_cells,
        zoom=zoom,
        meta=meta,
    ).compute()


class Namespace:
    input: list[str]
    output: list[str]
    verbose: bool
    zoom: list[int]
    cell_length: float
    max: float
    chunktiles: int


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate elevation map from geometric data')
    parser.add_argument(
        'input',
        nargs='+',
        help='input files for which elevation maps will be generated'
    )
    parser.add_argument(
        '--output',
        '-o',
        nargs='*',
        help='output directories in which the elevation maps will be generated',
        default=[os.getcwd()],
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true'
    )
    parser.add_argument(
        '--zoom',
        '-z',
        nargs='+',
        type=int,
        help='slippy tile zoom levels at which the elevation map will be generated',
    )
    parser.add_argument(
        '--length',
        type=float,
        help='projected length for each cell within a slippy tile',
        default=10.0
    )
    parser.add_argument(
        '--max',
        type=float,
        help='max height',
        default=0
    )
    args = parser.parse_args(namespace=Namespace)


    def gdfs() -> Iterator[GeoDataFrame]:
        for path in args.input:
            ext = path.rpartition('.')[2]
            if ext == 'feather':
                yield gpd.read_feather(path)
            elif ext == 'parquet':
                yield gpd.read_parquet(path)
            else:
                yield gpd.read_file(path)


    if not args.max:
        max_height = max(gdf['height'].max() for gdf in gdfs())
    else:
        max_height = args.max

    cwd = os.getcwd()
    if not args.output:
        args.output = (
            os.path.join(
                cwd,
                input.rpartition('/')[2].rpartition('.')[0]
            )
            for input in args.input
        )
    else:
        args.output = (
            os.path.join(cwd, output)
            if '/' not in output
            else output
            for output in args.output
        )

    if args.verbose:
        for gdf, output, input in zip(gdfs(), args.output, args.input):
            for zoom in args.zoom:
                t = time.time()
                run(gdf, zoom, max_height, output)
                dest = os.path.join(output, str(zoom))
                print(f"{(time.time() - t) / 60:.1f}; {input=} {dest=}")
    else:
        for gdf, output in zip(gdfs(), args.output):
            for zoom in args.zoom:
                run(gdf, zoom, max_height, output)
