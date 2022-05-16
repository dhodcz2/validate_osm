import argparse

if True:
    # TODO: For some strange reason, importing geopandas before shadow.cutil causes an ImportError
    from shadow.cutil import (
        load_image,
        deg2num,
        nums2degs,
        num2deg
    )
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import dask_geopandas as dg

import dask
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator

import cv2
import dask_geopandas as dgpd
import geopandas as gpd
import geopy.distance
import numpy as np
import pandas as pd
import pygeos.creation
import pygeos.creation
import pyproj
from geopandas import GeoDataFrame
from pandas import Series
from pyproj import Transformer
from dask import delayed
from pathlib import Path


def _max_height(gdf: GeoDataFrame | dgpd.GeoDataFrame | str | Path) -> tuple[float, dgpd.GeoDataFrame]:
    if isinstance(gdf, GeoDataFrame):
        if isinstance(gdf, GeoDataFrame):
            gdf = dgpd.from_dask_dataframe(gdf)
    elif isinstance(gdf, str | Path):
        if isinstance(gdf, str):
            ext = gdf.rpartition('.')[2]
        else:
            ext = gdf.name.rpartition('.')[2]

        match ext:
            case 'feather':
                gdf = dgpd.read_feather(gdf)
            case 'parquet':
                gdf = dgpd.read_parquet(gdf)
            case _:
                gdf = dgpd.read_file(gdf)
    else:
        raise TypeError
    gdf: dgpd.GeoDataFrame
    return gdf['height'].max().compute(), gdf


def get_tiles(gdf: dgpd.GeoDataFrame, zoom: int) -> GeoDataFrame:
    pw, ps, pe, pn = gdf.total_bounds.compute()

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
        # 'tn': tn, 'tw': tw,
        'tpw': tpw, 'tps': tps, 'tpe': tpe, 'tpn': tpn,
        'h': h, 'w': w,
    }, index=tntw, geometry=geometry, crs=gdf.crs)

    return tiles

    # itile, igdf = gdf.sindex.query_bulk(tiles.geometry)
    # gdf: GeoDataFrame = gdf.iloc[igdf]
    # tiles: GeoDataFrame = tiles.iloc[itile]
    # gdf = gdf.set_index(tiles.index)
    # # tiles: GeoDataFrame = tiles.loc[tiles.index.unique()]
    # tiles: GeoDataFrame = tiles.loc[~tiles.index.duplicated()]
    # return gdf, tiles


def get_cells(tiles: GeoDataFrame, cell_length: float = 10.0) -> tuple[dgpd.GeoDataFrame, int]:
    s, w, n, e = tiles.geometry.iloc[0].bounds
    cells_oned = math.ceil(
        abs(s - n) / cell_length
    )
    cells_twod = cells_oned ** 2
    tile_count = len(tiles)

    dh = tiles['h'].values / cells_oned
    dw = tiles['w'].values / cells_oned
    cn = np.repeat(
        np.arange(cells_oned, dtype=np.uint64), cells_oned,
    )
    cw = np.tile(
        np.arange(cells_oned, dtype=np.uint64), cells_oned,
    )
    cs = np.repeat(
        np.arange(1, cells_oned + 1, dtype=np.uint64), cells_oned
    )
    ce = np.tile(
        np.arange(1, cells_oned + 1, dtype=np.uint64), cells_oned
    )

    cnr = np.tile(cn, tile_count)
    cwr = np.tile(cw, tile_count)
    csr = np.tile(cs, tile_count)
    cer = np.tile(ce, tile_count)

    tpnr = np.repeat(tiles['tpn'].values, cells_twod)
    tpwr = np.repeat(tiles['tpw'].values, cells_twod)

    dhr = np.repeat(dh, cells_twod)
    dwr = np.repeat(dw, cells_twod)

    cpn = tpnr + (dhr * cnr)
    cps = tpnr + (dhr * csr)
    cpw = tpwr + (dwr * cwr)
    cpe = tpwr + (dwr * cer)

    index = tiles.index.repeat(cells_twod)
    cnr = np.tile(cn, tile_count)
    cwr = np.tile(cw, tile_count)

    area = np.abs(dh * dw)
    arear = np.repeat(area, cells_twod)

    cells = DataFrame({
        'cpn': cpn, 'cps': cps, 'cpw': cpw, 'cpe': cpe,
        'cn': cnr, 'cw': cwr,
        'area': arear,
    }, index=index)
    cells: dd.DataFrame = dask.dataframe.from_pandas(cells, chunksize=cells_twod,sort=False)
    def func(df: DataFrame):
        df = df.assign(
            # geometry=pygeos.creation.box(df['cpw'])
            geometry=pygeos.creation.box(
                df['cpw'].values,
                df['cps'].values,
                df['cpe'].values,
                df['cpn'].values,
            ))
        return df[['geometry', 'cn', 'cw', 'area']]

    meta = {
        'geometry': object,
        'cn': 'uint64', 'cw': 'uint64',
        'area': 'float64',
    }

    cells = cells.map_partitions(func=func, meta=meta)
    cells: dgpd.GeoDataFrame = dgpd.from_dask_dataframe(cells['geometry cn cw area'.split()])
    return cells, cells_oned


def partition_mapping(
        cells: GeoDataFrame,
        gdf: GeoDataFrame,
        max_height: float,
        directory: str,
        cell_length: int,
        zoom: int,
):
    # I don't care about chained assignment because after this is done the GDFs are just going to be thrown in the trash
    gdf = gdf.loc[cells.index.unique()]  # Get only the geometry relevant to the cells
    icell, igdf = gdf.sindex.query_bulk(cells.geometry)

    cells: GeoDataFrame = cells.iloc[icell]
    gdf: GeoDataFrame = gdf.iloc[igdf]

    intersection: Series = cells.intersection(gdf.geometry, align=False).area

    # weight: Series = intersection.values / cells['area'].values * gdf['height'].values / max_height * (2 ** 16 - 1)
    weight: np.ndarray = (
            intersection.values
            / cells['area'].values
            * gdf['height'].values
            / max_height
    )
    cells['weight'] = weight
    agg: GeoDataFrame = cells.groupby(['tntw', 'cn', 'cw'], sort=False).agg({'weight': 'sum'})
    agg['weight'] = (
            agg.values * (2 ** 16 - 1)
    ).astype(np.uint16)

    groups = agg.groupby('tntw', sort=False).groups
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
    with ThreadPoolExecutor() as te:
        te.map(os.makedirs, nodirs)
    subaggs: Iterator[Series] = (
        agg.loc[loc]
        for loc in groups.values()
    )
    images = (
        load_image(
            cn=subagg.index.get_level_values('cn').values,
            cw=subagg.index.get_level_values('cw').values,
            weights=subagg['weight'].values,
            cell_length=cell_length,
        )
        for subagg in subaggs
    )
    with ThreadPoolExecutor() as te:
        te.map(cv2.imwrite, paths, images)


def run(gdf: dgpd.GeoDataFrame, zoom: int, max_height: float, outputfolder: str):
    tiles = get_tiles(gdf, zoom)

    gdf = gdf.sjoin(tiles[['geometry']])
    gdf: dgpd.GeoDataFrame = gdf.rename(columns={'index_right': tiles.index.name})
    gdf = gdf.set_index('tntw')

    tiles = tiles.drop(tiles.index.difference(gdf.index))
    cells, length_cells = get_cells(tiles, 10.0)

    gdf.join(cells, 'tntw', 'outer', lsuffix='_gdf', rsuffix='_cells')


    grid_size = length_cells ** 2
    chunksize = grid_size * 50
    pd.set_option('mode.chained_assignment', None)
    cells.map_partitions(
        partition_mapping,
        gdf=gdf,
        meta=(None, None),
        max_height=max_height,
        directory=outputfolder,
        zoom=zoom,
        length_cells=length_cells,
        align_dataframes=True,
    ).compute()
    pd.set_option('mode.chained_assignment', 'warn')


class Namespace:
    input: list[str]
    output: list[str]
    verbose: bool
    zoom: list[int]
    cell_length: float
    max: float


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


    def gdfs() -> Iterator[dgpd.GeoDataFrame]:
        for path in args.input:
            match path.rpartition('.')[2]:
                case 'feather':
                    yield dgpd.read_feather(path)
                case 'parquet':
                    yield dgpd.read_parquet(path)
                case _:
                    yield dgpd.read_file(path)


    if not args.max:
        max_height = max(gdf['height'].max() for gdf in gdfs()).compute()
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
            if '.' not in output
            else output
            for output in args.output
        )

    for gdf, output in zip(gdfs(), args.output):
        for zoom in args.zoom:
            run(gdf, zoom, max_height, output)
