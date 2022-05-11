from shadow.cutil import (
    load_image,
    deg2num,
    nums2degs,
    num2deg,
    degs2nums
)

import functools
import pyproj
import pygeos.creation
import itertools
import math

import dask
import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import dask_geopandas as dg

import numpy
import pygeos.creation

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame
import geopy.distance
from geopandas import GeoSeries
import cv2
import geopandas as gpd
# TODO: For some strange reason, importing geopandas before shadow.cutil causes an ImportError
import numpy as np
import pandas as pd
import shapely.geometry
from geopandas import GeoDataFrame
from pandas import Series
from pyproj import Transformer
from dask_geopandas import GeoDataFrame as DaskGeoDataFrame
import dask_geopandas
import os


def get_tiles(gdf: GeoDataFrame, zoom: int) -> tuple[GeoDataFrame, GeoDataFrame]:
    # Get tile bounds from GDF
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

    # tn = np.repeat(tn, len(tw))
    # tw = np.tile(tw, len(tn))
    repeat_rows = len(tw)
    tile_columns = len(tn)
    tn, tw = np.repeat(tn, repeat_rows), np.tile(tw, tile_columns)
    tntw = pd.MultiIndex.from_arrays((tn, tw))
    tntw = pd.util.hash_pandas_object(tntw)
    tntw = pd.Index(tntw, name='tntw')

    tpw = np.tile(tpw, tile_columns)
    tps = np.repeat(tps, repeat_rows)
    tpe = np.tile(tpe, tile_columns)
    tpn = np.repeat(tpn, repeat_rows)
    geometry = pygeos.creation.box(tpw, tps, tpe, tpn)
    h = (tps - tpn)
    w = (tpe - tpw)

    tiles = GeoDataFrame({
        'tn': tn, 'tw': tw,
        'tpw': tpw, 'tps': tps, 'tpe': tpe, 'tpn': tpn,
        'h': h, 'w': w,
    }, index=tntw, geometry=geometry, crs=gdf.crs)

    itile, igdf = gdf.sindex.query_bulk(tiles.geometry)
    gdf: GeoDataFrame = gdf.iloc[igdf]
    tiles: GeoDataFrame = tiles.iloc[itile]
    gdf = gdf.set_index(tiles.index)
    # tiles: GeoDataFrame = tiles.loc[tiles.index.unique()]
    tiles: GeoDataFrame = tiles.loc[~tiles.index.duplicated()]
    return gdf, tiles


def get_cells(tiles: GeoDataFrame, cell_length: float = 10.0) -> GeoDataFrame:
    s, w, n, e = tiles.geometry.iloc[0].bounds
    trans = pyproj.Transformer.from_crs(tiles.crs, 4326, always_xy=True)
    (w, e), (s, n) = trans.transform((w, e), (n, s))
    distance = geopy.distance.distance((n, w), (s, w)).meters
    cell_count = math.ceil(distance / cell_length)
    grid_count = cell_count ** 2

    # h = np.repeat(tiles['h'], grid_count)
    # w = np.repeat(tiles['w'], grid_count)
    # area = np.abs(h * w)
    # area = np.repeat(area, grid_count)

    h = tiles['h'].values
    w = tiles['w'].values
    # TODO: Create dask_gdf of gdf and cells, and call map_partitions
    tpnr = np.repeat(tiles['tpn'].values, grid_count)
    tpwr = np.repeat(tiles['tpw'].values, grid_count)
    tpsr = np.repeat(tiles['tps'].values, grid_count)
    tper = np.repeat(tiles['tpe'].values, grid_count)

    area = np.abs(h * w)
    arear = np.repeat(area, grid_count)
    hr = np.repeat(h, grid_count)
    wr = np.repeat(w, grid_count)

    cn = np.repeat(range(cell_count), cell_count)
    cw = np.tile(range(cell_count), cell_count)
    cs = np.repeat(range(1, cell_count + 1), cell_count)
    ce = np.tile(range(1, cell_count + 1), cell_count)

    tile_count = len(tiles)
    cnr = np.tile(cn, tile_count)
    cwr = np.tile(cw, tile_count)
    csr = np.tile(cs, tile_count)
    cer = np.tile(ce, tile_count)

    cpn = tpnr + (cnr * hr)
    cpw = tpwr + (cwr * wr)
    cps = tpsr + (csr * hr)
    cpe = tper + (cer * wr)

    index = tiles.index.repeat(grid_count)
    geometry = pygeos.creation.box(cpw, cps, cpe, cpn)

    tnr = tiles['tn'].repeat(grid_count)
    twr = tiles['tw'].repeat(grid_count)
    cnr = cn.repeat(tile_count)
    cwr = cw.repeat(tile_count)

    cells = GeoDataFrame({
        'tn': tnr, 'tw': twr, 'cn': cnr, 'cw': cwr,
        'area': arear,
    }, index=index, geometry=geometry, crs=tiles.crs)
    return cells


def partition_mapping(
        cells: GeoDataFrame,
        gdf: GeoDataFrame,
        max_height: float,
        directory: str,
        cell_length: int,
):
    gdf = gdf.loc[cells.index.unique()]
    icell, igdf = gdf.sindex.query_bulk(cells.geometry)
    cells: GeoDataFrame = cells.iloc[icell]
    gdf: GeoDataFrame = gdf.iloc[igdf]

    t = time.time()
    intersection: Series = cells.intersection(gdf, align=False).area
    print(f'\t{(time.time() - t)/60} minutes for {len(cells)=}')

    weight: Series = intersection / cells['area'].values * gdf['height'].values / max_height
    cells['weight'] = weight

    agg: GeoDataFrame = cells.groupby('tn tw cn cw'.split()).agg({'weight': 'sum'})
    agg['weight'] = agg['weight'].astype(np.uint16)
    groups = agg.groupby(['tn', 'tw']).groups

    paths = [
        os.path.join(directory, f'{zoom}/{tn}/{tw}.png')
        for tn, tw in groups.keys()
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
    images = [
        load_image(
            cn=subagg.index.get_level_values('cn').values,
            cw=subagg.index.get_level_values('cw').values,
            weights=subagg['weight'].values,
            cell_length=cell_length,
        )
        for subagg in subaggs
    ]
    with ThreadPoolExecutor() as te:
        te.map(cv2.imwrite, paths, images)
    # TODO: figure out why all the cels are 0
    print()

def test_partition_mapping(
        gdf: GeoDataFrame,
        cells: GeoDataFrame,
        max_height: float,
        directory: str,
        cell_length: int,
):
    if not cells.index.difference(gdf.index).empty:
        raise ValueError

if __name__ == '__main__':
    zoom = 15
    print('reading file...')
    gdf = gpd.read_feather('/home/arstneio/Downloads/new_york_city.feather')

    gdf, tiles = get_tiles(gdf, zoom)
    cells = get_cells(tiles, 10.0)

    max_height = gdf['height'].max()
    cell_length = len(cells['cn'].unique())

    cells: dask_geopandas.GeoDataFrame = dask_geopandas.from_geopandas(cells, chunksize=5000, sort=True)
    gdf: dask_geopandas.GeoDataFrame = dask_geopandas.from_geopandas(gdf, chunksize=5000, sort=True)
    # max_height = gdf['height'].max()
    # cell_length = len(cells['cn'].unique())

    # cells: dask_geopandas.GeoDataFrame = cells.assign(
    #     height=gdf['height'],
    #     building=gdf['geometry'],
    # )
    #
    print('partition mapping')
    t = time.time()
    cells.map_partitions(
        partition_mapping,
        gdf=gdf,
        meta=(None, None),
        max_height=max_height,
        directory=os.getcwd(),
        cell_length=cell_length,
        align_dataframes=True
    ).compute()
    # gdf.map_partitions(
    #     partition_mapping,
    #     cells,
    #     meta=(None, None),
    #     max_height=max_height,
    #     directory=os.getcwd(),
    #     cell_length=cell_length,
    # )
    print(f'optimized took {(time.time() - t) / 60} minutes')







