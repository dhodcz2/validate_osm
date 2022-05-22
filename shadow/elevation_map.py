
from shadow.cutil import (
    load_image,
    deg2num,
    nums2degs,
    num2deg,
    degs2nums
)

import pygeos.creation
import itertools
import math

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

def run(
        gdf: GeoDataFrame,
        zoom: int,
        directory: str | Path | None = None
) -> None:
    """
    tn: tile north
    cn: cell north
    gn: geographic north
    pn: projected north
    pcn: projected cell north

    :param gdf:
    :param zoom:
    :return:
    """
    # directory = Path(
    #     os.getcwd() if directory is None
    #     else directory
    # )
    directory = os.getcwd() if directory is None else str(directory)

    # Get tile bounds from GDF
    pw, ps, pe, pn = gdf.total_bounds
    trans = Transformer.from_crs(gdf.crs, 4326, always_xy=True)
    gw, gn = trans.transform(pw, pn)
    ge, gs = trans.transform(pe, ps)
    # dist = geopy.distance.distance((n, w), (n + 1, w)).meters
    # cellsize = math.ceil(dist / 10)

    tw, tn = deg2num(gw, gn, zoom, always_xy=True)
    te, ts = deg2num(ge, gs, zoom, always_xy=True)

    # Just making sure that the tiles are actually north, west
    tn = min(tn, ts)
    tw = min(tw, te)
    ts = max(tn, ts)
    te = max(tw, te)

    total_dist = geopy.distance.distance((gn, gw), (gs, gw)).meters
    tile_dist = total_dist / (ts - tn)
    cellsize = math.ceil(tile_dist / 10)
    gridsize = cellsize ** 2

    # np.ndarray indexing is [row, column], so I am using [north, west] to maintain that convention
    # Convention: repeat rows, tile columns

    # Slippy Tiles
    tn = np.arange(tn, ts, dtype=np.uint)  # xtile goes from n to s
    tw = np.arange(tw, te, dtype=np.uint)  # ytile goes from w to e
    # ts = tn + 1
    # te = tw + 1

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

    index = pd.MultiIndex.from_arrays((
        np.repeat(tn, len(tw)),
        np.tile(tw, len(tn)),
    ), names=['tn', 'tw'])
    tpw = np.tile(tpw, len(tn))
    tps = np.repeat(tps, len(tw))
    tpe = np.tile(tpe, len(tn))
    tpn = np.repeat(tpn, len(tw))
    # TODO: cn, cw were done incorrectly for cells
    cells.to_feather('test_cells.feather')

    geometry = pygeos.creation.box(tpw, tps, tpe, tpn)
    tiles: GeoDataFrame = GeoDataFrame({
        'tpn': tpn, 'tpw': tpw, 'tps': tps, 'tpe': tpe,
    }, index=index, geometry=geometry)

    itile, igdf = gdf.sindex.query_bulk(tiles.geometry)
    del tiles['geometry']
    gdf = gdf.iloc[igdf]
    index = tiles.index[itile]
    loc = index.unique()
    tiles = tiles.loc[loc]
    index = pd.Index(index, name='tntw')

    tn = loc.get_level_values('tn').values
    tw = loc.get_level_values('tw').values

    tpn, tpw, tps, tpe = tiles.T.values
    dl = (tpe - tpw) / cellsize
    dh = (tps - tpn) / cellsize

    area = np.abs(dl * dh) / gridsize

    cn = np.repeat(range(cellsize), cellsize)
    cw = np.tile(range(cellsize), cellsize)
    cs = np.repeat(range(1, cellsize+1), cellsize)
    ce = np.tile(range(1, cellsize+1), cellsize)

    arear = np.repeat(area, gridsize)
    cnr = np.repeat(cn, len(loc))
    cwr = np.repeat(cw, len(loc))
    csr = np.repeat(cs, len(loc))
    cer = np.repeat(ce, len(loc))

    tpnr = np.repeat(tpn, gridsize)
    tpwr = np.repeat(tpw, gridsize)
    tpsr = np.repeat(tps, gridsize)
    tper = np.repeat(tpe, gridsize)

    dhr = np.repeat(dh, gridsize)
    dlr = np.repeat(dl, gridsize)

    cpn = tpnr + (cnr * dhr)
    cpw = tpwr + (cwr * dlr)
    cps = tpsr + (csr * dhr)
    cpe = tper + (cer * dlr)

    tnr = tn.repeat(gridsize)
    twr = tw.repeat(gridsize)

    # index = pd.MultiIndex.from_arrays((
    #     tnr, twr, cnr, cwr,
    # ), names='tn tw cn cw'.split())
    index = pd.MultiIndex.from_arrays((tnr, twr), names='tn tw'.split())
    cells = DataFrame({
        'cn': cnr, 'cw': cwr,
        'cpn': cpn, 'cpw': cpw, 'cps': cps, 'cpe': cpe,
        'area': arear,
    }, index=index)

    print()

    # TODO: How the fuck am I supposed to parallelize this?


if __name__ == '__main__':
    zoom = 15
    print('reading file...')
    gdf = gpd.read_feather('/home/arstneio/Downloads/new_york_city.feather')

    print('testing optimized...')
    t = time.time()
    # run(gdf, zoom)
    # run_parallel_tiles(gdf, zoom)
    run(gdf, zoom)
    print(f'optimized took {(time.time() - t) / 60} minutes')
