if True:
    from shadow.cutil import (
        xtiles_from_lons,
        ytiles_from_lats,
        lons_from_xtiles,
        lats_from_ytiles,
    )
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

import geopandas as gpd

import skimage.io
import concurrent.futures
import os

import pandas as pd
import pyproj

from shadow.cutil import deg2num, nums2degs
import pygeos.creation
import cv2
import cython
import numpy as np
import spatialpandas.geometry


def get_cells_from_image(image: np.ndarray, pw: float, ps: float, pe: float, pn: float):
    where = np.where(image > 0)

    dh = (ps - pn) / 256
    dw = (pe - pw) / 256

    cn = where // 256
    cw = where % 256
    area = np.repeat(abs(dh * dw), len(where))

    cgn = cn * dh
    cgw = cw * dw
    cgs = (cn + 1) * dh
    cge = (cw + 1) * dw

    geometry = np.concatenate((
        cgw, cgs, cge, cgs, cge, cgn, cgw, cgn, cgw, cgs
    ), axis=1, dtype=np.float64)
    geometry = spatialpandas.geometry.PolygonArray(geometry)
    gdf = spatialpandas.GeoDataFrame({
        'value': image[where],
        'area': area,
    }, geometry=geometry)
    return gdf


def get_cells_from_tiles(tiles: GeoDataFrame, dir: str):
    tn = tiles.index.get_level_values('tn').values
    tw = tiles.index.get_level_values('tw').values
    s, e, n, w = tiles['s e n w'.split()].T.values
    # TODO: tn and tw seem to be flipped
    paths = [
        path
        for xtile, ytile in zip(tw, tn)
        if os.path.exists(
            path := os.path.join(dir, str(xtile), f'{ytile}.png')
        )
    ]
    images: list[np.ndarray] = list(concurrent.futures.ThreadPoolExecutor().map(skimage.io.imread, paths))
    print()
    # TODO: get the relevant paths


def get_tiles_from_bounds(gw: float, gs: float, ge: float, gn: float, zoom: int, crs: 4326):
    tw, tn = deg2num(gw, gn, zoom, True)
    te, ts = deg2num(ge, gs, zoom, True)
    tn = np.arange(tn, ts, dtype=np.uint32)
    tw = np.arange(tw, te, dtype=np.uint32)
    ts = tn + 1
    te = tw + 1

    gw = lons_from_xtiles(tw, zoom)
    ge = lons_from_xtiles(te, zoom)
    gn = lats_from_ytiles(tn, zoom)
    gs = lats_from_ytiles(ts, zoom)

    trans = pyproj.Transformer.from_crs(4326, crs, always_xy=True).transform
    pw, pn = trans(gw, gn)
    pe, ps = trans(ge, gs)

    tile_columns = len(tn)
    repeat_rows = len(tw)

    tn = np.repeat(tn, repeat_rows)
    tw = np.tile(tw, tile_columns)
    index = pd.MultiIndex.from_arrays((tn, tw), names=['tn', 'tw'])

    # tntw = tn.astype(np.uint64) << 32
    # tntw = np.bitwise_or(tntw, tw)
    # index = tntw

    # pn = np.repeat(pn, repeat_rows)
    # pw = np.repeat(pw, repeat_rows)
    # ps = np.tile(ps, tile_columns)
    # pe = np.tile(pe, tile_columns)

    pn = np.repeat(pn, repeat_rows)
    ps = np.repeat(ps, repeat_rows)
    pw = np.tile(pw, tile_columns)
    pe = np.tile(pe, tile_columns)

    geometry = pygeos.creation.box(pw, ps, pn, pe)
    gdf = gpd.GeoDataFrame({'s': ps, 'n': pn, 'e': pe, 'w': pw, }, index=index, geometry=geometry)
    return gdf


if __name__ == '__main__':
    tiles = get_tiles_from_bounds(
        *(41.68690456932072, -87.7731951452115)[::-1],
        *(42.00812423120439, -87.49853693294519)[::-1],
        16,
        crs=4326
    )
    cells = get_cells_from_tiles(tiles, os.path.join('/home/arstneio/Downloads/shadows/nyc-sep-22/', str(16)))

    # TODO: get_xtiles_ytiles_from_bounds
    #   THEN get_tiles_from_xtiles_ytiles
