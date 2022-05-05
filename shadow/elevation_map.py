import os
from shadow.cutil import (
    load_image,
    deg2num,
    nums2degs
)

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import cv2
import geopandas as gpd
# TODO: For some strange reason, importing geopandas before shadow.cutil causes an ImportError
import numpy as np
import pandas as pd
import shapely.geometry
from geopandas import GeoDataFrame
from pandas import Series
from pyproj import Transformer


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
    w, s, e, n = gdf.total_bounds
    trans = Transformer.from_crs(gdf.crs, 4326, always_xy=True)
    w, n = trans.transform(w, n)
    e, s = trans.transform(e, s)
    w, n = deg2num(n, w, zoom)
    e, s = deg2num(s, e, zoom)

    # Just making sure that the tiles are actually north, west
    n = min(n, s)
    w = min(w, e)

    cellsize = 10
    gridsize = cellsize ** 2
    # np.ndarray indexing is [row, column], so I am using [north, west] to maintain that convention
    rn = np.arange(n, s, dtype=np.uint32)  # xtile goes from n to s
    rw = np.arange(w, e, dtype=np.uint32)  # ytile goes from w to e
    rs = rn + 1
    re = rw + 1
    index = pd.MultiIndex.from_product(
        (rn, rw, range(cellsize), range(cellsize)),
        names=['tn', 'tw', 'cn', 'cw']
    )
    tn = index.get_level_values('tn').values
    tw = index.get_level_values('tw').values
    ts = tn + 1
    te = tw + 1
    gn, gw = nums2degs(tw, tn, zoom)
    gs, ge = nums2degs(te, ts, zoom)
    trans = Transformer.from_crs(4326, gdf.crs, always_xy=True)
    pw, pn = trans.transform(gw, gn)
    pe, ps = trans.transform(ge, gs)
    l = pe - pw
    h = ps - pn
    a = np.abs(l * h) / gridsize
    dl = l / cellsize
    dh = h / cellsize

    cn = index.get_level_values('cn').values
    cw = index.get_level_values('cw').values
    pcn = pn + (cn * dh)
    pcw = pw + (cw * dl)
    pcs = pcn + dh
    pce = pcw + dl

    g = map(shapely.geometry.box, pcw, pcs, pce, pcn)
    cells = GeoDataFrame({
        'area': a,
        'geometry': g,
    }, index=index, crs=gdf.crs)

    icell, igdf = gdf.sindex.query_bulk(cells.geometry)
    cells = cells.iloc[icell]
    gdf = gdf.iloc[igdf]

    # abandon hope all ye who enter here
    intersection: Series = cells.intersection(gdf, align=False)

    area: Series = intersection.area

    max_height = gdf['height'].max()
    weight: Series = area / cells['area'].values * (gdf['height'].values / max_height) * 255
    # weight = weight.astype('uint8')

    agg: Series = weight.groupby(['tn', 'tw', 'cn', 'cw']).agg('sum')
    agg = agg.astype('uint8')
    # TODO: Better to not use pathlib, for sake of speed?
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
    images = (
        load_image(
            cn=subagg.index.get_level_values('cn').values,
            cw=subagg.index.get_level_values('cw').values,
            weights=subagg.values,
            cellsize=cellsize,
        )
        for subagg in subaggs
    )
    with ThreadPoolExecutor() as te:
        te.map(cv2.imwrite, paths, images)


if __name__ == '__main__':
    zoom = 15
    print('reading file...')
    gdf = gpd.read_feather('/home/arstneio/Downloads/new_york_city.feather')

    print('testing optimized...')
    t = time.time()
    run(gdf, zoom)
    print(f'optimized took {(time.time() - t) / 60} minutes')
