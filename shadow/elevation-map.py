import geopandas as gpd
import itertools
from shadow.util import load_grid
import math
import os

import cv2
import dask_geopandas
# TODO: Instead of every call to digital_elevation creating bounds, let's create bounds once inside run()
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame
from pyproj import Transformer
from shapely.geometry import box
from tqdm.notebook import trange
from shadow.util import load_grid, num2deg, deg2num

transformer = Transformer.from_crs(3395, 4326)
invtransformer = Transformer.from_crs(4326, 3395)

from tqdm.notebook import trange


def run(gdf: GeoDataFrame, zoom: int):
    # Get tile bounds from GDF
    bounds = gdf.total_bounds
    lat0, lng0 = transformer.transform(bounds[0], bounds[1])
    lat1, lng1 = transformer.transform(bounds[2], bounds[3])
    coord0 = deg2num(lat0, lng0, zoom)
    coord1 = deg2num(lat1, lng1, zoom)
    bottomleft = [min(coord0[0], coord1[0]), min(coord0[1], coord1[1])]
    topright = [max(coord0[0], coord1[0]), max(coord0[1], coord1[1])]

    max_height = gdf.height.max()

    # Get tile grid
    trans = Transformer.from_crs(4326, 3395, always_xy=True)

    tw = range(math.floor(bottomleft[0]), math.ceil(topright[0]))
    ts = range(math.floor(bottomleft[1]), math.ceil(topright[1]))
    sw = DataFrame((
        num2deg(x, y, zoom)
        for x, y in zip(tw, ts)
    ), columns=['x', 'y'])
    pw, ps = trans.transform(sw['x'], sw['y'])

    te = range(math.floor(bottomleft[0]) + 1, math.ceil(topright[0]) + 1)
    tn = range(math.floor(bottomleft[1]) + 1, math.ceil(topright[1]) + 1)
    ne = DataFrame((
        num2deg(x, y, zoom)
        for x, y in zip(te, tn)
    ), columns=['x', 'y'])
    pe, pn = trans.transform(ne['x'], ne['y'])

    # tw: tile west; ts: tile south
    # pw: projected west; ps: projected south
    tiles = GeoDataFrame({
        'tw': tw, 'ts': ts, 'tn': tn, 'te': te,
        'pw': pw, 'ps': ps, 'pn': pn, 'pe': pe,
        'l': abs(pe - pw),
        'h': abs(pn - ps),
        'geometry': GeoSeries((
            shapely.geometry.box(*vals)
            for vals in zip(pw, ps, pe, pn)
        ))
    }, index=['tw', 'ts'], crs=3395)

    cellsize = 10
    # TODO: group by tiles, and call create_image for each tile
    itiles, igdf = gdf.sindex.query_bulk(tiles.geometry)
    tiles: GeoDataFrame = tiles.iloc[itiles]
    gdf: GeoDataFrame = gdf.iloc[igdf]

    intersections: GeoSeries = tiles.geometry.iloc[itiles].intersection(gdf.geometry.iloc[igdf], align=False)
    del gdf.geometry
    # intersection.groupby(level=['tw', 'ts'])
    groups = intersections.groupby(level=['tw', 'ts']).groups
    for loc, intersection in groups.items():
        tile: Series = tiles.loc[loc]
        grid = digital_elevation(tile, intersection, cellsize)

    # # intersection = gdf.intersection(tiles, align=False)
    #
    # tiles = tiles.assign(
    #     geometry=intersection,
    #     height=gdf.height
    # )
    # groups = tiles.groupby(['tw', 'ts'], as_index=False, sort=False, group_keys=False).groups
    # for (tw, ts), loc in groups.items():
    #     grid = grid_from_tile(tiles.loc[loc], 10)
    #


def create_image(i, j, zoom, max_height):
    bb0 = num2deg(i, j, zoom)
    bb1 = num2deg(i + 1, j + 1, zoom)
    bb0 = invtransformer.transform(bb0[0], bb0[1])
    bb1 = invtransformer.transform(bb1[0], bb1[1])
    filtered = gdf.cx[bb0[0]:bb1[0], bb0[1]:bb1[1]]
    if len(filtered) > 0:

        dem = grid_from_tile(filtered, 10)
        dem_bounds = filtered.total_bounds
        dem_bb0 = transformer.transform(dem_bounds[0], dem_bounds[1])
        dem_bb1 = transformer.transform(dem_bounds[2], dem_bounds[3])
        dem_bb0 = deg2num(dem_bb0[0], dem_bb0[1], zoom)
        dem_bb1 = deg2num(dem_bb1[0], dem_bb1[1], zoom)
        dsize = (math.ceil(256 * abs(dem_bb1[0] - dem_bb0[0])), math.ceil(256 * abs(dem_bb1[1] - dem_bb0[1])))
        dem = cv2.resize(dem, dsize=dsize)

        startx = 256 - 256 * (dem_bb0[0] % 1)
        endx = (-256 * (dem_bb1[0] % 1))
        starty = 256 - 256 * (dem_bb1[1] % 1)
        endy = (-256 * (dem_bb0[1] % 1))

        values = dem[math.ceil(starty):math.floor(endy), math.ceil(startx):math.floor(endx)]
        if values.shape[0] != 0 and values.shape[1] != 0:

            values = cv2.resize(values, dsize=(256, 256))
            filename = 'data/nyc-heights/%d/%d/%d.png' % (zoom, i, j)
            directory = os.path.dirname(filename)
            if not os.path.exists(directory):
                os.makedirs(directory)
            cv2.imwrite(filename, 255.0 * (values / max_height))


def digital_elevation(tile: Series, gdf: GeoDataFrame, cellsize: Series):
    tw, ts, pw, ps, pn, pe, l, h = tile[['tw', 'ts', 'pw', 'pn', 'pe', 'l', 'h']]
    l /= cellsize
    h /= cellsize
    cw = range(pw, pe, l)
    ce = range(pw + l, pe + l, l)
    cs = range(ps, pn, h)
    cn = range(ps + h, pn + h, h)
    # TODO: Check if its cs and ce
    index = pd.MultiIndex(zip(cs, ce), names=['cs', 'ce'])
    cells = GeoSeries(map(shapely.geometry.box, cw, ce, cs, cn), index=index)

    icell, igdf = gdf.sindex.query_bulk(cells)
    cells: GeoSeries = cells.iloc[icell]
    gdf: GeoDataFrame = gdf.iloc[igdf]

    intersection: GeoSeries = cells.intersection(gdf.geometry, align=False)
    weight: Series = intersection.area / (l * h)
    weight.groupby(level=[])
    # TODO:


def grid_from_tile(tile: gpd.GeoDataFrame, cellsize):
    # TODO: Instead of every call to digital_elevation creating bounds, let's create bounds once inside run()
    bounds = tile.total_bounds
    bb0 = math.ceil(max(bounds[2], bounds[0]))
    bb1 = math.floor(min(bounds[2], bounds[0]))
    bb2 = math.ceil(max(bounds[3], bounds[1]))
    bb3 = math.floor(min(bounds[3], bounds[1]))

    width = bb0 - bb1
    height = bb2 - bb3
    grid = np.zeros((math.ceil(height / cellsize), math.ceil(width / cellsize)))

    xmin = range(bb1, bb0, cellsize)
    ymin = range(bb3, bb2, cellsize)
    mins = itertools.product(xmin, ymin)
    bounds = GeoDataFrame(mins, columns=['xmin', 'ymin'])
    bounds['xmax'] = bounds['xmin'] + cellsize
    bounds['ymax'] = bounds['ymin'] + cellsize
    bounds = bounds.set_geometry([
        shapely.geometry.box(*vals)
        for vals in bounds.values
    ], crs=3395)
    bounds['i'] = ((bounds['xmin'] - bb1) / cellsize).astype('uint16')
    bounds['j'] = ((bounds['ymin'] - bb3) / cellsize).astype('uint16')

    ibounds, igeometries = tile.sindex.query_bulk(bounds.geometry)
    bounds: GeoDataFrame = bounds.iloc[ibounds]
    tile: GeoDataFrame = tile.iloc[igeometries]

    intersection: GeoSeries = bounds.intersection(tile, align=False)
    bounds['weight'] = intersection.area.values / bounds.area.values * tile.height.values

    result = bounds.groupby(['j', 'i'], as_index=False, sort=False, group_keys=False).agg({'weight': 'sum'})
    load_grid(
        grid,
        i=result.index.get_level_values('i').values,
        j=result.index.get_level_values('j').values,
        val=result['weight'].values,
        length=len(result)
    )
    return grid


if __name__ == '__main__':
    ...
