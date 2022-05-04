from pandas import IndexSlice as idx
import time
import geopandas as gpd
import itertools
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

# from shadow.util import load_grid, num2deg, deg2num
transformer = Transformer.from_crs(3395, 4326)
invtransformer = Transformer.from_crs(4326, 3395)
from shadow.util import (
    nums2degs, degs2nums, load_grid
)

from tqdm.notebook import trange


def deg2num(lon_deg, lat_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


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

    # Generate Tiles
    trans = Transformer.from_crs(4326, 3395, always_xy=True)

    rw = range(math.floor(bottomleft[0]), math.ceil(topright[0]))
    rs = range(math.floor(bottomleft[1]), math.ceil(topright[1]))
    index = pd.MultiIndex.from_product((rw, rs), names=['tw', 'ts'])
    tw = index.get_level_values(0).values
    ts = index.get_level_values(1).values

    te = tw + 1
    tn = ts + 1

    # gw = geographic north
    gw, gs = nums2degs(tw, ts, zoom, len(tw))
    ge, gn = nums2degs(te, tn, zoom, len(te))

    # pw = projected west
    pw, ps = trans.transform(gw, gs)
    pe, pn = trans.transform(ge, gn)

    geometry = map(shapely.geometry.box, pw, ps, pe, pn)
    tiles = GeoDataFrame({
        'pw': pw, 'ps': ps, 'pn': pn, 'pe': pe,
        # 'area': abs(pe - pw) * abs(pn - ps),
        'l': abs(pe - pw),
        'h': abs(pn - ps),
        'geometry': geometry,
    }, index=index, crs=3395)
    tiles['area'] = tiles['l'] * tiles['h']

    # Query for tile, building matches
    itiles, igdf = gdf.sindex.query_bulk(tiles.geometry)

    # Apply the tiles to the gdf
    gdf = gdf.iloc[igdf]
    tiles = tiles.iloc[itiles]
    # TODO: Perhaps tiles aren't even necessary, and just create cells.
    intersection: GeoSeries = tiles.intersection(gdf, align=False)

    # Appropriate the transformation to the gdf
    gdf = gdf.set_geometry(intersection)
    gdf.index = intersection.index
    del tiles['geometry']

    # Generate cells
    cellsize = 10
    gridsize = cellsize ** 2

    # Generate two arrays of cartesian pairs

    cw = np.repeat(range(cellsize), cellsize)
    cs = np.tile(range(cellsize), cellsize)

    cw = np.tile(cw, len(tiles))
    cs = np.repeat(cs, len(tiles))

    tw = tw.repeat(gridsize)
    ts = ts.repeat(gridsize)

    index = pd.MultiIndex.from_arrays((
        tw, ts, cw, cs
    ), names=['tw', 'ts', 'cw', 'cs'])

    ce = cw + 1
    cn = cs + 1

    l = np.repeat(tiles['l'], gridsize)
    h = np.repeat(tiles['h'], gridsize)
    a = l * h / gridsize

    pw = np.repeat(tiles['pw'], gridsize) + (cw / l)
    pe = np.repeat(tiles['pw'], gridsize) + (ce / l)
    ps = np.repeat(tiles['ps'], gridsize) + (cs / h)
    pn = np.repeat(tiles['ps'], gridsize) + (cn / h)

    del tiles
    geometry = map(shapely.geometry.box, pw, pe, ps, pn)

    cells = GeoDataFrame({
        'area': a,
        'geometry': geometry,
    }, index=index)

    # Query for cell, building matches
    icell, igdf = gdf.sindex.query_bulk(cells.geometry)

    # Apply the cells to the gdf
    gdf: GeoDataFrame = gdf.iloc[igdf]
    cells: GeoDataFrame = cells.iloc[icell]

    intersection = cells.intersection(gdf, align=False).area / cells['area']
    tw = intersection.index.get_level_values('tw')
    ts = intersection.index.get_level_values('ts')
    height = gdf.loc[idx[tw, ts], 'height']
    weight: Series = intersection * height
    weights: gpd.oc = weight.groupby(level=[0, 1, 2, 3]).agg('sum')

    sum = weight.groupby(level=[0, 1, 2, 3]).agg('sum')
    groups = sum.groupby(level=['tw', 'ts'])
    for (tw, ts), loc in groups.items():
        tile: Series = sum.loc[loc]
        weights = tile.values
        cw = tile.index.get_level_values('cw')
        cs = tile.index.get_level_values('cs')
        grid = load_grid(cw, cs, weights, cellsize, len(tile))


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
    group = weight.groupby(level=[0, 1]).agg('sum')
    return load_grid(

    )


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
    zoom = 15
    print('reading file...')
    gdf = gpd.read_feather('/home/arstneio/Downloads/new_york_city.feather')

    print('testing optimized...')
    t = time.time()
    run(gdf, zoom)
    print(f'optimized took {(t - time.time()) / 60} minutes')

    ...
