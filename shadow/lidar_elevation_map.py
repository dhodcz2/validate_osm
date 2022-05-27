import glob
from typing import Optional
import time

if True:
    from shadow.cutil import deg2num, nums2degs, num2deg
import warnings

import shapely.geometry.polygon

warnings.filterwarnings('ignore', '.*Shapely GEOS.*')

import concurrent.futures

import tempfile

import pdal
import pygeos.creation
import pystac.item_collection

import pystac_client
import planetary_computer
import shapely.geometry

import os

import cv2
import numpy as np
import pandas as pd
import pygeos.creation
import pygeos.creation
from geopandas import GeoDataFrame


def _get_slippy_tiles(polygon: shapely.geometry.Polygon, zoom: int) -> GeoDataFrame:
    gw, gs, ge, gn = polygon.bounds
    tw, tn = deg2num(gw, gn, zoom, True)
    te, ts = deg2num(ge, gs, zoom, True)
    te += 1
    ts += 1

    tn = np.arange(tn, ts, dtype=np.uint)
    tw = np.arange(tw, te, dtype=np.uint)
    repeat_rows = len(tw)
    tile_columns = len(tn)

    _, tgn = nums2degs(np.repeat(tw[0], len(tn)), tn, zoom, always_xy=True)
    tgw, _ = nums2degs(tw, np.repeat(tn[0], len(tw)), zoom, always_xy=True)
    tgs = np.append(tgn[1:], num2deg(tw[0], ts, zoom, always_xy=True)[1])
    tge = np.append(tgw[1:], num2deg(te, tn[0], zoom, always_xy=True)[0])

    tgn = np.repeat(tgn, repeat_rows)
    tgs = np.repeat(tgs, repeat_rows)
    tn = np.repeat(tn, repeat_rows)

    tgw = np.tile(tgw, tile_columns)
    tge = np.tile(tge, tile_columns)
    tw = np.tile(tw, tile_columns)

    geometry = pygeos.creation.box(tgw, tgs, tge, tgn)
    index = pd.MultiIndex.from_arrays((tn, tw), names=['tn', 'tw'])
    tiles = GeoDataFrame(geometry=geometry, index=index, crs=4326)
    # tiles = GeoDataFrame({
    #     'tn': tn,
    #     'tw': tw,
    # }, geometry=geometry, crs=4326)

    # Only include tiles that are entirely within BBox
    #   1. we only make 1 call to pystac_client.Client.search()
    #   2. avoid unexpected behavior with the tiles being larger than the bbox,
    #       thus failing to use copc chunks that were previously not returned
    tiles = tiles[tiles.within(polygon)]
    return tiles


def _resize(path, dest):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite(dest, image)


def load(
        polygon: shapely.geometry.Polygon,
        zoom: int,
        max_height: float,
        outdir: Optional[str] = None,
):
    tiles = _get_slippy_tiles(polygon, zoom)
    union = tiles.unary_union
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(collections=['3dep-lidar-copc'], intersects=union)

    signed: pystac.item_collection.ItemCollection = planetary_computer.sign(search)
    hrefs = np.fromiter((
        item.assets['data'].href
        for item in signed
    ), dtype='U1024')
    crs = {
        item.properties['proj:projjson']['components'][0]['id']['code']
        for item in signed
    }
    if len(crs) > 1:
        raise ValueError(
            f"The LIDAR partitions have different CRSes. The current code doesn't yet account for this"
        )
    crs = crs.pop()

    polygon = union.wkt + ' / EPSG:4326'
    tempdir = os.path.join(tempfile.gettempdir(), str(zoom))

    readers = (
        pdal.Reader.copc(
            href,
            resolution=2,
            polygon=polygon
        )
        for href in hrefs
    )

    merge = pdal.Filter.merge()
    hag = pdal.Filter.hag_nn()
    saturated = 2 ** 16 - 1
    assign = pdal.Filter.assign(value=[
        f'HeightAboveGround = 0 WHERE HeightAboveGround < 7',
        f'HeightAboveGround = HeightAboveGround / {max_height} * {saturated}',
        f'HeightAboveGround = {saturated} WHERE HeightAboveGround > {saturated}'
    ])
    tn = tiles.index.get_level_values('tn').values.astype('U10')
    tw = tiles.index.get_level_values('tw').values.astype('U10')
    # it_bounds = np.fromiter((
    #     f'([{minx},{maxx}],[{miny},{maxy}])'
    #     for minx, miny, maxx, maxy in tiles.bounds.values
    # ), dtype='U128')
    it_bounds = [
        f'([{minx},{maxx}],[{miny},{maxy}])'
        for minx, miny, maxx, maxy in tiles.to_crs(crs).bounds.values
    ]

    zoom_ = str(zoom)
    tempdirs = np.fromiter((
        os.path.join(tempdir, xtile)
        for xtile, ytile in zip(tn, tw)
    ), dtype='U4086')
    with concurrent.futures.ThreadPoolExecutor() as threads:
        threads.map(os.makedirs, tempdirs)

    # Note to self: don't even bother with iterators when debugging. Once you have it fully implemented, switch
    #   from [ to (
    tempfiles = [
        os.path.join(dir, f'{ytile}.tif')
        for dir, ytile in zip(tempdirs, tw)
    ]
    writers = (
        pdal.Writer.gdal(
            output_type='mean',
            resolution=2,
            dimension='HeightAboveGround',
            data_type='uint16',
            nodata=0,
            bounds=bounds,
            filename=filename,
        ) for bounds, filename in zip(it_bounds, tempfiles)
    )

    if outdir is None:
        outdir = os.getcwd()
    outdir = os.path.join(outdir, zoom_)
    outdirs = [
        os.path.join(outdir, xtile)
        for xtile in tw
    ]
    outfiles = [
        os.path.join(outdir, f'{ytile}.png')
        for outdir, ytile in zip(outdirs, tn)
    ]

    with concurrent.futures.ThreadPoolExecutor() as threads:
        threads.map(os.makedirs, outdirs)

    pipeline = next(readers)
    for reader in readers:
        pipeline |= reader
    pipeline = pipeline | merge | hag | assign
    for writer in writers:
        pipeline |= writer
    pipeline.execute()

    with concurrent.futures.ThreadPoolExecutor() as threads:
        threads.map(_resize, tempfiles, outfiles)


if __name__ == '__main__':
    zoom = 16
    box = shapely.geometry.box(
        *(41.85520272864603, -87.64667114877022)[::-1],
        *(41.886776137830516, -87.62461266183428)[::-1],
    )
    outputfolder = os.getcwd()
    outputfolder = os.path.join(outputfolder, str(zoom))
    t = time.time()
    load(box, 16, 500)
    t = time.time() - t
    pathname = os.path.join(outputfolder, '*/*')
    files = glob.glob(pathname)
    print(f"{t // 60} minutes {t % 60} seconds for {len(files)} tiles at {zoom=}")
