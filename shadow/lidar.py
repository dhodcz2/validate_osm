import concurrent.futures
import itertools
from functools import partial

import cv2

if True:
    from shadow.cutil import deg2num, nums2degs, num2deg
import os
import tempfile
from typing import Optional

import pandas as pd
import pdal
import pygeos.creation
import pystac.item_collection

import numpy as np
import pystac_client
import planetary_computer
from geopandas import GeoDataFrame
import shapely.geometry


class LidarTiles:
    def __init__(self, bbox: shapely.geometry.Polygon):
        self._bbox = bbox
        # catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        # search = catalog.search(collections=['3dep-lidar-copc'], intersects=bbox)
        # self._items = search.get_all_items()

    def _get_slippy_tiles(self, zoom: int) -> GeoDataFrame:
        gw, gs, ge, gn = self._bbox.bounds
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
        tiles = tiles[tiles.within(self._bbox)]
        tiles['polygon'] = tiles.geometry.to_wkt() + ' / EPSG:4326'
        return tiles

    def _get_copc(self, union: shapely.geometry.Polygon) -> GeoDataFrame:
        catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        search = catalog.search(collections=['3dep-lidar-copc'], intersects=union)
        signed: pystac.item_collection.ItemCollection = planetary_computer.sign(search)
        geometry = [
            shapely.geometry.shape(item.geometry)
            for item in signed
        ]
        index = np.fromiter((
            item.id
            for item in signed
        ), dtype='U256')
        href = np.fromiter((
            item.assets['data'].href
            for item in signed
        ), dtype='U1024')
        gdf = GeoDataFrame({
            'href': href,
        }, geometry=geometry, crs=4326, index=index)
        return gdf

    def load(self, zoom: int, max_height: float, dir: Optional[str] = None):
        tiles = self._get_slippy_tiles(zoom)
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
        if dir is None:
            dir = os.getcwd()
        dir = os.path.join(dir, zoom_)
        outdirs = [
            os.path.join(dir, xtile)
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
            threads.map(resize, tempfiles, outfiles)


def resize(path, dest):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    cv2.imwrite(dest, image)


if __name__ == '__main__':
    box = shapely.geometry.box(
        *(41.85520272864603, -87.64667114877022)[::-1],
        *(41.886776137830516, -87.62461266183428)[::-1],
    )
    LidarTiles(box).load(16, 450)
    print('done')
