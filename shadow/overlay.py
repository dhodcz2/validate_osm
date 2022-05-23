import dask.dataframe as dd
import dask.array as da
import dask.bag as db
import dask_geopandas as dg

import itertools
import os

from cutil import load_cells, nums2degs

import pygeos.creation
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

# TODO: Load images given xtiles, ytiles, zoom into a GeoDataFrame
import concurrent.futures
import glob
from typing import Iterator

import numpy as np
import skimage.io


class Shadows:
    @classmethod
    def from_dir(cls, dir: str):
        # /zoom/xtile/ytile.png
        zoom = int(dir.rpartition('/')[2])
        paths = np.fromiter(
            glob.iglob(os.path.join(dir, '*/*')),
            dtype='U4096'
        )
        part = np.char.rpartition(paths, '/')
        # ytiles = np.ndarray.astype(part[:,2], np.uint32)
        ytiles = np.ndarray.astype(
            np.char.rpartition(part[:, 2], '.')[:, 0],
            np.uint64
        )
        xtiles = np.ndarray.astype(
            np.char.rpartition(part[:, 0], '/')[:, 2],
            np.uint64
        )

        def func(xtile, ytile, path):
            return xtile, ytile, skimage.io.imread(path)

        with concurrent.futures.ThreadPoolExecutor() as threads, concurrent.futures.ProcessPoolExecutor() as processes:
            # # TODO: is as_completed faster?
            # # images = [
            # #     threads.submit(func, x, y, p)
            # #     for x, y, p in zip(xtiles, ytiles, paths)
            # # ]
            # # cells = [
            # #     processes.submit(load_cells, arr, x, y, zoom)
            # #     for (x, y, arr) in concurrent.futures.as_completed(images)
            # # ]
            # # geometries = concurrent.futures.as_completed(cells)
            # # geometry = np.concatenate(geometries)
            #
            #
            # images = threads.map(skimage.io.imread, paths)
            # cells = processes.map(load_cells, images, xtiles, ytiles, itertools.repeat(zoom))
            #
            # results = list(cells)
            # geometries = itertools.chain.from_iterable(results[0::2])
            # values = itertools.chain.from_iterable(results[1::2])
            #
            # gdf = GeoDataFrame(
            #     data={
            #         'value': np.concatenate(values)
            #     },
            #     geometry=np.concatenate(geometries),
            #     crs=4326
            # )
            # # TODO: Instead of turning the map into a list of objects, can we get geometries, then values?
            # return gdf
            #

            # images: list[np.ndarray] = list(threads.map(skimage.io.imread, paths))
            # wheres: list[np.ndarray] = [
            #     np.where(image > 0)
            #     for image in images
            # ]
            # values: Iterator[np.ndarray] = [
            #     image[where]
            #     for where, image in zip(wheres, images)
            # ]
            # values = np.concatenate(values, dtype=np.uint16)
            # gw, gn = nums2degs(xtiles, ytiles, zoom, True)
            # ge, gs = nums2degs(xtiles + 1, ytiles + 1, zoom, True)
            # dh = gs - gn
            # dw = ge - gw
            # # TODO: Perhaps it is better to just concatenate all the arrays at once instead of coping
            # #   with iterators
            # cgn = (
            #     where[0] * h
            #     for where, h in zip(wheres, dh, gn)
            # )
            # cgw = (
            #     where[1] * w
            #     for where, w in zip(wheres, dw, gw)
            # )
            # cgs = (
            #     where[0] + 1 * h
            #     for where, h in zip(wheres, dh, gs)
            # )
            # cge = (
            #     te + (where[1] + 1 * w)
            #     for where, w, te in zip(wheres, dw, ge)
            # )
            # cgn = np.fromiter(itertools.chain.from_iterable(cgn), dtype=float)
            # cgw = np.fromiter(itertools.chain.from_iterable(cgw), dtype=float)
            # cgs = np.fromiter(itertools.chain.from_iterable(cgs), dtype=float)
            # cge = np.fromiter(itertools.chain.from_iterable(cge), dtype=float)
            #
            # geometry = pygeos.creation.box(cgw, cgs, cge, cgn)
            # gdf = GeoDataFrame({'value': values, }, geometry=geometry, crs=4326)
            # return gdf

            images: list[np.ndarray] = list(threads.map(skimage.io.imread, paths))
            images: np.ndarray = np.concatenate(images, dtype=np.uint16)
            where = np.where(images > 0)
            mod = np.mod(where, 255)
            gw, gn = nums2degs(xtiles, ytiles, zoom, True)
            ge, gs = nums2degs(xtiles + 1, ytiles + 1, zoom, True)
            dh = gs - gn
            dw = ge - gw

            cn = mod
            gn[mod] + (dh[mod] * )




if __name__ == '__main__':
    Shadows.from_dir('/home/arstneio/Downloads/15')
