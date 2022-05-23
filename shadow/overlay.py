if True:
    from cutil import nums2degs
import concurrent.futures
import glob
import os

import dask.array as da
import dask.dataframe as dd
import dask_geopandas as dgpd
import numpy as np
import pygeos.creation
import skimage.io


def get_shadows_from_dir(dir: str):
    # /zoom/xtile/ytile.png
    zoom = int(dir.rpartition('/')[2])
    paths = np.fromiter(
        glob.iglob(os.path.join(dir, '*/*')),
        dtype='U4096'
    )
    part = np.char.rpartition(paths, '/')
    ytiles = np.ndarray.astype(
        np.char.rpartition(part[:, 2], '.')[:, 0],
        np.uint64
    )
    xtiles = np.ndarray.astype(
        np.char.rpartition(part[:, 0], '/')[:, 2],
        np.uint64
    )

    images: list[np.ndarray] = list(concurrent.futures.ThreadPoolExecutor().map(skimage.io.imread, paths))
    images: np.ndarray = np.concatenate(images, dtype=np.uint16)
    where = np.where(images > 0)
    gw, gn = nums2degs(xtiles, ytiles, zoom, True)
    ge, gs = nums2degs(xtiles + 1, ytiles + 1, zoom, True)
    dh = gs - gn
    dw = ge - gw

    cn = np.mod(where[0], 256)
    cw = where[1]
    tile = where[0] // 256
    tgn = ytiles[tile]
    tgw = xtiles[tile]
    cgn = tgn + (cn * dh[tile])
    cgw = tgw + (cw * dw[tile])
    cgs = tgn + (cn + 1 * dh[tile])
    cge = tgw + (cw + 1 * dw[tile])
    # 100 MB per chunk, with each object taking 32 bytes, and each reference taking 8 bytes
    chunksize = 100 * 1024 * 1024 / (32 + 8)
    cgn = da.from_array(cgn, chunks=chunksize)
    cgw = da.from_array(cgw, chunks=chunksize)
    cgs = da.from_array(cgs, chunks=chunksize)
    cge = da.from_array(cge, chunks=chunksize)
    geometry = da.map_blocks(pygeos.creation.box, cgw, cgs, cge, cgn)
    geometry = dd.from_dask_array(geometry, 'geometry')
    values = da.from_array(images[where], chunks=chunksize)
    values = dd.from_dask_array(values, 'value')
    cells = dd.concat([geometry, values], axis=1)
    cells = dgpd.from_dask_dataframe(cells)
    return cells
