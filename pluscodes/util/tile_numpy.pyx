import itertools

from cython cimport view
import pygeos.geometry
import cython
from libc.string cimport strlen, strcpy
import math

import pandas as pd
import pygeos.creation
import shapely.geometry.base
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from typing import Optional
import numpy
from geopandas import GeoDataFrame, GeoSeries
from typing import Union

import numpy as np
cimport numpy as np
cimport util.cfuncs as cfuncs


from cpython cimport (
    PyList_SET_ITEM,
    PyList_New,
)

cdef extern from '<util/globals.h>':
    const char* ALPHABET
    const size_t SEP_POS
    const size_t BASE
    const size_t MAX_DIGITS
    const size_t PAIR_LENGTH
    const size_t GRID_LENGTH
    const size_t GRID_COLUMNS
    const size_t GRID_ROWS
    const size_t MIN_TRIMMABLE_CODE_LEN
    const int MAX_LAT
    const int MAX_LON
    const double GRID_SIZE_DEGREES
    unsigned long PAIR_PRECISION
    const size_t L
    unsigned long FINAL_LAT_PRECISION
    unsigned long FINAL_LON_PRECISION


ctypedef np.uint8_t UINT8
ctypedef np.uint64_t UINT64
ctypedef np.float64_t F64
ctypedef np.uint8_t BOOL

cdef :
    unsigned long[6] FINAL_LON_PRECISIONS
    unsigned long[6] FINAL_LAT_PRECISIONS
    unsigned long[6] TRIM_LONS
    unsigned long[6] TRIM_LATS
    double[6] LAT_RESOLUTIONS
    double[6] LON_RESOLUTIONS
    unsigned long[6] XSTEPS
    unsigned long[6] YSTEPS
    size_t n

for n in range(6):
    FINAL_LAT_PRECISIONS[n] = PAIR_PRECISION * pow(GRID_ROWS, n)
    FINAL_LON_PRECISIONS[n] = PAIR_PRECISION * pow(GRID_COLUMNS, n)
    TRIM_LATS[n] = pow(<unsigned long> GRID_ROWS, GRID_LENGTH - n)
    TRIM_LONS[n] = pow(<unsigned long> GRID_COLUMNS, GRID_LENGTH - n)
    LAT_RESOLUTIONS[n] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_ROWS, n)
    LON_RESOLUTIONS[n] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_COLUMNS, n)
    XSTEPS[n] = pow(GRID_COLUMNS, GRID_LENGTH - n)
    YSTEPS[n] = pow(GRID_ROWS, GRID_LENGTH - n)


def get_geoseries_tiles(
        gdf: Union[GeoDataFrame, GeoSeries],
        np.ndarray[UINT8, ndim=1] code_lengths,
        bounds: Optional[np.ndarray] = None,
) -> GeoSeries:
    cdef :
        size_t n, size, i, count, len_gdf, j
        size_t *sizes
        np.ndarray[UINT64, ndim=1] repeat

    len_gdf = len(gdf)
    list_longs = get_list_tiles(gdf, code_lengths, bounds)
    data = np.concatenate(list_longs, axis=0)

    repeat = np.ndarray(shape=(len_gdf,), dtype=np.uint64)
    for i in range(len_gdf):
        repeat[i] = len(list_longs[i])

    code_lengths = np.repeat(code_lengths, repeat)
    strings = cfuncs.get_strings(data[:, 0], data[:, 1], code_lengths)
    strings = np.repeat(strings, repeat)
    index = pd.MultiIndex.from_arrays((repeat, strings), names=('iloc', 'tile'))

    return GeoSeries(data=data, index=index, crs=4326)

cdef list get_list_tiles(
    gdf: Union[GeoDataFrame, GeoSeries],
    np.ndarray[UINT8, ndim=1] code_lengths,
    bounds: Optional[np.ndarray] = None,
):
    cdef :
        size_t size, i
        np.ndarray[UINT64, ndim = 2] tiles

    assert gdf.crs == 4326
    points = gdf.representative_point().geometry
    size = len(gdf)
    objects = gdf.geometry.values.data
    list_tiles = PyList_New(size)


    if bounds is not None:
        bw, bs, be, bn = bounds
    else:
        fw, fs, fe, fn = gdf.bounds.values.T

        w_view = np.ndarray.astype((
                (fw + MAX_LON) * FINAL_LON_PRECISION
        ), dtype=np.uint64)
        s_view = np.ndarray.astype((
                (fs + MAX_LAT) * FINAL_LAT_PRECISION
        ), dtype=np.uint64)
        e_view = np.ndarray.astype((
                (fe + MAX_LON) * FINAL_LON_PRECISION
        ), dtype=np.uint64)
        n_view = np.ndarray.astype((
                (fn + MAX_LAT) * FINAL_LAT_PRECISION
        ), dtype=np.uint64)

    for i in range(size):
        tiles = get_tiles(
            geometry=objects[i],
            w_bound=w_view[i],
            s_bound=s_view[i],
            e_bound=e_view[i],
            n_bound=n_view[i],
            grid_length=code_lengths[i] - PAIR_LENGTH,
        )
        Py_INCREF(tiles)
        PyList_SET_ITEM(list_tiles, i, tiles)

    return list_tiles

cdef np.ndarray[UINT64, ndim=2] get_tiles(
        # object geometry,
        geometry: pygeos.geometry.Geometry,
        unsigned long w_bound,
        unsigned long s_bound,
        unsigned long e_bound,
        unsigned long n_bound,
        size_t grid_length,
):
    cdef :
        size_t i, j, xtiles, ytiles, xpoints, ypoints, n
        unsigned long xstep, ystep, trim_lon, trim_lat, final_lon_precision, final_lat_precision
        double[:] fw, fs, fw_broad, fs_broad
        unsigned long[:] lw, ls
        unsigned long[:, :] out
        np.ndarray[object, ndim=1] points
        np.ndarray[UINT8, ndim=2] intersects

    xstep = XSTEPS[grid_length]
    ystep = YSTEPS[grid_length]

    trim_lon = TRIM_LONS[grid_length]
    trim_lat = TRIM_LATS[grid_length]
    final_lon_precision = FINAL_LON_PRECISIONS[grid_length]
    final_lat_precision = FINAL_LAT_PRECISIONS[grid_length]

    xtiles = (e_bound // xstep) - (w_bound // xstep)
    ytiles = (n_bound // ystep) - (s_bound // ystep)

    xpoints = xtiles + 1
    ypoints = ytiles + 1

    w_bound += xstep
    s_bound += ystep
    e_bound += xstep
    n_bound += ystep

    lw = view.array(shape=(xpoints,), itemsize=sizeof(unsigned long))
    ls = view.array(shape=(ypoints,), itemsize=sizeof(unsigned long))
    fw = view.array(shape=(xpoints,), itemsize=sizeof(double))
    fs = view.array(shape=(ypoints,), itemsize=sizeof(double))

    for i in range(xpoints):
        lw[i] = w_bound + (i * xstep)
        fw[i] = <double>(lw[i] // trim_lon) / final_lon_precision - MAX_LON

    for i in range(ypoints):
        ls[i] = s_bound + (i * ystep)
        fs[i] = <double>(ls[i] // trim_lat) / final_lat_precision - MAX_LAT

    fw_broad = view.array(shape=(xpoints*ypoints,), itemsize=sizeof(double))
    fs_broad = view.array(shape=(xpoints*ypoints,), itemsize=sizeof(double))
    for i in range(xpoints):
        for j in range(ypoints):
            fw_broad[i * ypoints + j] = fw[i]
            fs_broad[i * ypoints + j] = fs[j]

    # TODO: If only we had a stable C API for PyGEOS, we could iterate and instantiate ephemeral Point objects
    #   instead of needing to create a numpy array of PyGEOS objects.
    points = pygeos.creation.points( np.asarray(fw_broad), np.asarray(fs_broad), )
    intersects = (
        pygeos.intersects(points, geometry)
        .reshape(xpoints, ypoints)
    )
    intersects = (
        intersects[:xtiles, :ytiles]
        & intersects[1:xpoints, :ytiles]
        & intersects[1:xpoints, 1:ypoints]
        & intersects[:xtiles, 1:ypoints]
    )

    out = view.array(shape=(intersects.sum(),2), itemsize=sizeof(unsigned long))
    n = 0
    for i in range(xtiles):
        for j in range(ytiles):
            if intersects[i, j]:
                out[n, 0] = lw[i]
                out[n, 1] = ls[j]
                n += 1

    return np.asarray(out)

