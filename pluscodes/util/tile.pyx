import cython
import math

import pandas as pd
import pygeos.creation
import shapely.geometry.base
from cpython.object cimport PyObject
from cpython.ref cimport Py_INCREF
from typing import Optional
import numpy
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

from cpython cimport PyObject
cimport cython
from numpy.typing import NDArray
from typing import Union

import numpy as np
cimport numpy as np
import geopandas as gpd
from libc.stdlib cimport malloc, free

from .pygeos cimport (
    PyGEOS_CreateGeometry,
    PyGEOS_GetGEOSGeometry,
    import_pygeos_c_api,
)
cimport util.cfuncs as cfuncs


from ._geos cimport (
    GEOSPreparedGeometry,
    GEOSGeometry,
    GEOSPrepare,
    GEOSContextHandle_t,

    GEOSPreparedIntersects_r,
    GEOSPreparedContains,
    GEOSGeom_destroy_r,
    # get_geos_handle,
    GEOSPreparedGeom_destroy_r,
    GEOSGeom_createPointFromXY_r,
    GEOSPrepare_r,
    GEOS_init_r,

    GEOSPreparedIntersects,
GEOSPreparedDisjoint,
    GEOSGeom_destroy,
    GEOSPreparedGeom_destroy,
    GEOSGeom_createPointFromXY,
    GEOSPrepare,

GEOSIntersects
)

from cpython cimport (
    PyBytes_AsStringAndSize,
    PyList_SET_ITEM,
    PyList_New,
)

cdef extern from 'Python.h':
    PyObject* Py_BuildValue(const char*, ...) except NULL

import_pygeos_c_api()
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

# TODO: hardcoding 6 has a bad smell, is there a proper way to handle this such that each call to get_lengths
#   mustn't define the arrays at a local level?
# cdef unsigned long[6] FINAL_LON_PRECISIONS
# cdef unsigned long[6] FINAL_LAT_PRECISIONS
# cdef unsigned long[6] TRIM_LONS
# cdef unsigned long[6] TRIM_LATS
# cdef double[6] LAT_RESOLUTIONS
# cdef double[6] LON_RESOLUTIONS
#
# cdef size_t n
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

cdef struct Footprint:
    const GEOSPreparedGeometry *prepared
    GEOSGeometry *geom
    PyObject *obj
    unsigned char *accepted
    unsigned char *contained
    size_t xtiles, ytiles, xpoints, ypoints
    double *fw
    double *fs
    unsigned long *lw
    unsigned long *ls

cdef Footprint Footprint_init(
        GEOSGeometry *g,
        PyObject *obj,
        unsigned long w_bound,
        unsigned long s_bound,
        unsigned long e_bound,
        unsigned long n_bound,
        size_t grid_length
):
    cdef :
        size_t dw, dh, size, i, j, xtiles, ytiles, tiles, xpoints, ypoints, points
        unsigned long xstep, ystep, trim_lon, trim_lat, final_lon_precision, final_lat_precision
        double *fw
        double *fs
        unsigned long *lw
        unsigned long *ls
        unsigned char * contained
        unsigned char * accepted

    xstep = XSTEPS[grid_length]
    ystep = YSTEPS[grid_length]

    trim_lon = TRIM_LONS[grid_length]
    trim_lat = TRIM_LATS[grid_length]
    final_lon_precision = FINAL_LON_PRECISIONS[grid_length]
    final_lat_precision = FINAL_LAT_PRECISIONS[grid_length]

    xtiles = (e_bound // xstep) - (w_bound // xstep)
    ytiles = (n_bound // ystep) - (s_bound // ystep)

    tiles = xtiles * ytiles

    xpoints = xtiles + 1
    ypoints = ytiles + 1
    points = xpoints * ypoints

    contained = <unsigned char *> malloc(points * sizeof(unsigned char))
    accepted = <unsigned char *> malloc(points * sizeof(unsigned char))
    fw = <double *>malloc(xpoints * sizeof(double))
    fs = <double *>malloc(ypoints * sizeof(double))
    lw = <unsigned long *>malloc(xpoints * sizeof(unsigned long))
    ls = <unsigned long *>malloc(ypoints * sizeof(unsigned long))

    fw_ = set()
    lw_ = set()
    ls_ = set()
    fs_ = set()

    for i in range(xpoints):
        lw[i] = w_bound + i * xstep + xstep
        fw[i] = <double>(lw[i] // trim_lon) / final_lon_precision - MAX_LON

        fw_.add(fw[i])
        lw_.add(lw[i])
        # print('\t', fw[i], end=' ')
    # print()
    for i in range(ypoints):
        ls[i] = s_bound + i * ystep + ystep
        fs[i] = <double>(ls[i] // trim_lat) / final_lat_precision - MAX_LAT

        ls_.add(ls[i])
        fs_.add(fs[i])
        # print('\t', fs[i], end=' ')
    # print()

    assert len(lw_) == xpoints
    assert len(ls_) == ypoints
    assert len(fw_) == xpoints
    assert len(fs_) == ypoints



    return Footprint(
        prepared=GEOSPrepare(g),
        geom=g,
        obj=obj,
        accepted=accepted,
        contained=contained,
        xtiles=xtiles, ytiles=ytiles, xpoints=xpoints, ypoints=ypoints,
        fw=fw, fs=fs, lw=lw, ls=ls
    )

cdef inline void Footprint_destroy(Footprint f):
    free(f.accepted)
    free(f.contained)
    free(f.fw)
    free(f.fs)
    free(f.lw)
    free(f.ls)
    GEOSPreparedGeom_destroy(f.prepared)

# noinspection PyTypeChecker
def get_list_tiles(
        gdf: Union[GeoDataFrame, GeoSeries],
        unsigned char[:] lengths,
        bounds: Optional[NDArray[float]] = None,
) -> list[np.ndarray]:
    cdef :
        size_t n, size, i
        np.ndarray[F64, ndim=1] fw, fs, fe, fn
        unsigned long[:] w_view, s_view, e_view, n_view
        GEOSGeometry *geom = NULL
        Footprint footprint
        char c
        object [:] objects

    assert gdf.crs == 4326
    points = gdf.representative_point().geometry
    size = len(gdf)
    objects = gdf.geometry.values.data

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

    list_tiles: list[NDArray[np.uint64]] = PyList_New(size)
    for i in range(size):
        c = PyGEOS_GetGEOSGeometry(<PyObject *> objects[i], &geom)
        if c == 0:
            raise ValueError("Could not get GEOS geometry")
        obj = <PyObject *> objects[i]
        # print(f'[{fw[i]} {fe[i]}]', f'[{fs[i]} {fn[i]}]')
        # print(f'[{fs[i]} {fn[i]}]')
        footprint = Footprint_init(
            g=geom,
            obj=obj,
            w_bound=w_view[i],
            s_bound=s_view[i],
            e_bound=e_view[i],
            n_bound=n_view[i],
            grid_length=lengths[i] - PAIR_LENGTH,
        )
        print(objects[i])

        tiles = Footprint_getArray(footprint)
        Py_INCREF(tiles)
        PyList_SET_ITEM(list_tiles, i, tiles)
        Footprint_destroy(footprint)

    # print('return list_tiles')
    return list_tiles


cdef np.ndarray Footprint_getArray(const Footprint f):
    cdef :
        size_t r, c, k, n, nout
        GEOSGeometry *point
        # bint intersects
        char intersects
        unsigned long[:, :] out_view


    for c in range(f.xpoints):
        for r in range(f.ypoints):
            # TODO: For some reason, GEOSIntersects is always returning True

            k = c * f.ypoints + r
            point = GEOSGeom_createPointFromXY(f.fw[c], f.fs[r])
            # intersects = GEOSPreparedIntersects(f.geom, point)
            intersects = GEOSIntersects(f.geom, point)
            if intersects == 2:
                print(f'\tfw: {f.fw[c]}, fs: {f.fs[r]}')
                raise ValueError("intersects == 2")
            GEOSGeom_destroy(point)
            # print(f'\tintersects: {intersects}')

            f.contained[k] = intersects

            # obj: shapely.geometry.base.BaseGeometry = <object> f.obj
            # point_ = pygeos.Geometry(f'POINT({f.fw[c]} {f.fs[r]})')
            # intersects_ = pygeos.intersects(obj, point_)
            # # print(f'\tintersects: {intersects_}')
            #
            #
            # point = GEOSGeom_createPointFromXY(f.fw[c]*2, f.fs[r]*2)
            # intersects = GEOSIntersects(f.geom, point)
            # GEOSGeom_destroy(point)
            # print(f'\tdummy intersects: {intersects}')

            if intersects != 2:
                raise ValueError("Intersects is not 2")


    nout = 0
    for c in range(f.xtiles):
        for r in range(f.ytiles):
            k = c * f.ypoints + r
            if (
                f.contained[k]
                    and f.contained[c * f.ypoints + r + 1]
                    and f.contained[(c + 1) * f.ypoints + r]
                    and f.contained[(c + 1) * f.ypoints + r + 1]
            ):
                nout += 1
                f.accepted[k] = 1

    out = np.ndarray((nout, 2), dtype=np.uint64)

    out_view  = out
    n = 0
    for c in range(f.xtiles):
        for r in range(f.ytiles):
            k = c * f.ypoints + r
            if f.accepted[k]:
                # out_view[n, 0] = f.lw[r]
                # out_view[n, 1] = f.ls[c]
                out_view[n, 0] = f.lw[c]
                out_view[n, 1] = f.ls[r]
                n += 1
    assert n == nout

    try:
        assert set(
            (out[i, 0], out[i, 1])
            for i in range(nout)
        ).__len__() == nout
    except AssertionError:
        print(f'xtiles: {f.xtiles}', f'ytiles: {f.ytiles}', f'xpoints: {f.xpoints}', f'ypoints: {f.ypoints}')

        raise

    return out


def get_geoseries_tiles(
        gdf: Union[GeoDataFrame, GeoSeries],
        # unsigned char[:] lengths,
        np.ndarray[UINT8, ndim=1] lengths,
        bounds: Optional[NDArray[float]] = None,
) -> GeoSeries:
    cdef :
        size_t n, size, i, count, len_gdf, j
        size_t *sizes
        np.ndarray[UINT64, ndim=1] iloc
        unsigned long[:] viloc
        unsigned char[:] lengths_view = lengths

    len_gdf = len(gdf)

    list_longs: list[np.ndarray] = get_list_tiles(gdf, lengths_view, bounds)
    longs = np.concatenate(list_longs, axis=0)
    # print('concatenate')

    sizes = <size_t *> malloc(len_gdf * sizeof(size_t))
    count = 0
    for i in range(len_gdf):
        sizes[i] = list_longs[i].shape[0]
        count += sizes[i]


    iloc = np.ndarray((count,), dtype=np.uint64)
    viloc = iloc

    n = 0
    for i in range(len_gdf):
        for j in range(sizes[i]):
            viloc[n] = i
            n += 1

    free(sizes)

    # assert set(
    #     (longs[i, 0], longs[i, 1])
    #     for i in range(count)
    # ).__len__() == count

    lengths = lengths[iloc]
    strings = cfuncs.get_strings(longs[:, 0], longs[:, 1], lengths)
    strings = strings[iloc]
    index = pd.MultiIndex.from_arrays((
        iloc, strings,
    ), names=('index', 'tile'),)

    # assert set(
    #     strings[i]
    #     for i in range(count)
    # ).__len__() == count

    bounds = cfuncs.get_bounds(longs[:, 0], longs[:, 1], lengths)
    data = pygeos.creation.box( bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3] )

    result = GeoSeries(data=data, index=index, crs=4326)
    # print('return get_geoseries_tiles')
    return result
