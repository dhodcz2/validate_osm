import cython
import math

import pandas as pd
import pygeos.creation
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
from pygeos._geos cimport (
    get_geos_handle,
)
cimport util.cfuncs as cfuncs


from ._geos cimport (
    GEOSPreparedGeometry,
    GEOSGeometry,
    GEOSPrepare,
    GEOSContextHandle_t,

    GEOSPreparedIntersects_r,
    GEOSGeom_destroy_r,
    # get_geos_handle,
    GEOSPreparedGeom_destroy_r,
    GEOSGeom_createPointFromXY_r,
    GEOSPrepare_r,
    GEOS_init_r,

    GEOSPreparedIntersects,
    GEOSGeom_destroy,
    GEOSPreparedGeom_destroy,
    GEOSGeom_createPointFromXY,
    GEOSPrepare,


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
    unsigned char *accepted
    unsigned char *contained
    size_t xtiles, ytiles, xpoints, ypoints
    double *ftw
    double *fts
    unsigned long *ltw
    unsigned long *lts

cdef Footprint Footprint_init(
        GEOSGeometry *g,
        unsigned long bw,
        unsigned long bs,
        unsigned long be,
        unsigned long bn,
        size_t grid_length
):
    cdef :
        size_t dw, dh, size, i, j
        # unsigned long xstep = pow(GRID_COLUMNS, GRID_LENGTH - grid_length)
        # unsigned long ystep = pow(GRID_ROWS, GRID_LENGTH - grid_length)
        unsigned long xstep = XSTEPS[grid_length]
        unsigned long ystep = YSTEPS[grid_length]

        unsigned long trim_lon = TRIM_LONS[grid_length]
        unsigned long trim_lat = TRIM_LATS[grid_length]
        unsigned long final_lon_precision = FINAL_LON_PRECISIONS[grid_length]
        unsigned long final_lat_precision = FINAL_LAT_PRECISIONS[grid_length]

        unsigned long xtiles = (be // xstep) -  (bw // xstep)
        unsigned long ytiles = (bn // ystep) - (bs // ystep)

        unsigned long tiles = xtiles * ytiles

        unsigned long xpoints = xtiles + 1
        unsigned long ypoints = ytiles + 1
        unsigned long points = xpoints * ypoints

        double* ftw = <double *>malloc(xpoints * sizeof(double))
        double* fts = <double *>malloc(ypoints * sizeof(double))
        unsigned long* ltw = <unsigned long *>malloc(xpoints * sizeof(unsigned long))
        unsigned long* lts = <unsigned long *>malloc(ypoints * sizeof(unsigned long))

    for i in range(xpoints):
        ltw[i] = (bw + i * xstep + xstep) // trim_lon
        ftw[i] = <double> ltw[i] / final_lon_precision - MAX_LON

    for i in range(ypoints):
        lts[i] = (bs + i * ystep + ystep) // trim_lat
        fts[i] = <double> lts[i] / final_lat_precision - MAX_LAT

    cdef unsigned char* contained = <unsigned char*> malloc(points * sizeof(unsigned char))
    cdef unsigned char* accepted = <unsigned char *> malloc(points * sizeof(unsigned char))
    return Footprint(
        ltw=ltw,
        lts=lts,
        ftw=ftw,
        fts=fts,
        prepared=GEOSPrepare(g),
        accepted=accepted,
        contained=contained,
        xtiles=xtiles, ytiles=ytiles, xpoints=xpoints, ypoints=ypoints,
    )

cdef inline void Footprint_destroy(Footprint f):
    free(f.accepted)
    free(f.contained)
    free(f.ftw)
    free(f.fts)
    free(f.ltw)
    free(f.lts)
    GEOSPreparedGeom_destroy(f.prepared)

def get_list_tiles(
        gdf: Union[GeoDataFrame, GeoSeries],
        unsigned char[:] lengths,
        bounds: Optional[NDArray[float]] = None,
) -> list[np.ndarray]:
    cdef Py_ssize_t n, size, i
    cdef np.ndarray[F64, ndim=1] fw, fs, fe, fn,
    cdef unsigned long[:] vw, vs, ve, vn

    cdef GEOSGeometry *geom = NULL
    # cdef const GEOSPreparedGeometry *prepared = NULL
    cdef Footprint footprint
    cdef char c
    cdef GEOSContextHandle_t h

    gdf = gdf.to_crs(epsg=4326)
    points = gdf.representative_point().geometry
    size = len(gdf)
    cdef object [:] objects = gdf.geometry.values.data

    if bounds is not None:
        bw, bs, be, bn = bounds
    else:
        fw, fs, fe, fn = gdf.bounds.values.T
        vw = np.ndarray.astype((
            (fw + MAX_LON) * FINAL_LON_PRECISION
        ), dtype=np.uint64)
        vs = np.ndarray.astype((
            (fs + MAX_LAT) * FINAL_LAT_PRECISION
        ), dtype=np.uint64)
        ve = np.ndarray.astype((
            (fe + MAX_LON) * FINAL_LON_PRECISION
        ), dtype=np.uint64)
        vn = np.ndarray.astype((
            (fn + MAX_LAT) * FINAL_LAT_PRECISION
        ), dtype=np.uint64)

    list_tiles: list[NDArray[np.uint64]] = PyList_New(size)
    for i in range(size):
        c = PyGEOS_GetGEOSGeometry(<PyObject *> objects[i], &geom)
        if c == 0:
            raise ValueError("Could not get GEOS geometry")
        footprint = Footprint_init(
            g=geom,
            bw=vw[i],
            bs=vs[i],
            be=ve[i],
            bn=vn[i],
            grid_length=lengths[i] - PAIR_LENGTH,
        )
        tiles = Footprint_getArray(footprint)
        Py_INCREF(tiles)
        PyList_SET_ITEM(list_tiles, i, tiles)
        Footprint_destroy(footprint)

    print('return list_tiles')
    return list_tiles


cdef np.ndarray Footprint_getArray(const Footprint f):
    cdef :
        size_t r, c, k, n, nout
        GEOSGeometry *point


    for c in range(f.xpoints):
        for r in range(f.ypoints):
            k = c * f.ypoints + r
            point = GEOSGeom_createPointFromXY(f.ftw[c], f.fts[r])
            intersects = GEOSPreparedIntersects(f.prepared, point)
            GEOSGeom_destroy(point)
            f.contained[k] = intersects

    nout = 0
    for c in range(f.xtiles):
        for r in range(f.ytiles):
            k = c * f.ypoints + r
            if (
                f.contained[k] and
                f.contained[c * f.ypoints + r + 1] and
                f.contained[(c + 1) * f.ypoints + r] and
                f.contained[(c + 1) * f.ypoints + r + 1]
            ):
                nout += 1
                f.accepted[k] = 1

    cdef np.ndarray[UINT64, ndim=2] out = np.ndarray((nout, 2), dtype=np.uint64)

    cdef unsigned long[:, :] ov  = out
    n = 0
    for c in range(f.xtiles):
        for r in range(f.ytiles):
            k = c * f.ypoints + r
            if f.accepted[k]:
                ov[n, 0] = f.ltw[r]
                ov[n, 1] = f.lts[c]
                n += 1
    assert n == nout

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
    print('concatenate')

    sizes = <size_t *> malloc(len_gdf * sizeof(size_t))
    count = 0
    for i in range(len_gdf):
        sizes[i] = list_longs[i].shape[0]
        count += sizes[i]

    print('count', count)

    iloc = np.ndarray((count,), dtype=np.uint64)
    viloc = iloc

    n = 0
    for i in range(len_gdf):
        for j in range(sizes[i]):
            viloc[n] = i
            n += 1
    print('len(iloc)', len(iloc))

    free(sizes)

    lengths = lengths[iloc]
    print('len(lengths)', len(lengths))
    strings = cfuncs.get_strings(longs[:, 0], longs[:, 1], lengths)
    print('get_strings')
    strings = strings[iloc]
    index = pd.MultiIndex.from_arrays((
        iloc, strings,
    ), names=('index', 'tile'),)
    print('len(index)', len(index))

    bounds = cfuncs.get_bounds(longs[:, 0], longs[:, 1], lengths)
    data = pygeos.creation.box( bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3] )

    result = GeoSeries(data=data, index=index, crs=4326)
    print('return get_geoseries_tiles')
    return result