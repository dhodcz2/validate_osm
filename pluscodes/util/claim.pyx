import cython
import math
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
    GEOSPreparedGeometry *prepared
    unsigned char * accepted, *contained
    # double * tw,ts
    size_t xtiles, ytiles, xpoints, ypoints
    double *ftw, *fts
    unsigned long *ltw, *lts

cdef Footprint footprint_init(
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

    print('xpoints:', xpoints)
    for i in range(xpoints):
        ltw[i] = (bw + i * xstep + xstep) // trim_lon
        ftw[i] = <double> ltw[i] / final_lon_precision - MAX_LON
        # print(round(ftw[i], 2), end=" ",)

    print('ypoints:', ypoints)
    for i in range(ypoints):
        lts[i] = (bs + i * ystep + ystep) // trim_lat
        fts[i] = <double> lts[i] / final_lat_precision - MAX_LAT
        # print(round(fts[i], 2), end=" ",)

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

cdef inline void footprint_destroy(Footprint f):
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
) -> list[NDArray[np.uint64]]:
    cdef Py_ssize_t n, size, i
    cdef np.ndarray[F64, ndim=1] fw, fs, fe, fn,
    cdef unsigned long[:] vw, vs, ve, vn

    cdef GEOSGeometry *geom = NULL
    cdef const GEOSPreparedGeometry *prepared = NULL
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
        # fw, fs, fn, fe = gdf.bounds.T.values
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
        print(fw[i], fs[i], fe[i], fn[i])
        # print(f'[%d, %d, %d, %d]' % (fw[i], fs[i], fe[i], fn[i]))

    list_tiles: list[NDArray[np.uint64]] = PyList_New(size)
    for i in range(size):
        c = PyGEOS_GetGEOSGeometry(<PyObject *> objects[i], &geom)
        if c == 0:
            raise ValueError("Could not get GEOS geometry")
        footprint = footprint_init(
            g=geom,
            bw=vw[i],
            bs=vs[i],
            be=ve[i],
            bn=vn[i],
            grid_length=lengths[i] - PAIR_LENGTH,
        )
        tiles = get_tiles(footprint)
        Py_INCREF(tiles)
        PyList_SET_ITEM(list_tiles, i, tiles)
        footprint_destroy(footprint)
        return list_tiles
    print('return list_tiles')
    return list_tiles


cdef np.ndarray get_tiles(const Footprint f):
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
            if (
                f.contained[c * f.ypoints + r] and
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




