import cython
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



ctypedef np.uint8_t UINT8
ctypedef np.float64_t F64
ctypedef np.uint8_t BOOL

# TODO: hardcoding 6 has a bad smell, is there a proper way to handle this such that each call to get_lengths
#   mustn't define the arrays at a local level?
cdef unsigned long[6] FINAL_LON_PRECISIONS
cdef unsigned long[6] FINAL_LAT_PRECISIONS
cdef unsigned long[6] TRIM_LONS
cdef unsigned long[6] TRIM_LATS
cdef double[6] LAT_RESOLUTIONS
cdef double[6] LON_RESOLUTIONS

cdef size_t n

for n in range(6):
    FINAL_LAT_PRECISIONS[n] = PAIR_PRECISION * pow(GRID_ROWS, n)
    FINAL_LON_PRECISIONS[n] = PAIR_PRECISION * pow(GRID_COLUMNS, n)
    TRIM_LATS[n] = pow(<unsigned long> GRID_ROWS, GRID_LENGTH - n)
    TRIM_LONS[n] = pow(<unsigned long> GRID_COLUMNS, GRID_LENGTH - n)
    LAT_RESOLUTIONS[n] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_ROWS, n)
    LON_RESOLUTIONS[n] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_COLUMNS, n)

cdef struct Footprint:
    GEOSPreparedGeometry *prepared
    unsigned char * accepted, contained, visited
    double * w,s
    size_t xtiles, ytiles, xpoints, ypoints

cdef Footprint footprint_init(GEOSGeometry *g, double w, double s, double e, double n, size_t length):
    cdef size_t dw, dh, size, i, j
    cdef unsigned long xstep = pow(GRID_COLUMNS, MAX_DIGITS - length)
    cdef unsigned long ystep = pow(GRID_ROWS, MAX_DIGITS - length)

    cdef unsigned long trim_lon = TRIM_LONS[length]
    cdef unsigned long trim_lat = TRIM_LATS[length]
    cdef unsigned long final_lon_precision = FINAL_LON_PRECISIONS[length]
    cdef unsigned long final_lat_precision = FINAL_LAT_PRECISIONS[length]


    cdef unsigned long xtiles = footprint.e // xstep - footprint.w // xstep
    cdef unsigned long ytiles = footprint.n // ystep - footprint.s // ystep
    cdef unsigned long tiles = xtiles * ytiles

    cdef unsigned long xpoints = xtiles + 2
    cdef unsigned long ypoints = ytiles + 2
    cdef unsigned long points = xpoints * ypoints

    cdef double* w = malloc(xtiles * sizeof(double))
    cdef double* s = malloc(ytiles * sizeof(double))

    for i in range(xpoints):
        w[i] = <double> (
                footprint.w + i * xstep // trim_lon
        ) / final_lon_precision - MAX_LON
    for i in range(ypoints):
        s[i] = <double> (
                footprint.s + i * ystep // trim_lat
        ) / final_lat_precision - MAX_LAT

    cdef unsigned char* contained = malloc(points * sizeof(unsigned char))
    cdef unsigned char* accepted = malloc(points * sizeof(unsigned char))
    cdef unsigned char* visited = malloc(points * sizeof(unsigned char))

    return Footprint(
        w=w, s=s,
        prepared=GEOSPrepare(g),
        accepted=accepted,
        contained=contained,
        visited=visited,
        xtiles=xtiles, ytiles=ytiles, xpoints=xpoints, ypoints=ypoints,
    )

cdef void footprint_destroy(Footprint &f):
    free(f.visited)
    free(f.accepted)
    free(f.contained)
    free(f.w)
    free(f.s)
    GEOSPreparedGeom_destroy(f.prepared)

def get_list_tiles(
        gdf: Union[GeoDataFrame, GeoSeries],
        unsigned char[:] lengths,
        bounds: Optional[NDArray[float]] = None,
) -> NDArray[np.uint64]:
    cdef Py_ssize_t n, size, i
    cdef NDArray[F64, ndim=1] fw, fs, fe, fn,
    cdef double[:] vw, vs, ve, vn

    cdef GEOSGeometry *geom = NULL
    cdef const GEOSPreparedGeometry *prepared = NULL
    cdef Footprint footprint
    cdef char c
    cdef GEOSContextHandle_t h

    gdf = gdf.to_crs(epsg=4326)
    points = gdf.representative_point().geometry
    size = len(gdf)

    if bounds is not None:
        bw, bs, be, bn = bounds
    else:
        fw, fs, fn, fe = gdf.bounds.T.values
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


    list_tiles = PyList_New(size)
    for i in range(size):
        geom = PyGEOS_CreateGeometry(points.values[i])
        footprint = footprint_init(geom, vw[i], vs[i], ve[i], vn[i], lengths[i])
        PyList_SET_ITEM(list_tiles, i, get_tile(footprint, lengths[i]))
        footprint_destroy(footprint)

    return list_tiles

# cdef np.ndarray[UINT64, ndim=2] get_tiles( Footprint footprint, ) nogil:
#
#     cdef size_t nout = 0
#     for i in range(1, ytiles):
#         for j in range(1, xtiles):
#             k = i * xcount + j
#             nout += accept(k, contained, visited, accepted)
#
#     cdef np.ndarray[UINT64, ndim=2] out = np.ndarray((nout, 2), dtype=np.uint64)
#     cdef size_t out_i = 0
#     for i in range(1, ytiles):
#         for j in range(1, xtiles):
#             k = i * xcount + j
#             if contained[k]:
#                 out[out_i, 0] = w[i]
#                 out[out_i, 1] = s[j]
#                 out_i += 1
#     assert out_i == nout
#
#     free(lx)
#     free(ly)
#
# cdef inline char accept(
#         Footprint footprint,
#         unsigned long k,
# ) nogil:
#     ...

cdef np.ndarray[UINT64, ndim=2] get_tiles(Footprint footprint) nogil:
    cdef size_t nout = 0
    cdef size_t i, j, k

