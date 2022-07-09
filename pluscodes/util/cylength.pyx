import cython
from typing import Optional
from geopandas import GeoDataFrame, GeoSeries
from pandas import Series, DataFrame

from cpython cimport PyObject
cimport cython
from numpy.typing import NDArray
from typing import Union

import numpy as np
cimport numpy as np
import geopandas as gpd
# from libc.stdlib cimport malloc, free

from .pygeos cimport (
    PyGEOS_CreateGeometry,
    PyGEOS_GetGEOSGeometry,
    import_pygeos_c_api,
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
    double bw, bs, be, bn, px, py
    const GEOSPreparedGeometry *prepared


@cython.boundscheck(False)
@cython.wraparound(False)
def get_lengths(
        footprints: Union[GeoSeries, GeoDataFrame],
        bounds: Optional[NDArray[float]] = None,
        points: Optional[GeoSeries] = None,
) -> NDArray[np.uint8]:
    cdef :
        double[:] wv, sv, ev, nv, px, py
        size_t n, i
        np.ndarray[UINT8, ndim=1] lengths
        unsigned char[:] lv
        GEOSGeometry * geom = NULL
        const GEOSPreparedGeometry *prepared = NULL
        Footprint footprint
        char c
        GEOSContextHandle_t h
        object [:] objects

    lengths = np.ndarray((len(footprints),), dtype=np.uint8)
    lv = lengths
    objects = footprints.geometry.values.data

    if bounds is not None:
        bw, bs, be, bn = bounds
    else:
        bw, bs, be, bn = footprints.bounds.T.values
    if points is not None:
        px = points.x.values
        py = points.y.values
    else:
        points = footprints.representative_point().geometry
        px = points.x.values
        py = points.y.values

    for i in range(len(footprints)):
        c = PyGEOS_GetGEOSGeometry(<PyObject *> objects[i], &geom)
        if c == 0:
            raise ValueError("Could not get GEOSGeometry")
        prepared = GEOSPrepare(geom)

        footprint = Footprint(bw[i], bs[i], be[i], bn[i], px[i], py[i], prepared)
        lv[i] = Footprint_length(footprint)
        GEOSPreparedGeom_destroy(prepared)

    return lengths

cdef unsigned char Footprint_length(const Footprint footprint) nogil:
    cdef :
        unsigned char length = 0
        double dw, dh

    dw = footprint.be - footprint.bw
    dh = footprint.bn - footprint.bs

    while length <= GRID_LENGTH:
        if LAT_RESOLUTIONS[length] < dw and LON_RESOLUTIONS[length] < dh:
            break
        length += 1
    else:
        raise ValueError('footprint bounds too small')

    while length <= GRID_LENGTH:
        if Footprint_contained(footprint, length):
            break
        length += 1
    else:
        raise ValueError('footprint point not containable')

    return PAIR_LENGTH + length

cdef inline bint Footprint_contained(const Footprint footprint, unsigned char grid_length, ) nogil:
    cdef double w, s, n, e
    cdef GEOSGeometry *point
    cdef bint intersects

    w = <double> (footprint.px // TRIM_LONS[grid_length]) / FINAL_LON_PRECISIONS[grid_length] - MAX_LON
    s = <double> (footprint.py // TRIM_LATS[grid_length]) / FINAL_LAT_PRECISIONS[grid_length] - MAX_LAT
    e = w + LON_RESOLUTIONS[grid_length]
    n = s + LAT_RESOLUTIONS[grid_length]

    point = GEOSGeom_createPointFromXY(w, s)

    intersects = GEOSPreparedIntersects(footprint.prepared, point)
    GEOSGeom_destroy(point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY(e, s)
    intersects = GEOSPreparedIntersects(footprint.prepared, point)
    GEOSGeom_destroy(point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY(w, n)
    intersects = GEOSPreparedIntersects(footprint.prepared, point)
    GEOSGeom_destroy(point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY(e, n)
    intersects = GEOSPreparedIntersects(footprint.prepared, point)
    GEOSGeom_destroy(point)
    if not intersects:
        return False

    return True

