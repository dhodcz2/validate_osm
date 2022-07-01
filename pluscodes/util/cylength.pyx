import cython
from cpython cimport PyObject
cimport cython
from numpy.typing import NDArray
from typing import Union

import numpy as np
cimport numpy as np
import geopandas as gpd
from .pygeos cimport *

cdef extern from '<util/globals.h>':
    const char SEP = '+';
    const char *ALPHABET = "23456789CFGHJMPQRVWX";
    const char PAD

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

    const int DEBUG = 5;

    const unsigned long PAIR_PRECISION_FIRST_VALUE
    const unsigned long PAIR_PRECISION
    const unsigned long GRID_LAT_FIRST_PLACE_VALUE
    const unsigned long GRID_LON_FIRST_PLACE_VALUE
    const unsigned long FINAL_LAT_PRECISION
    const unsigned long FINAL_LON_PRECISION

    const double GRID_LAT_RESOLUTION
    const double GRID_LON_RESOLUTION

    const size_t L
    # const size_t L = GRID_LENGTH + 1;

ctypedef np.uint8_t UINT8
ctypedef np.float64_t F64
ctypedef np.uint8_t BOOL
# cdef const size_t L = GRID_LENGTH + 1


cdef unsigned long[L] FINAL_LON_PRECISIONS
cdef unsigned long[L] FINAL_LAT_PRECISIONS
cdef unsigned long[L] TRIM_LONS
cdef unsigned long[L] TRIM_LATS
cdef double[L] LAT_RESOLUTIONS
cdef double[L] LON_RESOLUTIONS

cdef size_t n
for n in range(GRID_LENGTH + 1):
    FINAL_LAT_PRECISIONS[n] = PAIR_PRECISION * pow(GRID_ROWS, n)
    FINAL_LON_PRECISIONS[n] = PAIR_PRECISION * pow(GRID_COLUMNS, n)
    TRIM_LATS[n] = pow(<unsigned long> GRID_ROWS, GRID_LENGTH - n)
    TRIM_LONS[n] = pow(<unsigned long> GRID_COLUMNS, GRID_LENGTH - n)
    LAT_RESOLUTIONS[n] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_ROWS, n)
    LON_RESOLUTIONS[n] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_COLUMNS, n)

from ._geos cimport *

import_pygeos_c_api()

cdef struct Footprint:
    double bw, bs, be, bn, px, py
    const GEOSPreparedGeometry *geom

cdef inline bint contained(const Footprint footprint, unsigned char grid_length, GEOSContextHandle_t handle) nogil:
    cdef double w, s, n, e
    cdef GEOSGeometry *point
    cdef bint intersects

    w = <double> (footprint.px // TRIM_LONS[grid_length]) / FINAL_LON_PRECISIONS[grid_length] - MAX_LON
    s = <double> (footprint.py // TRIM_LATS[grid_length]) / FINAL_LAT_PRECISIONS[grid_length] - MAX_LAT
    e = w + LON_RESOLUTIONS[grid_length]
    n = s + LAT_RESOLUTIONS[grid_length]

    point = GEOSGeom_createPointFromXY_r(handle, w, s)
    intersects = GEOSPreparedIntersects_r(handle, footprint.geom, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY_r(handle, e, s)
    intersects = GEOSPreparedIntersects_r(handle, footprint.geom, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY_r(handle, w, n)
    intersects = GEOSPreparedIntersects_r(handle, footprint.geom, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY_r(handle, e, n)
    intersects = GEOSPreparedIntersects_r(handle, footprint.geom, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    return True

cdef unsigned char get_length(const Footprint footprint, GEOSContextHandle_t handle) nogil:
    cdef unsigned char length = 0
    cdef double dw, dh
    dw = footprint.be - footprint.bw
    dh = footprint.bn - footprint.bs

    while length <= GRID_LENGTH:
        if LAT_RESOLUTIONS[length] < dw and LON_RESOLUTIONS[length] < dh:
            break
        length += 1
    else:
        raise ValueError('footprint bounds too small')

    while length <= GRID_LENGTH:
        if contained(footprint, length, handle):
            break
        length += 1
    else:
        raise ValueError('footprint point not containable')


@cython.boundscheck(False)
def get_lengths( footprints: Union[gpd.GeoSeries, gpd.GeoDataFrame] ) -> NDArray[np.uint8]:
    cdef double[:] wv, sv, nv, ev, xv, yv
    cdef Py_ssize_t n, i
    cdef np.ndarray[UINT8, ndim=1] lengths = np.ndarray((len(footprints),), dtype=np.uint8)
    cdef unsigned char [:] lv = lengths
    cdef GEOSGeometry *geom = NULL
    cdef const GEOSPreparedGeometry *prepared
    cdef Footprint footprint
    cdef char c

    footprints = footprints.to_crs(4326)
    points = footprints.representative_point().geometry.values.data
    cdef object [:] objects = footprints.geometry.values.data

    xv = points.x.values
    yv = points.y.values
    bw, bs, be, bn = footprints.bounds.T.values

    with get_geos_handle() as handle:
        for i in range(len(footprints)):
            if PyGEOS_GetGEOSGeometry(<PyObject *>objects[i], &geom) == 0:
                raise TypeError
            prepared = GEOSPrepare_r(handle, geom)
            footprint = Footprint(bw[i], bs[i], be[i], bn[i], xv[i], yv[i], prepared)
            lv[i] = get_length(footprint, handle)
            GEOSPreparedGeom_destroy_r(handle, prepared)

    return lengths

