import cython
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

cdef inline bint contained(
        const Footprint footprint, unsigned char grid_length, GEOSContextHandle_t handle
):
# ) nogil:
    print('inline contained')
    cdef double w, s, n, e
    cdef GEOSGeometry *point
    cdef bint intersects

    w = <double> (footprint.px // TRIM_LONS[grid_length]) / FINAL_LON_PRECISIONS[grid_length] - MAX_LON
    s = <double> (footprint.py // TRIM_LATS[grid_length]) / FINAL_LAT_PRECISIONS[grid_length] - MAX_LAT
    e = w + LON_RESOLUTIONS[grid_length]
    n = s + LAT_RESOLUTIONS[grid_length]


    point = GEOSGeom_createPointFromXY_r(handle, w, s)

    intersects = GEOSPreparedIntersects(footprint.prepared, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY_r(handle, e, s)
    intersects = GEOSPreparedIntersects_r(handle, footprint.prepared, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY_r(handle, w, n)
    intersects = GEOSPreparedIntersects_r(handle, footprint.prepared, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    point = GEOSGeom_createPointFromXY_r(handle, e, n)
    intersects = GEOSPreparedIntersects_r(handle, footprint.prepared, point)
    GEOSGeom_destroy_r(handle, point)
    if not intersects:
        return False

    return True

cdef unsigned char get_length(
        const Footprint footprint, GEOSContextHandle_t handle
):
# ) nogil:
    print("get_length")
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
    cdef const GEOSPreparedGeometry *prepared = NULL
    cdef Footprint footprint
    cdef char c
    cdef GEOSContextHandle_t h

    footprints = footprints.to_crs(4326)
    cdef object [:] objects = footprints.geometry.values.data

    points = footprints.representative_point().geometry
    xv = points.x.values
    yv = points.y.values
    bw, bs, be, bn = footprints.bounds.T.values

    with get_geos_handle() as handle:

        for i in range(len(footprints)):
            c = PyGEOS_GetGEOSGeometry(<PyObject *>objects[i], &geom)
            if c == 0:
                raise ValueError('could not get geometry')

            prepared = GEOSPrepare(geom)
            # GEOSPreparedGeom_destroy_r(handle, prepared)
            # prepared = GEOSPrepare_r(handle, geom)

            footprint = Footprint(bw[i], bs[i], be[i], bn[i], xv[i], yv[i], prepared)
            lv[i] = get_length(footprint, handle)
            GEOSPreparedGeom_destroy_r(handle, prepared)

    return lengths
