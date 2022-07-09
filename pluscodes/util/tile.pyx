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

import_pygeos_c_api()
cimport util.cfuncs as cfuncs

from ._geos cimport (
    get_geos_handle,
)
#
# from ._geos cimport (
#     GEOSEquals_r,
#     GEOSGeomTypeId_r,
#     GEOSPreparedGeometry,
#     GEOSGeometry,
#     GEOSPrepare,
#     GEOSContextHandle_t,
#     GEOSPreparedIntersects_r,
#     GEOSPreparedContains,
#     GEOSGeom_destroy_r,
#     GEOSPreparedGeom_destroy_r,
#     GEOSGeom_createPointFromXY_r,
#     GEOSPrepare_r,
#     GEOS_init_r,
#     GEOSPreparedIntersects,
#     GEOSPreparedDisjoint,
#     GEOSGeom_destroy,
#     GEOSPreparedGeom_destroy,
#     GEOSGeom_createPointFromXY,
#     GEOSPrepare,
#     GEOSisValid,
#     GEOSIntersects,
#     GEOSEquals,
#     GEOSGeom_clone,
#     GEOSGetNumGeometries,
#     GEOSGetGeometryN,
#     GEOSisEmpty,
#     get_geos_handle,
#     GEOSFree,
#     GEOSGeomType,
#     GEOSGetNumGeometries_r,
#     GEOSGeomTypeId,
# )
#
cdef extern from 'geos_c.h':
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSPreparedGeometry
    ctypedef struct GEOSCoordSequence
    ctypedef void (*GEOSMessageHandler_r)(const char *message, void *data)

    GEOSContextHandle_t GEOS_init_r() nogil
    void GEOS_finish_r(GEOSContextHandle_t handle) nogil
    void GEOSContext_setErrorMessageHandler_r(
            GEOSContextHandle_t handle,
            GEOSMessageHandler_r ef,
            void* data,
    ) nogil
    void GEOSContext_setNoticeMessageHandler_r(
            GEOSContextHandle_t handle,
            GEOSMessageHandler_r nf,
            void* data,
    ) nogil


    GEOSGeometry* GEOSGeom_createPointFromXY_r(GEOSContextHandle_t, double, double) nogil
    GEOSPreparedGeometry* GEOSPrepare_r(GEOSContextHandle_t, const GEOSGeometry *) nogil
    char GEOSIntersects_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry *) nogil
    char GEOSPreparedIntersects_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry *) nogil
    void GEOSGeom_destroy_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    void GEOSPreparedGeom_destroy_r(GEOSContextHandle_t, GEOSPreparedGeometry *) nogil
    int GEOSGetNumGeometries_r(GEOSContextHandle_t, const GEOSGeometry *) nogil
    char GEOSEquals_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry *) nogil
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t, const GEOSGeometry *) nogil
    const GEOSGeometry* GEOSGetGeometryN_r(GEOSContextHandle_t, const GEOSGeometry *, unsigned int) nogil


    GEOSGeometry* GEOSGeom_createPointFromXY(double, double) nogil
    GEOSPreparedGeometry* GEOSPrepare(const GEOSGeometry *) nogil
    char GEOSIntersects(const GEOSGeometry*, const GEOSGeometry *) nogil
    char GEOSContains(const GEOSGeometry*, const GEOSGeometry *) nogil
    char GEOSPreparedIntersects(const GEOSPreparedGeometry*, const GEOSGeometry *) nogil
    char GEOSPreparedContains(const GEOSPreparedGeometry*, const GEOSGeometry *) nogil
    char GEOSPreparedDisjoint(const GEOSPreparedGeometry*, const GEOSGeometry *) nogil
    void GEOSGeom_destroy(GEOSGeometry *) nogil
    void GEOSPreparedGeom_destroy(GEOSPreparedGeometry *) nogil
    char GEOSEquals(const GEOSGeometry*, const GEOSGeometry *) nogil
    char GEOSisValid(const GEOSGeometry *) nogil
    GEOSGeometry* GEOSGeom_clone(const GEOSGeometry *) nogil
    char GEOSisEmpty(const GEOSGeometry *) nogil


    const GEOSGeometry * GEOSGetGeometryN(const GEOSGeometry*, int) nogil
    int GEOSGetNumGeometries(const GEOSGeometry *) nogil
    int GEOSGeomTypeId(const GEOSGeometry *) nogil
    int GEOSGeomTypeId_r(GEOSContextHandle_t, const GEOSGeometry *) nogil

    char* GEOSGeomType(const GEOSGeometry *) nogil
    void GEOSFree(void *buffer) nogil

    GEOSGeometry* GEOSGeom_createEmptyPolygon_r(GEOSContextHandle_t) nogil
    GEOSGeometry* GEOSGeom_createCollection_r(GEOSContextHandle_t, int, GEOSGeometry **, unsigned int) nogil
    int GEOSGeomTypeId_r(GEOSContextHandle_t, const GEOSGeometry *) nogil
    GEOSGeometry* GEOSGeom_clone_r(GEOSContextHandle_t, const GEOSGeometry *) nogil
    GEOSGeometry* GEOSGeom_createPolygon_r(
            GEOSContextHandle_t,
            GEOSGeometry *,
            GEOSGeometry **,
            unsigned int,
    ) nogil



from cpython cimport (
    PyBytes_AsStringAndSize,
    PyList_SET_ITEM,
    PyList_New,
)

cdef extern from 'Python.h':
    PyObject* Py_BuildValue(const char*, ...) except NULL

# import_pygeos_c_api()
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
    # const GEOSPreparedGeometry *prepared
    GEOSGeometry *geom
    # PyObject *obj
    unsigned char *accepted
    unsigned char *contained
    size_t xtiles, ytiles, xpoints, ypoints
    double *fw
    double *fs
    unsigned long *lw
    unsigned long *ls

cdef Footprint Footprint_init(
        GEOSGeometry *g,
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

    for i in range(xpoints):
        lw[i] = w_bound + i * xstep + xstep
        fw[i] = <double>(lw[i] // trim_lon) / final_lon_precision - MAX_LON
    for i in range(ypoints):
        ls[i] = s_bound + i * ystep + ystep
        fs[i] = <double>(ls[i] // trim_lat) / final_lat_precision - MAX_LAT

    for i in range(points):
        contained[i] = 0
        accepted[i] = 0

    return Footprint(
        geom=g,
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
    # GEOSPreparedGeom_destroy(f.prepared)

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
        int num_geoms

    assert gdf.crs == 4326
    points = gdf.representative_point().geometry
    size = len(gdf)
    objects = gdf.geometry.values.data
    num_geometries = pygeos.get_num_geometries(objects)

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
    # import_pygeos_c_api()
    list_tiles: list[NDArray[np.uint64]] = PyList_New(size)
    # cdef GEOSGeometry * clone = <GEOSGeometry *> malloc(sizeof(GEOSGeometry))
    cdef GEOSGeometry * clone

    cdef long ptr
    cdef char* geomtype
    cdef int v
    cdef unsigned long test
    # import_pygeos_c_api()
    cdef GEOSContextHandle_t handle_2 = GEOS_init_r()
    context_handle = get_geos_handle()
    with context_handle as handle:
        for i in range(size):
            c = PyGEOS_GetGEOSGeometry(<PyObject *> objects[i], &geom)
            # geom = <GEOSGeometry * >(<unsigned long>objects[i]._ptr)

            print(f'geom: {<unsigned long> geom}')
            v = GEOSGetNumGeometries_r(handle, geom)
            if context_handle.last_error[0] != 0:
                print('error')
            if context_handle.last_warning[0] != 0:
                print('warning')
            print('v: ', v)



            #     raise ValueError('num_geometries[i] != v')

            # c = GEOSEquals(geom, geom)
            # print('c: ', c)
            #
            # c = GEOSEquals_r(handle, geom, geom)
            # print('c: ', c)
            # print('geomtype')
            # geomtype = GEOSGeomType(geom)
            # print('strl')
            # strl = strlen(geomtype)
            # print('print')
            # print(geomtype[:strl])
            # print('free')
            # GEOSFree(geomtype)

            # clone = <GEOSGeometry *> objects[i]._ptr
            # print(<unsigned long> clone)
            # print(<unsigned long> geom)

            # print('trying')
            # geom = <GEOSGeometry *> objects[i]._ptr
            # print('achieved')
            #
            # # num_geoms = GEOSGetNumGeometries_r(handle, geom)
            # # print('\thandle:', num_geoms)
            # num_geoms = GEOSGetNumGeometries(geom)
            # print('\thanle:', num_geoms)
            #


            # if num_geoms == -1:
            #     raise ValueError("Invalid geometry")
            # else:
            #     print("num_geoms:", num_geoms)
            #
            # if not c == 1:
            #     raise ValueError(f'c={c}')
            # if geom == NULL:
            #     raise ValueError('geom == NULL')
            #
            footprint = Footprint_init(
                g=geom,
                w_bound=w_view[i],
                s_bound=s_view[i],
                e_bound=e_view[i],
                n_bound=n_view[i],
                grid_length=lengths[i] - PAIR_LENGTH,
            )

            tiles = Footprint_getArray(footprint)
            Py_INCREF(tiles)
            PyList_SET_ITEM(list_tiles, i, tiles)
            Footprint_destroy(footprint)

    # print('return list_tiles')
    return list_tiles


# cdef np.ndarray Footprint_getArray(const Footprint f):
#     cdef :
#         size_t r, c, k, n, nout
#         GEOSGeometry *point
#         const GEOSGeometry *part
#         char intersects
#         unsigned long[:, :] out_view
#         int n_parts, part_num
#
#     n_parts = GEOSGetNumGeometries(f.geom)
#
#
#     for c in range(f.xpoints):
#         for r in range(f.ypoints):
#             # TODO: For some reason, GEOSIntersects is always returning True
#
#             k = c * f.ypoints + r
#             point = GEOSGeom_createPointFromXY(f.fw[c], f.fs[r])
#             # intersects = GEOSPreparedIntersects(f.geom, point)
#             intersects = GEOSIntersects(f.geom, point)
#             if intersects == 2:
#                 print(f'\tfw: {f.fw[c]}, fs: {f.fs[r]}')
#                 raise ValueError("intersects == 2")
#             GEOSGeom_destroy(point)
#
#             f.contained[k] = intersects
#
#             if intersects != 2:
#                 raise ValueError("Intersects is not 2")
#
#
#     nout = 0
#     for c in range(f.xtiles):
#         for r in range(f.ytiles):
#             k = c * f.ypoints + r
#             if (
#                 f.contained[k]
#                     and f.contained[c * f.ypoints + r + 1]
#                     and f.contained[(c + 1) * f.ypoints + r]
#                     and f.contained[(c + 1) * f.ypoints + r + 1]
#             ):
#                 nout += 1
#                 f.accepted[k] = 1
#
#     out = np.ndarray((nout, 2), dtype=np.uint64)
#
#     out_view  = out
#     n = 0
#     for c in range(f.xtiles):
#         for r in range(f.ytiles):
#             k = c * f.ypoints + r
#             if f.accepted[k]:
#                 # out_view[n, 0] = f.lw[r]
#                 # out_view[n, 1] = f.ls[c]
#                 out_view[n, 0] = f.lw[c]
#                 out_view[n, 1] = f.ls[r]
#                 n += 1
#     assert n == nout
#
#     try:
#         assert set(
#             (out[i, 0], out[i, 1])
#             for i in range(nout)
#         ).__len__() == nout
#     except AssertionError:
#         print(f'xtiles: {f.xtiles}', f'ytiles: {f.ytiles}', f'xpoints: {f.xpoints}', f'ypoints: {f.ypoints}')
#
#         raise
#
#     return out
#
cdef np.ndarray Footprint_getArray(const Footprint f):
    cdef :
        size_t r, c, k, n, nout
        GEOSGeometry *point
        const GEOSGeometry *part
        const GEOSPreparedGeometry *prepared
        char intersects
        unsigned long[:, :] out_view
        int n_parts, n_part

    n_parts = GEOSGetNumGeometries(f.geom)
    # print(f'n_parts: {n_parts}')
    for n_part in range(n_parts):
        part = GEOSGetGeometryN(f.geom, n_part)
        prepared = GEOSPrepare(part)

        if part == NULL:
            raise ValueError('part == NULL')

        for c in range(f.xpoints):
            for r in range(f.ypoints):
                k = c * f.ypoints + r
                point = GEOSGeom_createPointFromXY(f.fw[c], f.fs[r])
                intersects = GEOSPreparedIntersects(prepared, point)
                GEOSGeom_destroy(point)
                if intersects == 1:
                    f.contained[k] = 1
                elif intersects == 2:
                    print(f'\tfw: {f.fw[c]}, fs: {f.fs[r]}')
                    raise ValueError("intersects == 2")
        GEOSPreparedGeom_destroy(prepared)

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
                out_view[n, 0] = f.lw[c]
                out_view[n, 1] = f.ls[r]
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

    lengths = lengths[iloc]
    strings = cfuncs.get_strings(longs[:, 0], longs[:, 1], lengths)
    strings = strings[iloc]
    index = pd.MultiIndex.from_arrays((
        iloc, strings,
    ), names=('index', 'tile'),)


    bounds = cfuncs.get_bounds(longs[:, 0], longs[:, 1], lengths)
    data = pygeos.creation.box( bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3] )

    result = GeoSeries(data=data, index=index, crs=4326)
    # print('return get_geoseries_tiles')
    return result


# @cython.boundscheck(False)
def get_parts(object[:] array, bint extract_rings=0):
    cdef Py_ssize_t geom_idx = 0
    cdef Py_ssize_t part_idx = 0
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t count
    cdef GEOSGeometry *geom = NULL
    cdef const GEOSGeometry *part = NULL

    if extract_rings:
        counts = pygeos.get_num_interior_rings(array)
        is_polygon = (pygeos.get_type_id(array) == 3) & (~pygeos.is_empty(array))
        counts += is_polygon
        count = counts.sum()
    else:
        counts = pygeos.get_num_geometries(array)
        count = counts.sum()
    print(f'count: {count}')

    if count == 0:
        # return immediately if there are no geometries to return
        return (
            np.empty(shape=(0, ), dtype=object),
            np.empty(shape=(0, ), dtype=np.intp)
        )

    parts = np.empty(shape=(count, ), dtype=object)
    index = np.empty(shape=(count, ), dtype=np.intp)

    cdef int[:] counts_view = counts
    cdef object[:] parts_view = parts
    cdef np.intp_t[:] index_view = index

    with get_geos_handle() as geos_handle:
        for geom_idx in range(array.size):
            if counts_view[geom_idx] <= 0:
                # No parts to return, skip this item
                continue

            if PyGEOS_GetGEOSGeometry(<PyObject *>array[geom_idx], &geom) == 0:
                raise TypeError("One of the arguments is of incorrect type. "
                                "Please provide only Geometry objects.")

            if geom == NULL:
                continue

            print(f'counts_view: ', counts_view[geom_idx])
            for part_idx in range(counts_view[geom_idx]):
                index_view[idx] = geom_idx

                if extract_rings:
                    pass
                    # part = GetRingN(geos_handle, geom, part_idx)
                else:
                    part = GEOSGetGeometryN_r(geos_handle, geom, part_idx)
                print('got geometry')
                if part == NULL:
                    return  # GEOSException is raised by get_geos_handle

                # clone the geometry to keep it separate from the inputs
                part = GEOSGeom_clone_r(geos_handle, part)
                print('cloned geometry')
                if part == NULL:
                    return  # GEOSException is raised by get_geos_handle

                # cast part back to <GEOSGeometry> to discard const qualifier
                # pending issue #227
                parts_view[idx] = PyGEOS_CreateGeometry(<GEOSGeometry *>part, geos_handle)

                idx += 1

    # TODO: somehow the pygeos module is not failing but my util is
    #   so we need to experiment with putting the code in pygeos
    #   or changing our configuration
    return parts, index

def collections_1d(object geometries, object indices, int geometry_type = 7, object out = None):
    """Converts geometries + indices to collections
    Allowed geometry type conversions are:
    - linearrings to polygons
    - points to multipoints
    - linestrings/linearrings to multilinestrings
    - polygons to multipolygons
    - any to geometrycollections
    """
    cdef Py_ssize_t geom_idx_1 = 0
    cdef Py_ssize_t coll_idx = 0
    cdef unsigned int coll_size = 0
    cdef Py_ssize_t coll_geom_idx = 0
    cdef GEOSGeometry *geom = NULL
    cdef GEOSGeometry *coll = NULL
    cdef int expected_type = -1
    cdef int expected_type_alt = -1
    cdef int curr_type = -1

    if geometry_type == 3:  # POLYGON
        expected_type = 2
    elif geometry_type == 4:  # MULTIPOINT
        expected_type = 0
    elif geometry_type == 5:  # MULTILINESTRING
        expected_type = 1
        expected_type_alt = 2
    elif geometry_type == 6:  # MULTIPOLYGON
        expected_type = 3
    elif geometry_type == 7:
        pass
    else:
        raise ValueError(f"Invalid geometry_type: {geometry_type}.")

    # Cast input arrays and define memoryviews for later usage
    geometries = np.asarray(geometries, dtype=object)
    if geometries.ndim != 1:
        raise TypeError("geometries must be a one-dimensional array.")

    indices = np.asarray(indices, dtype=np.intp)  # intp is what bincount takes
    if indices.ndim != 1:
        raise TypeError("indices must be a one-dimensional array.")

    if geometries.shape[0] != indices.shape[0]:
        raise ValueError("geometries and indices do not have equal size.")

    if geometries.shape[0] == 0:
        # return immediately if there are no geometries to return
        return np.empty(shape=(0, ), dtype=object)

    if np.any(indices[1:] < indices[:indices.shape[0] - 1]):
        raise ValueError("The indices should be sorted.")

    # get the geometry count per collection (this raises on negative indices)
    cdef int[:] collection_size = np.bincount(indices).astype(np.int32)

    # A temporary array for the geometries that will be given to CreateCollection.
    # Its size equals max(collection_size) to accomodate the largest collection.
    temp_geoms = np.empty(shape=(np.max(collection_size), ), dtype=np.intp)
    cdef np.intp_t[:] temp_geoms_view = temp_geoms

    # The final target array
    cdef Py_ssize_t n_colls = collection_size.shape[0]
    # Allow missing indices only if 'out' was given explicitly (if 'out' is not
    # supplied by the user, we would have to come up with an output value ourselves).
    cdef char allow_missing = out is not None
    out = _check_out_array(out, n_colls)
    cdef object[:] out_view = out

    with get_geos_handle() as geos_handle:
        for coll_idx in range(n_colls):
            if collection_size[coll_idx] == 0:
                if allow_missing:
                    continue
                else:
                    raise ValueError(
                        f"Index {coll_idx} is missing from the input indices."
                    )
            coll_size = 0

            # fill the temporary array with geometries belonging to this collection
            for coll_geom_idx in range(collection_size[coll_idx]):
                if PyGEOS_GetGEOSGeometry(<PyObject *>geometries[geom_idx_1 + coll_geom_idx], &geom) == 0:
                    _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                    raise TypeError(
                        "One of the arguments is of incorrect type. Please provide only Geometry objects."
                    )

                # ignore missing values
                if geom == NULL:
                    continue

                # Check geometry subtype for non-geometrycollections
                if geometry_type != 7:
                    curr_type = GEOSGeomTypeId_r(geos_handle, geom)
                    if curr_type == -1:
                        _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                        return  # GEOSException is raised by get_geos_handle
                    if curr_type != expected_type and curr_type != expected_type_alt:
                        _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                        raise TypeError(
                            f"One of the arguments has unexpected geometry type {curr_type}."
                        )

                # assign to the temporary geometry array
                geom = GEOSGeom_clone_r(geos_handle, geom)
                if geom == NULL:
                    _deallocate_arr(geos_handle, temp_geoms_view, coll_size)
                    return  # GEOSException is raised by get_geos_handle
                temp_geoms_view[coll_size] = <np.intp_t>geom
                coll_size += 1

            # create the collection
            if geometry_type != 3:  # Collection
                coll = GEOSGeom_createCollection_r(
                    geos_handle,
                    geometry_type,
                    <GEOSGeometry**> &temp_geoms_view[0],
                    coll_size
                )
            elif coll_size != 0:  # Polygon, non-empty
                coll = GEOSGeom_createPolygon_r(
                    geos_handle,
                    <GEOSGeometry*> temp_geoms_view[0],
                    NULL if coll_size <= 1 else <GEOSGeometry**> &temp_geoms_view[1],
                    coll_size - 1
                )
            else:  # Polygon, empty
                coll = GEOSGeom_createEmptyPolygon_r(
                    geos_handle
                )

            if coll == NULL:
                return  # GEOSException is raised by get_geos_handle

            out_view[coll_idx] = PyGEOS_CreateGeometry(coll, geos_handle)

            geom_idx_1 += collection_size[coll_idx]

    return out


cdef _deallocate_arr(void* handle, np.intp_t[:] arr, Py_ssize_t last_geom_i):
    """Deallocate a temporary geometry array to prevent memory leaks"""
    cdef Py_ssize_t i = 0
    cdef GEOSGeometry *g

    for i in range(last_geom_i):
        g = <GEOSGeometry *>arr[i]
        if g != NULL:
            GEOSGeom_destroy_r(handle, <GEOSGeometry *>arr[i])

def _check_out_array(object out, Py_ssize_t size):
    if out is None:
        return np.empty(shape=(size,), dtype=object)
    if not isinstance(out, np.ndarray):
        raise TypeError("out array must be of numpy.ndarray type")
    if not out.flags.writeable:
        raise TypeError("out array must be writeable")
    if out.dtype != object:
        raise TypeError("out array dtype must be object")
    if out.ndim != 1:
        raise TypeError("out must be a one-dimensional array.")
    if out.shape[0] < size:
        raise ValueError(f"out array is too small ({out.shape[0]} < {size})")
    return out

