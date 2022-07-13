from libc.stdlib cimport malloc, free
import pygeos.geometry

from cython cimport view
import pygeos.geometry

import pandas as pd
import pygeos.creation
from geopandas import GeoDataFrame, GeoSeries
from typing import Union

import numpy as np
cimport numpy as np
np.import_array()
cimport util.cfuncs as cfuncs

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
ctypedef np.int64_t INT64
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


cdef struct Decomposition:
    size_t n_tiles
    unsigned char code_length
    unsigned long* longs

cdef Decomposition decompose(
        geometry: pygeos.geometry.Geometry,
        unsigned long w,
        unsigned long s,
        unsigned long e,
        unsigned long n,
        unsigned char code_length,
):
    cdef :
        unsigned char grid_length
        unsigned long xstep, ystep, trim_lon, trim_lat
        size_t xtiles, ytiles, xpoints, ypoints, n_tiles, i, j, k
        unsigned long[:] lw, ls
        double[:] fx, fy, fw_boad, fs_broad
        np.ndarray[UINT8, ndim=1] intersects
        unsigned long* longs
        double final_lon_precision, final_lat_precision

    grid_length = code_length - PAIR_LENGTH

    xstep = XSTEPS[grid_length]
    ystep = YSTEPS[grid_length]


    trim_lon = TRIM_LONS[grid_length]
    trim_lat = TRIM_LATS[grid_length]
    final_lon_precision = FINAL_LON_PRECISIONS[grid_length]
    final_lat_precision = FINAL_LAT_PRECISIONS[grid_length]

    xtiles = (e // xstep) - (w // xstep)
    ytiles = (n // ystep) - (s // ystep)

    xpoints = xtiles + 1
    ypoints = ytiles + 1

    # w += xstep
    # s += ystep
    # e += xstep
    # n += ystep
    #
    lw = view.array(shape=(xpoints,), itemsize=sizeof(unsigned long), format='L')
    ls = view.array(shape=(ypoints,), itemsize=sizeof(unsigned long), format='L')
    fw = view.array(shape=(xpoints,), itemsize=sizeof(double), format='d')
    fs = view.array(shape=(ypoints,), itemsize=sizeof(double), format='d')

    for i in range(xpoints):
        lw[i] = w + (i * xstep) + xstep
        fw[i] = <double> (lw[i] // trim_lon) / final_lon_precision - MAX_LON

    for i in range(ypoints):
        ls[i] = s + (i * ystep) + ystep
        fs[i] = <double> (ls[i] // trim_lat) / final_lat_precision - MAX_LAT

    coords = view.array(shape=(xpoints * ypoints, 2), itemsize=sizeof(double), format='d')
    for i in range(xpoints):
        for j in range(ypoints):
            coords[i * ypoints + j, 0] = fw[i]
            coords[i * ypoints + j, 1] = fs[j]

    # TODO: If only we had a stable C API for PyGEOS, we could iterate and instantiate ephemeral Point objects
    #   instead of needing to create a numpy array of PyGEOS objects.
    points = pygeos.creation.points(
        np.asarray(coords[:, 0], dtype=np.float64),
        np.asarray(coords[:, 1], dtype=np.float64),
    )
    intersects = pygeos.intersects(geometry, points)
    n_tiles = 0
    for i in range(xtiles):
        for j in range(ytiles):
            k = i * ypoints + j
            if (
                intersects[k]
                and intersects[k + 1]
                and intersects[k + ypoints + 1]
                and intersects[k + ypoints]
            ):
                n_tiles += 1
                intersects[k] = True
            else:
                intersects[k] = False

    if n_tiles == 0:
        # TODO: For now we are going to exclude absurdly small geometries
        #   our approach scales well with building size, however including "buildings" somehow smaller than a bus stop
        #   will generate many, many tiles for a very small area, stressing visualization
        if code_length >= 11:
            return Decomposition(n_tiles=0, code_length=code_length, longs=NULL)

        if grid_length == GRID_LENGTH:
            raise ValueError('Unable to decompose geometry into sufficiently small tiles')
        return decompose(
            geometry=geometry,
            w=w,
            s=s,
            e=e,
            n=n,
            code_length=code_length + 1,
        )

    longs = <unsigned long *> malloc(sizeof(unsigned long) * n_tiles * 2)
    k = 0
    for i in range(xtiles):
        for j in range(ytiles):
            if intersects[i * ypoints + j]:
                longs[k] = lw[i]
                longs[k + 1] = ls[j]
                k += 2

    assert k == n_tiles * 2

    return Decomposition(
        n_tiles=n_tiles,
        code_length=code_length,
        longs=longs,
    )

cdef class Decompositions:
    cdef :
        Decomposition * _decompositions

        size_t _n_tiles, _n_gdf
        np.ndarray _strings

        double[:, :] _bounds
        unsigned char[:] _code_lengths
        unsigned long[:,:] _longs
        unsigned long[:] _iloc, _spaces, _left_bounds,
        unsigned int[:] _repeats

    def __cinit__(self, gdf: Union[GeoDataFrame, GeoSeries]):
        cdef :
            Decomposition decomposition
            unsigned long[:] left_bounds, iloc
            unsigned char[:] code_lengths
            unsigned long[:] lw, ls, le, ln
            size_t i, j, k


        self._n_gdf = len(gdf)
        self._decompositions = <Decomposition *>malloc(self._n_gdf * sizeof(Decomposition))

        fw, fs, fe, fn = gdf.bounds.values.T
        lw = np.ndarray.astype((
                (fw + MAX_LON) * FINAL_LON_PRECISION
        ), dtype=np.uint64)
        ls = np.ndarray.astype((
                (fs + MAX_LAT) * FINAL_LAT_PRECISION
        ), dtype=np.uint64)
        le = np.ndarray.astype((
                (fe + MAX_LON) * FINAL_LON_PRECISION
        ), dtype=np.uint64)
        ln = np.ndarray.astype((
                (fn + MAX_LAT) * FINAL_LAT_PRECISION
        ), dtype=np.uint64)

        geometry = gdf.geometry.values.data

        self._repeats = view.array(shape=(self._n_gdf,), itemsize=sizeof(unsigned int), format='I')
        self._code_lengths = view.array(shape=(self._n_gdf,), itemsize=sizeof(unsigned char), format='B')
        self._left_bounds = view.array(shape=(self._n_gdf,), itemsize=sizeof(unsigned long), format='L')

        # TODO: Rather than using the ILOC, space needs to actually be determine with the Z-order curve.
        self._spaces = view.array(shape=(self._n_gdf,), itemsize=sizeof(unsigned long), format='L')

        for i in range(self._n_gdf):
            decomposition = decompose(
                geometry=geometry[i],
                w=lw[i],
                s=ls[i],
                e=le[i],
                n=ln[i],
                code_length=PAIR_LENGTH,
            )
            self._decompositions[i] = decomposition
            self._left_bounds[i] = self._n_tiles
            self._n_tiles += decomposition.n_tiles
            self._repeats[i] = decomposition.n_tiles
            self._code_lengths[i] = decomposition.code_length
            self._spaces[i] = i

        self._longs = view.array(shape=(self._n_tiles, 2), itemsize=sizeof(unsigned long), format='L')
        self._iloc = view.array(shape=(self._n_tiles,), itemsize=sizeof(unsigned long), format='L')
        code_lengths = view.array(shape=(self._n_tiles,), itemsize=sizeof(unsigned char), format='B')

        k = 0
        for i in range(self._n_gdf):
            decomposition = self._decompositions[i]
            for j in range(0, decomposition.n_tiles * 2, 2):
                self._longs[k, 0] = decomposition.longs[j]
                self._longs[k, 1] = decomposition.longs[j + 1]
                code_lengths[k] = decomposition.code_length
                self._iloc[k] = i
                k += 1

        assert k == self._n_tiles

        self._strings = cfuncs.get_strings(self._longs[:, 0], self._longs[:, 1], code_lengths)
        self._bounds = cfuncs.get_bounds(self._longs[:, 0], self._longs[:, 1], code_lengths)


    def __dealloc__(self):
        for i in range(self._n_gdf):
            free(self._decompositions[i].longs)
        free(self._decompositions)


    def geoseries(self) -> GeoSeries:
        spaces = np.asarray(self._spaces).repeat(np.asarray(self._repeats))
        index = pd.MultiIndex.from_arrays((
            np.asarray(self._iloc), np.asarray(spaces), self._strings
        ), names=('iloc', 'space', 'string'))
        data = pygeos.creation.box(self._bounds[:, 0], self._bounds[:, 1], self._bounds[:, 2], self._bounds[:, 3])
        return GeoSeries(data=data, index=index, crs=4326)

    def tiles(self) -> dict[str, str]:
        return {
            self._strings[j]: self._spaces[i]
            for i in range(self._n_gdf)
            for j in range(self._left_bounds[i], self._left_bounds[i] + self._repeats[i])
        }

    def spaces(self) -> dict[str, set[str]]:
        return {
            self._spaces[i]: {
                self._strings[j]
                for j in range(self._left_bounds[i], self._left_bounds[i] + self._repeats[i])
            }
            for i in range(self._n_gdf)
        }
