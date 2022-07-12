from libc.stdlib cimport malloc, free
import pygeos.geometry
from typing import Hashable
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
np.import_array()
cimport util.cfuncs as cfuncs
cimport util.z


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


cdef class Decomposition:
    cdef :
        pygeos.geometry.Geometry footprint
        size_t n_tiles
        unsigned char code_length
        unsigned long w, s, e, n
        unsigned long* longs

    def __cinit__(
            self,
            geometry: pygeos.geometry.Geometry,
            unsigned long w,
            unsigned long s,
            unsigned long e,
            unsigned long n,
    ):
        self.footprint = footprint
        self.code_length = PAIR_LENGTH
        self.tile_count = 0
        self.tiles = self.__tiles()

    cdef unsigned long* __tiles(self):
        grid_length = self.code_length - PAIR_LENGTH

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

        lw = view.array(shape=(xpoints,), itemsize=sizeof(unsigned long), format='L')
        ls = view.array(shape=(ypoints,), itemsize=sizeof(unsigned long), format='L')
        fw = view.array(shape=(xpoints,), itemsize=sizeof(double), format='d')
        fs = view.array(shape=(ypoints,), itemsize=sizeof(double), format='d')

        for i in range(xpoints):
            lw[i] = w_bound + (i * xstep)
            fw[i] = <double> (lw[i] // trim_lon) / final_lon_precision - MAX_LON

        for i in range(ypoints):
            ls[i] = s_bound + (i * ystep)
            fs[i] = <double> (ls[i] // trim_lat) / final_lat_precision - MAX_LAT

        fw_broad = view.array(shape=(xpoints * ypoints,), itemsize=sizeof(double), format='d')
        fs_broad = view.array(shape=(xpoints * ypoints,), itemsize=sizeof(double), format='d')
        for i in range(xpoints):
            for j in range(ypoints):
                fw_broad[i * ypoints + j] = fw[i]
                fs_broad[i * ypoints + j] = fs[j]

        # TODO: If only we had a stable C API for PyGEOS, we could iterate and instantiate ephemeral Point objects
        #   instead of needing to create a numpy array of PyGEOS objects.
        points = pygeos.creation.points(np.asarray(fw_broad), np.asarray(fs_broad), )
        intersects = (
            pygeos.intersects(self.footprint, points)
            .reshape(xpoints, ypoints)
        )
        intersects = (
                intersects[:xtiles, :ytiles]
                & intersects[1:xpoints, :ytiles]
                & intersects[1:xpoints, 1:ypoints]
                & intersects[:xtiles, 1:ypoints]
        )
        self.tile_count = intersects.sum()

        if self.tile_count == 0:
            if grid_length == GRID_LENGTH:
                raise ValueError('Infeasible tile request')
            else:
                self.code_length += 1
                return self.__tiles()
        else:
            longs = malloc(sizeof(unsigned long) * self.tile_count * 2)
            n = 0
            for i in range(xtiles):
                for j in range(ytiles):
                    if intersects[i, j]:
                        self.tiles[n] = lw[i]
                        self.tiles[n + 1] = ls[j]
                        n += 2
            return longs



    def __dealloc__(self):
        free(self.tiles)

cdef class Decompositions:
    cdef :
        Decomposition* decompositions

        size_t n_tiles, n_gdf
        np.ndarray strings

        double[:, :] bound
        unsigned char[:] code_length
        unsigned long[:,:] tile
        unsigned long[:] iloc
        unsigned long[:] space

    def __cinit__(self, gdf: Union[GeoDataFrame, GeoSeries]):
        cdef :
            Decomposition decomposition



        self.n_gdf = len(gdf)
        self.decompositions = malloc(self.len * sizeof(Decomposition))

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

        for i in range(self.len):
            decomposition = Decomposition( gdf.geometry[i], lw[i], ls[i], le[i], ln[i], )
            self.decompositions[i] = decomposition
            self.n_tiles += decomposition.n_tiles
            self.repeat[i] = decomposition.n_tiles

        self.repeat = view.array(shape=(self.n_tiles,), itemsize=sizeof(unsigned long), format='L')
        self.longs = view.array(shape=(self.n_tiles,2), itemsize=sizeof(unsigned long), format='L')
        self.iloc = view.array(shape=(self.n_tiles,), itemsize=sizeof(unsigned long), format='L')
        self.code_length = view.array(shape=(self.n_tiles,), itemsize=sizeof(unsigned char), format='B')

        n = 0
        for i in range(self.n_tiles):
            decomposition = self.decompositions[i]
            for j in range(decomposition.tile_count):
                self.tile[n, 0] = decomposition.tiles[j]
                self.tile[n, 1] = decomposition.tiles[j + 1]
                self.code_length[n] = decomposition.code_length
                self.iloc[n] = i
                n += 1

        self.strings = cfuncs.get_strings(self.longs[:, 0], self.longs[:, 1], self.code_lengths)
        self.bounds = cfuncs.get_bounds(self.longs[:, 0], self.longs[:, 1], self.code_lengths)
        # TODO: Rather than using the ILOC, space needs to actually be determine with the Z-order curve.
        self.space = self.iloc

    def __dealloc__(self):
        free(self.decompositions)


    def geoseries(self) -> GeoSeries:
        index = pd.MultiIndex.from_arrays(
            (self.iloc, self.space, self.strings),
            names=('iloc', 'space', 'tile')
        )
        data = pygeos.creation.box(self.bounds[:, 0], self.bounds[:, 1], self.bounds[:, 2], self.bounds[:, 3])
        return GeoSeries(data=data, index=index, crs=4326)



    def sets(self) -> dict[Hashable, set]:
        ...

    def spaces(self) -> dict[Hashable, Hashable]:
        ...

    def tiles(self) -> dict[Hashable, Hashable]:
        ...






