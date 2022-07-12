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

from cpython cimport (
PyList_SET_ITEM,
PyList_New,
PyList_Append,
PyList_GET_ITEM,
)
from cpython cimport (
PyDict_New,
PyDict_SetItem,
PyDict_SetItemString,
PySet_Add,
PySet_New,
PyFrozenSet_New,
)

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


cdef class FootprintDecomposition:
    cdef :
        pygeos.geometry.Geometry footprint
        unsigned char code_length
        unsigned int tile_count
        unsigned long[4] bounds
        unsigned long* tiles

    def __cinit__(
            self,
            geometry: pygeos.geometry.Geometry,
            unsigned long w_bound,
            unsigned long s_bound,
            unsigned long e_bound,
            unsigned long n_bound,
    ):
        self.footprint = footprint
        self.code_length = PAIR_LENGTH
        self.tile_count = 0
        self.tiles = self.__tiles()

    def __tiles(self):
        ...


    def __dealloc__(self):
        free(self.tiles)






cdef class FootprintsDecomposition:
    ...






