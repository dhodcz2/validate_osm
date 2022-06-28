import pandas as pd
import pygeos.creation
from numpy.typing import NDArray
from typing import Union

import spatialpandas.geometry.base
from libc.stdlib cimport malloc, free
# from cpython cimport Py_buffer
from libcpp cimport vector
import numpy as np
cimport numpy as np
import geopandas as gpd
import shapely.geometry
import spatialpandas.geometry
import shapely.geometry
cimport util.cfuncs as cfuncs

np.string_t = np.dtype('S%d' % (MAX_DIGITS + 1))
ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint64_t UINT64
ctypedef np.int64_t INT64
ctypedef np.uint_t UINT
ctypedef np.int_t INT
ctypedef np.uint8_t BOOL

cdef extern from '<util/globals.h>':
    char SEP
    unsigned int SEP_POS
    char* ALPHABET
    unsigned int BASE
    char PAD
    unsigned int MAX_LAT
    unsigned int MAX_LON
    unsigned int MAX_DIGITS
    unsigned int PAIR_LENGTH
    unsigned int PAIR_PRECISION_FIRST_VALUE
    unsigned long PAIR_PRECISION
    unsigned int GRID_LENGTH
    unsigned int GRID_COLUMNS
    unsigned int GRID_ROWS
    unsigned int GRID_LAT_FIRST_PLACE_VALUE
    unsigned int GRID_LON_FIRST_PLACE_VALUE
    unsigned long FINAL_LAT_PRECISION
    unsigned long FINAL_LON_PRECISION
    unsigned int MIN_TRIMMABLE_CODE_LEN
    double GRID_SIZE_DEGREES

    unsigned long GRID_LAT_RESOLUTION
    unsigned long GRID_LON_RESOLUTION

cdef unsigned char m = (GRID_LENGTH + 1) + sizeof(unsigned char)
cdef unsigned long *FINAL_LAT_PRECISIONS = <unsigned long *> malloc(m)
cdef unsigned long *FINAL_LON_PRECISIONS = <unsigned long *> malloc(m)
cdef unsigned long *TRIM_LATS = <unsigned long *> malloc(m)
cdef unsigned long *TRIM_LONS = <unsigned long *> malloc(m)
cdef double *LAT_RESOLUTIONS = <double *> malloc(m)
cdef double *LON_RESOLUTIONS = <double *> malloc(m)

cdef Py_ssize_t l
for l in range(GRID_LENGTH + 1):
    FINAL_LAT_PRECISIONS[l] = PAIR_PRECISION * pow(GRID_ROWS, l)
    FINAL_LON_PRECISIONS[l] = PAIR_PRECISION * pow(GRID_COLUMNS, l)
    TRIM_LATS[l] = pow(<unsigned long> GRID_ROWS, GRID_LENGTH-l)
    TRIM_LONS[l] = pow(<unsigned long> GRID_COLUMNS, GRID_LENGTH-l)
    LAT_RESOLUTIONS[l] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_ROWS, l)
    LON_RESOLUTIONS[l] = GRID_SIZE_DEGREES / pow(<unsigned long> GRID_COLUMNS, l)

cdef class ShapeLength:
    def __cinit__(
            self,
            footprint: shapely.geometry.base.BaseGeometry,
            unsigned long lw,
            unsigned long ls,
            unsigned long le,
            unsigned long ln,
    ):
        cdef Py_ssize_t r, c, i
        ...

    def __init__(self, footprint: shapely.geometry.base.BaseGeometry):
        ...

cdef class PolygonLength(ShapeLength):
    ...

cdef class MultiPolygonLength(ShapeLength):
    ...





