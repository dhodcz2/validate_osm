from libc.stdlib cimport malloc
cimport cython.array
from libc.stdlib cimport free

from numpy.typing import NDArray
from typing import Union, Type

import Cython
import cython
import numpy as np
cimport numpy as np
cimport libc.math as math
import pygeos
from dataclasses import dataclass

from numpy import ndarray

ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint64_t UINT64
ctypedef np.int64_t INT64
ctypedef np.uint_t UINT
ctypedef np.int_t INT
if False | False:
    import math

# https://github.com/google/open-location-code/blob/main/python/openlocationcode/openlocationcode.py


cdef char SEPARATOR_ = '+'
cdef unsigned int SEPARATOR_POSITION_ = 8
cdef char PADDING_CHARACTER_ = '0'
cdef char * CODE_ALPHABET_ = '23456789CFGHJMPQRVWX'
cdef int ENCODING_BASE_ = 20
cdef int LATITUDE_MAX_ = 90
cdef int LONGITUDE_MAX_ = 180
cdef int  MAX_DIGIT_COUNT_ = 15
cdef int PAIR_CODE_LENGTH_ = 10
cdef int PAIR_FIRST_PLACE_VALUE_ = ENCODING_BASE_ ** (PAIR_CODE_LENGTH_ // 2 - 1)
cdef int PAIR_PRECISION_ = ENCODING_BASE_ ** 3
# cdef PAIR_RESOLUTIONS_ = [20.0, 1.0, .05, .0025, .000125]
cdef int GRID_CODE_LENGTH_ = MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_
cdef int GRID_COLUMNS_ = 4
cdef int GRID_ROWS_ = 5
cdef int GRID_LAT_FIRST_PLACE_VALUE_ = GRID_ROWS_ ** (GRID_CODE_LENGTH_ - 1)
cdef int GRID_LNG_FIRST_PLACE_VALUE_ = GRID_COLUMNS_ ** (GRID_CODE_LENGTH_ - 1)
cdef int FINAL_LAT_PRECISION_ = PAIR_PRECISION_ * GRID_ROWS_ ** (MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_)
cdef int FINAL_LNG_PRECISION_ = PAIR_PRECISION_ * GRID_COLUMNS_ ** (MAX_DIGIT_COUNT_ - PAIR_CODE_LENGTH_)
cdef int MIN_TRIMMABLE_CODE_LEN_ = 6
cdef float GRID_SIZE_DEGREES_ = 0.000125


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef _get_codes(
        unsigned long[:] ix,
        unsigned long[:] iy,
        unsigned long[:] lengths,
):
# cdef _get_codes(
#         np.ndarray[UINT64, ndim=1] ix,
#         np.ndarray[UINT64, ndim=1] iy,
#         np.ndarray[UINT8, ndim=1] lengths,
# ):
#     cdef char *string[PAIR_CODE_LENGTH_]
    cdef char* string = <char *> malloc((PAIR_CODE_LENGTH_) * sizeof(char))
    cdef Py_ssize_t length = len(ix)
    print('length %d' % (length))
    cdef const char* alphabet = CODE_ALPHABET_
    cdef unsigned char strlen = MAX_DIGIT_COUNT_ + 1

    codes = np.ndarray(shape=length, dtype='U%d' % strlen)
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned int i
    cdef unsigned char c

    print('size %d' % (ix.size))

    for i in range(ix.size):
        x = ix[i]
        y = iy[i]
        length = lengths[i]

        # grid
        # for c in range(MAX_DIGIT_COUNT_ - 1, PAIR_CODE_LENGTH_-1,-1):
        for c in range(MAX_DIGIT_COUNT_, PAIR_CODE_LENGTH_, -1):
            string[c] = alphabet[
                y % GRID_ROWS_ * GRID_COLUMNS_
                + x % GRID_COLUMNS_
            ]
            x //= ENCODING_BASE_
            y //= ENCODING_BASE_

        #pair
        for c in range(PAIR_CODE_LENGTH_, SEPARATOR_POSITION_, -2):
            string[c] = alphabet[
                x % ENCODING_BASE_
            ]
            string[c-1] = alphabet[
                y % ENCODING_BASE_
            ]
            x //= ENCODING_BASE_
            y //= ENCODING_BASE_

        string[SEPARATOR_POSITION_] = SEPARATOR_

        for c in range(SEPARATOR_POSITION_-1, 0, -2):
            string[c] = alphabet[
                x % ENCODING_BASE_
                ]
            string[c-1] = alphabet[
                y % ENCODING_BASE_
                ]
            x //= ENCODING_BASE_
            y //= ENCODING_BASE_

        codes[i] = string[:MAX_DIGIT_COUNT_].decode('UTF-8')
        # codes[i] = string[:length].decode('ascii')
        # codes[i] = string[:length].decode('ascii')

    free(string)
    return codes

def get_codes(
        ix: NDArray[np.uint64],
        iy: NDArray[np.uint64],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.unicode]:
    """

    :param ix:
    :param iy:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(ix), MAX_DIGIT_COUNT_ + 1, dtype=np.uint8)
    if not len(ix) == len(iy) == len(lengths):
        raise ValueError('invalid lengths')
    return get_codes(ix, iy, lengths)

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
cdef _get_digits(
        unsigned long[:] ix,
        unsigned long[:] iy,
        unsigned long[:] lengths,
):
    # cdef char* string = <char *> malloc((PAIR_CODE_LENGTH_) * sizeof(char))
    cdef Py_ssize_t length = len(ix)
    print('length %d' % (length))
    # cdef const char* alphabet = CODE_ALPHABET_
    # cdef unsigned char strlen = MAX_DIGIT_COUNT_ + 1

    cdef np.ndarray[UINT8, ndim=2] digits = np.ndarray(shape=(length, MAX_DIGIT_COUNT_), dtype=np.uint8)
    cdef unsigned char[:,:] dv = digits
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned int i
    cdef unsigned char c
    cdef unsigned int r
    for r in range(length):
        y = iy[r]
        x = ix[r]

        print('grid')
        for c in range(MAX_DIGIT_COUNT_-1, PAIR_CODE_LENGTH_-1, -1):
            dv[r,c] = (
                (y % GRID_ROWS_ * GRID_COLUMNS_)
                +( x % GRID_COLUMNS_)
            )
            x //= GRID_COLUMNS_
            y //= GRID_ROWS_

        # print('pairs')
        # for c in range(PAIR_CODE_LENGTH_, SEPARATOR_POSITION_, -2):
        #     dv[r,c] = x % ENCODING_BASE_
        #     dv[r,c-1] = y % ENCODING_BASE_
        #     x //= ENCODING_BASE_
        #     y //= ENCODING_BASE_

        print('pairs')
        for c in range(PAIR_CODE_LENGTH_ -1, 0, -2):
            dv[r,c] = x % ENCODING_BASE_
            dv[r,c-1] = y % ENCODING_BASE_
            x //= ENCODING_BASE_
            y //= ENCODING_BASE_

    return digits



def get_digits(
        ix: NDArray[np.uint64],
        iy: NDArray[np.uint64],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.unicode]:
    """

    :param ix:
    :param iy:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(ix), MAX_DIGIT_COUNT_ + 1, dtype=np.uint8)
    if not len(ix) == len(iy) == len(lengths):
        raise ValueError('invalid lengths')
    return _get_digits(ix, iy, lengths)
