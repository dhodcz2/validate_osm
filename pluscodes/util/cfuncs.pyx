import os

cimport cython
from libc.stdlib cimport malloc
from libc.stdlib cimport free

from numpy.typing import NDArray

import numpy as np
cimport numpy as np

cdef extern from '<globals.h>':
    char SEP
    unsigned int SEP_POS
    char* ALPHABET
    unsigned int BASE
    char PAD
    unsigned int MAX_LAT
    unsigned int MAX_LON
    unsigned int MAX_DIGITS
    unsigned int PAIR_LENGTH
    unsigned int PAIR_FIRST_VALUE
    unsigned long PAIR_PRECISION
    unsigned int GRID_LENGTH
    unsigned int GRID_COLUMNS
    unsigned int GRID_ROWS
    unsigned int GRID_LAT_FIRST_PLACE_VALUE
    unsigned long FINAL_LAT_PRECISION
    unsigned long FINAL_LON_PRECISION
    unsigned int MIN_TRIMMABLE_CODE_LEN
    int DEBUG

np.string_t = np.dtype('S%d' % (MAX_DIGITS + 1))
ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint64_t UINT64
ctypedef np.int64_t INT64
ctypedef np.uint_t UINT
ctypedef np.int_t INT
# ctypedef np.str_ STRING
if False | False:
    pass




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef  _encode_string(
        unsigned long[:] ix,
        unsigned long[:] iy,
        unsigned long[:] lengths,
):
    cdef char* string = <char *> malloc((MAX_DIGITS + 1) * sizeof(char))
    cdef unsigned long length = ix.size
    cdef const char* alphabet = ALPHABET
    cdef unsigned char strlen = MAX_DIGITS + 1
    codes = np.ndarray(shape=(length,), dtype='U%d' % strlen)
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned int i
    cdef unsigned char c

    for i in range(ix.size):
        x = ix[i]
        y = iy[i]
        length = lengths[i]

        for c in range(MAX_DIGITS, PAIR_LENGTH, -1):
            string[c] = alphabet[
                y % GRID_ROWS * GRID_COLUMNS
                + x % GRID_COLUMNS
            ]
            x //= GRID_COLUMNS
            y //= GRID_ROWS

        for c in range(PAIR_LENGTH, SEP, -2):
            string[c] = alphabet[ x % BASE ]
            string[c-1] = alphabet[ y % BASE ]
            x //= BASE
            y //= BASE

        string[SEP_POS] = SEP

        for c in range(SEP_POS-1, 0, -2):
            string[c] = alphabet[ x % BASE ]
            string[c-1] = alphabet[ y % BASE ]
            x //= BASE
            y //= BASE

        codes[i] = string[:MAX_DIGITS].decode('UTF-8')

    free(string)
    codes = np.ndarray(dtype=np.uint8)
    return codes

def encode_string(
        ix: NDArray[np.uint64],
        iy: NDArray[np.uint64],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.string_t]:
    """

    :param ix:
    :param iy:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(ix), MAX_DIGITS + 1, dtype=np.uint8)
    if not len(ix) == len(iy) == len(lengths):
        raise ValueError('invalid lengths')
    return _encode_string(ix, iy, lengths)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[UINT8, ndim=2] _encode_digits(
        unsigned long[:] ix,
        unsigned long[:] iy,
        unsigned long[:] lengths,
):
    cdef Py_ssize_t length = ix.size
    cdef Py_ssize_t row
    cdef np.ndarray[UINT8, ndim=2] digits = np.ndarray(shape=(length, MAX_DIGITS), dtype=np.uint8)
    cdef unsigned char[:,:] dv = digits
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned char c

    for r in range(length):
        y = iy[r]
        x = ix[r]

        for c in range(MAX_DIGITS-1, PAIR_LENGTH-1, -1):
            dv[r,c] = (
                y % GRID_ROWS * GRID_COLUMNS
                + x % GRID_COLUMNS
            )
            x //= GRID_COLUMNS
            y //= GRID_ROWS

        for c in range(PAIR_LENGTH -1, 0, -2):
            dv[r,c] = x % BASE
            dv[r,c-1] = y % BASE
            x //= BASE
            y //= BASE

    return digits



def encode_digits(
        ix: NDArray[np.uint64],
        iy: NDArray[np.uint64],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.uint8]:
    """

    :param ix:
    :param iy:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(ix), MAX_DIGITS + 1, dtype=np.uint8)
    if not len(ix) == len(iy) == len(lengths):
        raise ValueError('invalid lengths')
    return _encode_digits(ix, iy, lengths)

cdef np.ndarray[UINT64, ndim=2] _decode_digits(
        unsigned char[:,:] digits,
        unsigned long[:] lengths,
):
    cdef Py_ssize_t length = digits.shape[0]
    cdef Py_ssize_t r
    cdef np.ndarray[UINT64, ndim=2] ix = np.ndarray(shape=(length,), dtype=np.uint64)
    cdef np.ndarray[UINT64, ndim=2] iy = np.ndarray(shape=(length,), dtype=np.uint64)
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned char c
    ...

def decode_digits(
        digits: NDArray[np.uint8],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.uint64]:
    """

    :param digits:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(digits), MAX_DIGITS + 1, dtype=np.uint8)
    if not len(digits) == len(lengths):
        raise ValueError('invalid lengths')
    return _decode_digits(digits, lengths)

# cdef NDArray[np.uint8] _decode_strings(
#         unsigned char[:,::1] strings,
# ):
#     cdef Py_ssize_t length = strings.shape[0]
#     cdef Py_ssize_t r
#     cdef np.ndarray[UINT64, ndim=2] ix = np.ndarray(shape=(length,), dtype=np.uint64)
#     cdef np.ndarray[UINT64, ndim=2] iy = np.ndarray(shape=(length,), dtype=np.uint64)
#     cdef unsigned long x
#     cdef unsigned long y
#     cdef unsigned char c
#
#     for r in range(length):
#         y = 0
#         x = 0
#         for c in range(PAIR_LENGTH):
#             # y += ALPHABET * BASE ** (PAIR_LENGTH - c - 1)
#             # x += alphabet.index(strings[r,c+PAIR_LENGTH]) * BASE ** (PAIR_LENGTH - c - 1)
#         ix[r] = x
#         iy[r] = y
#
#
#     return ix, iy
#
#
# def decode_strings( strings: NDArray[STRING], ) -> NDArray[np.uint8]:
#     return _decode_strings(strings)
