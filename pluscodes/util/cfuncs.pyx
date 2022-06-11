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
    unsigned int PAIR_PRECISION_FIRST_VALUE
    unsigned long PAIR_PRECISION
    unsigned int GRID_LENGTH
    unsigned int GRID_COLUMNS
    unsigned int GRID_ROWS
    unsigned int GRID_LAT_FIRST_PLACE_VALUE
    unsigned long FINAL_LAT_PRECISION
    unsigned long FINAL_LON_PRECISION
    unsigned int MIN_TRIMMABLE_CODE_LEN

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
        unsigned long[:] fx,
        unsigned long[:] fy,
        unsigned long[:] code_lengths,
):
    cdef char* string = <char *> malloc((MAX_DIGITS + 1) * sizeof(char))
    cdef unsigned long length = fx.size
    cdef const char* alphabet = ALPHABET
    codes = np.ndarray(shape=(length,), dtype='U%d' % (MAX_DIGITS+1))
    # codes = np.ndarray(shape=(length,MAX_DIGITS+1), dtype=np.uint8)
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned int i
    cdef unsigned char c
    cdef ssize_t strlen


    for i in range(length):
        x = fx[i]
        y = fy[i]
        strlen = code_lengths[i]

        for c in range(MAX_DIGITS, PAIR_LENGTH, -1):
            string[c] = alphabet[
                y % GRID_ROWS * GRID_COLUMNS
                + x % GRID_COLUMNS
            ]
            # codes[i, c] = (
            #     y % GRID_ROWS * GRID_COLUMNS
            #     + x % GRID_COLUMNS
            # )
            x //= GRID_COLUMNS
            y //= GRID_ROWS

        for c in range(PAIR_LENGTH, SEP_POS, -2):
            string[c] = alphabet[ x % BASE ]
            string[c-1] = alphabet[ y % BASE ]
            # codes[i, c] = x % BASE
            # codes[i, c-1] = y % BASE
            x //= BASE
            y //= BASE

        string[SEP_POS] = SEP

        for c in range(SEP_POS-1, 0, -2):
            string[c] = alphabet[ x % BASE ]
            string[c-1] = alphabet[ y % BASE ]
            # codes[i, c] = x % BASE
            # codes[i, c-1] = y % BASE
            x //= BASE
            y //= BASE

        codes[i] = string[:strlen].decode('utf-8')

    free(string)
    return codes

def encode_string(
        fx: NDArray[np.uint64],
        fy: NDArray[np.uint64],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.string_t]:
    """

    :param fx:
    :param fy:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(fx), MAX_DIGITS + 1, dtype=np.uint8)
    if not len(fx) == len(fy) == len(lengths):
        raise ValueError('invalid lengths')
    return _encode_string(fx, fy, lengths)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.ndarray[UINT8, ndim=2] _encode_digits(
        unsigned long[:] fx,
        unsigned long[:] fy,
        unsigned long[:] lengths,
):
    cdef Py_ssize_t length = fx.size
    cdef Py_ssize_t row
    cdef np.ndarray[UINT8, ndim=2] digits = np.ndarray(shape=(length, MAX_DIGITS), dtype=np.uint8)
    cdef unsigned char[:,:] dv = digits
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned char c

    for r in range(length):
        y = fy[r]
        x = fx[r]

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
        fx: NDArray[np.uint64],
        fy: NDArray[np.uint64],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.uint8]:
    """

    :param fx:
    :param fy:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(fx), MAX_DIGITS + 1, dtype=np.uint8)
    if not len(fx) == len(fy) == len(lengths):
        raise ValueError('invalid lengths')
    return _encode_digits(fx, fy, lengths)

cdef np.ndarray[UINT64, ndim=2] _decode_digits(
        unsigned char[:,:] digits,
        unsigned long[:] lengths,
):
    cdef Py_ssize_t length = digits.shape[0]
    cdef Py_ssize_t r
    cdef np.ndarray[UINT64, ndim=2] fx = np.ndarray(shape=(length,), dtype=np.uint64)
    cdef np.ndarray[UINT64, ndim=2] fy = np.ndarray(shape=(length,), dtype=np.uint64)
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

cdef np.ndarray[F64, ndim=2] _encode_bounds(
        unsigned long[:] fx,
        unsigned long[:] fy,
        unsigned char[:] px,
        unsigned char[:] py,
):
    cdef float [:, ::1] bounds = np.ndarray(shape=(fx.size, 4), dtype=np.float64)
    cdef Py_ssize_t r
    cdef float w, s, n, e

    cdef const unsigned long row_sizes[GRID_LENGTH]
    cdef const unsigned long col_sizes[GRID_LENGTH]
    cdef const unsigned long pair_sizes[PAIR_LENGTH]

    for r in range(fx.size):
        w = <float> fx[r] / FINAL_LON_PRECISION - MAX_LON
        s = <float> fy[r] / FINAL_LAT_PRECISION - MAX_LAT
        e = <float> px[r] / FINAL_LON_PRECISION + w
        n = <float> py[r] / FINAL_LAT_PRECISION + s

        bounds[r, 0] = w
        bounds[r, 1] = s
        bounds[r, 2] = e
        bounds[r, 3] = n

    return bounds

def encode_bounds(
        fx: NDArray[np.uint64],
        fy: NDArray[np.uint64],
        px: NDArray[np.uint64],
        py: NDArray[np.uint64],
) -> NDArray[np.float64]:
    """

    :param fx:  finest precision integer longitude
    :param fy:  finest precision integer latitude
    :param px:
    :param py:
    :return:
    """
    if not len(fx) == len(fy) == len(px) == len(py):
        raise ValueError('invalid lengths')
    return _encode_bounds(fx, fy, px, py)

cdef np.ndarray[UINT8, ndim=1] _suggest_lengths(
        float [:, ::1] w,
        float [:, ::1] s,
        float [:, ::1] e,
        float [:, ::1] n,
        bint contains,
):
    ...


def suggest_lengths(
        w: NDArray[np.float64],
        s: NDArray[np.float64],
        e: NDArray[np.float64],
        n: NDArray[np.float64],
        contains: bool = True
) -> NDArray[np.uint8]:
    """

    :param w:
    :param s:
    :param e:
    :param n:
    :param contains:
    :return:
    """
    return _suggest_lengths(w, s, e, n, contains)
