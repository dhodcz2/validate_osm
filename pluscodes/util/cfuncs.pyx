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
    double GRID_SIZE_DEGREES


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


cdef str _encode_string(
        unsigned long lx,
        unsigned long ly,
        unsigned char length,
):
    cdef char* string = <char *> malloc(length * sizeof(char))
    cdef unsigned char c
    cdef const char* alphabet = ALPHABET

    for c in range(MAX_DIGITS, PAIR_LENGTH, -1):
        string[c] = alphabet[
            ly % GRID_ROWS * GRID_COLUMNS
            + lx% GRID_COLUMNS
            ]
        lx//= GRID_COLUMNS
        ly //= GRID_ROWS

    for c in range(PAIR_LENGTH, SEP_POS, -2):
        string[c] = alphabet[lx% BASE]
        string[c - 1] = alphabet[ly % BASE]
        # codes[i, c] = lx% BASE
        # codes[i, c-1] = ly % BASE
        lx//= BASE
        ly //= BASE

    string[SEP_POS] = SEP

    for c in range(SEP_POS - 1, 0, -2):
        string[c] = alphabet[lx% BASE]
        string[c - 1] = alphabet[ly % BASE]
        # codes[i, c] = lx% BASE
        # codes[i, c-1] = ly % BASE
        lx//= BASE
        ly //= BASE

    return string[:length].decode('utf-8')

def encode_string(
        x: int | float,
        y: int | float,
        unsigned char length,
):
    if isinstance(x, float):
        x = int((x + MAX_LON) * FINAL_LON_PRECISION)
    if isinstance(y, float):
        y = int((y + MAX_LAT) * FINAL_LAT_PRECISION)
    return _encode_string(x, y, length)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef  _encode_strings(
        unsigned long[:] lx,
        unsigned long[:] ly,
        unsigned long[:] code_lengths,
):
    cdef char* string = <char *> malloc((MAX_DIGITS + 1) * sizeof(char))
    cdef unsigned long length = lx.size
    cdef const char* alphabet = ALPHABET
    codes = np.ndarray(shape=(length,), dtype='U%d' % (MAX_DIGITS+1))
    cdef unsigned long x
    cdef unsigned long y
    cdef unsigned int i
    cdef unsigned char c
    cdef ssize_t strlen


    for i in range(length):
        x = lx[i]
        y = ly[i]
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

def encode_strings(
        x: NDArray[np.uint64],
        y: NDArray[np.uint64],
        lengths: NDArray[np.uint8] = None,
) -> NDArray[np.string_t]:
    """
    :param x:
    :param y:
    :param lengths:
    :return:
    """
    if lengths is None:
        lengths = np.full(len(x), 11, dtype=np.uint8)
    if not len(x) == len(y) == len(lengths):
        raise ValueError('invalid lengths')
    if x.dtype != np.uint64:
        x = np.ndarray.astype((x + MAX_LON) * FINAL_LON_PRECISION, np.uint64)
    if y.dtype != np.uint64:
        y = np.ndarray.astype((y + MAX_LAT) * FINAL_LAT_PRECISION, np.uint64)
    return _encode_strings(x, y, lengths)

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cdef np.ndarray[UINT8, ndim=2] _encode_digits(
#         unsigned long[:] fx,
#         unsigned long[:] fy,
#         unsigned long[:] lengths,
# ):
#     cdef Py_ssize_t length = fx.size
#     cdef Py_ssize_t row
#     cdef np.ndarray[UINT8, ndim=2] digits = np.ndarray(shape=(length, MAX_DIGITS), dtype=np.uint8)
#     cdef unsigned char[:,:] dv = digits
#     cdef unsigned long x
#     cdef unsigned long y
#     cdef unsigned char c
#
#     for r in range(length):
#         y = fy[r]
#         x = fx[r]
#
#         for c in range(MAX_DIGITS-1, PAIR_LENGTH-1, -1):
#             dv[r,c] = (
#                 y % GRID_ROWS * GRID_COLUMNS
#                 + x % GRID_COLUMNS
#             )
#             x //= GRID_COLUMNS
#             y //= GRID_ROWS
#
#         for c in range(PAIR_LENGTH -1, 0, -2):
#             dv[r,c] = x % BASE
#             dv[r,c-1] = y % BASE
#             x //= BASE
#             y //= BASE
#
#     return digits
#
#
#
# def encode_digits(
#         fx: NDArray[np.uint64],
#         fy: NDArray[np.uint64],
#         lengths: NDArray[np.uint8] = None,
# ) -> NDArray[np.uint8]:
#     """
#
#     :param fx:
#     :param fy:
#     :param lengths:
#     :return:
#     """
#     if lengths is None:
#         lengths = np.full(len(fx), MAX_DIGITS + 1, dtype=np.uint8)
#     if not len(fx) == len(fy) == len(lengths):
#         raise ValueError('invalid lengths')
#     return _encode_digits(fx, fy, lengths)
#

# cdef np.ndarray[UINT64, ndim=2] _decode_digits(
#         unsigned char[:,:] digits,
#         unsigned long[:] lengths,
# ):
#     cdef Py_ssize_t length = digits.shape[0]
#     cdef Py_ssize_t r
#     cdef np.ndarray[UINT64, ndim=2] fx = np.ndarray(shape=(length,), dtype=np.uint64)
#     cdef np.ndarray[UINT64, ndim=2] fy = np.ndarray(shape=(length,), dtype=np.uint64)
#     cdef unsigned long x
#     cdef unsigned long y
#     cdef unsigned char c
#     ...
#
# def decode_digits(
#         digits: NDArray[np.uint8],
#         lengths: NDArray[np.uint8] = None,
# ) -> NDArray[np.uint64]:
#     """
#
#     :param digits:
#     :param lengths:
#     :return:
#     """
#     if lengths is None:
#         lengths = np.full(len(digits), MAX_DIGITS + 1, dtype=np.uint8)
#     if not len(digits) == len(lengths):
#         raise ValueError('invalid lengths')
#     return _decode_digits(digits, lengths)

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
#
# cdef unsigned char _suggest_length_where_box_contains(
#         unsigned long fw,
#         unsigned long fs,
#         unsigned long fe,
#         unsigned long fn,
#         bint contains,
# ):
#     cdef unsigned long lw = (fw + MAX_LON) * FINAL_LON_PRECISION
#     cdef unsigned long le = (fe + MAX_LON) * FINAL_LON_PRECISION
#     cdef unsigned long ls = (fs + MAX_LAT) * FINAL_LAT_PRECISION
#     cdef unsigned long ln = (fn + MAX_LAT) * FINAL_LAT_PRECISION
#     cdef unsigned char length = 0
#     cdef unsigned char p
#
#     # What is the largest length such that all precisions are matching?
#     for p in range(GRID_LENGTH):
#         if not (
#             le % GRID_COLUMNS == lw % GRID_COLUMNS
#             and ln % GRID_ROWS == ls % GRID_ROWS
#         ):
#             break
#         length += 1
#
#         le //= GRID_COLUMNS
#         lw //= GRID_COLUMNS
#         ln //= GRID_ROWS
#         ls //= GRID_ROWS
#
#     return length
#
#

cdef unsigned char _suggest_length_where_box_contained(
        double fw,
        double fs,
        double fe,
        double fn,
        bint contains,
):
    cdef unsigned long lw = (fw + MAX_LON) * FINAL_LON_PRECISION
    cdef unsigned long le = (fe + MAX_LON) * FINAL_LON_PRECISION
    cdef unsigned long ls = (fs + MAX_LAT) * FINAL_LAT_PRECISION
    cdef unsigned long ln = (fn + MAX_LAT) * FINAL_LAT_PRECISION
    cdef unsigned char length = 0
    cdef unsigned char p

    # TODO: THis one doesn't work. The bounds can be split across two PAIR tiles
    # # What is the largest length such that either  doesn't match?
    # for p in range(GRID_LENGTH -1):
    #     length += 1
    #     if not (
    #             le % GRID_COLUMNS == lw % GRID_COLUMNS
    #             and ln % GRID_ROWS == ls % GRID_ROWS
    #     ):
    #         break
    #     le //= GRID_COLUMNS
    #     lw //= GRID_COLUMNS
    #     ln //= GRID_ROWS
    #     ls //= GRID_ROWS
    #
    cdef row_degrees = GRID_SIZE_DEGREES
    cdef col_degrees = GRID_SIZE_DEGREES
    cdef double dw = le - lw
    cdef double dh = ln - ls
    cdef unsigned char c
    for c in range(GRID_LENGTH):
        length += 1
        if dw < col_degrees and dh < row_degrees:
            break
        row_degrees //= GRID_ROWS
        col_degrees //= GRID_COLUMNS



    return length
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.nonecheck(False)
# cdef np.ndarray[UINT8, ndim=1] _suggest_lengths(
#         unsigned long[:] fw,
#         unsigned long[:] fs,
#         unsigned long[:] fe,
#         unsigned long[:] fn,
#         bint contains,
# ):
#     cdef np.ndarray[UINT8, ndim=1] lengths = np.ndarray(shape=(fw.size,), dtype=np.uint8)
#     cdef Py_ssize_t r
#
#     if contains:
#         for r in range(fw.size):
#             lengths[r] = _suggest_length_where_box_contains( fw[r], fs[r], fe[r], fn[r], contains, )
#     else:
#         for r in range(fw.size):
#             lengths[r] = _suggest_length_where_box_contained( fw[r], fs[r], fe[r], fn[r], contains, )
#
#     return contains
#
#
# def suggest_lengths(
#         fw: NDArray[np.float64],
#         fs: NDArray[np.float64],
#         fe: NDArray[np.float64],
#         fn: NDArray[np.float64],
#         contains: bool = True
# ) -> NDArray[np.uint8]:
#     """
#
#     :param fw:
#     :param fs:
#     :param fe:
#     :param fn:
#     :param contains:
#     :return:
#     """
#     return _suggest_lengths(fw, fs, fe, fn, contains)



cdef np.ndarray[np.uint64, ndim=1] _get_claim(
    double fw,
    double fs,
    double fe,
    double fn,
    unsigned char length,
):
    # TODO: Get all UBIDs that intersect the bounds
    cdef unsigned long lw = (fw + MAX_LON) * FINAL_LON_PRECISION
    cdef unsigned long le = (fe + MAX_LON) * FINAL_LON_PRECISION
    cdef unsigned long ls = (fs + MAX_LAT) * FINAL_LAT_PRECISION
    cdef unsigned long ln = (fn + MAX_LAT) * FINAL_LAT_PRECISION
    cdef unsigned char trim_rows = pow(GRID_ROWS, MAX_DIGITS - length)
    cdef unsigned char trim_cols = pow(GRID_COLUMNS, MAX_DIGITS - length)

    lw //= GRID_COLUMNS
    le //= GRID_COLUMNS
    ln //= GRID_ROWS
    ls //= GRID_ROWS

    cdef unsigned int rows = len(range(le-lw)) * len(range(ln-ls))
    cdef unsigned long[:] lx = np.repeat(range(le-lw), len(range(ln-ls)))
    cdef unsigned long[:] ly = np.tile(range(ln-ls), len(range(le-lw)))

    return _encode_strings(lx, ly, length)

def get_claim(
    fw: float,
    fs: float,
    fe: float,
    fn: float,
    length: int = 11,
) -> NDArray[np.uint64]:
    """

    :param fw:
    :param fs:
    :param fe:
    :param fn:
    :param length:
    :return:
    """
    return _get_claim(fw, fs, fe, fn, length)
