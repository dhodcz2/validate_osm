import math
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
    unsigned int GRID_LON_FIRST_PLACE_VALUE
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


cdef get_string(
        unsigned long lx,
        unsigned long ly,
        unsigned char length,
):
    cdef char* string = <char *> malloc(length * sizeof(char))
    cdef unsigned char c
    cdef const char* alphabet = ALPHABET
    cdef str s

    for c in range(MAX_DIGITS, PAIR_LENGTH, -1):
        string[c] = alphabet[
            ly % GRID_ROWS * GRID_COLUMNS
            + lx % GRID_COLUMNS
            ]
        lx //= GRID_COLUMNS
        ly //= GRID_ROWS

    for c in range(PAIR_LENGTH, SEP_POS, -2):
        string[c] = alphabet[lx % BASE]
        string[c - 1] = alphabet[ly % BASE]
        lx //= BASE
        ly //= BASE

    string[SEP_POS] = SEP

    for c in range(SEP_POS - 1, -1, -2):
        string[c] = alphabet[lx % BASE]
        string[c - 1] = alphabet[ly % BASE]
        lx //= BASE
        ly //= BASE

    s =  string[:length+1].decode('utf-8')
    free(string)
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef  np.ndarray get_strings(
        unsigned long[:] lx,
        unsigned long[:] ly,
        unsigned char[:] lengths
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
        strlen = lengths[i]

        for c in range(MAX_DIGITS, PAIR_LENGTH+1, -1):
            string[c] = alphabet[
                y % GRID_ROWS * GRID_COLUMNS
                + x % GRID_COLUMNS
            ]
            x //= GRID_COLUMNS
            y //= GRID_ROWS

        for c in range(PAIR_LENGTH+1, SEP_POS, -2):
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

        codes[i] = string[:strlen].decode('utf-8')

    free(string)
    return codes

cdef  unsigned char get_length(
        double fw,
        double fs,
        double fe,
        double fn,
):
    cdef unsigned char length = 0
    cdef unsigned char p

    cdef double row_degrees = GRID_SIZE_DEGREES
    cdef double col_degrees = GRID_SIZE_DEGREES
    cdef double dw = fe - fw
    cdef double dh = fn - fs
    cdef unsigned char c

    while col_degrees >= dw or row_degrees >= dh:
        length += 1
        col_degrees /= GRID_COLUMNS
        row_degrees /= GRID_ROWS
    if length > GRID_LENGTH:
        raise RuntimeError(f'length {length} is too large')

    length += PAIR_LENGTH

    return length

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef  np.ndarray[UINT8, ndim=1] get_lengths(
        double[:] fw,
        double[:] fs,
        double[:] fe,
        double[:] fn,
):
    cdef unsigned char[:] lengths = np.ndarray(shape=(fw.size,), dtype=np.uint8)

    cdef Py_ssize_t r
    for r in range(fw.size):
        lengths[r] = get_length(fw[r], fs[r], fe[r], fn[r], )
    return lengths

cdef  np.ndarray[UINT64, ndim=2] get_claim(
    double fw,
    double fs,
    double fe,
    double fn,
    unsigned char length,
):
    fw = (fw + MAX_LON) * FINAL_LON_PRECISION
    fe = (fe + MAX_LON) * FINAL_LON_PRECISION
    fs = (fs + MAX_LAT) * FINAL_LAT_PRECISION
    fn = (fn + MAX_LAT) * FINAL_LAT_PRECISION
    cdef unsigned long lw = <unsigned long> fw
    cdef unsigned long le = <unsigned long> fe
    cdef unsigned long ls = <unsigned long> fs
    cdef unsigned long ln = <unsigned long> fn

    cdef unsigned char trim_rows = pow(GRID_ROWS, MAX_DIGITS - length)
    cdef unsigned char trim_cols = pow(GRID_COLUMNS, MAX_DIGITS - length)

    lw //= trim_cols
    le //= trim_cols
    ln //= trim_rows
    ls //= trim_rows

    lw += 1
    ls += 1

    cdef np.ndarray[UINT64, ndim=2] claim = np.ndarray(shape=(length,2), dtype=np.uint64)

    cdef unsigned int repeat = len(range(lw, le))
    cdef unsigned int tile = len(range(ls, ln))
    claim[:, 0] = np.repeat(range(lw, le), repeat)
    claim[:, 1] = np.tile(range(ls, ln), tile)

    return claim



cdef get_bound(
    unsigned long lx,
    unsigned long ly,
    unsigned char length,
):
    cdef double fw = <double>(lx) / FINAL_LON_PRECISION
    cdef double fs = <double>(ly) / FINAL_LAT_PRECISION
    cdef double fe = fw
    cdef double fn = fs
    fn += GRID_SIZE_DEGREES / math.pow(GRID_ROWS, length-PAIR_LENGTH)
    fe += GRID_SIZE_DEGREES / math.pow(GRID_COLUMNS, length-PAIR_LENGTH)
    return (fw, fs, fe, fn)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef  np.ndarray[F64, ndim=2] get_bounds(
    # np.ndarray[UINT64, ndim=1] lx,
    # np.ndarray[UINT64, ndim=1] ly,
    unsigned long[:] lx,
    unsigned long[:] ly,
    unsigned char[:] lengths,
):
    cdef np.ndarray[F64, ndim=2] bounds = np.ndarray(shape=(lx.size, 4), dtype=np.float64)
    cdef Py_ssize_t r


    for r in range(lx.size):
        bounds[r, 0] = lx[r] / FINAL_LON_PRECISION
        bounds[r, 1] = ly[r] / FINAL_LAT_PRECISION
        bounds[r, 2] = bounds[r, 0] + GRID_SIZE_DEGREES / math.pow(GRID_ROWS, lengths[r]-PAIR_LENGTH)
        bounds[r, 3] = bounds[r, 1] + GRID_SIZE_DEGREES / math.pow(GRID_COLUMNS, lengths[r]-PAIR_LENGTH)
    return bounds
