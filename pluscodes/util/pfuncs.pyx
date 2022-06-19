print('wtf')
import os

import numpy as np
cimport numpy as np
from numpy.typing import NDArray
print(os.getcwd())
from . cimport cfuncs

cdef extern from '<util/globals.h>':
    char SEP
    unsigned int SEP_POS
    char * ALPHABET
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

def get_string(
        x: int | float,
        y: int | float,
        unsigned char length,
) -> str:
    if isinstance(x, float):
        x = int((x + MAX_LON) * FINAL_LON_PRECISION)
    if isinstance(y, float):
        y = int((y + MAX_LAT) * FINAL_LAT_PRECISION)
    return cfuncs.get_string(x, y, length)

def get_strings(
        x: NDArray[np.uint64, np.float64],
        y: NDArray[np.uint64, np.float64],
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
    if x.dtype == np.float64:
        x = np.ndarray.astype((x + MAX_LON) * FINAL_LON_PRECISION, np.uint64)
    if y.dtype == np.float64:
        y = np.ndarray.astype((y + MAX_LAT) * FINAL_LAT_PRECISION, np.uint64)
    return cfuncs.get_strings(x, y, lengths)

def get_length(
        double fw,
        double fs,
        double fe,
        double fn,
) -> int:
    return cfuncs.get_length(fw, fs, fe, fn)

def get_lengths(
        fw: NDArray[np.float64],
        fs: NDArray[np.float64],
        fe: NDArray[np.float64],
        fn: NDArray[np.float64],
) -> NDArray[np.uint8]:
    return cfuncs.get_lengths(fw, fs, fe, fn)

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
    return cfuncs.get_claim(fw, fs, fe, fn, length)

def get_bound(
        x: int | float,
        y: int | float,
        length: int = 11,
) -> tuple[float, float, float, float]:
    """

    :param x:
    :param y:
    :param length:
    :return:
    """
    if isinstance(x, float):
        x = int((x + MAX_LON) * FINAL_LON_PRECISION)
    if isinstance(y, float):
        y = int((y + MAX_LAT) * FINAL_LAT_PRECISION)
    return cfuncs.get_bound(x, y, length)

def get_bounds(
        x: NDArray[np.float64] | NDArray[np.uint64],
        y: NDArray[np.float64] | NDArray[np.uint64],
        lengths: NDArray[np.uint8],
) -> NDArray[np.float64]:
    if x.dtype == np.float64:
        x = np.ndarray.astype((x + MAX_LON) * FINAL_LON_PRECISION, np.uint64)
    if y.dtype == np.float64:
        y = np.ndarray.astype((y + MAX_LAT) * FINAL_LAT_PRECISION, np.uint64)
    return cfuncs.get_bounds(x, y, lengths)
