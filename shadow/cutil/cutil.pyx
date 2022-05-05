import cv2
import cython
import numpy as np
cimport numpy as np
cimport libc.math as math

ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint_t UINT
ctypedef np.int_t INT
if False | False:
    import math

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef _load_image(
        np.ndarray[UINT8, ndim=1] cn,
        np.ndarray[UINT8, ndim=1] cw,
        np.ndarray[F64, ndim=1] weights,
        unsigned char cellsize,
        double absolute_max,
):
    cdef np.ndarray[UINT, ndim=2] grid = np.zeros((cellsize, cellsize), dtype=np.uint8)
    cdef unsigned char length = len(cw)
    cdef unsigned char k
    cdef unsigned short weight
    for k in range(length):
        weight = <unsigned short> (255.0 * weights[k] / absolute_max)
        grid[cn[k], cw[k]] = weight

    return cv2.resize(grid, dsize=(256, 256))

def load_image(
        cn: np.ndarray,
        cw: np.ndarray,
        weights: np.ndarray,
        cellsize: int,
        absolute_max: float
) -> np.ndarray:
    """

    :param cn:
    :param cw:
    :param weights:
    :param cellsize:
    :param absolute_max:
    :return: 256x256 image
    """
    return _load_image(cn, cw, weights, cellsize, absolute_max)

cdef _deg2num(
        double lat_deg,
        double lon_deg,
        unsigned int zoom
):
    cdef unsigned int n, xtile, ytile
    cdef double lat_rad
    # lat_rad = math.radians(lat_deg)
    lat_rad = lat_deg * math.pi / 180.0

    n = 2 ** zoom
    xtile = <unsigned int> ((lon_deg + 180) / 360 * n)
    ytile = <unsigned int> ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def deg2num(
        lat_deg: float,
        lon_deg: float,
        zoom: int
) -> tuple[int, int]:
    """

    :param lat_deg:
    :param lon_deg:
    :param zoom:
    :return: xtile, ytile
    """
    return _deg2num(lat_deg, lon_deg, zoom)

cdef _num2deg(
        double xtile,
        double ytile,
        unsigned int zoom
):
    cdef unsigned int n
    cdef double lon_deg, lat_rad, lat_deg
    n = 2 ** zoom
    lon_deg = xtile / n * 360 - 180
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytile / n)))
    # lat_deg = math.degrees(lat_rad)
    lat_deg = lat_rad * 180.0 / math.pi
    return lat_deg, lon_deg

def num2deg(xtile: float, ytile: float, zoom: int) -> tuple[float, float]:
    """

    :param xtile:
    :param ytile:
    :param zoom:
    :return: latitude, longitude
    """
    return _num2deg(xtile, ytile, zoom)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef _degs2nums(
        np.ndarray[F64, ndim=1] lat_degs,
        np.ndarray[F64, ndim=1] lon_degs,
        unsigned char zoom,
):
    cdef unsigned int length = len(lat_degs)
    cdef np.ndarray[UINT32, ndim = 1] xtiles = np.zeros((1, length), dtype=np.uint32)
    cdef np.ndarray[UINT32, ndim = 1] ytiles = np.zeros((1, length), dtype=np.uint32)
    cdef unsigned int n = 2 ** zoom
    cdef unsigned int k
    for k in range(length):
        lat_rad = lat_degs[k] * math.pi / 180.0
        xtile = <unsigned int> ((lon_degs[k] + 180) / 360 * n)
        ytile = <unsigned int> ((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        xtiles[k] = xtile
        ytiles[k] = ytile
    return xtiles, ytiles

def degs2nums(lat_degs: np.ndarray, lon_degs: np.ndarray, zoom: int):
    """

    :param lat_degs:
    :param lon_degs:
    :param zoom:
    :return: xtile, ytile
    """
    return _degs2nums(lat_degs, lon_degs, zoom)

@cython.boundscheck(False)
@cython.boundscheck(False)
cdef _nums2degs(
        np.ndarray[UINT32, ndim=1] xtiles,
        np.ndarray[UINT32, ndim=1] ytiles,
        unsigned int zoom,
):
    cdef unsigned int length = len(xtiles)
    cdef np.ndarray[F64, ndim = 1] lat_degs = np.zeros((1, length), dtype=np.float64)
    cdef np.ndarray[F64, ndim = 1] lon_degs = np.zeros((1, length), dtype=np.float64)
    cdef unsigned int n = 2 ** zoom
    cdef unsigned int k
    for k in range(length):
        lon_deg = 360.0 * xtiles[k] / n - 180
        lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * ytiles[k] / n)))
        lat_deg = lat_rad * 180.0 / math.pi
        lat_degs[k] = lat_deg
        lon_degs[k] = lon_deg
    return lat_degs, lon_degs

def nums2degs(xtiles: np.ndarray, ytiles: np.ndarray, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """

    :param xtiles:
    :param ytiles:
    :param zoom:
    :return: latitude, longitude
    """
    return _nums2degs(xtiles, ytiles, zoom)
