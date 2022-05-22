# import math
if False | False:
    import math
import cython
import numpy
cimport numpy

ctypedef numpy.uint8_t DTYPE_t

# from libc.math cimport radians, asinh, tan, pi
cimport libc.math as math
import math as _math

if False | False:
    import math
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cdisplacement(
        numpy.ndarray[DTYPE_t, ndim=3] image,
        double xlen,
        double ylen,
        double slope,
        int rcam,
        int ccam,
        int rinc,
        int cinc,
):
    cdef int rwall = rcam
    cdef int cwall = ccam
    cdef double x
    cdef double y
    cdef double buffer = 0

    if slope > 1:
        slope = 1 / slope
        while 0 <= cwall <= 255 and 0 <= rwall <= 255:
            if image[rwall, cwall, 0]:
                # BAD: does the math, and then creates a double out of it

                y = (rwall - rcam)
                y = y / 255 * ylen

                x = (cwall - ccam)
                x = x / 255 * xlen

                return x, y
            elif buffer >= 1:
                buffer += -1
                cwall += cinc
            else:
                buffer += slope
                rwall += rinc
    else:
        while 0 <= cwall <= 255 and 0 <= rwall <= 255:
            if image[rwall, cwall, 0]:
                y = (rwall - rcam)
                y = y / 255 * ylen

                x = (cwall - ccam)
                x = x / 255 * xlen
                return x, y
            elif buffer >= 1:
                buffer += -1
                rwall += rinc
            else:
                buffer += slope
                cwall += cinc
    return numpy.nan, numpy.nan

def _deg2num(lat_deg, lon_deg, zoom):
    lat_rad = _math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = ((lon_deg + 180.0) / 360.0 * n)
    ytile = ((1.0 - _math.asinh(_math.tan(lat_rad)) / _math.pi) / 2.0 * n)
    return (xtile, ytile)

def _num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = _math.atan(_math.sinh(_math.pi * (1 - 2 * ytile / n)))
    lat_deg = _math.degrees(lat_rad)
    return (lat_deg, lon_deg)

cpdef deg2num(
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

cpdef num2deg(
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
