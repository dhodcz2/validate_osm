import cython
import numpy
cimport numpy
cimport cython

ctypedef numpy.uint8_t DTYPE_t

# TODO: Instead of computing ccam in displacement(), compute it in _displacement()
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
                # y = (rwall - rcam) / 255 * ylen
                # x = (cwall - ccam) / 255 * xlen

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
                # y = (rwall - rcam) / 255 * ylen
                # x = (cwall - ccam) / 255 * xlen
                return x, y
            elif buffer >= 1:
                buffer += -1
                rwall += rinc
            else:
                buffer += slope
                cwall += cinc
    return numpy.nan, numpy.nan

def test(
        int first,
        int second
):
    return first + second
