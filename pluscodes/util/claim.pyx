import math

from cpython cimport Py_buffer
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
import geopandas as gpd
import shapely.geometry
import spatialpandas.geometry

np.string_t = np.dtype('S%d' % (MAX_DIGITS + 1))
ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint64_t UINT64
ctypedef np.int64_t INT64
ctypedef np.uint_t UINT
ctypedef np.int_t INT
ctypedef np.bool_t BOOL

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


cdef class ShapeClaim:
    cdef spatialpandas.geometry.base.Geometry footprint
    cdef unsigned char length

    cdef bint[:] visited
    cdef bint[:] contained
    # cdef spatialpandas.geometry.PointArray points
    points: spatialpandas.geometry.PointArray
    cdef Py_ssize_t nrows
    cdef Py_ssize_t ncols
    cdef bint initialized = 0
    cdef Py_ssize_t vrows
    cdef unsigned long[:] lw
    cdef unsigned long[:] ls

    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]
    cdef vector[unsigned long] v
    # cdef vector[float[2]] v


    def __cinit__(self, shapely.geometry.base.BaseGeometry footprint, unsigned char length):
        cdef double fw, fs, fe, fn = footprint.bounds

        cdef unsigned long w = <unsigned long> ((fw + MAX_LON) * FINAL_LON_PRECISION)
        cdef unsigned long e = <unsigned long> ((fe + MAX_LON) * FINAL_LON_PRECISION)
        cdef unsigned long s = <unsigned long> ((fs + MAX_LAT) * FINAL_LAT_PRECISION)
        cdef unsigned long n = <unsigned long> ( (fn + MAX_LAT) * FINAL_LAT_PRECISION)

        cdef unsigned long xstep = pow(GRID_COLUMNS, MAX_DIGITS - length)
        cdef unsigned long ystep = pow(GRID_ROWS, MAX_DIGITS - length)
        cdef unsigned long dx = e // xstep - w // xstep
        cdef unsigned long dy = n // ystep - s // ystep

        # (leftmost long, rightmost long), (topmost long, bottommost long) exclusive
        # cdef np.ndarray[UINT64, ndim=1] lx = np.arange(xstep, dx*xstep, xstep, dtype=np.uint64)
        # cdef np.ndarray[UINT64, ndim=1] ly = np.arange(ystep, dy*ystep, ystep, dtype=np.uint64)
        self.lx = np.arange(xstep, dx*xstep, xstep, dtype=np.uint64)
        self.ly = np.arange(ystep, dy*ystep, ystep, dtype=np.uint64)

        # self.footprint = spatialpandas.geometry.Polygon(footprint)
        self.length = length
        self.nrows = e - w
        self.ncols = n - s
        self.visited = np.full(shape=(self.nrows*self.ncols,), fill_value=False, dtype=np.bool)
        self.contained = np.full(shape=(self.nrows*self.ncols,), fill_value=False, dtype=np.bool)

    def __init__(self, shapely.geometry.base.BaseGeometry footprint, unsigned char length):
        # TODO: check for footprints


        cdef Py_ssize_t r, c, i
        self.recursion(0, 0, self.ncols, self.nrows)
        self.initialized = 1

        # TODO: populate vector
        self.v.resize(self.vrows)

        for r in range(self.nrows):
            for c in range(self.ncols):
                if self.contained[r * self.ncols + c]:
                    self.v[i][0] = self.lx[c]
                    self.v[i][1] = self.ly[r]
                    i += 1

    def __len__(self):
        return self.vrows

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        cdef Py_ssize_t itemsize = sizeof(self.v[0])
        self.shape[0] = self.vrows
        self.shape[1] = 2
        self.strides[1] = <Py_ssize_t> (
            <char *>&(self.v[1]) - <char *> &(self.v[0])
        )
        self.strides[0] = 2 * self.strides[1]

        buffer.buf = <char *>(self.v[0])
        buffer.format = 'i' # TODO
        buffer.internal = NULL
        buffer.itemsize = itemsize
        buffer.len = self.v.size() * itemsize
        buffer.ndim = 2
        buffer.obj = self
        buffer.readonly = 1
        buffer.shape = self.shape
        buffer.strides = self.strides
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        pass

    cdef base_case(self, Py_ssize_t s, Py_ssize_t w):
        cdef Py_ssize_t p = s * self.ncols + w
        if not self.visited[p]:
            self.visited[p] = 1
            self.contained[p] = self.points[p].intersects(self.footprint)
        if self.contained[p]:
            self.vrows += 1


    cdef recursion(self, Py_ssize_t w, Py_ssize_t s, Py_ssize_t e, Py_ssize_t n):
        cdef Py_ssize_t i, j
        cdef Py_ssize_t sw, se, nw, ne

        if e == w + 1 and n == s + 1:
            self.base_case(s, w)
            return

        sw = s * self.ncols + w
        se = s * self.ncols + e
        nw = n * self.ncols + w
        ne = n * self.ncols + e

        if not self.visited[sw]:
            self.contained[sw] = self.points[sw].intersects(self.footprint)
        if not self.visited[se]:
            self.contained[se] = self.points[se].intersects(self.footprint)
        if not self.visited[nw]:
            self.contained[nw] = self.points[nw].intersects(self.footprint)
        if not self.visited[ne]:
            self.contained[ne] = self.points[ne].intersects(self.footprint)

        cdef unsigned int midx
        cdef unsigned int midy

        if  self.contained[sw] and self.contained[se] and self.contained[nw] and self.contained[ne] :
            # Case 1: the box is completely contained in the polygon
            for i in range(s, n):
                for j in range(w, e):
                    # self.visited[i * self.ncols + j] = 1
                    self.contained[i * self.ncols + j] = 1
            self.vrows += (e - w) * (n - s)
        elif not ( self.contained[sw] or self.contained[se] or self.contained[nw] or self.contained[ne] ):
            # Case 2: the box is completely outside the polygon
            pass
        else:
            # Case 3: the box is partially contained in the polygon
            midx = (w + e)
            midx = midx // 2 + (midx % 2)
            midy = (s + n)
            midy = midy // 2 + (midy % 2)
            self.recursion(w, s, midx, midy)
            self.recursion(midx, s, e, midy)
            self.recursion(midx, midy, e, n)
            self.recursion(w, midy, midx, n)









cdef class PolygonClaim(ShapeClaim):
    cdef spatialpandas.geometry.Polygon footprint
    def __init__(self, shapely.geometry.base.BaseGeometry footprint, unsigned char length):
        self.footprint = spatialpandas.geometry.Polygon(footprint)
        super().__init__(footprint, length)


cdef class MultiPolygonClaim(ShapeClaim):
    cdef spatialpandas.geometry.MultiPolygon footprint

    def __init__(self, shapely.geometry.base.BaseGeometry footprint, unsigned char length):
        self.footprint = spatialpandas.geometry.MultiPolygon(footprint)
        super().__init__(footprint, length)


cdef get_claim(
    footprint: shapely.geometry.base.BaseGeometry,
    unsigned int length,
):
    if isinstance(footprint, shapely.geometry.Polygon):
        claim = PolygonClaim(footprint, length)
    elif isinstance(footprint, shapely.geometry.MultiPolygon):
        claim = MultiPolygonClaim(footprint, length)
    else:
        raise ValueError('Unsupported geometry type: %s' % type(footprint))




"""
results: np.ndarray[object] = np.ndarray(len(footprints), dtype=object)
for l, f in zip(lengths, footprints):
    results[l] = FactoryClaim(f, l)
sum = 0
np.ndarray[UINT64] claims = np.ndarray(sum, dtype=np.uint64)
i = 0
for r in results:
    claims[i] = r
    i += 1
    


"""
