import spatialpandas.geometry.base
from libc.stdlib cimport malloc, free
# from cpython cimport Py_buffer
from libcpp cimport vector
import numpy as np
cimport numpy as np
import geopandas as gpd
import shapely.geometry
import spatialpandas.geometry
import spatialpandas.geometry
import shapely.geometry
cimport util.cfuncs as cfuncs

np.string_t = np.dtype('S%d' % (MAX_DIGITS + 1))
ctypedef np.uint8_t UINT8
ctypedef np.uint16_t UINT16
ctypedef np.float64_t F64
ctypedef np.uint32_t UINT32
ctypedef np.uint64_t UINT64
ctypedef np.int64_t INT64
ctypedef np.uint_t UINT
ctypedef np.int_t INT
ctypedef np.uint8_t BOOL

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
    points: spatialpandas.geometry.PointArray
    footprint: spatialpandas.geometry.base.Geometry
    cdef unsigned char length

    cdef bint[:] visited
    cdef bint[:] contained
    cdef Py_ssize_t claimrows
    cdef Py_ssize_t claimcols
    cdef Py_ssize_t pointrows
    cdef Py_ssize_t pointcols
    cdef Py_ssize_t size

    cdef Py_ssize_t shape[2]
    cdef Py_ssize_t strides[2]

    cdef unsigned long[:, :] longs


    def __cinit__(self, footprint: shapely.geometry.base.BaseGeometry, unsigned char length):
        cdef Py_ssize_t r, c, i
        cdef double fw, fs, fe, fn

        fw, fs, fe, fn= footprint.bounds
        cdef unsigned long lw = <unsigned long> ((fw + MAX_LON) * FINAL_LON_PRECISION)
        cdef unsigned long le = <unsigned long> ((fe + MAX_LON) * FINAL_LON_PRECISION)
        cdef unsigned long ls = <unsigned long> ((fs + MAX_LAT) * FINAL_LAT_PRECISION)
        cdef unsigned long ln = <unsigned long> ( (fn + MAX_LAT) * FINAL_LAT_PRECISION)

        cdef unsigned long xstep = pow(GRID_COLUMNS, MAX_DIGITS - length)
        cdef unsigned long ystep = pow(GRID_ROWS, MAX_DIGITS - length)

        self.claimcols = le // xstep - lw // xstep
        self.claimrows = ln // ystep - ls // ystep
        self.pointrows = self.claimrows + 1
        self.pointcols = self.claimcols + 1

        # (left, right], (bottom, top]
        cdef np.ndarray[UINT64, ndim=1] dx = np.arange(xstep, self.pointcols * xstep + 1, xstep, dtype=np.uint64)
        cdef np.ndarray[UINT64, ndim=1] dy = np.arange(ystep, self.pointrows * ystep + 1, ystep, dtype=np.uint64)

        cdef longs = np.ndarray(shape=(self.pointrows * self.pointcols, 2), dtype=np.uint64)
        cdef unsigned long[:,:] longv = longs

        print(f'pointrows={self.pointrows} pointcols={self.pointcols}')
        for r in range(self.pointrows):
            for c in range(self.pointcols):
                longv[<Py_ssize_t> (r*self.pointcols+c), 0] = lw + dx[c]
                longv[<Py_ssize_t> (r*self.pointcols+c), 1] = ls + dy[r]


        cdef np.ndarray[F64, ndim=2] coords = np.ndarray(shape=(self.pointcols*self.pointrows, 2), dtype=np.float64)
        cdef double[:, :] floatv = coords
        cdef unsigned long final_lat_precision = PAIR_PRECISION * pow(GRID_ROWS, length)
        cdef unsigned long final_lon_precision = PAIR_PRECISION * pow(GRID_COLUMNS, length)
        cdef unsigned long trim_lats = pow(<unsigned long> GRID_ROWS, GRID_LENGTH-length)
        cdef unsigned long trim_lons = pow(<unsigned long> GRID_COLUMNS, GRID_LENGTH-length)

        print('populating floats')
        for r in range(self.pointrows):
            for c in range(self.pointcols):
                i = r*self.pointcols+c
                floatv[i, 0] = <double>(longv[i,0] // trim_lons) / final_lon_precision - MAX_LON
                floatv[i, 1] = <double>(longv[i,1] // trim_lats) / final_lat_precision - MAX_LAT
        print('done populating floats')

        self.length = length
        self.visited = np.full(shape=(self.pointrows*self.pointcols), fill_value=False, dtype=np.bool)
        self.contained = np.full(shape=(self.pointrows*self.pointcols), fill_value=False, dtype=np.bool)
        print('point array')
        self.points = spatialpandas.geometry.PointArray((coords[:, 0], coords[:, 1]))

        self.longs = longv
        print('cinit end')

    def __init__(self, footprint: shapely.geometry.base.BaseGeometry, unsigned char length):
        cdef Py_ssize_t r, c, n, i,
        print(f'init base start')
        self.recursion(0, 0, self.pointcols, self.pointrows)

        n = 0
        # replace buffer with only contained points
        for r in range(self.claimrows):
            for c in range(self.claimcols):
                i = r * self.pointcols + c
                if self.contained[i]:
                    self.longs[n, 0] = self.longs[i, 0]
                    self.longs[n, 1] = self.longs[i, 1]
                    n += 1
        # assert n == self.size
        print(f'init base end')


    def __len__(self):
        return self.size

    # def __getbuffer__(self, Py_buffer *buffer, int flags):
    #     cdef Py_ssize_t itemsize = sizeof(unsigned long)
    #     self.shape[0] = self.size
    #     self.shape[1] = 2
    #     self.strides[1] = <Py_ssize_t> (
    #         <char *>&(self.buf[1])
    #         - <char *> &(self.buf[0])
    #     )
    #     self.strides[0] = 2 * self.strides[1]
    #
    #     # buffer.buf = <char *>(self.v[0])
    #     buffer.buf = <char *>self.buf
    #     buffer.format = 'i' # TODO
    #     buffer.internal = NULL
    #     buffer.itemsize = itemsize
    #     buffer.len = self.size * 2 * itemsize
    #     buffer.ndim = 2
    #     buffer.obj = self
    #     buffer.readonly = 1
    #     buffer.shape = self.shape
    #     buffer.strides = self.strides
    #     buffer.suboffsets = NULL
    #
    # def __releasebuffer__(self, Py_buffer *buffer):
    #     pass


    cdef recursion(self, Py_ssize_t w, Py_ssize_t s, Py_ssize_t e, Py_ssize_t n):
        print(f'recursion {w} {s} {e} {n}')
        cdef Py_ssize_t sw, se, nw, ne, midx, midy, r, c

        if e == w and n == s:
            return # there is no tile

        sw = s * self.pointcols + w
        se = s * self.pointcols + e
        nw = n * self.pointcols + w
        ne = n * self.pointcols + e

        if not self.visited[sw]:
            self.contained[sw] = self.points[sw].intersects(self.footprint)
            self.visited[sw] = True
        if not self.visited[se]:
            self.contained[se] = self.points[se].intersects(self.footprint)
            self.visited[se] = True
        if not self.visited[nw]:
            self.contained[nw] = self.points[nw].intersects(self.footprint)
            self.visited[nw] = True
        if not self.visited[ne]:
            self.contained[ne] = self.points[ne].intersects(self.footprint)
            self.visited[ne] = True

        if  self.contained[sw] and self.contained[se] and self.contained[nw] and self.contained[ne] :
            # the box is completely contained in the polygon
            for r in range(s, n):
                for c in range(w, e):
                    self.contained[r * self.pointcols + c] = 1
            self.size += (e - w) * (n - s)
        else:
            midx = w + e
            midx = midx // 2 + (midx % 2)
            midy = s + n
            midy = midy // 2 + (midy % 2)
            if self.contained[sw]:
                self.recursion(w, s, midx, midy)
            if self.contained[se]:
                self.recursion(midx, s, e, midy)
            if self.contained[nw]:
                self.recursion(w, midy, midx, n)
            if self.contained[ne]:
                self.recursion(midx, midy, e, n)



cdef class PolygonClaim(ShapeClaim):
    footprint: spatialpandas.geometry.Polygon

    def __init__(self, footprint: shapely.geometry.Polygon, unsigned char length):
        print('init polygon start')
        self.footprint = spatialpandas.geometry.Polygon(footprint)
        super().__init__(footprint, length)

cdef class MultiPolygonClaim(ShapeClaim):
    footprint: spatialpandas.geometry.MultiPolygon

    def __init__(self, footprint: shapely.geometry.MultiPolygon, unsigned char length):
        print('init multipolygon')
        self.footprint = spatialpandas.geometry.MultiPolygon(footprint)
        super().__init__(footprint, length)


# cpdef get_claim( footprint, length: int):
cpdef np.ndarray[UINT64, ndim=2] get_claim(
        footprint: shapely.geometry.base.BaseGeometry,
        Py_ssize_t length,
):
    cdef Py_ssize_t size, r, c
    if isinstance(footprint, shapely.geometry.Polygon):
        claim = PolygonClaim(footprint, length)
    elif isinstance(footprint, shapely.geometry.MultiPolygon):
        claim = MultiPolygonClaim(footprint, length)
    else:
        raise ValueError('Unsupported geometry type: %s' % type(footprint))
    size = claim.size

    cdef np.ndarray[UINT64, ndim=2] longs = np.zeros((size, 2), dtype=np.uint64)
    cdef unsigned long[:, :] longv = longs
    for r in range(claim.claimrows):
        for c in range(claim.claimcols):
            i = r * claim.pointcols + c
            longv[i, 0] = claim.longs[i, 0]
            longv[i, 1] = claim.longs[i, 1]

    return longv






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
