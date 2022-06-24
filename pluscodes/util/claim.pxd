cdef class ShapeClaim:
    cdef base_case(self, Py_ssize_t s, Py_ssize_t w)
    cdef recursion(self, Py_ssize_t w, Py_ssize_t s, Py_ssize_t e, Py_ssize_t n)
    cdef process(self)

cdef class PolygonClaim(ShapeClaim):
    pass

cdef class MultiPolygonClaim(ShapeClaim):
    pass
