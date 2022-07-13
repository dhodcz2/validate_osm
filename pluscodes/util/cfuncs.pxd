import numpy as np
cimport numpy as np
# ctypedef np.uint8_t UINT8
# ctypedef np.uint64_t UINT64

cdef get_string(unsigned long lx, unsigned long ly, unsigned char length)
cdef np.ndarray get_strings( unsigned long[:] lx, unsigned long[:] ly, unsigned char[:] lengths )
cdef unsigned char get_length(double fw, double fs, double fe, double fn)
cdef np.ndarray get_lengths(double[:] fw, double[:] fs, double[:] fe, double[:] fn)
cdef np.ndarray get_claim(double fw, double fs, double fe, double fn, unsigned char length)
cdef get_bound(unsigned long lx, unsigned long ly, unsigned char length)
cdef np.ndarray get_bounds(unsigned long[:] lx, unsigned long[:] ly, unsigned char[:] lengths)
cdef np.ndarray get_points(unsigned long[:] lx, unsigned long[:] ly, unsigned char[:] lengths)
