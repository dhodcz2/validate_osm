# TODO: seems like _geos is not getting its dependencies
cdef extern from 'geos_c.h':
    ctypedef void *GEOSContextHandle_t
    ctypedef struct GEOSGeometry
    ctypedef struct GEOSPreparedGeometry
    ctypedef void (*GEOSMessageHandler_r)(const char *message, void *data)

    GEOSContextHandle_t GEOS_init_r() nogil
    void GEOS_finish_r(GEOSContextHandle_t handle) nogil
    void GEOSContext_setErrorMessageHandler_r(
            GEOSContextHandle_t handle,
            GEOSMessageHandler_r ef,
            void* data,
    ) nogil
    void GEOSContext_setNoticeMessageHandler_r(
            GEOSContextHandle_t handle,
            GEOSMessageHandler_r nf,
            void* data,
    ) nogil

    GEOSGeometry* GEOSGeom_createPointFromXY_r(GEOSContextHandle_t, double, double) nogil
    GEOSPreparedGeometry* GEOSPrepare_r(GEOSContextHandle_t, const GEOSGeometry *) nogil

    char GEOSIntersects_r(GEOSContextHandle_t, const GEOSGeometry*, const GEOSGeometry *) nogil
    char GEOSPreparedIntersects_r(GEOSContextHandle_t, const GEOSPreparedGeometry*, const GEOSGeometry *) nogil

    void GEOSGeom_destroy_r(GEOSContextHandle_t, GEOSGeometry *) nogil
    void GEOSPreparedGeom_destroy_r(GEOSContextHandle_t, GEOSPreparedGeometry *) nogil

cdef class get_geos_handle:
    cdef GEOSContextHandle_t handle
    cdef char* last_error
    cdef char* last_warning
    cdef GEOSContextHandle_t __enter__(self)
