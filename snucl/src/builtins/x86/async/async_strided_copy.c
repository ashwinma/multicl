#include <cl_cpu_ops.h>

#define SNUCL_ASYNC_STRIDE_G2L(type)  \
event_t async_work_group_strided_copy_g2l(type *dst, const type *src, size_t num_elements, size_t src_stride, event_t event) { \
  for (size_t i = 0, j = 0; j < num_elements; i += src_stride, j++) \
	  dst[j] = src[i];                             \
  return event;                                  \
}

#define SNUCL_ASYNC_STRIDE_L2G(type)  \
event_t async_work_group_strided_copy_l2g(type *dst, const type *src, size_t num_elements, size_t dst_stride, event_t event) { \
  for (size_t i = 0, j = 0; j < num_elements; i += dst_stride, j++) \
	  dst[i] = src[j];                             \
  return event;                                  \
}

#include "async_strided_copy.inc"
