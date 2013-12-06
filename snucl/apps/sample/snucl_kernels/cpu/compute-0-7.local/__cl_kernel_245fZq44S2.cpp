#include <cpu_launch_header.h>

__kernel void sample(__global int* dst, __global int* src, int offset);

static const char* KernelNames[] = {
  "sample"
};

TLB_STR_FUNC_START
TLS struct Function_0 { int* dst; int* src; int offset; void(*_p2W8Ht)(int*, int*, int); } func_0;
TLB_STR_FUNC_END

inline void kernel_launch() {
  TLB_GET_KEY;
  TLB_GET_KEY_FUNC;
  switch (TLB_GET(kernel_idx)) {
    case 0: (TLB_GET_FUNC(func_0))._p2W8Ht((TLB_GET_FUNC(func_0)).dst, (TLB_GET_FUNC(func_0)).src, (TLB_GET_FUNC(func_0)).offset); break;
  }
}

inline void kernel_set_stack(unsigned int* num, unsigned int* size) {
  TLB_GET_KEY;
  switch (TLB_GET(kernel_idx)) {
    case 0:
       *num = __CL_KERNEL_DEFAULT_STACK_NUM;
       *size = __CL_KERNEL_DEFAULT_STACK_SIZE;
       break;
    default: CL_DEV_ERR("[S%d] error %d\n", TLB_GET(cu_id),__LINE__);
  }
}

inline void kernel_set_arguments(void) {
  TLB_GET_KEY;
  TLB_GET_KEY_FUNC;
  CL_SET_ARG_INIT();
  switch (TLB_GET(kernel_idx)) {
    case 0: { (TLB_GET_FUNC(func_0))._p2W8Ht = sample; CL_SET_ARG_GLOBAL(TLB_GET_FUNC(func_0), int*, dst); CL_SET_ARG_GLOBAL(TLB_GET_FUNC(func_0), int*, src); CL_SET_ARG_PRIVATE(TLB_GET_FUNC(func_0), int, offset); } break;
    default: CL_DEV_ERR("[S%d] error %d\n", TLB_GET(cu_id),__LINE__);
  }
}

inline void kernel_fini(void) {
  TLB_GET_KEY;
  TLB_GET_KEY_FUNC;
   switch (TLB_GET(kernel_idx)) {
    case 0: break;
    default: CL_DEV_ERR("[S%d] error %d\n", TLB_GET(cu_id),__LINE__);
  }
}

#include <cpu_launch_footer.h>
