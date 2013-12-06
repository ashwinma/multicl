#include <cpu_kernel_header.h>

__kernel void sample(__global int* dst, __global int* src, int offset) {
  TLB_GET_KEY;
  int id;
  for (TLB_GET(__k) = 0; TLB_GET(__k) < TLB_GET(__local_size[2]); TLB_GET(__k)++) {
    for (TLB_GET(__j) = 0; TLB_GET(__j) < TLB_GET(__local_size[1]); TLB_GET(__j)++) {
      for (TLB_GET(__i) = 0; TLB_GET(__i) < TLB_GET(__local_size[0]); TLB_GET(__i)++) {
        id = get_global_id(0);
        dst[id] = src[id] + offset;
      }
    }
  }
}


#if 0
1
"sample"
TLS struct Function_# { int* dst; int* src; int offset; void(*_p2W8Ht)(int*, int*, int); } func_#;
    case #: (TLB_GET_FUNC(func_#))._p2W8Ht((TLB_GET_FUNC(func_#)).dst, (TLB_GET_FUNC(func_#)).src, (TLB_GET_FUNC(func_#)).offset); break;
    case #:
    case #: { (TLB_GET_FUNC(func_#))._p2W8Ht = sample; CL_SET_ARG_GLOBAL(TLB_GET_FUNC(func_#), int*, dst); CL_SET_ARG_GLOBAL(TLB_GET_FUNC(func_#), int*, src); CL_SET_ARG_PRIVATE(TLB_GET_FUNC(func_#), int, offset); } break;
    case #: break;
"sample"
3
""
{0, 0, 0}
{0, 0, 0}
0
4
"330"
"000"
"int*", "int*", "int"
0, 0, 0
"dst", "src", "offset"
3
#endif
#if 0
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned int _cl_kernel_num = 1;
/* SNUCL_KERNEL_COMPILE_INFO */const char* _cl_kernel_names[] = {"sample"};
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned int _cl_kernel_num_args[] = {3};
/* SNUCL_KERNEL_COMPILE_INFO */const char* _cl_kernel_attributes[] = {""};
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned int _cl_kernel_work_group_size_hint[][3] = {{0, 0, 0}};
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned int _cl_kernel_reqd_work_group_size[][3] = {{0, 0, 0}};
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned long long _cl_kernel_local_mem_size[] = {0};
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned long long _cl_kernel_private_mem_size[] = {4};
/* SNUCL_KERNEL_COMPILE_INFO */const char* _cl_kernel_arg_address_qualifier[] = {"330"};
/* SNUCL_KERNEL_COMPILE_INFO */const char* _cl_kernel_arg_access_qualifier[] = {"000"};
/* SNUCL_KERNEL_COMPILE_INFO */const char* _cl_kernel_arg_type_name[] = {"int*", "int*", "int"};
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned int _cl_kernel_arg_type_qualifier[] = {0, 0, 0};
/* SNUCL_KERNEL_COMPILE_INFO */const char* _cl_kernel_arg_name[] = {"dst", "src", "offset"};
/* SNUCL_KERNEL_COMPILE_INFO */const unsigned int _cl_kernel_dim[] = {3};
#endif
