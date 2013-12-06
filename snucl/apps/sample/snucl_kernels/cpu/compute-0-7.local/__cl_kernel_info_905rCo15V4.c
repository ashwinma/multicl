const unsigned int _cl_kernel_num = 1;

const char* _cl_kernel_names[] = {"sample"};

const unsigned int _cl_kernel_num_args[] = {3};

const char* _cl_kernel_attributes[] = {""};

const unsigned int _cl_kernel_work_group_size_hint[][3] = {{0, 0, 0}};

const unsigned int _cl_kernel_reqd_work_group_size[][3] = {{0, 0, 0}};

const unsigned long long _cl_kernel_local_mem_size[] = {0};

const unsigned long long _cl_kernel_private_mem_size[] = {4};

/* CL_KERNEL_ARG_ADDRESS_GLOBAL  : 3
   CL_KERNEL_ARG_ADDRESS_CONSTANT: 2
   CL_KERNEL_ARG_ADDRESS_LOCAL   : 1
   CL_KERNEL_ARG_ADDRESS_PRIVATE : 0 */
const char* _cl_kernel_arg_address_qualifier[] = {"330"};

/* CL_KERNEL_ARG_ACCESS_READ_ONLY : 3
   CL_KERNEL_ARG_ACCESS_WRITE_ONLY: 2
   CL_KERNEL_ARG_ACCESS_READ_WRITE: 1
   CL_KERNEL_ARG_ACCESS_NONE      : 0 */
const char* _cl_kernel_arg_access_qualifier[] = {"000"};

const char* _cl_kernel_arg_type_name[] = {"int*", "int*", "int"};

/* CL_KERNEL_ARG_TYPE_NONE    : 0x0
   CL_KERNEL_ARG_TYPE_CONST   : 0x1
   CL_KERNEL_ARG_TYPE_RESTRICT: 0x2
   CL_KERNEL_ARG_TYPE_VOLATILE: 0x4 */
const unsigned int _cl_kernel_arg_type_qualifier[] = {0, 0, 0};

const char* _cl_kernel_arg_name[] = {"dst", "src", "offset"};

const unsigned int _cl_kernel_dim[] = {3};

