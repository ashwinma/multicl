/*****************************************************************************/
/*                                                                           */
/* Copyright (c) 2011-2013 Seoul National University.                        */
/* All rights reserved.                                                      */
/*                                                                           */
/* Redistribution and use in source and binary forms, with or without        */
/* modification, are permitted provided that the following conditions        */
/* are met:                                                                  */
/*   1. Redistributions of source code must retain the above copyright       */
/*      notice, this list of conditions and the following disclaimer.        */
/*   2. Redistributions in binary form must reproduce the above copyright    */
/*      notice, this list of conditions and the following disclaimer in the  */
/*      documentation and/or other materials provided with the distribution. */
/*   3. Neither the name of Seoul National University nor the names of its   */
/*      contributors may be used to endorse or promote products derived      */
/*      from this software without specific prior written permission.        */
/*                                                                           */
/* THIS SOFTWARE IS PROVIDED BY SEOUL NATIONAL UNIVERSITY "AS IS" AND ANY    */
/* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED */
/* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE    */
/* DISCLAIMED. IN NO EVENT SHALL SEOUL NATIONAL UNIVERSITY BE LIABLE FOR ANY */
/* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL        */
/* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS   */
/* OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     */
/* HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,       */
/* STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN  */
/* ANY WAY OUT OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE          */
/* POSSIBILITY OF SUCH DAMAGE.                                               */
/*                                                                           */
/* Contact information:                                                      */
/*   Center for Manycore Programming                                         */
/*   School of Computer Science and Engineering                              */
/*   Seoul National University, Seoul 151-744, Korea                         */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Sangmin Seo, Jungwon Kim, Gangwon Jo, Jun Lee, Jeongho Nah,             */
/*   Jungho Park, Junghyun Kim, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/* This file is based on the SNU-SAMSUNG OpenCL Compiler and is distributed  */
/* under GNU General Public License.                                         */
/* See LICENSE.SNU-SAMSUNG_OpenCL_C_Compiler.TXT for details.                */
/*****************************************************************************/

//===--------------------------------------------------------------------===//
// This file defines some constants for OpenCL compiler.
//===--------------------------------------------------------------------===//

#ifndef SNUCLC_DEFINES_H
#define SNUCLC_DEFINES_H

//////////////////////////////////////////////////////////////
// Header and footer for kernel source
//////////////////////////////////////////////////////////////
#define OPENCL_KERNEL_FP            "_p2W8Ht"

//////////////////////////////////////////////////////////////
// Quailifiers
//////////////////////////////////////////////////////////////
/* function qualfiers */
#define OPENCL_FQ_KERNEL            "__kernel"

/* address qualifiers */
#define OPENCL_N_GLOBAL             3
#define OPENCL_N_CONSTANT           2
#define OPENCL_N_LOCAL              1
#define OPENCL_N_PRIVATE            0

/* access qualifiers */
#define OPENCL_N_READONLY           3
#define OPENCL_N_WRITEONLY          2
#define OPENCL_N_READWRITE          1
#define OPENCL_N_NONE               0

/* type qualifiers */
#define OPENCL_TYPE_NONE            0
#define OPENCL_TYPE_CONST           (1 << 0)
#define OPENCL_TYPE_RESTRICT        (1 << 1)
#define OPENCL_TYPE_VOLATILE        (1 << 2)


//////////////////////////////////////////////////////////////
// Types
//////////////////////////////////////////////////////////////
/* type name */
#define OPENCL_TYPE_ULONG           "ullong"

/* name for anonymous tag type */
#define OPENCL_ANONYMOUS_NAME       "__anonymous_"

/* local variables */
#define OPENCL_LOCAL_VAR_PREFIX     "__cl_lv"

/* prefix for duplicated functions */
#define OPENCL_DUP_FUN_PREFIX       "cl_dup_"

/* prefix for vector literal VarDecls */
#define OPENCL_VEC_LIT_VAR_PREFIX   "__cl_vl"

/* prefix for temporary return VarDecls */
#define OPENCL_RET_VAR_PREFIX       "__cl_ret"

/* prefix for argument VarDecls */
#define OPENCL_ARG_VAR_PREFIX       "__cl_arg"

/* prefix for conditional VarDecls */
#define OPENCL_COND_VAR_PREFIX      "__cl_cond"

/* prefix for ParmVarDecls */
#define OPENCL_PVD_PREFIX           "__cl_parm_"

/* prefix for variable expansion */
#define OPENCL_VE_PREFIX            "__cl_ve_"

/* prefix for the next label */
#define OPENCL_NEXT_LABEL_PREFIX    "__cl_next_"


//////////////////////////////////////////////////////////////
// Builtin Functions
//////////////////////////////////////////////////////////////
/* convert function for type conversion */
#define OPENCL_CONVERT_HEADER       "convert_"
#define OPENCL_CONVERT_MIDDLE       "("
#define OPENCL_CONVERT_FOOTER       ")"

/* vector load and store */
#define OPENCL_VECTOR_DATA_GLOBAL   "_global"
#define OPENCL_VECTOR_DATA_LOCAL    "_local"

/* async_work_group_copy */
#define OPENCL_ASYNC_GLOBAL_TO_LOCAL  "_g2l"
#define OPENCL_ASYNC_LOCAL_TO_GLOBAL  "_l2g"


#endif //SNUCLC_DEFINES_H

