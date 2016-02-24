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

//===--- CLUtils.cpp - Utility class for OpenCL compiler ----------------===//
//
//===--------------------------------------------------------------------===//
//
// This file implements CLUtils class for OpenCL compiler.
//
//===--------------------------------------------------------------------===//

#include <string>
#include "Defines.h"
#include "CLUtils.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;
using std::string;


// NOTE: Keep string arrays in the sorted order!
static string clang_types[] = {
  "__builtin_va_list",
  "__int128_t",
  "__uint128_t",
  "__va_list_tag"
};

static string cl_types[] = {
  "cl_mem_fence_flags",
  "event_t",
  "image1d_array_t",
  "image1d_buffer_t",
  "image1d_t",
  "image2d_array_t",
  "image2d_t",
  "image3d_t",
  "intptr_t",
  "ptrdiff_t",
  "sampler_t",
  "size_t",
  "uchar",
  "uint",
  "uintptr_t",
  "ulong",
  "ushort",
  "wchar_t"
};

static string cl_image_types[] = {
  "image1d_array_t",
  "image1d_buffer_t",
  "image1d_t",
  "image2d_array_t",
  "image2d_t",
  "image3d_t"
};

static string cl_vectors[] = {
  "char16",
  "char2",
  "char3",
  "char4",
  "char8",
  "double16",
  "double2",
  "double3",
  "double4",
  "double8",
  "float16",
  "float2",
  "float3",
  "float4",
  "float8",
  "half16",
  "half2",
  "half3",
  "half4",
  "half8",
  "int16",
  "int2",
  "int3",
  "int4",
  "int8",
  "long16",
  "long2",
  "long3",
  "long4",
  "long8",
  "short16",
  "short2",
  "short3",
  "short4",
  "short8",
  "uchar16",
  "uchar2",
  "uchar3",
  "uchar4",
  "uchar8",
  "uint16",
  "uint2",
  "uint3",
  "uint4",
  "uint8",
  "ulong16",
  "ulong2",
  "ulong3",
  "ulong4",
  "ulong8",
  "ushort16",
  "ushort2",
  "ushort3",
  "ushort4",
  "ushort8"
};

static string cl_enums[] = {
  "_cl_mem_fence_flags",
  "_sampler_addressing_mode",
  "_sampler_filter_mode",
  "_sampler_normalized_coords"
};

static string cl_builtins[] = {
  "abs",
  "abs_diff",
  "acos",
  "acosh",
  "acospi",
  "add_sat",
  "all",
  "any",
  "as_char",
  "as_char16",
  "as_char2",
  "as_char3",
  "as_char4",
  "as_char8",
  "as_double",
  "as_double16",
  "as_double2",
  "as_double3",
  "as_double4",
  "as_double8",
  "as_float",
  "as_float16",
  "as_float2",
  "as_float3",
  "as_float4",
  "as_float8",
  "as_int",
  "as_int16",
  "as_int2",
  "as_int3",
  "as_int4",
  "as_int8",
  "as_long",
  "as_long16",
  "as_long2",
  "as_long3",
  "as_long4",
  "as_long8",
  "as_short",
  "as_short16",
  "as_short2",
  "as_short3",
  "as_short4",
  "as_short8",
  "as_uchar",
  "as_uchar16",
  "as_uchar2",
  "as_uchar3",
  "as_uchar4",
  "as_uchar8",
  "as_uint",
  "as_uint16",
  "as_uint2",
  "as_uint3",
  "as_uint4",
  "as_uint8",
  "as_ulong",
  "as_ulong16",
  "as_ulong2",
  "as_ulong3",
  "as_ulong4",
  "as_ulong8",
  "as_ushort",
  "as_ushort16",
  "as_ushort2",
  "as_ushort3",
  "as_ushort4",
  "as_ushort8",
  "asin",
  "asinh",
  "asinpi",
  "async_work_group_copy",
  "async_work_group_strided_copy",
  "atan",
  "atan2",
  "atan2pi",
  "atanh",
  "atanpi",
  "atom_add",
  "atom_and",
  "atom_cmpxchg",
  "atom_dec",
  "atom_inc",
  "atom_max",
  "atom_min",
  "atom_or",
  "atom_sub",
  "atom_xchg",
  "atom_xor",
  "atomic_add",
  "atomic_and",
  "atomic_cmpxchg",
  "atomic_dec",
  "atomic_inc",
  "atomic_max",
  "atomic_min",
  "atomic_or",
  "atomic_sub",
  "atomic_xchg",
  "atomic_xor",
  "barrier",
  "bitselect",
  "cbrt",
  "ceil",
  "clamp",
  "clz",
  "convert_bool",
  "convert_bool_rte",
  "convert_bool_rtn",
  "convert_bool_rtp",
  "convert_bool_rtz",
  "convert_bool_sat",
  "convert_bool_sat_rte",
  "convert_bool_sat_rtn",
  "convert_bool_sat_rtp",
  "convert_bool_sat_rtz",
  "convert_char",
  "convert_char16",
  "convert_char16_rte",
  "convert_char16_rtn",
  "convert_char16_rtp",
  "convert_char16_rtz",
  "convert_char16_sat",
  "convert_char16_sat_rte",
  "convert_char16_sat_rtn",
  "convert_char16_sat_rtp",
  "convert_char16_sat_rtz",
  "convert_char2",
  "convert_char2_rte",
  "convert_char2_rtn",
  "convert_char2_rtp",
  "convert_char2_rtz",
  "convert_char2_sat",
  "convert_char2_sat_rte",
  "convert_char2_sat_rtn",
  "convert_char2_sat_rtp",
  "convert_char2_sat_rtz",
  "convert_char3",
  "convert_char3_rte",
  "convert_char3_rtn",
  "convert_char3_rtp",
  "convert_char3_rtz",
  "convert_char3_sat",
  "convert_char3_sat_rte",
  "convert_char3_sat_rtn",
  "convert_char3_sat_rtp",
  "convert_char3_sat_rtz",
  "convert_char4",
  "convert_char4_rte",
  "convert_char4_rtn",
  "convert_char4_rtp",
  "convert_char4_rtz",
  "convert_char4_sat",
  "convert_char4_sat_rte",
  "convert_char4_sat_rtn",
  "convert_char4_sat_rtp",
  "convert_char4_sat_rtz",
  "convert_char8",
  "convert_char8_rte",
  "convert_char8_rtn",
  "convert_char8_rtp",
  "convert_char8_rtz",
  "convert_char8_sat",
  "convert_char8_sat_rte",
  "convert_char8_sat_rtn",
  "convert_char8_sat_rtp",
  "convert_char8_sat_rtz",
  "convert_char_rte",
  "convert_char_rtn",
  "convert_char_rtp",
  "convert_char_rtz",
  "convert_char_sat",
  "convert_char_sat_rte",
  "convert_char_sat_rtn",
  "convert_char_sat_rtp",
  "convert_char_sat_rtz",
  "convert_double",
  "convert_double16",
  "convert_double16_rte",
  "convert_double16_rtn",
  "convert_double16_rtp",
  "convert_double16_rtz",
  "convert_double16_sat",
  "convert_double16_sat_rte",
  "convert_double16_sat_rtn",
  "convert_double16_sat_rtp",
  "convert_double16_sat_rtz",
  "convert_double2",
  "convert_double2_rte",
  "convert_double2_rtn",
  "convert_double2_rtp",
  "convert_double2_rtz",
  "convert_double2_sat",
  "convert_double2_sat_rte",
  "convert_double2_sat_rtn",
  "convert_double2_sat_rtp",
  "convert_double2_sat_rtz",
  "convert_double3",
  "convert_double3_rte",
  "convert_double3_rtn",
  "convert_double3_rtp",
  "convert_double3_rtz",
  "convert_double3_sat",
  "convert_double3_sat_rte",
  "convert_double3_sat_rtn",
  "convert_double3_sat_rtp",
  "convert_double3_sat_rtz",
  "convert_double4",
  "convert_double4_rte",
  "convert_double4_rtn",
  "convert_double4_rtp",
  "convert_double4_rtz",
  "convert_double4_sat",
  "convert_double4_sat_rte",
  "convert_double4_sat_rtn",
  "convert_double4_sat_rtp",
  "convert_double4_sat_rtz",
  "convert_double8",
  "convert_double8_rte",
  "convert_double8_rtn",
  "convert_double8_rtp",
  "convert_double8_rtz",
  "convert_double8_sat",
  "convert_double8_sat_rte",
  "convert_double8_sat_rtn",
  "convert_double8_sat_rtp",
  "convert_double8_sat_rtz",
  "convert_double_rte",
  "convert_double_rtn",
  "convert_double_rtp",
  "convert_double_rtz",
  "convert_double_sat",
  "convert_double_sat_rte",
  "convert_double_sat_rtn",
  "convert_double_sat_rtp",
  "convert_double_sat_rtz",
  "convert_float",
  "convert_float16",
  "convert_float16_rte",
  "convert_float16_rtn",
  "convert_float16_rtp",
  "convert_float16_rtz",
  "convert_float16_sat",
  "convert_float16_sat_rte",
  "convert_float16_sat_rtn",
  "convert_float16_sat_rtp",
  "convert_float16_sat_rtz",
  "convert_float2",
  "convert_float2_rte",
  "convert_float2_rtn",
  "convert_float2_rtp",
  "convert_float2_rtz",
  "convert_float2_sat",
  "convert_float2_sat_rte",
  "convert_float2_sat_rtn",
  "convert_float2_sat_rtp",
  "convert_float2_sat_rtz",
  "convert_float3",
  "convert_float3_rte",
  "convert_float3_rtn",
  "convert_float3_rtp",
  "convert_float3_rtz",
  "convert_float3_sat",
  "convert_float3_sat_rte",
  "convert_float3_sat_rtn",
  "convert_float3_sat_rtp",
  "convert_float3_sat_rtz",
  "convert_float4",
  "convert_float4_rte",
  "convert_float4_rtn",
  "convert_float4_rtp",
  "convert_float4_rtz",
  "convert_float4_sat",
  "convert_float4_sat_rte",
  "convert_float4_sat_rtn",
  "convert_float4_sat_rtp",
  "convert_float4_sat_rtz",
  "convert_float8",
  "convert_float8_rte",
  "convert_float8_rtn",
  "convert_float8_rtp",
  "convert_float8_rtz",
  "convert_float8_sat",
  "convert_float8_sat_rte",
  "convert_float8_sat_rtn",
  "convert_float8_sat_rtp",
  "convert_float8_sat_rtz",
  "convert_float_rte",
  "convert_float_rtn",
  "convert_float_rtp",
  "convert_float_rtz",
  "convert_float_sat",
  "convert_float_sat_rte",
  "convert_float_sat_rtn",
  "convert_float_sat_rtp",
  "convert_float_sat_rtz",
  "convert_int",
  "convert_int16",
  "convert_int16_rte",
  "convert_int16_rtn",
  "convert_int16_rtp",
  "convert_int16_rtz",
  "convert_int16_sat",
  "convert_int16_sat_rte",
  "convert_int16_sat_rtn",
  "convert_int16_sat_rtp",
  "convert_int16_sat_rtz",
  "convert_int2",
  "convert_int2_rte",
  "convert_int2_rtn",
  "convert_int2_rtp",
  "convert_int2_rtz",
  "convert_int2_sat",
  "convert_int2_sat_rte",
  "convert_int2_sat_rtn",
  "convert_int2_sat_rtp",
  "convert_int2_sat_rtz",
  "convert_int3",
  "convert_int3_rte",
  "convert_int3_rtn",
  "convert_int3_rtp",
  "convert_int3_rtz",
  "convert_int3_sat",
  "convert_int3_sat_rte",
  "convert_int3_sat_rtn",
  "convert_int3_sat_rtp",
  "convert_int3_sat_rtz",
  "convert_int4",
  "convert_int4_rte",
  "convert_int4_rtn",
  "convert_int4_rtp",
  "convert_int4_rtz",
  "convert_int4_sat",
  "convert_int4_sat_rte",
  "convert_int4_sat_rtn",
  "convert_int4_sat_rtp",
  "convert_int4_sat_rtz",
  "convert_int8",
  "convert_int8_rte",
  "convert_int8_rtn",
  "convert_int8_rtp",
  "convert_int8_rtz",
  "convert_int8_sat",
  "convert_int8_sat_rte",
  "convert_int8_sat_rtn",
  "convert_int8_sat_rtp",
  "convert_int8_sat_rtz",
  "convert_int_rte",
  "convert_int_rtn",
  "convert_int_rtp",
  "convert_int_rtz",
  "convert_int_sat",
  "convert_int_sat_rte",
  "convert_int_sat_rtn",
  "convert_int_sat_rtp",
  "convert_int_sat_rtz",
  "convert_long",
  "convert_long16",
  "convert_long16_rte",
  "convert_long16_rtn",
  "convert_long16_rtp",
  "convert_long16_rtz",
  "convert_long16_sat",
  "convert_long16_sat_rte",
  "convert_long16_sat_rtn",
  "convert_long16_sat_rtp",
  "convert_long16_sat_rtz",
  "convert_long2",
  "convert_long2_rte",
  "convert_long2_rtn",
  "convert_long2_rtp",
  "convert_long2_rtz",
  "convert_long2_sat",
  "convert_long2_sat_rte",
  "convert_long2_sat_rtn",
  "convert_long2_sat_rtp",
  "convert_long2_sat_rtz",
  "convert_long3",
  "convert_long3_rte",
  "convert_long3_rtn",
  "convert_long3_rtp",
  "convert_long3_rtz",
  "convert_long3_sat",
  "convert_long3_sat_rte",
  "convert_long3_sat_rtn",
  "convert_long3_sat_rtp",
  "convert_long3_sat_rtz",
  "convert_long4",
  "convert_long4_rte",
  "convert_long4_rtn",
  "convert_long4_rtp",
  "convert_long4_rtz",
  "convert_long4_sat",
  "convert_long4_sat_rte",
  "convert_long4_sat_rtn",
  "convert_long4_sat_rtp",
  "convert_long4_sat_rtz",
  "convert_long8",
  "convert_long8_rte",
  "convert_long8_rtn",
  "convert_long8_rtp",
  "convert_long8_rtz",
  "convert_long8_sat",
  "convert_long8_sat_rte",
  "convert_long8_sat_rtn",
  "convert_long8_sat_rtp",
  "convert_long8_sat_rtz",
  "convert_long_rte",
  "convert_long_rtn",
  "convert_long_rtp",
  "convert_long_rtz",
  "convert_long_sat",
  "convert_long_sat_rte",
  "convert_long_sat_rtn",
  "convert_long_sat_rtp",
  "convert_long_sat_rtz",
  "convert_short",
  "convert_short16",
  "convert_short16_rte",
  "convert_short16_rtn",
  "convert_short16_rtp",
  "convert_short16_rtz",
  "convert_short16_sat",
  "convert_short16_sat_rte",
  "convert_short16_sat_rtn",
  "convert_short16_sat_rtp",
  "convert_short16_sat_rtz",
  "convert_short2",
  "convert_short2_rte",
  "convert_short2_rtn",
  "convert_short2_rtp",
  "convert_short2_rtz",
  "convert_short2_sat",
  "convert_short2_sat_rte",
  "convert_short2_sat_rtn",
  "convert_short2_sat_rtp",
  "convert_short2_sat_rtz",
  "convert_short3",
  "convert_short3_rte",
  "convert_short3_rtn",
  "convert_short3_rtp",
  "convert_short3_rtz",
  "convert_short3_sat",
  "convert_short3_sat_rte",
  "convert_short3_sat_rtn",
  "convert_short3_sat_rtp",
  "convert_short3_sat_rtz",
  "convert_short4",
  "convert_short4_rte",
  "convert_short4_rtn",
  "convert_short4_rtp",
  "convert_short4_rtz",
  "convert_short4_sat",
  "convert_short4_sat_rte",
  "convert_short4_sat_rtn",
  "convert_short4_sat_rtp",
  "convert_short4_sat_rtz",
  "convert_short8",
  "convert_short8_rte",
  "convert_short8_rtn",
  "convert_short8_rtp",
  "convert_short8_rtz",
  "convert_short8_sat",
  "convert_short8_sat_rte",
  "convert_short8_sat_rtn",
  "convert_short8_sat_rtp",
  "convert_short8_sat_rtz",
  "convert_short_rte",
  "convert_short_rtn",
  "convert_short_rtp",
  "convert_short_rtz",
  "convert_short_sat",
  "convert_short_sat_rte",
  "convert_short_sat_rtn",
  "convert_short_sat_rtp",
  "convert_short_sat_rtz",
  "convert_uchar",
  "convert_uchar16",
  "convert_uchar16_rte",
  "convert_uchar16_rtn",
  "convert_uchar16_rtp",
  "convert_uchar16_rtz",
  "convert_uchar16_sat",
  "convert_uchar16_sat_rte",
  "convert_uchar16_sat_rtn",
  "convert_uchar16_sat_rtp",
  "convert_uchar16_sat_rtz",
  "convert_uchar2",
  "convert_uchar2_rte",
  "convert_uchar2_rtn",
  "convert_uchar2_rtp",
  "convert_uchar2_rtz",
  "convert_uchar2_sat",
  "convert_uchar2_sat_rte",
  "convert_uchar2_sat_rtn",
  "convert_uchar2_sat_rtp",
  "convert_uchar2_sat_rtz",
  "convert_uchar3",
  "convert_uchar3_rte",
  "convert_uchar3_rtn",
  "convert_uchar3_rtp",
  "convert_uchar3_rtz",
  "convert_uchar3_sat",
  "convert_uchar3_sat_rte",
  "convert_uchar3_sat_rtn",
  "convert_uchar3_sat_rtp",
  "convert_uchar3_sat_rtz",
  "convert_uchar4",
  "convert_uchar4_rte",
  "convert_uchar4_rtn",
  "convert_uchar4_rtp",
  "convert_uchar4_rtz",
  "convert_uchar4_sat",
  "convert_uchar4_sat_rte",
  "convert_uchar4_sat_rtn",
  "convert_uchar4_sat_rtp",
  "convert_uchar4_sat_rtz",
  "convert_uchar8",
  "convert_uchar8_rte",
  "convert_uchar8_rtn",
  "convert_uchar8_rtp",
  "convert_uchar8_rtz",
  "convert_uchar8_sat",
  "convert_uchar8_sat_rte",
  "convert_uchar8_sat_rtn",
  "convert_uchar8_sat_rtp",
  "convert_uchar8_sat_rtz",
  "convert_uchar_rte",
  "convert_uchar_rtn",
  "convert_uchar_rtp",
  "convert_uchar_rtz",
  "convert_uchar_sat",
  "convert_uchar_sat_rte",
  "convert_uchar_sat_rtn",
  "convert_uchar_sat_rtp",
  "convert_uchar_sat_rtz",
  "convert_uint",
  "convert_uint16",
  "convert_uint16_rte",
  "convert_uint16_rtn",
  "convert_uint16_rtp",
  "convert_uint16_rtz",
  "convert_uint16_sat",
  "convert_uint16_sat_rte",
  "convert_uint16_sat_rtn",
  "convert_uint16_sat_rtp",
  "convert_uint16_sat_rtz",
  "convert_uint2",
  "convert_uint2_rte",
  "convert_uint2_rtn",
  "convert_uint2_rtp",
  "convert_uint2_rtz",
  "convert_uint2_sat",
  "convert_uint2_sat_rte",
  "convert_uint2_sat_rtn",
  "convert_uint2_sat_rtp",
  "convert_uint2_sat_rtz",
  "convert_uint3",
  "convert_uint3_rte",
  "convert_uint3_rtn",
  "convert_uint3_rtp",
  "convert_uint3_rtz",
  "convert_uint3_sat",
  "convert_uint3_sat_rte",
  "convert_uint3_sat_rtn",
  "convert_uint3_sat_rtp",
  "convert_uint3_sat_rtz",
  "convert_uint4",
  "convert_uint4_rte",
  "convert_uint4_rtn",
  "convert_uint4_rtp",
  "convert_uint4_rtz",
  "convert_uint4_sat",
  "convert_uint4_sat_rte",
  "convert_uint4_sat_rtn",
  "convert_uint4_sat_rtp",
  "convert_uint4_sat_rtz",
  "convert_uint8",
  "convert_uint8_rte",
  "convert_uint8_rtn",
  "convert_uint8_rtp",
  "convert_uint8_rtz",
  "convert_uint8_sat",
  "convert_uint8_sat_rte",
  "convert_uint8_sat_rtn",
  "convert_uint8_sat_rtp",
  "convert_uint8_sat_rtz",
  "convert_uint_rte",
  "convert_uint_rtn",
  "convert_uint_rtp",
  "convert_uint_rtz",
  "convert_uint_sat",
  "convert_uint_sat_rte",
  "convert_uint_sat_rtn",
  "convert_uint_sat_rtp",
  "convert_uint_sat_rtz",
  "convert_ulong",
  "convert_ulong16",
  "convert_ulong16_rte",
  "convert_ulong16_rtn",
  "convert_ulong16_rtp",
  "convert_ulong16_rtz",
  "convert_ulong16_sat",
  "convert_ulong16_sat_rte",
  "convert_ulong16_sat_rtn",
  "convert_ulong16_sat_rtp",
  "convert_ulong16_sat_rtz",
  "convert_ulong2",
  "convert_ulong2_rte",
  "convert_ulong2_rtn",
  "convert_ulong2_rtp",
  "convert_ulong2_rtz",
  "convert_ulong2_sat",
  "convert_ulong2_sat_rte",
  "convert_ulong2_sat_rtn",
  "convert_ulong2_sat_rtp",
  "convert_ulong2_sat_rtz",
  "convert_ulong3",
  "convert_ulong3_rte",
  "convert_ulong3_rtn",
  "convert_ulong3_rtp",
  "convert_ulong3_rtz",
  "convert_ulong3_sat",
  "convert_ulong3_sat_rte",
  "convert_ulong3_sat_rtn",
  "convert_ulong3_sat_rtp",
  "convert_ulong3_sat_rtz",
  "convert_ulong4",
  "convert_ulong4_rte",
  "convert_ulong4_rtn",
  "convert_ulong4_rtp",
  "convert_ulong4_rtz",
  "convert_ulong4_sat",
  "convert_ulong4_sat_rte",
  "convert_ulong4_sat_rtn",
  "convert_ulong4_sat_rtp",
  "convert_ulong4_sat_rtz",
  "convert_ulong8",
  "convert_ulong8_rte",
  "convert_ulong8_rtn",
  "convert_ulong8_rtp",
  "convert_ulong8_rtz",
  "convert_ulong8_sat",
  "convert_ulong8_sat_rte",
  "convert_ulong8_sat_rtn",
  "convert_ulong8_sat_rtp",
  "convert_ulong8_sat_rtz",
  "convert_ulong_rte",
  "convert_ulong_rtn",
  "convert_ulong_rtp",
  "convert_ulong_rtz",
  "convert_ulong_sat",
  "convert_ulong_sat_rte",
  "convert_ulong_sat_rtn",
  "convert_ulong_sat_rtp",
  "convert_ulong_sat_rtz",
  "convert_ushort",
  "convert_ushort16",
  "convert_ushort16_rte",
  "convert_ushort16_rtn",
  "convert_ushort16_rtp",
  "convert_ushort16_rtz",
  "convert_ushort16_sat",
  "convert_ushort16_sat_rte",
  "convert_ushort16_sat_rtn",
  "convert_ushort16_sat_rtp",
  "convert_ushort16_sat_rtz",
  "convert_ushort2",
  "convert_ushort2_rte",
  "convert_ushort2_rtn",
  "convert_ushort2_rtp",
  "convert_ushort2_rtz",
  "convert_ushort2_sat",
  "convert_ushort2_sat_rte",
  "convert_ushort2_sat_rtn",
  "convert_ushort2_sat_rtp",
  "convert_ushort2_sat_rtz",
  "convert_ushort3",
  "convert_ushort3_rte",
  "convert_ushort3_rtn",
  "convert_ushort3_rtp",
  "convert_ushort3_rtz",
  "convert_ushort3_sat",
  "convert_ushort3_sat_rte",
  "convert_ushort3_sat_rtn",
  "convert_ushort3_sat_rtp",
  "convert_ushort3_sat_rtz",
  "convert_ushort4",
  "convert_ushort4_rte",
  "convert_ushort4_rtn",
  "convert_ushort4_rtp",
  "convert_ushort4_rtz",
  "convert_ushort4_sat",
  "convert_ushort4_sat_rte",
  "convert_ushort4_sat_rtn",
  "convert_ushort4_sat_rtp",
  "convert_ushort4_sat_rtz",
  "convert_ushort8",
  "convert_ushort8_rte",
  "convert_ushort8_rtn",
  "convert_ushort8_rtp",
  "convert_ushort8_rtz",
  "convert_ushort8_sat",
  "convert_ushort8_sat_rte",
  "convert_ushort8_sat_rtn",
  "convert_ushort8_sat_rtp",
  "convert_ushort8_sat_rtz",
  "convert_ushort_rte",
  "convert_ushort_rtn",
  "convert_ushort_rtp",
  "convert_ushort_rtz",
  "convert_ushort_sat",
  "convert_ushort_sat_rte",
  "convert_ushort_sat_rtn",
  "convert_ushort_sat_rtp",
  "convert_ushort_sat_rtz",
  "copysign",
  "cos",
  "cosh",
  "cospi",
  "cross",
  "degrees",
  "distance",
  "dot",
  "erf",
  "erfc",
  "exp",
  "exp10",
  "exp2",
  "expm1",
  "fabs",
  "fast_distance",
  "fast_length",
  "fast_normalize",
  "fdim",
  "floor",
  "fma",
  "fmax",
  "fmin",
  "fmod",
  "fract",
  "frexp",
  "get_global_id",
  "get_global_offset",
  "get_global_size",
  "get_group_id",
  "get_image_array_size",
  "get_image_channel_data_type",
  "get_image_channel_order",
  "get_image_depth",
  "get_image_dim",
  "get_image_height",
  "get_image_width",
  "get_local_id",
  "get_local_size",
  "get_num_groups",
  "get_work_dim",
  "hadd",
  "half_cos",
  "half_divide",
  "half_exp",
  "half_exp10",
  "half_exp2",
  "half_log",
  "half_log10",
  "half_log2",
  "half_powr",
  "half_recip",
  "half_rsqrt",
  "half_sin",
  "half_sqrt",
  "half_tan",
  "hypot",
  "ilogb",
  "isequal",
  "isfinite",
  "isgreater",
  "isgreaterequal",
  "isinf",
  "isless",
  "islessequal",
  "islessgreater",
  "isnan",
  "isnormal",
  "isnotequal",
  "isordered",
  "isunordered",
  "ldexp",
  "length",
  "lgamma",
  "lgamma_r",
  "log",
  "log10",
  "log1p",
  "log2",
  "logb",
  "mad",
  "mad24",
  "mad_hi",
  "mad_sat",
  "max",
  "maxmag",
  "mem_fence",
  "min",
  "minmag",
  "mix",
  "modf",
  "mul24",
  "mul_hi",
  "nan",
  "native_cos",
  "native_divide",
  "native_exp",
  "native_exp10",
  "native_exp2",
  "native_log",
  "native_log10",
  "native_log2",
  "native_powr",
  "native_recip",
  "native_rsqrt",
  "native_sin",
  "native_sqrt",
  "native_tan",
  "nextafter",
  "normalize",
  "popcount",
  "pow",
  "pown",
  "powr",
  "prefetch",
  "printf",
  "radians",
  "read_imagef",
  "read_imagei",
  "read_imageui",
  "read_mem_fence",
  "remainder",
  "remquo",
  "rhadd",
  "rint",
  "rootn",
  "rotate",
  "round",
  "rsqrt",
  "select",
  "shuffle",
  "shuffle2",
  "sign",
  "signbit",
  "sin",
  "sincos",
  "sinh",
  "sinpi",
  "smoothstep",
  "sqrt",
  "step",
  "sub_sat",
  "tan",
  "tanh",
  "tanpi",
  "tgamma",
  "trunc",
  "upsample",
  "vec_step",
  "vload16",
  "vload2",
  "vload3",
  "vload4",
  "vload8",
  "vload_half",
  "vload_half16",
  "vload_half2",
  "vload_half3",
  "vload_half4",
  "vload_half8",
  "vloada_half",
  "vloada_half16",
  "vloada_half2",
  "vloada_half3",
  "vloada_half4",
  "vloada_half8",
  "vstore16",
  "vstore2",
  "vstore3",
  "vstore4",
  "vstore8",
  "vstore_half",
  "vstore_half16",
  "vstore_half16_rte",
  "vstore_half16_rtn",
  "vstore_half16_rtp",
  "vstore_half16_rtz",
  "vstore_half2",
  "vstore_half2_rte",
  "vstore_half2_rtn",
  "vstore_half2_rtp",
  "vstore_half2_rtz",
  "vstore_half3",
  "vstore_half3_rte",
  "vstore_half3_rtn",
  "vstore_half3_rtp",
  "vstore_half3_rtz",
  "vstore_half4",
  "vstore_half4_rte",
  "vstore_half4_rtn",
  "vstore_half4_rtp",
  "vstore_half4_rtz",
  "vstore_half8",
  "vstore_half8_rte",
  "vstore_half8_rtn",
  "vstore_half8_rtp",
  "vstore_half8_rtz",
  "vstore_half_rte",
  "vstore_half_rtn",
  "vstore_half_rtp",
  "vstore_half_rtz",
  "vstorea_half",
  "vstorea_half16",
  "vstorea_half16_rte",
  "vstorea_half16_rtn",
  "vstorea_half16_rtp",
  "vstorea_half16_rtz",
  "vstorea_half2",
  "vstorea_half2_rte",
  "vstorea_half2_rtn",
  "vstorea_half2_rtp",
  "vstorea_half2_rtz",
  "vstorea_half3",
  "vstorea_half3_rte",
  "vstorea_half3_rtn",
  "vstorea_half3_rtp",
  "vstorea_half3_rtz",
  "vstorea_half4",
  "vstorea_half4_rte",
  "vstorea_half4_rtn",
  "vstorea_half4_rtp",
  "vstorea_half4_rtz",
  "vstorea_half8",
  "vstorea_half8_rte",
  "vstorea_half8_rtn",
  "vstorea_half8_rtp",
  "vstorea_half8_rtz",
  "vstorea_half_rte",
  "vstorea_half_rtn",
  "vstorea_half_rtp",
  "vstorea_half_rtz",
  "wait_group_events",
  "write_imagef",
  "write_imagei",
  "write_imageui",
  "write_mem_fence"
};

static string cl_vloads[] = {
  "vload16",
  "vload2",
  "vload3",
  "vload4",
  "vload8",
  "vload_half",
  "vload_half16",
  "vload_half2",
  "vload_half3",
  "vload_half4",
  "vload_half8",
  "vloada_half",
  "vloada_half16",
  "vloada_half2",
  "vloada_half3",
  "vloada_half4",
  "vloada_half8"
};

static string cl_vstores[] = {
  "vstore16",
  "vstore2",
  "vstore3",
  "vstore4",
  "vstore8",
  "vstore_half",
  "vstore_half16",
  "vstore_half16_rte",
  "vstore_half16_rtn",
  "vstore_half16_rtp",
  "vstore_half16_rtz",
  "vstore_half2",
  "vstore_half2_rte",
  "vstore_half2_rtn",
  "vstore_half2_rtp",
  "vstore_half2_rtz",
  "vstore_half3",
  "vstore_half3_rte",
  "vstore_half3_rtn",
  "vstore_half3_rtp",
  "vstore_half3_rtz",
  "vstore_half4",
  "vstore_half4_rte",
  "vstore_half4_rtn",
  "vstore_half4_rtp",
  "vstore_half4_rtz",
  "vstore_half8",
  "vstore_half8_rte",
  "vstore_half8_rtn",
  "vstore_half8_rtp",
  "vstore_half8_rtz",
  "vstore_half_rte",
  "vstore_half_rtn",
  "vstore_half_rtp",
  "vstore_half_rtz",
  "vstorea_half",
  "vstorea_half16",
  "vstorea_half16_rte",
  "vstorea_half16_rtn",
  "vstorea_half16_rtp",
  "vstorea_half16_rtz",
  "vstorea_half2",
  "vstorea_half2_rte",
  "vstorea_half2_rtn",
  "vstorea_half2_rtp",
  "vstorea_half2_rtz",
  "vstorea_half3",
  "vstorea_half3_rte",
  "vstorea_half3_rtn",
  "vstorea_half3_rtp",
  "vstorea_half3_rtz",
  "vstorea_half4",
  "vstorea_half4_rte",
  "vstorea_half4_rtn",
  "vstorea_half4_rtp",
  "vstorea_half4_rtz",
  "vstorea_half8",
  "vstorea_half8_rte",
  "vstorea_half8_rtn",
  "vstorea_half8_rtp",
  "vstorea_half8_rtz",
  "vstorea_half_rte",
  "vstorea_half_rtn",
  "vstorea_half_rtp",
  "vstorea_half_rtz"
};

static string cl_async_copies[] = {
  "async_work_group_copy",
  "async_work_group_strided_copy"
};

static string cl_atomics[] = {
  "atom_add",
  "atom_and",
  "atom_cmpxchg",
  "atom_dec",
  "atom_inc",
  "atom_max",
  "atom_min",
  "atom_or",
  "atom_sub",
  "atom_xchg",
  "atom_xor",
  "atomic_add",
  "atomic_and",
  "atomic_cmpxchg",
  "atomic_dec",
  "atomic_inc",
  "atomic_max",
  "atomic_min",
  "atomic_or",
  "atomic_sub",
  "atomic_xchg",
  "atomic_xor"
};


static int SearchString(string &value, string arr[], int size) {
  int pos;
  int begin = 0;
  int end = size - 1;
  int cond = 0;

  while (begin <= end) {
    pos = (begin + end) / 2;
    cond = value.compare(arr[pos]);
    if (cond == 0)
      return pos;
    else if (cond < 0)
      end = pos - 1;
    else
      begin = pos + 1;
  }

  return -1;  // not found
}


//-------------------------------------------------------------------------//
// OpenCL identifiers, such as type name, built-in function names
//-------------------------------------------------------------------------//
bool CLUtils::IsClangTypeName(std::string &S) {
  const int CLANG_NUM_TYPES = sizeof(clang_types) / sizeof(string);
  if (SearchString(S, clang_types, CLANG_NUM_TYPES) != -1)
    return true;
  else
    return false;
}

bool CLUtils::IsCLTypeName(string &S) {
  const int NUM_TYPES = sizeof(cl_types) / sizeof(string);
  if (SearchString(S, cl_types, NUM_TYPES) != -1 ||
      IsCLVectorName(S))
    return true;
  else
    return false;
}

bool CLUtils::IsCLImageType(QualType T) {
  string TypeName = T.getAsString();
  return IsCLImageTypeName(TypeName);
}

bool CLUtils::IsCLImageTypeName(string &S) {
  const int NUM_IMAGE_TYPES = sizeof(cl_image_types) / sizeof(string);
  if (SearchString(S, cl_image_types, NUM_IMAGE_TYPES) != -1)
    return true;
  else
    return false;
}

bool CLUtils::IsCLVectorName(string &S) {
  const int NUM_VECTORS = sizeof(cl_vectors) / sizeof(string); 
  if (SearchString(S, cl_vectors, NUM_VECTORS) != -1)
    return true;
  else
    return false;
}

bool CLUtils::IsCLEnumName(string &S) {
  const int NUM_ENUMS = sizeof(cl_enums) / sizeof(string);
  if (SearchString(S, cl_enums, NUM_ENUMS) != -1)
    return true;
  else
    return false;
}

bool CLUtils::IsCLBuiltinName(string &S) {
  const int NUM_BUILTINS = sizeof(cl_builtins) / sizeof(string);
  if (SearchString(S, cl_builtins, NUM_BUILTINS) != -1)
    return true;
  else
    return false;
}

bool CLUtils::IsPredefinedName(std::string &S) {
  if (IsCLBuiltinName(S)) return true;
  if (IsCLTypeName(S)) return true;
  if (IsCLVectorName(S)) return true;
  if (IsCLEnumName(S)) return true;

  return false;
}

bool CLUtils::IsWorkItemFunction(StringRef S) {
  return S.equals("get_global_id")   || S.equals("get_local_id") ||
         S.equals("get_global_size") || S.equals("get_local_size") ||
         S.equals("get_num_groups")  || S.equals("get_group_id") ||
         S.equals("get_global_offset");
}
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Kernel function, address qualifiers, access qualifiers
//-------------------------------------------------------------------------//
bool CLUtils::IsKernel(Decl *D) {
  return D->hasAttr<OpenCLKernelAttr>();
}

bool CLUtils::IsGlobal(Decl *D) {
  if (D->getAddrQualifier() == AQ_Global) {
    return true;
  } else if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    QualType VDTy = VD->getType();
    return IsCLImageType(VDTy);
  }
  return false;
}

bool CLUtils::IsConstant(Decl *D) {
  return D->getAddrQualifier() == AQ_Constant;
}

bool CLUtils::IsLocal(Decl *D) {
  return D->getAddrQualifier() == AQ_Local;
}

bool CLUtils::IsPrivate(Decl *D) {
  return D->getAddrQualifier() == AQ_Private;
}

unsigned CLUtils::GetAddrQualifier(Decl *D) {
  if (IsGlobal(D)) return OPENCL_N_GLOBAL;

  switch (D->getAddrQualifier()) {
    case AQ_Global:   return OPENCL_N_GLOBAL;
    case AQ_Constant: return OPENCL_N_CONSTANT;
    case AQ_Local:    return OPENCL_N_LOCAL;
    case AQ_Private:  return OPENCL_N_PRIVATE;
  }
  return OPENCL_N_PRIVATE;
}


bool CLUtils::IsReadOnly(Decl *D) {
  return D->getAccessQualifier() == ACQ_ReadOnly;
}

bool CLUtils::IsWriteOnly(Decl *D) {
  return D->getAccessQualifier() == ACQ_WriteOnly;
}

bool CLUtils::IsReadWrite(Decl *D) {
  return D->getAccessQualifier() == ACQ_ReadWrite;
}

unsigned CLUtils::GetAccessQualifier(Decl *D) {
  switch (D->getAccessQualifier()) {
    case ACQ_ReadOnly:  return OPENCL_N_READONLY;
    case ACQ_WriteOnly: return OPENCL_N_WRITEONLY;
    case ACQ_ReadWrite: return OPENCL_N_READWRITE;
    case ACQ_None:      return OPENCL_N_NONE;
  }
  return OPENCL_N_NONE;
}

unsigned CLUtils::GetTypeQualifier(ValueDecl *D) {
  unsigned TQ = OPENCL_TYPE_NONE;
  QualType Ty = D->getType();
  if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
    Ty = Parm->getOriginalType();
  }

  // Obtain all internal CVR qualifiers
  unsigned CVRQuals = Ty.getCVRQualifiers();
  if (Ty->isPointerType()) {
    CVRQuals &= ~Qualifiers::Volatile;

    do {
      const PointerType *PTy = Ty->getAs<PointerType>();
      Ty = PTy->getPointeeType();
      CVRQuals |= Ty.getCVRQualifiers();
    } while (Ty->isPointerType());
  }

  if (CVRQuals & Qualifiers::Restrict)
    TQ |= OPENCL_TYPE_RESTRICT;
  if ((CVRQuals & Qualifiers::Const) || IsConstant(D))
    TQ |= OPENCL_TYPE_CONST;
  if (CVRQuals & Qualifiers::Volatile)
    TQ |= OPENCL_TYPE_VOLATILE;
  return TQ;
}

bool CLUtils::HasAttributeAnnotate(Decl *D, StringRef *Str) {
  if (const AnnotateAttr *AA = D->getAttr<AnnotateAttr>()) {
    return Str->equals(AA->getAnnotation());
  }
  return false;
}
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Type functions
//-------------------------------------------------------------------------//
void CLUtils::GetTypeStrWithoutCVR(QualType Ty, string &S, 
                                   PrintingPolicy &P) {
  Ty.removeLocalFastQualifiers();

  if (const PointerType *PT = dyn_cast<PointerType>(Ty.getTypePtr())) {
    if (!S.empty() && S[0] != '*')
      S = ' ' + S;
    S = '*' + S;
    QualType PointeeType = PT->getPointeeType();
    if (isa<ArrayType>(PointeeType))
      S = '(' + S + ')';
    PointeeType.removeLocalFastQualifiers();
    PointeeType.getAsStringInternal(S, P);
  } else {
    Ty.getAsStringInternal(S, P);
  }
}

unsigned CLUtils::GetTypeSize(QualType Ty, unsigned PointerSize) {
  if (Ty->isPointerType())
    return (PointerSize / 8);
  
  if (const BuiltinType *BT = Ty->getAs<BuiltinType>()) {
    switch (BT->getKind()) {
      case BuiltinType::Char_S:
      case BuiltinType::Char_U:
      case BuiltinType::SChar:
      case BuiltinType::UChar:       return 1;
      case BuiltinType::Short:
      case BuiltinType::UShort:      return 2;
      case BuiltinType::Bool:
      case BuiltinType::Int:
      case BuiltinType::UInt:        return 4;
      case BuiltinType::Long:
      case BuiltinType::ULong:       return 8;
      case BuiltinType::Half:        return 2;
      case BuiltinType::Float:       return 4;
      case BuiltinType::Double:      return 8;
      default: return 0;
    }
  }

  if (const ExtVectorType *EVT = Ty->getAs<ExtVectorType>()) {
    return GetTypeSize(EVT->getElementType(), PointerSize)
           * EVT->getNumElements();
  }
 
  if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(Ty)) {
    return GetTypeSize(CAT->getElementType(), PointerSize)
           * CAT->getSize().getZExtValue();
  }

  if (Ty->isEnumeralType())
    return 4;

  if (const TypedefType *TT = Ty->getAs<TypedefType>()) {
    QualType BaseTy = TT->desugar();
    return GetTypeSize(BaseTy, PointerSize);
  }

  if (const RecordType *RT = Ty->getAs<RecordType>()) {
    unsigned TySize = 0;
    RecordDecl *RD = RT->getDecl()->getDefinition();
    if (RD->isStruct()) {
      for (RecordDecl::field_iterator I = RD->field_begin(),
           E = RD->field_end(); I != E; ++I) {
        ValueDecl *Field = dyn_cast<ValueDecl>(*I);
        if (!Field) continue;

        QualType FieldTy = Field->getType();
        TySize += GetTypeSize(FieldTy, PointerSize);
      }
    } else if (RD->isUnion()) {
      for (RecordDecl::field_iterator I = RD->field_begin(),
           E = RD->field_end(); I != E; ++I) {
        ValueDecl *Field = dyn_cast<ValueDecl>(*I);
        if (!Field) continue;

        QualType FieldTy = Field->getType();
        unsigned FieldTySize = GetTypeSize(FieldTy, PointerSize);
        if (FieldTySize > TySize) TySize = FieldTySize;
      }
    } else {
      assert(0 && "Which RecordDecl?");
    }
    return TySize;
  }

  assert(0 && "What kind of type?");
  return 0;
}

//-------------------------------------------------------------------------//
// Vector functions
//-------------------------------------------------------------------------//
unsigned CLUtils::CeilVecNumElements(QualType Ty) {
  if (Ty->isExtVectorType()) {
    const ExtVectorType *EVT = Ty->getAs<ExtVectorType>();
    unsigned numElems = EVT->getNumElements();
    if (numElems <= 2)       return 2;
    else if (numElems <= 4)  return numElems;
    else if (numElems <= 8)  return 8;
    else if (numElems <= 16) return 16;
  }
  return 1;
}

unsigned CLUtils::GetVecAccessorNumber(Expr *E) {
  ExtVectorElementExpr *Node = dyn_cast<ExtVectorElementExpr>(E);
  assert(Node && "E is not ExtVectorElementExpr");

  // get accessor indices
  llvm::SmallVector<unsigned, 4> Indices;
  Node->getEncodedElementAccess(Indices);

  return Indices[0];
}
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Un-overloadable functions
//-------------------------------------------------------------------------//
bool CLUtils::IsBuiltinFunction(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    string FuncName = DRE->getDecl()->getNameAsString();
    return IsCLBuiltinName(FuncName);
  }
  else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
    return IsBuiltinFunction(ICE->getSubExpr());

  return false;
}

static bool IsVectorDataLoadFunction(string &S) {
  const int NUM_VLOADS = sizeof(cl_vloads) / sizeof(string);
  if (SearchString(S, cl_vloads, NUM_VLOADS) != -1)
    return true;
  else
    return false;
}
static bool IsVectorDataStoreFunction(string &S) {
  const int NUM_VSTORES = sizeof(cl_vstores) / sizeof(string);
  if (SearchString(S, cl_vstores, NUM_VSTORES) != -1)
    return true;
  else
    return false;
}

// Check if Expr is vector data load or store function.
// return value
// -1: neither vector data load nor store function
//  1: vector load
//  2: vector store
int CLUtils::IsVectorDataFunction(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    string FuncName = DRE->getDecl()->getNameAsString();
    if (IsVectorDataLoadFunction(FuncName))
      return 1;
    else if (IsVectorDataStoreFunction(FuncName))
      return 2;
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    return IsVectorDataFunction(ICE->getSubExpr());
  }

  return -1;
}

// Check if Expr is an async copy function.
bool CLUtils::IsAsyncCopyFunction(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    string FuncName = DRE->getDecl()->getNameAsString();
    const int NUM_ASYNC_COPIES = sizeof(cl_async_copies) / sizeof(string);
    if (SearchString(FuncName, cl_async_copies, NUM_ASYNC_COPIES) != -1)
      return true;
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    return IsAsyncCopyFunction(ICE->getSubExpr());
  }

  return false;
}

// Check if Expr is a barrier function.
bool CLUtils::IsBarrierFunction(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    string FuncName = DRE->getDecl()->getNameAsString();
    if (FuncName.compare("barrier") == 0)
      return true;
  }
  else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
    return IsBarrierFunction(ICE->getSubExpr());

  return false;
}

// Check if Expr is an atomic function
bool CLUtils::IsAtomicFunction(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    string FuncName = DRE->getDecl()->getNameAsString();
    const int NUM_ATOMICS = sizeof(cl_atomics) / sizeof(string);
    if (SearchString(FuncName, cl_atomics, NUM_ATOMICS) != -1)
      return true;
  }
  else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
    return IsAtomicFunction(ICE->getSubExpr());

  return false;
}

bool CLUtils::IsWorkItemIDFunction(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    string FuncName = DRE->getDecl()->getNameAsString();
    if (FuncName.compare("get_global_id") == 0 ||
        FuncName.compare("get_local_id") == 0)
      return true;
  }
  else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
    return IsWorkItemIDFunction(ICE->getSubExpr());

  return false;
}

bool CLUtils::IsInvariantFunction(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    string FuncName = DRE->getDecl()->getNameAsString();
    if (FuncName.compare("__CL_SAFE_INT_DIV_ZERO") == 0)
      return true;
  }
  else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E))
    return IsWorkItemIDFunction(ICE->getSubExpr());

  return false;
}
//-------------------------------------------------------------------------//

//-------------------------------------------------------------------------//
// Global expression
//-------------------------------------------------------------------------//
bool CLUtils::IsGlobalPointerExprForDeref(Expr *E) {
  QualType ETy = E->getType();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (ETy->isPointerType()) {
      if (VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
        if (IsGlobal(VD)) return true;
        else if (IsConstant(VD)) {
          if (!VD->isDefinedOutsideFunctionOrMethod())
            return true;
        }
      }
    }
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    // need to know whether sub-expression is global
    return IsGlobalPointerExprForDeref(PE->getSubExpr());
  } else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(E)) {
    // FIXME: Currently, single global expression makes whole expression 
    //        global
    return IsGlobalPointerExprForDeref(BO->getLHS()) ||
           IsGlobalPointerExprForDeref(BO->getRHS());
  } else if (CStyleCastExpr *CSCE = dyn_cast<CStyleCastExpr>(E)) {
    // FIXME: how can we handle (__global ... *)(...)?
    QualType Ty = CSCE->getTypeAsWritten();
    return IsGlobalPointerExprForDeref(CSCE->getSubExpr());
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    return IsGlobalPointerExprForDeref(ICE->getSubExpr());
  }

  return false;
}
//-------------------------------------------------------------------------//

Expr *CLUtils::IgnoreImpCasts(Expr *E) {
  while (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    E = ICE->getSubExpr();
  }
  return E;
}

bool CLUtils::IsVectorArrayType(QualType Ty) {
  if (const ArrayType *AT = Ty->getAsArrayTypeUnsafe()) {
    QualType ElemTy = AT->getElementType();
    if (ElemTy->isExtVectorType())
      return true;
    else if (ElemTy->isScalarType())
      return false;
    else
      return IsVectorArrayType(ElemTy);
  }

  return false;
}

