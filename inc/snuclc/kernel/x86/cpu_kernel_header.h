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
/*   Jungwon Kim, Sangmin Seo, Gangwon Jo, Jun Lee, Jeongho Nah,             */
/*   Jungho Park, Junghyun Kim, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>

#ifdef __cplusplus
}
#endif

#include <cpu_common.h>


/////////////////////////////////////////////////////////////////////////////
#ifdef USE_TLB
/////////////////////////////////////////////////////////////////////////////
#define TLS
#define TLB_KEYS                  extern pthread_key_t key_data;
#define USE_INLINE_BUILT_IN_FUNC


/////////////////////////////////////////////////////////////////////////////
#else //USE_TLB
#ifdef USE_TLS
/////////////////////////////////////////////////////////////////////////////
#define TLS                       extern __thread
#define TLB_KEYS
#define USE_MACRO_BUILT_IN_FUNC


/////////////////////////////////////////////////////////////////////////////
#else //USE_TLS - TLB and TLS are not used.
/////////////////////////////////////////////////////////////////////////////
#define TLS
#define TLB_KEYS
#define USE_MACRO_BUILT_IN_FUNC


#endif //USE_TLS
#endif //USE_TLB


/***************************************************************************/
/* Thread-Local Data                                                       */
/***************************************************************************/
#include <cpu_thread_data.h>


/***************************************************************************/
/* Declare TLB Keys                                                        */
/***************************************************************************/
TLB_KEYS;


/***************************************************************************/
/* Types and Builtins                                                      */
/***************************************************************************/
#ifdef USE_INLINE_BUILT_IN_FUNC
inline unsigned int get_work_dim() {
  TLB_GET_KEY;
  return TLB_GET(work_dim);
}
inline size_t get_global_size(unsigned N) {
  TLB_GET_KEY;
  return TLB_GET(__global_size[N]);
}
inline size_t get_global_id(unsigned N)    {
  TLB_GET_KEY;
  return TLB_GET(__global_id[N]) + (N == 0 ? TLB_GET(__i) : (N == 1 ? TLB_GET(__j) : TLB_GET(__k)));
}
inline size_t get_local_size(unsigned N) {
  TLB_GET_KEY;
  return TLB_GET(__local_size[N]);
}
inline size_t get_local_id(unsigned N) {
  TLB_GET_KEY;
  return (N == 0 ? TLB_GET(__i) : (N == 1 ? TLB_GET(__j) : TLB_GET(__k)));
}
inline size_t get_num_groups(unsigned N) {
  TLB_GET_KEY;
  return TLB_GET(__num_groups[N]);
}
inline size_t get_group_id(unsigned N) {
  TLB_GET_KEY;
  return TLB_GET(__group_id[N]);
}
inline size_t get_global_offset(unsigned N) {
  TLB_GET_KEY;
  return TLB_GET(__global_offset[N]);
}
inline size_t get_group_offset(unsigned N) {
  TLB_GET_KEY;
  return TLB_GET(__group_offset[N]);
}
#endif //USE_INLINE_BUILT_IN_FUNC

#ifdef USE_MACRO_BUILT_IN_FUNC
#define get_work_dim()        (work_dim)
#define get_global_size(N)    (__global_size[N])
#define get_global_id(N)      (__global_id[N] + (N == 0 ? __i : (N == 1 ? __j : __k)))
#define get_local_size(N)     (__local_size[N])
#define get_local_id(N)       (N == 0 ? __i : (N == 1 ? __j : __k))
#define get_num_groups(N)     (__num_groups[N])
#define get_group_id(N)       (__group_id[N])
#define get_global_offset(N)  (__global_offset[N])
#define get_group_offset(N)   (__group_offset[N])
#endif //USE_MACRO_BUILT_IN_FUNC

#include <cl_cpu_util.h>

extern void mem_fence(cl_mem_fence_flags flags);
extern void read_mem_fence(cl_mem_fence_flags flags);
extern void write_mem_fence(cl_mem_fence_flags flags);
extern void barrier(cl_mem_fence_flags flags);
