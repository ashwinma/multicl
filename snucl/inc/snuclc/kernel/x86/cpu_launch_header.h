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
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <malloc.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>

#ifdef __cplusplus
}
#endif

#include <cpu_common.h>
#include <cl_cpu_types.h>


#ifdef __CL_DEVICE_DEBUG
#define CL_DEV_LOG(format,...) fprintf(stdout, "[LOG] "format, __VA_ARGS__);
#define CL_DEV_ERR(format,...) fprintf(stderr, "[ERR] "format, __VA_ARGS__);
#else
#define CL_DEV_LOG(A,B,C)
#define CL_DEV_ERR(A,B,C)
#endif //__CL_DEVICE_DEBUG


/////////////////////////////////////////////////////////////////////////////
#ifdef USE_TLB
/////////////////////////////////////////////////////////////////////////////
#define TLS

#define TLB_STR_FUNC_START        typedef struct {
#define TLB_STR_FUNC_END          } tlb_function;

#define TLB_GET_KEY_FUNC          \
  tlb_function *tlb_func = (tlb_function *)pthread_getspecific(key_function);
#define TLB_GET_FUNC(X)           (tlb_func->X)

#define TLB_INIT                                        \
  pthread_key_t key_data;                               \
  static pthread_key_t key_function;                    \
  static void __attribute__((constructor)) start() {    \
    pthread_key_create(&key_data,NULL);                 \
    pthread_key_create(&key_function,NULL);             \
  }                                                     \
  static void __attribute__((destructor)) end() {       \
    pthread_key_delete(key_data);                       \
    pthread_key_delete(key_function);                   \
  }

#define TLB_STATIC_ALLOC                                        \
  static tlb_data tlb_data_pool[MAX_CONTEXT_COUNT];             \
  static tlb_function tlb_func_pool[MAX_CONTEXT_COUNT];  

#define TLB_SET_KEY                                     \
  tlb_data *tlb = &tlb_data_pool[id];                   \
  tlb_function *tlb_func = &tlb_func_pool[id];          \
  pthread_setspecific(key_data,tlb);                    \
  pthread_setspecific(key_function,tlb_func);
#define TLB_FREE_KEY                                    \
  pthread_setspecific(key_data,NULL);                   \
  pthread_setspecific(key_function,NULL);


/////////////////////////////////////////////////////////////////////////////
#else //USE_TLB
#ifdef USE_TLS
/////////////////////////////////////////////////////////////////////////////
#define TLS                       __thread

#define TLB_STR_FUNC_START  
#define TLB_STR_FUNC_END    

#define TLB_GET_KEY_FUNC  
#define TLB_GET_FUNC(X)           (X)

#define TLB_INIT
#define TLB_STATIC_ALLOC
#define TLB_SET_KEY
#define TLB_FREE_KEY          


/////////////////////////////////////////////////////////////////////////////
#else //USE_TLS - TLB and TLS are not used.
/////////////////////////////////////////////////////////////////////////////
#define TLS

#define TLB_STR_DATA_START  
#define TLB_STR_DATA_END    
#define TLB_STR_FUNC_START  
#define TLB_STR_FUNC_END    

#define TLB_GET_KEY_FUNC  
#define TLB_GET_FUNC(X)           (X)

#define TLB_INIT
#define TLB_STATIC_ALLOC
#define TLB_SET_KEY
#define TLB_FREE_KEY

#endif //USE_TLS
#endif //USE_TLB


/***************************************************************************/
/* Thread-Local Data                                                       */
/***************************************************************************/
#include <cpu_thread_data.h>


/***************************************************************************/
/* Initialize TLB                                                          */
/***************************************************************************/
TLB_INIT;


/***************************************************************************/
/* Constants                                                               */
/***************************************************************************/
// TODO:Delete 
#define __CL_KERNEL_DEFAULT_STACK_NUM     1
#define __CL_KERNEL_DEFAULT_STACK_SIZE    1


/***************************************************************************/
/* Macro Functions                                                         */
/***************************************************************************/
#define CL_SET_ARG_INIT()                   \
  unsigned int args_offset = 0;

#define CL_SET_ARG_GLOBAL(func,type,name)   \
  unsigned long _##name = *(unsigned long*) (TLB_GET(param_ctx).args + args_offset);  \
  func.name = (type)_##name;                \
  args_offset += sizeof(type);                              

#define CL_SET_ARG_LOCAL(func,type,name)                                    \
uint32_t _##name = *(uint32_t*) (TLB_GET(param_ctx).args + args_offset);  \
func.name = (type) memalign(128,(size_t)_##name);                         \
args_offset += sizeof(uint32_t);

#define CL_SET_ARG_PRIVATE(func,type,name)  \
  memcpy(&func.name, (void*)(TLB_GET(param_ctx).args + args_offset), sizeof(type));  \
  args_offset += sizeof(type);

/*
#define CL_FREE_LOCAL(func,name)              \
  free(func.name);
  */
#define CL_FREE_LOCAL(func,name)

