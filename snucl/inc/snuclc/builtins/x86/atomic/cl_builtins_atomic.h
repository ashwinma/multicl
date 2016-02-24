/*****************************************************************************/
/* Copyright (C) 2010, 2011 Seoul National University                        */
/* and Samsung Electronics Co., Ltd.                                         */
/*                                                                           */
/* Contributed by Sangmin Seo <sangmin@aces.snu.ac.kr>, Jungwon Kim          */
/* <jungwon@aces.snu.ac.kr>, Jaejin Lee <jlee@cse.snu.ac.kr>, Seungkyun Kim  */
/* <seungkyun@aces.snu.ac.kr>, Jungho Park <jungho@aces.snu.ac.kr>,          */
/* Honggyu Kim <honggyu@aces.snu.ac.kr>, Jeongho Nah                         */
/* <jeongho@aces.snu.ac.kr>, Sung Jong Seo <sj1557.seo@samsung.com>,         */
/* Seung Hak Lee <s.hak.lee@samsung.com>, Seung Mo Cho                       */
/* <seungm.cho@samsung.com>, Hyo Jung Song <hjsong@samsung.com>,             */
/* Sang-Bum Suh <sbuk.suh@samsung.com>, and Jong-Deok Choi                   */
/* <jd11.choi@samsung.com>                                                   */
/*                                                                           */
/* All rights reserved.                                                      */
/*                                                                           */
/* This file is part of the SNU-SAMSUNG OpenCL runtime.                      */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is free software: you can redistribute it  */
/* and/or modify it under the terms of the GNU Lesser General Public License */
/* as published by the Free Software Foundation, either version 3 of the     */
/* License, or (at your option) any later version.                           */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is distributed in the hope that it will be */
/* useful, but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  */
/* Public License for more details.                                          */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with the SNU-SAMSUNG OpenCL runtime. If not, see                    */
/* <http://www.gnu.org/licenses/>.                                           */
/*****************************************************************************/

#ifndef _cl_cpu_atom_h_
#define _cl_cpu_atom_h_

//#include <asm/system.h>

//atomic functions for global variables
#define atomic_add_global		atomic_add
#define atomic_sub_global		atomic_sub
#define atomic_inc_global		atomic_inc
#define atomic_dec_global		atomic_dec
#define atomic_xchg_global	atomic_xchg
#define atomic_cmpxchg_global	  atomic_cmpxchg
#define atomic_min_global   atomic_min
#define atomic_max_global   atomic_max
#define atomic_and_global   atomic_and
#define atomic_or_global    atomic_or
#define atomic_xor_global   atomic_xor


//atomic functions for local variables
#define atomic_add_local		atomic_add
#define atomic_sub_local		atomic_sub
#define atomic_inc_local		atomic_inc
#define atomic_dec_local		atomic_dec
#define atomic_xchg_local 	atomic_xchg
#define atomic_cmpxchg_local	atomic_cmpxchg
#define atomic_min_local    atomic_min
#define atomic_max_local    atomic_max
#define atomic_and_local    atomic_and
#define atomic_or_local     atomic_or
#define atomic_xor_local    atomic_xor

#define atomic_read(v)          (*v)

// int

inline int atomic_add(volatile int *v, int i)
{
	return __sync_fetch_and_add(v, i);
}

inline int atomic_sub(volatile int *v, int i)
{
	return __sync_fetch_and_sub(v, i);
}

inline int atomic_inc(volatile int *v)
{
	return __sync_fetch_and_add(v, 1);
}

inline int atomic_dec(volatile int *v)
{
	return __sync_fetch_and_sub(v, 1);
}

inline int atomic_cmpxchg(volatile int* v, int oldv, int newv)
{
	return __sync_val_compare_and_swap(v, oldv, newv);
}

inline int atomic_xchg(volatile int* v, int i)
{
	return __sync_lock_test_and_set(v, i);
}

inline int atomic_max(volatile int* v, int i)
{
	int old;
	do {
		old = *v;
		if (old >= i)
			break;
	} while (!__sync_bool_compare_and_swap(v, old, i));
	return old;
}

inline int atomic_min(volatile int* v, int i)
{
	int old;
	do {
		old = *v;
		if (old <= i)
			break;
	} while (!__sync_bool_compare_and_swap(v, old, i));
	return old;
}

inline int atomic_and(volatile int *v, int i)
{
	return __sync_fetch_and_and(v, i);
}

inline int atomic_or(volatile int *v, int i)
{
	return __sync_fetch_and_or(v, i);
}

inline int atomic_xor(volatile int *v, int i)
{
	return __sync_fetch_and_xor(v, i);
}

// float

inline float atomic_xchg(volatile float * v, float i)
{
	unsigned int r = __sync_lock_test_and_set((volatile unsigned int*)v, *((unsigned int*)(&i)));
	return *((float*)(&r));
}

// unsigned int

inline unsigned int atomic_add(volatile unsigned int *v, unsigned int i)
{
	return __sync_fetch_and_add(v, i);
}

inline unsigned int atomic_sub(volatile unsigned int *v, unsigned int i)
{
	return __sync_fetch_and_sub(v, i);
}

inline unsigned int atomic_inc(volatile unsigned int *v)
{
	return __sync_fetch_and_add(v, 1);
}

inline unsigned int atomic_dec(volatile unsigned int *v)
{
	return __sync_fetch_and_sub(v, 1);
}

inline unsigned int atomic_cmpxchg(volatile unsigned int* v, unsigned int oldv, unsigned int newv)
{
	return __sync_val_compare_and_swap(v, oldv, newv);
}

inline unsigned int atomic_xchg(volatile unsigned int* v, unsigned int i)
{
	return __sync_lock_test_and_set(v, i);  
}

inline unsigned int atomic_max(volatile unsigned int* v, unsigned int i)
{
	unsigned int old;
	do {
		old = *v;
		if (old >= i)
			break;
	} while (!__sync_bool_compare_and_swap(v, old, i));
	return old;
}

inline unsigned int atomic_min(volatile unsigned int* v, unsigned int i)
{
	unsigned int old;
	do {
		old = *v;
		if (old <= i)
			break;
	} while (!__sync_bool_compare_and_swap(v, old, i));
	return old;
}

inline unsigned int atomic_and(volatile unsigned int *v, unsigned int i)
{
	return __sync_fetch_and_and(v, i);
}

inline unsigned int atomic_or(volatile unsigned int *v, unsigned int i)
{
	return __sync_fetch_and_or(v, i);
}

inline unsigned int atomic_xor(volatile unsigned int *v, unsigned int i)
{
	return __sync_fetch_and_xor(v, i);
}

#endif /* _cl_cpu_atom_h_ */
