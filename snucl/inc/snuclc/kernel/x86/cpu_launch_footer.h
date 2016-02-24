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

void mem_fence(cl_mem_fence_flags flags) {}
void read_mem_fence(cl_mem_fence_flags flags) {}
void write_mem_fence(cl_mem_fence_flags flags) {}
void barrier(cl_mem_fence_flags flags) {}

#define FLAT_SPLIT

int handle_kernel(uint64_t argp)
{
  TLB_GET_KEY;

  unsigned int i, j, k;
  unsigned int job_id;

#ifdef FLAT_SPLIT
  size_t wg_id;
  size_t wg_id_start;
  size_t wg_id_end;
#else
  size_t wg_id[3];
  size_t wg_id_start[3];
  size_t wg_id_size[3];
  size_t wg_id_end[3];
#endif
  size_t wg_size[3];
    
  memcpy((void*) &TLB_GET(param_ctx), (void*) argp, sizeof(TLB_GET(param_ctx)));

  TLB_GET(work_dim) = TLB_GET(param_ctx).work_dim;
  job_id = TLB_GET(param_ctx).launch_id;
  TLB_GET(kernel_idx) = TLB_GET(param_ctx).kernel_idx;

  kernel_set_arguments();

  for (i = 0; i < 3; i++)
  {
    TLB_GET(__global_size[i]) = TLB_GET(param_ctx).orig_grid[i] * TLB_GET(param_ctx).lws[i];
    TLB_GET(__local_size[i]) = TLB_GET(param_ctx).lws[i];
    TLB_GET(__num_groups[i]) = TLB_GET(param_ctx).orig_grid[i];
    TLB_GET(__global_offset[i]) = TLB_GET(param_ctx).gwo[i];
    wg_size[i] = TLB_GET(param_ctx).orig_grid[i];
#ifndef FLAT_SPLIT
    wg_id_start[i] = TLB_GET(param_ctx).wg_id_begin[i];
    wg_id_size[i] = TLB_GET(param_ctx).wg_id_size[i];
    wg_id_end[i] = wg_id_start[i] + wg_id_size[i];
#endif
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

#ifndef FLAT_SPLIT
  for (wg_id[2] = wg_id_start[2]; wg_id[2] < wg_id_end[2]; wg_id[2]++) {
    for (wg_id[1] = wg_id_start[1]; wg_id[1] < wg_id_end[1]; wg_id[1]++) {
      for (wg_id[0] = wg_id_start[0]; wg_id[0] < wg_id_end[0]; wg_id[0]++) {
        TLB_GET(__group_id[0]) = wg_id[0];
        TLB_GET(__group_id[1]) = wg_id[1];
        TLB_GET(__group_id[2]) = wg_id[2];
        TLB_GET(__global_id[0]) = wg_id[0] * TLB_GET(__local_size[0]);
        TLB_GET(__global_id[1]) = wg_id[1] * TLB_GET(__local_size[1]);
        TLB_GET(__global_id[2]) = wg_id[2] * TLB_GET(__local_size[2]);
        kernel_launch();
      }
    }
  }
#else
  wg_id_start = TLB_GET(param_ctx).wg_id_start;
  wg_id_end = TLB_GET(param_ctx).wg_id_end;

  for (wg_id = wg_id_start; wg_id < wg_id_end; wg_id++) {
    switch (TLB_GET(work_dim)) {
      case 1:
        TLB_GET(__group_id[0]) = wg_id;
        TLB_GET(__global_id[0]) = TLB_GET(__group_id[0]) * TLB_GET(__local_size[0]) + TLB_GET(__global_offset[0]);
        break;
      case 2:
        TLB_GET(__group_id[0]) = wg_id % wg_size[0];
        TLB_GET(__group_id[1]) = wg_id / wg_size[0];

        TLB_GET(__global_id[0]) = TLB_GET(__group_id[0]) * TLB_GET(__local_size[0]) + TLB_GET(__global_offset[0]);
        TLB_GET(__global_id[1]) = TLB_GET(__group_id[1]) * TLB_GET(__local_size[1]) + TLB_GET(__global_offset[1]);
        break;
      case 3:
        TLB_GET(__group_id[0]) = wg_id % wg_size[0];
        TLB_GET(__group_id[1]) = (wg_id % (wg_size[0] * wg_size[1])) / wg_size[0];
        TLB_GET(__group_id[2]) = wg_id / (wg_size[0] * wg_size[1]);

        TLB_GET(__global_id[0]) = TLB_GET(__group_id[0]) * TLB_GET(__local_size[0]) + TLB_GET(__global_offset[0]);
        TLB_GET(__global_id[1]) = TLB_GET(__group_id[1]) * TLB_GET(__local_size[1]) + TLB_GET(__global_offset[1]);
        TLB_GET(__global_id[2]) = TLB_GET(__group_id[2]) * TLB_GET(__local_size[2]) + TLB_GET(__global_offset[2]);
        break;
    }
    kernel_launch();
  }
#endif
  kernel_fini();

  return 0;
}

TLB_STATIC_ALLOC;

extern "C" {
int dev_entry(int id, void *argp) {
	TLB_SET_KEY;
	TLB_GET(cu_id) = (int) id;
	handle_kernel((uint64_t) argp);
	TLB_FREE_KEY;
	return 0;
} 
}
