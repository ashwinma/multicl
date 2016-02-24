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
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 151-744, Korea                         */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jungwon Kim, Sangmin Seo, Gangwon Jo, Jun Lee, Jeongho Nah,             */
/*   Jungho Park, Junghyun Kim, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

#ifndef __SNUCL__CLUSTER_MESSAGE_H
#define __SNUCL__CLUSTER_MESSAGE_H

#include <vector>
#include <mpi.h>
#include <stdint.h>

#define CLUSTER_MESSAGE_SIZE 5120

#define CLUSTER_TAG_COMMAND 0
#define CLUSTER_TAG_NODE_INFO 1
#define CLUSTER_TAG_DEVICE_INFO 2
#define CLUSTER_TAG_EVENT_WAIT(id) (10 + 2 * (id) + 0)
#define CLUSTER_TAG_SEND_BODY(id) (10 + 2 * (id) + 1)
#define CLUSTER_TAG_RECV_BODY(id) (10 + 2 * (id) + 1)

#define CLUSTER_REQUEST_EXIT 0
#define CLUSTER_REQUEST_NODE_INFO 1
#define CLUSTER_REQUEST_DEVICE_INFO 2
#define CLUSTER_REQUEST_LAUNCH_KERNEL 3
#define CLUSTER_REQUEST_LAUNCH_NATIVE_KERNEL 4
#define CLUSTER_REQUEST_SEND_BUFFER 5
#define CLUSTER_REQUEST_RECV_BUFFER 6
#define CLUSTER_REQUEST_COPY_BUFFER 7
#define CLUSTER_REQUEST_SEND_IMAGE 8
#define CLUSTER_REQUEST_RECV_IMAGE 9
#define CLUSTER_REQUEST_COPY_IMAGE 10
#define CLUSTER_REQUEST_COPY_IMAGE_TO_BUFFER 11
#define CLUSTER_REQUEST_COPY_BUFFER_TO_IMAGE 12
#define CLUSTER_REQUEST_SEND_BUFFER_RECT 13
#define CLUSTER_REQUEST_RECV_BUFFER_RECT 14
#define CLUSTER_REQUEST_COPY_BUFFER_RECT 15
#define CLUSTER_REQUEST_FILL_BUFFER 16
#define CLUSTER_REQUEST_FILL_IMAGE 17
#define CLUSTER_REQUEST_BUILD_PROGRAM 18
#define CLUSTER_REQUEST_COMPILE_PROGRAM 19
#define CLUSTER_REQUEST_LINK_PROGRAM 20
#define CLUSTER_REQUEST_ALLTOALL 21
#define CLUSTER_REQUEST_BROADCAST 22
#define CLUSTER_REQUEST_FREE_MEM 23
#define CLUSTER_REQUEST_FREE_SAMPLER 24
#define CLUSTER_REQUEST_FREE_PROGRAM 25
#define CLUSTER_REQUEST_FREE_KERNEL 26

class ClusterSendMessage {
 public:
  ClusterSendMessage(bool fixed = true);
  ~ClusterSendMessage();

  char* ptr() { return current_chunk_ + offset_; }

  void WriteInt(int32_t v);
  void WriteUInt(uint32_t v);
  void WriteLong(int64_t v);
  void WriteULong(uint64_t v);
  void WriteBool(bool v);
  void WriteBuffer(const void* p, size_t size);
  void WriteString(const char* str);

  void Send(int target, int tag);

 private:
  void Write(const void* data, size_t size);

  std::vector<char*> chunks_;
  char* current_chunk_;
  size_t offset_;
  bool fixed_;
};

class ClusterRecvMessage {
 public:
  ClusterRecvMessage(bool fixed = true);
  ~ClusterRecvMessage();

  char* ptr() { return message_ + offset_; }

  int32_t ReadInt();
  uint32_t ReadUInt();
  int64_t ReadLong();
  uint64_t ReadULong();
  bool ReadBool();
  void ReadBuffer(void* p, size_t size);
  void* ReadBuffer(size_t size); // this does not allocate memory
  char* ReadString(); // this does not allocate memory

  void Recv(int source, int tag);
  MPI_Request Irecv(int source, int tag); // for fixed_ == false

 private:
  char* first_chunk_;
  char* message_;
  size_t offset_;
  bool fixed_;
};

#endif // __SNUCL__CLUSTER_MESSAGE_H
