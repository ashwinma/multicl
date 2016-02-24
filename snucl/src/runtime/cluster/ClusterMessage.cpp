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

#include "cluster/ClusterMessage.h"
#include <algorithm>
#include <cstring>
#include <vector>
#include <malloc.h>
#include <mpi.h>
#include <stdint.h>

using namespace std;

ClusterSendMessage::ClusterSendMessage(bool fixed) {
  current_chunk_ = (char*)memalign(4096, CLUSTER_MESSAGE_SIZE);
  if (!fixed)
    chunks_.push_back(current_chunk_);
  offset_ = (fixed ? 0 : sizeof(uint64_t));
  fixed_ = fixed;
}

ClusterSendMessage::~ClusterSendMessage() {
  if (fixed_) {
    free(current_chunk_);
  } else {
    for (vector<char*>::iterator it = chunks_.begin();
         it != chunks_.end();
         ++it) {
      free(*it);
    }
  }
}

void ClusterSendMessage::WriteInt(int32_t v) {
  Write(&v, sizeof(int32_t));
}

void ClusterSendMessage::WriteUInt(uint32_t v) {
  Write(&v, sizeof(uint32_t));
}

void ClusterSendMessage::WriteLong(int64_t v) {
  Write(&v, sizeof(int64_t));
}

void ClusterSendMessage::WriteULong(uint64_t v) {
  Write(&v, sizeof(uint64_t));
}

void ClusterSendMessage::WriteBool(bool v) {
  Write(&v, sizeof(bool));
}

void ClusterSendMessage::WriteBuffer(const void* p, size_t size) {
  Write(p, size);
}

void ClusterSendMessage::WriteString(const char* str) {
  Write(str, strlen(str) + 1);
}

void ClusterSendMessage::Send(int target, int tag) {
  if (fixed_) {
    MPI_Send(current_chunk_, CLUSTER_MESSAGE_SIZE, MPI_BYTE, target, tag,
             MPI_COMM_WORLD);
  } else {
    *((uint64_t*)chunks_.front()) = chunks_.size();
    for (vector<char*>::iterator it = chunks_.begin();
         it != chunks_.end();
         ++it) {
      MPI_Send(*it, CLUSTER_MESSAGE_SIZE, MPI_BYTE, target, tag,
               MPI_COMM_WORLD);
    }
  }
}

void ClusterSendMessage::Write(const void* data, size_t size) {
  if (fixed_) {
    memcpy(current_chunk_ + offset_, data, size);
    offset_ += size;
  } else {
    while (size > 0) {
      size_t copy_size = size;
      if (copy_size > CLUSTER_MESSAGE_SIZE - offset_)
        copy_size = CLUSTER_MESSAGE_SIZE - offset_;
      if (copy_size == 0) {
        current_chunk_ = (char*)memalign(4096, CLUSTER_MESSAGE_SIZE);
        chunks_.push_back(current_chunk_);
        offset_ = 0;
        copy_size = (size < CLUSTER_MESSAGE_SIZE ? size :
                                                   CLUSTER_MESSAGE_SIZE);
      }
      memcpy(current_chunk_ + offset_, data, copy_size);
      offset_ += copy_size;
      data = (const void*)((size_t)data + copy_size);
      size -= copy_size;
    }
  }
}

ClusterRecvMessage::ClusterRecvMessage(bool fixed) {
  first_chunk_ = (char*)memalign(4096, CLUSTER_MESSAGE_SIZE);
  message_ = first_chunk_;
  offset_ = 0;
  fixed_ = fixed;
}

ClusterRecvMessage::~ClusterRecvMessage() {
  free(first_chunk_);
  if (first_chunk_ != message_)
    free(message_);
}

int32_t ClusterRecvMessage::ReadInt() {
  int32_t value;
  memcpy(&value, message_ + offset_, sizeof(int32_t));
  offset_ += sizeof(int32_t);
  return value;
}

uint32_t ClusterRecvMessage::ReadUInt() {
  uint32_t value;
  memcpy(&value, message_ + offset_, sizeof(uint32_t));
  offset_ += sizeof(uint32_t);
  return value;
}

int64_t ClusterRecvMessage::ReadLong() {
  int64_t value;
  memcpy(&value, message_ + offset_, sizeof(int64_t));
  offset_ += sizeof(int64_t);
  return value;
}

uint64_t ClusterRecvMessage::ReadULong() {
  uint64_t value;
  memcpy(&value, message_ + offset_, sizeof(uint64_t));
  offset_ += sizeof(uint64_t);
  return value;
}

bool ClusterRecvMessage::ReadBool() {
  bool value;
  memcpy(&value, message_ + offset_, sizeof(bool));
  offset_ += sizeof(bool);
  return value;
}

void ClusterRecvMessage::ReadBuffer(void* p, size_t size) {
  memcpy(p, message_ + offset_, size);
  offset_ += size;
}

void* ClusterRecvMessage::ReadBuffer(size_t size) {
  void* p = message_ + offset_;
  offset_ += size;
  return p;
}

char* ClusterRecvMessage::ReadString() {
  char* str = message_ + offset_;
  offset_ += strlen(str) + 1;
  return str;
}

void ClusterRecvMessage::Recv(int source, int tag) {
  if (fixed_) {
    MPI_Recv(message_, CLUSTER_MESSAGE_SIZE, MPI_BYTE, source, tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    offset_ = 0;
  } else {
    if (first_chunk_ != message_)
      free(message_);
    MPI_Recv(first_chunk_, CLUSTER_MESSAGE_SIZE, MPI_BYTE, source, tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    uint64_t num_chunks = *((uint64_t*)first_chunk_);
    if (num_chunks == 1) {
      message_ = first_chunk_;
    } else {
      message_ = (char*)memalign(4096, CLUSTER_MESSAGE_SIZE * num_chunks);
      memcpy(message_, first_chunk_, CLUSTER_MESSAGE_SIZE);
      for (uint64_t i = 1; i < num_chunks; i++) {
        MPI_Recv(message_ + (i * CLUSTER_MESSAGE_SIZE), CLUSTER_MESSAGE_SIZE,
                 MPI_BYTE, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
    offset_ = sizeof(uint64_t);
  }
}

MPI_Request ClusterRecvMessage::Irecv(int source, int tag) {
  MPI_Request mpi_request;
  MPI_Irecv(message_, CLUSTER_MESSAGE_SIZE, MPI_BYTE, source, tag,
            MPI_COMM_WORLD, &mpi_request);
  offset_ = 0;
  return mpi_request;
}
