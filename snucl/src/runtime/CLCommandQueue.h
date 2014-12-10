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

#ifndef __SNUCL__CL_COMMAND_QUEUE_H
#define __SNUCL__CL_COMMAND_QUEUE_H

#include <list>
#include <pthread.h>
#include <CL/cl.h>
#include "CLObject.h"
#include "Structs.h"
#include "Utils.h"
#include "RealTimer.h"

class CLCommand;
class CLContext;
class CLDevice;
class CLEvent;

class CLCommandQueue: public CLObject<struct _cl_command_queue,
                                      CLCommandQueue, struct _emu_cl_command_queue> {
 protected:
  CLCommandQueue(CLContext* context, CLDevice* device,
                 cl_command_queue_properties properties);

 public:
  virtual ~CLCommandQueue();

  CLContext* context() const { return context_; }
  CLDevice* device() const { return device_; }
  cl_int set_device(CLDevice *d);

  cl_int GetCommandQueueInfo(cl_command_queue_info param_name,
                             size_t param_value_size, void* param_value,
                             size_t* param_value_size_ret);

  bool IsProfiled() const {
    return (properties_ & CL_QUEUE_PROFILING_ENABLE);
  }

  bool IsAutoDeviceSelection() const {
  	//return (type_ == CL_QUEUE_AUTO_DEVICE_SELECTION);
  	return (properties_ & CL_QUEUE_AUTO_DEVICE_SELECTION);
  }

  virtual CLCommand* Peek() = 0;
  virtual void Enqueue(CLCommand* command) = 0;
  virtual void Dequeue(CLCommand* command) = 0;
  virtual void Flush(bool special_event = false) = 0;
  void PrintInfo();
  std::list<CLCommand *> commands() const {
  	return commands_;
  }

  void clearCommands() {commands_.clear(); }

  //bool isEpochRecorded(std::string epoch);
  //void recordEpoch(std::string epoch, std::vector<double> performances);
  void accumulateEpoch(std::vector<double> &performances);
  //std::vector<double> getEpochCosts(std::string epoch);
  std::vector<double> getAccumulatedEpochCosts();
  void resetAccumulatedEpochCosts();
  void set_perf_model_done(bool val) {perfModDone_ = val;}
  bool get_perf_model_done() {return perfModDone_;}
  void set_properties(cl_command_queue_properties properties) {properties_ = properties; }
  cl_command_queue_properties get_properties() {return properties_; }

  size_t get_h2d_size() const { return cumulative_h2d_sizes; } 
  size_t get_d2h_size() const { return cumulative_d2h_sizes; } 
 protected:
  CLDevice* SelectBestDevice(CLContext *context, CLDevice* device, 
                               cl_command_queue_properties properties);
  void InvokeScheduler();
  Global::RealTimer gQueueTimer;
  CLDevice* device_;
  std::list<CLCommand*> commands_;
  size_t cumulative_h2d_sizes;
  size_t cumulative_d2h_sizes;
 private:
  typedef std::vector<double> devicePerfVector;
  devicePerfVector accumulatedPerformances_;
  //std::map<std::string, devicePerfVector> epochPerformances_; 
  bool perfModDone_;
  bool explicitEpochMarkerSet_;
  CLContext* context_;
  cl_command_queue_properties properties_;
  cl_command_queue_type type_;

 public:
  static CLCommandQueue* CreateCommandQueue(
      CLContext* context, CLDevice* device,
      cl_command_queue_properties properties, cl_int* err);
};

class CLInOrderCommandQueue: public CLCommandQueue {
 public:
  CLInOrderCommandQueue(CLContext* context, CLDevice* device,
                        cl_command_queue_properties properties);
  ~CLInOrderCommandQueue();

  CLCommand* Peek();
  void Enqueue(CLCommand* command);
  void Dequeue(CLCommand* command);
  void Flush(bool special_event = false);
  //std::list<CLCommand *> commands() const {
  //	return commands_;
  //}

 private:
  //std::list<CLCommand*> commands_;
  LockFreeQueueMS queue_;
  CLEvent* last_event_;
};

class CLOutOfOrderCommandQueue: public CLCommandQueue {
 public:
  CLOutOfOrderCommandQueue(CLContext* context, CLDevice* device,
                           cl_command_queue_properties properties);
  ~CLOutOfOrderCommandQueue();

  CLCommand* Peek();
  void Flush(bool special_event = false) {}
  void Enqueue(CLCommand* command);
  void Dequeue(CLCommand* command);
  //std::list<CLCommand *> commands() const {
  //	return commands_;
  //}

 private:
  //std::list<CLCommand*> commands_;
  pthread_mutex_t mutex_commands_;
};

#endif // __SNUCL__CL_COMMAND_QUEUE_H
