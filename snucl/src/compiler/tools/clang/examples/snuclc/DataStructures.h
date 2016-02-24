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

#ifndef SNUCLC_DATASTRUCTURES_H
#define SNUCLC_DATASTRUCTURES_H

#include "clang/AST/Type.h"
#include "llvm/ADT/SmallVector.h"
#include "TypeDefs.h"
#include <string>

namespace clang {
namespace snuclc {

class FunctionCallInfo;

class KernelInfo {
public:
  static const unsigned TASKDIM = 100;

private:
  std::string KernelName;

  // parameter information
  unsigned NumParams;
  llvm::SmallVector<std::string, 32> ParamName;   // name
  llvm::SmallVector<unsigned, 32>    ParamAddrQ;  // address qualifier
  llvm::SmallVector<unsigned, 32>    ParamAccQ;   // access qualifier
  llvm::SmallVector<unsigned, 32>    ParamTQ;     // type qualifier
  llvm::SmallVector<QualType, 32>    ParamTy;     // type

  // attribute information
  unsigned RX, RY, RZ;    // reqd_work_group_size
  unsigned WX, WY, WZ;    // work_group_size_hint
  llvm::SmallVector<std::string, 4>  AttrStr;

  // __local memory and __local declarations
  uint64_t LocalMemSize;

  // private memory size
  uint64_t PrivateMemSize;

  // possible maximum dimension
  // 0: this function can be used for any dimension.
  // 1, 2, 3: this function should be used for N dimension.
  // 100: specify that this function is for a task.
  unsigned MaxDim;

  // whether the coalescing is applied
  bool IsCoalescingApplied;

public:
  KernelInfo() {}

  KernelInfo(FunctionDecl *FD) {
    KernelName = FD->getNameAsString();
    NumParams  = FD->getNumParams();
    RX = RY = RZ = 0;
    WX = WY = WZ = 0;
    LocalMemSize = 0;
    PrivateMemSize = 0;
    MaxDim = 0;
    IsCoalescingApplied = false;
  }

  const std::string &getName()             { return KernelName; }

  unsigned getNumParams()                  { return NumParams; }
  const std::string &getParamName(int i)   { return ParamName[i]; }
  unsigned getParamAddrQualifier(int i)    { return ParamAddrQ[i]; }
  unsigned getParamAccessQualifier(int i)  { return ParamAccQ[i]; }
  unsigned getParamTypeQualifier(int i)    { return ParamTQ[i]; }
  QualType getParamType(int i)             { return ParamTy[i]; }

  void setNumParams(unsigned n)               { NumParams = n; }
  void pushParamName(std::string name)        { ParamName.push_back(name); }
  void pushParamAddrQualifier(unsigned aq)    { ParamAddrQ.push_back(aq); }
  void pushParamAccessQualifier(unsigned aq)  { ParamAccQ.push_back(aq); }
  void pushParamTypeQualifier(unsigned tq)    { ParamTQ.push_back(tq); }
  void pushParamType(QualType qt)             { ParamTy.push_back(qt); }

  unsigned getReqdDimX()                   { return RX; }
  unsigned getReqdDimY()                   { return RY; }
  unsigned getReqdDimZ()                   { return RZ; }
  unsigned getWorkDimX()                   { return WX; }
  unsigned getWorkDimY()                   { return WY; }
  unsigned getWorkDimZ()                   { return WZ; }
  void setReqdWorkGroupSize(unsigned X, unsigned Y, unsigned Z) {
    RX = X; RY = Y; RZ = Z;
  }
  void setWorkGroupSizeHint(unsigned X, unsigned Y, unsigned Z) {
    WX = X; WY = Y; WZ = Z;
  }

  unsigned getNumAttrs()                   { return AttrStr.size(); }
  const std::string &getAttrStr(int i)     { return AttrStr[i]; }
  void pushAttrStr(std::string attr)       { AttrStr.push_back(attr); }

  uint64_t getLocalMemSize()               { return LocalMemSize; }
  void setLocalMemSize(uint64_t size)      { LocalMemSize = size; }

  uint64_t getPrivateMemSize()             { return PrivateMemSize; }
  void setPrivateMemSize(uint64_t size)    { PrivateMemSize = size; }

  unsigned getMaxDim()                     { return MaxDim; }
  void setMaxDim(unsigned MD)              { MaxDim = MD; }

  bool isCoalescingApplied()               { return IsCoalescingApplied; }
  void setCoalescingApplied(bool B)        { IsCoalescingApplied = B; }
}; //end class KernelInfo


class FunctionCallInfo {
  // possible maximum dimension
  // 0: this function can be used for any dimension.
  // 1, 2, 3: this function should be used for N dimension.
  // 100: specify that this function is for a task.
  unsigned MaxDim;

public:
  enum NODE_COLOR {
    WHITE,
    GREY,
    BLACK
  };

  NODE_COLOR    Mark;
  FuncDeclSetTy Callers;
  FuncDeclSetTy Callees;

public:
  FunctionCallInfo() {
    MaxDim = 0;
    Mark = WHITE;
  };

  void ResetMark() { Mark =  WHITE; }

  unsigned getMaxDim()          { return MaxDim; }
  void setMaxDim(unsigned MD)   { MaxDim = MD; }
}; //end class FunctionCallInfo


} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_DATASTRUCTURES_H

