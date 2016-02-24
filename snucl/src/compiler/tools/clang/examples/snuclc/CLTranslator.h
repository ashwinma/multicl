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

#ifndef SNUCLC_CLTRANSLATOR_H
#define SNUCLC_CLTRANSLATOR_H

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/AST.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/SmallVector.h"
#include "CLOptions.h"
#include "CLUtils.h"
#include "CLExpressions.h"
#include "DataStructures.h"

namespace clang {
namespace snuclc {

/// CLTranslator - Translate ASTs to C code
class CLTranslator : public ASTConsumer {
  CLOptions         &CLOpts;
  llvm::raw_ostream &Out;
  PrintingPolicy    *Policy;
  CLExpressions     *CLExprs;
  ASTContext        *ASTCtx;

  KernelMapTy  KernelMap;   // mapping between a kernel FD and its KernelInfo
  CallMapTy    CallMap;     // callers & callees for each FD
  VarDeclSetTy LocalVDSet;  // set of VDs declared in the kernel code

public:

  CLTranslator(CLOptions &clOpts, llvm::raw_ostream *o = NULL)
    : CLOpts(clOpts), Out(o ? *o : llvm::outs()) {
    Policy = NULL;
    CLExprs = NULL;
    ASTCtx = NULL;
  }

  ~CLTranslator();

  virtual void Initialize(ASTContext &Context);
  virtual void HandleTopLevelDecl(DeclGroupRef D);
  virtual void HandleInterestingDecl(DeclGroupRef D) {}
  virtual void HandleTranslationUnit(ASTContext &Context);
  virtual void HandleTagDeclDefinition(TagDecl *D) {}
  virtual void CompleteTentativeDefinition(VarDecl *D) {}
  virtual void HandleVTable(CXXRecordDecl *RD, bool DefinitionRequired) {}
  virtual ASTMutationListener *GetASTMutationListener() { return 0; }
  virtual ASTDeserializationListener *GetASTDeserializationListener() { return 0; }
  virtual void PrintStats() {}

private:
  void GatherKernelInfo();
  void TransformAST();
  void GenerateCode();

  void InsertTlbGetKey(FunctionDecl *FD);
  void TransformLocalVDName();

  void PrepareTransformation();
  void CheckCyclesInCallGraph();
  bool VisitInCallGraph(FunctionCallInfo &V);
  void UpdateAllMaxDimension();
  void UpdateMaxDimension(FuncDeclSetTy &Callers, unsigned CalleeMaxDim);
  void UpdateAllHasGotoStmt();
  void UpdateHasGotoStmt(FuncDeclSetTy &Callers);
  void UpdateAllHasIndirectBarrierCall();
  void UpdateHasIndirectBarrierCall(FuncDeclSetTy &Callers);
  void InlineFunctions();
  void UpdateFullyInlined();
  void UpdateAllHasBarrier();
  void DuplicateKernelFunctionDecls();

  void PrintHeaderCode();
  void PrintLocalVarDecls();

  void PrintFooterCode();
  void PrintKernelDependentCode();
  void PrintKernelNamesArray();
  void PrintFunctionStructs();
  void PrintKernelLaunch();
  void PrintKernelSetStack();
  void PrintKernelSetArguments();
  void PrintKernelFini();
  void PrintKernelInfoForMerge();
  void PrintKernelInfo();

  DiagnosticBuilder Diag(unsigned DiagID);
  void HandleError();

  // Debugging functions
  void printCallMap(CallMapTy &CallMap);

};

} //end namespace snuclc
} //end namespace clang


#endif //SNUCLC_CLTRANSLATOR_H
