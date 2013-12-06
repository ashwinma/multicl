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

#include "clang/Analysis/CFG.h"
#include "TransformCoalescing.h"
#include "StmtSimplifier.h"
#include "BarrierMarker.h"
#include "IdiomDuplicator.h"
#include "ReachingDefinitions.h"
#include "DUChains.h"
#include "TransformWCR.h"
#include "TransformWCL.h"
#include "TransformLICM.h"
#include "TransformVE.h"
#include "TransformFlowKeyword.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

//#define CL_DEBUG


void TransformCoalescing::Transform(FunctionDecl *FD, KernelInfo &KInfo) {
  if (!FD->hasBody()) return;

  // If FD has both barrier() invocation and GotoStmt, currently we cannot
  // apply the coalescing transformation.
  if (FD->hasGotoStmt() && FD->hasBarrierCall()) {
    // warning
    llvm::errs() << "warning: " << FD->getName() << " is not transformed "
                    "with the coalescing due to 'goto'.\n";

    setIsTransformed(false);
    return;
  }

  // FIXME: Need to modify the runtime code to support MaxDim.
  KInfo.setMaxDim(3);

  // If FD is for the task, we do not apply the coalescing.
  if (KInfo.getMaxDim() == KernelInfo::TASKDIM) {
    setIsTransformed(false);
    return;
  }

  // Simplify AST nodes before coalescing.
  StmtSimplifier SS(ASTCtx, CLExprs);
  SS.Transform(FD);

  // Mark true for each AST node whose subtrees contain a barrier() call.
  if (FD->hasBarrierCall()) {
    BarrierMarker BM;
    BM.Mark(FD);
  }

  // Identify Work-item Coalescing Regions (WCRs).
  TransformWCR TWCR(ASTCtx, DeclCtx, CLExprs, OS);
  TWCR.Transform(FD);

  // Copy idiom expressions between WCRs.
  // - This reduces the amount of variable expansions.
  if (FD->hasBarrierCall()) {
    IdiomDuplicator IDup(ASTCtx, CLExprs, OS);
    IDup.Duplicate(FD);
  }

  // FIXME: Apply loop invariant code motion to each WCR.
//  TransformLICM TLICM(ASTCtx, CLExprs, TWCR, OS);
//  TLICM.Transform(FD);

  // If FD has barrier() calls, it may need the variable expansion.
  if (FD->hasBarrierCall()) {
    Stmt *FDBody = FD->getBody();

    // Construct a source-level CFG.
    CFG::BuildOptions BO;   // CFG build options
    BO.NeedWCL = true;
    CFG *BodyCFG = CFG::buildCFG(NULL, FDBody, &ASTCtx, BO);
#ifdef CL_DEBUG
    BodyCFG->print(OS, ASTCtx.getLangOptions());
    OS << "\n";
#endif

    // Reaching definitions analysis
    ReachingDefinitions RD(ASTCtx, OS);
    RD.Solve(*BodyCFG, true);

    // Find du-chains for each variable.
    DUChains duchain(ASTCtx, OS, *BodyCFG, RD);
    duchain.FindDUChains();

    // Find webs for each variable.
    duchain.FindWebs();

    // Partition webs to find references to replace.
    duchain.PartitionWebs();

    // Variable Expansion
    // - Get a new three dimensional array t.
    // - Replace each reference x in the partition with t[__k][__j][__i].
    // - Add malloc() and free() nodes.
    TransformVE TVE(ASTCtx, CLExprs, *BodyCFG, TWCR, duchain, OS);
    TVE.Transform(FD, KInfo.getMaxDim());

    delete BodyCFG;
  }

  // Replace control flow kerword such as break and return in WCRs.
  TransformFlowKeyword TFK(ASTCtx, TWCR, OS);

  // Enclose each WCR with a WCL.
  TransformWCL TWCL(ASTCtx, CLExprs);
  TWCL.Transform(FD, KInfo.getMaxDim());

  setIsTransformed(true);
}

