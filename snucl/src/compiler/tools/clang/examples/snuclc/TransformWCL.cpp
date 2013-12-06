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

#include "TransformWCL.h"
#include "TypeDefs.h"
using namespace clang;
using namespace clang::snuclc;


void TransformWCL::Transform(FunctionDecl *FD, unsigned MaxDim) {
  setMaxDim(MaxDim);

  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  assert(FDBody && "Function body must be a CompoundStmt");

  FD->setBody(VisitCompoundStmt(FDBody));
}

Stmt *TransformWCL::VisitStmt(Stmt *S) {
  switch (S->getStmtClass()) {
  default: return S;
  case Stmt::CompoundStmtClass:
    return VisitCompoundStmt(static_cast<CompoundStmt*>(S));
  case Stmt::IfStmtClass:
    return VisitIfStmt(static_cast<IfStmt*>(S));
  case Stmt::WhileStmtClass:
    return VisitWhileStmt(static_cast<WhileStmt*>(S));
  }

  return S;
}

Stmt *TransformWCL::VisitCompoundStmt(CompoundStmt *S) {
  if (S->isWCR()) {
    // If S is a WCR, we enclose it with a WCL.
    if (S->size() > 0)
      return CLExprs.getWCL(S, getMaxDim());
    else {
      ASTCtx.Deallocate(S);
      return 0;
    }
  } else {
    StmtVector Stmts;
    for (CompoundStmt::body_iterator I = S->body_begin(), E = S->body_end();
         I != E; ++I) {
      Stmt *Node = VisitStmt(*I);
      if (Node) Stmts.push_back(Node);
    }
    S->setStmts(ASTCtx, Stmts.data(), Stmts.size());
  }

  return S;
}

Stmt *TransformWCL::VisitIfStmt(IfStmt *S) {
  // Then
  S->setThen(VisitStmt(S->getThen()));
  
  // Else
  if (Stmt *Else = S->getElse()) {
    S->setElse(VisitStmt(Else));
  }

  return S;
}

Stmt *TransformWCL::VisitWhileStmt(WhileStmt *S) {
  // Body
  S->setBody(VisitStmt(S->getBody()));
  return S;
}
