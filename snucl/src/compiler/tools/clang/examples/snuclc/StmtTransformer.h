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

#ifndef SNUCLC_STMTTRANSFORMER_H
#define SNUCLC_STMTTRANSFORMER_H

#include "clang/AST/StmtVisitor.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/Support/Format.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"

namespace clang {
namespace snuclc {

/// StmtTransformer
class StmtTransformer : public StmtVisitor<StmtTransformer, Stmt *> {
public:
  StmtTransformer() {}
  virtual ~StmtTransformer() {}

  Stmt *TransformStmt(Stmt *S) {
    if (S) return Visit(S);
    return S;
  }

  Expr *TransformExpr(Expr *E) {
    if (E) return dyn_cast<Expr>(Visit(E));
    return E;
  }

  Stmt *Visit(Stmt* S) {
    return StmtVisitor<StmtTransformer, Stmt *>::Visit(S);
  }

  Stmt *VisitStmt(Stmt *Node) LLVM_ATTRIBUTE_UNUSED {
    assert(0 && "<<unknown stmt type>>");
    return Node;
  }

  Stmt *VisitExpr(Expr *Node) LLVM_ATTRIBUTE_UNUSED {
    assert(0 && "<<unknown expr type>>");
    return Node;
  }

  CompoundStmt *TransformRawCompoundStmt(CompoundStmt *Node);
  Stmt *TransformRawDeclStmt(DeclStmt *S);
  Stmt *TransformRawIfStmt(IfStmt *If);

#define ABSTRACT_STMT(CLASS)
#define STMT(CLASS, PARENT) \
  virtual Stmt *Visit##CLASS(CLASS *Node);
#include "clang/AST/StmtNodes.inc"
}; //end class StmtTransformer

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_STMTTRANSFORMER_H

