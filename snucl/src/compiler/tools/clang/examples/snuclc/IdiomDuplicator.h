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

#ifndef SNUCLC_IDIOMDUPLICATOR_H
#define SNUCLC_IDIOMDUPLICATOR_H

#include "CLExpressions.h"
#include "TypeDefs.h"

namespace clang {
namespace snuclc {

/// IdiomDuplicator
class IdiomDuplicator {
  ASTContext    &ASTCtx;
  CLExpressions &CLExprs;
  llvm::raw_ostream &OS;

  DeclContext *DeclCtx;
  CompoundStmt *CurWCR;

  typedef std::set<DeclRefExpr *>  UseSetTy;
  typedef std::map<CompoundStmt *, UseSetTy> WCRUsesMapTy;

  class DefUseInfo {
  public:
    Stmt         *DefStmt;
    CompoundStmt *DefWCR;
    WCRUsesMapTy WCRUsesMap;

    DefUseInfo() : DefStmt(NULL), DefWCR(NULL) {}

    DefUseInfo(Stmt *defS, CompoundStmt *defWCR)
      : DefStmt(defS), DefWCR(defWCR) {}
  };

  typedef std::map<VarDecl *, DefUseInfo> DefUseInfoMapTy;

  /// VarDecl -> <Def, Uses>
  DefUseInfoMapTy DefUseMap;

  /// VarDecls that are defined.
  VarDeclSetTy DefinedVDs;

public:
  IdiomDuplicator(ASTContext &C, CLExpressions &Exprs, llvm::raw_ostream &O)
    : ASTCtx(C), CLExprs(Exprs), OS(O) {
    DeclCtx = ASTCtx.getTranslationUnitDecl();
    CurWCR = NULL;
  }

  void Duplicate(FunctionDecl *FD);

private:
  void FindAllIdioms(Stmt *S);
  bool IsIdiomExpr(Expr *E);
  void InsertDefStmt(VarDecl *VD, Stmt *Def);
  VarDecl *GetDefinedVarDecl(Expr *E);

  void CopyAllIdioms();
  void CopyStmtIntoWCR(Stmt *S, CompoundStmt *WCR);
  void RemoveStmtInWCR(Stmt *S, CompoundStmt *WCR);

  // Debugging functions
  void printDefUseMap();
}; //end class IdiomDuplicator

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_IDIOMDUPLICATOR_H

