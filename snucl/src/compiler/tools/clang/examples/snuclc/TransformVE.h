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

#ifndef SNUCLC_TRANSFORMVE_H
#define SNUCLC_TRANSFORMVE_H

#include "llvm/Support/raw_ostream.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/CFG.h"
#include "TypeDefs.h"
#include "CLExpressions.h"
#include "TransformWCR.h"
#include "DUChains.h"
#include "Dominator.h"
#include "PostDominator.h"

namespace clang {
namespace snuclc {

/// TransformVE
/// - class for variable expansion
class TransformVE {
  ASTContext    &ASTCtx;
  DeclContext   *DeclCtx;
  CLExpressions &CLExprs;
  CFG           &cfg;
  TransformWCR  &TWCR;
  DUChains      &DU;
  llvm::raw_ostream &OS;

  typedef DUChains::WebTy           WebTy;
  typedef DUChains::WebPtrSetTy     WebPtrSetTy;
  typedef DUChains::ParWebTy        ParWebTy;
  typedef DUChains::VarDeclParWebTy VarDeclParWebTy;

  /// Maximum dimension
  unsigned MaxDim;

  /// Numerical ID of Three-dimensional array VarDecl
  unsigned MaxDimVDNum;

  /// The set of all uses to be replaced.
  typedef std::set<DeclRefExpr *> UseSetTy;
  UseSetTy UseSet;

  /// To find the LCA from dominator tree and postdominator tree.
  /// VarDecl is a created one to be used in reference replacements.
  typedef std::set<CFGBlock *>                    CFGBlockSetTy;
  typedef std::pair<CFGBlockSetTy, CFGBlockSetTy> CFGBlockPairTy;
  typedef std::map<VarDecl *, CFGBlockPairTy>     VarDeclCFGTy;
  VarDeclCFGTy VarDeclCFGs;

  /// DeclStmts of malloc variables.
  StmtVector MallocDecls;

  /// Default stmt where malloc/free codes are attached.
  Stmt *DefaultPos;
  typedef std::map<Stmt *, VarDeclSetTy> StmtDeclMapTy;
  StmtDeclMapTy MallocStmtMap;
  StmtDeclMapTy FreeStmtMap;

  /// Free position
  typedef enum { FREE_BEFORE, FREE_AFTER } FreePosTy;
  typedef std::map<VarDecl *, FreePosTy> FreePosMapTy;
  FreePosMapTy FreePosMap;

public:
  TransformVE(ASTContext &C, CLExpressions &exprs, CFG &cfg,
              TransformWCR &twcr, DUChains &du, llvm::raw_ostream &os)
    : ASTCtx(C), CLExprs(exprs), cfg(cfg), TWCR(twcr), DU(du), OS(os) {
    // DeclContext
    DeclCtx = ASTCtx.getTranslationUnitDecl();

    MaxDimVDNum = 0;
    DefaultPos = 0;
  }
  void Transform(FunctionDecl *FD, unsigned MaxDim);

private:
  unsigned getNewVarDeclID()      { return MaxDimVDNum++; }
  void     setMaxDim(unsigned MD) { MaxDim = MD; }
  unsigned getMaxDim()            { return MaxDim; }

  void  ReplaceRefs(FunctionDecl *FD);
  void  ReplaceRefsInPartition(VarDecl *VD, const WebPtrSetTy &webs);
  void  ReplaceDef(Stmt *def, VarDecl *oldVD, VarDecl *newVD);
  Expr *ReplaceDefInExpr(VarDecl *oldVD, VarDecl *newVD, Expr *E);
  bool  ReplaceUseVarDecl(DeclRefExpr *use, VarDecl *oldVD, VarDecl *newVD);
  Stmt *ReplaceUseInStmt(Stmt *S);
  Expr *ReplaceUseInExpr(Expr *S);

  // FIXME: Are the below three functions necessary?
  void ReplaceUnusedDeclStmt(VarDecl *VD, DeclStmt *DS);
  void ReplaceStmtInWCR(Stmt *def, Stmt *newDef, CompoundStmt *wcr);
  void InsertDeclStmtBeforeWCR(DeclStmt *DS, CompoundStmt *wcr);

  bool IdiomRecognition(VarDecl *VD, ParWebTy &pars);
  bool IsIdiomStmt(Stmt *S);
  bool IsIdiomExpr(Expr *E);
  void InsertDefIntoWCR(Stmt *def, CompoundStmt *wcr);
  void RemoveDefInWCR(Stmt *def, CompoundStmt *wcr);

  VarDecl *NewMaxDimVarDecl(VarDecl *VD);
  ArraySubscriptExpr *NewMaxDimArrayRef(DeclRefExpr *LHS, QualType T);

  // malloc/free code
  void InsertMemoryCodes(FunctionDecl *FD);
  void MarkMallocPosition(Dominator &Dom, CFGBlock *B, VarDecl *VD);
  void MarkFreePosition(PostDominator &PostDom, CFGBlock *B, VarDecl *VD);
  Stmt *InsertMallocFree(Stmt *S, Stmt *upS=0);
  Stmt *InsertRemainingMalloc(CompoundStmt *CS);
  Stmt *InsertRemainingFree(CompoundStmt *CS);
  Stmt *MergeMallocCodeAndStmt(VarDeclSetTy &declSet, Stmt *S);
  Stmt *MergeFreeCodeAndStmt(VarDeclSetTy &declSet, Stmt *S);

  Stmt *Make1DMallocCode(VarDeclSetTy &declSet, Stmt *S);

  Stmt *Make2DMallocCode(VarDeclSetTy &declSet, Stmt *S);
  CompoundStmt *Make2DSecondLevelMalloc(VarDeclSetTy &declSet);

  Stmt *Make3DMallocCode(VarDeclSetTy &declSet, Stmt *S);
  CompoundStmt *Make3DSecondLevelMalloc(VarDeclSetTy &declSet);
  CompoundStmt *Make3DThirdLevelMalloc(VarDeclSetTy &declSet);

  void MakeFreeCode(StmtVector &Stmts, VarDeclSetTy &declSet);
  CompoundStmt *Make2DSecondLevelFree(VarDeclSetTy &declSet);
  CompoundStmt *Make3DSecondLevelFree(VarDeclSetTy &declSet);
  CompoundStmt *Make3DThirdLevelFree(VarDeclSetTy &declSet);
  Stmt *MergeMallocDeclsAndBody(Stmt *S);

  // Debugging functions
  void printLCA(VarDecl *VD, CFGBlockSetTy &blkSet,
                CFGBlock *LCA, CFGBlock *newLCA);
  void printStmtDeclMap(StmtDeclMapTy &stmtDeclMap);
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_TRANSFORMVE_H
