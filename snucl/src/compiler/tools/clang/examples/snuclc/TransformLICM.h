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

#ifndef SNUCLC_TRANSFORMLICM_H
#define SNUCLC_TRANSFORMLICM_H

#include "clang/Analysis/CFG.h"
#include "TransformWCR.h"
#include "ReachingDefinitions.h"
#include "CLExpressions.h"
#include "TypeDefs.h"

namespace clang {
namespace snuclc {

class TransformLICM {
  ASTContext          &ASTCtx;
  CLExpressions       &CLExprs;
  TransformWCR        &WCR;
  llvm::raw_ostream   &OS;
  ReachingDefinitions *RD;

  StmtSetTy    AllStmts;           // All stmts in all CFGBlocks
  VarDeclSetTy VariantSet;         // WCL variant VarDecls

  /// DU-Chains
  typedef std::map<VarDecl*, StmtSetTy>   VarDeclStmtsMapTy;
  typedef std::map<Stmt*, StmtSetTy>      DUChainsTy;
  typedef std::map<VarDecl*, DUChainsTy>  VarDeclDUChainsTy;
  VarDeclStmtsMapTy CurDefMap;
  VarDeclDUChainsTy VarDeclDUChains;

  /// Loop invariant code motion
  typedef std::set<CFGBlock*>          BlkSetTy;
  typedef std::map<int, BlkSetTy>      WCRBlksMapTy;
  typedef std::map<Stmt*, bool>        StmtBoolMapTy;
  typedef std::map<VarDecl*, unsigned> VarDeclUIntMapTy;
  WCRBlksMapTy     WCRBlksMap;      // WCR -> set of CFGBlocks
  StmtBoolMapTy    StmtInvMap;      // Stmt -> invariant? (true is invariant)
  VarDeclUIntMapTy VarDeclInvMap;   // VarrDecl -> invariant?

  CFGBlock *CurCFGBlk;      // current CFGBlock
  Stmt     *CurStmt;        // current Stmt in CFGBlock
  int       CurWCRID;       // current WCR ID
  StmtVector *CurBody;      // current body of CompoundStmt

public:
  TransformLICM(ASTContext &C, CLExpressions &CLE, TransformWCR &wcr,
                llvm::raw_ostream &os)
    : ASTCtx(C), CLExprs(CLE), WCR(wcr), OS(os) {
    RD = NULL;

    CurCFGBlk = NULL;
    CurStmt   = NULL;
    CurWCRID  = -1;
    CurBody   = NULL;
  }

  void Transform(FunctionDecl *FD);

private:
  // Loop-variant VarDecls
  void FindLoopVariants(FunctionDecl *FD, CFG &cfg);
  void FindLoopVariantsInStmt(Stmt *S);
  void FindLoopVariantsInSubStmt(Stmt *S);
  bool IsVariantExpr(Expr *S);
  VarDecl *FindVarDeclOfExpr(Expr *E);

  // DU-Chains
  void FindDUChains(CFG &cfg);
  void FindDUChainsInStmt(Stmt *S);
  void FindDUChainsInSubStmt(Stmt *S);
  void InsertDefIntoCurDefMap(VarDecl *VD);
  VarDecl *GetDefinedVarDecl(Expr *E, bool wrMode=true);
  VarDecl *GetDefinedPointerVarDecl(Expr *E);

  // Loop invariant code motion
  void DoLICM(CFG &cfg);
  void FindCFGBlocksOfWCR(CFG &cfg);
  void FindInvariantCode(int wcrID, CFG &cfg);
  unsigned DetermineInvariance(Stmt *S);
  bool IsWCLInvariant(Stmt *S);
  bool HasOtherReachingDef(VarDecl *VD);
  bool HasReachableUse(VarDecl *VD);

  // Code rearrangement
  void RearrangeCode(CompoundStmt *body);
  Stmt *RearrangeCodeInStmt(Stmt *S);
  void RearrangeCodeInCompoundStmt(CompoundStmt *S);
  void MergeConsecutiveWCRs(CompoundStmt *WCR1, CompoundStmt *WCR2);

  /// Printing method
  void printWCRBlksMap();
  void printVariantSet();
  void printDUChains(CFG &cfg);
  void printCurDefMap();
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_TRANSFORMLICM_H

