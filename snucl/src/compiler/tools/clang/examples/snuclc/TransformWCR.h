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

#ifndef SNUCLC_TRANSFORMWCR_H
#define SNUCLC_TRANSFORMWCR_H

#include "CLExpressions.h"
#include "DataStructures.h"

namespace clang {
namespace snuclc {

/// TransformWCR
/// - Identify WCRs.
class TransformWCR {
public:
  typedef std::set<CompoundStmt *> WCRSetTy;
  typedef WCRSetTy::iterator       iterator;
  iterator begin()  { return WCRs.begin(); }
  iterator end()    { return WCRs.end(); }

private:
  ASTContext    &ASTCtx;
  DeclContext   *DeclCtx;
  CLExpressions &CLExprs;
  llvm::raw_ostream &OS;

  /// Numerical ID of WCR (Work-item Coalescing Region)
  int WCR_ID;

  /// The set of WCRs
  WCRSetTy WCRs;

  /// Landing pad map
  typedef std::map<Stmt *, Stmt *> LandPadMapTy;
  LandPadMapTy LandPadMap;

  /// DeclStmts for condition VarDecls created by transformation.
  StmtVector CondDecls;
  unsigned   CondNum;

public:
  TransformWCR(ASTContext &C, DeclContext *D, CLExpressions &Exprs,
               llvm::raw_ostream &O)
    : ASTCtx(C), DeclCtx(D), CLExprs(Exprs), OS(O) {
    WCR_ID = 0;
    CondNum = 0;
  }
  void Transform(FunctionDecl *FD);

  unsigned getWCRSize() { return WCRs.size(); }
  CompoundStmt *getWCR(int id);
  void setWCR(int id, CompoundStmt *WCR);
  void eraseWCR(CompoundStmt *WCR);

  Stmt *getLoopOfLandPad(Stmt *pad);
  bool isLandPad(Stmt *S);

private:
  void CheckParmVarDecls(FunctionDecl *FD, StmtVector &Body, StmtVector &WCR);

  void VisitRawCompoundStmt(CompoundStmt *S, StmtVector &Body, 
                             StmtVector &WCR);
  void VisitRawDeclStmt(DeclStmt *DS, StmtVector &Body, StmtVector &WCR);
  Stmt *VisitStmt(Stmt *S);
  Stmt *VisitCompoundStmt(CompoundStmt *S);
  Stmt *VisitSwitchStmt(SwitchStmt *S);
  Stmt *VisitIfStmt(IfStmt *S);
  Stmt *VisitWhileStmt(WhileStmt *S);
  Stmt *VisitDoStmt(DoStmt *S);
  Stmt *VisitForStmt(ForStmt *S);

  void SerializeStmtList(StmtVector &Stmts, CompoundStmt *SL);
  void UnwrapStmtList(StmtVector &Stmts, StmtVector &Body);
  CompoundStmt *MergeConsecutiveWCRs(CompoundStmt *WCR1, CompoundStmt *WCR2);
  Stmt *MergeWCRsAndMakeCompoundStmt(StmtVector &Body);
  Stmt *MergeCondDeclsAndBody(CompoundStmt *Body);
  void  MergeBodyAndWCR(StmtVector &Body, StmtVector &WCR);
  void  MergeBodyAndCond(StmtVector &Body, Stmt *Cond);

  VarDecl *NewCondVarDecl(QualType T);
  VarDecl *NewVarDeclForParameter(ParmVarDecl *PD);
  DeclStmt *NewDeclStmt(VarDecl *VD);
  CompoundStmt *NewCompoundStmt(StmtVector &Stmts);
  CompoundStmt *NewStmtList(StmtVector &Stmts);
  CompoundStmt *NewWCR(StmtVector &Stmts);
  void MakeWCR(CompoundStmt *CS);
  Stmt *NewLandingPad(Stmt *loop);

  // For debugging
  void printWCRs();
  void printLandPads();
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_TRANSFORMWCR_H

