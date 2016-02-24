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

#include "llvm/ADT/StringRef.h"
#include "TransformWCR.h"
#include "Defines.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

#include <string>
#include <sstream>
using std::string;
using std::stringstream;

//#define CL_DEBUG

void TransformWCR::Transform(FunctionDecl *FD) {
  if (!FD->hasBody()) return;

  StmtVector BodyStmts;
  StmtVector WCRStmts;

  CheckParmVarDecls(FD, BodyStmts, WCRStmts);

  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  assert(FDBody && "Function body must be a CompoundStmt");

  // WCR identification.
  VisitRawCompoundStmt(FDBody, BodyStmts, WCRStmts);
  ASTCtx.Deallocate(FDBody);

  MergeBodyAndWCR(BodyStmts, WCRStmts);
  Stmt *NewBody = MergeWCRsAndMakeCompoundStmt(BodyStmts);
  NewBody = MergeCondDeclsAndBody(dyn_cast<CompoundStmt>(NewBody));

  FD->setBody(NewBody);

#ifdef CL_DEBUG
  printWCRs();
  printLandPads();
#endif
}

CompoundStmt *TransformWCR::getWCR(int id) {
  for (WCRSetTy::iterator I = WCRs.begin(), E = WCRs.end(); I != E; ++I) {
    CompoundStmt *WCR = *I;
    if (WCR->getWCRID() == id) return WCR;
  }
  assert(0 && "Invalid WCR ID");
  return 0;
}

void TransformWCR::setWCR(int id, CompoundStmt *WCR) {
  CompoundStmt *OldWCR = getWCR(id);
  WCRs.erase(OldWCR);
  WCRs.insert(WCR);
}

void TransformWCR::eraseWCR(CompoundStmt *WCR) {
  WCRs.erase(WCR);
}

Stmt *TransformWCR::getLoopOfLandPad(Stmt *pad) {
  if (LandPadMap.find(pad) != LandPadMap.end())
    return LandPadMap[pad];
  else
    return 0;
}

bool TransformWCR::isLandPad(Stmt *S) {
  return LandPadMap.find(S) != LandPadMap.end();
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void TransformWCR::CheckParmVarDecls(FunctionDecl *FD, StmtVector &Body,
                                     StmtVector &WCR) {
  // If some function parameters are modified in the function body,
  // they should be saved in some temporal variables.
  for (unsigned i = 0, e = FD->getNumParams(); i < e; ++i) {
    ParmVarDecl *PD = FD->getParamDecl(i);
    if (PD->isModified()) {
      SourceLocation SL;

      // T __cl_parm_PD;
      VarDecl *VD = NewVarDeclForParameter(PD);
      VD->setInit(NULL);
      Body.push_back(NewDeclStmt(VD));

      // __cl_parm_PD = PD;
      DeclRefExpr *Init_LHS = new (ASTCtx) DeclRefExpr(VD, VD->getType(),
                                                       VK_RValue, SL);
      DeclRefExpr *Init_RHS = new (ASTCtx) DeclRefExpr(PD, PD->getType(),
                                                       VK_RValue, SL);
      Expr *InitExpr = new (ASTCtx) BinaryOperator(Init_LHS, Init_RHS,
          BO_Assign, PD->getType(), VK_RValue, OK_Ordinary, SL);
      Body.push_back(InitExpr);

      // PD = __cl_parm_PD;
      DeclRefExpr *LHS = new (ASTCtx) DeclRefExpr(PD, PD->getType(),
                                                  VK_RValue, SL);
      DeclRefExpr *RHS = new (ASTCtx) DeclRefExpr(VD, VD->getType(),
                                                  VK_RValue, SL);
      Expr *RestoreExpr = new (ASTCtx) BinaryOperator(LHS, RHS, BO_Assign,
          PD->getType(), VK_RValue, OK_Ordinary, SL);
      WCR.push_back(RestoreExpr);
    }
  }
}

/// Identify each WCR iterating the body of CompoundStmt.
void TransformWCR::VisitRawCompoundStmt(CompoundStmt *S,
                                        StmtVector &Body, 
                                        StmtVector &WCR) {
  CompoundStmt::body_iterator I, E;
  for (I = S->body_begin(), E = S->body_end(); I != E; ++I) {
    Stmt *stmt = *I;
    if (stmt->hasBarrier()) {
      MergeBodyAndWCR(Body, WCR);
      WCR.clear();
      Body.push_back(VisitStmt(stmt));
    } else {
      if (DeclStmt *DS = dyn_cast<DeclStmt>(stmt)) {
        // If there is a Decl that does not have an init expr, 
        //  it is code-motioned to the Body.
        VisitRawDeclStmt(DS, Body, WCR);
        continue;
      } else if (CompoundStmt *CS = dyn_cast<CompoundStmt>(stmt)) {
        // StmtList is divided into multiple stmts.
        if (CS->isStmtList()) {
          VisitRawCompoundStmt(CS, Body, WCR);
          ASTCtx.Deallocate((void *)CS);
          continue;
        }
      }

      // Most stmts are inserted into the WCR.
      WCR.push_back(stmt);
    }
  }
}

void TransformWCR::VisitRawDeclStmt(DeclStmt *DS, StmtVector &Body, 
                                    StmtVector &WCR) {
  // If there is a Decl that does not have an init expr, it is inserted into 
  // the Body. Otherwise, it is added to the WCR.
  assert(DS->isSingleDecl() && "Not a SingleDecl");

  if (VarDecl *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
    if (!VD->getInit()) {
      Body.push_back(DS);
      return;
    }
  } else {
    Body.push_back(DS);
    return;
  }

  WCR.push_back(DS);
}

Stmt *TransformWCR::VisitStmt(Stmt *S) {
  switch (S->getStmtClass()) {
    default: return S;
    case Stmt::CompoundStmtClass:
      return VisitCompoundStmt(static_cast<CompoundStmt*>(S));
    case Stmt::SwitchStmtClass:
      return VisitSwitchStmt(static_cast<SwitchStmt*>(S));
    case Stmt::IfStmtClass:
      return VisitIfStmt(static_cast<IfStmt*>(S));
    case Stmt::WhileStmtClass:
      return VisitWhileStmt(static_cast<WhileStmt*>(S));
    case Stmt::DoStmtClass:
      return VisitDoStmt(static_cast<DoStmt*>(S));
    case Stmt::ForStmtClass:
      return VisitForStmt(static_cast<ForStmt*>(S));
  }
  return S;
}

Stmt *TransformWCR::VisitCompoundStmt(CompoundStmt *S) {
  StmtVector Body;
  StmtVector WCR;

  // Identify each WCR
  VisitRawCompoundStmt(S, Body, WCR);

  // the last WCR
  MergeBodyAndWCR(Body, WCR);

  // Deallocate the current CompoundStmt
  ASTCtx.Deallocate(S);

  return MergeWCRsAndMakeCompoundStmt(Body);
}

/// When SwitchStmt has a barrier call in its body, it is converted into
///  a IfStmt.
Stmt *TransformWCR::VisitSwitchStmt(SwitchStmt *S) {
  assert(0 && "Currently, we do not support the case that SwitchStmt has "
              "a barrier() call in its body.");
  return S;
}

Stmt *TransformWCR::VisitIfStmt(IfStmt *S) {
  StmtVector OuterBody;

  // Declaration of the condition variable
  VarDecl *CondVD = NewCondVarDecl(S->getCond()->getType());
  CondDecls.push_back(NewDeclStmt(CondVD));
  
  // Computation of the condition
  DeclRefExpr *LHS = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  BinaryOperator *CondBinOp = new (ASTCtx) BinaryOperator(
      LHS, S->getCond(), BO_Assign, LHS->getType(), 
      VK_RValue, OK_Ordinary, SourceLocation());
  MergeBodyAndCond(OuterBody, CondBinOp);

  // Cond
  DeclRefExpr *CondRef = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  S->setCond(CondRef);

  // Then
  Stmt *Then = S->getThen();
  CompoundStmt *ThenCS = dyn_cast<CompoundStmt>(Then);
  assert(ThenCS && "Not a CompoundStmt");
  S->setThen(VisitCompoundStmt(ThenCS));
  
  // Else
  if (Stmt *Else = S->getElse()) {
    CompoundStmt *ElseCS = dyn_cast<CompoundStmt>(Else);
    assert(ElseCS && "Not a CompoundStmt");
    S->setElse(VisitCompoundStmt(ElseCS));
  }

  OuterBody.push_back(S);

  return NewStmtList(OuterBody);
}

Stmt *TransformWCR::VisitWhileStmt(WhileStmt *S) {
  StmtVector OuterBody;
  StmtVector BodyStmts;
  StmtVector WCRBody;

  // Declaration of condition variable
  VarDecl *CondVD = NewCondVarDecl(S->getCond()->getType());
  CondDecls.push_back(NewDeclStmt(CondVD));

  // Computation of condition
  DeclRefExpr *LHS = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  BinaryOperator *CondBinOp = new (ASTCtx) BinaryOperator(
      LHS, S->getCond(), BO_Assign, LHS->getType(), 
      VK_RValue, OK_Ordinary, SourceLocation());

  // Cond WCR
  MergeBodyAndCond(BodyStmts, CondBinOp);

  // Exit condition
  DeclRefExpr *CondRef = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  UnaryOperator *NotCond = new (ASTCtx) UnaryOperator(
      CondRef, UO_LNot, CondVD->getType(), VK_RValue, OK_Ordinary,
      SourceLocation());
  BreakStmt *ThenStmt = new (ASTCtx) BreakStmt(SourceLocation());
  IfStmt *ExitStmt = new (ASTCtx) IfStmt(ASTCtx, SourceLocation(), 
                                         NULL, NotCond, ThenStmt);
  BodyStmts.push_back(ExitStmt);
  
  // Body
  Stmt *Body = S->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  assert(CS && "Not a CompoundStmt");

  // Identify each WCR
  VisitRawCompoundStmt(CS, BodyStmts, WCRBody);
  MergeBodyAndWCR(BodyStmts, WCRBody);

  ASTCtx.Deallocate(S);
  ASTCtx.Deallocate(CS);

  // Outer WhileStmt 
  Expr *Cond = CLExprs.getExpr(CLExpressions::ONE);
  Body = MergeWCRsAndMakeCompoundStmt(BodyStmts);
  WhileStmt *WS = new (ASTCtx) WhileStmt(ASTCtx, NULL, Cond, Body,
                                         SourceLocation());
  OuterBody.push_back(WS);

  // Landing pad - empty WCR
  OuterBody.push_back(NewLandingPad(WS));

  return NewStmtList(OuterBody);
}

Stmt *TransformWCR::VisitDoStmt(DoStmt *S) {
  StmtVector OuterBody;
  StmtVector BodyStmts;
  StmtVector WCRBody;

  // Declaration of condition variable
  VarDecl *CondVD = NewCondVarDecl(S->getCond()->getType());
  CondDecls.push_back(NewDeclStmt(CondVD));

  // Computation of condition
  DeclRefExpr *LHS = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  BinaryOperator *CondBinOp = new (ASTCtx) BinaryOperator(
      LHS, S->getCond(), BO_Assign, LHS->getType(), 
      VK_RValue, OK_Ordinary, SourceLocation());

  // Body
  Stmt *Body = S->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  assert(CS && "Not a CompoundStmt");

  // Identify each WCR
  VisitRawCompoundStmt(CS, BodyStmts, WCRBody);
  MergeBodyAndWCR(BodyStmts, WCRBody);
  MergeBodyAndCond(BodyStmts, CondBinOp);

  ASTCtx.Deallocate(S);
  ASTCtx.Deallocate(CS);

  // Exit condition
  DeclRefExpr *CondRef = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  UnaryOperator *NotCond = new (ASTCtx) UnaryOperator(
      CondRef, UO_LNot, CondVD->getType(), VK_RValue, OK_Ordinary,
      SourceLocation());
  BreakStmt *ThenStmt = new (ASTCtx) BreakStmt(SourceLocation());
  IfStmt *ExitStmt = new (ASTCtx) IfStmt(ASTCtx, SourceLocation(), 
                                         NULL, NotCond, ThenStmt);
  BodyStmts.push_back(ExitStmt);
  
  // Outer WhileStmt 
  Expr *Cond = CLExprs.getExpr(CLExpressions::ONE);
  Body = MergeWCRsAndMakeCompoundStmt(BodyStmts);
  WhileStmt *WS = new (ASTCtx) WhileStmt(ASTCtx, NULL, Cond, Body,
                                         SourceLocation());
  OuterBody.push_back(WS);

  // Landing pad - empty WCR
  OuterBody.push_back(NewLandingPad(WS));

  return NewStmtList(OuterBody);
}

Stmt *TransformWCR::VisitForStmt(ForStmt *S) {
  StmtVector OuterBody;
  StmtVector BodyStmts;
  StmtVector WCRBody;

  // If Init of ForStmt has a DeclStmt, we make a CompoundStmt that encloses
  // an Init stmt and WhileStmt made from orginal ForStmt.
  bool HasDecl = false;

  // Init
  if (Stmt *Init = S->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Init)) {
      HasDecl = true;
      StmtVector tmpWCR;
      VisitRawDeclStmt(DS, OuterBody, tmpWCR);
      MergeBodyAndWCR(OuterBody, tmpWCR);
    } else {
      MergeBodyAndCond(OuterBody, Init);
    }
  }

  // Declaration of condition variable
  VarDecl *CondVD = NewCondVarDecl(ASTCtx.IntTy);
  CondDecls.push_back(NewDeclStmt(CondVD));

  // Computation of condition
  DeclRefExpr *LHS = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  Expr *Cond = S->getCond();
  if (!Cond) Cond = CLExprs.getExpr(CLExpressions::ONE);
  BinaryOperator *CondBinOp = new (ASTCtx) BinaryOperator(
      LHS, Cond, BO_Assign, LHS->getType(), 
      VK_RValue, OK_Ordinary, SourceLocation());

  // Cond WCR
  MergeBodyAndCond(BodyStmts, CondBinOp);

  // Exit condition
  DeclRefExpr *CondRef = new (ASTCtx) DeclRefExpr(CondVD,
      CondVD->getType(), VK_RValue, SourceLocation());
  UnaryOperator *NotCond = new (ASTCtx) UnaryOperator(
      CondRef, UO_LNot, CondVD->getType(), VK_RValue, OK_Ordinary,
      SourceLocation());
  BreakStmt *ThenStmt = new (ASTCtx) BreakStmt(SourceLocation());
  IfStmt *ExitStmt = new (ASTCtx) IfStmt(ASTCtx, SourceLocation(), 
                                         NULL, NotCond, ThenStmt);
  BodyStmts.push_back(ExitStmt);
  
  // Body
  Stmt *Body = S->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  assert(CS && "Not a CompoundStmt");

  // Identify each WCR
  VisitRawCompoundStmt(CS, BodyStmts, WCRBody);
  MergeBodyAndWCR(BodyStmts, WCRBody);

  // Inc
  if (Expr *Inc = S->getInc()) MergeBodyAndCond(BodyStmts, Inc);

  ASTCtx.Deallocate(S);
  ASTCtx.Deallocate(CS);

  // Outer WhileStmt 
  Cond = CLExprs.getExpr(CLExpressions::ONE);
  Body = MergeWCRsAndMakeCompoundStmt(BodyStmts);
  WhileStmt *WS = new (ASTCtx) WhileStmt(ASTCtx, NULL, Cond, Body,
                                         SourceLocation());
  OuterBody.push_back(WS);

  // Landing pad - empty WCR
  OuterBody.push_back(NewLandingPad(WS));

  CompoundStmt *Out = NewCompoundStmt(OuterBody);
  if (!HasDecl) Out->setIsStmtList(true);
  return Out;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void TransformWCR::SerializeStmtList(StmtVector &Stmts, CompoundStmt *SL) {
  for (CompoundStmt::body_iterator I = SL->body_begin(), E = SL->body_end();
       I != E; ++I) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(*I);
    if (CS && CS->isStmtList())
      SerializeStmtList(Stmts, CS);
    else
      Stmts.push_back(*I);
  }
  ASTCtx.Deallocate(SL);
}

void TransformWCR::UnwrapStmtList(StmtVector &Stmts, StmtVector &Body) {
  for (unsigned i = 0, e = Body.size(); i < e; i++) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Body[i]);
    if (CS && CS->isStmtList()) {
      SerializeStmtList(Stmts, CS);
    } else {
      Stmts.push_back(Body[i]);
    }
  }
}

CompoundStmt *TransformWCR::MergeConsecutiveWCRs(CompoundStmt *WCR1,
                                                 CompoundStmt *WCR2) {
  StmtVector Stmts;
  CompoundStmt::body_iterator I, E;
  for (I = WCR1->body_begin(), E = WCR1->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }
  for (I = WCR2->body_begin(), E = WCR2->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }

  // Make a new WCR.
  CompoundStmt *MergedWCR = NewWCR(Stmts);

  // Remove merged WCRs from WCR set.
  WCRs.erase(WCR1);
  WCRs.erase(WCR2);
  ASTCtx.Deallocate(WCR1);
  ASTCtx.Deallocate(WCR2);

  return MergedWCR;
}

Stmt *TransformWCR::MergeWCRsAndMakeCompoundStmt(StmtVector &Body) {
  StmtVector Stmts;
  UnwrapStmtList(Stmts, Body);

#if 1
  return NewCompoundStmt(Stmts);
#else
  StmtVector NewBody;
  for (unsigned i = 0, e = Stmts.size(); i < e; i++) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Stmts[i]);
    if (CS && CS->isWCR() && !isLandPad(CS)) {
      // Check if the next stmt is a WCR.
      while (i + 1 < e) {
        CompoundStmt *nextCS = dyn_cast<CompoundStmt>(Stmts[i+1]);
        if (nextCS && nextCS->isWCR() && !isLandPad(CS)) {
          CS = MergeConsecutiveWCRs(CS, nextCS);
          i++;
        } else break;
      }
      NewBody.push_back(CS);
    } else {
      NewBody.push_back(Stmts[i]);
    }
  }

  return NewCompoundStmt(NewBody);
#endif
}

Stmt *TransformWCR::MergeCondDeclsAndBody(CompoundStmt *Body) {
  StmtVector Stmts;

  for (unsigned i = 0, e = CondDecls.size(); i < e; i++) {
    Stmts.push_back(CondDecls[i]);

    // WE DON'T NEED THE INITIALIZATION!
#if 0
    // Initialize CondVD with CondVD = 0.
    if (DeclStmt *DS = dyn_cast<DeclStmt>(CondDecls[i])) {
      if (VarDecl *VD = dyn_cast<VarDecl>(DS->getSingleDecl())) {
        Expr *LHS = new (ASTCtx) DeclRefExpr(
            VD, VD->getType(), VK_RValue, SourceLocation());
        Expr *RHS = CLExprs.NewIntegerLiteral(0);
        Expr *Init = CLExprs.NewBinaryOperator(
            LHS, RHS, BO_Assign, VD->getType());
        Stmts.push_back(Init);
      }
    }
#endif
  }
  
  CompoundStmt::body_iterator I, E;
  for (I = Body->body_begin(), E = Body->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }
  ASTCtx.Deallocate(Body);

  return NewCompoundStmt(Stmts);
}

void TransformWCR::MergeBodyAndWCR(StmtVector &Body, StmtVector &WCR) {
  Body.push_back(NewWCR(WCR));
}

void TransformWCR::MergeBodyAndCond(StmtVector &Body, Stmt *Cond) {
  StmtVector Stmts;
  Stmts.push_back(Cond);
  CompoundStmt *CondWCR = NewWCR(Stmts);

  Body.push_back(CondWCR);
}


//---------------------------------------------------------------------------
VarDecl *TransformWCR::NewCondVarDecl(QualType T) {
  stringstream Name;
  Name << OPENCL_COND_VAR_PREFIX << CondNum++;

  string NameStr = Name.str();
  StringRef DeclName(NameStr);
  IdentifierInfo &DeclID = ASTCtx.Idents.get(DeclName);
  TypeSourceInfo *TSI = ASTCtx.getNullTypeSourceInfo();

  return VarDecl::Create(ASTCtx, DeclCtx, SourceLocation(), &DeclID, 
                         T, TSI, SC_None, SC_None);
}

VarDecl *TransformWCR::NewVarDeclForParameter(ParmVarDecl *PD) {
  stringstream Name;
  Name << "__cl_parm_" << PD->getNameAsString();

  string NameStr = Name.str();
  StringRef DeclName(NameStr);
  IdentifierInfo &DeclID = ASTCtx.Idents.get(DeclName);

  VarDecl *VD = VarDecl::Create(ASTCtx, DeclCtx, SourceLocation(), &DeclID,
                                PD->getType(), PD->getTypeSourceInfo(),
                                PD->getStorageClass(),
                                PD->getStorageClassAsWritten());
  if (PD->hasAttrs()) VD->setAttrs(PD->getAttrs());
  return VD;
}

DeclStmt *TransformWCR::NewDeclStmt(VarDecl *VD) {
  SourceLocation SL;
  Decl *Decls[1] = { VD };
  DeclGroupRef DG = DeclGroupRef::Create(ASTCtx, Decls, 1);
  return new (ASTCtx) DeclStmt(DG, SL, SL);
}

CompoundStmt *TransformWCR::NewCompoundStmt(StmtVector &Stmts) {
  SourceLocation SL;
  return new (ASTCtx) CompoundStmt(ASTCtx, Stmts.data(), Stmts.size(),
                                   SL, SL);
}

CompoundStmt *TransformWCR::NewStmtList(StmtVector &Stmts) {
  CompoundStmt *CS = NewCompoundStmt(Stmts);
  CS->setIsStmtList(true);
  return CS;
}

CompoundStmt *TransformWCR::NewWCR(StmtVector &Stmts) {
  CompoundStmt *CS = NewCompoundStmt(Stmts);
  MakeWCR(CS);
  return CS;
}

void TransformWCR::MakeWCR(CompoundStmt *CS) {
  CS->setWCR(WCR_ID++);
  WCRs.insert(CS);
}

Stmt *TransformWCR::NewLandingPad(Stmt *loop) {
  SourceLocation loc;
  CompoundStmt *pad = new (ASTCtx) CompoundStmt(ASTCtx, 0, 0, loc, loc);
  MakeWCR(pad);
  LandPadMap[pad] = loop;
  return pad;
}


////////////////////////////////////////////////////////////////////////////
/// Printing functions for debugging
////////////////////////////////////////////////////////////////////////////
void TransformWCR::printWCRs() {
  OS << "All WCRs(" << WCRs.size() << ") = { ";
  for (WCRSetTy::iterator I = WCRs.begin(), E = WCRs.end(); I != E; ++I) {
    if (I != WCRs.begin()) OS << ", ";
    CompoundStmt *WCR = *I;
    OS << "W" << WCR->getWCRID();
    if (isLandPad(WCR)) OS << "(LP)";
  }
  OS << " }\n";
  OS.flush();
}

void TransformWCR::printLandPads() {
  OS << "All Landing Pads(" << LandPadMap.size() << ") = { ";
  for (LandPadMapTy::iterator I = LandPadMap.begin(), E = LandPadMap.end();
       I != E; ++I) {
    if (I != LandPadMap.begin()) OS << ", ";
    if (CompoundStmt *pad = dyn_cast<CompoundStmt>((*I).first)) {
      OS << "W" << pad->getWCRID();
    } else {
      OS << "??";
    }
  }
  OS << " }\n";
  OS.flush();
}
