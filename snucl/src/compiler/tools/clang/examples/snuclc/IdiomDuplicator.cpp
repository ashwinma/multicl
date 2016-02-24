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

#include "IdiomDuplicator.h"
#include "FunctionDuplicator.h"
#include "CLStmtPrinter.h"
#include "CLUtils.h"
using namespace clang;
using namespace clang::snuclc;

//#define CL_DEBUG

void IdiomDuplicator::Duplicate(FunctionDecl *FD) {
  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());

  // Find all idiom definitions and uses.
  FindAllIdioms(FDBody);

#ifdef CL_DEBUG
  printDefUseMap();
#endif

  // Copy each idiom definition stmt to the WCR that has its uses.
  CopyAllIdioms();
}


void IdiomDuplicator::FindAllIdioms(Stmt *S) {
  switch (S->getStmtClass()) {
    // statements
    case Stmt::DeclStmtClass: {
      DeclStmt *Node = static_cast<DeclStmt*>(S);
      if (!Node->isSingleDecl()) return;
      if (VarDecl *VD = dyn_cast<VarDecl>(Node->getSingleDecl())) {
        if (VD->hasInit()) {
          assert(0 && "Was StmtSimplifier executed?");
          //if (IsIdiomExpr(Init)) InsertDefStmt(VD, S);
          //DefinedVDs.insert(VD);
        }
      }
      break;
    }
    case Stmt::CompoundStmtClass: {
      CompoundStmt *Node = static_cast<CompoundStmt*>(S);
      if (Node->isWCR()) CurWCR = Node;

      CompoundStmt::body_iterator I, E;
      for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
        FindAllIdioms(*I);
      }

      if (Node->isWCR()) CurWCR = NULL;
      break;
    }
    case Stmt::IfStmtClass: {
      IfStmt *Node = static_cast<IfStmt*>(S);
      FindAllIdioms(Node->getCond());
      FindAllIdioms(Node->getThen());
      if (Stmt *Else = Node->getElse()) FindAllIdioms(Else);
      break;
    }
    case Stmt::SwitchStmtClass: {
      SwitchStmt *Node = static_cast<SwitchStmt*>(S);
      FindAllIdioms(Node->getCond());
      FindAllIdioms(Node->getBody());
      break;
    }
    case Stmt::WhileStmtClass: {
      WhileStmt *Node = static_cast<WhileStmt*>(S);
      FindAllIdioms(Node->getCond());
      FindAllIdioms(Node->getBody());
      break;
    }
    case Stmt::DoStmtClass: {
      DoStmt *Node = static_cast<DoStmt*>(S);
      FindAllIdioms(Node->getBody());
      FindAllIdioms(Node->getCond());
      break;
    }
    case Stmt::ForStmtClass: {
      ForStmt *Node = static_cast<ForStmt*>(S);
      if (Stmt *Init = Node->getInit()) FindAllIdioms(Init);
      if (Expr *Cond = Node->getCond()) FindAllIdioms(Cond);
      if (Expr *Inc  = Node->getInc())  FindAllIdioms(Inc);
      FindAllIdioms(Node->getBody());
      break;
    }
    case Stmt::ReturnStmtClass: {
      ReturnStmt *Node = static_cast<ReturnStmt*>(S);
      if (Expr *Ret = Node->getRetValue()) FindAllIdioms(Ret);
      break;
    }
    case Stmt::CaseStmtClass: {
      CaseStmt *Node = static_cast<CaseStmt*>(S);
      FindAllIdioms(Node->getLHS());
      if (Expr *RHS = Node->getRHS()) FindAllIdioms(RHS);
      if (Stmt *SubStmt = Node->getSubStmt()) FindAllIdioms(SubStmt);
      break;
    }
    case Stmt::DefaultStmtClass: {
      DefaultStmt *Node = static_cast<DefaultStmt*>(S);
      if (Stmt *SubStmt = Node->getSubStmt()) FindAllIdioms(SubStmt);
      break;
    }
    case Stmt::LabelStmtClass: {
      LabelStmt *Node = static_cast<LabelStmt*>(S);
      if (Stmt *SubStmt = Node->getSubStmt()) FindAllIdioms(SubStmt);
      break;
    }
    case Stmt::IndirectGotoStmtClass: {
      IndirectGotoStmt *Node = static_cast<IndirectGotoStmt*>(S);
      FindAllIdioms(Node->getTarget());
      break;
    }

    // expressions
    case Stmt::DeclRefExprClass: {
      DeclRefExpr *Node = static_cast<DeclRefExpr*>(S);
      if (VarDecl *VD = dyn_cast<VarDecl>(Node->getDecl())) {
        if (DefUseMap.find(VD) != DefUseMap.end()) {
          DefUseInfo &DefUse = DefUseMap[VD];
          DefUse.WCRUsesMap[CurWCR].insert(Node);
        }
      }
      break;
    }
    case Stmt::UnaryOperatorClass: {
      UnaryOperator *Node = static_cast<UnaryOperator*>(S);
      Expr *SubExpr = Node->getSubExpr();
      switch (Node->getOpcode()) {
        default: FindAllIdioms(SubExpr);
                 break;
        case UO_PostInc:
        case UO_PostDec:
        case UO_PreInc:
        case UO_PreDec:
        case UO_AddrOf: {
          if (VarDecl *VD = GetDefinedVarDecl(SubExpr)) {
            DefinedVDs.insert(VD);
            DefUseMap.erase(VD);
          } else {
            FindAllIdioms(SubExpr);
          }
          break;
        }
      }
      break;
    }
    case Stmt::BinaryOperatorClass: {
      BinaryOperator *Node = static_cast<BinaryOperator*>(S);
      Expr *LHS = Node->getLHS();
      Expr *RHS = Node->getRHS();

      if (Node->getOpcode() == BO_Assign) {
        if (VarDecl *VD = GetDefinedVarDecl(LHS)) {
          if (IsIdiomExpr(RHS)) {
            InsertDefStmt(VD, S);
          } else {
            DefUseMap.erase(VD);
            FindAllIdioms(RHS);
          }
          DefinedVDs.insert(VD);
          break;
        }
      }

      FindAllIdioms(LHS);
      FindAllIdioms(RHS);
      break;
    }
    case Stmt::CompoundAssignOperatorClass: {
      CompoundAssignOperator *Node = static_cast<CompoundAssignOperator*>(S);
      Expr *LHS = Node->getLHS();
      Expr *RHS = Node->getRHS();
      FindAllIdioms(RHS);

      if (VarDecl *VD = GetDefinedVarDecl(LHS)) {
        DefUseMap.erase(VD);
        DefinedVDs.insert(VD);
      } else {
        FindAllIdioms(LHS);
      }

      break;
    }
    case Stmt::ConditionalOperatorClass: {
      ConditionalOperator *Node = static_cast<ConditionalOperator*>(S);
      FindAllIdioms(Node->getCond());
      FindAllIdioms(Node->getLHS());
      FindAllIdioms(Node->getRHS());
      break;
    }
    case Stmt::ArraySubscriptExprClass: {
      ArraySubscriptExpr *Node = static_cast<ArraySubscriptExpr*>(S);
      FindAllIdioms(Node->getLHS());
      FindAllIdioms(Node->getRHS());
      break;
    }
    case Stmt::CallExprClass: {
      CallExpr *Node = static_cast<CallExpr*>(S);
      FindAllIdioms(Node->getCallee());
      for (unsigned i = 0, e = Node->getNumArgs(); i < e; i++) {
        FindAllIdioms(Node->getArg(i));
      }
      break;
    }
    case Stmt::MemberExprClass: {
      MemberExpr *Node = static_cast<MemberExpr*>(S);
      FindAllIdioms(Node->getBase());
      break;
    }
    case Stmt::ParenExprClass: {
      ParenExpr *Node = static_cast<ParenExpr*>(S);
      FindAllIdioms(Node->getSubExpr());
      break;
    }
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr *Node = static_cast<ImplicitCastExpr*>(S);
      FindAllIdioms(Node->getSubExpr());
      break;
    }
    case Stmt::CStyleCastExprClass: {
      CStyleCastExpr *Node = static_cast<CStyleCastExpr*>(S);
      FindAllIdioms(Node->getSubExpr());
      break;
    }
    case Stmt::CompoundLiteralExprClass: {
      CompoundLiteralExpr *Node = static_cast<CompoundLiteralExpr*>(S);
      FindAllIdioms(Node->getInitializer());
      break;
    }
    case Stmt::ExtVectorElementExprClass: {
      ExtVectorElementExpr *Node = static_cast<ExtVectorElementExpr*>(S);
      FindAllIdioms(Node->getBase());
      break;
    }
    case Stmt::InitListExprClass: {
      InitListExpr *Node = static_cast<InitListExpr*>(S);
      for (unsigned i = 0, e = Node->getNumInits(); i < e; i++) {
        if (Expr *Init = Node->getInit(i)) FindAllIdioms(Init);
      }
      break;
    }
    case Stmt::OffsetOfExprClass: {
      OffsetOfExpr *Node = static_cast<OffsetOfExpr*>(S);
      for (unsigned i = 0, n = Node->getNumComponents(); i < n; i++) {
        OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
        if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
          // Array node
          unsigned Idx = ON.getArrayExprIndex();
          FindAllIdioms(Node->getIndexExpr(Idx));
        }
      }
      break;
    }
    case Stmt::SizeOfAlignOfExprClass: {
      SizeOfAlignOfExpr *Node = static_cast<SizeOfAlignOfExpr*>(S);
      if (!Node->isArgumentType())
        FindAllIdioms(Node->getArgumentExpr());
      break;
    }
    case Stmt::VecStepExprClass: {
      VecStepExpr *Node = static_cast<VecStepExpr*>(S);
      if (!Node->isArgumentType())
        FindAllIdioms(Node->getArgumentExpr());
      break;
    }
    case Stmt::DesignatedInitExprClass: {
      DesignatedInitExpr *Node = static_cast<DesignatedInitExpr*>(S);
      for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; i++) {
        if (Expr *SubExpr = Node->getSubExpr(i)) {
          FindAllIdioms(SubExpr);
        }
      }
      FindAllIdioms(Node->getInit());
      break;
    }
    case Stmt::ParenListExprClass: {
      ParenListExpr *Node = static_cast<ParenListExpr*>(S);
      for (unsigned i = 0, e = Node->getNumExprs(); i < e; i++) {
        FindAllIdioms(Node->getExpr(i));
      }
      break;
    }
    case Stmt::VAArgExprClass: {
      VAArgExpr *Node = static_cast<VAArgExpr*>(S);
      FindAllIdioms(Node->getSubExpr());
      break;
    }

    default: break;
  }
}

bool IdiomDuplicator::IsIdiomExpr(Expr *E) {
  E = E->IgnoreParenCasts();
  if (CallExpr *CE = dyn_cast<CallExpr>(E)) {
    if (CLUtils::IsWorkItemIDFunction(CE->getCallee())) {
      Expr *arg = CE->getArg(0);
      arg = arg->IgnoreParenCasts();
      return isa<IntegerLiteral>(arg);
    }
  }
  
  return false;
}

void IdiomDuplicator::InsertDefStmt(VarDecl *VD, Stmt *Def) {
  if (DefinedVDs.find(VD) != DefinedVDs.end()) {
    // Since VD has multiple definitions, it cannot be an idiom.
    DefUseMap.erase(VD);
  } else {
    DefUseMap.insert(
        std::pair<VarDecl *, DefUseInfo>(VD, DefUseInfo(Def, CurWCR)));
  }
}

/// Find the defined VarDecl of assignment LHS.
VarDecl *IdiomDuplicator::GetDefinedVarDecl(Expr *E) {
  VarDecl *VD = NULL;

  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    VD = dyn_cast<VarDecl>(DRE->getDecl());
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    VD = GetDefinedVarDecl(ICE->getSubExpr());
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    VD = GetDefinedVarDecl(PE->getSubExpr());
  }

  return VD;
}


void IdiomDuplicator::CopyAllIdioms() {
  for (DefUseInfoMapTy::reverse_iterator I = DefUseMap.rbegin(), E = DefUseMap.rend();
       I != E; ++I) {
    DefUseInfo &DefUse = (*I).second;
    Stmt *DefStmt = DefUse.DefStmt;
    CompoundStmt *DefWCR = DefUse.DefWCR;

    WCRUsesMapTy &WCRUsesMap = DefUse.WCRUsesMap;
    for (WCRUsesMapTy::iterator WI = WCRUsesMap.begin(),
         WE = WCRUsesMap.end(); WI != WE; ++WI) {
      CompoundStmt *UseWCR = (*WI).first;
      if (DefWCR == UseWCR) continue;

      // Insert DefStmt into UseWCR.
      CopyStmtIntoWCR(DefStmt, UseWCR);
    }

    // If the definition is not used in its WCR, remove it.
    if (WCRUsesMap.find(DefWCR) == WCRUsesMap.end()) {
      RemoveStmtInWCR(DefStmt, DefWCR);
    }
  }
}

void IdiomDuplicator::CopyStmtIntoWCR(Stmt *S, CompoundStmt *WCR) {
  StmtVector Stmts;

  // Duplicate S and insert it into the first position of WCR.
  FunctionDuplicator FDup(ASTCtx, CLExprs);
  Stmts.push_back(FDup.TransformStmt(S));

  CompoundStmt::body_iterator I, E;
  for (I = WCR->body_begin(), E = WCR->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }

  WCR->setStmts(ASTCtx, Stmts.data(), Stmts.size());
}

void IdiomDuplicator::RemoveStmtInWCR(Stmt *S, CompoundStmt *WCR) {
  StmtVector Stmts;

  // FIXME: Currently, only the top-level definition is removed.
  CompoundStmt::body_iterator I, E;
  for (I = WCR->body_begin(), E = WCR->body_end(); I != E; ++I) {
    if (*I != S) Stmts.push_back(*I);
  }

  WCR->setStmts(ASTCtx, Stmts.data(), Stmts.size());
}


//---------------------------------------------------------------------------
// Debugging Functions
//---------------------------------------------------------------------------
void IdiomDuplicator::printDefUseMap() {
  for (DefUseInfoMapTy::iterator I = DefUseMap.begin(), E = DefUseMap.end();
       I != E; ++I) {
    VarDecl    *VD     = (*I).first;
    DefUseInfo &DefUse = (*I).second;
    OS << VD->getName() << ":\n";

    // Definition
    OS << "  Def:\n";
    OS << "    WCR(" << DefUse.DefWCR->getWCRID() << "): ";
    CLStmt::printPretty(OS, ASTCtx, 0, ASTCtx.PrintingPolicy, 0,
                        DefUse.DefStmt);
    OS << "\n";

    // Uses
    OS << "  Uses:\n";
    WCRUsesMapTy &WCRUsesMap = DefUse.WCRUsesMap;
    for (WCRUsesMapTy::iterator WI = WCRUsesMap.begin(),
         WE = WCRUsesMap.end(); WI != WE; ++WI) {
      CompoundStmt *wcr = (*WI).first;
      UseSetTy &UseSet  = (*WI).second;
      OS << "    WCR(" << wcr->getWCRID() << "): " << UseSet.size() << "\n";
    }
  }
}

