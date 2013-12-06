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
#include "TransformVE.h"
#include "FunctionDuplicator.h"
#include "CLStmtPrinter.h"
#include "CLUtils.h"
#include "Defines.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

#include <string>
#include <sstream>
using namespace std;

//#define CL_DEBUG


void TransformVE::Transform(FunctionDecl *FD, unsigned MaxDim) {
  setMaxDim(MaxDim);

  // Replace each reference x in the partition with the expanded array 
  // reference.
  ReplaceRefs(FD);

  InsertMemoryCodes(FD);
}

//---------------------------------------------------------------------------
// Get a new array t whose dimension is determined by the maximum dimension
// of FD.
// Replace each reference x in the partition with the array t reference.
// For example, t is three dimensional, the reference is t[__k][__j][__i].
//---------------------------------------------------------------------------
void TransformVE::ReplaceRefs(FunctionDecl *FD) {
  VarDeclParWebTy &VarDeclParWeb = DU.getVarDeclParWeb();
  for (VarDeclParWebTy::iterator I = VarDeclParWeb.begin(),
                                 E = VarDeclParWeb.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    ParWebTy &pars = (*I).second;   // partitions

    if (!DU.hasRemainingWebs(VD) && !DU.hasUnusedDefUse(VD)) {
      // All webs are included in partitions.
      if (!IdiomRecognition(VD, pars)) {
        // Iterate partitions
        for (ParWebTy::iterator PI = pars.begin(), PE = pars.end();
             PI != PE; ++PI) {
          ReplaceRefsInPartition(VD, *PI);
        }
      }
    } else {
      // If a DeclStmt for the VarDecl is not used, the declaration of 
      //  the VarDecl should exist outside the WCR.
      if (DeclStmt *DS = DU.getUnusedDeclStmt(VD)) {
        ReplaceUnusedDeclStmt(VD, DS);
      }

      // There are some webs that are not included in partitions.
      for (ParWebTy::iterator PI = pars.begin(), PE = pars.end();
          PI != PE; ++PI) {
        ReplaceRefsInPartition(VD, *PI);
      }
    }
  }

  // Replace references of all uses with array references
  FD->setBody(ReplaceUseInStmt(FD->getBody()));
}

void TransformVE::ReplaceRefsInPartition(VarDecl *VD,
                                         const WebPtrSetTy &webs) {
  // Create a new MaxDim-dimensional array VarDecl
  VarDecl *newVD = NewMaxDimVarDecl(VD);

  CFGBlockSetTy defBlks;
  CFGBlockSetTy useBlks;

  // Iterate webs
  for (WebPtrSetTy::const_iterator I = webs.begin(), E = webs.end();
       I != E; ++I) {
    WebTy *web = *I;
    for (WebTy::iterator WI = web->begin(), WE = web->end(); 
         WI != WE; ++WI) {
      // Replace def with MaxDim array reference.
      Stmt *def = (*WI).first;
      defBlks.insert(DU.getCFGBlockofDef(def));
      ReplaceDef(def, VD, newVD);

      UseSetTy &uses = (*WI).second;
      for (UseSetTy::iterator UI = uses.begin(), UE = uses.end();
           UI != UE; ++UI) {
        DeclRefExpr *use = *UI;
        useBlks.insert(DU.getCFGBlockofUse(use));

        // VarDecl of DeclRefExpr is replaced here.
        if (ReplaceUseVarDecl(use, VD, newVD)) {
          // Since a certain use such as one of CompoundAssignOperator or
          //  UnaryOperator has same DeclRefExpr with a def, it may have 
          //  been already replaced with three-dim array references.
          //  Therefore, when a use was not replaces, it is inserted into
          //  UseSet.
          // After collecting all uses, replace them in the AST.
          UseSet.insert(use);
        }
      }
    }
  }

  VarDeclCFGs[newVD] = CFGBlockPairTy(defBlks, useBlks);
}

void TransformVE::ReplaceDef(Stmt *def, VarDecl *oldVD, VarDecl *newVD) {
  if (BinaryOperator *E = dyn_cast<BinaryOperator>(def)) {
    E->setLHS(ReplaceDefInExpr(oldVD, newVD, E->getLHS()));
  } else if (UnaryOperator *E = dyn_cast<UnaryOperator>(def)) {
    E->setSubExpr(ReplaceDefInExpr(oldVD, newVD, E->getSubExpr()));
  } else {
    assert(0 && "What is def?");
  }
}

Expr *TransformVE::ReplaceDefInExpr(VarDecl *oldVD, VarDecl *newVD, Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    VarDecl *VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (VD && VD == oldVD) {
      DRE->setDecl(newVD);
      return NewMaxDimArrayRef(DRE, oldVD->getType());   
    }
  } else if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    ASE->setLHS(ReplaceDefInExpr(oldVD, newVD, ASE->getLHS()));
    return ASE;
  } else if (MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
    ME->setBase(ReplaceDefInExpr(oldVD, newVD, ME->getBase()));
    return ME;
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    ICE->setSubExpr(ReplaceDefInExpr(oldVD, newVD, ICE->getSubExpr()));
    return ICE;
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    PE->setSubExpr(ReplaceDefInExpr(oldVD, newVD, PE->getSubExpr()));
    return PE;
  }

  return E;
}

bool TransformVE::ReplaceUseVarDecl(DeclRefExpr *use, VarDecl *oldVD, 
                                    VarDecl *newVD) {
  VarDecl *VD = dyn_cast<VarDecl>(use->getDecl());
  assert(VD && "use must have a VarDecl");

  if (VD == oldVD) {
    use->setDecl(newVD);
    return true;
  }
  return false;
}

Stmt *TransformVE::ReplaceUseInStmt(Stmt *S) {
  switch (S->getStmtClass()) {
  default: {
    if (Expr *E = dyn_cast<Expr>(S))
       return ReplaceUseInExpr(E);
    else
      return S;
  }

  // statements
  case Stmt::CompoundStmtClass: {
    CompoundStmt *Node = static_cast<CompoundStmt*>(S);
    CompoundStmt::body_iterator I, E;
    for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
      *I = ReplaceUseInStmt(*I);
    }
    return Node;
  }

  case Stmt::SwitchStmtClass: {
    SwitchStmt *Node = static_cast<SwitchStmt*>(S);
    Node->setCond(ReplaceUseInExpr(Node->getCond()));
    Node->setBody(ReplaceUseInStmt(Node->getBody()));
    return Node;
  }
  case Stmt::CaseStmtClass: {
    CaseStmt *Node = static_cast<CaseStmt*>(S);
    Node->setLHS(ReplaceUseInExpr(Node->getLHS()));
    if (Expr *RHS = Node->getRHS()) {
      Node->setRHS(ReplaceUseInExpr(RHS));
    }
    Node->setSubStmt(ReplaceUseInStmt(Node->getSubStmt()));
    return Node;
  }
  case Stmt::DefaultStmtClass: {
    DefaultStmt *Node = static_cast<DefaultStmt*>(S);
    if (Stmt *SubStmt = Node->getSubStmt())
      Node->setSubStmt(ReplaceUseInStmt(SubStmt));
    return Node;
  }
                               
  case Stmt::LabelStmtClass: {
    LabelStmt *Node = static_cast<LabelStmt*>(S);
    if (Stmt *SubStmt = Node->getSubStmt())
      Node->setSubStmt(ReplaceUseInStmt(SubStmt));
    return Node;
  }

  case Stmt::IfStmtClass: {
    IfStmt *Node = static_cast<IfStmt*>(S);
    Node->setCond(ReplaceUseInExpr(Node->getCond()));
    Node->setThen(ReplaceUseInStmt(Node->getThen()));
    if (Stmt *Else = Node->getElse()) {
      Node->setElse(ReplaceUseInStmt(Else));
    }
    return Node;
  }

  case Stmt::WhileStmtClass: {
    WhileStmt *Node = static_cast<WhileStmt*>(S);
    Node->setCond(ReplaceUseInExpr(Node->getCond()));
    Node->setBody(ReplaceUseInStmt(Node->getBody()));
    return Node;
  }

  case Stmt::DoStmtClass: {
    DoStmt *Node = static_cast<DoStmt*>(S);
    Node->setBody(ReplaceUseInStmt(Node->getBody()));
    Node->setCond(ReplaceUseInExpr(Node->getCond()));
    return Node;
  }

  case Stmt::ForStmtClass: {
    ForStmt *Node = static_cast<ForStmt*>(S);
    if (Stmt *Init = Node->getInit()) {
      Node->setInit(ReplaceUseInStmt(Init));
    }
    if (Expr *Cond = Node->getCond()) {
      Node->setCond(ReplaceUseInExpr(Cond));
    }
    if (Expr *Inc = Node->getInc()) {
      Node->setInc(ReplaceUseInExpr(Inc));
    }
    Node->setBody(ReplaceUseInStmt(Node->getBody()));
    return Node;
  }

  case Stmt::IndirectGotoStmtClass: {
    IndirectGotoStmt *Node = static_cast<IndirectGotoStmt*>(S);
    Node->setTarget(ReplaceUseInExpr(Node->getTarget()));
    return Node;
  }
  case Stmt::ReturnStmtClass: {
    ReturnStmt *Node = static_cast<ReturnStmt*>(S);
    if (Expr *RetExpr = Node->getRetValue()) {
      Node->setRetValue(ReplaceUseInExpr(RetExpr));
    }
    return Node;
  }

  case Stmt::DeclStmtClass: {
    DeclStmt *Node = static_cast<DeclStmt*>(S);
    for (DeclStmt::decl_iterator I = Node->decl_begin(), E = Node->decl_end();
         I != E; ++I) {    
      if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
        if (Expr *Init = VD->getInit()) {
          VD->setInit(ReplaceUseInExpr(Init));
        }
      }
    }
    return Node;
  }
  } // end switch

  return S;
}

Expr *TransformVE::ReplaceUseInExpr(Expr *S) {
  switch (S->getStmtClass()) {
  default: break;

  case Stmt::DeclRefExprClass: {
    DeclRefExpr *E = static_cast<DeclRefExpr*>(S);
    if (UseSet.find(E) != UseSet.end()) {
      return NewMaxDimArrayRef(E, E->getType());
    }
    return E;
  }

  case Stmt::ParenExprClass: {
    ParenExpr *E = static_cast<ParenExpr*>(S);
    E->setSubExpr(ReplaceUseInExpr(E->getSubExpr()));
    return E;
  }

  case Stmt::UnaryOperatorClass: {
    UnaryOperator *E = static_cast<UnaryOperator*>(S);
    E->setSubExpr(ReplaceUseInExpr(E->getSubExpr()));
    return E;
  }

  case Stmt::SizeOfAlignOfExprClass: {
    SizeOfAlignOfExpr *E = static_cast<SizeOfAlignOfExpr*>(S);
    if (!E->isArgumentType()) {
      E->setArgument(ReplaceUseInExpr(E->getArgumentExpr()));
    }
    return E;
  }

  case Stmt::VecStepExprClass: {
    VecStepExpr *E = static_cast<VecStepExpr*>(S);
    if (!E->isArgumentType()) {
      E->setArgument(ReplaceUseInExpr(E->getArgumentExpr()));
    }
    return E;
  }

  case Stmt::ArraySubscriptExprClass: {
    ArraySubscriptExpr *E = static_cast<ArraySubscriptExpr*>(S);
    E->setLHS(ReplaceUseInExpr(E->getLHS()));
    E->setRHS(ReplaceUseInExpr(E->getRHS()));
    return E;
  }

  case Stmt::CallExprClass: {
    CallExpr *E = static_cast<CallExpr*>(S);
    E->setCallee(ReplaceUseInExpr(E->getCallee()));
    for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
      E->setArg(i, ReplaceUseInExpr(E->getArg(i)));
    }
    return E;
  }

  case Stmt::MemberExprClass: {
    MemberExpr *E = static_cast<MemberExpr*>(S);
    E->setBase(ReplaceUseInExpr(E->getBase()));
    return E;
  }

  case Stmt::BinaryOperatorClass: {
    BinaryOperator *E = static_cast<BinaryOperator*>(S);
    E->setLHS(ReplaceUseInExpr(E->getLHS()));
    E->setRHS(ReplaceUseInExpr(E->getRHS()));
    return E;
  }

  case Stmt::CompoundAssignOperatorClass: {
    CompoundAssignOperator *E = static_cast<CompoundAssignOperator*>(S);
    E->setRHS(ReplaceUseInExpr(E->getRHS()));
    E->setLHS(ReplaceUseInExpr(E->getLHS()));
    return E;
  }

  case Stmt::ConditionalOperatorClass: {
    ConditionalOperator *E = static_cast<ConditionalOperator*>(S);
    E->setCond(ReplaceUseInExpr(E->getCond()));
    E->setLHS(ReplaceUseInExpr(E->getLHS()));
    E->setRHS(ReplaceUseInExpr(E->getRHS()));
    return E;
  }

  case Stmt::ImplicitCastExprClass: {
    ImplicitCastExpr *E = static_cast<ImplicitCastExpr*>(S);
    E->setSubExpr(ReplaceUseInExpr(E->getSubExpr()));
    return E;
  }

  case Stmt::CStyleCastExprClass: {
    CStyleCastExpr *E = static_cast<CStyleCastExpr*>(S);
    E->setSubExpr(ReplaceUseInExpr(E->getSubExpr()));
    return E;
  }

  case Stmt::CompoundLiteralExprClass: {
    CompoundLiteralExpr *E = static_cast<CompoundLiteralExpr*>(S);
    E->setInitializer(ReplaceUseInExpr(E->getInitializer()));
    return E;
  }

  case Stmt::ExtVectorElementExprClass: {
    ExtVectorElementExpr *E = static_cast<ExtVectorElementExpr*>(S);
    E->setBase(ReplaceUseInExpr(E->getBase()));
    return E;
  }

  case Stmt::InitListExprClass: {
    InitListExpr *E = static_cast<InitListExpr*>(S);
    for (unsigned i = 0, e = E->getNumInits(); i != e; ++i) {
      if (E->getInit(i)) {
        E->setInit(i, ReplaceUseInExpr(E->getInit(i)));
      }
    }
    return E;
  }

  case Stmt::DesignatedInitExprClass: {
    DesignatedInitExpr *E = static_cast<DesignatedInitExpr*>(S);
    for (unsigned i = 0, e = E->getNumSubExprs(); i < e; ++i) {
      if (E->getSubExpr(i)) {
        E->setSubExpr(i, ReplaceUseInExpr(E->getSubExpr(i)));
      }
    }
    E->setInit(ReplaceUseInExpr(E->getInit()));
    return E;
  }

  case Stmt::ParenListExprClass: {
    ParenListExpr *E = static_cast<ParenListExpr*>(S);
    for (unsigned i = 0, e = E->getNumExprs(); i != e; i++) {
      E->setExpr(i, ReplaceUseInExpr(E->getExpr(i)));
    }
    return E;
  }

  case Stmt::VAArgExprClass: {
    VAArgExpr *E = static_cast<VAArgExpr*>(S);
    E->setSubExpr(ReplaceUseInExpr(E->getSubExpr()));
    return E;
  }
  }

  return S;
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
void TransformVE::ReplaceUnusedDeclStmt(VarDecl *VD, DeclStmt *DS) {
  int defWCRID = DU.getWCRIDofDef(DS);
  if (Expr *init = VD->getInit()) {
    DeclRefExpr *lhs = new (ASTCtx) DeclRefExpr(VD, VD->getType(),
                                                VK_RValue, SourceLocation());
    BinaryOperator *newDef = CLExprs.NewBinaryOperator(
        lhs, init, BO_Assign, VD->getType());

    // DeclStmt is changed to an assignment expression.
    CompoundStmt *wcr = TWCR.getWCR(defWCRID);
    ReplaceStmtInWCR(DS, newDef, wcr);

    VD->setInit(0);
    InsertDeclStmtBeforeWCR(DS, wcr);
  }
}

void TransformVE::ReplaceStmtInWCR(Stmt *def, Stmt *newDef,
                                   CompoundStmt *wcr) {
  CompoundStmt::body_iterator I, E;
  for (I = wcr->body_begin(), E = wcr->body_end(); I != E; ++I) {
    if (*I == def) {
      *I = newDef;
      return;
    }
  }
  assert(0 && "Cannot find the def in the WCR");
}

void TransformVE::InsertDeclStmtBeforeWCR(DeclStmt *DS, CompoundStmt *wcr) {
  // Copy the WCR
  StmtVector Stmts;

  CompoundStmt::body_iterator I, E;
  for (I = wcr->body_begin(), E = wcr->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }

  CompoundStmt *newWCR = new (ASTCtx) CompoundStmt(
      ASTCtx, Stmts.data(), Stmts.size(), SourceLocation(), SourceLocation());

  int WCRID = wcr->getWCRID();
  newWCR->setWCR(WCRID);
  TWCR.setWCR(WCRID, newWCR);   // update WCR lookup table

  // Make the original wcr a StmtList
  wcr->setWCR(-1);
  wcr->setIsStmtList(true);
  Stmt *lists[2] = { DS, newWCR };
  wcr->setStmts(ASTCtx, lists, 2);
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
/// Recognize idiom expressions.
/// An expression such as "int tid = get_global_id(0);" is treated as an 
///  idiom if it is defined only once and is include in only a single web
///  and a single partition.
/// If the idiom expression is recognized, true is returned.
///  Otherwise, false is returned.
bool TransformVE::IdiomRecognition(VarDecl *VD, ParWebTy &pars) {
  if (pars.size() != 1) return false;

  const WebPtrSetTy &par = *(pars.begin());
  if (par.size() != 1) return false;

  WebTy *web = *(par.begin());
  if (web->size() != 1) return false;

  // Check if the def is an idiom expression.
  WebTy::iterator webIt = web->begin();
  Stmt *def = (*webIt).first;
  if (!IsIdiomStmt(def)) return false;

  // Get WCR IDs of def and uses.
  int defWCRID = DU.getWCRIDofDef(def);
  set<int> useWCRIDs;
  UseSetTy &uses = (*webIt).second;
  for (UseSetTy::const_iterator I = uses.begin(), E = uses.end(); 
      I != E; ++I) {
    DeclRefExpr *use = *I;
    useWCRIDs.insert(DU.getWCRIDofUse(use));
  }

  // Insert the def stmt into each use's WCR.
  for (set<int>::iterator I = useWCRIDs.begin(), E = useWCRIDs.end();
       I != E; ++I) {
    if (defWCRID != *I) {
      CompoundStmt *wcr = TWCR.getWCR(*I);
      InsertDefIntoWCR(def, wcr);
    }
  }

  // If the WCR of def does not have a use of this def, 
  //  remove this def in the WCR.
  if (useWCRIDs.find(defWCRID) == useWCRIDs.end()) {
    CompoundStmt *defWCR = TWCR.getWCR(defWCRID);
    RemoveDefInWCR(def, defWCR);
  }

  return true;
}

bool TransformVE::IsIdiomStmt(Stmt *S) {
  if (BinaryOperator *E = dyn_cast<BinaryOperator>(S)) {
    if (E->getOpcode() == BO_Assign) {
      Expr *lhs = E->getLHS();
      if (!isa<DeclRefExpr>(lhs)) return false;
      return IsIdiomExpr(E->getRHS());
    }
  }

  return false;
}

bool TransformVE::IsIdiomExpr(Expr *E) {
  E = E->IgnoreParenCasts();
  if (CallExpr *CE = dyn_cast<CallExpr>(E)) {
    if (CLUtils::IsWorkItemIDFunction(CE->getCallee())) {
      Expr *arg = CE->getArg(0);
      return isa<IntegerLiteral>(arg->IgnoreParenCasts());
    }
  }
  
  return false;
}

void TransformVE::InsertDefIntoWCR(Stmt *def, CompoundStmt *wcr) {
  assert(wcr->isWCR() && "Not WCR");

  StmtVector Stmts;

  // Duplicate S and insert it into the first position of WCR.
  FunctionDuplicator FDup(ASTCtx, CLExprs);
  Stmts.push_back(FDup.TransformStmt(def));

  CompoundStmt::body_iterator I, E;
  for (I = wcr->body_begin(), E = wcr->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }

  wcr->setStmts(ASTCtx, Stmts.data(), Stmts.size());
}

void TransformVE::RemoveDefInWCR(Stmt *def, CompoundStmt *wcr) {
  StmtVector Stmts;

  CompoundStmt::body_iterator I, E;
  for (I = wcr->body_begin(), E = wcr->body_end(); I != E; ++I) {
    if (*I != def) Stmts.push_back(*I);
  }

  wcr->setStmts(ASTCtx, Stmts.data(), Stmts.size());
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
VarDecl *TransformVE::NewMaxDimVarDecl(VarDecl *VD) {
  stringstream Name;
  Name << OPENCL_VE_PREFIX << VD->getNameAsString() << getNewVarDeclID() << "e";

  string NameStr = Name.str();
  StringRef DeclName(NameStr);
  IdentifierInfo &DeclID = ASTCtx.Idents.get(DeclName);

  // Make a MaxDim-dimensional pointer type.
  QualType T = VD->getType();
  for (unsigned i = 0; i < getMaxDim(); i++) {
    T = ASTCtx.getPointerType(T);
  }

  VarDecl *NewVD = VarDecl::Create(ASTCtx, DeclCtx, SourceLocation(), &DeclID,
                                   T, ASTCtx.getTrivialTypeSourceInfo(T),
                                   VD->getStorageClass(), 
                                   VD->getStorageClassAsWritten());
  if (VD->hasAttrs()) NewVD->setAttrs(VD->getAttrs());
  return NewVD;
}

ArraySubscriptExpr *TransformVE::NewMaxDimArrayRef(DeclRefExpr *LHS,
                                                   QualType T) {
  if (getMaxDim() == 1) {
    return CLExprs.NewArraySubscriptExpr(
        LHS, CLExprs.getExpr(CLExpressions::WCL_IDX_I), T);

  } else if (getMaxDim() == 2) {
    QualType T1 = ASTCtx.getPointerType(T);   // Type of LHS[__j]

    ArraySubscriptExpr *LHS1 = CLExprs.NewArraySubscriptExpr(
        LHS, CLExprs.getExpr(CLExpressions::WCL_IDX_J), T1);
    return CLExprs.NewArraySubscriptExpr(
        LHS1, CLExprs.getExpr(CLExpressions::WCL_IDX_I), T);

  } else {
    assert(getMaxDim() == 3 && "MaxDim should be between 1 to 3.");

    QualType T1 = ASTCtx.getPointerType(T);   // Type of LHS[__k][__j]
    QualType T2 = ASTCtx.getPointerType(T1);  // Type of LHS[__k]

    ArraySubscriptExpr *LHS2 = CLExprs.NewArraySubscriptExpr(
        LHS, CLExprs.getExpr(CLExpressions::WCL_IDX_K), T2);
    ArraySubscriptExpr *LHS1 = CLExprs.NewArraySubscriptExpr(
        LHS2, CLExprs.getExpr(CLExpressions::WCL_IDX_J), T1);
    return CLExprs.NewArraySubscriptExpr(
        LHS1, CLExprs.getExpr(CLExpressions::WCL_IDX_I), T);
  }
}
//---------------------------------------------------------------------------


//---------------------------------------------------------------------------
// malloc/free code
//---------------------------------------------------------------------------
void TransformVE::InsertMemoryCodes(FunctionDecl *FD) {
  // Construct the dominator tree.
  Dominator Dom(OS, cfg);

  // Construct the postdominator tree.
  PostDominator PostDom(OS, cfg);

  // Set DefaultPos as function's body.
  DefaultPos = FD->getBody();
  
  for (VarDeclCFGTy::iterator I = VarDeclCFGs.begin(), E = VarDeclCFGs.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    CFGBlockPairTy &blkPair = (*I).second;
    CFGBlockSetTy &defs = blkPair.first;
    CFGBlockSetTy &uses = blkPair.second;
    assert(defs.size() > 0 && "defs should be greater than 0.");
    assert(uses.size() > 0 && "uses should be greater than 0.");

    // 1. Find the LCA node in the dominator tree for all definitions.
    CFGBlockSetTy::iterator CI = defs.begin();
    CFGBlockSetTy::iterator CE = defs.end();
    CFGBlock *defLCA = *CI;
    while (++CI != CE) {
      defLCA = Dom.FindLCA(defLCA, *CI);
    }

    // 2. Find the LCA node in the postdominator tree for all uses.
    CI = uses.begin();
    CE = uses.end();
    CFGBlock *useLCA = *CI;
    while (++CI != CE) {
      useLCA = PostDom.FindLCA(useLCA, *CI);
    }

    // 3. Find the LCA between defLCA and useLCA in the dominator tree.
    CFGBlock *newDefLCA = Dom.FindLCA(defLCA, useLCA);

    // 4. Find the LCA between defLCA and useLCA in the postdominator tree.
    CFGBlock *newUseLCA = PostDom.FindLCA(useLCA, defLCA);

#ifdef CL_DEBUG
    OS.flush();
    OS << "[Dominator tree    ]: ";
    printLCA(VD, defs, defLCA, newDefLCA);
    OS.flush();
    OS << "[PostDominator tree]: ";
    printLCA(VD, uses, useLCA, newUseLCA);
    OS.flush();
#endif

    // For correctness, if newDefLCA does not dominate newUseLCA or
    //  newUseLCA does not postdominate newDefLCA, find nodes that satisfy
    //  the dominance relation.
    while (!Dom.IsDominator(newDefLCA, newUseLCA) ||
           !PostDom.IsPostDominator(newUseLCA, newDefLCA)) {
      newUseLCA = PostDom.getImmediatePostDominator(newUseLCA);
      if (newUseLCA == &cfg.getExit())
        newDefLCA = Dom.getImmediateDominator(newDefLCA);

#ifdef CL_DEBUG
      OS << "/* " << VD->getNameAsString() 
         << " (B" << newDefLCA->getBlockID() 
         << ", B" << newUseLCA->getBlockID() << ") */\n";
      OS.flush();
#endif
    }

    // 5. Find the stmt where malloc() node is attached.
    MarkMallocPosition(Dom, newDefLCA, VD);

    // 6. Find the stmt where free() node is attached.
    FreePosMap[VD] = FREE_AFTER;
    MarkFreePosition(PostDom, newUseLCA, VD);
  }

  // Insert malloc/free codes
  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  FDBody->setIsStmtList(true);
  FDBody = dyn_cast<CompoundStmt>(InsertMallocFree(FDBody));
  FD->setBody(FDBody);

  if (MallocStmtMap.size() != 0) {
#ifdef CL_DEBUG
    OS << "MallocStmtMap: \n";
    printStmtDeclMap(MallocStmtMap);
#endif
    FD->setBody(InsertRemainingMalloc(FDBody));
  }
  if (FreeStmtMap.size() != 0) {
#ifdef CL_DEBUG
    OS << "FreeStmtMap: \n";
    printStmtDeclMap(FreeStmtMap);
#endif
    FD->setBody(InsertRemainingFree(FDBody));
  }

  FD->setBody(MergeMallocDeclsAndBody(FD->getBody()));
}

void TransformVE::MarkMallocPosition(Dominator &Dom, CFGBlock *B, 
                                     VarDecl *VD) {
  Stmt *mallocStmt = DefaultPos;
  if (B->getWCRID() > -1) {
    mallocStmt = TWCR.getWCR(B->getWCRID());
  } else {
    if (B->getTerminator()) {
      mallocStmt = B->getTerminator().getStmt();
    } else if (B->size() > 0) {
      const CFGStmt *SE = B->rbegin()->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");
      mallocStmt = SE->getStmt();
    } else if (B != &cfg.getEntry()) {
      MarkMallocPosition(Dom, Dom.getImmediateDominator(B), VD);
      return;
    }
  }

  MallocStmtMap[mallocStmt].insert(VD);
}

void TransformVE::MarkFreePosition(PostDominator &PostDom, CFGBlock *B,
                                   VarDecl *VD) {
  Stmt *freeStmt = DefaultPos;
  if (B->getWCRID() > -1) {
    freeStmt = TWCR.getWCR(B->getWCRID());

    // If the WCR is a landing pad (empty WCR) of loop, free position is 
    //  located below the loop.
    if (Stmt *loopStmt = TWCR.getLoopOfLandPad(freeStmt)) {
      freeStmt = loopStmt;
      FreePosMap[VD] = FREE_AFTER;
    }
  } else {
    if (B->getTerminator()) {
      freeStmt = B->getTerminator().getStmt();
    } else if (B->size() > 0) {
      const CFGStmt *SE = B->rbegin()->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");
      freeStmt = SE->getStmt();
    } else if (B != &cfg.getExit()) {
      MarkFreePosition(PostDom, PostDom.getImmediatePostDominator(B), VD);
      return;
    }
  }

  FreeStmtMap[freeStmt].insert(VD);
}

Stmt *TransformVE::InsertMallocFree(Stmt *S, Stmt *upS) {
  switch (S->getStmtClass()) {
  default: break;

  // statements
  case Stmt::CompoundStmtClass: {
    CompoundStmt *Node = static_cast<CompoundStmt*>(S);
    CompoundStmt::body_iterator I, E;
    for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
      *I = InsertMallocFree(*I, upS);
    }
    break;
  }
  case Stmt::SwitchStmtClass: {
    SwitchStmt *Node = static_cast<SwitchStmt*>(S);
    Node->setBody(InsertMallocFree(Node->getBody(), upS));
    break;
  }
  case Stmt::CaseStmtClass: {
    CaseStmt *Node = static_cast<CaseStmt*>(S);
    Node->setSubStmt(InsertMallocFree(Node->getSubStmt(), upS));
    break;
  }
  case Stmt::DefaultStmtClass: {
    DefaultStmt *Node = static_cast<DefaultStmt*>(S);
    Node->setSubStmt(InsertMallocFree(Node->getSubStmt(), upS));
    break;
  }
  case Stmt::LabelStmtClass: {
    LabelStmt *Node = static_cast<LabelStmt*>(S);
    Node->setSubStmt(InsertMallocFree(Node->getSubStmt(), upS));
    break;
  }
  case Stmt::IfStmtClass: {
    IfStmt *Node = static_cast<IfStmt*>(S);
    Node->setThen(InsertMallocFree(Node->getThen(), upS));
    if (Stmt *Else = Node->getElse()) {
      Node->setElse(InsertMallocFree(Else, upS));
    }
    break;
  }
  case Stmt::WhileStmtClass: {
    WhileStmt *Node = static_cast<WhileStmt*>(S);
    Node->setBody(InsertMallocFree(Node->getBody(), upS ? upS : Node));
    break;
  }
  case Stmt::DoStmtClass: {
    DoStmt *Node = static_cast<DoStmt*>(S);
    Node->setBody(InsertMallocFree(Node->getBody(), upS ? upS : Node));
    break;
  }
  case Stmt::ForStmtClass: {
    ForStmt *Node = static_cast<ForStmt*>(S);
    Node->setBody(InsertMallocFree(Node->getBody(), upS ? upS : Node));
    break;
  }
  } // end switch

  // malloc() loop
  Stmt *origS = S;
  if (MallocStmtMap.find(S) != MallocStmtMap.end()) {
    if (!upS) {
      S = MergeMallocCodeAndStmt(MallocStmtMap[S], S);;
    } else {
      // Move malloc() to upS stmt.
      if (MallocStmtMap.find(upS) != MallocStmtMap.end()) {
        VarDeclSetTy &upSdecls = MallocStmtMap[upS];
        VarDeclSetTy &Sdecls   = MallocStmtMap[S];
        for (VarDeclSetTy::iterator I = Sdecls.begin(), E = Sdecls.end();
             I != E; ++I) {
          upSdecls.insert(*I);
        }
      } else {
        MallocStmtMap[upS] = MallocStmtMap[S];
      }
    }
    MallocStmtMap.erase(origS);
  }

  // free() loop
  if (FreeStmtMap.find(origS) != FreeStmtMap.end()) {
    if (!upS) {
      S = MergeFreeCodeAndStmt(FreeStmtMap[origS], S);
    } else {
      // Make the free position FREE_AFTER.
      for (VarDeclSetTy::iterator I = FreeStmtMap[origS].begin(),
                                  E = FreeStmtMap[origS].end();
           I != E; ++I) {
        FreePosMap[*I] = FREE_AFTER;
      }
      
      // Move free() below upS stmt.
      if (FreeStmtMap.find(upS) != FreeStmtMap.end()) {
        VarDeclSetTy &upSdecls = FreeStmtMap[upS];
        VarDeclSetTy &Sdecls   = FreeStmtMap[origS];
        for (VarDeclSetTy::iterator I = Sdecls.begin(), E = Sdecls.end();
             I != E; ++I) {
          upSdecls.insert(*I);
        }
      } else {
        FreeStmtMap[upS] = FreeStmtMap[origS];
      }
    }
    FreeStmtMap.erase(origS);
  }

  return S;
}

Stmt *TransformVE::InsertRemainingMalloc(CompoundStmt *CS) {
  VarDeclSetTy decls;
  for (StmtDeclMapTy::iterator I = MallocStmtMap.begin(),
                               E = MallocStmtMap.end();
       I != E; ++I) {
    VarDeclSetTy &declSet = (*I).second;
    for (VarDeclSetTy::iterator VI = declSet.begin(), VE = declSet.end();
         VI != VE; ++VI) {
      decls.insert(*VI);
    }
  }
  MallocStmtMap.clear();

  CS->setIsStmtList(true);
  Stmt *merged = MergeMallocCodeAndStmt(decls, CS);
  CS = dyn_cast<CompoundStmt>(merged);
  CS->setIsStmtList(false);
  return CS;
}

Stmt *TransformVE::InsertRemainingFree(CompoundStmt *CS) {
  VarDeclSetTy decls;
  for (StmtDeclMapTy::iterator I = FreeStmtMap.begin(),
                               E = FreeStmtMap.end();
       I != E; ++I) {
    VarDeclSetTy &declSet = (*I).second;
    for (VarDeclSetTy::iterator VI = declSet.begin(), VE = declSet.end();
         VI != VE; ++VI) {
      decls.insert(*VI);
    }
  }
  FreeStmtMap.clear();

  CS->setIsStmtList(true);
  Stmt *merged = MergeFreeCodeAndStmt(decls, CS);
  CS = dyn_cast<CompoundStmt>(merged);
  CS->setIsStmtList(false);
  return CS;
}

Stmt *TransformVE::MergeMallocCodeAndStmt(VarDeclSetTy &declSet, Stmt *S) {
  switch (getMaxDim()) {
    case 1: return Make1DMallocCode(declSet, S);
    case 2: return Make2DMallocCode(declSet, S);
    case 3: return Make3DMallocCode(declSet, S);
    default: assert(0 && "MaxDim should be between 1 and 3.");
  }

  return NULL;
}

Stmt *TransformVE::MergeFreeCodeAndStmt(VarDeclSetTy &declSet, Stmt *S) {
  SourceLocation loc;
  
  StmtVector body;
  VarDeclSetTy beforeSet;
  VarDeclSetTy afterSet;

  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;
    if (FreePosMap[VD] == FREE_BEFORE) {
      beforeSet.insert(VD);
    } else {
      afterSet.insert(VD);
    }
  }

  // free code for beforeSet
  if (beforeSet.size() > 0) {
    MakeFreeCode(body, beforeSet);
  }

  // statement
  CompoundStmt *CS = dyn_cast<CompoundStmt>(S);
  if (CS && CS->isStmtList()) {
    // Copy the StmtList
    for (CompoundStmt::body_iterator I = CS->body_begin(), 
                                     E = CS->body_end();
         I != E; ++I) {
      body.push_back(*I);
    }
    ASTCtx.Deallocate(CS);
  } else {
    body.push_back(S);
  }

  // free code for afterSet
  if (afterSet.size() > 0) {
    MakeFreeCode(body, afterSet);
  }

  return CLExprs.NewStmtList(body);
}


Stmt *TransformVE::Make1DMallocCode(VarDeclSetTy &declSet, Stmt *S) {
  StmtVector Stmts;

  // 1st level malloc
  QualType voidPtrTy = ASTCtx.getPointerType(ASTCtx.VoidTy);
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    // make a malloc() expression.
    const PointerType *VDTy = VD->getType()->getAs<PointerType>();
    assert(VDTy && "Type of VarDecl must be a poiner type.");
    QualType T = VDTy->getPointeeType();

    // __local_size[0] * sizeof(T)
    Expr *binOp = CLExprs.NewBinaryOperator(
        CLExprs.getExpr(CLExpressions::LOCAL_SIZE_0),
        CLExprs.NewSizeOfExpr(T),
        BO_Mul, ASTCtx.UnsignedIntTy);

    // malloc(__local_size[0] * sizeof(T))
    Expr *callExpr = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::MALLOC), &binOp, 1, voidPtrTy);

    // VD = (PT)malloc(__local_size[0] * sizeof(T))
    QualType PT = VD->getType();
    Expr *LHS = CLExprs.NewDeclRefExpr(VD, PT);
    Expr *RHS = CLExprs.NewCStyleCastExpr(PT, callExpr, PT);
    Expr *mallocCode = CLExprs.NewBinaryOperator(LHS, RHS, BO_Assign, PT);
    Stmts.push_back(mallocCode);

    DeclStmt *DS = CLExprs.NewSingleDeclStmt(VD);
    MallocDecls.push_back(DS);
  }

  Stmts.push_back(S);

  return CLExprs.NewStmtList(Stmts);
}


Stmt *TransformVE::Make2DMallocCode(VarDeclSetTy &declSet, Stmt *S) {
  StmtVector Stmts;

  // 1st level malloc
  QualType voidPtrTy = ASTCtx.getPointerType(ASTCtx.VoidTy);
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    // make a malloc() expression.
    const PointerType *VDTy = VD->getType()->getAs<PointerType>();
    assert(VDTy && "Type of VarDecl must be a poiner type.");
    QualType T = VDTy->getPointeeType();

    // __local_size[1] * sizeof(T)
    Expr *binOp = CLExprs.NewBinaryOperator(
        CLExprs.getExpr(CLExpressions::LOCAL_SIZE_1),
        CLExprs.NewSizeOfExpr(T),
        BO_Mul, ASTCtx.UnsignedIntTy);

    // malloc(__local_size[1] * sizeof(T))
    Expr *callExpr = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::MALLOC), &binOp, 1, voidPtrTy);

    // VD = (PT)malloc(__local_size[1] * sizeof(T))
    QualType PT = VD->getType();
    Expr *LHS = CLExprs.NewDeclRefExpr(VD, PT);
    Expr *RHS = CLExprs.NewCStyleCastExpr(PT, callExpr, PT);
    Expr *mallocCode = CLExprs.NewBinaryOperator(LHS, RHS, BO_Assign, PT);
    Stmts.push_back(mallocCode);

    DeclStmt *DS = CLExprs.NewSingleDeclStmt(VD);
    MallocDecls.push_back(DS);
  }

  Stmts.push_back(CLExprs.getLoopJ(Make2DSecondLevelMalloc(declSet)));
  Stmts.push_back(S);

  return CLExprs.NewStmtList(Stmts);
}

CompoundStmt *TransformVE::Make2DSecondLevelMalloc(VarDeclSetTy &declSet) {
  StmtVector Stmts;

  // 2nd level malloc
  QualType voidPtrTy = ASTCtx.getPointerType(ASTCtx.VoidTy);
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    // make a malloc() expression.
    const PointerType *VDTy1 = VD->getType()->getAs<PointerType>();
    QualType type1 = VDTy1->getPointeeType(); // T*
    const PointerType *VDTy2 = type1->getAs<PointerType>();
    assert(VDTy2 && "Type of VarDecl must be a poiner type.");
    QualType type2 = VDTy2->getPointeeType(); // T

    // __local_size[0] * sizeof(type2)
    Expr *binOp = CLExprs.NewBinaryOperator(
        CLExprs.getExpr(CLExpressions::LOCAL_SIZE_0),
        CLExprs.NewSizeOfExpr(type2),
        BO_Mul, ASTCtx.UnsignedIntTy);

    // malloc(__local_size[0] * sizeof(type2))
    Expr *callExpr = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::MALLOC), &binOp, 1, voidPtrTy);

    // RHS: (type1)malloc(__local_size[0] * sizeof(type2))
    Expr *RHS = CLExprs.NewCStyleCastExpr(type1, callExpr, type1);

    // LHS: VD[__j]
    Expr *declRef = CLExprs.NewDeclRefExpr(VD, VD->getType());
    Expr *LHS = CLExprs.NewArraySubscriptExpr(
        declRef, CLExprs.getExpr(CLExpressions::WCL_IDX_J), type1);

    // VD[__j] = (type1)malloc(__local_size[1] * sizeof(type2))
    Expr *mallocCode = CLExprs.NewBinaryOperator(LHS, RHS, BO_Assign, type1);
    Stmts.push_back(mallocCode);
  }

  return CLExprs.NewCompoundStmt(Stmts);
}


Stmt *TransformVE::Make3DMallocCode(VarDeclSetTy &declSet, Stmt *S) {
  StmtVector Stmts;

  // 1st level malloc
  QualType voidPtrTy = ASTCtx.getPointerType(ASTCtx.VoidTy);
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    // make a malloc() expression.
    const PointerType *VDTy = VD->getType()->getAs<PointerType>();
    assert(VDTy && "Type of VarDecl must be a poiner type.");
    QualType type = VDTy->getPointeeType();

    // __local_size[2] * sizeof(type)
    Expr *binOp = CLExprs.NewBinaryOperator(
        CLExprs.getExpr(CLExpressions::LOCAL_SIZE_2),
        CLExprs.NewSizeOfExpr(type),
        BO_Mul, ASTCtx.UnsignedIntTy);

    // malloc(__local_size[2] * sizeof(type))
    Expr *callExpr = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::MALLOC), &binOp, 1, voidPtrTy);

    // VD = (PTy)malloc(__local_size[2] * sizeof(type))
    QualType PTy = VD->getType();
    Expr *LHS = CLExprs.NewDeclRefExpr(VD, PTy);
    Expr *RHS = CLExprs.NewCStyleCastExpr(PTy, callExpr, PTy);
    Expr *mallocCode = CLExprs.NewBinaryOperator(LHS, RHS, BO_Assign, PTy);
    Stmts.push_back(mallocCode);

    DeclStmt *DS = CLExprs.NewSingleDeclStmt(VD);
    MallocDecls.push_back(DS);
  }

  Stmts.push_back(CLExprs.getLoopK(Make3DSecondLevelMalloc(declSet)));
  Stmts.push_back(S);

  return CLExprs.NewStmtList(Stmts);
}

CompoundStmt *TransformVE::Make3DSecondLevelMalloc(VarDeclSetTy &declSet) {
  StmtVector Stmts;

  // 2nd level malloc
  QualType voidPtrTy = ASTCtx.getPointerType(ASTCtx.VoidTy);
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    // make a malloc() expression.
    const PointerType *VDTy1 = VD->getType()->getAs<PointerType>();
    QualType type1 = VDTy1->getPointeeType(); // T**
    const PointerType *VDTy2 = type1->getAs<PointerType>();
    assert(VDTy2 && "Type of VarDecl must be a poiner type.");
    QualType type2 = VDTy2->getPointeeType(); // T*

    // __local_size[1] * sizeof(type2)
    Expr *binOp = CLExprs.NewBinaryOperator(
        CLExprs.getExpr(CLExpressions::LOCAL_SIZE_1),
        CLExprs.NewSizeOfExpr(type2),
        BO_Mul, ASTCtx.UnsignedIntTy);

    // malloc(__local_size[1] * sizeof(type2))
    Expr *callExpr = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::MALLOC), &binOp, 1, voidPtrTy);

    // RHS: (type1)malloc(__local_size[1] * sizeof(type2))
    Expr *RHS = CLExprs.NewCStyleCastExpr(type1, callExpr, type1);

    // LHS: VD[__k]
    Expr *declRef = CLExprs.NewDeclRefExpr(VD, VD->getType());
    Expr *LHS = CLExprs.NewArraySubscriptExpr(
        declRef, CLExprs.getExpr(CLExpressions::WCL_IDX_K), type1);

    // VD[__k] = (type1)malloc(__local_size[1] * sizeof(type2))
    Expr *mallocCode = CLExprs.NewBinaryOperator(LHS, RHS, BO_Assign, type1);
    Stmts.push_back(mallocCode);
  }

  Stmts.push_back(CLExprs.getLoopJ(Make3DThirdLevelMalloc(declSet)));

  return CLExprs.NewCompoundStmt(Stmts);
}

CompoundStmt *TransformVE::Make3DThirdLevelMalloc(VarDeclSetTy &declSet) {
  StmtVector Stmts;

  // 3rd level malloc
  QualType voidPtrTy = ASTCtx.getPointerType(ASTCtx.VoidTy);
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    // make a malloc() expression.
    const PointerType *VDTy1 = VD->getType()->getAs<PointerType>();
    QualType type1 = VDTy1->getPointeeType(); // T**
    const PointerType *VDTy2 = type1->getAs<PointerType>();
    QualType type2 = VDTy2->getPointeeType(); // T*
    const PointerType *VDTy3 = type2->getAs<PointerType>();
    assert(VDTy3 && "Type of VarDecl must be a poiner type.");
    QualType type3 = VDTy3->getPointeeType(); // T

    // __local_size[0] * sizeof(type3)
    Expr *binOp = CLExprs.NewBinaryOperator(
        CLExprs.getExpr(CLExpressions::LOCAL_SIZE_0),
        CLExprs.NewSizeOfExpr(type3),
        BO_Mul, ASTCtx.UnsignedIntTy);

    // malloc(__local_size[0] * sizeof(type3))
    Expr *callExpr = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::MALLOC), &binOp, 1, voidPtrTy);

    // RHS: (type2)malloc(__local_size[0] * sizeof(type3))
    Expr *RHS = CLExprs.NewCStyleCastExpr(type2, callExpr, type2);

    // LHS: VD[__k][__j]
    Expr *declRef = CLExprs.NewDeclRefExpr(VD, VD->getType());
    Expr *LHS1 = CLExprs.NewArraySubscriptExpr(
        declRef, CLExprs.getExpr(CLExpressions::WCL_IDX_K), type1);
    Expr *LHS  = CLExprs.NewArraySubscriptExpr(
        LHS1, CLExprs.getExpr(CLExpressions::WCL_IDX_J), type2);

    // VD[__k][__j] = (type2)malloc(__local_size[0] * sizeof(type3))
    Expr *mallocCode = CLExprs.NewBinaryOperator(LHS, RHS, BO_Assign, type2);
    Stmts.push_back(mallocCode);
  }

  return CLExprs.NewCompoundStmt(Stmts);
}


void TransformVE::MakeFreeCode(StmtVector &Stmts, VarDeclSetTy &declSet) {
  // loop for 2nd level free
  if (getMaxDim() == 2) {
    Stmts.push_back(CLExprs.getLoopJ(Make2DSecondLevelFree(declSet)));
  } else if (getMaxDim() == 3) {
    Stmts.push_back(CLExprs.getLoopK(Make3DSecondLevelFree(declSet)));
  }

  // 1st level free
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    // free(VD)
    Expr *declRef  = CLExprs.NewDeclRefExpr(VD, VD->getType());
    Expr *freeExpr = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::FREE), &declRef, 1, ASTCtx.VoidTy);
    Stmts.push_back(freeExpr);
  }
}

CompoundStmt *TransformVE::Make2DSecondLevelFree(VarDeclSetTy &declSet) {
  StmtVector Stmts;

  // 2nd level free
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    const PointerType *VDTy1 = VD->getType()->getAs<PointerType>();
    QualType type1 = VDTy1->getPointeeType(); // T*

    // VD[__j];
    Expr *argExpr = CLExprs.NewArraySubscriptExpr(
        CLExprs.NewDeclRefExpr(VD, VD->getType()),
        CLExprs.getExpr(CLExpressions::WCL_IDX_J),
        type1);

    // free(VD[__j])
    Expr *freeCode = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::FREE), &argExpr, 1, ASTCtx.VoidTy);
    Stmts.push_back(freeCode);
  }

  return CLExprs.NewCompoundStmt(Stmts);
}

CompoundStmt *TransformVE::Make3DSecondLevelFree(VarDeclSetTy &declSet) {
  StmtVector Stmts;

  // loop for 3rd level free
  Stmts.push_back(CLExprs.getLoopJ(Make3DThirdLevelFree(declSet)));

  // 2nd level free
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    const PointerType *VDTy1 = VD->getType()->getAs<PointerType>();
    QualType type1 = VDTy1->getPointeeType(); // T**

    // VD[__k];
    Expr *argExpr = CLExprs.NewArraySubscriptExpr(
        CLExprs.NewDeclRefExpr(VD, VD->getType()),
        CLExprs.getExpr(CLExpressions::WCL_IDX_K),
        type1);

    // free(VD[__k])
    Expr *freeCode = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::FREE), &argExpr, 1, ASTCtx.VoidTy);
    Stmts.push_back(freeCode);
  }

  return CLExprs.NewCompoundStmt(Stmts);
}

CompoundStmt *TransformVE::Make3DThirdLevelFree(VarDeclSetTy &declSet) {
  StmtVector Stmts;

  // 3rd level free
  for (VarDeclSetTy::iterator I = declSet.begin(), E = declSet.end();
       I != E; ++I) {
    VarDecl *VD = *I;

    const PointerType *VDTy1 = VD->getType()->getAs<PointerType>();
    QualType type1 = VDTy1->getPointeeType(); // T**
    const PointerType *VDTy2 = type1->getAs<PointerType>();
    QualType type2 = VDTy2->getPointeeType(); // T*

    // VD[__k]
    Expr *argExpr = CLExprs.NewArraySubscriptExpr(
        CLExprs.NewDeclRefExpr(VD, VD->getType()),
        CLExprs.getExpr(CLExpressions::WCL_IDX_K),
        type1);

    // VD[__k][__j]
    argExpr = CLExprs.NewArraySubscriptExpr(
        argExpr, CLExprs.getExpr(CLExpressions::WCL_IDX_J), type2);

    // free(VD[__k][__j])
    Expr *freeCode = CLExprs.NewCallExpr(
        CLExprs.getExpr(CLExpressions::FREE), &argExpr, 1, ASTCtx.VoidTy);
    Stmts.push_back(freeCode);
  }

  return CLExprs.NewCompoundStmt(Stmts);
}

Stmt *TransformVE::MergeMallocDeclsAndBody(Stmt *S) {
  if (MallocDecls.size() == 0) return S;
  
  CompoundStmt *CS = dyn_cast<CompoundStmt>(S);
  for (CompoundStmt::body_iterator I = CS->body_begin(), E = CS->body_end();
       I != E; ++I) {
    MallocDecls.push_back(*I);
  }

  return CLExprs.NewCompoundStmt(MallocDecls);
}


//---------------------------------------------------------------------------
// Debugging functions
//---------------------------------------------------------------------------
void TransformVE::printLCA(VarDecl *VD, CFGBlockSetTy &blkSet,
                           CFGBlock *LCA, CFGBlock *newLCA) {
  OS << VD->getNameAsString() << " : LCA of { ";
  for (CFGBlockSetTy::iterator I = blkSet.begin(), E = blkSet.end();
       I != E; ++I) {
    if (I != blkSet.begin()) OS << ", ";
    OS << "B" << (*I)->getBlockID();
  }
  OS << " } => B" << LCA->getBlockID() 
     << " => B" << newLCA->getBlockID() << "\n";
}

void TransformVE::printStmtDeclMap(StmtDeclMapTy &stmtDeclMap) {
  CLStmtPrinter P(OS, ASTCtx, NULL, ASTCtx.PrintingPolicy);

  for (StmtDeclMapTy::iterator I = stmtDeclMap.begin(), E = stmtDeclMap.end();
       I != E; ++I) {
    Stmt *S = (*I).first;
    VarDeclSetTy &declSet = (*I).second;
    P.Visit(S);
    OS << " => { ";
    for (VarDeclSetTy::iterator VI = declSet.begin(), VE = declSet.end();
         VI != VE; ++VI) {
      if (VI != declSet.begin()) OS << ", ";
      OS << (*VI)->getNameAsString();
    }
    OS << " }\n";
  }
}
