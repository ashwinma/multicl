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

#include "FunctionDuplicator.h"
using namespace clang;
using namespace clang::snuclc;


void FunctionDuplicator::Duplicate(FunctionDecl *FD) {
  FunctionDecl *DupFD = FD->getDuplication();
  assert(DupFD && "FD does not have its duplication");

  // Duplicate parameters.
  if (FD->getNumParams() > 0) {
    llvm::SmallVector<ParmVarDecl *, 4> NewParamInfo;
    for (FunctionDecl::param_iterator I = FD->param_begin(),
         E = FD->param_end(); I != E; ++I) {
      ParmVarDecl *PVD = *I;

      Expr *DefArg = PVD->getDefaultArg();
      if (DefArg) DefArg = TransformExpr(DefArg);

      ParmVarDecl *DupPVD = ParmVarDecl::Create(ASTCtx, DeclCtx,
          PVD->getLocation(), PVD->getIdentifier(),
          PVD->getType(), PVD->getTypeSourceInfo(),
          PVD->getStorageClass(), PVD->getStorageClassAsWritten(),
          DefArg);
      if (PVD->hasAttrs()) DupPVD->setAttrs(PVD->getAttrs());
      NewParamInfo.push_back(DupPVD);

      // Make the mapping between two ParmVarDecls.
      ParmVarDeclMap[PVD] = DupPVD;
    }

    DupFD->setParams(NewParamInfo.data(), NewParamInfo.size());
  }

  // Duplicate body.
  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  assert(FDBody && "Function body must be a CompoundStmt");
  DupFD->setBody(VisitCompoundStmt(FDBody));
}

void FunctionDuplicator::SetParmVarDeclMap(ParmVarDeclMapTy &PDMap) {
  ParmVarDeclMap.clear();
  ParmVarDeclMap.insert(PDMap.begin(), PDMap.end());
}


CompoundStmt *FunctionDuplicator::TransformRawCompoundStmt(CompoundStmt *Node) {
  StmtVector StmtVec;

  CompoundStmt::body_iterator I, E;
  for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
    StmtVec.push_back(TransformStmt(*I));
  }

  return new (ASTCtx) CompoundStmt(ASTCtx,
      StmtVec.data(), StmtVec.size(), SourceLocation(), SourceLocation());
}

Stmt *FunctionDuplicator::TransformRawDeclStmt(DeclStmt *S) {
  DeclVector DeclVec;

  DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
  for ( ; Begin != End; ++Begin) {
    if (VarDecl *VD = dyn_cast<VarDecl>(*Begin)) {
      DeclVec.push_back(DuplicateVarDecl(VD));
    } else {
      // FIXME: we need to handle other decls.
      DeclVec.push_back(*Begin);
    }
  }

  DeclGroupRef DGRef = DeclGroupRef::Create(ASTCtx,
                                            DeclVec.data(), DeclVec.size());
  return new (ASTCtx) DeclStmt(DGRef, SourceLocation(), SourceLocation());
}

Stmt *FunctionDuplicator::TransformRawIfStmt(IfStmt *If) {
  Expr *Cond = TransformExpr(If->getCond());
  Stmt *Then = TransformStmt(If->getThen());
  VarDecl *Var = DuplicateVarDecl(If->getConditionVariable());

  IfStmt *DupIf = new (ASTCtx) IfStmt(ASTCtx, SourceLocation(),
      Var, Cond, Then);

  if (Stmt *Else = If->getElse()) {
    if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Else)) {
      DupIf->setElse(TransformRawCompoundStmt(CS));
    } else if (IfStmt *ElseIf = dyn_cast<IfStmt>(Else)) {
      DupIf->setElse(TransformRawIfStmt(ElseIf));
    } else {
      DupIf->setElse(TransformStmt(Else));
    }
  }

  return DupIf;
}


//===--------------------------------------------------------------------===//
//  Stmt methods.
//===--------------------------------------------------------------------===//
Stmt *FunctionDuplicator::VisitNullStmt(NullStmt *Node) {
  return new (ASTCtx) NullStmt(SourceLocation());
}

Stmt *FunctionDuplicator::VisitCompoundStmt(CompoundStmt *Node) {
  return TransformRawCompoundStmt(Node);
}

Stmt *FunctionDuplicator::VisitLabelStmt(LabelStmt *Node) {
  LabelDecl *LD = DuplicateLabelDecl(Node->getDecl());
  Stmt *SubStmt = TransformStmt(Node->getSubStmt());

  return new (ASTCtx) LabelStmt(SourceLocation(), LD, SubStmt);
}

Stmt *FunctionDuplicator::VisitIfStmt(IfStmt *If) {
  return TransformRawIfStmt(If);
}

Stmt *FunctionDuplicator::VisitSwitchStmt(SwitchStmt *Node) {
  VarDecl *Var = DuplicateVarDecl(Node->getConditionVariable());
  Expr *Cond = Node->getCond() ? TransformExpr(Node->getCond()) : NULL;

  SwitchStmt *DupSS = new (ASTCtx) SwitchStmt(ASTCtx, Var, Cond);
  if (Stmt *Body = Node->getBody()) {
    DupSS->setBody(TransformStmt(Body));
  }
  return DupSS;
}

Stmt *FunctionDuplicator::VisitWhileStmt(WhileStmt *Node) {
  VarDecl *Var = DuplicateVarDecl(Node->getConditionVariable());
  Expr *Cond = Node->getCond();
  Stmt *Body = Node->getBody();
  if (Cond) Cond = TransformExpr(Cond);
  if (Body) Body = TransformStmt(Body);

  return new (ASTCtx) WhileStmt(ASTCtx, Var, Cond, Body, SourceLocation());
}

Stmt *FunctionDuplicator::VisitDoStmt(DoStmt *Node) {
  Stmt *Body = Node->getBody();
  Expr *Cond = Node->getCond();
  if (Body) Body = TransformStmt(Body);
  if (Cond) Cond = TransformExpr(Cond);

  SourceLocation SL;
  return new (ASTCtx) DoStmt(Body, Cond, SL, SL, SL); 
}

Stmt *FunctionDuplicator::VisitForStmt(ForStmt *Node) {
  VarDecl *CondVar = DuplicateVarDecl(Node->getConditionVariable());
  Stmt *Init = Node->getInit();
  Expr *Cond = Node->getCond();
  Expr *Inc  = Node->getInc();
  Stmt *Body = Node->getBody();
  if (Init) Init = TransformStmt(Init);
  if (Cond) Cond = TransformExpr(Cond);
  if (Inc)  Inc  = TransformExpr(Inc);
  if (Body) Body = TransformStmt(Body);

  SourceLocation SL;
  return new (ASTCtx) ForStmt(ASTCtx, Init, Cond, CondVar, Inc, Body,
                              SL, SL, SL);
}

Stmt *FunctionDuplicator::VisitGotoStmt(GotoStmt *Node) {
  LabelDecl *LD = DuplicateLabelDecl(Node->getLabel());
  return new (ASTCtx) GotoStmt(LD, SourceLocation(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Expr *Target = Node->getTarget();
  if (Target) Target = TransformExpr(Target);
  SourceLocation SL;
  return new (ASTCtx) IndirectGotoStmt(SL, SL, Target);
}

Stmt *FunctionDuplicator::VisitContinueStmt(ContinueStmt *Node) {
  return new (ASTCtx) ContinueStmt(SourceLocation());
}

Stmt *FunctionDuplicator::VisitBreakStmt(BreakStmt *Node) {
  return new (ASTCtx) BreakStmt(SourceLocation());
}

Stmt *FunctionDuplicator::VisitReturnStmt(ReturnStmt *Node) {
  if (InFunctionInliner) {
    // When FunctionDuplicator is used in FunctionInliner, ReturnStmt needs
    // to be transformed.
    if (Expr *Ret = Node->getRetValue()) {
      assert(RetVD && "RetVD is not specified!");
      Expr *LHS = new (ASTCtx) DeclRefExpr(RetVD, RetVD->getType(),
                                           VK_RValue, SourceLocation());
      Expr *RHS = TransformExpr(Ret);
      return new (ASTCtx) BinaryOperator(LHS, RHS, BO_Assign,
          RetVD->getType(), VK_RValue, OK_Ordinary, SourceLocation());
    } else {
      return new (ASTCtx) NullStmt(SourceLocation());
    }
  } else {
    // Normal ReturnStmt.
    ReturnStmt *DupRS = new (ASTCtx) ReturnStmt(SourceLocation());
    if (Expr *Ret = Node->getRetValue()) {
      DupRS->setRetValue(TransformExpr(Ret));
    }
    return DupRS;
  }
}

Stmt *FunctionDuplicator::VisitDeclStmt(DeclStmt *Node) {
  return TransformRawDeclStmt(Node);
}

Stmt *FunctionDuplicator::VisitCaseStmt(CaseStmt *Node) {
  Expr *LHS = Node->getLHS();
  Expr *RHS = Node->getRHS();
  if (LHS) LHS = TransformExpr(LHS);
  if (RHS) RHS = TransformExpr(RHS);

  SourceLocation SL;
  CaseStmt *DupCS = new (ASTCtx) CaseStmt(LHS, RHS, SL, SL, SL);
  if (Stmt *SubStmt = Node->getSubStmt()) {
    DupCS->setSubStmt(TransformStmt(SubStmt));
  }
  return DupCS;
}

Stmt *FunctionDuplicator::VisitDefaultStmt(DefaultStmt *Node) {
  Stmt *SubStmt = Node->getSubStmt();
  if (SubStmt) SubStmt = TransformStmt(SubStmt);

  SourceLocation SL;
  return new (ASTCtx) DefaultStmt(SL, SL, SubStmt);
}


//===--------------------------------------------------------------------===//
// Expr methods.
//===--------------------------------------------------------------------===//
Stmt *FunctionDuplicator::VisitPredefinedExpr(PredefinedExpr *Node) {
  return new (ASTCtx) PredefinedExpr(SourceLocation(),
      Node->getType(), Node->getIdentType());
}

Stmt *FunctionDuplicator::VisitDeclRefExpr(DeclRefExpr *Node) {
  ValueDecl *D = Node->getDecl();

  // If D has a duplicated declaration, it is used for the new DeclRefExpr.
  // Otherwise, the original D is used.
  if (VarDecl *VD = dyn_cast<VarDecl>(D)) {
    if (VarDeclMap.find(VD) != VarDeclMap.end()) {
      // VD is declared inside the fundtion and was duplicated.
      D = VarDeclMap[VD];
    } else if (ParmVarDecl *PD = dyn_cast<ParmVarDecl>(D)) {
      if (ParmVarDeclMap.find(PD) != ParmVarDeclMap.end()) {
        D = ParmVarDeclMap[PD];
      }
    }
  }
  
  return new (ASTCtx) DeclRefExpr(D, Node->getType(),
      Node->getValueKind(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitIntegerLiteral(IntegerLiteral *Node) {
  return new (ASTCtx) IntegerLiteral(ASTCtx, Node->getValue(),
      Node->getType(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitFloatingLiteral(FloatingLiteral *Node) {
  return FloatingLiteral::Create(ASTCtx, Node->getValue(), Node->isExact(),
      Node->getType(), SourceLocation(), Node->getValueAsString());
}

Stmt *FunctionDuplicator::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  Expr *Val = Node->getSubExpr();
  if (Val) Val = TransformExpr(Val);
  return new (ASTCtx) ImaginaryLiteral(Val, Node->getType());
}

Stmt *FunctionDuplicator::VisitStringLiteral(StringLiteral *Str) {
  const char *StrData = Str->getString().data();
  return StringLiteral::Create(ASTCtx, StrData, Str->getByteLength(),
      Str->isWide(), Str->getType(), SourceLocation()); 
}

Stmt *FunctionDuplicator::VisitCharacterLiteral(CharacterLiteral *Node) {
  return new (ASTCtx) CharacterLiteral(Node->getValue(), Node->isWide(),
      Node->getType(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitParenExpr(ParenExpr *Node) {
  Expr *Val = Node->getSubExpr();
  if (Val) Val = TransformExpr(Val);
  return new (ASTCtx) ParenExpr(SourceLocation(), SourceLocation(), Val);
}

Stmt *FunctionDuplicator::VisitUnaryOperator(UnaryOperator *Node) {
  Expr *Input = TransformExpr(Node->getSubExpr());
  return new (ASTCtx) UnaryOperator(Input, Node->getOpcode(), Node->getType(),
      Node->getValueKind(), Node->getObjectKind(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitOffsetOfExpr(OffsetOfExpr *Node) {
  typedef OffsetOfExpr::OffsetOfNode OffsetOfNode;
  llvm::SmallVector<OffsetOfNode, 4> Comps;
  llvm::SmallVector<Expr*, 4> Exprs;

  for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
    if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
      // Array node
      unsigned Idx = ON.getArrayExprIndex();
      Exprs.push_back(TransformExpr(Node->getIndexExpr(Idx)));
    }
    Comps.push_back(ON);
  }

  return OffsetOfExpr::Create(ASTCtx, Node->getType(),
      SourceLocation(), Node->getTypeSourceInfo(),
      Comps.data(), Comps.size(),
      Exprs.data(), Exprs.size(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (Node->isArgumentType()) {
    return new (ASTCtx) SizeOfAlignOfExpr(Node->isSizeOf(), 
        Node->getArgumentTypeInfo(), Node->getType(), 
        SourceLocation(), SourceLocation());
  } else {
    Expr *E = TransformExpr(Node->getArgumentExpr());
    return new (ASTCtx) SizeOfAlignOfExpr(Node->isSizeOf(),
        E, Node->getType(),
        SourceLocation(), SourceLocation());
  }
}

Stmt *FunctionDuplicator::VisitVecStepExpr(VecStepExpr *Node) {
  if (Node->isArgumentType()) {
    return new (ASTCtx) VecStepExpr(
        Node->getArgumentTypeInfo(), Node->getType(), 
        SourceLocation(), SourceLocation());
  } else {
    Expr *E = TransformExpr(Node->getArgumentExpr());
    return new (ASTCtx) VecStepExpr(
        E, Node->getType(), SourceLocation(), SourceLocation());
  }
}

Stmt *FunctionDuplicator::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Expr *LHS = TransformExpr(Node->getLHS());
  Expr *RHS = TransformExpr(Node->getRHS());
  return new (ASTCtx) ArraySubscriptExpr(LHS, RHS, Node->getType(),
      Node->getValueKind(), Node->getObjectKind(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitCallExpr(CallExpr *Call) {
  llvm::SmallVector<Expr*, 4> Exprs;

  Expr *Fn = TransformExpr(Call->getCallee());
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Exprs.push_back(TransformExpr(Call->getArg(i)));
  }

  return new (ASTCtx) CallExpr(ASTCtx, Fn, Exprs.data(), Exprs.size(),
      Call->getType(), Call->getValueKind(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitMemberExpr(MemberExpr *Node) {
  Expr *Base = TransformExpr(Node->getBase());
  return new (ASTCtx) MemberExpr(Base, Node->isArrow(),
      Node->getMemberDecl(), SourceLocation(), Node->getType(),
      Node->getValueKind(), Node->getObjectKind());
}

Stmt *FunctionDuplicator::VisitBinaryOperator(BinaryOperator *Node) {
  Expr *LHS = TransformExpr(Node->getLHS());
  Expr *RHS = TransformExpr(Node->getRHS());
  return new (ASTCtx) BinaryOperator(LHS, RHS, Node->getOpcode(),
      Node->getType(), Node->getValueKind(), Node->getObjectKind(),
      SourceLocation());
}

Stmt *FunctionDuplicator::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  Expr *LHS = TransformExpr(Node->getLHS());
  Expr *RHS = TransformExpr(Node->getRHS());
  return new (ASTCtx) CompoundAssignOperator(LHS, RHS, Node->getOpcode(),
      Node->getType(), Node->getValueKind(), Node->getObjectKind(),
      Node->getComputationLHSType(), Node->getComputationResultType(),
      SourceLocation());
}

Stmt *FunctionDuplicator::VisitConditionalOperator(ConditionalOperator *Node) {
  Expr *Cond = TransformExpr(Node->getCond());
  Expr *LHS  = TransformExpr(Node->getLHS());
  Expr *RHS  = TransformExpr(Node->getRHS());
  return new (ASTCtx) ConditionalOperator(Cond, SourceLocation(), LHS,
      SourceLocation(), RHS, Node->getType(),
      Node->getValueKind(), Node->getObjectKind());
}

Stmt *FunctionDuplicator::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Expr *Op = TransformExpr(Node->getSubExpr());
  return new (ASTCtx) ImplicitCastExpr(ImplicitCastExpr::OnStack,
      Node->getType(), Node->getCastKind(), Op, Node->getValueKind());
}

Stmt *FunctionDuplicator::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Expr *Op = TransformExpr(Node->getSubExpr());
  return CStyleCastExpr::Create(ASTCtx, Node->getType(),
      Node->getValueKind(), Node->getCastKind(), Op, NULL,
      Node->getTypeInfoAsWritten(), SourceLocation(),
      SourceLocation());
}

Stmt *FunctionDuplicator::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Expr *Init = Node->getInitializer();
  if (Init) Init = TransformExpr(Init);
  return new (ASTCtx) CompoundLiteralExpr(SourceLocation(),
      Node->getTypeSourceInfo(), Node->getType(), Node->getValueKind(),
      Init, Node->isFileScope());
}

Stmt *FunctionDuplicator::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Expr *Base = TransformExpr(Node->getBase());
  return new (ASTCtx) ExtVectorElementExpr(Node->getType(),
      Node->getValueKind(), Base, Node->getAccessor(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitInitListExpr(InitListExpr *Node) {
  llvm::SmallVector<Expr*, 4> Exprs;
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (Expr *Init = Node->getInit(i))
      Exprs.push_back(TransformExpr(Init));
    else
      Exprs.push_back(NULL);
  }
  InitListExpr *NewNode = new (ASTCtx) InitListExpr(
      ASTCtx, SourceLocation(), Exprs.data(), Exprs.size(), SourceLocation());
  NewNode->setType(Node->getType());
  return NewNode;
}

Stmt *FunctionDuplicator::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  llvm::SmallVector<Expr*, 4> Exprs;
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Expr *SubExpr = Node->getSubExpr(i)) {
      Exprs.push_back(TransformExpr(SubExpr));
    } else {
      Exprs.push_back(NULL);
    }
  }
  Expr *Init = TransformExpr(Node->getInit());

  return DesignatedInitExpr::Create(ASTCtx, Node->designators_begin(),
      Node->size(), Exprs.data(), Exprs.size(),
      SourceLocation(), Node->usesGNUSyntax(), Init);
}

Stmt *FunctionDuplicator::VisitParenListExpr(ParenListExpr* Node) {
  llvm::SmallVector<Expr*, 4> Exprs;
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    Exprs.push_back(TransformExpr(Node->getExpr(i)));
  }
  return new (ASTCtx) ParenListExpr(ASTCtx, SourceLocation(),
      Exprs.data(), Exprs.size(), SourceLocation());
}

Stmt *FunctionDuplicator::VisitVAArgExpr(VAArgExpr *Node) {
  Expr *E = TransformExpr(Node->getSubExpr());
  return new (ASTCtx) VAArgExpr(SourceLocation(), E,
      Node->getWrittenTypeInfo(), SourceLocation(), Node->getType());
}

VarDecl *FunctionDuplicator::DuplicateVarDecl(VarDecl *VD) {
  if (VD == NULL) return NULL;

  VarDecl *DupVD = VarDecl::Create(ASTCtx, DeclCtx, SourceLocation(),
      VD->getIdentifier(), VD->getType(), VD->getTypeSourceInfo(),
      VD->getStorageClass(), VD->getStorageClassAsWritten());
  if (VD->hasAttrs()) DupVD->setAttrs(VD->getAttrs());

  if (VD->hasInit()) {
    DupVD->setInit(TransformExpr(VD->getInit()));
  }

  // Make the mapping between two VarDecls.
  assert(VarDeclMap.find(VD) == VarDeclMap.end() && 
         "Multiple declaration of VarDecl");
  VarDeclMap[VD] = DupVD;

  return DupVD;
}

/// LabelDecl can be used before it is appeared in the source code.
/// For example, 
///   goto L;
///   L: ...
LabelDecl *FunctionDuplicator::DuplicateLabelDecl(LabelDecl *LD) {
  if (LabelDeclMap.find(LD) == LabelDeclMap.end()) {
    LabelDecl *DupLD = LabelDecl::Create(ASTCtx, DeclCtx, SourceLocation(),
                                         LD->getIdentifier());

    // Make the mapping between two LabelDecls.
    LabelDeclMap[LD] = DupLD;
  }
  return LabelDeclMap[LD];
}

