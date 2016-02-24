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

#include "StmtSimplifier.h"
#include "FunctionDuplicator.h"
#include "CLUtils.h"
using namespace clang;
using namespace clang::snuclc;


void StmtSimplifier::Transform(FunctionDecl *FD) {
  CompoundStmt *Body = dyn_cast<CompoundStmt>(FD->getBody());
  assert(Body && "Function body must be a CompoundStmt");

  FD->setBody(VisitCompoundStmt(Body));
}


CompoundStmt *StmtSimplifier::TransformRawCompoundStmt(CompoundStmt *Node) {
  StmtVector *PrvBody = CurBody;
  StmtVector Stmts;

  CurBody = &Stmts;

  CompoundStmt::body_iterator I, E;
  for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
    if (Expr *Ex = dyn_cast<Expr>(*I)) {
      Ex = Ex->IgnoreParenImpCasts();
      if (UnaryOperator *UO = dyn_cast<UnaryOperator>(Ex)) {
        Stmts.push_back(VisitTopLevelUnaryOperator(UO));
      } else {
        Stmts.push_back(TransformExpr(Ex));
      }

      // Check if there were PostInc/PostDec UnaryOperator.
      if (PostExprVec.size() > 0) {
        for (unsigned i = 0; i < PostExprVec.size(); i++) {
          Stmts.push_back(PostExprVec[i]);
        }
        PostExprVec.clear();
      }
    } else {
      Stmts.push_back(TransformStmt(*I));
    }
  }
  Node->setStmts(ASTCtx, Stmts.data(), Stmts.size());

  CurBody = PrvBody;

  return Node;
}


//===--------------------------------------------------------------------===//
//  Stmt methods.
//===--------------------------------------------------------------------===//
Stmt *StmtSimplifier::VisitNullStmt(NullStmt *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitCompoundStmt(CompoundStmt *Node) {
  return TransformRawCompoundStmt(Node);
}

Stmt *StmtSimplifier::VisitLabelStmt(LabelStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *StmtSimplifier::VisitIfStmt(IfStmt *If) {
  If->setCond(VisitExprInStmt(If->getCond()));

  CompoundStmt *CS = dyn_cast<CompoundStmt>(If->getThen());
  if (!CS) CS = ConvertToCompoundStmt(If->getThen());
  If->setThen(TransformRawCompoundStmt(CS));

  if (Stmt *Else = If->getElse()) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Else);
    if (!CS) CS = ConvertToCompoundStmt(Else);
    If->setElse(TransformRawCompoundStmt(CS));
  }

  return If;
}

Stmt *StmtSimplifier::VisitSwitchStmt(SwitchStmt *Node) {
  Node->setCond(VisitExprInStmt(Node->getCond()));

  CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody());
  if (!CS) CS = ConvertToCompoundStmt(Node->getBody());
  Node->setBody(TransformRawCompoundStmt(CS));

  return Node;
}

Stmt *StmtSimplifier::VisitWhileStmt(WhileStmt *Node) {
  Node->setCond(VisitExprInStmt(Node->getCond()));
  if (Stmt *Body = Node->getBody()) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
    if (!CS) CS = ConvertToCompoundStmt(Body);
    Node->setBody(TransformRawCompoundStmt(CS));
  }
  return Node;
}

Stmt *StmtSimplifier::VisitDoStmt(DoStmt *Node) {
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody());
  if (!CS) CS = ConvertToCompoundStmt(Node->getBody());
  Node->setBody(TransformRawCompoundStmt(CS));

  Node->setCond(VisitExprInStmt(Node->getCond()));
  return Node;
}

Stmt *StmtSimplifier::VisitForStmt(ForStmt *Node) {
  StmtVector Stmts;

  if (Node->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getInit())) {
      StmtVector *PrvBody = CurBody;
      CurBody = &Stmts;

      // DeclGroup is divided into single Decls.
      Stmts.push_back(VisitDeclStmt(DS));
      Node->setInit(NULL);

      CurBody = PrvBody;
    } else {
      Node->setInit(VisitExprInStmt(cast<Expr>(Node->getInit())));
    }
  }
  if (Node->getCond()) {
    Node->setCond(VisitExprInStmt(Node->getCond()));
  }
  if (Node->getInc()) {
    Node->setInc(VisitExprInStmt(Node->getInc()));
  }

  Stmt *Body = Node->getBody();
  if (!Body) Body = new (ASTCtx) NullStmt(SourceLocation());
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS) CS = ConvertToCompoundStmt(Body);
  Node->setBody(TransformRawCompoundStmt(CS));

  // We convert ForStmt into CompoundStmt if its Init is a DeclStmt.
  if (Stmts.size() > 0) {
    Stmts.push_back(Node);
    return new (ASTCtx) CompoundStmt(ASTCtx, Stmts.data(), Stmts.size(),
        SourceLocation(), SourceLocation());
  }

  return Node;
}

Stmt *StmtSimplifier::VisitGotoStmt(GotoStmt *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Node->setTarget(VisitExprInStmt(Node->getTarget()));
  return Node;
}

Stmt *StmtSimplifier::VisitContinueStmt(ContinueStmt *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitBreakStmt(BreakStmt *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitReturnStmt(ReturnStmt *Node) {
  if (Node->getRetValue()) {
    Node->setRetValue(VisitExprInStmt(Node->getRetValue()));
  }
  return Node;
}

Stmt *StmtSimplifier::VisitDeclStmt(DeclStmt *Node) {
  StmtVector Decls;
  StmtVector Defs;

  SourceLocation SL;
  DeclStmt::decl_iterator I = Node->decl_begin(), E = Node->decl_end();
  for ( ; I != E; ++I) {
    Decl *SD = *I;

    // DeclStmt without Init expr.
    DeclGroupRef DG = DeclGroupRef::Create(ASTCtx, &SD, 1);
    DeclStmt *DS = new (ASTCtx) DeclStmt(DG, SL, SL);
    Decls.push_back(DS);

    // Make the definition expression.
    if (VarDecl *VD = dyn_cast<VarDecl>(SD)) {
      if (Expr *Init = VD->getInit()) {
        VD->setInit(NULL);

        if (VD->getType()->isConstantArrayType()) {
          Expr *RHS = VisitExprInStmt(Init);
          MakeArrayInitExprs(VD, RHS, Defs);
        } else {
          DeclRefExpr *LHS = new (ASTCtx) DeclRefExpr(VD, VD->getType(),
              VK_RValue, SourceLocation());
          Expr *RHS = VisitExprInStmt(Init);
          Expr *Def = new (ASTCtx) BinaryOperator(LHS, RHS, BO_Assign,
              VD->getType(), VK_RValue, OK_Ordinary, SourceLocation());

          Defs.push_back(Def);
        }
      }
    }
  }

  // Insert all Decls and Defs except the last one into CurBody.
  if (Defs.size() == 0) {
    for (unsigned i = 0, e = Decls.size(); i < e; i++) {
      if (i + 1 < e)
        CurBody->push_back(Decls[i]);
      else
        return Decls[i];
    }
  } else {
    // We have separated Decls and Defs.
    for (unsigned i = 0, e = Decls.size(); i < e; i++) {
      CurBody->push_back(Decls[i]);
    }

    for (unsigned i = 0, e = Defs.size(); i < e; i++) {
      if (i + 1 < e)
        CurBody->push_back(Defs[i]);
      else
        return Defs[i];
    }
  }

  return Node;
}

Stmt *StmtSimplifier::VisitCaseStmt(CaseStmt *Node) {
  Node->setLHS(VisitExprInStmt(Node->getLHS()));
  if (Node->getRHS()) {
    Node->setRHS(VisitExprInStmt(Node->getRHS()));
  }

  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *StmtSimplifier::VisitDefaultStmt(DefaultStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}


//===--------------------------------------------------------------------===//
// Expr methods.
//===--------------------------------------------------------------------===//
Stmt *StmtSimplifier::VisitPredefinedExpr(PredefinedExpr *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitDeclRefExpr(DeclRefExpr *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitIntegerLiteral(IntegerLiteral *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitFloatingLiteral(FloatingLiteral *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitStringLiteral(StringLiteral *Str) {
  return Str;
}

Stmt *StmtSimplifier::VisitCharacterLiteral(CharacterLiteral *Node) {
  return Node;
}

Stmt *StmtSimplifier::VisitParenExpr(ParenExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *StmtSimplifier::VisitUnaryOperator(UnaryOperator *Node) {
  Expr *Ex = VisitTopLevelUnaryOperator(Node);

  UnaryOperator::Opcode Op = Node->getOpcode();
  if (Op == UO_PreInc || Op == UO_PreDec) {
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Ex);

    // Since this needs to be executed before the current stmt, it should be
    // inserted into CurBody.
    CurBody->push_back(BO);

    // Duplicate the LHS.
    FunctionDuplicator FDup(ASTCtx, CLExprs);
    return FDup.TransformExpr(BO->getLHS());
  } else if (Op == UO_PostInc || Op == UO_PostDec) {
    BinaryOperator *BO = dyn_cast<BinaryOperator>(Ex);

    // This should be executed after the current stmt.
    PostExprVec.push_back(BO);

    // Duplicate the LHS.
    FunctionDuplicator FDup(ASTCtx, CLExprs);
    return FDup.TransformExpr(BO->getLHS());
  }

  return Ex;
}

Stmt *StmtSimplifier::VisitOffsetOfExpr(OffsetOfExpr *Node) {
  for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
    if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
      // Array node
      unsigned Idx = ON.getArrayExprIndex();
      Node->setIndexExpr(Idx, TransformExpr(Node->getIndexExpr(Idx)));
    }
  }
  return Node;
}

Stmt *StmtSimplifier::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *StmtSimplifier::VisitVecStepExpr(VecStepExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *StmtSimplifier::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *StmtSimplifier::VisitCallExpr(CallExpr *Call) {
  Call->setCallee(TransformExpr(Call->getCallee()));
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Call->setArg(i, TransformExpr(Call->getArg(i)));
  }
  return Call;
}

Stmt *StmtSimplifier::VisitMemberExpr(MemberExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *StmtSimplifier::VisitBinaryOperator(BinaryOperator *Node) {
  if (Node->getOpcode() == BO_Comma) {
    // Divide the comma expression into two expressions.
    CurBody->push_back(TransformExpr(Node->getLHS()));
    Expr *RHS = TransformExpr(Node->getRHS());
    ASTCtx.Deallocate(Node);
    return RHS;
  } else if (Node->isAssignmentOp()) {
    Expr *RHS = VisitExprInAssignExpr(Node->getRHS());
    Expr *LHS = TransformExpr(Node->getLHS());
    Node->setRHS(RHS);
    Node->setLHS(LHS);
    return Node;
  }

  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

/// Transform CompoundAssignOperator to BinaryOperator.
Stmt *StmtSimplifier::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  Expr *RHS = VisitExprInAssignExpr(Node->getRHS());
  Expr *LHS = TransformExpr(Node->getLHS());

  // Duplicate the LHS.
  FunctionDuplicator FDup(ASTCtx, CLExprs);
  Expr *DupLHS = FDup.TransformExpr(LHS);

  BinaryOperator::Opcode Op = BO_Add;
  switch (Node->getOpcode()) {
    case BO_MulAssign: Op = BO_Mul; break;
    case BO_DivAssign: Op = BO_Div; break;
    case BO_RemAssign: Op = BO_Rem; break;
    case BO_AddAssign: Op = BO_Add; break;
    case BO_SubAssign: Op = BO_Sub; break;
    case BO_ShlAssign: Op = BO_Shl; break;
    case BO_ShrAssign: Op = BO_Shr; break;
    case BO_AndAssign: Op = BO_And; break;
    case BO_XorAssign: Op = BO_Xor; break;
    case BO_OrAssign:  Op = BO_Or;  break;
    default: assert(0 && "What kind of operator?");
  }

  // If RHS is a BinaryOperator, it should be converted into ParenExpr.
  SourceLocation SL;
  RHS = CLUtils::IgnoreImpCasts(RHS);
  if (isa<BinaryOperator>(RHS)) {
    RHS = new (ASTCtx) ParenExpr(SL, SL, RHS);
  }

  BinaryOperator *NewRHS = new (ASTCtx) BinaryOperator(DupLHS, RHS, Op,
      Node->getType(), Node->getValueKind(), Node->getObjectKind(), SL);
  BinaryOperator *BO = new (ASTCtx) BinaryOperator(LHS, NewRHS, BO_Assign,
      Node->getType(), Node->getValueKind(), Node->getObjectKind(), SL);

  return BO;
}

Stmt *StmtSimplifier::VisitConditionalOperator(ConditionalOperator *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *StmtSimplifier::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *StmtSimplifier::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *StmtSimplifier::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Node->setInitializer(TransformExpr(Node->getInitializer()));
  return Node;
}

Stmt *StmtSimplifier::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *StmtSimplifier::VisitInitListExpr(InitListExpr *Node) {
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (Node->getInit(i))
      Node->setInit(i, TransformExpr(Node->getInit(i)));
  }
  return Node;
}

Stmt *StmtSimplifier::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Node->getSubExpr(i)) {
      Node->setSubExpr(i, TransformExpr(Node->getSubExpr(i)));
    }
  }

  Node->setInit(TransformExpr(Node->getInit()));
  return Node;
}

Stmt *StmtSimplifier::VisitParenListExpr(ParenListExpr* Node) {
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    Node->setExpr(i, TransformExpr(Node->getExpr(i)));
  }
  return Node;
}

Stmt *StmtSimplifier::VisitVAArgExpr(VAArgExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}


//----------------------------------------------------------------------------
Expr *StmtSimplifier::VisitExprInStmt(Expr *E) {
  switch (E->getStmtClass()) {
    case Stmt::ParenExprClass: {
      ParenExpr *Node = static_cast<ParenExpr *>(E);
      Node->setSubExpr(VisitExprInStmt(Node->getSubExpr()));
      return Node;
    }
    case Stmt::UnaryOperatorClass: {
      UnaryOperator *Node = static_cast<UnaryOperator *>(E);
      Node->setSubExpr(VisitExprInStmt(Node->getSubExpr()));
      return Node;
    }
    case Stmt::OffsetOfExprClass: {
      OffsetOfExpr *Node = static_cast<OffsetOfExpr *>(E);
      for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
        OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
        if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
          // Array node
          unsigned Idx = ON.getArrayExprIndex();
          Node->setIndexExpr(Idx, VisitExprInStmt(Node->getIndexExpr(Idx)));
        }
      }
      return Node;
    }
    case Stmt::SizeOfAlignOfExprClass: {
      SizeOfAlignOfExpr *Node = static_cast<SizeOfAlignOfExpr *>(E);
      if (!Node->isArgumentType()) {
        Node->setArgument(VisitExprInStmt(Node->getArgumentExpr()));
      }
      return Node;
    }
    case Stmt::VecStepExprClass: {
      VecStepExpr *Node = static_cast<VecStepExpr *>(E);
      if (!Node->isArgumentType()) {
        Node->setArgument(VisitExprInStmt(Node->getArgumentExpr()));
      }
      return Node;
    }
    case Stmt::ArraySubscriptExprClass: {
      ArraySubscriptExpr *Node = static_cast<ArraySubscriptExpr *>(E);
      Node->setLHS(VisitExprInStmt(Node->getLHS()));
      Node->setRHS(VisitExprInStmt(Node->getRHS()));
      return Node;
    }
    case Stmt::CallExprClass: {
      CallExpr *Call = static_cast<CallExpr *>(E);
      Call->setCallee(VisitExprInStmt(Call->getCallee()));
      for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
        Call->setArg(i, VisitExprInStmt(Call->getArg(i)));
      }
      return Call;
    }
    case Stmt::MemberExprClass: {
      MemberExpr *Node = static_cast<MemberExpr *>(E);
      Node->setBase(VisitExprInStmt(Node->getBase()));
      return Node;
    }
    case Stmt::BinaryOperatorClass: {
      BinaryOperator *Node = static_cast<BinaryOperator *>(E);
      Node->setLHS(VisitExprInStmt(Node->getLHS()));
      Node->setRHS(VisitExprInStmt(Node->getRHS()));
      return Node;
    }
    case Stmt::CompoundAssignOperatorClass: {
      CompoundAssignOperator *Node = static_cast<CompoundAssignOperator *>(E);
      Node->setLHS(VisitExprInStmt(Node->getLHS()));
      Node->setRHS(VisitExprInStmt(Node->getRHS()));
      return Node;
    }
    case Stmt::ConditionalOperatorClass: {
      ConditionalOperator *Node = static_cast<ConditionalOperator *>(E);
      Node->setCond(VisitExprInStmt(Node->getCond()));
      Node->setLHS(VisitExprInStmt(Node->getLHS()));
      Node->setRHS(VisitExprInStmt(Node->getRHS()));
      return Node;
    }
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr *Node = static_cast<ImplicitCastExpr *>(E);
      Node->setSubExpr(VisitExprInStmt(Node->getSubExpr()));
      return Node;
    }
    case Stmt::CStyleCastExprClass: {
      CStyleCastExpr *Node = static_cast<CStyleCastExpr *>(E);
      Node->setSubExpr(VisitExprInStmt(Node->getSubExpr()));
      return Node;
    }
    case Stmt::CompoundLiteralExprClass: {
      CompoundLiteralExpr *Node = static_cast<CompoundLiteralExpr *>(E);
      Node->setInitializer(VisitExprInStmt(Node->getInitializer()));
      return Node;
    }
    case Stmt::ExtVectorElementExprClass: {
      ExtVectorElementExpr *Node = static_cast<ExtVectorElementExpr *>(E);
      Node->setBase(VisitExprInStmt(Node->getBase()));
      return Node;
    }
    case Stmt::InitListExprClass: {
      InitListExpr *Node = static_cast<InitListExpr *>(E);
      for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
        if (Node->getInit(i))
          Node->setInit(i, VisitExprInStmt(Node->getInit(i)));
      }
      return Node;
    }
    case Stmt::DesignatedInitExprClass: {
      DesignatedInitExpr *Node = static_cast<DesignatedInitExpr *>(E);
      for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
        if (Node->getSubExpr(i)) {
          Node->setSubExpr(i, VisitExprInStmt(Node->getSubExpr(i)));
        }
      }

      Node->setInit(VisitExprInStmt(Node->getInit()));
      return Node;
    }
    case Stmt::ParenListExprClass: {
      ParenListExpr *Node = static_cast<ParenListExpr *>(E);
      for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
        Node->setExpr(i, VisitExprInStmt(Node->getExpr(i)));
      }
      return Node;
    }
    case Stmt::VAArgExprClass: {
      VAArgExpr *Node = static_cast<VAArgExpr *>(E);
      Node->setSubExpr(VisitExprInStmt(Node->getSubExpr()));
      return Node;
    }

    default: break;
  }

  return E;
}
//----------------------------------------------------------------------------

Expr *StmtSimplifier::VisitExprInAssignExpr(Expr *E) {
  Expr *Node = E->IgnoreParenCasts();
  if (CompoundAssignOperator *CAO = dyn_cast<CompoundAssignOperator>(Node)) {
    BinaryOperator *BO = dyn_cast<BinaryOperator>(TransformExpr(CAO));
    assert(BO && "Not BinaryOperator!");

    CurBody->push_back(BO);

    // Duplicate the LHS.
    FunctionDuplicator FDup(ASTCtx, CLExprs);
    return FDup.TransformExpr(BO->getLHS());
  }
  else if (BinaryOperator *BO = dyn_cast<BinaryOperator>(Node)) {
    if (BO->getOpcode() == BO_Assign) {
      Expr *RHS = VisitExprInAssignExpr(BO->getRHS());
      Expr *LHS = TransformExpr(BO->getLHS());
      BO->setRHS(RHS);
      BO->setLHS(LHS);

      CurBody->push_back(BO);

      // Duplicate the LHS.
      FunctionDuplicator FDup(ASTCtx, CLExprs);
      return FDup.TransformExpr(LHS);
    }
  }

  return TransformExpr(E);
}

Expr *StmtSimplifier::VisitTopLevelUnaryOperator(UnaryOperator *Node) {
  BinaryOperator::Opcode BinOp = BO_Add;

  switch (Node->getOpcode()) {
    case UO_PostInc: BinOp = BO_Add; break;
    case UO_PostDec: BinOp = BO_Sub; break;
    case UO_PreInc:  BinOp = BO_Add; break;
    case UO_PreDec:  BinOp = BO_Sub; break;
    default:
      Node->setSubExpr(TransformExpr(Node->getSubExpr()));
      return Node;
  }

  // We convert the UnaryOperator of PostInc/PostDec/PreInc/PreDec into 
  // the BinaryOperator.
  Expr *SubExpr = TransformExpr(Node->getSubExpr());

  // Duplicate the SubExpr.
  FunctionDuplicator FDup(ASTCtx, CLExprs);
  Expr *DupSubExpr = FDup.TransformExpr(SubExpr);

  SourceLocation SL;
  BinaryOperator *RHS = new (ASTCtx) BinaryOperator(
      DupSubExpr, CLExprs.getExpr(CLExpressions::ONE), BinOp,
      Node->getType(), Node->getValueKind(), Node->getObjectKind(), SL);
  BinaryOperator *BO = new (ASTCtx) BinaryOperator(SubExpr, RHS, BO_Assign,
      Node->getType(), Node->getValueKind(), Node->getObjectKind(), SL);
  
  ASTCtx.Deallocate(Node);

  return BO;
}

void StmtSimplifier::MakeArrayInitExprs(VarDecl *VD, Expr *Init,
                                        StmtVector &Defs) {
  QualType VDTy = VD->getType();
  const ArrayType *ArrayTy = VDTy->getAsArrayTypeUnsafe();
  const ConstantArrayType *CATy = dyn_cast<ConstantArrayType>(ArrayTy);
  assert(CATy && "Wrong ConstantArrayType");

  Init = CLUtils::IgnoreImpCasts(Init);
  InitListExpr *ILE = dyn_cast<InitListExpr>(Init);
  assert(ILE && "Only InitListExpr is supported.");

  unsigned ArraySize = (unsigned)CATy->getSize().getLimitedValue();
  unsigned InitSize = ILE->getNumInits();
  InitSize = ArraySize < InitSize ? ArraySize : InitSize;
  for (unsigned i = 0; i < InitSize; i++) {
    if (Expr *Init = ILE->getInit(i)) {
      QualType T = Init->getType();
      Expr *LHS = CLExprs.NewArraySubscriptExpr(
          CLExprs.NewDeclRefExpr(VD, VD->getType()),
          CLExprs.NewIntegerLiteral(i),
          T);
      Expr *NewInit = CLExprs.NewBinaryOperator(LHS, Init, BO_Assign, T);
      Defs.push_back(NewInit);
    }
  }
}


CompoundStmt *StmtSimplifier::ConvertToCompoundStmt(Stmt *S) {
  SourceLocation SL;
  Stmt *Stmts[1] = { S };
  return new (ASTCtx) CompoundStmt(ASTCtx, Stmts, 1, SL, SL);
}

