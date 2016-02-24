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

#include "StmtTransformer.h"
using namespace clang;
using namespace clang::snuclc;


CompoundStmt *StmtTransformer::TransformRawCompoundStmt(CompoundStmt *Node) {
  CompoundStmt::body_iterator I, E;
  for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
    *I = TransformStmt(*I);
  }
  return Node;
}

Stmt *StmtTransformer::TransformRawDeclStmt(DeclStmt *S) {
  DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
  for ( ; Begin != End; ++Begin) {
    if (VarDecl *VD = dyn_cast<VarDecl>(*Begin)) {
      if (VD->hasInit()) {
        VD->setInit(TransformExpr(VD->getInit()));
      }
    }
  }
  return S;
}

Stmt *StmtTransformer::TransformRawIfStmt(IfStmt *If) {
  If->setCond(TransformExpr(If->getCond()));

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(If->getThen())) {
    If->setThen(TransformRawCompoundStmt(CS));
  } else {
    If->setThen(TransformStmt(If->getThen()));
  }

  if (Stmt *Else = If->getElse()) {
    if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Else)) {
      If->setElse(TransformRawCompoundStmt(CS));
    } else if (IfStmt *ElseIf = dyn_cast<IfStmt>(Else)) {
      If->setElse(TransformRawIfStmt(ElseIf));
    } else {
      If->setElse(TransformStmt(If->getElse()));
    }
  }

  return If;
}


//===--------------------------------------------------------------------===//
//  Stmt methods.
//===--------------------------------------------------------------------===//
Stmt *StmtTransformer::VisitNullStmt(NullStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCompoundStmt(CompoundStmt *Node) {
  return TransformRawCompoundStmt(Node);
}

Stmt *StmtTransformer::VisitLabelStmt(LabelStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *StmtTransformer::VisitIfStmt(IfStmt *If) {
  return TransformRawIfStmt(If);
}

Stmt *StmtTransformer::VisitSwitchStmt(SwitchStmt *Node) {
  Node->setCond(TransformExpr(Node->getCond()));

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }
  return Node;
}

Stmt *StmtTransformer::VisitWhileStmt(WhileStmt *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setBody(TransformStmt(Node->getBody()));
  return Node;
}

Stmt *StmtTransformer::VisitDoStmt(DoStmt *Node) {
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }

  Node->setCond(TransformExpr(Node->getCond()));
  return Node;
}

Stmt *StmtTransformer::VisitForStmt(ForStmt *Node) {
  if (Stmt *Init = Node->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Init))
      Node->setInit(TransformRawDeclStmt(DS));
    else
      Node->setInit(TransformExpr(cast<Expr>(Init)));
  }
  if (Expr *Cond = Node->getCond()) {
    Node->setCond(TransformExpr(Cond));
  }
  if (Expr *Inc = Node->getInc()) {
    Node->setInc(TransformExpr(Inc));
  }

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }
  return Node;
}

Stmt *StmtTransformer::VisitGotoStmt(GotoStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Node->setTarget(TransformExpr(Node->getTarget()));
  return Node;
}

Stmt *StmtTransformer::VisitContinueStmt(ContinueStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitBreakStmt(BreakStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitReturnStmt(ReturnStmt *Node) {
  if (Expr *Ret = Node->getRetValue()) {
    Node->setRetValue(TransformExpr(Ret));
  }
  return Node;
}

Stmt *StmtTransformer::VisitDeclStmt(DeclStmt *Node) {
  return TransformRawDeclStmt(Node);
}

Stmt *StmtTransformer::VisitCaseStmt(CaseStmt *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  if (Expr *RHS = Node->getRHS()) {
    Node->setRHS(TransformExpr(RHS));
  }

  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *StmtTransformer::VisitDefaultStmt(DefaultStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}


//===--------------------------------------------------------------------===//
// Expr methods.
//===--------------------------------------------------------------------===//
Stmt *StmtTransformer::VisitPredefinedExpr(PredefinedExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitDeclRefExpr(DeclRefExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitIntegerLiteral(IntegerLiteral *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitFloatingLiteral(FloatingLiteral *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitStringLiteral(StringLiteral *Str) {
  return Str;
}

Stmt *StmtTransformer::VisitCharacterLiteral(CharacterLiteral *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitParenExpr(ParenExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *StmtTransformer::VisitUnaryOperator(UnaryOperator *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *StmtTransformer::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

Stmt *StmtTransformer::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *StmtTransformer::VisitVecStepExpr(VecStepExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *StmtTransformer::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *StmtTransformer::VisitCallExpr(CallExpr *Call) {
  Call->setCallee(TransformExpr(Call->getCallee()));
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Call->setArg(i, TransformExpr(Call->getArg(i)));
  }
  return Call;
}

Stmt *StmtTransformer::VisitMemberExpr(MemberExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *StmtTransformer::VisitBinaryOperator(BinaryOperator *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *StmtTransformer::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *StmtTransformer::VisitConditionalOperator(ConditionalOperator *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *StmtTransformer::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *StmtTransformer::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *StmtTransformer::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Node->setInitializer(TransformExpr(Node->getInitializer()));
  return Node;
}

Stmt *StmtTransformer::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *StmtTransformer::VisitInitListExpr(InitListExpr *Node) {
  for (unsigned i = 0, e = Node->getNumInits(); i < e; i++) {
    if (Expr *Init = Node->getInit(i))
      Node->setInit(i, TransformExpr(Init));
  }
  return Node;
}

Stmt *StmtTransformer::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; i++) {
    if (Expr *SubExpr = Node->getSubExpr(i)) {
      Node->setSubExpr(i, TransformExpr(SubExpr));
    }
  }

  Node->setInit(TransformExpr(Node->getInit()));
  return Node;
}

Stmt *StmtTransformer::VisitParenListExpr(ParenListExpr* Node) {
  for (unsigned i = 0, e = Node->getNumExprs(); i < e; i++) {
    Node->setExpr(i, TransformExpr(Node->getExpr(i)));
  }
  return Node;
}

Stmt *StmtTransformer::VisitVAArgExpr(VAArgExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}


//===--------------------------------------------------------------------===//
// GNU extensions, Obj-C, C++ statements
//===--------------------------------------------------------------------===//
// GNU Extensions
Stmt *StmtTransformer::VisitAsmStmt(AsmStmt *Node) {
  return Node;
}

// Obj-C statements
Stmt *StmtTransformer::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCForCollectionStmt(
                                            ObjCForCollectionStmt *Node) {
  return Node;
}

// C++ statements
Stmt *StmtTransformer::VisitCXXCatchStmt(CXXCatchStmt *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXTryStmt(CXXTryStmt *Node) {
  return Node;
}

//===--------------------------------------------------------------------===//
// GNU, Clang, Microsoft extensions, C++, Obj-C, CUDA expressions 
//===--------------------------------------------------------------------===//
Stmt *StmtTransformer::VisitImplicitValueInitExpr(
                                          ImplicitValueInitExpr *Node) {
  return Node;
}

// GNU Extensions.
Stmt *StmtTransformer::VisitBinaryConditionalOperator(
                                          BinaryConditionalOperator *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitStmtExpr(StmtExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitChooseExpr(ChooseExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitGNUNullExpr(GNUNullExpr *Node) {
  return Node;
}

// Clang Extensions.
Stmt *StmtTransformer::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitBlockExpr(BlockExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {
  return Node;
}

// C++ Expressions.
Stmt *StmtTransformer::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXReinterpretCastExpr(
                                         CXXReinterpretCastExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXFunctionalCastExpr(
                                         CXXFunctionalCastExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXNullPtrLiteralExpr(
                                         CXXNullPtrLiteralExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXThisExpr(CXXThisExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXThrowExpr(CXXThrowExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXScalarValueInitExpr(
                                         CXXScalarValueInitExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXNewExpr(CXXNewExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitCXXPseudoDestructorExpr(
                                        CXXPseudoDestructorExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitDependentScopeDeclRefExpr(
                                         DependentScopeDeclRefExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXConstructExpr(CXXConstructExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitCXXBindTemporaryExpr(
                                         CXXBindTemporaryExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitExprWithCleanups(ExprWithCleanups *E) {
  return E;
}

Stmt *StmtTransformer::VisitCXXTemporaryObjectExpr(
                                         CXXTemporaryObjectExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXUnresolvedConstructExpr(
                                        CXXUnresolvedConstructExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXDependentScopeMemberExpr(
                                        CXXDependentScopeMemberExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitUnresolvedLookupExpr(
                                         UnresolvedLookupExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitUnresolvedMemberExpr(
                                        UnresolvedMemberExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitPackExpansionExpr(PackExpansionExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  return E;
}

Stmt *StmtTransformer::VisitSubstNonTypeTemplateParmPackExpr(
                                   SubstNonTypeTemplateParmPackExpr *Node) {
  return Node;
}

// Obj-C Expressions.
Stmt *StmtTransformer::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
  return Mess;
}

Stmt *StmtTransformer::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  return Node;
}

Stmt *StmtTransformer::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
  return Node;
}

// CUDA Expressions.
Stmt *StmtTransformer::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
  return Node;
}

// Microsoft Extensions.
Stmt *StmtTransformer::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
  return Node;
}

