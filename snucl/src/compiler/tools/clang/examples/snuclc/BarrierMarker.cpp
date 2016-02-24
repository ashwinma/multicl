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

#include "BarrierMarker.h"
using namespace clang;
using namespace clang::snuclc;


void BarrierMarker::Mark(FunctionDecl *FD) {
  if (FD->hasBody()) Visit(FD->getBody());
}

/// MarkRawCompoundStmt - Mark a compound stmt
bool BarrierMarker::MarkRawCompoundStmt(CompoundStmt *Node) {
  bool HasBarrier = false;
  for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
       I != E; ++I) {
    HasBarrier = Visit(*I) || HasBarrier;
  }
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::MarkRawDeclStmt(DeclStmt *S) {
  bool HasBarrier = false;
  DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
  for ( ; Begin != End; ++Begin) {
    if (VarDecl *VD = dyn_cast<VarDecl>(*Begin)) {
      if (VD->hasInit()) {
        HasBarrier = Visit(VD->getInit()) || HasBarrier;
      }
    }
  }
  S->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::MarkRawIfStmt(IfStmt *If) {
  bool HasBarrier = Visit(If->getCond());

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(If->getThen())) {
    HasBarrier = MarkRawCompoundStmt(CS) || HasBarrier;
  } else {
    HasBarrier = Visit(If->getThen()) || HasBarrier;
  }

  if (Stmt *Else = If->getElse()) {
    if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Else)) {
      HasBarrier = MarkRawCompoundStmt(CS) || HasBarrier;
    } else if (IfStmt *ElseIf = dyn_cast<IfStmt>(Else)) {
      HasBarrier = MarkRawIfStmt(ElseIf) || HasBarrier;
    } else {
      HasBarrier = Visit(If->getElse()) || HasBarrier;
    }
  }

  If->setBarrier(HasBarrier);
  return HasBarrier;
}


//===--------------------------------------------------------------------===//
//  Stmt methods.
//===--------------------------------------------------------------------===//
bool BarrierMarker::VisitNullStmt(NullStmt *Node) {
  return false;
}

bool BarrierMarker::VisitDeclStmt(DeclStmt *Node) {
  return MarkRawDeclStmt(Node);
}

bool BarrierMarker::VisitCompoundStmt(CompoundStmt *Node) {
  return MarkRawCompoundStmt(Node);
}

bool BarrierMarker::VisitIfStmt(IfStmt *If) {
  return MarkRawIfStmt(If);
}

bool BarrierMarker::VisitSwitchStmt(SwitchStmt *Node) {
  bool HasBarrier = Visit(Node->getCond());

  // Pretty print compoundstmt bodies (very common).
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    HasBarrier = MarkRawCompoundStmt(CS) || HasBarrier;
  } else {
    HasBarrier = Visit(Node->getBody()) || HasBarrier;
  }

  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitCaseStmt(CaseStmt *Node) {
  bool HasBarrier = Visit(Node->getLHS());
  if (Node->getRHS()) {
    HasBarrier = Visit(Node->getRHS()) || HasBarrier;
  }

  HasBarrier = Visit(Node->getSubStmt()) || HasBarrier;

  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitDefaultStmt(DefaultStmt *Node) {
  Node->setBarrier(Visit(Node->getSubStmt()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitWhileStmt(WhileStmt *Node) {
  bool HasBarrier = Visit(Node->getCond());
  HasBarrier = Visit(Node->getBody()) || HasBarrier;
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitDoStmt(DoStmt *Node) {
  bool HasBarrier = false;
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    HasBarrier = MarkRawCompoundStmt(CS);
  } else {
    HasBarrier = Visit(Node->getBody());
  }

  HasBarrier = Visit(Node->getCond()) || HasBarrier;

  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitForStmt(ForStmt *Node) {
  bool HasBarrier = false;
  if (Node->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getInit()))
      HasBarrier = MarkRawDeclStmt(DS);
    else
      HasBarrier = Visit(cast<Expr>(Node->getInit()));
  }
  if (Node->getCond()) {
    HasBarrier = Visit(Node->getCond()) || HasBarrier;
  }
  if (Node->getInc()) {
    HasBarrier = Visit(Node->getInc()) || HasBarrier;
  }

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    HasBarrier = MarkRawCompoundStmt(CS) || HasBarrier;
  } else {
    HasBarrier = Visit(Node->getBody()) || HasBarrier;
  }

  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitReturnStmt(ReturnStmt *Node) {
  bool HasBarrier = false;
  if (Node->getRetValue()) {
    HasBarrier = Visit(Node->getRetValue());
  }
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitLabelStmt(LabelStmt *Node) {
  Node->setBarrier(Visit(Node->getSubStmt()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitGotoStmt(GotoStmt *Node) {
  return false;
}

bool BarrierMarker::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Node->setBarrier(Visit(Node->getTarget()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitContinueStmt(ContinueStmt *Node) {
  return false;
}

bool BarrierMarker::VisitBreakStmt(BreakStmt *Node) {
  return false;
}


//===--------------------------------------------------------------------===//
//  Expr methods.
//===--------------------------------------------------------------------===//

bool BarrierMarker::VisitDeclRefExpr(DeclRefExpr *Node) {
  return false;
}

bool BarrierMarker::VisitPredefinedExpr(PredefinedExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCharacterLiteral(CharacterLiteral *Node) {
  return false;
}

bool BarrierMarker::VisitIntegerLiteral(IntegerLiteral *Node) {
  return false;
}

bool BarrierMarker::VisitFloatingLiteral(FloatingLiteral *Node) {
  return false;
}

bool BarrierMarker::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  return false;
}

bool BarrierMarker::VisitStringLiteral(StringLiteral *Str) {
  return false;
}

bool BarrierMarker::VisitParenExpr(ParenExpr *Node) {
  Node->setBarrier(Visit(Node->getSubExpr()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitUnaryOperator(UnaryOperator *Node) {
  Node->setBarrier(Visit(Node->getSubExpr()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitOffsetOfExpr(OffsetOfExpr *Node) {
  bool HasBarrier = false;
  for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
    if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
      // Array node
      HasBarrier = Visit(Node->getIndexExpr(ON.getArrayExprIndex())) 
                 || HasBarrier;
    }
  }
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType())
    Node->setBarrier(Visit(Node->getArgumentExpr()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitVecStepExpr(VecStepExpr *Node) {
  if (!Node->isArgumentType())
    Node->setBarrier(Visit(Node->getArgumentExpr()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  bool HasBarrier = Visit(Node->getLHS());
  HasBarrier = Visit(Node->getRHS()) || HasBarrier;
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

static FunctionDecl *getFunctionDecl(Expr *E) {
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    return dyn_cast<FunctionDecl>(DRE->getDecl());
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    return getFunctionDecl(PE->getSubExpr());
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    return getFunctionDecl(ICE->getSubExpr());
  }

  return NULL;
}

bool BarrierMarker::VisitCallExpr(CallExpr *Call) {
  bool HasBarrier = false;
  if (FunctionDecl *Callee = getFunctionDecl(Call->getCallee())) {
    if (Callee->getName().equals("barrier")) {
      HasBarrier = true;
    }
  } else {
    HasBarrier = Visit(Call->getCallee());
  }

  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    HasBarrier = Visit(Call->getArg(i)) || HasBarrier;
  }

  Call->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitMemberExpr(MemberExpr *Node) {
  Node->setBarrier(Visit(Node->getBase()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Node->setBarrier(Visit(Node->getBase()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Node->setBarrier(Visit(Node->getSubExpr()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Node->setBarrier(Visit(Node->getInitializer()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Node->setBarrier(Visit(Node->getSubExpr()));
  return Node->hasBarrier();
}

bool BarrierMarker::VisitBinaryOperator(BinaryOperator *Node) {
  bool HasBarrier = Visit(Node->getLHS());
  HasBarrier = Visit(Node->getRHS()) || HasBarrier;
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  bool HasBarrier = Visit(Node->getLHS());
  HasBarrier = Visit(Node->getRHS()) || HasBarrier;
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitConditionalOperator(ConditionalOperator *Node) {
  bool HasBarrier = Visit(Node->getCond());
  HasBarrier = Visit(Node->getLHS()) || HasBarrier;
  HasBarrier = Visit(Node->getRHS()) || HasBarrier;
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitInitListExpr(InitListExpr *Node) {
  bool HasBarrier = false;
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (Node->getInit(i))
      HasBarrier = Visit(Node->getInit(i)) || HasBarrier;
  }
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  bool HasBarrier = false;
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Node->getSubExpr(i)) {
      HasBarrier = Visit(Node->getSubExpr(i)) || HasBarrier;
    }
  }

  HasBarrier = Visit(Node->getInit()) || HasBarrier;
  
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitParenListExpr(ParenListExpr* Node) {
  bool HasBarrier = false;
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    HasBarrier = Visit(Node->getExpr(i)) || HasBarrier;
  }
  Node->setBarrier(HasBarrier);
  return HasBarrier;
}

bool BarrierMarker::VisitVAArgExpr(VAArgExpr *Node) {
  Node->setBarrier(Visit(Node->getSubExpr()));
  return Node->hasBarrier();
}


//===--------------------------------------------------------------------===//
// GNU extensions, Obj-C, C++ statements
//===--------------------------------------------------------------------===//
// GNU Extensions
bool BarrierMarker::VisitAsmStmt(AsmStmt *Node) {
  return false;
}


// Obj-C statements
bool BarrierMarker::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
  return false;
}

bool BarrierMarker::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
  return false;
}

bool BarrierMarker::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
  return false;
}

bool BarrierMarker::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
  return false;
}

bool BarrierMarker::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
  return false;
}

bool BarrierMarker::VisitObjCForCollectionStmt(
                                            ObjCForCollectionStmt *Node) {
  return false;
}

// C++ statements
bool BarrierMarker::VisitCXXCatchStmt(CXXCatchStmt *Node) {
  return false;
}

bool BarrierMarker::VisitCXXTryStmt(CXXTryStmt *Node) {
  return false;
}


//===--------------------------------------------------------------------===//
// GNU, Clang, Microsoft extensions, C++, Obj-C, CUDA expressions 
//===--------------------------------------------------------------------===//
bool BarrierMarker::VisitImplicitValueInitExpr(
                                          ImplicitValueInitExpr *Node) {
  return false;
}

// GNU Extensions.
bool BarrierMarker::VisitBinaryConditionalOperator(
                                          BinaryConditionalOperator *Node) {
  return false;
}

bool BarrierMarker::VisitAddrLabelExpr(AddrLabelExpr *Node) {
  return false;
}

bool BarrierMarker::VisitStmtExpr(StmtExpr *E) {
  return false;
}

bool BarrierMarker::VisitChooseExpr(ChooseExpr *Node) {
  return false;
}

bool BarrierMarker::VisitGNUNullExpr(GNUNullExpr *Node) {
  return false;
}

// Clang Extensions.
bool BarrierMarker::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
  return false;
}

bool BarrierMarker::VisitBlockExpr(BlockExpr *Node) {
  return false;
}

bool BarrierMarker::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
  return false;
}

bool BarrierMarker::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {
  return false;
}

// C++ Expressions.
bool BarrierMarker::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXReinterpretCastExpr(
                                         CXXReinterpretCastExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXFunctionalCastExpr(
                                         CXXFunctionalCastExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXNullPtrLiteralExpr(
                                         CXXNullPtrLiteralExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXThisExpr(CXXThisExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXThrowExpr(CXXThrowExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXScalarValueInitExpr(
                                         CXXScalarValueInitExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXNewExpr(CXXNewExpr *E) {
  return false;
}

bool BarrierMarker::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
  return false;
}

bool BarrierMarker::VisitCXXPseudoDestructorExpr(
                                        CXXPseudoDestructorExpr *E) {
  return false;
}

bool BarrierMarker::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
  return false;
}

bool BarrierMarker::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
  return false;
}

bool BarrierMarker::VisitDependentScopeDeclRefExpr(
                                         DependentScopeDeclRefExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXConstructExpr(CXXConstructExpr *E) {
  return false;
}

bool BarrierMarker::VisitCXXBindTemporaryExpr(
                                         CXXBindTemporaryExpr *Node) {
  return false;
}

bool BarrierMarker::VisitExprWithCleanups(ExprWithCleanups *E) {
  return false;
}

bool BarrierMarker::VisitCXXTemporaryObjectExpr(
                                         CXXTemporaryObjectExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXUnresolvedConstructExpr(
                                        CXXUnresolvedConstructExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXDependentScopeMemberExpr(
                                        CXXDependentScopeMemberExpr *Node) {
  return false;
}

bool BarrierMarker::VisitUnresolvedLookupExpr(
                                         UnresolvedLookupExpr *Node) {
  return false;
}

bool BarrierMarker::VisitUnresolvedMemberExpr(
                                        UnresolvedMemberExpr *Node) {
  return false;
}

bool BarrierMarker::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
  return false;
}

bool BarrierMarker::VisitPackExpansionExpr(PackExpansionExpr *E) {
  return false;
}

bool BarrierMarker::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
  return false;
}

bool BarrierMarker::VisitSubstNonTypeTemplateParmPackExpr(
                                   SubstNonTypeTemplateParmPackExpr *Node) {
  return false;
}

// Obj-C Expressions.
bool BarrierMarker::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
  return false;
}

bool BarrierMarker::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
  return false;
}

bool BarrierMarker::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
  return false;
}

bool BarrierMarker::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
  return false;
}

bool BarrierMarker::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
  return false;
}

bool BarrierMarker::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
  return false;
}

bool BarrierMarker::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
  return false;
}

bool BarrierMarker::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
  return false;
}

// CUDA Expressions.
bool BarrierMarker::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
  return false;
}

// Microsoft Extensions.
bool BarrierMarker::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
  return false;
}

