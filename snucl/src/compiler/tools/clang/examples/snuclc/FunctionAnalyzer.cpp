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
#include "FunctionAnalyzer.h"
#include "CLUtils.h"
using llvm::StringRef;
using namespace clang;
using namespace clang::snuclc;


//===--------------------------------------------------------------------===//
//  Stmt analyzing methods.
//===--------------------------------------------------------------------===//

/// AnalyzeRawCompoundStmt - Analyze a compound stmt
void FunctionAnalyzer::AnalyzeRawCompoundStmt(CompoundStmt *Node) {
  for (CompoundStmt::body_iterator I = Node->body_begin(), E = Node->body_end();
       I != E; ++I)
    Visit(*I);
}

void FunctionAnalyzer::AnalyzeRawDeclStmt(DeclStmt *S) {
  DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();
  for ( ; Begin != End; ++Begin) {
    if (VarDecl *VD = dyn_cast<VarDecl>(*Begin)) {
      if (VD->hasInit()) {
        Visit(VD->getInit());
      }
    }
  }
}

void FunctionAnalyzer::AnalyzeRawIfStmt(IfStmt *If) {
  Visit(If->getCond());

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(If->getThen())) {
    AnalyzeRawCompoundStmt(CS);
  } else {
    Visit(If->getThen());
  }

  if (Stmt *Else = If->getElse()) {
    if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Else)) {
      AnalyzeRawCompoundStmt(CS);
    } else if (IfStmt *ElseIf = dyn_cast<IfStmt>(Else)) {
      AnalyzeRawIfStmt(ElseIf);
    } else {
      Visit(If->getElse());
    }
  }
}


//===--------------------------------------------------------------------===//
//  Stmt methods.
//===--------------------------------------------------------------------===//
void FunctionAnalyzer::VisitNullStmt(NullStmt *Node) {
}

void FunctionAnalyzer::VisitDeclStmt(DeclStmt *Node) {
  AnalyzeRawDeclStmt(Node);
}

void FunctionAnalyzer::VisitCompoundStmt(CompoundStmt *Node) {
  AnalyzeRawCompoundStmt(Node);
}

void FunctionAnalyzer::VisitIfStmt(IfStmt *If) {
  AnalyzeRawIfStmt(If);
}

void FunctionAnalyzer::VisitSwitchStmt(SwitchStmt *Node) {
  Visit(Node->getCond());

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    AnalyzeRawCompoundStmt(CS);
  } else {
    Visit(Node->getBody());
  }
}

void FunctionAnalyzer::VisitCaseStmt(CaseStmt *Node) {
  Visit(Node->getLHS());
  if (Node->getRHS()) {
    Visit(Node->getRHS());
  }

  Visit(Node->getSubStmt());
}

void FunctionAnalyzer::VisitDefaultStmt(DefaultStmt *Node) {
  Visit(Node->getSubStmt());
}

void FunctionAnalyzer::VisitWhileStmt(WhileStmt *Node) {
  Visit(Node->getCond());
  Visit(Node->getBody());
}

void FunctionAnalyzer::VisitDoStmt(DoStmt *Node) {
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    AnalyzeRawCompoundStmt(CS);
  } else {
    Visit(Node->getBody());
  }

  Visit(Node->getCond());
}

void FunctionAnalyzer::VisitForStmt(ForStmt *Node) {
  if (Node->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getInit()))
      AnalyzeRawDeclStmt(DS);
    else
      Visit(cast<Expr>(Node->getInit()));
  }
  if (Node->getCond()) {
    Visit(Node->getCond());
  }
  if (Node->getInc()) {
    Visit(Node->getInc());
  }

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    AnalyzeRawCompoundStmt(CS);
  } else {
    Visit(Node->getBody());
  }
}

void FunctionAnalyzer::VisitReturnStmt(ReturnStmt *Node) {
  if (Node->getRetValue()) {
    Visit(Node->getRetValue());
  }
}

void FunctionAnalyzer::VisitLabelStmt(LabelStmt *Node) {
  Visit(Node->getSubStmt());
}

void FunctionAnalyzer::VisitGotoStmt(GotoStmt *Node) {
  FD->setHasGotoStmt(true);
}

void FunctionAnalyzer::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Visit(Node->getTarget());
  FD->setHasGotoStmt(true);
}

void FunctionAnalyzer::VisitContinueStmt(ContinueStmt *Node) {
}

void FunctionAnalyzer::VisitBreakStmt(BreakStmt *Node) {
}


//===--------------------------------------------------------------------===//
//  Expr analyzing methods.
//===--------------------------------------------------------------------===//

void FunctionAnalyzer::VisitDeclRefExpr(DeclRefExpr *Node) {
}

void FunctionAnalyzer::VisitPredefinedExpr(PredefinedExpr *Node) {
}

void FunctionAnalyzer::VisitCharacterLiteral(CharacterLiteral *Node) {
}

void FunctionAnalyzer::VisitIntegerLiteral(IntegerLiteral *Node) {
}

void FunctionAnalyzer::VisitFloatingLiteral(FloatingLiteral *Node) {
}

void FunctionAnalyzer::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
}

void FunctionAnalyzer::VisitStringLiteral(StringLiteral *Str) {
}

void FunctionAnalyzer::VisitParenExpr(ParenExpr *Node) {
  Visit(Node->getSubExpr());
}

void FunctionAnalyzer::VisitUnaryOperator(UnaryOperator *Node) {
  switch (Node->getOpcode()) {
    case UO_PostInc:
    case UO_PostDec:
    case UO_PreInc:
    case UO_PreDec:
    case UO_AddrOf:
      CheckParmVarDeclModified(Node->getSubExpr());
      break;
    default: break;
  }

  Visit(Node->getSubExpr());
}

void FunctionAnalyzer::VisitOffsetOfExpr(OffsetOfExpr *Node) {
  for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
    OffsetOfExpr::OffsetOfNode ON = Node->getComponent(i);
    if (ON.getKind() == OffsetOfExpr::OffsetOfNode::Array) {
      // Array node
      Visit(Node->getIndexExpr(ON.getArrayExprIndex()));
    }
  }
}

void FunctionAnalyzer::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType())
    Visit(Node->getArgumentExpr());
}

void FunctionAnalyzer::VisitVecStepExpr(VecStepExpr *Node) {
  if (!Node->isArgumentType())
    Visit(Node->getArgumentExpr());
}

void FunctionAnalyzer::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Visit(Node->getLHS());
  Visit(Node->getRHS());
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

static IntegerLiteral *getIntegerLiteral(Expr *E) {
  if (IntegerLiteral *IL = dyn_cast<IntegerLiteral>(E)) {
    return IL;
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    return getIntegerLiteral(PE->getSubExpr());
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    return getIntegerLiteral(ICE->getSubExpr());
  }

  return NULL;
}

void FunctionAnalyzer::VisitCallExpr(CallExpr *Call) {
  if (FunctionDecl *Callee = getFunctionDecl(Call->getCallee())) {
    FunctionDecl *Definition;
    if (Callee->isDefined(Definition)) {
      CalleeSet.insert(Definition);
    } else {
      CalleeSet.insert(Callee);
    }

    StringRef CalleeName = Callee->getName();
    if (CalleeName.equals("barrier")) {
      FD->setBarrierCall(true);
    } else if (CLUtils::IsWorkItemFunction(CalleeName)) {
      // Check if the first argument is IntegerLiteral.
      if (IntegerLiteral *IL = getIntegerLiteral(Call->getArg(0))) {
        unsigned Val = (unsigned)IL->getValue().getLimitedValue() + 1;
        if (Val >= 1 && Val <= 3) {
          if (CallInfo.getMaxDim() < Val) CallInfo.setMaxDim(Val);
        }
      }
    }
  } else {
    Visit(Call->getCallee());
  }

  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Visit(Call->getArg(i));
  }
}

void FunctionAnalyzer::VisitMemberExpr(MemberExpr *Node) {
  Visit(Node->getBase());
}

void FunctionAnalyzer::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Visit(Node->getBase());
}

void FunctionAnalyzer::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Visit(Node->getSubExpr());
}

void FunctionAnalyzer::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Visit(Node->getInitializer());
}

void FunctionAnalyzer::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Visit(Node->getSubExpr());
}

void FunctionAnalyzer::VisitBinaryOperator(BinaryOperator *Node) {
  // Check if the function parameter is modified.
  if (Node->getOpcode() == BO_Assign) {
    CheckParmVarDeclModified(Node->getLHS());
  }

  Visit(Node->getLHS());
  Visit(Node->getRHS());
}

void FunctionAnalyzer::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  CheckParmVarDeclModified(Node->getLHS());

  Visit(Node->getLHS());
  Visit(Node->getRHS());
}

void FunctionAnalyzer::VisitConditionalOperator(ConditionalOperator *Node) {
  Visit(Node->getCond());
  Visit(Node->getLHS());
  Visit(Node->getRHS());
}

void FunctionAnalyzer::VisitInitListExpr(InitListExpr *Node) {
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (Node->getInit(i))
      Visit(Node->getInit(i));
  }
}

void FunctionAnalyzer::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Node->getSubExpr(i)) {
      Visit(Node->getSubExpr(i));
    }
  }

  Visit(Node->getInit());
}

void FunctionAnalyzer::VisitParenListExpr(ParenListExpr* Node) {
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    Visit(Node->getExpr(i));
  }
}

void FunctionAnalyzer::VisitVAArgExpr(VAArgExpr *Node) {
  Visit(Node->getSubExpr());
}


//===--------------------------------------------------------------------===//
// GNU extensions, Obj-C, C++ statements
//===--------------------------------------------------------------------===//
// GNU Extensions
void FunctionAnalyzer::VisitAsmStmt(AsmStmt *Node) {
}


// Obj-C statements
void FunctionAnalyzer::VisitObjCAtTryStmt(ObjCAtTryStmt *Node) {
}

void FunctionAnalyzer::VisitObjCAtCatchStmt (ObjCAtCatchStmt *Node) {
}

void FunctionAnalyzer::VisitObjCAtFinallyStmt(ObjCAtFinallyStmt *Node) {
}

void FunctionAnalyzer::VisitObjCAtThrowStmt(ObjCAtThrowStmt *Node) {
}

void FunctionAnalyzer::VisitObjCAtSynchronizedStmt(ObjCAtSynchronizedStmt *Node) {
}

void FunctionAnalyzer::VisitObjCForCollectionStmt(
                                            ObjCForCollectionStmt *Node) {
}

// C++ statements
void FunctionAnalyzer::VisitCXXCatchStmt(CXXCatchStmt *Node) {
}

void FunctionAnalyzer::VisitCXXTryStmt(CXXTryStmt *Node) {
}


//===--------------------------------------------------------------------===//
// GNU, Clang, Microsoft extensions, C++, Obj-C, CUDA expressions 
//===--------------------------------------------------------------------===//
void FunctionAnalyzer::VisitImplicitValueInitExpr(
                                          ImplicitValueInitExpr *Node) {
}

// GNU Extensions.
void FunctionAnalyzer::VisitBinaryConditionalOperator(
                                          BinaryConditionalOperator *Node) {
}

void FunctionAnalyzer::VisitAddrLabelExpr(AddrLabelExpr *Node) {
}

void FunctionAnalyzer::VisitStmtExpr(StmtExpr *E) {
}

void FunctionAnalyzer::VisitChooseExpr(ChooseExpr *Node) {
}

void FunctionAnalyzer::VisitGNUNullExpr(GNUNullExpr *Node) {
}

// Clang Extensions.
void FunctionAnalyzer::VisitShuffleVectorExpr(ShuffleVectorExpr *Node) {
}

void FunctionAnalyzer::VisitBlockExpr(BlockExpr *Node) {
}

void FunctionAnalyzer::VisitBlockDeclRefExpr(BlockDeclRefExpr *Node) {
}

void FunctionAnalyzer::VisitOpaqueValueExpr(OpaqueValueExpr *Node) {
}

// C++ Expressions.
void FunctionAnalyzer::VisitCXXOperatorCallExpr(CXXOperatorCallExpr *Node) {
}

void FunctionAnalyzer::VisitCXXMemberCallExpr(CXXMemberCallExpr *Node) {
}

void FunctionAnalyzer::VisitCXXStaticCastExpr(CXXStaticCastExpr *Node) {
}

void FunctionAnalyzer::VisitCXXDynamicCastExpr(CXXDynamicCastExpr *Node) {
}

void FunctionAnalyzer::VisitCXXReinterpretCastExpr(
                                         CXXReinterpretCastExpr *Node) {
}

void FunctionAnalyzer::VisitCXXConstCastExpr(CXXConstCastExpr *Node) {
}

void FunctionAnalyzer::VisitCXXFunctionalCastExpr(
                                         CXXFunctionalCastExpr *Node) {
}

void FunctionAnalyzer::VisitCXXTypeidExpr(CXXTypeidExpr *Node) {
}

void FunctionAnalyzer::VisitCXXBoolLiteralExpr(CXXBoolLiteralExpr *Node) {
}

void FunctionAnalyzer::VisitCXXNullPtrLiteralExpr(
                                         CXXNullPtrLiteralExpr *Node) {
}

void FunctionAnalyzer::VisitCXXThisExpr(CXXThisExpr *Node) {
}

void FunctionAnalyzer::VisitCXXThrowExpr(CXXThrowExpr *Node) {
}

void FunctionAnalyzer::VisitCXXDefaultArgExpr(CXXDefaultArgExpr *Node) {
}

void FunctionAnalyzer::VisitCXXScalarValueInitExpr(
                                         CXXScalarValueInitExpr *Node) {
}

void FunctionAnalyzer::VisitCXXNewExpr(CXXNewExpr *E) {
}

void FunctionAnalyzer::VisitCXXDeleteExpr(CXXDeleteExpr *E) {
}

void FunctionAnalyzer::VisitCXXPseudoDestructorExpr(
                                        CXXPseudoDestructorExpr *E) {
}

void FunctionAnalyzer::VisitUnaryTypeTraitExpr(UnaryTypeTraitExpr *E) {
}

void FunctionAnalyzer::VisitBinaryTypeTraitExpr(BinaryTypeTraitExpr *E) {
}

void FunctionAnalyzer::VisitDependentScopeDeclRefExpr(
                                         DependentScopeDeclRefExpr *Node) {
}

void FunctionAnalyzer::VisitCXXConstructExpr(CXXConstructExpr *E) {
}

void FunctionAnalyzer::VisitCXXBindTemporaryExpr(
                                         CXXBindTemporaryExpr *Node) {
}

void FunctionAnalyzer::VisitExprWithCleanups(ExprWithCleanups *E) {
}

void FunctionAnalyzer::VisitCXXTemporaryObjectExpr(
                                         CXXTemporaryObjectExpr *Node) {
}

void FunctionAnalyzer::VisitCXXUnresolvedConstructExpr(
                                        CXXUnresolvedConstructExpr *Node) {
}

void FunctionAnalyzer::VisitCXXDependentScopeMemberExpr(
                                        CXXDependentScopeMemberExpr *Node) {
}

void FunctionAnalyzer::VisitUnresolvedLookupExpr(
                                         UnresolvedLookupExpr *Node) {
}

void FunctionAnalyzer::VisitUnresolvedMemberExpr(
                                        UnresolvedMemberExpr *Node) {
}

void FunctionAnalyzer::VisitCXXNoexceptExpr(CXXNoexceptExpr *E) {
}

void FunctionAnalyzer::VisitPackExpansionExpr(PackExpansionExpr *E) {
}

void FunctionAnalyzer::VisitSizeOfPackExpr(SizeOfPackExpr *E) {
}

void FunctionAnalyzer::VisitSubstNonTypeTemplateParmPackExpr(
                                   SubstNonTypeTemplateParmPackExpr *Node) {
}

// Obj-C Expressions.
void FunctionAnalyzer::VisitObjCStringLiteral(ObjCStringLiteral *Node) {
}

void FunctionAnalyzer::VisitObjCEncodeExpr(ObjCEncodeExpr *Node) {
}

void FunctionAnalyzer::VisitObjCMessageExpr(ObjCMessageExpr *Mess) {
}

void FunctionAnalyzer::VisitObjCSelectorExpr(ObjCSelectorExpr *Node) {
}

void FunctionAnalyzer::VisitObjCProtocolExpr(ObjCProtocolExpr *Node) {
}

void FunctionAnalyzer::VisitObjCIvarRefExpr(ObjCIvarRefExpr *Node) {
}

void FunctionAnalyzer::VisitObjCPropertyRefExpr(ObjCPropertyRefExpr *Node) {
}

void FunctionAnalyzer::VisitObjCIsaExpr(ObjCIsaExpr *Node) {
}

// CUDA Expressions.
void FunctionAnalyzer::VisitCUDAKernelCallExpr(CUDAKernelCallExpr *Node) {
}

// Microsoft Extensions.
void FunctionAnalyzer::VisitCXXUuidofExpr(CXXUuidofExpr *Node) {
}


void FunctionAnalyzer::CheckParmVarDeclModified(Expr *E) {
  E = E->IgnoreParenCasts();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (ParmVarDecl *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl())) {
      PVD->setModified(true);
    }
  }
}

