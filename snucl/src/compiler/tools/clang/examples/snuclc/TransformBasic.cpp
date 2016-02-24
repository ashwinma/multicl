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
#include "TransformBasic.h"
#include "TransformVector.h"
#include "FunctionDuplicator.h"
#include "Defines.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;
using std::string;


void TransformBasic::Transform(FunctionDecl *FD) {
  if (!FD->hasBody()) return;

  // First, transform vector literals, components, and operations
  TransformVector TV(ASTCtx, CLOpts, CLExprs);
  TV.Transform(FD);

  // Main basic transformation
  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  assert(FDBody && "Function body must be a CompoundStmt");

  FDBody = TransformRawCompoundStmt(FDBody);
  FD->setBody(FDBody);
}


//===--------------------------------------------------------------------===//
//  Stmt methods.
//===--------------------------------------------------------------------===//
Stmt *TransformBasic::VisitNullStmt(NullStmt *Node) {
  return Node;
}

Stmt *TransformBasic::VisitCompoundStmt(CompoundStmt *Node) {
  return TransformRawCompoundStmt(Node);
}

Stmt *TransformBasic::VisitLabelStmt(LabelStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *TransformBasic::VisitIfStmt(IfStmt *If) {
  return TransformRawIfStmt(If);
}

Stmt *TransformBasic::VisitSwitchStmt(SwitchStmt *Node) {
  Node->setCond(TransformExpr(Node->getCond()));

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }
  return Node;
}

Stmt *TransformBasic::VisitWhileStmt(WhileStmt *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setBody(TransformStmt(Node->getBody()));
  return Node;
}

Stmt *TransformBasic::VisitDoStmt(DoStmt *Node) {
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }

  Node->setCond(TransformExpr(Node->getCond()));
  return Node;
}

Stmt *TransformBasic::VisitForStmt(ForStmt *Node) {
  if (Node->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Node->getInit()))
      Node->setInit(TransformRawDeclStmt(DS));
    else
      Node->setInit(TransformExpr(cast<Expr>(Node->getInit())));
  }
  if (Node->getCond()) {
    Node->setCond(TransformExpr(Node->getCond()));
  }
  if (Node->getInc()) {
    Node->setInc(TransformExpr(Node->getInc()));
  }

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }
  return Node;
}

Stmt *TransformBasic::VisitGotoStmt(GotoStmt *Node) {
  return Node;
}

Stmt *TransformBasic::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Node->setTarget(TransformExpr(Node->getTarget()));
  return Node;
}

Stmt *TransformBasic::VisitContinueStmt(ContinueStmt *Node) {
  return Node;
}

Stmt *TransformBasic::VisitBreakStmt(BreakStmt *Node) {
  return Node;
}

Stmt *TransformBasic::VisitReturnStmt(ReturnStmt *Node) {
  if (Node->getRetValue()) {
    Node->setRetValue(TransformExpr(Node->getRetValue()));
  }
  return Node;
}

Stmt *TransformBasic::VisitDeclStmt(DeclStmt *Node) {
  return TransformRawDeclStmt(Node);
}

Stmt *TransformBasic::VisitCaseStmt(CaseStmt *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  if (Node->getRHS()) {
    Node->setRHS(TransformExpr(Node->getRHS()));
  }

  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *TransformBasic::VisitDefaultStmt(DefaultStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}


//===--------------------------------------------------------------------===//
// Expr methods.
//===--------------------------------------------------------------------===//
Stmt *TransformBasic::VisitPredefinedExpr(PredefinedExpr *Node) {
  return Node;
}

Stmt *TransformBasic::VisitDeclRefExpr(DeclRefExpr *Node) {
  NamedDecl *ND = Node->getDecl();
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(ND)) {
    // If FD has a duplicated definition, we need a prefix.
    if (FunctionDecl *DupFD = FD->getDuplication()) {
      DeclRefExpr *NewNode = new (ASTCtx) DeclRefExpr(DupFD,
          Node->getType(), Node->getValueKind(), Node->getLocation());
      ASTCtx.Deallocate(Node);
      return NewNode;
    }
  }
  return Node;
}

Stmt *TransformBasic::VisitIntegerLiteral(IntegerLiteral *Node) {
  return Node;
}

Stmt *TransformBasic::VisitFloatingLiteral(FloatingLiteral *Node) {
  return Node;
}

Stmt *TransformBasic::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  return Node;
}

Stmt *TransformBasic::VisitStringLiteral(StringLiteral *Str) {
  return Str;
}

Stmt *TransformBasic::VisitCharacterLiteral(CharacterLiteral *Node) {
  return Node;
}

Stmt *TransformBasic::VisitParenExpr(ParenExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformBasic::VisitUnaryOperator(UnaryOperator *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformBasic::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

Stmt *TransformBasic::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *TransformBasic::VisitVecStepExpr(VecStepExpr *Node) {
  if (Node->isArgumentType()) {
    int NumElem = 0;
    QualType T = Node->getArgumentType();
    if (T->isScalarType()) {
      NumElem = 1;
    } else if (const ExtVectorType *EVT = T->getAs<ExtVectorType>()) {
      NumElem = EVT->getNumElements();
      if (NumElem == 3) NumElem = 4;
    }
    return CLExprs.NewIntegerLiteral(NumElem);
  } else {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *TransformBasic::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *TransformBasic::VisitCallExpr(CallExpr *Call) {
  Call->setCallee(TransformExpr(Call->getCallee()));

  Expr *CalleeExpr = Call->getCallee();
  if (CLUtils::IsAsyncCopyFunction(CalleeExpr)) {
    if (CLUtils::IsGlobalPointerExprForDeref(Call->getArg(0))) {
      Call = TransformCalleeName(Call, OPENCL_ASYNC_LOCAL_TO_GLOBAL);
    } else {
      Call = TransformCalleeName(Call, OPENCL_ASYNC_GLOBAL_TO_LOCAL);
    }
  } else if (CLUtils::IsAtomicFunction(CalleeExpr)) {
    if (CLUtils::IsGlobalPointerExprForDeref(Call->getArg(0))) {
      Call = TransformCalleeName(Call, OPENCL_VECTOR_DATA_GLOBAL);
    } else {
      Call = TransformCalleeName(Call, OPENCL_VECTOR_DATA_LOCAL);
    }
  }

  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Call->setArg(i, TransformExpr(Call->getArg(i)));
  }
  return Call;
}

CallExpr *TransformBasic::TransformCalleeName(CallExpr *Call, string suffix) {
  FunctionDecl *FD = Call->getDirectCallee();
  assert(FD && "Not a FunctionDecl");

  string CalleeName = FD->getNameAsString() + suffix;
  IdentifierInfo &declID = ASTCtx.Idents.get(StringRef(CalleeName));
  DeclarationName N(&declID);
  SourceLocation l;
  FD = FunctionDecl::Create(ASTCtx, ASTCtx.getTranslationUnitDecl(), 
      l, N, FD->getType(), FD->getTypeSourceInfo());

  DeclRefExpr *fn = new (ASTCtx) DeclRefExpr(FD, FD->getType(), VK_RValue, l);
  Expr **args = Call->getArgs();
  unsigned numargs = Call->getNumArgs();
  QualType t = Call->getType();
  ExprValueKind VK = Call->getValueKind();
  SourceLocation rparenloc = Call->getRParenLoc();

  ASTCtx.Deallocate(Call);
  
  return new (ASTCtx) CallExpr(ASTCtx, fn, args, numargs, t, VK, rparenloc);
}

Stmt *TransformBasic::VisitMemberExpr(MemberExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *TransformBasic::VisitBinaryOperator(BinaryOperator *Node) {
  Expr *LHS = TransformExpr(Node->getLHS());
  Expr *RHS = TransformExpr(Node->getRHS());

  // Integer division (/) and remainder (%) are converted to 
  // __CL_SAFE_INT_DIV_ZERO() macro call.
  BinaryOperator::Opcode Op = Node->getOpcode();
  if (Op == BO_Div || Op == BO_Rem) {
    QualType ETy = Node->getType();
    if (!ETy->isExtVectorType() && ETy->isIntegerType()) {
      ASTCtx.Deallocate(Node);

      Expr *args[3];
      args[0] = LHS;
      args[1] = (Op == BO_Div) ? CLExprs.getExpr(CLExpressions::DIV)
                               : CLExprs.getExpr(CLExpressions::REM);
      args[2] = RHS;
      SourceLocation loc;

      if (ETy->hasSignedIntegerRepresentation()) {
        return new (ASTCtx) CallExpr(ASTCtx, 
            CLExprs.getExpr(CLExpressions::CL_SAFE_INT_DIV_ZERO), 
            args, 3, ETy, VK_RValue, loc);
      } else {
        return new (ASTCtx) CallExpr(ASTCtx, 
            CLExprs.getExpr(CLExpressions::CL_SAFE_UINT_DIV_ZERO), 
            args, 3, ETy, VK_RValue, loc);
      }
    }
  }

  Node->setLHS(LHS);
  Node->setRHS(RHS);

  return Node;
}

Stmt *TransformBasic::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  // Integer division-assign (/=) and remainder-assign (%=) are converted to 
  // __CL_SAFE_INT_DIV_ZERO() macro call.
  BinaryOperator::Opcode Op = Node->getOpcode();
  if (Op == BO_DivAssign || Op == BO_RemAssign) {
    QualType ETy = Node->getType();
    if (!ETy->isExtVectorType() && ETy->isIntegerType()) {
      Expr *LHS = Node->getLHS();
      Expr *RHS = Node->getRHS();
      ASTCtx.Deallocate(Node);
      
      // Duplicate the LHS.
      FunctionDuplicator FDup(ASTCtx, CLExprs);
      Expr *DupLHS = FDup.TransformExpr(LHS);

      BinaryOperator::Opcode Op = BO_Add;
      switch (Node->getOpcode()) {
        case BO_DivAssign: Op = BO_Div; break;
        case BO_RemAssign: Op = BO_Rem; break;
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

      return VisitBinaryOperator(BO);
    }
  }

  Expr *LHS = TransformExpr(Node->getLHS());
  Expr *RHS = TransformExpr(Node->getRHS());
  Node->setLHS(LHS);
  Node->setRHS(RHS);

  return Node;
}

Stmt *TransformBasic::VisitConditionalOperator(ConditionalOperator *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *TransformBasic::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  QualType NodeTy = Node->getType();
  Expr *SubExpr = TransformExpr(Node->getSubExpr());

  QualType SubExprTy = SubExpr->getType();
  SubExprTy = SubExprTy.getUnqualifiedType();
  if (NodeTy == SubExprTy) return SubExpr;

  // If the type of Node is a vector type, add a vector conversion function.
  if (NodeTy->isVectorType()) {
    ASTCtx.Deallocate(Node);

    SourceLocation loc;
    Expr *args[1] = { SubExpr };
    return new (ASTCtx) CallExpr(ASTCtx, 
        CLExprs.getConvertExpr(NodeTy), args, 1, NodeTy, VK_RValue, loc);
  }

  // For implicit pointer to integer casting.
  if (SubExprTy->isPointerType() && NodeTy->isIntegerType()) {
    Expr *NewNode = CStyleCastExpr::Create(
        ASTCtx, NodeTy, VK_RValue, CK_PointerToIntegral,
        SubExpr, NULL, ASTCtx.getTrivialTypeSourceInfo(NodeTy),
        SourceLocation(), SourceLocation());
    return NewNode;
  }

  Node->setSubExpr(SubExpr);
  return Node;
}

Stmt *TransformBasic::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  QualType NodeTy = Node->getType();
  Expr *SubExpr = TransformExpr(Node->getSubExpr());

  // If the type of Node is a vector type, add a vector conversion function.
  if (NodeTy->isVectorType()) {
    ASTCtx.Deallocate(Node);

    SourceLocation loc;
    Expr *args[1] = { SubExpr };
    return new (ASTCtx) CallExpr(ASTCtx, 
        CLExprs.getConvertExpr(NodeTy), args, 1, NodeTy, VK_RValue, loc);
  }

  Node->setSubExpr(SubExpr);
  return Node;
}

Stmt *TransformBasic::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Node->setInitializer(TransformExpr(Node->getInitializer()));
  return Node;
}

Stmt *TransformBasic::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *TransformBasic::VisitInitListExpr(InitListExpr *Node) {
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (Node->getInit(i))
      Node->setInit(i, TransformExpr(Node->getInit(i)));
  }
  return Node;
}

Stmt *TransformBasic::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Node->getSubExpr(i)) {
      Node->setSubExpr(i, TransformExpr(Node->getSubExpr(i)));
    }
  }

  Node->setInit(TransformExpr(Node->getInit()));
  return Node;
}

Stmt *TransformBasic::VisitParenListExpr(ParenListExpr* Node) {
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    Node->setExpr(i, TransformExpr(Node->getExpr(i)));
  }
  return Node;
}

Stmt *TransformBasic::VisitVAArgExpr(VAArgExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

