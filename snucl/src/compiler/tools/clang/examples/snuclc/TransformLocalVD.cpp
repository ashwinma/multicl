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

#include "TransformLocalVD.h"
#include "TypeDefs.h"
using namespace clang;
using namespace clang::snuclc;


void TransformLocalVD::Transform(FunctionDecl *FD) {
  if (!FD->hasBody()) return;

  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  assert(FDBody && "Function body must be a CompoundStmt");

  FDBody = TransformRawCompoundStmt(FDBody);

  if (!CLOpts.UseTLBLocal) {
    StmtVector Stmts;
    for (VarDeclSetTy::iterator I = LocalVDSet.begin(), E = LocalVDSet.end();
         I != E; ++I) {
      DeclStmt *S = CLExprs.NewSingleDeclStmt(*I);
      Stmts.push_back(S);
    }
    LocalVDSet.clear();

    if (Stmts.size() > 0) {
      CompoundStmt::body_iterator I, E;
      for (I = FDBody->body_begin(), E = FDBody->body_end(); I != E; ++I) {
        Stmts.push_back(*I);
      }
      FDBody->setStmts(ASTCtx, Stmts.data(), Stmts.size());
    }
  }

  FD->setBody(FDBody);
}


CompoundStmt *TransformLocalVD::TransformRawCompoundStmt(CompoundStmt *Node) {
  StmtVector Stmts;

  CompoundStmt::body_iterator I, E;
  for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
    if (Stmt *S = TransformStmt(*I)) Stmts.push_back(S);
  }
  Node->setStmts(ASTCtx, Stmts.data(), Stmts.size());

  return Node;
}

Stmt *TransformLocalVD::TransformRawDeclStmt(DeclStmt *S) {
  for (DeclStmt::decl_iterator I = S->decl_begin(), E = S->decl_end();
       I != E; ++I) {    
    if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
      if (Expr *Init = VD->getInit()) {
        VD->setInit(TransformExpr(Init));
      }

      // private memory size
      PrivateMemSize += CLUtils::GetTypeSize(VD->getType(), PointerSize);
    }
  }

  DeclStmt::decl_iterator Begin = S->decl_begin(), End = S->decl_end();

  if (VarDecl *VD = dyn_cast<VarDecl>(*Begin)) {
    if (!CLUtils::IsLocal(VD)) return S;

    QualType VDTy = VD->getType();
    if (VDTy->isPointerType()) return S;

    unsigned typeSize = CLUtils::GetTypeSize(VDTy, PointerSize);
    unsigned numDecls = 0;
    Expr *InitExpr = NULL;

    while (true) {
      numDecls++;

      // save this VarDecl
      LocalVDSet.insert(VD);

      // convert VarDecl with init expr to BinaryOperator
      if (VD->hasInit()) {
        DeclRefExpr *DRE = new (ASTCtx) DeclRefExpr(
            VD, VD->getType(), VK_LValue, VD->getLocation());
        Expr *LHS = TransformExpr(DRE);

        BinaryOperator *BinOp = new (ASTCtx) BinaryOperator(
            LHS, VD->getInit(), BO_Assign, VD->getType(), VK_RValue, 
            OK_Ordinary, VD->getLocation());

        if (InitExpr) {
          InitExpr = new (ASTCtx) BinaryOperator(
              InitExpr, BinOp, BO_Comma, VD->getType(), VK_RValue,
              OK_Ordinary, VD->getLocation());
        } else {
          InitExpr = BinOp;
        }

        VD->setInit(NULL);
      }

      if (++Begin == End) break;

      VD = dyn_cast<VarDecl>(*Begin);
      assert(VD && "Not a VarDecl");
    }

    LocalMemSize += (typeSize * numDecls);

    if (InitExpr != NULL) return InitExpr;
    else return NULL;
  }

  return S;
}

Stmt *TransformLocalVD::TransformRawIfStmt(IfStmt *If) {
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
Stmt *TransformLocalVD::VisitNullStmt(NullStmt *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitCompoundStmt(CompoundStmt *Node) {
  return TransformRawCompoundStmt(Node);
}

Stmt *TransformLocalVD::VisitLabelStmt(LabelStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *TransformLocalVD::VisitIfStmt(IfStmt *If) {
  return TransformRawIfStmt(If);
}

Stmt *TransformLocalVD::VisitSwitchStmt(SwitchStmt *Node) {
  Node->setCond(TransformExpr(Node->getCond()));

  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }
  return Node;
}

Stmt *TransformLocalVD::VisitWhileStmt(WhileStmt *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setBody(TransformStmt(Node->getBody()));
  return Node;
}

Stmt *TransformLocalVD::VisitDoStmt(DoStmt *Node) {
  if (CompoundStmt *CS = dyn_cast<CompoundStmt>(Node->getBody())) {
    Node->setBody(TransformRawCompoundStmt(CS));
  } else {
    Node->setBody(TransformStmt(Node->getBody()));
  }

  Node->setCond(TransformExpr(Node->getCond()));
  return Node;
}

Stmt *TransformLocalVD::VisitForStmt(ForStmt *Node) {
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

Stmt *TransformLocalVD::VisitGotoStmt(GotoStmt *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Node->setTarget(TransformExpr(Node->getTarget()));
  return Node;
}

Stmt *TransformLocalVD::VisitContinueStmt(ContinueStmt *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitBreakStmt(BreakStmt *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitReturnStmt(ReturnStmt *Node) {
  if (Node->getRetValue()) {
    Node->setRetValue(TransformExpr(Node->getRetValue()));
  }
  return Node;
}

Stmt *TransformLocalVD::VisitDeclStmt(DeclStmt *Node) {
  return TransformRawDeclStmt(Node);
}

Stmt *TransformLocalVD::VisitCaseStmt(CaseStmt *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  if (Node->getRHS()) {
    Node->setRHS(TransformExpr(Node->getRHS()));
  }

  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *TransformLocalVD::VisitDefaultStmt(DefaultStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}


//===--------------------------------------------------------------------===//
// Expr methods.
//===--------------------------------------------------------------------===//
Stmt *TransformLocalVD::VisitPredefinedExpr(PredefinedExpr *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitDeclRefExpr(DeclRefExpr *Node) {
  if (CLOpts.UseTLBLocal) {
    if (VarDecl *VD = dyn_cast<VarDecl>(Node->getDecl())) {
      if (LocalVDSet.find(VD) != LocalVDSet.end()) {
        Expr *args[1] = { Node };
        return new (ASTCtx) CallExpr(ASTCtx,
          CLExprs.getExpr(CLExpressions::TLB_GET_LOCAL),
          args, 1, Node->getType(), Node->getValueKind(), Node->getLocation());
      }
    }
  }
  return Node;
}

Stmt *TransformLocalVD::VisitIntegerLiteral(IntegerLiteral *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitFloatingLiteral(FloatingLiteral *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitStringLiteral(StringLiteral *Str) {
  return Str;
}

Stmt *TransformLocalVD::VisitCharacterLiteral(CharacterLiteral *Node) {
  return Node;
}

Stmt *TransformLocalVD::VisitParenExpr(ParenExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformLocalVD::VisitUnaryOperator(UnaryOperator *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformLocalVD::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

Stmt *TransformLocalVD::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *TransformLocalVD::VisitVecStepExpr(VecStepExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *TransformLocalVD::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *TransformLocalVD::VisitCallExpr(CallExpr *Call) {
  Call->setCallee(TransformExpr(Call->getCallee()));
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Call->setArg(i, TransformExpr(Call->getArg(i)));
  }
  return Call;
}

Stmt *TransformLocalVD::VisitMemberExpr(MemberExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *TransformLocalVD::VisitBinaryOperator(BinaryOperator *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *TransformLocalVD::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *TransformLocalVD::VisitConditionalOperator(ConditionalOperator *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *TransformLocalVD::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformLocalVD::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformLocalVD::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Node->setInitializer(TransformExpr(Node->getInitializer()));
  return Node;
}

Stmt *TransformLocalVD::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *TransformLocalVD::VisitInitListExpr(InitListExpr *Node) {
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (Node->getInit(i))
      Node->setInit(i, TransformExpr(Node->getInit(i)));
  }
  return Node;
}

Stmt *TransformLocalVD::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Node->getSubExpr(i)) {
      Node->setSubExpr(i, TransformExpr(Node->getSubExpr(i)));
    }
  }

  Node->setInit(TransformExpr(Node->getInit()));
  return Node;
}

Stmt *TransformLocalVD::VisitParenListExpr(ParenListExpr* Node) {
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    Node->setExpr(i, TransformExpr(Node->getExpr(i)));
  }
  return Node;
}

Stmt *TransformLocalVD::VisitVAArgExpr(VAArgExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

