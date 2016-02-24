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
#include "TransformVector.h"
#include "Defines.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

#include <string>
#include <sstream>
using std::string;
using std::stringstream;


void TransformVector::Transform(FunctionDecl *FD) {
  if (!FD->hasBody()) return;

  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  assert(FDBody && "Function body must be a CompoundStmt");

  FDBody = TransformRawCompoundStmt(FDBody);
  FD->setBody(FDBody);
}


CompoundStmt *TransformVector::TransformRawCompoundStmt(CompoundStmt *Node) {
  StmtVector StmtVec;
  
  // Save the current StmtVec pointer and set the new one.
  StmtVector *PrvStmtVec = CurStmtVec;
  CurStmtVec = &StmtVec;

  CompoundStmt::body_iterator I, E;
  for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
    if (Expr *Ex = dyn_cast<Expr>(*I)) {
      if (Ex->getType()->isVectorType()) {
        // Vector literal conversion
        DeclVector DeclVec;
        Ex = ConvertVecLiteralInExpr(DeclVec, Ex);
        if (DeclVec.size() > 0) {
          PushBackDeclStmts(StmtVec, DeclVec);
        }
        StmtVec.push_back(Ex);
        continue;
      }
    }
    
    StmtVec.push_back(TransformStmt(*I));
  }

  // Make a new body of CompoundStmt.
  Node->setStmts(ASTCtx, StmtVec.data(), StmtVec.size());

  // Restore CurStmtVec.
  CurStmtVec = PrvStmtVec;

  return Node;
}


Stmt *TransformVector::TransformRawDeclStmt(DeclStmt *S) {
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


//===--------------------------------------------------------------------===//
//  Stmt methods.
//===--------------------------------------------------------------------===//
Stmt *TransformVector::VisitNullStmt(NullStmt *Node) {
  return Node;
}

Stmt *TransformVector::VisitCompoundStmt(CompoundStmt *Node) {
  return TransformRawCompoundStmt(Node);
}

Stmt *TransformVector::VisitLabelStmt(LabelStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}


Stmt *TransformVector::VisitIfStmt(IfStmt *If) {
  // Cond
  Expr *Cond = If->getCond();
  if (Cond->getType()->isVectorType()) {
    DeclVector DeclVec;
    If->setCond(ConvertVecLiteralInExpr(DeclVec, Cond));

    if (DeclVec.size() > 0) {
      PushBackDeclStmts(*CurStmtVec, DeclVec);
    }
  } else {
    If->setCond(TransformExpr(Cond));
  }

  // Then
  Stmt *Then = If->getThen();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Then);
  if (!CS) {
    // Convert a Then stmt into a CompoundStmt.
    SourceLocation loc;
    CS = new (ASTCtx) CompoundStmt(ASTCtx, &Then, 1, loc, loc);
  }
  If->setThen(TransformStmt(CS));
  
  // Else
  if (Stmt *Else = If->getElse()) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Else);
    if (!CS) {
      // Convert a single stmt Then into a compound stmt.
      SourceLocation loc;
      CS = new (ASTCtx) CompoundStmt(ASTCtx, &Else, 1, loc, loc);
    }
    If->setElse(TransformStmt(CS));
  }

  return If;
}


Stmt *TransformVector::VisitSwitchStmt(SwitchStmt *Node) {
  DeclVector DeclVec;

  // Cond
  Expr *Cond = Node->getCond();
  if (Cond->getType()->isVectorType()) {
    Node->setCond(ConvertVecLiteralInExpr(DeclVec, Cond));
  } else {
    Node->setCond(TransformExpr(Cond));
  }

  // Body
  Stmt *Body = Node->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS) {
    // Convert a single stmt body into a CompoundStmt.
    SourceLocation loc;
    CS = new (ASTCtx) CompoundStmt(ASTCtx, &Body, 1, loc, loc);
  }

  // Check all SwitchCase stmts themselves.
  SwitchCase *CurCase = Node->getSwitchCaseList();
  while (CurCase) {
    if (CaseStmt *CaseS = dyn_cast<CaseStmt>(CurCase)) {
      Expr *CaseLHS = CaseS->getLHS();
      if (CaseLHS->getType()->isVectorType()) {
        CaseS->setLHS(ConvertVecLiteralInExpr(DeclVec, CaseLHS));
      } else {
        CaseS->setLHS(TransformExpr(CaseLHS));
      }

      Expr *CaseRHS = CaseS->getRHS();
      if (CaseRHS && CaseRHS->getType()->isVectorType()) {
        CaseS->setRHS(ConvertVecLiteralInExpr(DeclVec, CaseRHS));
      } else {
        CaseS->setRHS(TransformExpr(CaseRHS));
      }
    }

    CurCase = CurCase->getNextSwitchCase();
  }

  // Check if there was a vector literal code motion.
  if (DeclVec.size() > 0) {
    PushBackDeclStmts(*CurStmtVec, DeclVec);
  }

  // If case stmts are not CompoundStmts, convert them into CompoundStmts
  StmtVector BodyStmts;
  CompoundStmt::body_iterator I, E;
  for (I = CS->body_begin(), E = CS->body_end(); I != E; ++I) {
    // Save the current stmt
    BodyStmts.push_back(*I);

    if (SwitchCase *SC = dyn_cast<SwitchCase>(*I)) {
      CompoundStmt::body_iterator NextI = I + 1; 
      if (NextI == E) break;

      if (isa<SwitchCase>(*NextI)) {
        // No stmt between current case stmt and next case stmt.
        if (Stmt *SubStmt = SC->getSubStmt()) {
          if (!isa<CompoundStmt>(SubStmt)) {
            SourceLocation loc;
            CompoundStmt *SubCS = new (ASTCtx) CompoundStmt(ASTCtx,
                &SubStmt, 1, loc, loc);

            if (CaseStmt *CaseS = dyn_cast<CaseStmt>(SC)) {
              CaseS->setSubStmt(SubCS);
            } else if (DefaultStmt *DefaultS = dyn_cast<DefaultStmt>(SC)) {
              DefaultS->setSubStmt(SubCS);
            } else {
              assert(0 && "What statement?");
            }
          }
        }
      } else {
        StmtVector StmtVec;

        // SubStmt
        if (Stmt *SubStmt = SC->getSubStmt()) {
          StmtVec.push_back(SubStmt);
        }

        // Following stmts
        do {
          I = NextI;
          StmtVec.push_back(*I);
        } while ((++NextI != E) && !isa<SwitchCase>(*NextI));

        // Convert all stmts into a CompoundStmt.
        SourceLocation loc;
        CompoundStmt *SubCS = new (ASTCtx) CompoundStmt(ASTCtx,
            StmtVec.data(), StmtVec.size(), loc, loc);

        if (CaseStmt *CaseS = dyn_cast<CaseStmt>(SC)) {
          CaseS->setSubStmt(SubCS);
        } else if (DefaultStmt *DefaultS = dyn_cast<DefaultStmt>(SC)) {
          DefaultS->setSubStmt(SubCS);
        } else {
          assert(0 && "What statement?");
        }
      }
    } //end if
  } //end for

  CS = new (ASTCtx) CompoundStmt(ASTCtx, BodyStmts.data(), BodyStmts.size(),
      SourceLocation(), SourceLocation());
  ASTCtx.Deallocate(Node->getBody());
  Node->setBody(TransformStmt(CS));

  return Node;
}


Stmt *TransformVector::VisitWhileStmt(WhileStmt *Node) {
  DeclVector DeclVec;

  // Cond
  Expr *Cond = Node->getCond();
  if (Cond->getType()->isVectorType()) {
    Node->setCond(ConvertVecLiteralInExpr(DeclVec, Cond));
    if (DeclVec.size() > 0) {
      PushBackDeclStmts(*CurStmtVec, DeclVec);
    }
  } else {
    Node->setCond(TransformExpr(Cond));
  }
  
  // Body
  Stmt *Body = Node->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS) {
    // Convert a single stmt Body into a compound stmt.
    SourceLocation loc;
    CS = new (ASTCtx) CompoundStmt(ASTCtx, &Body, 1, loc, loc);
  }
  Node->setBody(TransformStmt(CS));

  // Check if there was a vector literal code motion.
  if (DeclVec.size() > 0) {
    // Add vector literal assignment stmts at the end of the body.
    Node->setBody(MergeBodyAndDecls(Node->getBody(), DeclVec));
  }

  return Node;
}


Stmt *TransformVector::VisitDoStmt(DoStmt *Node) {
  // Body
  Stmt *Body = Node->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS) {
    // Convert a single stmt Body into a compound stmt.
    SourceLocation loc;
    CS = new (ASTCtx) CompoundStmt(ASTCtx, &Body, 1, loc, loc);
  }
  Node->setBody(TransformStmt(CS));

  // Cond
  Expr *Cond = Node->getCond();
  if (Cond->getType()->isVectorType()) {
    DeclVector DeclVec;
    Node->setCond(ConvertVecLiteralInExpr(DeclVec, Cond));

    if (DeclVec.size() > 0) {
      CS = dyn_cast<CompoundStmt>(Node->getBody());

      StmtVector StmtVec;
      CompoundStmt::body_iterator I, E;
      for (I = CS->body_begin(), E = CS->body_end(); I != E; ++I) {
        StmtVec.push_back(*I);
      }
      PushBackDeclStmts(StmtVec, DeclVec);

      CS->setStmts(ASTCtx, StmtVec.data(), StmtVec.size());
    }
  } else {
    Node->setCond(TransformExpr(Cond));
  }

  return Node;
}


Stmt *TransformVector::VisitForStmt(ForStmt *Node) {
  DeclVector DeclVec;   // Decls to be declared before ForStmt
  DeclVector ReDeclVec; // Decls to be re-declared at the end of the body
  DeclVector CondDeclVec;
  DeclVector IncDeclVec;

  // Init
  if (Stmt *Init = Node->getInit()) {
    if (Expr *E = dyn_cast<Expr>(Init)) {
      if (E->getType()->isVectorType()) {
        Node->setInit(ConvertVecLiteralInExpr(DeclVec, cast<Expr>(Init)));
      } else {
        Node->setInit(TransformExpr(E));
      }
    } else {
      Node->setInit(TransformStmt(Init));
    }
  }

  // Cond
  if (Expr *Cond = Node->getCond()) {
    if (Cond->getType()->isVectorType()) {
      Node->setCond(ConvertVecLiteralInExpr(CondDeclVec, Cond));
    } else {
      Node->setCond(TransformExpr(Cond));
    }
  }

  // Inc
  if (Expr *Inc = Node->getInc()) {
    if (Inc->getType()->isVectorType()) {
      Node->setInc(ConvertVecLiteralInExpr(IncDeclVec, Inc));
    } else {
      Node->setInc(TransformExpr(Inc));
    }
  }

  // Merge decl vectors
  for (unsigned i = 0, e = CondDeclVec.size(); i < e; i++)
    DeclVec.push_back(CondDeclVec[i]);
  for (unsigned i = 0, e = IncDeclVec.size(); i < e; i++) {
    DeclVec.push_back(IncDeclVec[i]);
    ReDeclVec.push_back(IncDeclVec[i]);
  }
  for (unsigned i = 0, e = CondDeclVec.size(); i < e; i++)
    ReDeclVec.push_back(CondDeclVec[i]);

  // Check if there was a vector literal code motion.
  if (DeclVec.size() > 0) {
    PushBackDeclStmts(*CurStmtVec, DeclVec);
  }

  // Body
  if (Stmt *Body = Node->getBody()) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
    if (!CS) {
      // Convert a single stmt Body into a compound stmt.
      SourceLocation loc;
      CS = new (ASTCtx) CompoundStmt(ASTCtx, &Body, 1, loc, loc);
    }
    Node->setBody(TransformStmt(CS));
  }

  // Add vector literal assignment stmts at the end of the body.
  if (ReDeclVec.size() > 0) {
    Node->setBody(MergeBodyAndDecls(Node->getBody(), ReDeclVec));
  }

  return Node;
}


Stmt *TransformVector::VisitGotoStmt(GotoStmt *Node) {
  return Node;
}

Stmt *TransformVector::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Expr *Target = Node->getTarget();
  if (Target->getType()->isVectorType()) {
    DeclVector DeclVec;
    Node->setTarget(ConvertVecLiteralInExpr(DeclVec, Target));
    if (DeclVec.size() > 0) {
      PushBackDeclStmts(*CurStmtVec, DeclVec);
    }
  } else {
    Node->setTarget(TransformExpr(Node->getTarget()));
  }

  return Node;
}

Stmt *TransformVector::VisitContinueStmt(ContinueStmt *Node) {
  return Node;
}

Stmt *TransformVector::VisitBreakStmt(BreakStmt *Node) {
  return Node;
}

Stmt *TransformVector::VisitReturnStmt(ReturnStmt *Node) {
  if (Expr *RetExpr = Node->getRetValue()) {
    if (RetExpr->getType()->isVectorType()) {
      DeclVector DeclVec;
      Node->setRetValue(ConvertVecLiteralInExpr(DeclVec, RetExpr));
      if (DeclVec.size() > 0) {
        PushBackDeclStmts(*CurStmtVec, DeclVec);
      }
    } else {
      Node->setRetValue(TransformExpr(RetExpr));
    }
  }
  return Node;
}

Stmt *TransformVector::VisitDeclStmt(DeclStmt *Node) {
  return TransformRawDeclStmt(Node);
}

Stmt *TransformVector::VisitCaseStmt(CaseStmt *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  if (Node->getRHS()) {
    Node->setRHS(TransformExpr(Node->getRHS()));
  }

  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}

Stmt *TransformVector::VisitDefaultStmt(DefaultStmt *Node) {
  Node->setSubStmt(TransformStmt(Node->getSubStmt()));
  return Node;
}


//===--------------------------------------------------------------------===//
// Expr methods.
//===--------------------------------------------------------------------===//
Stmt *TransformVector::VisitPredefinedExpr(PredefinedExpr *Node) {
  return Node;
}

Stmt *TransformVector::VisitDeclRefExpr(DeclRefExpr *Node) {
  return Node;
}

Stmt *TransformVector::VisitIntegerLiteral(IntegerLiteral *Node) {
  return Node;
}

Stmt *TransformVector::VisitFloatingLiteral(FloatingLiteral *Node) {
  return Node;
}

Stmt *TransformVector::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  return Node;
}

Stmt *TransformVector::VisitStringLiteral(StringLiteral *Str) {
  return Str;
}

Stmt *TransformVector::VisitCharacterLiteral(CharacterLiteral *Node) {
  return Node;
}

Stmt *TransformVector::VisitParenExpr(ParenExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformVector::VisitUnaryOperator(UnaryOperator *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *TransformVector::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

Stmt *TransformVector::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *TransformVector::VisitVecStepExpr(VecStepExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *TransformVector::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *TransformVector::VisitCallExpr(CallExpr *Call) {
  Call->setCallee(TransformExpr(Call->getCallee()));
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Call->setArg(i, TransformExpr(Call->getArg(i)));
  }
  return Call;
}

Stmt *TransformVector::VisitMemberExpr(MemberExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *TransformVector::VisitBinaryOperator(BinaryOperator *Node) {
  if (Node->getType()->isVectorType()) {
    DeclVector DeclVec;
    Node->setLHS(ConvertVecLiteralInExpr(DeclVec, Node->getLHS()));
    Node->setRHS(ConvertVecLiteralInExpr(DeclVec, Node->getRHS()));
    if (DeclVec.size() > 0) {
      PushBackDeclStmts(*CurStmtVec, DeclVec);
    }
  } else {
    Node->setLHS(TransformExpr(Node->getLHS()));
    Node->setRHS(TransformExpr(Node->getRHS()));
  }
  return Node;
}

Stmt *TransformVector::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}


Stmt *TransformVector::VisitConditionalOperator(ConditionalOperator *Node) {
  Expr *Cond = Node->getCond();
  QualType CondTy = Cond->getType();
  Expr *LHS = Node->getLHS();
  Expr *RHS = Node->getRHS();

  if (CondTy->isVectorType()) {
    // If the type of Cond is a vector type, change this expr to select().
    DeclVector DeclVec;
    Cond = ConvertVecLiteralInExpr(DeclVec, Cond);
    LHS = ConvertVecLiteralInExpr(DeclVec, LHS);
    RHS = ConvertVecLiteralInExpr(DeclVec, RHS);

    if (DeclVec.size() > 0) {
      PushBackDeclStmts(*CurStmtVec, DeclVec);
    }

    QualType NodeTy = Node->getType();

    ASTCtx.Deallocate(Node);

    Expr *Args[3] = { RHS, LHS, Cond };
    return new (ASTCtx) CallExpr(ASTCtx, 
        CLExprs.getExpr(CLExpressions::SELECT), Args, 3, NodeTy, 
        VK_RValue, SourceLocation());
  } else {
    Node->setCond(TransformExpr(Cond));
    Node->setLHS(TransformExpr(LHS));
    Node->setRHS(TransformExpr(RHS));
  }
  return Node;
}


Stmt *TransformVector::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  QualType NodeTy = Node->getType();
  QualType SubExprTy = Node->getSubExpr()->getType();

  Expr *SubExpr = TransformExpr(Node->getSubExpr());
  if (NodeTy == SubExprTy) {
    ASTCtx.Deallocate(Node);
    return SubExpr;
  }

  Node->setSubExpr(SubExpr);
  return Node;
}

Stmt *TransformVector::VisitCStyleCastExpr(CStyleCastExpr *Node) {
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


Stmt *TransformVector::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  QualType ETy = Node->getType();
  if (const ExtVectorType *EVT = ETy->getAs<ExtVectorType>()) {
    // This is a vector literal expression.
    unsigned NumElems = EVT->getNumElements();
    Expr **Args = new (ASTCtx) Expr*[NumElems];

    TransformVectorLiteralExpr(Node, Args, 0);

    Expr *NewExpr = NULL;
    if (NeedFlattening) {
      assert(CurInitExprs && "CurInitExprs is NULL");
      for (unsigned i = 0; i < NumElems - 1; i++) {
        CurInitExprs->push_back(Args[i]);
      }
      NewExpr = Args[NumElems - 1];
    } else {
      NewExpr = new (ASTCtx) CallExpr(ASTCtx,
          CLExprs.getVectorLiteralExpr(ETy), Args, NumElems, ETy, VK_RValue,
          SourceLocation());
    }

    ASTCtx.Deallocate(Args);
    ASTCtx.Deallocate(Node);
    return NewExpr;
  } else {
    Node->setInitializer(TransformExpr(Node->getInitializer()));
  }
  return Node;
} 
void TransformVector::TransformVectorLiteralExpr(
    CompoundLiteralExpr *E, Expr **Args, unsigned StartPos) {
  InitListExpr *ILE = dyn_cast<InitListExpr>(E->getInitializer());
  assert(ILE && "ERROR: Vector literal is not an InitListExpr");

  QualType ETy = E->getType();
  const ExtVectorType *EVT = ETy->getAs<ExtVectorType>();
  unsigned NumElems = EVT->getNumElements();
  unsigned VecPos   = 0;
  bool HasOneInitElem = (ILE->getNumInits() == 1);

  for (unsigned i = StartPos; i < StartPos + NumElems; i++) {
    Expr *InitExpr = ILE->getInit(VecPos);
    if (!HasOneInitElem) VecPos++;
    if (InitExpr == NULL) {
      // zero
      Args[i] = CLExprs.getExpr(CLExpressions::ZERO);
      continue;
    }

    QualType InitType = InitExpr->getType();
    if (!InitType->isVectorType()) {
      // scalar element
      Args[i] = TransformExpr(InitExpr);
      continue;
    }

    // vector element
    const ExtVectorType *InitVec = InitType->getAs<ExtVectorType>();
    unsigned InitNumElems = InitVec->getNumElements();

    // Strip off any ParenExpr or CastExprs
    InitExpr = InitExpr->IgnoreParenCasts();
    if (CompoundLiteralExpr *CE = dyn_cast<CompoundLiteralExpr>(InitExpr)) {
      TransformVectorLiteralExpr(CE, Args, i);
      i += (InitNumElems - 1);
    } else {
      CLExpressions::ExprKind kind;
      for (unsigned t = 0; t < InitNumElems; ++t, ++i) {
        // ArraySubscriptExpr
        kind = (CLExpressions::ExprKind)(CLExpressions::ZERO + t);
        Args[i] = new (ASTCtx) ArraySubscriptExpr(
            InitExpr, CLExprs.getExpr(kind), 
            InitType, VK_RValue, OK_Ordinary, SourceLocation());
      }
      --i;
    }
  }

  ASTCtx.Deallocate(ILE);
  ASTCtx.Deallocate(E);
}


Stmt *TransformVector::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  unsigned NumElems = Node->getNumElements();
  if (NumElems == 0) {
    // array subscripting syntax
    Expr *ExprBase = TransformExpr(Node->getBase());
    ASTCtx.Deallocate(Node);
    return ExprBase;
  } else {
    DeclVector DeclVec;
    ExprVector ExprVec;
    MakeElementExprs(DeclVec, ExprVec, Node);
    assert((ExprVec.size() == NumElems) && "Wrong accessor?");
    if (DeclVec.size() > 0) {
      PushBackDeclStmts(*CurStmtVec, DeclVec);
    }

    if (NumElems == 1) {
      return ExprVec[0];
    } else {
      QualType NodeTy = Node->getType();
      CallExpr *NewExpr = new (ASTCtx) CallExpr(ASTCtx,
          CLExprs.getVectorLiteralExpr(NodeTy), ExprVec.data(), NumElems,
          NodeTy, VK_RValue, SourceLocation());
      return NewExpr;
    }
  }
}


Stmt *TransformVector::VisitInitListExpr(InitListExpr *Node) {
  // For conventional vector literals such as '{1, 2, 3, 4}'
  QualType ETy = Node->getType();
  if (ETy->isExtVectorType()) {
    CompoundLiteralExpr *CLE = new (ASTCtx) CompoundLiteralExpr(
        SourceLocation(), ASTCtx.getTrivialTypeSourceInfo(ETy),
        ETy, Node->getValueKind(), Node, false);
    return VisitCompoundLiteralExpr(CLE);
  }

  if (NeedFlattening) {
    ExprVector InitExprs;
    ExprVector *PrvInitExprs = CurInitExprs;
    CurInitExprs = &InitExprs;

    for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
      assert(Node->getInit(i) && "NULL InitExpr?");
      Expr *InitExpr = TransformExpr(Node->getInit(i));
      InitExprs.push_back(InitExpr);
    }

    for (unsigned i =0, e = InitExprs.size(); i < e; ++i) {
      Node->updateInit(ASTCtx, i, InitExprs[i]);
    }

    CurInitExprs = PrvInitExprs;

  } else {
    for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
      if (Node->getInit(i)) {
        Node->setInit(i, TransformExpr(Node->getInit(i)));
      }
    }
  }
  
  return Node;
}

Stmt *TransformVector::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Node->getSubExpr(i)) {
      Node->setSubExpr(i, TransformExpr(Node->getSubExpr(i)));
    }
  }

  Node->setInit(TransformExpr(Node->getInit()));
  return Node;
}

Stmt *TransformVector::VisitParenListExpr(ParenListExpr* Node) {
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    Node->setExpr(i, TransformExpr(Node->getExpr(i)));
  }
  return Node;
}

Stmt *TransformVector::VisitVAArgExpr(VAArgExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}


//---------------------------------------------------------------------------
// Conversion for vector literals
//---------------------------------------------------------------------------
void TransformVector::PushBackDeclStmts(StmtVector &StmtVec, 
                                        DeclVector &DeclVec) {
  SourceLocation SL;
  for (unsigned i = 0; i < DeclVec.size(); i++) {
    DeclGroupRef DGRef = DeclGroupRef::Create(ASTCtx, &DeclVec[i], 1);
    DeclStmt *DS = new (ASTCtx) DeclStmt(DGRef, SL, SL);
    StmtVec.push_back(TransformStmt(DS));
  }
}


Expr *TransformVector::ConvertVecLiteralInExpr(DeclVector &DeclVec,
                                               Expr *E,
                                               bool IsTopDecl) {
  if (CompoundLiteralExpr *CLE = dyn_cast<CompoundLiteralExpr>(E)) {
    QualType LitTy = CLE->getType();
    if (LitTy->isVectorType()) { // vector literal
      VecLiteralMode = true;
      Expr *Init = ConvertVecLiteralInExpr(DeclVec, CLE->getInitializer());
      CLE->setInitializer(Init);
      VecLiteralMode = false;

      // Make a VarDecl with an initialization expr.
      VarDecl *VD = NewVecLiteralVarDecl(LitTy);
      VD->setInit(CLE);
      DeclVec.push_back(VD);

      // Return a DeclRefExpr that refers above VarDecl.
      return new (ASTCtx) DeclRefExpr(VD, LitTy, VK_RValue, SourceLocation());
    } else {
      Expr *Init = ConvertVecLiteralInExpr(DeclVec, CLE->getInitializer());
      CLE->setInitializer(Init);
    }

    return CLE;
  }
  else if (ExtVectorElementExpr *EVEE = dyn_cast<ExtVectorElementExpr>(E)) {
    unsigned NumElems = EVEE->getNumElements();
    Expr *NewE = TransformExpr(E);

    if (IsTopDecl && NumElems > 1) {
      // vector literal
      VarDecl *VD = NewVecLiteralVarDecl(EVEE->getType());
      VD->setInit(NewE);
      DeclVec.push_back(VD);

      // Return a DeclRefExpr that refers above VarDecl.
      SourceLocation loc;
      return new (ASTCtx) DeclRefExpr(VD, VD->getType(), VK_RValue, loc);
    }
    return NewE;
  }
  else if (CompoundAssignOperator *CA = dyn_cast<CompoundAssignOperator>(E)) {
    Expr *CLHS = CA->getLHS();
    Expr *CRHS = CA->getRHS();

    if (ExtVectorElementExpr *LHS = dyn_cast<ExtVectorElementExpr>(CLHS)) {
      Expr *NewE = ConvertAssignExpr(DeclVec, LHS, CA->getOpcode(), CRHS);
      ASTCtx.Deallocate(CA);
      return NewE;
    } else {
      // LHS & RHS
      CA->setLHS(ConvertVecLiteralInExpr(DeclVec, CLHS));
      CA->setRHS(ConvertVecLiteralInExpr(DeclVec, CRHS));
      return CA;
    }
  }
  else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(E)) {
    Expr *BLHS = BinOp->getLHS();
    Expr *BRHS = BinOp->getRHS();
    BinaryOperatorKind Op = BinOp->getOpcode();

    if (Op == BO_Assign) {
      if (ExtVectorElementExpr *LHS = dyn_cast<ExtVectorElementExpr>(BLHS)) {
        Expr *NewE = ConvertAssignExpr(DeclVec, LHS, Op, BRHS);
        ASTCtx.Deallocate(BinOp);
        return NewE;
      } 
    } 

    // LHS & RHS
    BinOp->setLHS(ConvertVecLiteralInExpr(DeclVec, BLHS));
    BinOp->setRHS(ConvertVecLiteralInExpr(DeclVec, BRHS));

    return BinOp;
  }
  else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    PE->setSubExpr(ConvertVecLiteralInExpr(DeclVec, PE->getSubExpr()));
    return PE;
  }
  else if (UnaryOperator *UOp = dyn_cast<UnaryOperator>(E)) {
    UOp->setSubExpr(ConvertVecLiteralInExpr(DeclVec, UOp->getSubExpr()));
    return UOp;
  }
  else if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    ASE->setLHS(ConvertVecLiteralInExpr(DeclVec, ASE->getLHS()));
    ASE->setRHS(ConvertVecLiteralInExpr(DeclVec, ASE->getRHS()));
    return ASE;
  }
  else if (CallExpr *Call = dyn_cast<CallExpr>(E)) {
    for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
      Call->setArg(i, ConvertVecLiteralInExpr(DeclVec, Call->getArg(i)));
    }

    // If a vload function exists in a vector literal, it can be 
    // code-motioned in order to improve performance.
    if (VecLiteralMode) {
      Expr *Callee = Call->getCallee();
      if (CLUtils::IsVectorDataFunction(Callee) != -1) {
        QualType CallTy = Call->getType();
        VarDecl *VD = NewVecLiteralVarDecl(CallTy);
        VD->setInit(Call);
        DeclVec.push_back(VD);
        
        SourceLocation loc;
        return new (ASTCtx) DeclRefExpr(VD, CallTy, VK_RValue, loc);
      }
    }

    return Call;
  }
  else if (CStyleCastExpr *CSCE = dyn_cast<CStyleCastExpr>(E)) {
    CSCE->setSubExpr(ConvertVecLiteralInExpr(DeclVec, CSCE->getSubExpr()));
    return CSCE;
  }
  else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    ICE->setSubExpr(ConvertVecLiteralInExpr(DeclVec, ICE->getSubExpr()));
    return ICE;
  }
  else if (ConditionalOperator *CO = dyn_cast<ConditionalOperator>(E)) {
    return TransformExpr(CO);
  }
  else if (InitListExpr *ILE = dyn_cast<InitListExpr>(E)) {
    for (unsigned i = 0, e = ILE->getNumInits(); i != e; ++i) {
      if (Expr *init = ILE->getInit(i)) {
        init = ConvertVecLiteralInExpr(DeclVec, init);
        ILE->setInit(i, init);
      }
    }
    return ILE;
  }
  else if (DesignatedInitExpr *DIE = dyn_cast<DesignatedInitExpr>(E)) {
    DIE->setInit(ConvertVecLiteralInExpr(DeclVec, DIE->getInit()));
    return DIE;
  }

  return E;
}


Expr *TransformVector::ConvertAssignExpr(DeclVector &DeclVec,
                                         ExtVectorElementExpr *LHS,
                                         BinaryOperator::Opcode Op,
                                         Expr *BRHS) {
  QualType BRHSTy = BRHS->getType();
  Expr *RHS = BRHS->IgnoreParenCasts();
  if (!(isa<CompoundLiteralExpr>(RHS) || isa<ExtVectorElementExpr>(RHS))) {
    RHS = ConvertVecLiteralInExpr(DeclVec, RHS);

    QualType RHSTy = RHS->getType();
    if (RHSTy->isVectorType() && !isa<DeclRefExpr>(RHS)) {
      // Make a VarDecl with RHS
      VarDecl *VD = NewVecLiteralVarDecl(BRHSTy);
      VD->setInit(RHS);
      DeclVec.push_back(VD);

      // Make a DeclRefExpr
      SourceLocation loc;
      RHS = new (ASTCtx) DeclRefExpr(VD, BRHSTy, VK_RValue, loc);
    }
  }

  ExprVector LHSVec;
  MakeElementExprs(DeclVec, LHSVec, LHS);
  assert((LHSVec.size() > 0) && "Wrong element exprs");

  bool IsScalarRHS = RHS->getType()->isScalarType();
  if (LHSVec.size() == 1 && IsScalarRHS) {
    return NewBinaryOperator(LHSVec[0], Op, RHS);
  }

  Expr *NewExpr = 0;
  if (IsScalarRHS) {
    // scalar RHS
    for (unsigned i = 0, e = LHSVec.size(); i < e; i++) {
      Expr *OneExpr = NewBinaryOperator(LHSVec[i], Op, RHS);
      if (NewExpr) {
        NewExpr = NewBinaryOperator(NewExpr, BO_Comma, OneExpr);
      } else {
        NewExpr = OneExpr;
      }
    }
  } else if (CompoundLiteralExpr *CLE = dyn_cast<CompoundLiteralExpr>(RHS)) {
    unsigned NumElems = LHSVec.size();
    Expr **Args = new (ASTCtx) Expr*[NumElems];
    TransformVectorLiteralExpr(CLE, Args, 0);

    for (unsigned i = 0; i < NumElems; i++) {
      Expr *OneExpr = NewBinaryOperator(LHSVec[i], Op, Args[i]);
      if (NewExpr) {
        NewExpr = NewBinaryOperator(NewExpr, BO_Comma, OneExpr);
      } else {
        NewExpr = OneExpr;
      }
    }
  } else if (ExtVectorElementExpr *EE = dyn_cast<ExtVectorElementExpr>(RHS)) {
    ExprVector RHSVec;
    MakeElementExprs(DeclVec, RHSVec, EE);
    assert((LHSVec.size() == RHSVec.size()) && "Different LHS and RHS?");

    for (unsigned i = 0, e = LHSVec.size(); i < e; i++) {
      Expr *OneExpr = NewBinaryOperator(LHSVec[i], Op, RHSVec[i]);
      if (NewExpr) {
        NewExpr = NewBinaryOperator(NewExpr, BO_Comma, OneExpr);
      } else {
        NewExpr = OneExpr;
      }
    }
  } else {
    // vector RHS
    for (unsigned i = 0, e = LHSVec.size(); i < e; i++) {
      QualType Ty = LHSVec[i]->getType();

      // RHS[i]
      ArraySubscriptExpr *ElemRHS = new (ASTCtx) ArraySubscriptExpr(
          RHS, 
          CLExprs.getExpr((CLExpressions::ExprKind)(CLExpressions::ZERO + i)), 
          Ty, VK_RValue, OK_Ordinary, SourceLocation());

      Expr *OneExpr = NewBinaryOperator(LHSVec[i], Op, ElemRHS);
      if (NewExpr) {
        NewExpr = NewBinaryOperator(NewExpr, BO_Comma, OneExpr);
      } else {
        NewExpr = OneExpr;
      }
    }
  }

  return NewExpr;
}


void TransformVector::MakeElementExprs(DeclVector &DeclVec, 
                                       ExprVector &ExprVec,
                                       ExtVectorElementExpr *E) {
  llvm::SmallVector<unsigned, 4> Indices;
  E->getEncodedElementAccess(Indices);
  Expr *BE = E->getBase();

  // If E is an arrow expression, the base pointer expression needs to be 
  // converted into a vector value expression.
  if (E->isArrow()) {
    QualType BaseTy = BE->getType();
    const PointerType *PTy = BaseTy->getAs<PointerType>();
    assert(PTy && "Not a pointer type");
    BE = new (ASTCtx) UnaryOperator(BE, UO_Deref, PTy->getPointeeType(),
                                    VK_RValue, OK_Ordinary, SourceLocation());
    BE = new (ASTCtx) ParenExpr(SourceLocation(), SourceLocation(), BE);
  }

  if (ExtVectorElementExpr *BP = dyn_cast<ExtVectorElementExpr>(BE)) {
    ExprVector BaseExprVec;
    MakeElementExprs(DeclVec, BaseExprVec, BP);
    for (unsigned i = 0, e = Indices.size(); i < e; i++) {
      ExprVec.push_back(BaseExprVec[Indices[i]]);
    }
  } else if (CompoundLiteralExpr *BP = dyn_cast<CompoundLiteralExpr>(BE)) {
    for (unsigned i = 0, e = Indices.size(); i < e; i++) {
      Expr *ElemE = GetSingleValueOfVecLiteral(DeclVec, BP, Indices[i]);
      ExprVec.push_back(ElemE);
    }
  } else {
    Expr *NewBE = ConvertVecLiteralInExpr(DeclVec, BE);
    const ExtVectorType *VecTy = NewBE->getType()->getAs<ExtVectorType>();
    assert(VecTy && "The type of BaseExpr is not a vector type.");

    QualType ElemTy = VecTy->getElementType();
    SourceLocation loc;

    for (unsigned i = 0, e = Indices.size(); i < e; i++) {
      unsigned Kind = CLExpressions::ZERO + Indices[i];
      ArraySubscriptExpr *ElemE = new (ASTCtx) ArraySubscriptExpr(
          NewBE,
          CLExprs.getExpr((CLExpressions::ExprKind)(Kind)), 
          ElemTy, VK_RValue, OK_Ordinary, loc);
      ExprVec.push_back(ElemE);
    }
  }
}


Expr *TransformVector::GetSingleValueOfVecLiteral(DeclVector &DeclVec, 
                                                  CompoundLiteralExpr *CLE, 
                                                  unsigned Idx) {
  InitListExpr *ILE = dyn_cast<InitListExpr>(CLE->getInitializer());

  // Get the type info of the current vector type
  const ExtVectorType *EVT = CLE->getType()->getAs<ExtVectorType>();
  unsigned numElems = EVT->getNumElements();
  unsigned vecPos   = 0;
  bool hasOneInitElem = (ILE->getNumInits() == 1);

  // Choose the specified element
  for (unsigned i = 0; i < numElems; ++i) {
    if (Expr *initExpr = ILE->getInit(vecPos)) {
      QualType initTy = initExpr->getType();
      if (initTy->isExtVectorType()) {
        // vector
        const ExtVectorType *initVec = initTy->getAs<ExtVectorType>();
        unsigned initNumElem = initVec->getNumElements();
        for (unsigned t = 0; t < initNumElem; ++t, ++i) {
          if (i != Idx) continue;

          if (ExtVectorElementExpr *P = dyn_cast<ExtVectorElementExpr>(initExpr)) {
            ExprVector ExprVec;
            MakeElementExprs(DeclVec, ExprVec, P);
            return ExprVec[t];
          } else if (CompoundLiteralExpr *P = dyn_cast<CompoundLiteralExpr>(initExpr)) {
            return GetSingleValueOfVecLiteral(DeclVec, P, t);
          } else { 
            Expr *LHS = ConvertVecLiteralInExpr(DeclVec, initExpr);

            QualType elemTy = initVec->getElementType();
            SourceLocation loc;
            
            unsigned kind = CLExpressions::ZERO + i;
            return new (ASTCtx) ArraySubscriptExpr(
                LHS,
                CLExprs.getExpr((CLExpressions::ExprKind)kind), 
                elemTy, VK_RValue, OK_Ordinary, loc);
          }
        } //for t
        --i;
      } else {
        // scalar
        if (i == Idx) {
          return ConvertVecLiteralInExpr(DeclVec, initExpr);
        }
      }
    } else {
      assert(0 && "L-value cannot be empty.");
    }

    if (!hasOneInitElem) vecPos++;
  }

  assert(0 && "Wrong index??");

  return CLE;
}


Stmt *TransformVector::MergeBodyAndDecls(Stmt *Body, DeclVector &DeclVec) {
  unsigned NumDecls = DeclVec.size();
  unsigned NumStmts = NumDecls + 1;

  Stmt **Stmts = new (ASTCtx) Stmt*[NumStmts];
  Stmts[0] = Body;

  SourceLocation loc;
  for (unsigned i = 0; i < NumDecls; i++) {
    VarDecl *VD = dyn_cast<VarDecl>(DeclVec[i]);
    assert(VD && "Element of DeclVec should a VarDecl");

    QualType Ty = VD->getType();
    DeclRefExpr *DRE = new (ASTCtx) DeclRefExpr(VD, Ty, VK_LValue, loc);
    Expr *RHS = VD->getInit();
    Stmts[i + 1] = new (ASTCtx) BinaryOperator(
        DRE, RHS, BO_Assign, Ty, VK_RValue, OK_Ordinary, loc);
  }

  CompoundStmt *CS = new (ASTCtx) CompoundStmt(ASTCtx, Stmts, NumStmts, 
                                               loc, loc);
  ASTCtx.Deallocate(Stmts);

  return CS;
}


VarDecl *TransformVector::NewVecLiteralVarDecl(QualType DeclTy) {
  IdentifierTable &Idents = ASTCtx.Idents;

  stringstream Name;
  Name << OPENCL_VEC_LIT_VAR_PREFIX << LiteralNum++ << "e";
  
  string NameStr = Name.str();
  StringRef DeclName(NameStr);
  IdentifierInfo &DeclID = Idents.get(DeclName);

  VarDecl *VD = VarDecl::Create(ASTCtx, DeclCtx, SourceLocation(), &DeclID,
                                DeclTy, ASTCtx.getNullTypeSourceInfo(),
                                SC_None, SC_None);
  return VD;
}


Expr *TransformVector::NewBinaryOperator(Expr *LHS,
                                         BinaryOperator::Opcode Op,
                                         Expr *RHS) {
  QualType T = LHS->getType();

  if (Op == BO_Assign || Op == BO_Comma) {
    return new (ASTCtx) BinaryOperator(LHS, RHS, Op, T, VK_RValue, 
                                       OK_Ordinary, SourceLocation());
  } else {
    return new (ASTCtx) CompoundAssignOperator(
        LHS, RHS, Op, T, VK_RValue, OK_Ordinary, T, T, SourceLocation());
  }
}


