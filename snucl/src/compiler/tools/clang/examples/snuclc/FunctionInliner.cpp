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
#include "FunctionInliner.h"
#include "FunctionDuplicator.h"
#include "Defines.h"
#include "CLUtils.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

#include <string>
#include <sstream>
using std::string;
using std::stringstream;


void FunctionInliner::Inline(FunctionDecl *FD) {
  CompoundStmt *Body = dyn_cast<CompoundStmt>(FD->getBody());
  assert(Body && "Function body must be a CompoundStmt");

  // Inline the function calls.
  FD->setBody(VisitCompoundStmt(Body));

  // Since all invocations of callees that have barrier() in their body
  // we remove FD in the caller set of each callee and clear the set of 
  // callees of FD. 
  FuncDeclSetTy InlinedFDs;
  FuncDeclSetTy &Callees = CallMap[FD].Callees;
  for (FuncDeclSetTy::iterator I = Callees.begin(), E = Callees.end();
       I != E; ++I) {
    FunctionDecl *Callee = *I;
    if (Callee->hasBarrierCall() || Callee->hasIndirectBarrierCall()) {
      FuncDeclSetTy &Callers = CallMap[Callee].Callers;
      assert(Callers.find(FD) != Callers.end() && "FD is not found!");
      Callers.erase(FD);

      InlinedFDs.insert(Callee);
    }
  }
  for (FuncDeclSetTy::iterator I = InlinedFDs.begin(), E = InlinedFDs.end();
       I != E; ++I) {
    Callees.erase(*I);
  }

  // Now, FD has direct barrier() invocations.
  FD->setBarrierCall(true);
  FD->setIndirectBarrierCall(false);
}


CompoundStmt *FunctionInliner::TransformRawCompoundStmt(CompoundStmt *Node) {
  StmtVector *PrvBody = CurBody;
  StmtVector Stmts;

  CurBody = &Stmts;

  CompoundStmt::body_iterator I, E;
  for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
    if (CallExpr *Call = dyn_cast<CallExpr>(*I)) {
      Stmt *S = VisitTopLevelCallExpr(Call);
      if (S) Stmts.push_back(S);
    } else {
      Stmts.push_back(TransformStmt(*I));
    }
  }
  Node->setStmts(ASTCtx, Stmts.data(), Stmts.size());

  CurBody = PrvBody;

  return Node;
}

Stmt *FunctionInliner::TransformRawDeclStmt(DeclStmt *S) {
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
Stmt *FunctionInliner::VisitNullStmt(NullStmt *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitCompoundStmt(CompoundStmt *Node) {
  return TransformRawCompoundStmt(Node);
}

Stmt *FunctionInliner::VisitLabelStmt(LabelStmt *Node) {
  if (Stmt *SubStmt = Node->getSubStmt())
    Node->setSubStmt(TransformStmt(SubStmt));
  return Node;
}

Stmt *FunctionInliner::VisitIfStmt(IfStmt *If) {
  // Cond
  If->setCond(TransformExpr(If->getCond()));

  // Then
  Stmt *Then = If->getThen();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Then);
  if (!CS) CS = ConvertToCompoundStmt(Then);
  If->setThen(TransformRawCompoundStmt(CS));

  // Else
  if (Stmt *Else = If->getElse()) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Else);
    if (!CS) CS = ConvertToCompoundStmt(Else);
    If->setElse(TransformRawCompoundStmt(CS));
  }

  return If;
}

Stmt *FunctionInliner::VisitSwitchStmt(SwitchStmt *Node) {
  // Cond
  if (Expr *Cond = Node->getCond())
    Node->setCond(TransformExpr(Cond));

  // Body
  if (Stmt *Body = Node->getBody()) {
    CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
    if (!CS) CS = ConvertToCompoundStmt(Body);
    Node->setBody(TransformRawCompoundStmt(CS));
  }

  return Node;
}

Stmt *FunctionInliner::VisitCaseStmt(CaseStmt *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  if (Node->getRHS()) {
    Node->setRHS(TransformExpr(Node->getRHS()));
  }

  if (Stmt *SubStmt = Node->getSubStmt()) {
    Node->setSubStmt(TransformStmt(SubStmt));
  }
  return Node;
}

Stmt *FunctionInliner::VisitDefaultStmt(DefaultStmt *Node) {
  if (Stmt *SubStmt = Node->getSubStmt())
    Node->setSubStmt(TransformStmt(SubStmt));
  return Node;
}

Stmt *FunctionInliner::VisitWhileStmt(WhileStmt *Node) {
  // Cond
  Node->setCond(TransformExpr(Node->getCond()));

  // Body
  Stmt *Body = Node->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS) CS = ConvertToCompoundStmt(Body);
  Node->setBody(TransformRawCompoundStmt(CS));

  return Node;
}

Stmt *FunctionInliner::VisitDoStmt(DoStmt *Node) {
  // Body
  Stmt *Body = Node->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS) CS = ConvertToCompoundStmt(Body);
  Node->setBody(TransformRawCompoundStmt(CS));

  // Cond
  Node->setCond(TransformExpr(Node->getCond()));

  return Node;
}

Stmt *FunctionInliner::VisitForStmt(ForStmt *Node) {
  // Init
  if (Stmt *Init = Node->getInit()) {
    if (DeclStmt *DS = dyn_cast<DeclStmt>(Init))
      Node->setInit(TransformRawDeclStmt(DS));
    else
      Node->setInit(TransformExpr(cast<Expr>(Init)));
  }

  // Cond
  if (Expr *Cond = Node->getCond()) {
    Node->setCond(TransformExpr(Cond));
  }

  // Inc
  if (Expr *Inc = Node->getInc()) {
    Node->setInc(TransformExpr(Inc));
  }

  // Body
  Stmt *Body = Node->getBody();
  CompoundStmt *CS = dyn_cast<CompoundStmt>(Body);
  if (!CS) CS = ConvertToCompoundStmt(Body);
  Node->setBody(TransformRawCompoundStmt(CS));

  return Node;
}

Stmt *FunctionInliner::VisitGotoStmt(GotoStmt *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitIndirectGotoStmt(IndirectGotoStmt *Node) {
  Node->setTarget(TransformExpr(Node->getTarget()));
  return Node;
}

Stmt *FunctionInliner::VisitContinueStmt(ContinueStmt *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitBreakStmt(BreakStmt *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitReturnStmt(ReturnStmt *Node) {
  if (Expr *RetExpr = Node->getRetValue()) {
    Node->setRetValue(TransformExpr(RetExpr));
  }
  return Node;
}

Stmt *FunctionInliner::VisitDeclStmt(DeclStmt *Node) {
  return TransformRawDeclStmt(Node);
}



//===--------------------------------------------------------------------===//
// Expr methods.
//===--------------------------------------------------------------------===//
Stmt *FunctionInliner::VisitPredefinedExpr(PredefinedExpr *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitDeclRefExpr(DeclRefExpr *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitIntegerLiteral(IntegerLiteral *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitFloatingLiteral(FloatingLiteral *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitImaginaryLiteral(ImaginaryLiteral *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitStringLiteral(StringLiteral *Str) {
  return Str;
}

Stmt *FunctionInliner::VisitCharacterLiteral(CharacterLiteral *Node) {
  return Node;
}

Stmt *FunctionInliner::VisitParenExpr(ParenExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *FunctionInliner::VisitUnaryOperator(UnaryOperator *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *FunctionInliner::VisitOffsetOfExpr(OffsetOfExpr *Node) {
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

Stmt *FunctionInliner::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *FunctionInliner::VisitVecStepExpr(VecStepExpr *Node) {
  if (!Node->isArgumentType()) {
    Node->setArgument(TransformExpr(Node->getArgumentExpr()));
  }
  return Node;
}

Stmt *FunctionInliner::VisitArraySubscriptExpr(ArraySubscriptExpr *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

/// We have to inline the function that has HasBarrierCall or 
/// HasIndirectBarrierCall set.
Stmt *FunctionInliner::VisitTopLevelCallExpr(CallExpr *Call) {
  for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
    Call->setArg(i, TransformExpr(Call->getArg(i)));
  }

  Expr *Callee = Call->getCallee()->IgnoreParenCasts();
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(Callee)) {
    if (FunctionDecl *FD = dyn_cast<FunctionDecl>(DRE->getDecl())) {
      if (FD->hasBarrierCall() || FD->hasIndirectBarrierCall()) {
        return InlineFunctionCall(Call, FD);
      }
    }
  }

  Call->setCallee(TransformExpr(Callee));

  return Call;
}

Stmt *FunctionInliner::VisitCallExpr(CallExpr *Call) {
  // Since this function is called in other Exprs, it should not return NULL.
  Stmt *Ret = VisitTopLevelCallExpr(Call);
  if (!Ret) Ret = CLExprs.getDummyCallExpr();
  return Ret;
}

Stmt *FunctionInliner::VisitMemberExpr(MemberExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *FunctionInliner::VisitBinaryOperator(BinaryOperator *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *FunctionInliner::VisitCompoundAssignOperator(CompoundAssignOperator *Node) {
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *FunctionInliner::VisitConditionalOperator(ConditionalOperator *Node) {
  Node->setCond(TransformExpr(Node->getCond()));
  Node->setLHS(TransformExpr(Node->getLHS()));
  Node->setRHS(TransformExpr(Node->getRHS()));
  return Node;
}

Stmt *FunctionInliner::VisitImplicitCastExpr(ImplicitCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *FunctionInliner::VisitCStyleCastExpr(CStyleCastExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}

Stmt *FunctionInliner::VisitCompoundLiteralExpr(CompoundLiteralExpr *Node) {
  Node->setInitializer(TransformExpr(Node->getInitializer()));
  return Node;
}

Stmt *FunctionInliner::VisitExtVectorElementExpr(ExtVectorElementExpr *Node) {
  Node->setBase(TransformExpr(Node->getBase()));
  return Node;
}

Stmt *FunctionInliner::VisitInitListExpr(InitListExpr *Node) {
  for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
    if (Node->getInit(i))
      Node->setInit(i, TransformExpr(Node->getInit(i)));
  }
  return Node;
}

Stmt *FunctionInliner::VisitDesignatedInitExpr(DesignatedInitExpr *Node) {
  for (unsigned i = 0, e = Node->getNumSubExprs(); i < e; ++i) {
    if (Node->getSubExpr(i)) {
      Node->setSubExpr(i, TransformExpr(Node->getSubExpr(i)));
    }
  }

  Node->setInit(TransformExpr(Node->getInit()));
  return Node;
}

Stmt *FunctionInliner::VisitParenListExpr(ParenListExpr* Node) {
  for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
    Node->setExpr(i, TransformExpr(Node->getExpr(i)));
  }
  return Node;
}

Stmt *FunctionInliner::VisitVAArgExpr(VAArgExpr *Node) {
  Node->setSubExpr(TransformExpr(Node->getSubExpr()));
  return Node;
}


//---------------------------------------------------------------------------
Stmt *FunctionInliner::InlineFunctionCall(CallExpr *CE, FunctionDecl *FD) {
  // FD may be a FunctionDecl for the prototype, so we need to find the
  // definition of FD.
  FunctionDecl *DefFD = FD->getDefinition();
  assert(DefFD && "No definition?");

  // FIXME: Check if FD has 'return' statements that are not the last stmts
  // in the function body. If true, currently we cannot inline FD's body.
  // We generate an error in this case.

  // If FD has indirect barrier() calls, they should be inlined into FD's body
  // before FD's body is inlined.
  if (DefFD->hasIndirectBarrierCall()) {
    FunctionInliner FIn(ASTCtx, CLExprs, CallMap);
    FIn.Inline(DefFD);
  }

  // If the return type of FD is not void, we decalre a temporary VarDecl to
  // save the return value.
  QualType RetTy = CE->getType();
  if (!RetTy->isVoidType()) {
    RetVD = NewTmpRetVarDecl(RetTy);
    CurBody->push_back(NewDeclStmt(RetVD));
  } else {
    RetVD = NULL;
  }

  // The body of FD is inlined as a CompoundStmt.
  StmtVector Stmts;

  // Setup function arguments.
  ParmVarDeclMapTy ParmVarDeclMap;
  for (unsigned i = 0, e = DefFD->getNumParams(); i < e; i++) {
    // Parameter of the function definition.
    ParmVarDecl *PD = DefFD->getParamDecl(i);
    assert(i < CE->getNumArgs() && "Wrong FunctionDecl?");

    // A new VarDecl is defined to pass the argument.
    VarDecl *NewVD = NewParmVarDecl(PD);
    NewVD->setInit(CE->getArg(i));
    ParmVarDeclMap[PD] = NewVD;

    // Add a DeclStmt for NewVD.
    CurBody->push_back(NewDeclStmt(NewVD));
  }

  // Cody the function body.
  // If the return type of FD is not void, the DeclRefExpr of RetVD is 
  // returned. Otherwise, return a dummay function call expression.
  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(DefFD->getBody());
  assert(FDBody && "Not a CompoundStmt");
  if (RetVD) {
    FunctionDuplicator FDup(ASTCtx, CLExprs, true, RetVD);
    FDup.SetParmVarDeclMap(ParmVarDeclMap);
    CurBody->push_back(FDup.VisitCompoundStmt(FDBody));
    return new (ASTCtx) DeclRefExpr(RetVD, RetTy, VK_RValue, 
                                    SourceLocation());
  } else {
    // void function call
    FunctionDuplicator FDup(ASTCtx, CLExprs, true);
    FDup.SetParmVarDeclMap(ParmVarDeclMap);
    CurBody->push_back(FDup.VisitCompoundStmt(FDBody));
    return NULL;
  }
}


CompoundStmt *FunctionInliner::ConvertToCompoundStmt(Stmt *S) {
  SourceLocation SL;
  Stmt *Stmts[1] = { S };
  return new (ASTCtx) CompoundStmt(ASTCtx, Stmts, 1, SL, SL);
}

VarDecl *FunctionInliner::NewTmpRetVarDecl(QualType T) {
  stringstream Name;
  Name << OPENCL_RET_VAR_PREFIX << CLExprs.getTmpVarNum() << "e";

  string NameStr = Name.str();
  StringRef DeclName(NameStr);
  IdentifierInfo &DeclID = ASTCtx.Idents.get(DeclName);
  TypeSourceInfo *TSI = ASTCtx.getNullTypeSourceInfo();

  return VarDecl::Create(ASTCtx, DeclCtx, SourceLocation(), &DeclID,
                         T, TSI, SC_None, SC_None);
}

VarDecl *FunctionInliner::NewParmVarDecl(ParmVarDecl *PD) {
  stringstream VarNum;
  VarNum << CLExprs.getTmpVarNum() << "e";

  std::string Name(OPENCL_ARG_VAR_PREFIX);
  Name += VarNum.str();
  Name += "_";
  Name += PD->getNameAsString();

  StringRef DeclName(Name);
  IdentifierInfo &DeclID = ASTCtx.Idents.get(DeclName);

  VarDecl *VD = VarDecl::Create(ASTCtx, DeclCtx, SourceLocation(), &DeclID,
                                PD->getType(), PD->getTypeSourceInfo(),
                                PD->getStorageClass(),
                                PD->getStorageClassAsWritten());
  if (PD->hasAttrs()) VD->setAttrs(PD->getAttrs());
  return VD;
}

DeclStmt *FunctionInliner::NewDeclStmt(VarDecl *VD) {
  Decl *Decls[1] = { VD };
  DeclGroupRef DGRef = DeclGroupRef::Create(ASTCtx, Decls, 1);
  return new (ASTCtx) DeclStmt(DGRef, SourceLocation(), SourceLocation());
}

