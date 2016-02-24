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

#include "TransformLICM.h"
#include "CLStmtPrinter.h"
#include "CLUtils.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

//#define CL_DEBUG


void TransformLICM::Transform(FunctionDecl *FD) {
  // 1. Construct a source-level CFG.
  Stmt *Body = FD->getBody();

  CFG::BuildOptions BO;   // CFG build options
  BO.NeedWCL = true;
  CFG *BodyCFG = CFG::buildCFG(NULL, Body, &ASTCtx, BO);
#ifdef CL_DEBUG
  BodyCFG->print(OS, ASTCtx.getLangOptions());
  OS << "\n";
#endif

  // 2. Find VarDecls that are loop-variant.
  FindLoopVariants(FD, *BodyCFG);

  // 3. Perform reaching definitions analysis for whole CFG.
  RD = new ReachingDefinitions(ASTCtx, OS);
  RD->Solve(*BodyCFG, false);

  // 4. Find du-chains
  FindDUChains(*BodyCFG);

  // 5. Loop invariant code motion for each WCR.
  DoLICM(*BodyCFG);

  // 6. Arrange the transformed code.
  RearrangeCode(dyn_cast<CompoundStmt>(Body));

  delete RD;
  delete BodyCFG;

  RD = NULL;
}


//---------------------------------------------------------------------------
// Find VarDecls that are loop-variant.
//---------------------------------------------------------------------------
void TransformLICM::FindLoopVariants(FunctionDecl *FD, CFG &cfg) {
  // Find parameters that are modified in WCR.
  for (unsigned i = 0, e = FD->getNumParams(); i < e; ++i) {
    ParmVarDecl *PD = FD->getParamDecl(i);
    if (PD->isModified()) {
      VariantSet.insert(PD);
    }
  }

  // Save all stmts in all CFGBlocks.
  for (CFG::const_iterator I = cfg.begin(), E = cfg.end(); I != E; ++I ) {
    for (CFGBlock::const_iterator BI = (*I)->begin(), BEnd = (*I)->end();
         BI != BEnd; ++BI) {
      const CFGStmt *SE = BI->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");

      AllStmts.insert(SE->getStmt());
    }
  }

  // Find all loop-variant VarDecls.
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *B = *I;

    // Find VarDecls in each stmt.
    for (CFGBlock::const_iterator BI = B->begin(), BE = B->end(); 
         BI != BE; ++BI) {
      const CFGStmt *SE = BI->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");

      FindLoopVariantsInStmt(SE->getStmt());
    }
  }

#ifdef CL_DEBUG
  printVariantSet();
#endif
}

void TransformLICM::FindLoopVariantsInStmt(Stmt *S) {
  // Visit this Stmt
  switch (S->getStmtClass()) {
    default: assert(0 && "Unknown or unsupported stmt kind!");
             break;

    // statements
    case Stmt::NullStmtClass:
    case Stmt::GotoStmtClass:
    case Stmt::ContinueStmtClass:
    case Stmt::BreakStmtClass:    break;
    case Stmt::ReturnStmtClass: {
      ReturnStmt *Node = static_cast<ReturnStmt*>(S);
      if (Expr *retExpr = Node->getRetValue()) {
        FindLoopVariantsInSubStmt(retExpr);
      }
      break;
    }
    case Stmt::DeclStmtClass: {
      DeclStmt *Node = static_cast<DeclStmt*>(S);
      for (DeclStmt::decl_iterator I = Node->decl_begin(),
                                   E = Node->decl_end();
           I != E; ++I) {
        if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
          if (Expr *init = VD->getInit()) {
            assert(0 && "Was StmtSimplifier executed?");
            if (IsVariantExpr(init)) {
              VariantSet.insert(VD);
            }
          }
        }
      }
      break;
    }

    // expressions
    case Stmt::PredefinedExprClass:   
    case Stmt::IntegerLiteralClass:   
    case Stmt::FloatingLiteralClass:  
    case Stmt::ImaginaryLiteralClass: 
    case Stmt::StringLiteralClass:    
    case Stmt::CharacterLiteralClass: 
    case Stmt::ImplicitValueInitExprClass: break;
    case Stmt::DeclRefExprClass: {
      break;
    }
    case Stmt::CallExprClass: {
      CallExpr *E = static_cast<CallExpr*>(S);
      for (unsigned i = 0, e = E->getNumArgs(); i < e; i++) {
        Expr *arg = E->getArg(i);

        // If the argument type is a pointer type, VarDecl of argument
        // may be variant.
        if (arg->getType()->isPointerType()) {
          VarDecl *VD = FindVarDeclOfExpr(arg);
          if (VD) VariantSet.insert(VD);
        }

        FindLoopVariantsInSubStmt(arg);
      }
      break;
    }
    case Stmt::BinaryOperatorClass: {
      BinaryOperator *E = static_cast<BinaryOperator*>(S);

      // && , || and comma(,) operator are used as a terminator.
      BinaryOperator::Opcode Op = E->getOpcode();
      if (Op == BO_LAnd || Op == BO_LOr || Op == BO_Comma) break;

      Expr *LHS = E->getLHS();
      Expr *RHS = E->getRHS();
      if (Op == BO_Assign) {
        FindLoopVariantsInSubStmt(RHS);
        FindLoopVariantsInSubStmt(LHS);
        if (VarDecl *VD = FindVarDeclOfExpr(LHS)) {
          if (IsVariantExpr(RHS) || IsVariantExpr(LHS)) {
            VariantSet.insert(VD);
          }
        }
      } else {
        FindLoopVariantsInSubStmt(LHS);
        FindLoopVariantsInSubStmt(RHS);
      }

      break;
    }
    case Stmt::CompoundAssignOperatorClass: {
      CompoundAssignOperator *E = static_cast<CompoundAssignOperator*>(S);
      Expr *LHS = E->getLHS();
      Expr *RHS = E->getRHS();

      FindLoopVariantsInSubStmt(RHS);
      FindLoopVariantsInSubStmt(LHS);
      if (VarDecl *VD = FindVarDeclOfExpr(LHS)) {
        if (IsVariantExpr(RHS) || IsVariantExpr(LHS)) {
          VariantSet.insert(VD);
        }
      }
      break;
    }
    case Stmt::UnaryOperatorClass: {
      UnaryOperator *E = static_cast<UnaryOperator*>(S);
      FindLoopVariantsInSubStmt(E->getSubExpr());
      switch (E->getOpcode()) {
        default: break;
        case UO_PostInc:
        case UO_PostDec:
        case UO_PreInc:
        case UO_PreDec: {
          if (VarDecl *VD = FindVarDeclOfExpr(E->getSubExpr())) {
            if (IsVariantExpr(E->getSubExpr()))
              VariantSet.insert(VD);
          }
          break;
        }
        case UO_AddrOf: {
          if (VarDecl *VD = FindVarDeclOfExpr(E->getSubExpr())) {
            VariantSet.insert(VD);
          }
          break;
        }
      }
      break;
    }
    case Stmt::ParenExprClass: {
      ParenExpr *E = static_cast<ParenExpr*>(S);
      FindLoopVariantsInSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::SizeOfAlignOfExprClass: {
      SizeOfAlignOfExpr *E = static_cast<SizeOfAlignOfExpr*>(S);
      if (!E->isArgumentType())
        FindLoopVariantsInSubStmt(E->getArgumentExpr());
      break;
    }
    case Stmt::VecStepExprClass: {
      VecStepExpr *E = static_cast<VecStepExpr*>(S);
      if (!E->isArgumentType())
        FindLoopVariantsInSubStmt(E->getArgumentExpr());
      break;
    }
    case Stmt::ArraySubscriptExprClass: {
      ArraySubscriptExpr *E = static_cast<ArraySubscriptExpr*>(S);
      FindLoopVariantsInSubStmt(E->getLHS());
      FindLoopVariantsInSubStmt(E->getRHS());
      break;
    }
    case Stmt::MemberExprClass: {
      MemberExpr *E = static_cast<MemberExpr*>(S);
      FindLoopVariantsInSubStmt(E->getBase());
      break;
    }
    case Stmt::ConditionalOperatorClass: {
      break;
    }
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr *E = static_cast<ImplicitCastExpr*>(S);
      FindLoopVariantsInSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::CStyleCastExprClass: {
      CStyleCastExpr *E = static_cast<CStyleCastExpr*>(S);
      FindLoopVariantsInSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::CompoundLiteralExprClass: {
      CompoundLiteralExpr *E = static_cast<CompoundLiteralExpr*>(S);
      FindLoopVariantsInSubStmt(E->getInitializer());
      break;
    }
    case Stmt::ExtVectorElementExprClass: {
      ExtVectorElementExpr *E = static_cast<ExtVectorElementExpr*>(S);
      FindLoopVariantsInSubStmt(E->getBase());
      break;
    }
    case Stmt::InitListExprClass: {
      InitListExpr *E = static_cast<InitListExpr*>(S);
      for (unsigned i = 0, e = E->getNumInits(); i < e; i++) {
        if (E->getInit(i)) {
          FindLoopVariantsInSubStmt(E->getInit(i));
        }
      }
      break;
    }
    case Stmt::DesignatedInitExprClass: {
      DesignatedInitExpr *E = static_cast<DesignatedInitExpr*>(S);
      for (unsigned i = 0, e = E->getNumSubExprs(); i < e; ++i) {
        if (E->getSubExpr(i)) {
          FindLoopVariantsInSubStmt(E->getSubExpr(i));
        }
      }
      FindLoopVariantsInSubStmt(E->getInit());
      break;
    }
    case Stmt::ParenListExprClass: {
      ParenListExpr *E = static_cast<ParenListExpr*>(S);
      for (unsigned i = 0, e = E->getNumExprs(); i != e; i++) {
        FindLoopVariantsInSubStmt(E->getExpr(i));
      }
      break;
    }
    case Stmt::VAArgExprClass: {
      VAArgExpr *E = static_cast<VAArgExpr*>(S);
      FindLoopVariantsInSubStmt(E->getSubExpr());
      break;
    }
  } // end switch
}

void TransformLICM::FindLoopVariantsInSubStmt(Stmt *S) {
  // Only when S does not appear in the top level of CFGBlocks, 
  // invoke FindLoopVariatnsInStmt().
  if (AllStmts.find(S) == AllStmts.end()) {
    FindLoopVariantsInStmt(S);
  }
}

/// If E is WCL variant, return true.
/// Otherwise, return false.
bool TransformLICM::IsVariantExpr(Expr *S) {
  // Visit this Expr
  switch (S->getStmtClass()) {
    default: assert(0 && "Unknown or unsupported expr kind!");
             break;

    case Stmt::PredefinedExprClass:   
    case Stmt::IntegerLiteralClass:   
    case Stmt::FloatingLiteralClass:  
    case Stmt::ImaginaryLiteralClass: 
    case Stmt::StringLiteralClass:    
    case Stmt::CharacterLiteralClass: 
    case Stmt::ImplicitValueInitExprClass: return false;
    case Stmt::DeclRefExprClass: {
      DeclRefExpr *E = static_cast<DeclRefExpr*>(S);
      if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
        if (VariantSet.find(VD) != VariantSet.end())
          return true;
      }
      return false;
    }
    case Stmt::CallExprClass: {
      CallExpr *E = static_cast<CallExpr*>(S);
      bool rst = false;
      for (unsigned i = 0, e = E->getNumArgs(); i < e; i++) {
        Expr *arg = E->getArg(i);

        // If the argument type is a pointer type, VarDecl of argument
        // may be variant.
        if (arg->getType()->isPointerType()) {
          VarDecl *VD = FindVarDeclOfExpr(arg);
          if (VD) VariantSet.insert(VD);
        }

        rst = IsVariantExpr(arg) || rst;
      }
      if (!rst) {
        Expr *Callee = E->getCallee();
        if (CLUtils::IsWorkItemIDFunction(Callee)) return true;
        if (CLUtils::IsBuiltinFunction(Callee))    return false;
        if (CLUtils::IsInvariantFunction(Callee))  return false;
      }
      return true;
    }
    case Stmt::BinaryOperatorClass: {
      BinaryOperator *E = static_cast<BinaryOperator*>(S);

      // && , || and comma(,) operator are used as a terminator.
      BinaryOperator::Opcode Op = E->getOpcode();
      if (Op == BO_LAnd || Op == BO_LOr || Op == BO_Comma) return true;

      bool rst;
      Expr *LHS = E->getLHS();
      Expr *RHS = E->getRHS();
      if (Op == BO_Assign) {
        rst = IsVariantExpr(RHS);
        rst = IsVariantExpr(LHS) || rst;
        if (VarDecl *VD = FindVarDeclOfExpr(LHS)) {
          if (rst) VariantSet.insert(VD);
        }
      } else {
        rst = IsVariantExpr(E->getLHS());
        rst = IsVariantExpr(E->getRHS()) || rst;
      }

      return rst;
    }
    case Stmt::CompoundAssignOperatorClass: {
      CompoundAssignOperator *E = static_cast<CompoundAssignOperator*>(S);
      bool rst;
      rst = IsVariantExpr(E->getRHS());
      rst = IsVariantExpr(E->getLHS()) || rst;
      if (VarDecl *VD = FindVarDeclOfExpr(E->getLHS())) {
        if (rst) VariantSet.insert(VD);
      }
      return rst;
    }
    case Stmt::UnaryOperatorClass: {
      UnaryOperator *E = static_cast<UnaryOperator*>(S);
      bool rst = IsVariantExpr(E->getSubExpr());
      switch (E->getOpcode()) {
        default: break;
        case UO_PostInc:
        case UO_PostDec:
        case UO_PreInc:
        case UO_PreDec: {
          if (VarDecl *VD = FindVarDeclOfExpr(E->getSubExpr())) {
            if (rst) VariantSet.insert(VD);
          }
        }
      }
      return rst;
    }
    case Stmt::ParenExprClass: {
      ParenExpr *E = static_cast<ParenExpr*>(S);
      return IsVariantExpr(E->getSubExpr());
    }
    case Stmt::SizeOfAlignOfExprClass: {
      SizeOfAlignOfExpr *E = static_cast<SizeOfAlignOfExpr*>(S);
      if (!E->isArgumentType())
        return IsVariantExpr(E->getArgumentExpr());
      return false;
    }
    case Stmt::VecStepExprClass: {
      VecStepExpr *E = static_cast<VecStepExpr*>(S);
      if (!E->isArgumentType())
        return IsVariantExpr(E->getArgumentExpr());
      return false;
    }
    case Stmt::ArraySubscriptExprClass: {
      ArraySubscriptExpr *E = static_cast<ArraySubscriptExpr*>(S);
      bool rst;
      rst = IsVariantExpr(E->getLHS());
      rst = IsVariantExpr(E->getRHS()) || rst;
      return rst;
    }
    case Stmt::MemberExprClass: {
      MemberExpr *E = static_cast<MemberExpr*>(S);
      return IsVariantExpr(E->getBase());
    }
    case Stmt::ConditionalOperatorClass: {
      return true;
    }
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr *E = static_cast<ImplicitCastExpr*>(S);
      return IsVariantExpr(E->getSubExpr());
    }
    case Stmt::CStyleCastExprClass: {
      CStyleCastExpr *E = static_cast<CStyleCastExpr*>(S);
      return IsVariantExpr(E->getSubExpr());
    }
    case Stmt::CompoundLiteralExprClass: {
      CompoundLiteralExpr *E = static_cast<CompoundLiteralExpr*>(S);
      return IsVariantExpr(E->getInitializer());
    }
    case Stmt::ExtVectorElementExprClass: {
      ExtVectorElementExpr *E = static_cast<ExtVectorElementExpr*>(S);
      return IsVariantExpr(E->getBase());
    }
    case Stmt::InitListExprClass: {
      InitListExpr *E = static_cast<InitListExpr*>(S);
      bool rst = false;
      for (unsigned i = 0, e = E->getNumInits(); i < e; i++) {
        if (E->getInit(i)) {
          rst = IsVariantExpr(E->getInit(i)) || rst;
        }
      }
      return rst;
    }
    case Stmt::DesignatedInitExprClass: {
      DesignatedInitExpr *E = static_cast<DesignatedInitExpr*>(S);
      bool rst = false;
      for (unsigned i = 0, e = E->getNumSubExprs(); i < e; ++i) {
        if (E->getSubExpr(i)) {
          rst = IsVariantExpr(E->getSubExpr(i)) || rst;
        }
      }
      rst = IsVariantExpr(E->getInit()) || rst;
      return rst;
    }
    case Stmt::ParenListExprClass: {
      ParenListExpr *E = static_cast<ParenListExpr*>(S);
      bool rst = false;
      for (unsigned i = 0, e = E->getNumExprs(); i != e; i++) {
        rst = IsVariantExpr(E->getExpr(i)) || rst;
      }
      return rst;
    }
    case Stmt::VAArgExprClass: {
      VAArgExpr *E = static_cast<VAArgExpr*>(S);
      return IsVariantExpr(E->getSubExpr());
    }
  } // end switch
}

VarDecl *TransformLICM::FindVarDeclOfExpr(Expr *E) {
  VarDecl *VD = 0;

  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    VD = dyn_cast<VarDecl>(DRE->getDecl());
  } else if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    VD = FindVarDeclOfExpr(ASE->getLHS());
  } else if (MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
    VD = FindVarDeclOfExpr(ME->getBase());
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    VD = FindVarDeclOfExpr(ICE->getSubExpr());
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    VD = FindVarDeclOfExpr(PE->getSubExpr());
  }

  return VD;
}


//---------------------------------------------------------------------------
// DU-Chains
//---------------------------------------------------------------------------
void TransformLICM::FindDUChains(CFG &cfg) {
  // Iterate through the CFGBlocks. 
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *B = *I;

    // Skip the entry block and the exit block.
    if (B == &cfg.getEntry() || B == &cfg.getExit())
      continue;

    UIntStmtsMapTy *REACHin = RD->getREACHinData(B);
    if (!REACHin) continue;

    // Initialize CurDefMap using REACHin of the CFGBlock.
    CurDefMap.clear();
    for (UIntStmtsMapTy::iterator SI = REACHin->begin(), SE = REACHin->end();
         SI != SE; ++SI) {
      unsigned pos    = (*SI).first;
      StmtSetTy &defs = (*SI).second;
      VarDecl *VD   = RD->getVarDecl(pos);
      CurDefMap[VD] = defs;
    }

    // Skip non-WCR CFGBlock.
    if (B->getWCRID() == -1) continue;

    // Iterate through the statements in the CFGBlock.
    for (CFGBlock::const_iterator BI = B->begin(), BE = B->end(); 
         BI != BE; ++BI) {
      const CFGStmt *SE = BI->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");

      // Set the current stmt.
      CurStmt = SE->getStmt();

      // find du-chains
      FindDUChainsInStmt(CurStmt);
    }
  }

#ifdef CL_DEBUG
  printDUChains(cfg);
#endif
}

void TransformLICM::FindDUChainsInStmt(Stmt *S) {
  // S is a non-terminator statement.
  switch (S->getStmtClass()) {
    default: assert(0 && "Unknown or unsupported expr kind!");
             break;

    // statements
    case Stmt::NullStmtClass:
    case Stmt::GotoStmtClass:
    case Stmt::ContinueStmtClass:
    case Stmt::BreakStmtClass:    break;
    case Stmt::ReturnStmtClass: {
      ReturnStmt *Node = static_cast<ReturnStmt*>(S);
      if (Expr *retExpr = Node->getRetValue()) {
        FindDUChainsInSubStmt(retExpr);
      }
      break;
    }
    case Stmt::DeclStmtClass: {
      DeclStmt *Node = static_cast<DeclStmt*>(S);
      DeclStmt::decl_iterator I, E;
      for (I = Node->decl_begin(), E = Node->decl_end(); I != E; ++I) {
        if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
          if (VD->getInit()) {
            assert(0 && "Was StmtSimplifier executed?");
            InsertDefIntoCurDefMap(VD);
          }
        }
      }
      break;
    }

    case Stmt::PredefinedExprClass:   
    case Stmt::IntegerLiteralClass:   
    case Stmt::FloatingLiteralClass:  
    case Stmt::ImaginaryLiteralClass: 
    case Stmt::StringLiteralClass:    
    case Stmt::CharacterLiteralClass: 
    case Stmt::ImplicitValueInitExprClass: break;
    case Stmt::DeclRefExprClass: {
      DeclRefExpr *E = static_cast<DeclRefExpr*>(S);
      if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
        // If definitions of this VarDecl exist, make du-chains.
        if (CurDefMap.find(VD) != CurDefMap.end()) {
          // If VarDeclDUChains does not have an entry for this VarDecl,
          // make an entry.
          DUChainsTy &chainMap = VarDeclDUChains[VD];

          // Make a du-chain for each definition.
          StmtSetTy &defs = CurDefMap[VD];
          for (StmtSetTy::iterator I = defs.begin(), IE = defs.end();
               I != IE; ++I) {
            Stmt *def = *I;
            chainMap[def].insert(CurStmt);
          } // end for
        }
      }
      break;
    }
    case Stmt::BinaryOperatorClass: {
      BinaryOperator *E = static_cast<BinaryOperator*>(S);

      // && , || and comma(,) operator are used as a terminator.
      BinaryOperator::Opcode Op = E->getOpcode();
      if (Op == BO_LAnd || Op == BO_LOr || Op == BO_Comma) break;

      Expr *LHS = E->getLHS();
      Expr *RHS = E->getRHS();
      if (Op == BO_Assign) {
        FindDUChainsInSubStmt(RHS);
        VarDecl *VD = GetDefinedVarDecl(LHS);
        if (!VD) VD = GetDefinedPointerVarDecl(LHS);
        if (VD)  InsertDefIntoCurDefMap(VD);
      } else {
        FindDUChainsInSubStmt(LHS);
        FindDUChainsInSubStmt(RHS);
      }
      break;
    }
    case Stmt::CompoundAssignOperatorClass: {
      CompoundAssignOperator *E = static_cast<CompoundAssignOperator*>(S);
      FindDUChainsInSubStmt(E->getRHS());
      FindDUChainsInSubStmt(E->getLHS());
      VarDecl *VD = GetDefinedVarDecl(E->getLHS());
      if (!VD) VD = GetDefinedPointerVarDecl(E->getLHS());
      if (VD)  InsertDefIntoCurDefMap(VD);
      break;
    }
    case Stmt::UnaryOperatorClass: {
      UnaryOperator *E = static_cast<UnaryOperator*>(S);
      // Because we cannot divide def and use in UnaryOperator,
      // we do not add SubExpr as a definition.
      FindDUChainsInSubStmt(E->getSubExpr());
      switch (E->getOpcode()) {
        default: break;
        case UO_PostInc:
        case UO_PostDec:
        case UO_PreInc:
        case UO_PreDec: {
          VarDecl *VD = GetDefinedVarDecl(E->getSubExpr());
          if (!VD) VD = GetDefinedPointerVarDecl(E->getSubExpr());
          if (VD)  InsertDefIntoCurDefMap(VD);
          break;
        }
      }
      break;
    }

    case Stmt::CallExprClass: {
      CallExpr *E = static_cast<CallExpr*>(S);
      for (unsigned i = 0, e = E->getNumArgs(); i < e; i++) {
        FindDUChainsInSubStmt(E->getArg(i));
      }
      break;
    }
    case Stmt::ParenExprClass: {
      ParenExpr *E = static_cast<ParenExpr*>(S);
      FindDUChainsInSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::SizeOfAlignOfExprClass: {
      SizeOfAlignOfExpr *E = static_cast<SizeOfAlignOfExpr*>(S);
      if (!E->isArgumentType())
        FindDUChainsInSubStmt(E->getArgumentExpr());
      break;
    }
    case Stmt::VecStepExprClass: {
      VecStepExpr *E = static_cast<VecStepExpr*>(S);
      if (!E->isArgumentType())
        FindDUChainsInSubStmt(E->getArgumentExpr());
      break;
    }
    case Stmt::ArraySubscriptExprClass: {
      ArraySubscriptExpr *E = static_cast<ArraySubscriptExpr*>(S);
      FindDUChainsInSubStmt(E->getRHS());
      FindDUChainsInSubStmt(E->getLHS());
      break;
    }
    case Stmt::MemberExprClass: {
      MemberExpr *E = static_cast<MemberExpr*>(S);
      FindDUChainsInSubStmt(E->getBase());
      break;
    }
    case Stmt::ConditionalOperatorClass: {
      // ConditionalOperator is a terminator.
      break;
    }
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr *E = static_cast<ImplicitCastExpr*>(S);
      FindDUChainsInSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::CStyleCastExprClass: {
      CStyleCastExpr *E = static_cast<CStyleCastExpr*>(S);
      FindDUChainsInSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::CompoundLiteralExprClass: {
      CompoundLiteralExpr *E = static_cast<CompoundLiteralExpr*>(S);
      FindDUChainsInSubStmt(E->getInitializer());
      break;
    }
    case Stmt::ExtVectorElementExprClass: {
      ExtVectorElementExpr *E = static_cast<ExtVectorElementExpr*>(S);
      FindDUChainsInSubStmt(E->getBase());
      break;
    }
    case Stmt::InitListExprClass: {
      InitListExpr *E = static_cast<InitListExpr*>(S);
      for (unsigned i = 0, e = E->getNumInits(); i < e; i++) {
        if (E->getInit(i)) FindDUChainsInSubStmt(E->getInit(i));
      }
      break;
    }
    case Stmt::DesignatedInitExprClass: {
      DesignatedInitExpr *E = static_cast<DesignatedInitExpr*>(S);
      for (unsigned i = 0, e = E->getNumSubExprs(); i < e; ++i) {
        if (E->getSubExpr(i)) FindDUChainsInSubStmt(E->getSubExpr(i));
      }
      FindDUChainsInSubStmt(E->getInit());
      break;
    }
    case Stmt::ParenListExprClass: {
      ParenListExpr *E = static_cast<ParenListExpr*>(S);
      for (unsigned i = 0, e = E->getNumExprs(); i != e; i++) {
        FindDUChainsInSubStmt(E->getExpr(i));
      }
      break;
    }
    case Stmt::VAArgExprClass: {
      VAArgExpr *E = static_cast<VAArgExpr*>(S);
      FindDUChainsInSubStmt(E->getSubExpr());
      break;
    }
  } // end switch
}

void TransformLICM::FindDUChainsInSubStmt(Stmt *S) {
  // Only when S does not appear in the top level of CFGBlocks, 
  // invoke FindDUChainsInStmt().
  if (AllStmts.find(S) == AllStmts.end()) {
    FindDUChainsInStmt(S);
  }
}

void TransformLICM::InsertDefIntoCurDefMap(VarDecl *VD) {
  // Insert this VarDecl into CurDefMap.
  // Because this definition kills the previous definition, the previous
  //  entry of CurDefMap is overwritten.
  StmtSetTy defSet;
  defSet.insert(CurStmt);
  CurDefMap[VD] = defSet;
}


/// Find really defined VarDecl of assignment LHS.
VarDecl *TransformLICM::GetDefinedVarDecl(Expr *E, bool wrMode) {
  VarDecl *VD = 0;

  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    if (wrMode) {
      VD = dyn_cast<VarDecl>(DRE->getDecl());
    } else {
      FindDUChainsInStmt(DRE);
    }
  } else if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    GetDefinedVarDecl(ASE->getRHS(), false);
    VD = GetDefinedVarDecl(ASE->getLHS(), wrMode);
  } else if (MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
    VD = GetDefinedVarDecl(ME->getBase(), wrMode);
  } else if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E)) {
    FindDUChainsInStmt(UO);
  } else if (CompoundAssignOperator *CAO = dyn_cast<CompoundAssignOperator>(E)) {
    FindDUChainsInSubStmt(CAO);
  } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(E)) {
    FindDUChainsInSubStmt(BinOp);
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    VD = GetDefinedVarDecl(ICE->getSubExpr(), wrMode);
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    VD = GetDefinedVarDecl(PE->getSubExpr(), wrMode);
  }

  return VD;
}

/// Find VarDecl with pointer type
VarDecl *TransformLICM::GetDefinedPointerVarDecl(Expr *E) {
  VarDecl *VD = 0;
  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    VD = dyn_cast<VarDecl>(DRE->getDecl());
    if (VD) {
      QualType VDTy = VD->getType();
      if (!VDTy->isPointerType()) VD = 0;
    }
  }
  return VD;
}


//---------------------------------------------------------------------------
// Loop Invariant Code Motion
//---------------------------------------------------------------------------
#define UNKNOWN     0
#define INVARIANT   1
#define VARIANT     2

void TransformLICM::DoLICM(CFG &cfg) {
  // Find CFGBlocks of each WCR.
  FindCFGBlocksOfWCR(cfg);

  for (WCRBlksMapTy::iterator I = WCRBlksMap.begin(), E = WCRBlksMap.end();
       I != E; ++I) {
    int wcrID = (*I).first;
    CurWCRID = wcrID;

    // If this WCR is a landing pad, skip this WCR.
    CompoundStmt *wcr = WCR.getWCR(wcrID);
    if (WCR.isLandPad(wcr)) continue;

    // Find invariant code and mark it.
    StmtInvMap.clear();
    VarDeclInvMap.clear();
    for (VarDeclSetTy::iterator VI = VariantSet.begin(), 
                                VE = VariantSet.end();
         VI != VE; ++VI) {
      VarDeclInvMap[*VI] = VARIANT;
    }
    FindInvariantCode(wcrID, cfg);

    // Invariant code motion.
    StmtVector invBody;
    StmtVector wcrBody;
    for (CompoundStmt::body_iterator BI = wcr->body_begin(),
         BE = wcr->body_end(); BI != BE; ++BI) {
      Stmt *S = *BI;
      if (IsWCLInvariant(S)) {
        invBody.push_back(S);
      } else {
        wcrBody.push_back(S);
      }
    }

    if (invBody.size() > 0) {
      CompoundStmt *invCS = CLExprs.NewCompoundStmt(invBody);
      invCS->setIsStmtList(true);

      CompoundStmt *wcrCS = CLExprs.NewCompoundStmt(wcrBody);
      wcrCS->setWCR(wcrID);
      WCR.setWCR(wcrID, wcrCS);

      // Unset WCRID of original WCR
      wcr->setWCR(-1);
      Stmt *stmts[2] = { invCS, wcrCS };
      wcr->setStmts(ASTCtx, stmts, 2);
      wcr->setIsStmtList(true);
    }
  }
}

void TransformLICM::FindCFGBlocksOfWCR(CFG &cfg) {
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *B = *I;
    int wcrID = B->getWCRID();
    if (wcrID != -1) {
      // CFGBlocks consisting of WCR
      if (WCRBlksMap.find(wcrID) == WCRBlksMap.end()) {
        WCRBlksMap[wcrID] = BlkSetTy();
      }
      BlkSetTy &blks = WCRBlksMap[wcrID];
      blks.insert(B);
    }
  }

#ifdef CL_DEBUG
  printWCRBlksMap();
#endif
}

void TransformLICM::FindInvariantCode(int wcrID, CFG &cfg) {
  bool allDetermined = false;
  while (!allDetermined) {
    allDetermined = true;
    BlkSetTy &blks = WCRBlksMap[wcrID];
    for (BlkSetTy::iterator I = blks.begin(), E = blks.end(); I != E; ++I) {
      CFGBlock *B = *I;
      if (B == &cfg.getEntry() || B == &cfg.getExit()) continue;

      // Set the current CFGBlock.
      CurCFGBlk = B;
      for (CFGBlock::iterator BI = B->begin(), BE = B->end();
           BI != BE; ++BI) {
        const CFGStmt *SE = BI->getAs<CFGStmt>();
        assert(SE && "Which CFGElement?");

        // Set the current Stmt.
        CurStmt = SE->getStmt();
        if (StmtInvMap.find(CurStmt) != StmtInvMap.end()) continue;
        
        unsigned rst = DetermineInvariance(CurStmt);
        if (rst == UNKNOWN)
          allDetermined = false;
        else if (rst == INVARIANT)
          StmtInvMap[CurStmt] = true;
        else
          StmtInvMap[CurStmt] = false;
      }
    }

    if (!allDetermined) {
      for (VarDeclUIntMapTy::iterator I = VarDeclInvMap.begin(),
                                      E = VarDeclInvMap.end();
           I != E; ++I) {
        VarDecl *VD = (*I).first;
        unsigned inv = (*I).second;
        if (inv == UNKNOWN) VarDeclInvMap[VD] = INVARIANT;
      }
    }
  } //end while
}


unsigned TransformLICM::DetermineInvariance(Stmt *S) {
  // If there is an entry for Stmt S, return the invariance.
  if (StmtInvMap.find(S) != StmtInvMap.end()) {
    if (StmtInvMap[S]) return INVARIANT;
    else return VARIANT;
  }

  // Visit this Stmt
  switch (S->getStmtClass()) {
    default: assert(0 && "Unknown or unsupported stmt kind!");
             break;

    // statements
    case Stmt::NullStmtClass:     return INVARIANT;
    case Stmt::GotoStmtClass:
    case Stmt::ContinueStmtClass:
    case Stmt::BreakStmtClass:    return VARIANT;
    case Stmt::ReturnStmtClass: {
      ReturnStmt *Node = static_cast<ReturnStmt*>(S);
      if (Expr *retExpr = Node->getRetValue()) {
        DetermineInvariance(retExpr);
      }
      return VARIANT;
    }
    case Stmt::DeclStmtClass: {
      DeclStmt *Node = static_cast<DeclStmt*>(S);
      for (DeclStmt::decl_iterator I = Node->decl_begin(),
                                   E = Node->decl_end();
           I != E; ++I) {
        if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
          if (Expr *init = VD->getInit()) {
            assert(0 && "Was StmtSimplifier executed?");
            unsigned rst = DetermineInvariance(init);

            // If there is other reaching definition for this VarDecl 
            //   and there is a use of this definition,
            //   this VarDecl is variant.
            if (HasOtherReachingDef(VD) && HasReachableUse(VD)) {
              VarDeclInvMap[VD] = VARIANT;
              return VARIANT;
            }

            if (VarDeclInvMap.find(VD) != VarDeclInvMap.end()) {
              if (VarDeclInvMap[VD] == VARIANT)
                return VARIANT;
            }

            VarDeclInvMap[VD] = rst;
            return rst;
          } else {
            return INVARIANT;
          }
        }
      }
      return VARIANT;
    }

    // expressions
    case Stmt::PredefinedExprClass:   
    case Stmt::IntegerLiteralClass:   
    case Stmt::FloatingLiteralClass:  
    case Stmt::ImaginaryLiteralClass: 
    case Stmt::StringLiteralClass:    
    case Stmt::CharacterLiteralClass: 
    case Stmt::ImplicitValueInitExprClass: return INVARIANT;
    case Stmt::DeclRefExprClass: {
      DeclRefExpr *E = static_cast<DeclRefExpr*>(S);
      if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
        if (VarDeclInvMap.find(VD) == VarDeclInvMap.end()) {
          VarDeclInvMap[VD] = UNKNOWN;
        }
        return VarDeclInvMap[VD];
      }
      return VARIANT;
    }
    case Stmt::CallExprClass: {
      CallExpr *E = static_cast<CallExpr*>(S);
      unsigned rst = INVARIANT;
      for (unsigned i = 0, e = E->getNumArgs(); i < e; i++) {
        unsigned tmp = DetermineInvariance(E->getArg(i));
        if (rst == VARIANT || tmp == VARIANT) rst = VARIANT;
        else if (tmp == UNKNOWN) rst = UNKNOWN;
      }
      if (rst == INVARIANT) {
        Expr *Callee = E->getCallee();
        if (CLUtils::IsWorkItemIDFunction(Callee)) return VARIANT;
        if (CLUtils::IsBuiltinFunction(Callee))    return INVARIANT;
        if (CLUtils::IsInvariantFunction(Callee))  return INVARIANT;
        return VARIANT;
      }
      return rst;
    }
    case Stmt::BinaryOperatorClass: {
      BinaryOperator *E = static_cast<BinaryOperator*>(S);

      Expr *LHS = E->getLHS();
      Expr *RHS = E->getRHS();

      // && , || and comma(,) operator are used as a terminator.
      if (E->getOpcode() == BO_Assign) {
        ///////
        unsigned invRHS = DetermineInvariance(RHS);
        unsigned invLHS = DetermineInvariance(LHS);

        // Find VarDecl of LHS
        if (VarDecl *VD = FindVarDeclOfExpr(LHS)) {
          if (HasOtherReachingDef(VD) && HasReachableUse(VD)) {
            VarDeclInvMap[VD] = VARIANT;
            return VARIANT;
          }
          
          if (invRHS == VARIANT || invLHS == VARIANT) {
            VarDeclInvMap[VD] = VARIANT;
            return VARIANT;
          } else if (invRHS == UNKNOWN || invLHS == UNKNOWN) {
            VarDeclInvMap[VD] = UNKNOWN;
            return UNKNOWN;
          } else {
            VarDeclInvMap[VD] = INVARIANT;
            return INVARIANT;
          }
        }

        if (invLHS == VARIANT || invRHS == VARIANT) return VARIANT;
        if (invLHS == UNKNOWN || invRHS == UNKNOWN) return UNKNOWN;
        return INVARIANT;
      } else {
        unsigned invLHS = DetermineInvariance(E->getLHS());
        unsigned invRHS = DetermineInvariance(E->getRHS());
        if (invLHS == VARIANT || invRHS == VARIANT) return VARIANT;
        if (invLHS == UNKNOWN || invRHS == UNKNOWN) return UNKNOWN;
        return INVARIANT;
      }

      return VARIANT;
    }
    case Stmt::CompoundAssignOperatorClass: {
      ////////
      CompoundAssignOperator *E = static_cast<CompoundAssignOperator*>(S);
      unsigned invRHS = DetermineInvariance(E->getRHS());
      unsigned invLHS = DetermineInvariance(E->getLHS());

      // Find VarDecl of LHS
      if (VarDecl *VD = FindVarDeclOfExpr(E->getLHS())) {
        if (HasOtherReachingDef(VD) && HasReachableUse(VD)) {
          VarDeclInvMap[VD] = VARIANT;
          return VARIANT;
        }

        if (invRHS == VARIANT || invLHS == VARIANT) {
          VarDeclInvMap[VD] = VARIANT;
          return VARIANT;
        } else if (invRHS == UNKNOWN || invLHS == UNKNOWN) {
          VarDeclInvMap[VD] = UNKNOWN;
          return UNKNOWN;
        } else {
          VarDeclInvMap[VD] = INVARIANT;
          return INVARIANT;
        }
      }
      if (invLHS == VARIANT || invRHS == VARIANT) return VARIANT;
      if (invLHS == UNKNOWN || invRHS == UNKNOWN) return UNKNOWN;
      return INVARIANT;
    }
    case Stmt::UnaryOperatorClass: {
      UnaryOperator *E = static_cast<UnaryOperator*>(S);
      unsigned rst = DetermineInvariance(E->getSubExpr());
      switch (E->getOpcode()) {
        default: break;
        case UO_PostInc:
        case UO_PostDec:
        case UO_PreInc:
        case UO_PreDec: {
          ////////////
          if (VarDecl *VD = FindVarDeclOfExpr(E->getSubExpr())) {
            VarDeclInvMap[VD] = VARIANT;
            if (HasOtherReachingDef(VD) && HasReachableUse(VD)) {
              VarDeclInvMap[VD] = VARIANT;
              return VARIANT;
            }
            VarDeclInvMap[VD] = rst;
          }
        }
      }
      return rst;
    }
    case Stmt::ParenExprClass: {
      ParenExpr *E = static_cast<ParenExpr*>(S);
      return DetermineInvariance(E->getSubExpr());
    }
    case Stmt::SizeOfAlignOfExprClass: {
      SizeOfAlignOfExpr *E = static_cast<SizeOfAlignOfExpr*>(S);
      if (!E->isArgumentType())
        return DetermineInvariance(E->getArgumentExpr());
      return INVARIANT;
    }
    case Stmt::VecStepExprClass: {
      VecStepExpr *E = static_cast<VecStepExpr*>(S);
      if (!E->isArgumentType())
        return DetermineInvariance(E->getArgumentExpr());
      return INVARIANT;
    }
    case Stmt::ArraySubscriptExprClass: {
      ArraySubscriptExpr *E = static_cast<ArraySubscriptExpr*>(S);
      unsigned invLHS = DetermineInvariance(E->getLHS());
      unsigned invRHS = DetermineInvariance(E->getRHS());
      if (invLHS == VARIANT || invRHS == VARIANT) return VARIANT;
      if (invLHS == UNKNOWN || invRHS == UNKNOWN) return UNKNOWN;
      return INVARIANT;
    }
    case Stmt::MemberExprClass: {
      MemberExpr *E = static_cast<MemberExpr*>(S);
      return DetermineInvariance(E->getBase());
    }
    case Stmt::ConditionalOperatorClass: {
      ConditionalOperator *E = static_cast<ConditionalOperator*>(S);
      unsigned invCond = DetermineInvariance(E->getCond());
      unsigned invLHS  = DetermineInvariance(E->getLHS());
      unsigned invRHS  = DetermineInvariance(E->getRHS());
      if (invCond == VARIANT || invLHS == VARIANT || invRHS == VARIANT)
        return VARIANT;
      if (invCond == UNKNOWN || invLHS == UNKNOWN || invRHS == UNKNOWN)
        return UNKNOWN;
      return INVARIANT;
    }
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr *E = static_cast<ImplicitCastExpr*>(S);
      return DetermineInvariance(E->getSubExpr());
    }
    case Stmt::CStyleCastExprClass: {
      CStyleCastExpr *E = static_cast<CStyleCastExpr*>(S);
      return DetermineInvariance(E->getSubExpr());
    }
    case Stmt::CompoundLiteralExprClass: {
      CompoundLiteralExpr *E = static_cast<CompoundLiteralExpr*>(S);
      return DetermineInvariance(E->getInitializer());
    }
    case Stmt::ExtVectorElementExprClass: {
      ExtVectorElementExpr *E = static_cast<ExtVectorElementExpr*>(S);
      return DetermineInvariance(E->getBase());
    }
    case Stmt::InitListExprClass: {
      InitListExpr *E = static_cast<InitListExpr*>(S);
      unsigned rst = INVARIANT;
      for (unsigned i = 0, e = E->getNumInits(); i < e; i++) {
        if (E->getInit(i)) {
          unsigned tmp = DetermineInvariance(E->getInit(i));
          if (rst == VARIANT || tmp == VARIANT)      rst = VARIANT;
          else if (rst == UNKNOWN || tmp == UNKNOWN) rst = UNKNOWN;
          else rst = INVARIANT;
        }
      }
      return rst;
    }
    case Stmt::DesignatedInitExprClass: {
      DesignatedInitExpr *E = static_cast<DesignatedInitExpr*>(S);
      unsigned rst = INVARIANT;
      for (unsigned i = 0, e = E->getNumSubExprs(); i < e; ++i) {
        if (E->getSubExpr(i)) {
          unsigned tmp = DetermineInvariance(E->getSubExpr(i));
          if (rst == VARIANT || tmp == VARIANT)      rst = VARIANT;
          else if (rst == UNKNOWN || tmp == UNKNOWN) rst = UNKNOWN;
          else rst = INVARIANT;
        }
      }
      unsigned tmp = DetermineInvariance(E->getInit());
      if (rst == VARIANT || tmp == VARIANT)      rst = VARIANT;
      else if (rst == UNKNOWN || tmp == UNKNOWN) rst = UNKNOWN;
      else rst = INVARIANT;
      return rst;
    }
    case Stmt::ParenListExprClass: {
      ParenListExpr *E = static_cast<ParenListExpr*>(S);
      unsigned rst = INVARIANT;
      for (unsigned i = 0, e = E->getNumExprs(); i != e; i++) {
        unsigned tmp = DetermineInvariance(E->getExpr(i));
        if (rst == VARIANT || tmp == VARIANT)      rst = VARIANT;
        else if (rst == UNKNOWN || tmp == UNKNOWN) rst = UNKNOWN;
        else rst = INVARIANT;
      }
      return rst;
    }
    case Stmt::VAArgExprClass: {
      VAArgExpr *E = static_cast<VAArgExpr*>(S);
      return DetermineInvariance(E->getSubExpr());
    }
  } // end switch

  return UNKNOWN;
}

bool TransformLICM::IsWCLInvariant(Stmt *S) {
  // If there is an entry for Stmt S, return the invariance.
  if (StmtInvMap.find(S) != StmtInvMap.end())
    return StmtInvMap[S];

  // Visit this statement.
  switch (S->getStmtClass()) {
  default: assert(0 && "Unknown or unsupported stmt kind!");
           return false;

  // statements
  case Stmt::NullStmtClass:
  case Stmt::GotoStmtClass:
  case Stmt::BreakStmtClass:    return false;
  case Stmt::ContinueStmtClass: return false;
  case Stmt::ReturnStmtClass:   return false;

  case Stmt::CompoundStmtClass: {
    CompoundStmt *Node = static_cast<CompoundStmt*>(S);
    CompoundStmt::body_iterator I, E;
    for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
      if (!IsWCLInvariant(*I)) return false;
    }
    return true;
  }
  case Stmt::SwitchStmtClass: {
    SwitchStmt *Node = static_cast<SwitchStmt*>(S);
    if (!IsWCLInvariant(Node->getCond())) return false;
    return IsWCLInvariant(Node->getBody());
  }
  case Stmt::CaseStmtClass: {
    CaseStmt *Node = static_cast<CaseStmt*>(S);
    if (!IsWCLInvariant(Node->getLHS())) return false;
    if (Node->getRHS()) {
      if (!IsWCLInvariant(Node->getRHS())) return false;
    }
    return IsWCLInvariant(Node->getSubStmt());
  }
  case Stmt::DefaultStmtClass: {
    DefaultStmt *Node = static_cast<DefaultStmt*>(S);
    return IsWCLInvariant(Node->getSubStmt());
  }
  case Stmt::LabelStmtClass: {
    LabelStmt *Node = static_cast<LabelStmt*>(S);
    return IsWCLInvariant(Node->getSubStmt());
  }
  case Stmt::IfStmtClass: {
    IfStmt *Node = static_cast<IfStmt*>(S);
    if (!IsWCLInvariant(Node->getCond())) return false;
    if (!IsWCLInvariant(Node->getThen())) return false;
    if (Stmt *Else = Node->getElse()) {
      return IsWCLInvariant(Else);
    }
    return true;
  }
  case Stmt::WhileStmtClass: {
    WhileStmt *Node = static_cast<WhileStmt*>(S);
    if (!IsWCLInvariant(Node->getCond())) return false;
    return IsWCLInvariant(Node->getBody());
  }
  case Stmt::DoStmtClass: {
    DoStmt *Node = static_cast<DoStmt*>(S);
    if (!IsWCLInvariant(Node->getBody())) return false;
    return IsWCLInvariant(Node->getCond());
  }
  case Stmt::ForStmtClass: {
    ForStmt *Node = static_cast<ForStmt*>(S);
    if (Stmt *Init = Node->getInit()) {
      if (!IsWCLInvariant(Init)) return false;
    }
    if (Expr *Cond = Node->getCond()) {
      if (!IsWCLInvariant(Cond)) return false;
    }
    if (Expr *Inc = Node->getInc()) {
      if (!IsWCLInvariant(Inc)) return false;
    }
    return IsWCLInvariant(Node->getBody());
  }
  } // end switch

  return false;
}

bool TransformLICM::HasOtherReachingDef(VarDecl *VD) {
  // Find reaching definitions for VD (REACHin of CurCFGBlk).
  // Find defs only in the same WCR.
  unsigned pos = RD->getBitPosition(VD);
  UIntStmtsMapTy *inStmtsMap = RD->getREACHinData(CurCFGBlk);
  if (inStmtsMap->find(pos) != inStmtsMap->end()) {
    StmtSetTy &stmts = (*inStmtsMap)[pos];
    for (StmtSetTy::iterator I = stmts.begin(), E = stmts.end();
         I != E; ++I) {
      Stmt *def   = *I;
      CFGBlock *B = RD->getCFGBlock(def);
      if (B->getWCRID() != CurWCRID) continue;

      // If there is other reaching definition of the same WCR,
      // return true.
      if (def != CurStmt) return true;

      // FIXME: If all definitions are invariant, return false.
    }
  }

  return false;
}

bool TransformLICM::HasReachableUse(VarDecl *VD) {
  if (VarDeclDUChains.find(VD) == VarDeclDUChains.end()) return false;

  DUChainsTy &DUChains = VarDeclDUChains[VD];
  
  // Check if there is any use of def(CurStmt) in WCR.
  if (DUChains.find(CurStmt) != DUChains.end()) {
    StmtSetTy &uses = DUChains[CurStmt];
    for (StmtSetTy::iterator I = uses.begin(), E = uses.end(); I != E; ++I) {
      Stmt *use = *I;
      CFGBlock *B = RD->getCFGBlock(use);
      if (B->getWCRID() != CurWCRID) continue;

      if (CurStmt != use) return true;
    }
  }

  return false;
}


//---------------------------------------------------------------------------
// Code Rearrangement
//---------------------------------------------------------------------------
void TransformLICM::RearrangeCode(CompoundStmt *body) {
  StmtVector newBody;
  CurBody = &newBody;
  RearrangeCodeInCompoundStmt(body);

  // Set stmts as the new body.
  body->setStmts(ASTCtx, newBody.data(), newBody.size());
}

Stmt *TransformLICM::RearrangeCodeInStmt(Stmt *S) {
  switch (S->getStmtClass()) {
    default: return S;

    case Stmt::CompoundStmtClass: {
      CompoundStmt *Node = static_cast<CompoundStmt*>(S);

      if (Node->isWCR()) {
        // Remove empty WCRs that are not landing pads.
        if ((Node->size() == 0) && (!WCR.isLandPad(Node))) {
          WCR.eraseWCR(Node);
          ASTCtx.Deallocate(Node);
          return 0;
        }
        return Node;
      } else if (Node->isStmtList()) {
        // Unwrap StmtList.
        RearrangeCodeInCompoundStmt(Node);
        return 0;
      }

      StmtVector *prevBody = CurBody;
      StmtVector newBody;
      CurBody = &newBody;
      RearrangeCodeInCompoundStmt(Node);
      Stmt *newCS = CLExprs.NewCompoundStmt(newBody);
      ASTCtx.Deallocate(Node);
      CurBody = prevBody;
      return newCS;
    }

    case Stmt::IfStmtClass: {
      IfStmt *Node = static_cast<IfStmt*>(S);
      Node->setThen(RearrangeCodeInStmt(Node->getThen()));
      if (Stmt *Else = Node->getElse()) {
        Node->setElse(RearrangeCodeInStmt(Else));
      }
      return Node;
    }
    case Stmt::WhileStmtClass: {
      WhileStmt *Node = static_cast<WhileStmt*>(S);
      Node->setBody(RearrangeCodeInStmt(Node->getBody()));
      return Node;
    }
  }

  return S;
}

void TransformLICM::RearrangeCodeInCompoundStmt(CompoundStmt *S) {
  CompoundStmt *prevWCR = 0;

  for (CompoundStmt::body_iterator I = S->body_begin(),
                                   E = S->body_end(); I != E; ++I) {
    Stmt *newS = RearrangeCodeInStmt(*I);
    if (newS) {
      // merge consecutive WCRs.
      CompoundStmt *CS = dyn_cast<CompoundStmt>(newS);
      if (CS && CS->isWCR() && !WCR.isLandPad(CS)) {
        if (prevWCR) {
          MergeConsecutiveWCRs(prevWCR, CS);
        } else {
          prevWCR = CS;
        }
      } else {
        if (prevWCR) {
          CurBody->push_back(prevWCR);
          prevWCR = 0;
        }
        CurBody->push_back(newS);
      }
    }
  }

  if (prevWCR) CurBody->push_back(prevWCR);
}

void TransformLICM::MergeConsecutiveWCRs(CompoundStmt *WCR1,
                                         CompoundStmt *WCR2) {
  StmtVector Stmts;

  CompoundStmt::body_iterator I, E;
  for (I = WCR1->body_begin(), E = WCR1->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }
  for (I = WCR2->body_begin(), E = WCR2->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }

  WCR1->setStmts(ASTCtx, Stmts.data(), Stmts.size());

  // remove the merged WCR.
  WCR.eraseWCR(WCR2);
  ASTCtx.Deallocate(WCR2);
}


//---------------------------------------------------------------------------
// Printing functions for debugging
//---------------------------------------------------------------------------
void TransformLICM::printWCRBlksMap() {
  OS << "=== CFGBlocks of WCR ===\n";
  for (WCRBlksMapTy::iterator I = WCRBlksMap.begin(), E = WCRBlksMap.end();
       I != E; ++I) {
    OS << "W" << (*I).first << " : ";
    BlkSetTy &blks = (*I).second;
    for (BlkSetTy::iterator BI = blks.begin(), BE = blks.end(); 
         BI != BE; ++BI) {
      if (BI != blks.begin()) OS << ", ";
      CFGBlock *B = *BI;
      OS << "B" << B->getBlockID();
      if (B->isFirstOfWCR()) OS << "(F)";
    }
    OS << "\n";
  }
  OS << "\n";
  OS.flush();
}

void TransformLICM::printVariantSet() {
  OS << "=== Variant Set ===\n";
  for (VarDeclSetTy::iterator I = VariantSet.begin(), E = VariantSet.end();
       I != E; ++I) {
    if (I != VariantSet.begin()) OS << ", ";
    VarDecl *VD = *I;
    OS << VD->getNameAsString();
  }
  OS << "\n\n";
  OS.flush();
}

void TransformLICM::printDUChains(CFG &cfg) {
  CLStmtPrinter P(OS, ASTCtx, NULL, ASTCtx.PrintingPolicy);

  OS << "=== DU-Chains ===\n";
  for (VarDeclDUChainsTy::iterator I = VarDeclDUChains.begin(),
                                   E = VarDeclDUChains.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    OS << VD->getNameAsString() << " :\n";
    DUChainsTy &chains = (*I).second;
    for (DUChainsTy::iterator DI = chains.begin(), DE = chains.end();
         DI != DE; ++DI) {
      Stmt *def = (*DI).first;
      CFGBlock *defB = RD->getCFGBlock(def);
      OS << "  ";
      P.Visit(def);
      OS << " (B" << defB->getBlockID() << ")\n";
      
      StmtSetTy &uses = (*DI).second;
      for (StmtSetTy::iterator SI = uses.begin(), SE = uses.end();
           SI != SE; ++SI) {
        Stmt *use = *SI;
        CFGBlock *useB = RD->getCFGBlock(use);
        OS << "   --> ";
        P.Visit(use);
        OS << " (B" << useB->getBlockID() << ")\n";
      }
    }
  }
  OS << "\n\n";
  OS.flush();
}

void TransformLICM::printCurDefMap() {
  CLStmtPrinter P(OS, ASTCtx, NULL, ASTCtx.PrintingPolicy);

  for (VarDeclStmtsMapTy::iterator I = CurDefMap.begin(), E = CurDefMap.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    OS << VD->getNameAsString() << " :\n";
    StmtSetTy &stmts = (*I).second;
    for (StmtSetTy::iterator SI = stmts.begin(), SE = stmts.end();
         SI != SE; ++SI) {
      Stmt *def = *SI;
      CFGBlock *defB = RD->getCFGBlock(def);
      OS << "  ";
      P.Visit(def);
      OS << " (B" << defB->getBlockID() << ")\n";
    }
  }
  OS.flush();
}

