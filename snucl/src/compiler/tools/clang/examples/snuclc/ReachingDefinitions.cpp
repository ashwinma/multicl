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

#include "ReachingDefinitions.h"
#include "CLStmtPrinter.h"
#include "CLUtils.h"
using namespace clang;
using namespace clang::snuclc;

using std::string;

//#define CL_DEBUG


void ReachingDefinitions::Solve(CFG &cfg, bool privateOnly) {
  PrivateOnly = privateOnly;

  // Save all stmts in all CFGBlocks.
  SaveStmtsInCFGBlocks(cfg);
  
  // Find generated and killed VarDecls.
  FindGenKillSet(cfg);

  // Find reaching definitions for each CFBBlock.
  FindReachingDefinitions(cfg);
}


void ReachingDefinitions::SaveStmtsInCFGBlocks(CFG &cfg) {
  for (CFG::const_iterator I = cfg.begin(), E = cfg.end(); I != E; ++I ) {
    for (CFGBlock::const_iterator BI = (*I)->begin(), BEnd = (*I)->end();
         BI != BEnd; ++BI) {
      const CFGStmt *SE = BI->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");

      Stmt *S = SE->getStmt();
      AllStmts.insert(S);

      if (S->getStmtClass() == Stmt::DeclStmtClass) {
        DeclStmt *Node = cast<DeclStmt>(S);
        if (VarDecl *VD = dyn_cast<VarDecl>(Node->getSingleDecl())) {
          AllDecls.insert(VD);
        }
      }
    }
  }
}


void ReachingDefinitions::FindGenKillSet(CFG &cfg) {
  for (CFG::reverse_iterator I = cfg.rbegin(), E = cfg.rend(); I != E; ++I) {
    CFGBlock *B = *I;

    // Initialization
    GenMap[B]  = BitVector();
    KillMap[B] = BitVector();
    GenStmtMap[B] = UIntStmtMapTy();
    REACHinStmtsMap[B]  = UIntStmtsMapTy();
    REACHoutStmtsMap[B] = UIntStmtsMapTy();

    // Set the current CFG block.
    CurCFGBlk = B;
    
    // Find Gen and Kill set in each stmt.
    for (CFGBlock::const_iterator BI = B->begin(), BE = B->end(); 
         BI != BE; ++BI) {
      // Set the current Stmt.
      const CFGStmt *SE = BI->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");

      CurStmt = SE->getStmt();
      StmtBlkMap[CurStmt] = CurCFGBlk;

      FindGenKillSetInStmt(CurStmt);
    }
  }
}

void ReachingDefinitions::VisitSubStmt(Stmt *S) {
  // Only when S does not appear in the top level of CFGBlocks, 
  // invoke FindGenKillSetInStmt().
  if (AllStmts.find(S) == AllStmts.end()) {
    FindGenKillSetInStmt(S);
  }
}

void ReachingDefinitions::FindGenKillSetInStmt(Stmt *S) {
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
      if (Expr *RetExpr = Node->getRetValue()) {
        VisitSubStmt(RetExpr);
      }
      break;
    }
    case Stmt::DeclStmtClass: {
      DeclStmt *Node = static_cast<DeclStmt*>(S);
      for (DeclStmt::decl_iterator I = Node->decl_begin(),
                                   E = Node->decl_end();
           I != E; ++I) {
        if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
          if (VD->hasInit()) {
            assert(0 && "Was StmtSimplifier executed?");
            UpdateGenSet(VD);
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
        VisitSubStmt(E->getArg(i));
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
        VisitSubStmt(RHS);
        VisitSubStmt(LHS);
        UpdateGenKillSet(LHS);
      } else {
        VisitSubStmt(LHS);
        VisitSubStmt(RHS);
      }

      break;
    }
    case Stmt::CompoundAssignOperatorClass: {
      CompoundAssignOperator *E = static_cast<CompoundAssignOperator*>(S);
      Expr *LHS = E->getLHS();
      Expr *RHS = E->getRHS();

      VisitSubStmt(RHS);
      VisitSubStmt(LHS);
      UpdateGenKillSet(LHS);
      break;
    }
    case Stmt::UnaryOperatorClass: {
      UnaryOperator *E = static_cast<UnaryOperator*>(S);
      VisitSubStmt(E->getSubExpr());
      switch (E->getOpcode()) {
        default: break;
        case UO_PostInc:
        case UO_PostDec:
        case UO_PreInc:
        case UO_PreDec: 
        case UO_AddrOf: {
          UpdateGenKillSet(E->getSubExpr());
          break;
        }
      }
      break;
    }
    case Stmt::ParenExprClass: {
      ParenExpr *E = static_cast<ParenExpr*>(S);
      VisitSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::SizeOfAlignOfExprClass: {
      SizeOfAlignOfExpr *E = static_cast<SizeOfAlignOfExpr*>(S);
      if (!E->isArgumentType())
        VisitSubStmt(E->getArgumentExpr());
      break;
    }
    case Stmt::VecStepExprClass: {
      VecStepExpr *E = static_cast<VecStepExpr*>(S);
      if (!E->isArgumentType())
        VisitSubStmt(E->getArgumentExpr());
      break;
    }
    case Stmt::ArraySubscriptExprClass: {
      ArraySubscriptExpr *E = static_cast<ArraySubscriptExpr*>(S);
      VisitSubStmt(E->getLHS());
      VisitSubStmt(E->getRHS());
      break;
    }
    case Stmt::MemberExprClass: {
      MemberExpr *E = static_cast<MemberExpr*>(S);
      VisitSubStmt(E->getBase());
      break;
    }
    case Stmt::ConditionalOperatorClass: {
      break;
    }
    case Stmt::ImplicitCastExprClass: {
      ImplicitCastExpr *E = static_cast<ImplicitCastExpr*>(S);
      VisitSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::CStyleCastExprClass: {
      CStyleCastExpr *E = static_cast<CStyleCastExpr*>(S);
      VisitSubStmt(E->getSubExpr());
      break;
    }
    case Stmt::CompoundLiteralExprClass: {
      CompoundLiteralExpr *E = static_cast<CompoundLiteralExpr*>(S);
      VisitSubStmt(E->getInitializer());
      break;
    }
    case Stmt::ExtVectorElementExprClass: {
      ExtVectorElementExpr *E = static_cast<ExtVectorElementExpr*>(S);
      VisitSubStmt(E->getBase());
      break;
    }
    case Stmt::InitListExprClass: {
      InitListExpr *E = static_cast<InitListExpr*>(S);
      for (unsigned i = 0, e = E->getNumInits(); i < e; i++) {
        if (E->getInit(i)) {
          VisitSubStmt(E->getInit(i));
        }
      }
      break;
    }
    case Stmt::DesignatedInitExprClass: {
      DesignatedInitExpr *E = static_cast<DesignatedInitExpr*>(S);
      for (unsigned i = 0, e = E->getNumSubExprs(); i < e; ++i) {
        if (E->getSubExpr(i)) {
          VisitSubStmt(E->getSubExpr(i));
        }
      }
      VisitSubStmt(E->getInit());
      break;
    }
    case Stmt::ParenListExprClass: {
      ParenListExpr *Node = static_cast<ParenListExpr*>(S);
      for (unsigned i = 0, e = Node->getNumExprs(); i != e; i++) {
        VisitSubStmt(Node->getExpr(i));
      }
      break;
    }
    case Stmt::VAArgExprClass: {
      VAArgExpr *E = static_cast<VAArgExpr*>(S);
      VisitSubStmt(E->getSubExpr());
      break;
    }
  } // end switch
}

VarDecl *ReachingDefinitions::FindVarDeclOfExpr(Expr *E) {
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

void ReachingDefinitions::UpdateGenKillSet(Expr *E) {
  if (VarDecl *VD = FindVarDeclOfExpr(E)) {
    if (PrivateOnly) {
      // All VarDecls declared in the current function are handled.
      E = E->IgnoreParenCasts();
      if (!CLUtils::IsPrivate(VD) && !isa<DeclRefExpr>(E)) return;

      if (AllDecls.find(VD) != AllDecls.end()) {
        UpdateKillSet(VD);
        UpdateGenSet(VD);
      }
    } else {
      UpdateKillSet(VD);
      UpdateGenSet(VD);
    }
  }
}

unsigned ReachingDefinitions::GetBitPosition(VarDecl *VD) {
  unsigned pos;
  if (BitPosMap.find(VD) != BitPosMap.end()) {
    pos = BitPosMap[VD];
  } else {
    GenDecls.push_back(VD);
    pos = GenDecls.size() - 1;
    BitPosMap[VD] = pos;
  }
  return pos;
}

void ReachingDefinitions::UpdateGenSet(VarDecl *VD) {
  unsigned pos = GetBitPosition(VD);

  // GEN set for the current CFGBlock.
  BitVector &gen = GenMap[CurCFGBlk];
  gen.Set(pos);

  // GEN Stmt
  UIntStmtMapTy &stmts = GenStmtMap[CurCFGBlk];
  stmts[pos] = CurStmt;
}

void ReachingDefinitions::UpdateKillSet(VarDecl *VD) {
  unsigned pos = GetBitPosition(VD);

  // KILL set for the current CFGBlock.
  BitVector &kill = KillMap[CurCFGBlk];
  kill.Set(pos);
}


//---------------------------------------------------------------------------
void ReachingDefinitions::FindReachingDefinitions(CFG &cfg) {
  WorkListTy WorkList;

  // Make the initial block data for the Entry block.
  CFGBlock *entry = &cfg.getEntry();
  REACHinMap[entry] = BitVector();
  REACHoutMap[entry] = BitVector();

  // Add all CFGBlock but entry to WorkList.
  for (CFG::reverse_iterator I = cfg.rbegin(), E = cfg.rend(); I != E; ++I) {
    if (*I != entry) WorkList.insert(*I);
  }

  while (!WorkList.empty()) {
    CFGBlock *curBlock = *(WorkList.begin());

    // 1. Compute the REACHin from predecessors.
    BitVector REACHin = ComputeREACHin(curBlock);

    // 2. Compute the REACHout of the current CFGBlock.
    UIntStmtsMapTy curStmtsMap = REACHoutStmtsMap[curBlock];
    BitVector REACHout = ComputeREACHout(REACHin, curBlock);

    // 3. Remove the current CFGBlock in the worklist.
    WorkList.erase(curBlock);

    // 4. Compare the current REACHout with the previous REACHout.
    //    If the previous REACHout does not exist or they are different, 
    //    the successors of the current CFGBlock are inserted into the 
    //    worklist.
    REACHinMap[curBlock] = REACHin;
    BlkBitVectorMapTy::iterator curIt = REACHoutMap.find(curBlock);
    if (curIt == REACHoutMap.end()) {
      // REACHoutMap entry does not exist.
      REACHoutMap[curBlock] = REACHout;
      InsertSuccessors(WorkList, curBlock);
    } else {
      BitVector &curREACHout = (*curIt).second;
      if (BitVector::CheckDifference(REACHout, curREACHout) ||
          CheckDifference(REACHoutStmtsMap[curBlock], curStmtsMap)) {
        REACHoutMap[curBlock] = REACHout;
        InsertSuccessors(WorkList, curBlock);
      }
    }
  }

#ifdef CL_DEBUG
  printReachingDefinitions(cfg);
#endif
}

/// Insert successors of CFGBlock B into the worklist. If the successor is
///  the Exit block, it is not inserted.
void ReachingDefinitions::InsertSuccessors(WorkListTy &WorkList, CFGBlock *B) {
  CFGBlock::succ_iterator I, E; 
  for (I = B->succ_begin(), E = B->succ_end(); I != E; ++I) {
    // Inifinite loop may have the NULL successor.
    if (*I) WorkList.insert(*I);
  }
}

/// REACHin[B] = Union of REACHout[p] where p is a predecessor of B.
BitVector ReachingDefinitions::ComputeREACHin(CFGBlock *B) {
  BitVector result;
  for (CFGBlock::pred_iterator I = B->pred_begin(), E = B->pred_end();
       I != E; ++I) {
    CFGBlock *predB = *I;   // previous CFGBlock

    BitVector &predOut = REACHoutMap[predB];
    if (predOut.GetVecSize() != 0) {
      if (result.GetVecSize() == 0) {
        // the first predecessor having REACHout.
        result = predOut;

        // save stmts in the bit vector
        REACHinStmtsMap[B] = REACHoutStmtsMap[predB];
      } else {
        // union the current result with other REACHout.
        BitVector newResult = BitVector::Union(result, predOut);

        // update saved stmts in the bit vector. (union)
        UIntStmtsMapTy &inStmtsMap  = REACHinStmtsMap[B];
        UIntStmtsMapTy &outStmtsMap = REACHoutStmtsMap[predB];
        for (UIntStmtsMapTy::iterator SI = outStmtsMap.begin(),
                                      SE = outStmtsMap.end();
             SI != SE; ++SI) {
          unsigned pos        = (*SI).first;
          StmtSetTy &outStmts = (*SI).second;
          if (inStmtsMap.find(pos) != inStmtsMap.end()) {
            StmtSetTy &inStmts = inStmtsMap[pos];
            MergeStmtSets(inStmts, outStmts);
          } else {
            inStmtsMap[pos] = outStmts;
          }
        }

        // update result.
        result = newResult;
      }
    }
  }
  return result;
}

/// Transfer function
/// REACHout[B] = GEN[B] U (REACHin[B] - KILL[B])
BitVector ReachingDefinitions::ComputeREACHout(BitVector &in, CFGBlock *B) {
  BitVector result;
  BitVector &gen = GenMap[B];

  // REACHin[B] - KILL[B]
  result = BitVector::Subtract(in, KillMap[B]);

  // save stmts in the bit vector.
  REACHoutStmtsMap[B] = REACHinStmtsMap[B];
  UIntStmtsMapTy &outStmtsMap = REACHoutStmtsMap[B];
  for (unsigned i = 0, e = in.GetSize(); i < e; ++i) {
    if (in.Get(i) && !result.Get(i)) {
      outStmtsMap.erase(i);
    }
  }

  // update outStmtsMap.
  UIntStmtMapTy &genStmt = GenStmtMap[B];
  for (unsigned i = 0, e = gen.GetSize(); i < e; ++i) {
    if (gen.Get(i)) {
      outStmtsMap[i] = StmtSetTy();
      StmtSetTy &stmtSet = outStmtsMap[i];
      stmtSet.insert(genStmt[i]);
    }
  }

  // GEN[B] U (REACHin[B] - KILL[B])
  result = BitVector::Union(gen, result);

  return result;
}

/// Check if two REACH sets are different.
///  true  - different
///  false - same
bool ReachingDefinitions::CheckDifference(UIntStmtsMapTy &oldMap, 
                                    UIntStmtsMapTy &newMap) {
  // If the size of each set is not same, two sets are different.
  if (oldMap.size() != newMap.size()) return true;

  // Compare each element of two sets.
  for (UIntStmtsMapTy::iterator I = oldMap.begin(), E = oldMap.end(); 
       I != E; ++I) {
    unsigned pos = (*I).first;
    if (newMap.find(pos) == newMap.end()) return true;

    StmtSetTy &oldStmts = (*I).second;
    StmtSetTy &newStmts = newMap[pos];
    if (oldStmts.size() != newStmts.size()) return true;
    for (StmtSetTy::iterator SI = oldStmts.begin(), SE = oldStmts.end();
         SI != SE; ++SI) {
      if (newStmts.find(*SI) == newStmts.end()) return true;
    }
  }

  return false;
}

/// Merge two StmtSets.
void ReachingDefinitions::MergeStmtSets(StmtSetTy &in, StmtSetTy &out) {
  for (StmtSetTy::iterator I = out.begin(), E = out.end(); I != E; ++I) {
    in.insert(*I);
  }
}


//---------------------------------------------------------------------------
// Printing functions for debugging
//---------------------------------------------------------------------------
void ReachingDefinitions::printReachingDefinitions(CFG &cfg) {
  OS << "=== Reaching Definitions ===\n";

  for (CFG::reverse_iterator I = cfg.rbegin(), E = cfg.rend(); I != E; ++I) {
    CFGBlock *blk = *I;
    OS << " [ B" << blk->getBlockID();
    if (blk == &cfg.getEntry())     OS << " (ENTRY)";
    else if (blk == &cfg.getExit()) OS << " (EXIT)";
    OS << " ]\n";
    OS.flush();

    if (REACHinMap.find(blk) == REACHinMap.end()) continue;

    string indent = "         ";
    OS << "   GEN : "; GenMap[blk].print(OS); OS.flush();
    printGenSet(blk, indent);

    OS << "   KILL: "; KillMap[blk].print(OS); OS.flush();
    printVarDecls(KillMap[blk], indent);

    OS << "   Rin : "; REACHinMap[blk].print(OS); OS.flush();
    printReachSet(REACHinStmtsMap[blk], indent);

    OS << "   Rout: "; REACHoutMap[blk].print(OS); OS.flush();
    printReachSet(REACHoutStmtsMap[blk], indent);
  }
  OS << "\n";
  OS.flush();
}

void ReachingDefinitions::printVarDecls(BitVector &bitVec, string &indent) {
  if (bitVec.IsZero()) return;

  bool next = false;
  OS << indent << "{ ";
  for (unsigned i = 0, e = bitVec.GetSize(); i < e; i++) {
    if (bitVec.Get(i)) {
      if (next) OS << ", ";

      VarDecl *VD = GenDecls[i];
      OS << VD->getNameAsString();

      next = true;
    }
  }
  OS << " }\n";
  OS.flush();
}

void ReachingDefinitions::printGenSet(CFGBlock *B, string &indent) {
  if (GenStmtMap.find(B) == GenStmtMap.end()) return;

  CLStmtPrinter P(OS, ASTCtx, NULL, ASTCtx.PrintingPolicy);
  UIntStmtMapTy &genStmt = GenStmtMap[B];
  for (UIntStmtMapTy::iterator I = genStmt.begin(), E = genStmt.end();
       I != E; ++I) {
    unsigned pos = (*I).first;
    Stmt *def    = (*I).second;
    VarDecl *VD  = GenDecls[pos];

    OS << indent << pos << " : " << VD->getNameAsString() << " : ";
    P.Visit(def);
    OS << " (B" << B->getBlockID() << ")\n";
  }
  OS.flush();
}

void ReachingDefinitions::printReachSet(UIntStmtsMapTy &stmtsMap, string &indent) {
  CLStmtPrinter P(OS, ASTCtx, NULL, ASTCtx.PrintingPolicy);
  for (UIntStmtsMapTy::iterator I = stmtsMap.begin(), E = stmtsMap.end();
       I != E; ++I) {
    unsigned pos     = (*I).first;
    StmtSetTy &stmts = (*I).second;
    VarDecl *VD      = GenDecls[pos];

    OS << indent << pos << " : " << VD->getNameAsString() << " : ";
    for (StmtSetTy::iterator SI = stmts.begin(), SE = stmts.end();
         SI != SE; ++SI) {
      Stmt *def = *SI;
      CFGBlock *B = StmtBlkMap[def];
      if (SI != stmts.begin()) OS << ", ";
      P.Visit(def);
      OS << " (B" << B->getBlockID() << ")";
    }
    OS << "\n";
  }
  OS.flush();
}

