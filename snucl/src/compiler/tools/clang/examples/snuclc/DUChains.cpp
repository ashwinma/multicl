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

#include "DUChains.h"
#include "CLUtils.h"
#include "CLStmtPrinter.h"
using namespace clang;
using namespace clang::snuclc;

//#define CL_DEBUG

//---------------------------------------------------------------------------
// Find du-chains for each variable.
//---------------------------------------------------------------------------
void DUChains::FindDUChains() {
  MakeStmtMap();

  // Iterate through the CFGBlocks. 
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *B = *I;

    // Skip the entry block and the exit block.
    if (B == &cfg.getEntry() || B == &cfg.getExit())
      continue;

    UIntStmtsMapTy *REACHin = RD.getREACHinData(B);
    if (!REACHin) continue;

    // Initialize CurDefMap using REACHin of the CFGBlock.
    CurDefMap.clear();
    for (UIntStmtsMapTy::iterator RI = REACHin->begin(), RE = REACHin->end();
         RI != RE; ++RI) {
      // Filter defs defined outside of WCR.
      StmtSetTy newDefs;
      StmtSetTy &defs = (*RI).second;
      for (StmtSetTy::iterator DI = defs.begin(), DE = defs.end();
           DI != DE; ++DI) {
        Stmt *def = *DI;
        if (StmtMap[def].first > -1)
          newDefs.insert(def);
      }

      if (newDefs.size() > 0) {
        VarDecl *VD = RD.getVarDecl((*RI).first);
        CurDefMap[VD] = newDefs;
      }
    }

    CurCFGBlock = B;
    CurWCRID = B->getWCRID();
    
    // Skip non-WCR CFGBlock.
    if (CurWCRID == -1) continue;

    // Iterate through the statements in the CFGBlock.
    for (CFGBlock::const_iterator BI = B->begin(), BE = B->end(); 
         BI != BE; ++BI) {
      const CFGStmt *SE = BI->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");

      // find du-chains
      Stmt *S = SE->getStmt();
      FindDUChainsInStmt(S);
    }
  }

  // Check if there are unused definitions.
  for (VarDeclDUChainTy::iterator I = VarDeclDUChain.begin(),
                                  E = VarDeclDUChain.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    DUChainMapTy &chainMap = (*I).second;
    assert(DefNum.find(VD) != DefNum.end() && "No definitions");
    if (chainMap.size() < DefNum[VD])
      UnusedVarDecls.insert(VD);

    // If a DeclStmt for the VarDecl is not used, the declaration of 
    //  the VarDecl should exist outside the WCR.
    if (DeclStmtMap.find(VD) != DeclStmtMap.end()) {
      if (chainMap.find(DeclStmtMap[VD]) != chainMap.end())
        DeclStmtMap.erase(VD);
    }
  }

#ifdef CL_DEBUG
  printDUChains();
  OS.flush();
#endif
}

/// Make a statement map
void DUChains::MakeStmtMap() {
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *B = *I;

    // Skip the entry block and the exit block.
    if (B == &cfg.getEntry() || B == &cfg.getExit())
      continue;

    if (!RD.hasREACHinData(B)) continue;

    CurCFGBlock = B;
    CurWCRID = B->getWCRID();
    
    // Iterate through the statements in the CFGBlock.
    for (CFGBlock::const_iterator BI = B->begin(), BE = B->end(); 
         BI != BE; ++BI) {
      const CFGStmt *SE = BI->getAs<CFGStmt>();
      assert(SE && "Which CFGElement?");

      // save the current statement
      Stmt *S = SE->getStmt();
      StmtMap[S] = WCRCFGBlockTy(CurWCRID, CurCFGBlock);
    }
  }
}

void DUChains::VisitSubStmt(Stmt *S) {
  if (StmtMap.find(S) == StmtMap.end()) {
    FindDUChainsInStmt(S);
  }
}

void DUChains::FindDUChainsInStmt(Stmt *S) {
  // S is a non-terminator statement.
  switch (S->getStmtClass()) {
  default: break;

  case Stmt::ReturnStmtClass: {
    ReturnStmt *Node = static_cast<ReturnStmt*>(S);
    if (Expr *retExpr = Node->getRetValue()) {
      VisitSubStmt(retExpr);
    }
    break;
  }
  case Stmt::DeclStmtClass: {
    DeclStmt *Node = static_cast<DeclStmt*>(S);
    DeclStmt::decl_iterator I, E;
    for (I = Node->decl_begin(), E = Node->decl_end(); I != E; ++I) {
      if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
        if (VD->hasInit()) {
          assert(0 && "Was StmtSimplifier executed?");
          InsertDefIntoCurDefMap(VD, Node);
          DeclStmtMap[VD] = Node;
        }
      }
    }
    break;
  }
  case Stmt::DeclRefExprClass: {
    DeclRefExpr *E = static_cast<DeclRefExpr*>(S);
    if (VarDecl *VD = dyn_cast<VarDecl>(E->getDecl())) {
      // Insert this DeclRefExpr into the UseMap.
      UseMap[E] = WCRCFGBlockTy(CurWCRID, CurCFGBlock);

      // If definitions of this VarDecl exist, make a DU-chain.
      if (CurDefMap.find(VD) != CurDefMap.end()) {
        DUChainMapTy &chainMap = VarDeclDUChain[VD];
        StmtSetTy &defs = CurDefMap[VD];

        for (StmtSetTy::iterator I = defs.begin(), IE = defs.end();
             I != IE; ++I) {
          Stmt *def = *I;
          chainMap[def].insert(E);
        } // end for
      } else {
        UnusedVarDecls.insert(VD);
      }
    }
    break;
  }
  case Stmt::BinaryOperatorClass: {
    BinaryOperator *E = static_cast<BinaryOperator*>(S);

    // && , || and comma(,) operator are used as a terminator.
    BinaryOperator::Opcode Op = E->getOpcode();
    if (Op == BO_LAnd || Op == BO_LOr || Op == BO_Comma) break;

    if (E->getOpcode() == BO_Assign) {
      VisitSubStmt(E->getRHS());
      VarDecl *VD = GetDefinedVarDecl(E->getLHS());
      if (VD && RD.isDeclaredInFunction(VD))
        InsertDefIntoCurDefMap(VD, E);
    } else {
      VisitSubStmt(E->getLHS());
      VisitSubStmt(E->getRHS());
    }
    break;
  }
  case Stmt::CompoundAssignOperatorClass: {
    CompoundAssignOperator *E = static_cast<CompoundAssignOperator*>(S);
    // FIXME: Because we cannot divide def and use in CompoundAssignOperator,
    // we do not add LHS as a definition.
    VisitSubStmt(E->getRHS());
    VisitSubStmt(E->getLHS());
    break;
  }
  case Stmt::UnaryOperatorClass: {
    UnaryOperator *E = static_cast<UnaryOperator*>(S);
    // FIXME: Because we cannot divide def and use in UnaryOperator,
    // we do not add SubExpr as a definition.
    if (E->getOpcode() == UO_AddrOf) {
      VarDecl *VD = GetDefinedVarDecl(E->getSubExpr());
      if (VD && RD.isDeclaredInFunction(VD))
        InsertDefIntoCurDefMap(VD, E);
    } else {
      VisitSubStmt(E->getSubExpr());
    }
    break;
  }

  case Stmt::CallExprClass: {
    CallExpr *E = static_cast<CallExpr*>(S);
    for (unsigned i = 0, e = E->getNumArgs(); i < e; i++) {
      VisitSubStmt(E->getArg(i));
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
    VisitSubStmt(E->getRHS());
    VisitSubStmt(E->getLHS());
    break;
  }
  case Stmt::MemberExprClass: {
    MemberExpr *E = static_cast<MemberExpr*>(S);
    VisitSubStmt(E->getBase());
    break;
  }
  case Stmt::ConditionalOperatorClass: {
    // ConditionalOperator is a terminator.
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
      if (E->getInit(i)) VisitSubStmt(E->getInit(i));
    }
    break;
  }
  case Stmt::DesignatedInitExprClass: {
    DesignatedInitExpr *E = static_cast<DesignatedInitExpr*>(S);
    for (unsigned i = 0, e = E->getNumSubExprs(); i < e; ++i) {
      if (E->getSubExpr(i)) VisitSubStmt(E->getSubExpr(i));
    }
    VisitSubStmt(E->getInit());
    break;
  }
  case Stmt::ParenListExprClass: {
    ParenListExpr *E = static_cast<ParenListExpr*>(S);
    for (unsigned i = 0, e = E->getNumExprs(); i != e; i++) {
      VisitSubStmt(E->getExpr(i));
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

/// Find really defined VarDecl of assignment LHS.
VarDecl *DUChains::GetDefinedVarDecl(Expr *E) {
  VarDecl *VD = 0;

  if (DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(E)) {
    VD = dyn_cast<VarDecl>(DRE->getDecl());
  } else if (ArraySubscriptExpr *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    VisitSubStmt(ASE->getRHS());
    VD = GetDefinedVarDecl(ASE->getLHS());
  } else if (MemberExpr *ME = dyn_cast<MemberExpr>(E)) {
    VD = GetDefinedVarDecl(ME->getBase());
  } else if (CompoundAssignOperator *CAO = dyn_cast<CompoundAssignOperator>(E)) {
    VisitSubStmt(CAO);
  } else if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(E)) {
    VisitSubStmt(BinOp);
  } else if (ImplicitCastExpr *ICE = dyn_cast<ImplicitCastExpr>(E)) {
    VD = GetDefinedVarDecl(ICE->getSubExpr());
  } else if (ParenExpr *PE = dyn_cast<ParenExpr>(E)) {
    VD = GetDefinedVarDecl(PE->getSubExpr());
  }

  return VD;
}

void DUChains::InsertDefIntoCurDefMap(VarDecl *VD, Stmt *def) {
  // Insert this VarDecl into CurDefMap.
  // Because this definition kills the previous definition, the previous
  //  entry of CurDefMap is overwritten.
  StmtSetTy defSet;
  defSet.insert(def);
  CurDefMap[VD] = defSet;

  // Increase the number of definitions for this VarDecl.
  if (DefNum.find(VD) != DefNum.end()) {
    DefNum[VD]++;
  } else {
    DefNum[VD] = 1;
  }

  // Special case:
  // If VD is not a builtin type, all its definitions need to be expanded.
  // Therefore, all definitions must have their du-chains.
  if (!VD->getType()->isBuiltinType()) {
    DUChainMapTy &chainMap = VarDeclDUChain[VD];
    if (chainMap.find(def) == chainMap.end())
      chainMap[def] = UseSetTy();
  }

  // If this def stmt is not registered in StmtMap, insert it.
  if (StmtMap.find(def) == StmtMap.end())
    StmtMap[def] = WCRCFGBlockTy(CurWCRID, CurCFGBlock);
}


//---------------------------------------------------------------------------
// Find webs for each VarDecl.
// - A web for a VarDecl is all du-chains of the VarDecl that contain a 
//   common use of the VarDecl.
//---------------------------------------------------------------------------
void DUChains::FindWebs() {
  VarDeclDUChainTy::iterator I, E;
  for (I = VarDeclDUChain.begin(), E = VarDeclDUChain.end(); I != E; ++I) {
    // Make an empty web-set for this VarDecl
    VarDecl *VD = (*I).first;
    VarDeclWeb[VD] = WebSetTy();
    WebSetTy &webs = VarDeclWeb[VD];

    // Iterate through du-chains map
    DUChainMapTy &chainMap = (*I).second;
    for (DUChainMapTy::iterator MI = chainMap.begin(), ME = chainMap.end();
         MI != ME; ++MI) {
      llvm::SmallVector<WebTy *, 8> webPtrs;

      // Find webs that have a common use with this du-chains iterating
      //  through the web set
      for (WebSetTy::iterator SI = webs.begin(), SE = webs.end();
           SI != SE; ++SI) {
        WebTy *web = const_cast<WebTy *>(&(*SI));
        for (WebTy::iterator WI = web->begin(), WE = web->end(); 
             WI != WE; ++WI) {
          if (HasCommonUse((*MI).second, (*WI).second)) {
            webPtrs.push_back(web);
            break;
          }
        }
      }

      // Create a web or merge this du-chains with the previous web.
      unsigned comWebNum = webPtrs.size();
      if (comWebNum == 0) {
        // No web has a common use.
        WebTy newWeb;
        newWeb.insert(*MI);
        webs.insert(newWeb);
      } else if (comWebNum == 1) {
        // A single web has a common use.
        // Merge this web and this du-chain.
        WebTy *web = webPtrs[0];
        web->insert(*MI);
      } else {
        // Multiple webs have a common use.
        // Merge all webs and this du-chain.
        WebTy *web = webPtrs[0];
        web->insert(*MI);
        for (unsigned i = 1; i < comWebNum; i++) {
          WebTy *webMerged = webPtrs[i];
          web->insert(webMerged->begin(), webMerged->end());
          webs.erase(*webMerged);
        }
      }
    }
  } // end for

#ifdef CL_DEBUG
  printWebs();
  OS.flush();
#endif
}

/// If there is a common use in two sets, return true.
/// Otherwise, return false.
bool DUChains::HasCommonUse(UseSetTy &setA, UseSetTy &setB) {
  if (setA.size() < setB.size()) {
    for (UseSetTy::iterator I = setA.begin(), E = setA.end(); I != E; ++I) {
      if (setB.find(*I) != setB.end()) return true;
    }
  } else {
    for (UseSetTy::iterator I = setB.begin(), E = setB.end(); I != E; ++I) {
      if (setA.find(*I) != setA.end()) return true;
    }
  }

  return false;
}


//---------------------------------------------------------------------------
// Partition webs.
// Each partition has only webs that have a common WCR.
// A web that can be included in a partition should have references that 
//  belong to more than one WCRs in it.
//---------------------------------------------------------------------------
void DUChains::PartitionWebs() {
  for (VarDeclWebTy::iterator I = VarDeclWeb.begin(), E = VarDeclWeb.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    WebSetTy &webs = (*I).second;
    if (VD->getType()->isBuiltinType()) {
      for (WebSetTy::iterator WI = webs.begin(), WE = webs.end();
          WI != WE; ++WI) {
        WebTy *web = const_cast<WebTy *>(&(*WI));
        if (GetBelongedWCRNum(web) > 1) {
          UpdateWebPartitions(VD, web);
        } else {
          UpdateRemainingWebs(VD, web);
        }
      }
    } else {
      // If VD is not a builtin type and its webs need to be partitioned,
      // we make all webs be a single partition.
      bool NeedPartition = false;
      for (WebSetTy::iterator WI = webs.begin(), WE = webs.end();
          WI != WE; ++WI) {
        WebTy *web = const_cast<WebTy *>(&(*WI));
        if (GetBelongedWCRNum(web) > 1) NeedPartition = true;
      }

      if (NeedPartition) {
        assert(VarDeclParWeb.find(VD) == VarDeclParWeb.end() &&
               "Does the partition exist?");

        WebPtrSetTy partition;
        for (WebSetTy::iterator WI = webs.begin(), WE = webs.end();
             WI != WE; ++WI) {
          WebTy *web = const_cast<WebTy *>(&(*WI));
          partition.insert(web);
        }
        ParWebTy &newParWeb = VarDeclParWeb[VD];
        newParWeb.insert(partition);
      }
    }
  }

#ifdef CL_DEBUG
  printPartitions();
  OS.flush();
  printRemainingWebs();
  OS.flush();
#endif
}

unsigned DUChains::GetBelongedWCRNum(WebTy *web) {
  WCRSetTy wcrSet;
  for (WebTy::iterator I = web->begin(), E = web->end(); I != E; ++I) {
    Stmt *def = (*I).first;
    wcrSet.insert(StmtMap[def].first);
    UseSetTy &uses = (*I).second;
    for (UseSetTy::iterator UI = uses.begin(), UE = uses.end(); 
         UI != UE; ++UI) {
      wcrSet.insert(UseMap[*UI].first);
    }
  }

  // Insert this WCR set into WebWCRMap.
  WebWCRMap[web] = wcrSet;

  return wcrSet.size();
}

void DUChains::UpdateWebPartitions(VarDecl *VD, WebTy *web) {
  if (VarDeclParWeb.find(VD) == VarDeclParWeb.end()) {
    // No entry
    WebPtrSetTy webPtrSet;
    webPtrSet.insert(web);
    ParWebTy newParWeb;
    newParWeb.insert(webPtrSet);
    VarDeclParWeb[VD] = newParWeb;
    return;
  }

  ParWebTy &parWeb = VarDeclParWeb[VD];
  llvm::SmallVector<WebPtrSetTy *, 8> pars;   // partitions to be merged
  
  // Find partitions that have a common WCR with this web iterating through
  //  all web partitions.
  for (ParWebTy::iterator I = parWeb.begin(), E = parWeb.end(); 
       I != E; ++I) {
    WebPtrSetTy *par = const_cast<WebPtrSetTy *>(&(*I));
    for (WebPtrSetTy::iterator SI = par->begin(), SE = par->end();
         SI != SE; ++SI) {
      if (HasCommonWCR(web, *SI)) {
        pars.push_back(par);
        break;
      }
    }
  }

  // Create a partition or merge this web with the previous partition.
  unsigned comParNum = pars.size();
  if (comParNum == 0) {
    // No partition has a common WCR.
    WebPtrSetTy webPtrSet;
    webPtrSet.insert(web);
    parWeb.insert(webPtrSet);
  } else if (comParNum == 1) {
    // A single partition has a common WCR.
    // Insert this web into the partition.
    WebPtrSetTy *partition = pars[0];
    partition->insert(web);
  } else {
    // Multiple partitions have a common WCR.
    // Merge all partitions and insert this web into the merged partition.
    WebPtrSetTy *partition = pars[0];
    partition->insert(web);
    for (unsigned i = 1; i < comParNum; i++) {
      WebPtrSetTy *parMerged = pars[i];
      partition->insert(parMerged->begin(), parMerged->end());
      parWeb.erase(*parMerged);
    }
  }
}

bool DUChains::HasCommonWCR(WebTy *webA, WebTy *webB) {
  WCRSetTy &setA = WebWCRMap[webA];
  WCRSetTy &setB = WebWCRMap[webB];

  if (setA.size() < setB.size()) {
    for (WCRSetTy::iterator I = setA.begin(), E = setA.end(); I != E; ++I) {
      if (setB.find(*I) != setB.end()) return true;
    }
  } else {
    for (WCRSetTy::iterator I = setB.begin(), E = setB.end(); I != E; ++I) {
      if (setA.find(*I) != setA.end()) return true;
    }
  }

  return false;
}

void DUChains::UpdateRemainingWebs(VarDecl *VD, WebTy *web) {
  VarDeclRemWeb[VD].insert(web);
}


//---------------------------------------------------------------------------
// Printing functions
//---------------------------------------------------------------------------
void DUChains::printDUChains() {
  OS << "=== DU-Chains ===\n";
  for (VarDeclDUChainTy::iterator I = VarDeclDUChain.begin(), 
                                  E = VarDeclDUChain.end(); 
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    OS << " [ " << VD->getNameAsString() << " ]:\n";
    OS.flush();

    DUChainMapTy &chainMap = (*I).second;
    for (DUChainMapTy::iterator I = chainMap.begin(), E = chainMap.end();
         I != E; ++I) {
      printDUChain((*I).first, (*I).second);
    }
    OS.flush();
  }
  OS << "\n";
}

void DUChains::printDUChain(Stmt *def, const UseSetTy &uses, unsigned indent) {
  CLStmtPrinter P(OS, ASTCtx, NULL, ASTCtx.PrintingPolicy);

  for (unsigned i = 0; i < indent; i++) OS << ' ';
  OS << "def ";
  OS.flush();
  if (StmtMap.find(def) != StmtMap.end()) {
    OS << "(W" << StmtMap[def].first << ",";
    CFGBlock *B = StmtMap[def].second;
    OS << "B" << B->getBlockID() << ")";
  }
  OS << ": ";
  if (def) P.Visit(def);
  else OS << "ParmVarDecl";
  if (!def || isa<Expr>(def)) OS << "\n";

  for (unsigned i = 0; i < (indent + 1); i++) OS << ' ';
  OS << "-> use (" << uses.size() << "): { ";
  OS.flush();
  for (UseSetTy::const_iterator UI = uses.begin(), UE = uses.end(); 
      UI != UE; ++UI) {
    if (UI != uses.begin()) OS << ", ";
    DeclRefExpr *use = *UI;
    P.Visit(use);
    if (UseMap.find(use) != UseMap.end()) {
      OS << " (W" << UseMap[use].first << ",";
      CFGBlock *B = UseMap[use].second;
      OS << "B" << B->getBlockID() << ")";
    }
    OS.flush();
  }
  OS << " }\n";
}

void DUChains::printWebs() {
  OS << "=== Webs ===\n";
  for (VarDeclWebTy::iterator I = VarDeclWeb.begin(), E = VarDeclWeb.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    OS << " [ " << VD->getNameAsString() << " ]:\n";

    unsigned webNum = 0;
    WebSetTy &webs = (*I).second;
    for (WebSetTy::iterator WI = webs.begin(), WE = webs.end();
         WI != WE; ++WI) {
      OS << "   <Web" << webNum++ << ">\n";
      const WebTy &web = *WI;
      for (WebTy::const_iterator I = web.begin(), E = web.end();
           I != E; ++I) {
        printDUChain((*I).first, (*I).second, 5);
      }
    }
  }
  OS << "\n";
}

void DUChains::printPartitions() {
  OS << "=== Partitions ===\n";
  for (VarDeclParWebTy::iterator I = VarDeclParWeb.begin(), 
                                 E = VarDeclParWeb.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    OS << " [ " << VD->getNameAsString() << " ]:\n";

    unsigned parNum = 0;
    ParWebTy &pars = (*I).second;
    for (ParWebTy::iterator PI = pars.begin(), PE = pars.end();
         PI != PE; ++PI) {
      OS << "   Partition " << parNum++ << ":\n";

      unsigned webNum = 0;
      const WebPtrSetTy &webs = *PI;
      for (WebPtrSetTy::const_iterator WI = webs.begin(), WE = webs.end();
           WI != WE; ++WI) {
        OS << "     <Web" << webNum++ << ">\n";
        WebTy *web = *WI;
        for (WebTy::const_iterator I = web->begin(), E = web->end();
            I != E; ++I) {
          printDUChain((*I).first, (*I).second, 7);
        }
      }
    }
  }
  OS << "\n";
}

void DUChains::printRemainingWebs() {
  OS << "=== Remaining Webs ===\n";
  for (VarDeclRemWebTy::iterator I = VarDeclRemWeb.begin(),
                                 E = VarDeclRemWeb.end();
       I != E; ++I) {
    VarDecl *VD = (*I).first;
    OS << " [ " << VD->getNameAsString() << " ]:\n";

    unsigned webNum = 0;
    WebPtrSetTy &webs = (*I).second;
    for (WebPtrSetTy::iterator WI = webs.begin(), WE = webs.end();
         WI != WE; ++WI) {
      OS << "   <Web" << webNum++ << ">\n";
      WebTy *web = *WI;
      for (WebTy::const_iterator I = web->begin(), E = web->end();
           I != E; ++I) {
        printDUChain((*I).first, (*I).second, 5);
      }
    }
  }
  OS << "\n";
}
