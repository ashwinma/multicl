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

#ifndef SNUCLC_REACHINGDEFINITIONS_H
#define SNUCLC_REACHINGDEFINITIONS_H

#include "llvm/Support/raw_ostream.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/CFG.h"
#include "BitVector.h"
#include "TypeDefs.h"

namespace clang {
namespace snuclc {

/// ReachingDefinitions
/// - Implement the reaching definition dataflow analysis
class ReachingDefinitions {
  ASTContext        &ASTCtx;
  llvm::raw_ostream &OS;

  typedef std::set<CFGBlock *> WorkListTy;

  typedef llvm::SmallVector<VarDecl*, 16>      VarDeclVectorTy;
  typedef std::map<VarDecl*, unsigned>         VarDeclUIntMapTy;
  typedef std::map<Stmt*, CFGBlock*>           StmtBlkMapTy;
  typedef std::map<CFGBlock*, BitVector>       BlkBitVectorMapTy;
  typedef std::map<CFGBlock*, UIntStmtMapTy>   BlkUIntStmtMapTy;
  typedef std::map<CFGBlock*, UIntStmtsMapTy>  BlkUIntStmtsMapTy;

  StmtSetTy         AllStmts;           // all stmts in all CFGBlocks
  VarDeclSetTy      AllDecls;           // all VarDecls declared

  VarDeclVectorTy   GenDecls;           // all defined VarDecls
  VarDeclUIntMapTy  BitPosMap;          // VarDecl -> Bit position
  StmtBlkMapTy      StmtBlkMap;         // Stmt -> CFGBlock

  BlkBitVectorMapTy GenMap;             // GEN for each basic block
  BlkBitVectorMapTy KillMap;            // KILL for each basic block
  BlkUIntStmtMapTy  GenStmtMap;         // Stmt for each GEN set

  BlkBitVectorMapTy REACHinMap;         // REACHin for each basic block
  BlkBitVectorMapTy REACHoutMap;        // REACHout for each basic block
  BlkUIntStmtsMapTy REACHinStmtsMap;    // Stmts for each REACHin
  BlkUIntStmtsMapTy REACHoutStmtsMap;   // Stmts for each REACHout

  CFGBlock *CurCFGBlk;    // current CFGBlock
  Stmt     *CurStmt;      // current Stmt in CFGBlock
  bool PrivateOnly;       // Whether only private variables are handled.

public:
  ReachingDefinitions(ASTContext &C, llvm::raw_ostream &os)
    : ASTCtx(C), OS(os) {
    CurCFGBlk = NULL;
    CurStmt = NULL;
    PrivateOnly = false;
  }
  void Solve(CFG &cfg, bool privateOnly=true);

  bool hasREACHinData(CFGBlock *B) {
    return REACHinStmtsMap.find(B) != REACHinStmtsMap.end();
  }

  UIntStmtsMapTy *getREACHinData(CFGBlock *B) {
    if (REACHinStmtsMap.find(B) != REACHinStmtsMap.end())
      return &REACHinStmtsMap[B];
    else
      return NULL;
  }

  VarDecl *getVarDecl(int pos) {
    return GenDecls[pos];
  }
  unsigned getBitPosition(VarDecl *VD) {
    return BitPosMap[VD];
  }

  CFGBlock *getCFGBlock(Stmt *S) {
    return StmtBlkMap[S];
  }

  bool isDeclaredInFunction(VarDecl *VD) {
    return AllDecls.find(VD) != AllDecls.end();
  }

private:
  void SaveStmtsInCFGBlocks(CFG &cfg);

  void FindGenKillSet(CFG &cfg);
  void VisitSubStmt(Stmt *S);
  void FindGenKillSetInStmt(Stmt *S);
  void UpdateGenKillSet(Expr *E);
  unsigned GetBitPosition(VarDecl *VD);
  VarDecl *FindVarDeclOfExpr(Expr *E);
  void UpdateGenSet(VarDecl *VD);
  void UpdateKillSet(VarDecl *VD);

  void FindReachingDefinitions(CFG &cfg);
  void InsertSuccessors(WorkListTy &WorkList, CFGBlock *B);
  BitVector ComputeREACHin(CFGBlock *B);
  BitVector ComputeREACHout(BitVector &in, CFGBlock *B);
  bool CheckDifference(UIntStmtsMapTy &oldMap, UIntStmtsMapTy &newMap);
  void MergeStmtSets(StmtSetTy &in, StmtSetTy &out);

  // Printing functions for debugging
  void printReachingDefinitions(CFG &cfg);
  void printVarDecls(BitVector &bitVec, std::string &indent);
  void printGenSet(CFGBlock *B, std::string &indent);
  void printReachSet(UIntStmtsMapTy &stmtsMap, std::string &indent);
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_REACHINGDEFINITIONS_H
