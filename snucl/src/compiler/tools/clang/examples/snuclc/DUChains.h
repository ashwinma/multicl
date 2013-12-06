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

#ifndef SNUCLC_DUCHAINS_H
#define SNUCLC_DUCHAINS_H

#include "llvm/Support/raw_ostream.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Analysis/CFG.h"
#include "ReachingDefinitions.h"
#include "TypeDefs.h"

namespace clang {
namespace snuclc {

class DUChains {
  ASTContext        &ASTCtx;
  llvm::raw_ostream &OS;

  CFG &cfg;
  ReachingDefinitions &RD;

  bool PrivateOnly;

  /// Current block information
  CFGBlock *CurCFGBlock;
  int CurWCRID;

  /// Map between a stmt and (WCR_ID, CFGBlock *).
  typedef std::pair<int, CFGBlock *>      WCRCFGBlockTy;
  typedef std::map<Stmt *, WCRCFGBlockTy> StmtMapTy;
  StmtMapTy StmtMap;

  /// Map between a use (DeclRefExpr) and (WCR_ID, CFGBlock *).
  typedef std::map<DeclRefExpr *, WCRCFGBlockTy> UseMapTy;
  UseMapTy UseMap;

  /// Current reaching definitions
  typedef std::map<VarDecl*, StmtSetTy> VarDeclStmtsMapTy;
  VarDeclStmtsMapTy CurDefMap;

public:
  /// DU-chains for each VarDecl
  typedef std::set<DeclRefExpr *>           UseSetTy;
  typedef std::pair<Stmt *, UseSetTy>       DUPairTy;
  typedef std::map<Stmt *, UseSetTy>        DUChainMapTy;
  typedef std::map<VarDecl *, DUChainMapTy> VarDeclDUChainTy;
  VarDeclDUChainTy VarDeclDUChain;

  /// Webs for each VarDecl
  typedef DUChainMapTy                   WebTy;
  typedef std::set<WebTy>                WebSetTy;
  typedef std::map<VarDecl *, WebSetTy>  VarDeclWebTy;
  VarDeclWebTy VarDeclWeb;

  /// The set of WCRs to which the references in the web belong
  typedef std::set<int>               WCRSetTy;
  typedef std::map<WebTy *, WCRSetTy> WebWCRMapTy;
  WebWCRMapTy WebWCRMap;

  /// Partitioned webs for each VarDecl
  /// Each partition has only webs that have no common WCR with other webs
  ///  in other partitions.
  typedef std::set<WebTy *>             WebPtrSetTy;
  typedef std::set<WebPtrSetTy>         ParWebTy;
  typedef std::map<VarDecl *, ParWebTy> VarDeclParWebTy;
  VarDeclParWebTy VarDeclParWeb;

private:
  /// Remaining webs that are not included in any partition.
  typedef std::map<VarDecl *, WebPtrSetTy> VarDeclRemWebTy;
  VarDeclRemWebTy VarDeclRemWeb;

  /// Number of definitions for each VarDecl.
  typedef std::map<VarDecl *, unsigned> DefNumTy;
  DefNumTy DefNum;

  /// DeclStmt that has Init expr for each VarDecl.
  /// This map is used to find unused DeclStmts.
  typedef std::map<VarDecl *, DeclStmt *> DeclStmtMapTy;
  DeclStmtMapTy DeclStmtMap;

  /// Set of VarDecls that have unused def or use.
  typedef std::set<VarDecl *> VarDeclSetTy;
  VarDeclSetTy UnusedVarDecls;

public:
  DUChains(ASTContext &C, llvm::raw_ostream &os, CFG &_cfg,
           ReachingDefinitions &rd)
    : ASTCtx(C), OS(os), cfg(_cfg), RD(rd) {
    CurCFGBlock = NULL;
    CurWCRID = -1;
    PrivateOnly = true;
  }

  void FindDUChains();
  void FindWebs();
  void PartitionWebs();
  
public:
  VarDeclParWebTy &getVarDeclParWeb() { return VarDeclParWeb; }

  bool hasRemainingWebs(VarDecl *VD) {
    return VarDeclRemWeb.find(VD) != VarDeclRemWeb.end();
  }

  bool hasUnusedDefUse(VarDecl *VD) {
    return UnusedVarDecls.find(VD) != UnusedVarDecls.end();
  }

  DeclStmt *getUnusedDeclStmt(VarDecl *VD) {
    if (DeclStmtMap.find(VD) != DeclStmtMap.end())
      return DeclStmtMap[VD];
    else
      return NULL;
  }

  int getWCRIDofDef(Stmt *def) {
    assert(StmtMap.find(def) != StmtMap.end() && "Unknown def");
    return StmtMap[def].first;
  }

  int getWCRIDofUse(DeclRefExpr *use) {
    assert(UseMap.find(use) != UseMap.end() && "Unknown use");
    return UseMap[use].first;
  }

  CFGBlock *getCFGBlockofDef(Stmt *def) {
    assert(StmtMap.find(def) != StmtMap.end() && "Unknown def");
    return StmtMap[def].second;
  }

  CFGBlock *getCFGBlockofUse(DeclRefExpr *use) {
    assert(UseMap.find(use) != UseMap.end() && "Unknown use");
    return UseMap[use].second;
  }

private:
  void MakeStmtMap();
  void VisitSubStmt(Stmt *S);
  void FindDUChainsInStmt(Stmt *S);
  VarDecl *GetDefinedVarDecl(Expr *E);
  void InsertDefIntoCurDefMap(VarDecl *VD, Stmt *def);

  bool HasCommonUse(UseSetTy &setA, UseSetTy &setB);

  unsigned GetBelongedWCRNum(WebTy *web);
  void UpdateWebPartitions(VarDecl *VD, WebTy *web);
  bool HasCommonWCR(WebTy *webA, WebTy *webB);
  void UpdateRemainingWebs(VarDecl *VD, WebTy *web);

  void printDUChains();
  void printDUChain(Stmt *def, const UseSetTy &uses, unsigned indent=4);
  void printWebs();
  void printPartitions();
  void printRemainingWebs();
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_DUCHAINS_H
