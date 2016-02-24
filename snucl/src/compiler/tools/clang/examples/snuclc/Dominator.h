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

#ifndef SNUCLC_DOMINATOR_H
#define SNUCLC_DOMINATOR_H

#include "llvm/Support/raw_ostream.h"
#include "clang/Analysis/CFG.h"
#include <set>
#include <map>

namespace clang {
namespace snuclc {

class Dominator {
  llvm::raw_ostream &OS;
  CFG &cfg;

  /// Worklist to solve data-flow equation.
  typedef std::set<CFGBlock *> WorkListTy;

  /// Dominators of a CFGBlock
  typedef std::set<CFGBlock *> BlkSetTy;
  typedef std::map<CFGBlock *, BlkSetTy> BlkToBlkSetTy;
  BlkToBlkSetTy DomMap;

  /// Immediate dominator
  typedef std::map<CFGBlock *, CFGBlock *> BlkToBlkTy;
  BlkToBlkTy ImmediateDomMap;

  /// Children of a CFGBlock (dominator tree)
  BlkToBlkSetTy ChildrenMap;

  /// CFGBlock pair and its LCA (lowest common ancestor)
  CFGBlock *NodeA;
  CFGBlock *NodeB;
  CFGBlock *NodeLCA;

public:
  Dominator(llvm::raw_ostream &os, CFG &C) 
    : OS(os), cfg(C) {
    // Initialize dominators of all nodes and dominator tree.
    Initialize();

    Solve();
    ConstructDominatorTree();
  }

  bool IsDominator(CFGBlock *blkA, CFGBlock *blkB);
  CFGBlock *getImmediateDominator(CFGBlock *B);
  CFGBlock *FindLCA(CFGBlock *X, CFGBlock *Y);

private:
  void Initialize();
  void Solve();
  void ConstructDominatorTree();
  void InsertSuccessors(WorkListTy &WorkList, CFGBlock *B);
  void ComputeDominators(BlkSetTy &dom, CFGBlock *B);
  bool CheckDifference(BlkSetTy &newDom, BlkSetTy &oldDom);

  /// Tarjan's off-line least common ancestors algorithm
  typedef enum { WHITE, BLACK } ColorTy;
  typedef struct {
    CFGBlock *ancestor; 
    CFGBlock *parent;
    unsigned  rank;
    ColorTy   color;
  } LCANodeTy;
  typedef std::map<CFGBlock *, LCANodeTy> LCANodeMapTy;
  LCANodeMapTy LCANodeMap;

  void TarjanOLCA(CFGBlock *U);
  void MakeSet(CFGBlock *X);
  void Union(CFGBlock *X, CFGBlock *Y);
  CFGBlock *Find(CFGBlock *X);

  /// printing functions
  void printDomMap();
  void printBlockSet(BlkSetTy &blkSet);
  void printDominatorTree();
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_DOMINATOR_H
