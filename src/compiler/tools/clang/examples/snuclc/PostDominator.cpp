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

#include "PostDominator.h"
using namespace clang;
using namespace clang::snuclc;

//#define CL_DEBUG

void PostDominator::Initialize() {
  // PostDominator of the Exit block is the Exit block itself.
  CFGBlock *exit = &cfg.getExit();
  BlkSetTy exitDom;
  exitDom.insert(exit);
  PostDomMap[exit] = exitDom;

  // Exit block is the root node of PostDominator tree.
  BlkSetTy emptyChildren;
  ImmediatePostDomMap[exit] = exit;   // special case
  ChildrenMap[exit] = emptyChildren;

  BlkSetTy blockDom;
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    blockDom.insert(*I);
  }

  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *block = *I;
    if (block == exit) continue;
    PostDomMap[block] = blockDom;

    ImmediatePostDomMap[block] = 0;
    ChildrenMap[block] = emptyChildren;
  }
}

void PostDominator::Solve() {
  WorkListTy WorkList;

  // Add all CFGBlock but the exit block to WorkList.
  CFGBlock *Exit = &cfg.getExit();
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    if (*I != Exit) WorkList.insert(*I);
  }

  while (!WorkList.empty()) {
    CFGBlock *curBlock = *(WorkList.begin());

    // 1. Compute postdominators from successors.
    BlkSetTy newDom;
    ComputePostDominators(newDom, curBlock);

    // 2. Compare the current postdominators with the previous.
    if (CheckDifference(newDom, PostDomMap[curBlock])) {
      PostDomMap[curBlock] = newDom;
      InsertPredecessors(WorkList, curBlock);
    }

    // 3. Remove the current CFGBlock from the worklist.
    WorkList.erase(curBlock);
  }

#ifdef CL_DEBUG
  printPostDomMap();
#endif
}

void PostDominator::ConstructPostDominatorTree() {
  for (BlkToBlkSetTy::iterator I = PostDomMap.begin(), E = PostDomMap.end();
       I != E; ++I) {
    CFGBlock *block = (*I).first;
    if (block == &cfg.getExit()) continue;

    // Exit is a postdominator of all nodes.
    CFGBlock *imPostDom = &cfg.getExit();

    // Find the immediate postdominator.
    BlkSetTy &doms = (*I).second;
    for (BlkSetTy::iterator DI = doms.begin(), DE = doms.end();
         DI != DE; ++DI) {
      CFGBlock *postdominator = *DI;
      if (postdominator == block) continue;

      // If the current immediate postdominator is a postdominator of the 
      //  current postdominator, the current postdominator becomes the 
      //  immediate postdominator.
      if (IsPostDominator(imPostDom, postdominator))
        imPostDom = postdominator;
    }
    ImmediatePostDomMap[block] = imPostDom;

    // Make a link from the parent node to the current node.
    ChildrenMap[imPostDom].insert(block);
  }

#ifdef CL_DEBUG
  printPostDominatorTree();
#endif
}

/// Insert predecessor of CFGBlock B into the worklist.
void PostDominator::InsertPredecessors(WorkListTy &WorkList, CFGBlock *B) {
  CFGBlock::pred_iterator I, E;
  for (I = B->pred_begin(), E = B->pred_end(); I != E; ++I) {
    WorkList.insert(*I);
  }
}

void PostDominator::ComputePostDominators(BlkSetTy &dom, CFGBlock *B) {
  // Intersection over postdominators of sucessors. 
  CFGBlock::succ_iterator I = B->succ_begin(), E = B->succ_end();
  while (*I == NULL) ++I;
  if (I != E) {
    // Initialization.
    BlkSetTy &succPostDom = PostDomMap[*I];
    for (BlkSetTy::iterator DI = succPostDom.begin(), DE = succPostDom.end();
         DI != DE; ++DI) {
      dom.insert(*DI);
    }
    ++I;
  }

  for ( ; I != E; ++I) {
    if (*I) {
      BlkSetTy eraseSet;

      // Find postdominators to be removed in dom.
      BlkSetTy &succPostDom = PostDomMap[*I];
      for (BlkSetTy::iterator DI = dom.begin(), DE = dom.end();
           DI != DE; ++DI) {
        if (succPostDom.find(*DI) == succPostDom.end())
          eraseSet.insert(*DI);
      }

      // Remove postdominators of eraseSet from dom.
      for (BlkSetTy::iterator DI = eraseSet.begin(), DE = eraseSet.end();
           DI != DE; ++DI) {
        dom.erase(*DI);
      }
    }
  }

  // Union
  dom.insert(B);
}

/// Check if two dominator sets are different.
///  true  - different
///  false - same
bool PostDominator::CheckDifference(BlkSetTy &newDom, BlkSetTy &oldDom) {
  // If the size of each set is not same, two sets are different.
  if (newDom.size() != oldDom.size()) return true;

  // Compare each element of two sets.
  for (BlkSetTy::iterator I = newDom.begin(), E = newDom.end(); I != E; ++I) {
    if (oldDom.find(*I) == oldDom.end()) return true;
  }

  return false;
}

/// If blkA is a postdominator of blkB, return true.
///  Otherwise, return false.
bool PostDominator::IsPostDominator(CFGBlock *blkA, CFGBlock *blkB) {
  return PostDomMap[blkB].find(blkA) != PostDomMap[blkB].end();
}

CFGBlock *PostDominator::getImmediatePostDominator(CFGBlock *B) {
  assert(ImmediatePostDomMap.find(B) != ImmediatePostDomMap.end() &&
         "Is it a correct CFGBlock?");
  return ImmediatePostDomMap[B];
}

/// Find the LCA CFGBlock between node X and node Y.
CFGBlock *PostDominator::FindLCA(CFGBlock *X, CFGBlock *Y) {
  NodeA = X;
  NodeB = Y;
  NodeLCA = &cfg.getExit();
  LCANodeMap.clear();

  TarjanOLCA(&cfg.getExit());

  return NodeLCA;
}

/// Tarjan's off-line least common ancestors algorithm
void PostDominator::TarjanOLCA(CFGBlock *U) {
  MakeSet(U);

  LCANodeMap[U].ancestor = U;

  BlkSetTy &children = ChildrenMap[U];
  for (BlkSetTy::iterator I = children.begin(), E = children.end();
       I != E; ++I) {
    CFGBlock *V = *I;
    TarjanOLCA(V);
    Union(U, V);
    LCANodeMap[Find(U)].ancestor = U;
  }

  LCANodeMap[U].color = BLACK;

  if (U == NodeA) {
    if (LCANodeMap.find(NodeB) != LCANodeMap.end()) {
      if (LCANodeMap[NodeB].color == BLACK)
        NodeLCA = LCANodeMap[Find(NodeB)].ancestor;
    }
  } else if (U == NodeB) {
    if (LCANodeMap.find(NodeA) != LCANodeMap.end()) {
      if (LCANodeMap[NodeA].color == BLACK)
        NodeLCA = LCANodeMap[Find(NodeA)].ancestor;
    }
  }
}

void PostDominator::MakeSet(CFGBlock *X) {
  LCANodeTy nodeData;
  nodeData.ancestor = 0;
  nodeData.parent = X;
  nodeData.rank = 0;
  nodeData.color = WHITE;
  LCANodeMap[X] = nodeData;
}

void PostDominator::Union(CFGBlock *X, CFGBlock *Y) {
  CFGBlock *xRoot = Find(X);
  CFGBlock *yRoot = Find(Y);

  LCANodeTy &xNode = LCANodeMap[xRoot];
  LCANodeTy &yNode = LCANodeMap[yRoot];
  if (xNode.rank > yNode.rank)
    yNode.parent = xRoot;
  else if (xNode.rank < yNode.rank)
    xNode.parent = yRoot;
  else if (xRoot != yRoot) {
    yNode.parent = xRoot;
    xNode.rank = xNode.rank + 1;
  }
}

CFGBlock *PostDominator::Find(CFGBlock *X) {
  CFGBlock *xParent = LCANodeMap[X].parent;
  if (xParent == X)
    return X;
  else {
    xParent = Find(xParent);
    LCANodeMap[X].parent = xParent;
    return xParent;
  }
}


void PostDominator::printPostDomMap() {
  OS << "=== PostDominators ===\n";
  for (BlkToBlkSetTy::iterator I = PostDomMap.begin(), E = PostDomMap.end();
       I != E; ++I) {
    CFGBlock *cfgBlock = (*I).first;
    OS << " [ B" << cfgBlock->getBlockID();
    if (cfgBlock == &cfg.getEntry())     OS << " (ENTRY)";
    else if (cfgBlock == &cfg.getExit()) OS << " (EXIT)";
    OS << " ]\n";

    BlkSetTy &doms = (*I).second;
    OS << "   PostDominators (" << doms.size() << "): ";
    printBlockSet(doms);
    OS << "\n";
  }
  OS << "\n";
}

void PostDominator::printBlockSet(BlkSetTy &blkSet) {
  OS << "{ ";
  for (BlkSetTy::iterator I = blkSet.begin(), E = blkSet.end(); I != E; ++I) {
    CFGBlock *block = *I;
    if (I != blkSet.begin()) OS << ", ";
    OS << "B" << block->getBlockID();
  }
  OS << " }";
}

void PostDominator::printPostDominatorTree() {
  OS  << "=== PostDominator Tree ===\n";
  for (CFG::reverse_iterator I = cfg.rbegin(), E = cfg.rend(); I != E; ++I) {
    CFGBlock *cfgBlock = *I;
    OS << " [ B" << cfgBlock->getBlockID();
    if (cfgBlock == &cfg.getEntry())     OS << " (ENTRY)";
    else if (cfgBlock == &cfg.getExit()) OS << " (EXIT)";
    OS << " ]\n";

    OS << "   Immediate postdominator: ";
    if (ImmediatePostDomMap[cfgBlock])
      OS << "B" << ImmediatePostDomMap[cfgBlock]->getBlockID();
    OS << "\n";

    BlkSetTy &children = ChildrenMap[cfgBlock];
    OS << "   Children (" << children.size() << "): ";
    printBlockSet(children);
    OS << "\n";
  }
  OS << "\n";
}

