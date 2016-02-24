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

#include "Dominator.h"
using namespace clang;
using namespace clang::snuclc;

//#define CL_DEBUG

void Dominator::Initialize() {
  // Dominator of the Entry block is the Entry block itself.
  CFGBlock *entry = &cfg.getEntry();
  BlkSetTy entryDom;
  entryDom.insert(entry);
  DomMap[entry] = entryDom;

  // Entry block is the root node of Dominator tree.
  BlkSetTy emptyChildren;
  ImmediateDomMap[entry] = entry;   // special case
  ChildrenMap[entry] = emptyChildren;

  BlkSetTy blockDom;
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    blockDom.insert(*I);
  }

  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *block = *I;
    if (block == entry) continue;
    DomMap[block] = blockDom;

    ImmediateDomMap[block] = 0;
    ChildrenMap[block] = emptyChildren;
  }
}

void Dominator::Solve() {
  WorkListTy WorkList;

  // Add all CFGBlock but the entry block to WorkList.
  CFGBlock *Entry = &cfg.getEntry();
  for (CFG::reverse_iterator I = cfg.rbegin(), E = cfg.rend(); I != E; ++I) {
    if (*I != Entry) WorkList.insert(*I);
  }

  while (!WorkList.empty()) {
    CFGBlock *curBlock = *(WorkList.begin());

    // 1. Compute dominators from predecessors.
    BlkSetTy newDom;
    ComputeDominators(newDom, curBlock);

    // 2. Compare the current dominators with the previous.
    if (CheckDifference(newDom, DomMap[curBlock])) {
      DomMap[curBlock] = newDom;
      InsertSuccessors(WorkList, curBlock);
    }

    // 3. Remove the current CFGBlock from the worklist.
    WorkList.erase(curBlock);
  }

#ifdef CL_DEBUG
  printDomMap();
#endif
}

void Dominator::ConstructDominatorTree() {
  for (BlkToBlkSetTy::iterator I = DomMap.begin(), E = DomMap.end();
       I != E; ++I) {
    CFGBlock *block = (*I).first;
    if (block == &cfg.getEntry()) continue;

    // Entry is a dominator of all nodes.
    CFGBlock *imDom = &cfg.getEntry();

    // Find the immediate dominator.
    BlkSetTy &doms = (*I).second;
    for (BlkSetTy::iterator DI = doms.begin(), DE = doms.end();
         DI != DE; ++DI) {
      CFGBlock *dominator = *DI;
      if (dominator == block) continue;

      // If the current immediate dominator is a dominator of the current 
      //  dominator, the current dominator becomes the immediate dominator.
      if (IsDominator(imDom, dominator))
        imDom = dominator;
    }
    ImmediateDomMap[block] = imDom;

    // Make a link from the parent node to the current node.
    ChildrenMap[imDom].insert(block);
  }

#ifdef CL_DEBUG
  printDominatorTree();
#endif
}

/// Insert successors of CFGBlock B into the worklist.
void Dominator::InsertSuccessors(WorkListTy &WorkList, CFGBlock *B) {
  CFGBlock::succ_iterator I, E;
  for (I = B->succ_begin(), E = B->succ_end(); I != E; ++I) {
    if (*I) WorkList.insert(*I);
  }
}

void Dominator::ComputeDominators(BlkSetTy &dom, CFGBlock *B) {
  // Intersection over dominators of predecessors
  CFGBlock::pred_iterator I = B->pred_begin(), E = B->pred_end();
  while (*I == NULL) ++I;
  if (I != E) {
    // Initialization.
    BlkSetTy &predDom = DomMap[*I];
    for (BlkSetTy::iterator DI = predDom.begin(), DE = predDom.end();
         DI != DE; ++DI) {
      dom.insert(*DI);
    }
    ++I;
  }

  for ( ; I != E; ++I) {
    if (*I) {
      BlkSetTy eraseSet;

      // Find dominators to be removed in dom.
      BlkSetTy &predDom = DomMap[*I];
      for (BlkSetTy::iterator DI = dom.begin(), DE = dom.end();
           DI != DE; ++DI) {
        if (predDom.find(*DI) == predDom.end())
          eraseSet.insert(*DI);
      }

      // Remove dominators of eraseSet from dom.
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
bool Dominator::CheckDifference(BlkSetTy &newDom, BlkSetTy &oldDom) {
  // If the size of each set is not same, two sets are different.
  if (newDom.size() != oldDom.size()) return true;

  // Compare each element of two sets.
  for (BlkSetTy::iterator I = newDom.begin(), E = newDom.end(); I != E; ++I) {
    if (oldDom.find(*I) == oldDom.end()) return true;
  }

  return false;
}

/// If blkA is a dominator of blkB, return true.
///  Otherwise, return false.
bool Dominator::IsDominator(CFGBlock *blkA, CFGBlock *blkB) {
  return DomMap[blkB].find(blkA) != DomMap[blkB].end();
}

CFGBlock *Dominator::getImmediateDominator(CFGBlock *B) {
  assert(ImmediateDomMap.find(B) != ImmediateDomMap.end() &&
         "Is it a correct CFGBlock?");
  return ImmediateDomMap[B];
}

/// Find the LCA CFGBlock between node X and node Y.
CFGBlock *Dominator::FindLCA(CFGBlock *X, CFGBlock *Y) {
  NodeA = X;
  NodeB = Y;
  NodeLCA = &cfg.getEntry();
  LCANodeMap.clear();

  TarjanOLCA(&cfg.getEntry());

  return NodeLCA;
}

/// Tarjan's off-line least common ancestors algorithm
void Dominator::TarjanOLCA(CFGBlock *U) {
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

void Dominator::MakeSet(CFGBlock *X) {
  LCANodeTy nodeData;
  nodeData.ancestor = 0;
  nodeData.parent = X;
  nodeData.rank = 0;
  nodeData.color = WHITE;
  LCANodeMap[X] = nodeData;
}

void Dominator::Union(CFGBlock *X, CFGBlock *Y) {
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

CFGBlock *Dominator::Find(CFGBlock *X) {
  CFGBlock *xParent = LCANodeMap[X].parent;
  if (xParent == X)
    return X;
  else {
    xParent = Find(xParent);
    LCANodeMap[X].parent = xParent;
    return xParent;
  }
}


void Dominator::printDomMap() {
  OS << "=== Dominators ===\n";
  for (BlkToBlkSetTy::reverse_iterator I = DomMap.rbegin(), E = DomMap.rend();
       I != E; ++I) {
    CFGBlock *cfgBlock = (*I).first;
    OS << " [ B" << cfgBlock->getBlockID();
    if (cfgBlock == &cfg.getEntry())     OS << " (ENTRY)";
    else if (cfgBlock == &cfg.getExit()) OS << " (EXIT)";
    OS << " ]\n";

    BlkSetTy &doms = (*I).second;
    OS << "   Dominators (" << doms.size() << "): ";
    printBlockSet(doms);
    OS << "\n";
  }
  OS << "\n";
}

void Dominator::printBlockSet(BlkSetTy &blkSet) {
  OS << "{ ";
  for (BlkSetTy::iterator I = blkSet.begin(), E = blkSet.end(); I != E; ++I) {
    CFGBlock *block = *I;
    if (I != blkSet.begin()) OS << ", ";
    OS << "B" << block->getBlockID();
  }
  OS << " }";
}

void Dominator::printDominatorTree() {
  OS  << "=== Dominator Tree ===\n";
  for (CFG::iterator I = cfg.begin(), E = cfg.end(); I != E; ++I) {
    CFGBlock *cfgBlock = *I;
    OS << " [ B" << cfgBlock->getBlockID();
    if (cfgBlock == &cfg.getEntry())     OS << " (ENTRY)";
    else if (cfgBlock == &cfg.getExit()) OS << " (EXIT)";
    OS << " ]\n";

    OS << "   Immediate dominator: ";
    if (ImmediateDomMap[cfgBlock])
      OS << "B" << ImmediateDomMap[cfgBlock]->getBlockID();
    OS << "\n";

    BlkSetTy &children = ChildrenMap[cfgBlock];
    OS << "   Children (" << children.size() << "): ";
    printBlockSet(children);
    OS << "\n";
  }
  OS << "\n";
}
