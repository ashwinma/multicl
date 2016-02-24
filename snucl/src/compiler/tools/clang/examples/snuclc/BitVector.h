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

#ifndef SNUCLC_BITVECTOR_H
#define SNUCLC_BITVECTOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace snuclc {

class BitVector {
  typedef unsigned int                  ElemTy;
  typedef llvm::SmallVector<ElemTy, 16> BitVectorTy;
  BitVectorTy BitVec;
  unsigned    BitNum;   // # of bits in ElemTy

public:
  BitVector() {
    BitNum = sizeof(ElemTy) * 8;
  }
  BitVector(bool init) {
    BitNum = sizeof(ElemTy) * 8;
    if (init) {
      BitVec.push_back((ElemTy)0);
    }
  }
  BitVector(BitVectorTy &bitVec) {
    BitVec = bitVec;
    BitNum = sizeof(ElemTy) * 8;
  }

  void Set(unsigned pos);
  void Unset(unsigned pos);
  bool Get(unsigned pos);
  bool IsZero();

  unsigned GetSize()    { return BitVec.size()*BitNum; }  // size of all bits
  unsigned GetVecSize() { return BitVec.size(); }         // size of BitVec
  
  static BitVector Union(BitVector &bv1, BitVector &bv2);
  static BitVector Intersection(BitVector &bv1, BitVector &bv2);
  static BitVector Subtract(BitVector &bv1, BitVector &bv2);
  static BitVector Complement(BitVector &bv);
  static bool CheckDifference(BitVector &bv1, BitVector &bv2);

  void   print(llvm::raw_ostream &OS);

private:
  void   CheckSize(unsigned pos);
  void   IncreaseSize(unsigned size);
  unsigned AdjustSize(BitVector &bv);
  ElemTy GetVector(unsigned vecpos) { return BitVec[vecpos]; }
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_BITVECTOR_H

