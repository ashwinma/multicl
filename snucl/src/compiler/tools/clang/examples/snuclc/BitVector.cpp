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

#include "BitVector.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;


/// Set the bit (make it one) of the bit position pos.
void BitVector::Set(unsigned pos) {
  CheckSize(pos);
  
  unsigned vecpos = pos / BitNum;
  unsigned bitpos = pos - vecpos * BitNum;
  BitVec[vecpos] = BitVec[vecpos] | (1 << bitpos);
}

/// Unset the bit (make it zero) of the bit position pos.
void BitVector::Unset(unsigned pos) {
  CheckSize(pos);
  
  unsigned vecpos = pos / BitNum;
  unsigned bitpos = pos - vecpos * BitNum;
  BitVec[vecpos] = BitVec[vecpos] & ~(1 << bitpos);
}

/// Get the bit value of the position pos.
bool BitVector::Get(unsigned pos) {
  CheckSize(pos);
  
  unsigned vecpos = pos / BitNum;
  unsigned bitpos = pos - vecpos * BitNum;
  ElemTy bit = BitVec[vecpos] & (1 << bitpos);
  return bit ? true : false;
}

/// Check if all bits of the bit vector are zero.
///  true - all bits are zero
///  false - otherwise
bool BitVector::IsZero() {
  ElemTy zero = (ElemTy)0;
  for (unsigned i = 0, e = BitVec.size(); i < e; i++)
    zero |= BitVec[i];
  if (zero == (ElemTy)0) return true;
  else return false;
}

/// Union of two bit vectors
BitVector BitVector::Union(BitVector &bv1, BitVector &bv2) {
  unsigned size = bv1.AdjustSize(bv2);

  // union
  BitVectorTy newVec;
  for (unsigned i = 0; i < size; i++) {
    newVec.push_back(bv1.GetVector(i) | bv2.GetVector(i));
  }
  BitVector result(newVec);
  return result;
}

/// Intersection of two bit vectors
BitVector BitVector::Intersection(BitVector &bv1, BitVector &bv2) {
  unsigned size = bv1.AdjustSize(bv2);

  // intersection
  BitVectorTy newVec;
  for (unsigned i = 0; i < size; i++) {
    newVec.push_back(bv1.GetVector(i) & bv2.GetVector(i));
  }
  BitVector result(newVec);
  return result;
}

/// Bit vector subtraction
BitVector BitVector::Subtract(BitVector &bv1, BitVector &bv2) {
  bv1.AdjustSize(bv2);

  // subtraction
  BitVector bv2Comp = Complement(bv2);
  return Intersection(bv1, bv2Comp);
}

/// Complement of the bit vector
BitVector BitVector::Complement(BitVector &bv) {
  BitVectorTy newVec;
  for (unsigned i = 0, e = bv.GetVecSize(); i < e; i++) {
    ElemTy elem = bv.GetVector(i);
    newVec.push_back(~elem);
  }
  BitVector result(newVec);
  return result;
}

/// Check if two bit vectors are different.
///  true  - different
///  false - same
bool BitVector::CheckDifference(BitVector &bv1, BitVector &bv2) {
  unsigned size = bv1.AdjustSize(bv2);
  for (unsigned i = 0; i < size; i++) {
    if (bv1.GetVector(i) != bv2.GetVector(i)) return true;
  }
  return false;
}

/// If the requested pos is beyond the current size of bit vector,
/// we increase the bit vector.
void BitVector::CheckSize(unsigned pos) {
  unsigned size = pos / BitNum + 1;
  IncreaseSize(size);
}

/// Increase the vector size and initialize the increased size as zero.
void BitVector::IncreaseSize(unsigned size) {
  for (unsigned i = BitVec.size(); i < size; i++) {
    BitVec.push_back((ElemTy)0);
  }
}

/// Make two bit vectors have same size.
unsigned BitVector::AdjustSize(BitVector &bv) {
  unsigned size1 = this->GetVecSize();
  unsigned size2 = bv.GetVecSize();
  unsigned size = size1;
  if (size1 > size2) {
    bv.IncreaseSize(size1);
  } else if (size1 < size2) {
    this->IncreaseSize(size2);
    size = size2;
  }
  return size; 
}

/// Print the bit vector as a bit pattern.
/// It is printed from LSB to MSB.
void BitVector::print(llvm::raw_ostream &OS) {
  for (unsigned i = 0, e = BitVec.size(); i < e; i++) {
    ElemTy val = BitVec[i];
    for (unsigned k = 0; k < BitNum; k++) {
      ElemTy bit = val & (1 << k);
      OS << (bit ? 1 : 0);
    }
    OS << ' ';
  }
  OS << '\n';
}
