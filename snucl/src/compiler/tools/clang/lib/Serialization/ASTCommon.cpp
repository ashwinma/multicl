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

//===--- ASTCommon.cpp - Common stuff for ASTReader/ASTWriter----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines common functions that both ASTReader and ASTWriter use.
//
//===----------------------------------------------------------------------===//

#include "ASTCommon.h"
#include "clang/Serialization/ASTDeserializationListener.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;

// Give ASTDeserializationListener's VTable a home.
ASTDeserializationListener::~ASTDeserializationListener() { }

serialization::TypeIdx
serialization::TypeIdxFromBuiltin(const BuiltinType *BT) {
  unsigned ID = 0;
  switch (BT->getKind()) {
  case BuiltinType::Void:       ID = PREDEF_TYPE_VOID_ID;       break;
  case BuiltinType::Bool:       ID = PREDEF_TYPE_BOOL_ID;       break;
  case BuiltinType::Char_U:     ID = PREDEF_TYPE_CHAR_U_ID;     break;
  case BuiltinType::UChar:      ID = PREDEF_TYPE_UCHAR_ID;      break;
  case BuiltinType::UShort:     ID = PREDEF_TYPE_USHORT_ID;     break;
  case BuiltinType::UInt:       ID = PREDEF_TYPE_UINT_ID;       break;
  case BuiltinType::ULong:      ID = PREDEF_TYPE_ULONG_ID;      break;
  case BuiltinType::ULongLong:  ID = PREDEF_TYPE_ULONGLONG_ID;  break;
  case BuiltinType::UInt128:    ID = PREDEF_TYPE_UINT128_ID;    break;
  case BuiltinType::Char_S:     ID = PREDEF_TYPE_CHAR_S_ID;     break;
  case BuiltinType::SChar:      ID = PREDEF_TYPE_SCHAR_ID;      break;
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:    ID = PREDEF_TYPE_WCHAR_ID;      break;
  case BuiltinType::Short:      ID = PREDEF_TYPE_SHORT_ID;      break;
  case BuiltinType::Int:        ID = PREDEF_TYPE_INT_ID;        break;
  case BuiltinType::Long:       ID = PREDEF_TYPE_LONG_ID;       break;
  case BuiltinType::LongLong:   ID = PREDEF_TYPE_LONGLONG_ID;   break;
  case BuiltinType::Int128:     ID = PREDEF_TYPE_INT128_ID;     break;
#ifdef __SNUCL_COMPILER__
  case BuiltinType::Half:       ID = PREDEF_TYPE_HALF_ID;      break;
#endif
  case BuiltinType::Float:      ID = PREDEF_TYPE_FLOAT_ID;      break;
  case BuiltinType::Double:     ID = PREDEF_TYPE_DOUBLE_ID;     break;
  case BuiltinType::LongDouble: ID = PREDEF_TYPE_LONGDOUBLE_ID; break;
  case BuiltinType::NullPtr:    ID = PREDEF_TYPE_NULLPTR_ID;    break;
  case BuiltinType::Char16:     ID = PREDEF_TYPE_CHAR16_ID;     break;
  case BuiltinType::Char32:     ID = PREDEF_TYPE_CHAR32_ID;     break;
  case BuiltinType::Overload:   ID = PREDEF_TYPE_OVERLOAD_ID;   break;
  case BuiltinType::Dependent:  ID = PREDEF_TYPE_DEPENDENT_ID;  break;
  case BuiltinType::ObjCId:     ID = PREDEF_TYPE_OBJC_ID;       break;
  case BuiltinType::ObjCClass:  ID = PREDEF_TYPE_OBJC_CLASS;    break;
  case BuiltinType::ObjCSel:    ID = PREDEF_TYPE_OBJC_SEL;      break;
  }

  return TypeIdx(ID);
}

unsigned serialization::ComputeHash(Selector Sel) {
  unsigned N = Sel.getNumArgs();
  if (N == 0)
    ++N;
  unsigned R = 5381;
  for (unsigned I = 0; I != N; ++I)
    if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(I))
      R = llvm::HashString(II->getName(), R);
  return R;
}
