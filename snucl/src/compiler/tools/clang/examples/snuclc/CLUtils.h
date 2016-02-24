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

#ifndef SNUCLC_CLUTILS_H
#define SNUCLC_CLUTILS_H

#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/StringRef.h"
#include "TypeDefs.h"
#include <set>

namespace clang {
namespace snuclc {

class CLUtils {
public:
  CLUtils() {}
  ~CLUtils() {}

  static bool IsClangTypeName(std::string &S);
  static bool IsCLTypeName(std::string &S);
  static bool IsCLImageType(QualType T);
  static bool IsCLImageTypeName(std::string &S);
  static bool IsCLVectorName(std::string &S);
  static bool IsCLEnumName(std::string &S);
  static bool IsCLBuiltinName(std::string &S);
  static bool IsPredefinedName(std::string &S);
  static bool IsWorkItemFunction(llvm::StringRef S);

  static bool IsKernel(Decl *D);
  static bool IsGlobal(Decl *D);
  static bool IsConstant(Decl *D);
  static bool IsLocal(Decl *D);
  static bool IsPrivate(Decl *D);
  static unsigned GetAddrQualifier(Decl *D);

  static bool IsReadOnly(Decl *D);
  static bool IsWriteOnly(Decl *D);
  static bool IsReadWrite(Decl *D);
  static unsigned GetAccessQualifier(Decl *D);

  static unsigned GetTypeQualifier(ValueDecl *D);

  static void GetTypeStrWithoutCVR(QualType Ty, std::string &S,
                                   PrintingPolicy &P);
  static unsigned GetTypeSize(QualType Ty, unsigned PointerSize);

  static unsigned CeilVecNumElements(QualType Ty);
  static unsigned GetVecAccessorNumber(Expr *E);
  
  static bool IsBuiltinFunction(Expr *E);
  static int  IsVectorDataFunction(Expr *E);
  static bool IsAsyncCopyFunction(Expr *E);
  static bool IsBarrierFunction(Expr *E);
  static bool IsAtomicFunction(Expr *E);
  static bool IsWorkItemIDFunction(Expr *E);
  static bool IsInvariantFunction(Expr *E);

  static bool IsGlobalPointerExprForDeref(Expr *E);

  static Expr *IgnoreImpCasts(Expr *E);

  static bool IsVectorArrayType(QualType Ty);

private:
  static bool HasAttributeAnnotate(Decl *D, llvm::StringRef *Str);

};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_CLUTILS_H

