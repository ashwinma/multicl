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

//===--- TypeLoc.cpp - Type Source Info Wrapper -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TypeLoc subclasses implementations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/ErrorHandling.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// TypeLoc Implementation
//===----------------------------------------------------------------------===//

namespace {
  class TypeLocRanger : public TypeLocVisitor<TypeLocRanger, SourceRange> {
  public:
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    SourceRange Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return TyLoc.getLocalSourceRange(); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

SourceRange TypeLoc::getLocalSourceRangeImpl(TypeLoc TL) {
  if (TL.isNull()) return SourceRange();
  return TypeLocRanger().Visit(TL);
}

namespace {
  class TypeSizer : public TypeLocVisitor<TypeSizer, unsigned> {
  public:
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    unsigned Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return TyLoc.getFullDataSize(); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

/// \brief Returns the size of the type source info data block.
unsigned TypeLoc::getFullDataSizeForType(QualType Ty) {
  if (Ty.isNull()) return 0;
  return TypeSizer().Visit(TypeLoc(Ty, 0));
}

namespace {
  class NextLoc : public TypeLocVisitor<NextLoc, TypeLoc> {
  public:
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    TypeLoc Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return TyLoc.getNextTypeLoc(); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}

/// \brief Get the next TypeLoc pointed by this TypeLoc, e.g for "int*" the
/// TypeLoc is a PointerLoc and next TypeLoc is for "int".
TypeLoc TypeLoc::getNextTypeLocImpl(TypeLoc TL) {
  return NextLoc().Visit(TL);
}

/// \brief Initializes a type location, and all of its children
/// recursively, as if the entire tree had been written in the
/// given location.
void TypeLoc::initializeImpl(ASTContext &Context, TypeLoc TL, 
                             SourceLocation Loc) {
  while (true) {
    switch (TL.getTypeLocClass()) {
#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT)        \
    case CLASS: {                     \
      CLASS##TypeLoc TLCasted = cast<CLASS##TypeLoc>(TL); \
      TLCasted.initializeLocal(Context, Loc);  \
      TL = TLCasted.getNextTypeLoc(); \
      if (!TL) return;                \
      continue;                       \
    }
#include "clang/AST/TypeLocNodes.def"
    }
  }
}

SourceLocation TypeLoc::getBeginLoc() const {
  TypeLoc Cur = *this;
  while (true) {
    switch (Cur.getTypeLocClass()) {
    // FIXME: Currently QualifiedTypeLoc does not have a source range
    // case Qualified:
    case Elaborated:
    case DependentName:
    case DependentTemplateSpecialization:
      break;
    default:
      TypeLoc Next = Cur.getNextTypeLoc();
      if (Next.isNull()) break;
      Cur = Next;
      continue;
    }
    break;
  }
  return Cur.getLocalSourceRange().getBegin();
}

SourceLocation TypeLoc::getEndLoc() const {
  TypeLoc Cur = *this;
  while (true) {
    switch (Cur.getTypeLocClass()) {
    default:
      break;
    case Qualified:
    case Elaborated:
      Cur = Cur.getNextTypeLoc();
      continue;
    }
    break;
  }
  return Cur.getLocalSourceRange().getEnd();
}


namespace {
  struct TSTChecker : public TypeLocVisitor<TSTChecker, bool> {
    // Overload resolution does the real work for us.
    static bool isTypeSpec(TypeSpecTypeLoc _) { return true; }
    static bool isTypeSpec(TypeLoc _) { return false; }

#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    bool Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc) { \
      return isTypeSpec(TyLoc); \
    }
#include "clang/AST/TypeLocNodes.def"
  };
}


/// \brief Determines if the given type loc corresponds to a
/// TypeSpecTypeLoc.  Since there is not actually a TypeSpecType in
/// the type hierarchy, this is made somewhat complicated.
///
/// There are a lot of types that currently use TypeSpecTypeLoc
/// because it's a convenient base class.  Ideally we would not accept
/// those here, but ideally we would have better implementations for
/// them.
bool TypeSpecTypeLoc::classof(const TypeLoc *TL) {
  if (TL->getType().hasLocalQualifiers()) return false;
  return TSTChecker().Visit(*TL);
}

// Reimplemented to account for GNU/C++ extension
//     typeof unary-expression
// where there are no parentheses.
SourceRange TypeOfExprTypeLoc::getLocalSourceRange() const {
  if (getRParenLoc().isValid())
    return SourceRange(getTypeofLoc(), getRParenLoc());
  else
    return SourceRange(getTypeofLoc(),
                       getUnderlyingExpr()->getSourceRange().getEnd());
}


TypeSpecifierType BuiltinTypeLoc::getWrittenTypeSpec() const {
  if (needsExtraLocalData())
    return static_cast<TypeSpecifierType>(getWrittenBuiltinSpecs().Type);
  else {
    switch (getTypePtr()->getKind()) {
    case BuiltinType::Void:
      return TST_void;
    case BuiltinType::Bool:
      return TST_bool;
    case BuiltinType::Char_U:
    case BuiltinType::Char_S:
      return TST_char;
    case BuiltinType::Char16:
      return TST_char16;        
    case BuiltinType::Char32:
      return TST_char32;
    case BuiltinType::WChar_S:
    case BuiltinType::WChar_U:
      return TST_wchar;

    case BuiltinType::UChar:
    case BuiltinType::UShort:
    case BuiltinType::UInt:
    case BuiltinType::ULong:
    case BuiltinType::ULongLong:
    case BuiltinType::UInt128:
    case BuiltinType::SChar:
    case BuiltinType::Short:
    case BuiltinType::Int:
    case BuiltinType::Long:
    case BuiltinType::LongLong:
    case BuiltinType::Int128:
#ifdef __SNUCL_COMPILER__
    case BuiltinType::Half:
#endif
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble:
      llvm_unreachable("Builtin type needs extra local data!");
      // Fall through, if the impossible happens.
        
    case BuiltinType::NullPtr:
    case BuiltinType::Overload:
    case BuiltinType::Dependent:
    case BuiltinType::ObjCId:
    case BuiltinType::ObjCClass:
    case BuiltinType::ObjCSel:
      return TST_unspecified;
    }
  }
  
  return TST_unspecified;
}

TypeLoc TypeLoc::IgnoreParensImpl(TypeLoc TL) {
  while (ParenTypeLoc* PTL = dyn_cast<ParenTypeLoc>(&TL))
    TL = PTL->getInnerLoc();
  return TL;
}

void ElaboratedTypeLoc::initializeLocal(ASTContext &Context, 
                                        SourceLocation Loc) {
  setKeywordLoc(Loc);
  NestedNameSpecifierLocBuilder Builder;
  Builder.MakeTrivial(Context, getTypePtr()->getQualifier(), Loc);
  setQualifierLoc(Builder.getWithLocInContext(Context));
}

void DependentNameTypeLoc::initializeLocal(ASTContext &Context, 
                                           SourceLocation Loc) {
  setKeywordLoc(Loc);
  NestedNameSpecifierLocBuilder Builder;
  Builder.MakeTrivial(Context, getTypePtr()->getQualifier(), Loc);
  setQualifierLoc(Builder.getWithLocInContext(Context));
  setNameLoc(Loc);
}

void 
DependentTemplateSpecializationTypeLoc::initializeLocal(ASTContext &Context, 
                                                        SourceLocation Loc) {
  setKeywordLoc(Loc);
  if (getTypePtr()->getQualifier()) {
    NestedNameSpecifierLocBuilder Builder;
    Builder.MakeTrivial(Context, getTypePtr()->getQualifier(), Loc);
    setQualifierLoc(Builder.getWithLocInContext(Context));
  } else {
    setQualifierLoc(NestedNameSpecifierLoc());
  }
  
  setNameLoc(Loc);
  setLAngleLoc(Loc);
  setRAngleLoc(Loc);
  TemplateSpecializationTypeLoc::initializeArgLocs(Context, getNumArgs(),
                                                   getTypePtr()->getArgs(),
                                                   getArgInfos(), Loc);
}

void TemplateSpecializationTypeLoc::initializeArgLocs(ASTContext &Context, 
                                                      unsigned NumArgs,
                                                  const TemplateArgument *Args,
                                              TemplateArgumentLocInfo *ArgInfos,
                                                      SourceLocation Loc) {
  for (unsigned i = 0, e = NumArgs; i != e; ++i) {
    switch (Args[i].getKind()) {
    case TemplateArgument::Null: 
    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
    case TemplateArgument::Pack:
    case TemplateArgument::Expression:
      // FIXME: Can we do better for declarations and integral values?
      ArgInfos[i] = TemplateArgumentLocInfo();
      break;
      
    case TemplateArgument::Type:
      ArgInfos[i] = TemplateArgumentLocInfo(
                          Context.getTrivialTypeSourceInfo(Args[i].getAsType(), 
                                                           Loc));
      break;
        
    case TemplateArgument::Template:
    case TemplateArgument::TemplateExpansion: {
      NestedNameSpecifierLocBuilder Builder;
      TemplateName Template = Args[i].getAsTemplate();
      if (DependentTemplateName *DTN = Template.getAsDependentTemplateName())
        Builder.MakeTrivial(Context, DTN->getQualifier(), Loc);
      else if (QualifiedTemplateName *QTN = Template.getAsQualifiedTemplateName())
        Builder.MakeTrivial(Context, QTN->getQualifier(), Loc);
      
      ArgInfos[i] = TemplateArgumentLocInfo(
                                           Builder.getWithLocInContext(Context),
                                            Loc, 
                                Args[i].getKind() == TemplateArgument::Template
                                            ? SourceLocation()
                                            : Loc);
      break;
    }        
    }
  }
}

