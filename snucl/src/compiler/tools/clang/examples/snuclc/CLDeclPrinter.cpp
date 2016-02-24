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

#include "CLDeclPrinter.h"
#include "CLStmtPrinter.h"
#include "CLUtils.h"
#include "Defines.h"
using namespace clang;
using namespace clang::snuclc;

static QualType GetBaseType(QualType T) {
  // FIXME: This should be on the Type class!
  QualType BaseType = T;
  while (!BaseType->isSpecifierType()) {
    if (isa<TypedefType>(BaseType))
      break;
    else if (const PointerType* PTy = BaseType->getAs<PointerType>())
      BaseType = PTy->getPointeeType();
    else if (const ArrayType* ATy = dyn_cast<ArrayType>(BaseType))
      BaseType = ATy->getElementType();
    else if (const FunctionType* FTy = BaseType->getAs<FunctionType>())
      BaseType = FTy->getResultType();
    else if (const VectorType *VTy = BaseType->getAs<VectorType>())
      BaseType = VTy->getElementType();
    else
      assert(0 && "Unknown declarator!");
  }
  return BaseType;
}

static QualType getDeclType(Decl* D) {
  if (TypedefDecl* TDD = dyn_cast<TypedefDecl>(D))
    return TDD->getUnderlyingType();
  if (ValueDecl* VD = dyn_cast<ValueDecl>(D))
    return VD->getType();
  return QualType();
}

llvm::raw_ostream& CLDeclPrinter::Indent(unsigned Indentation) {
  for (unsigned i = 0; i != Indentation; ++i)
    Out << "  ";
  return Out;
}

void CLDeclPrinter::ProcessDeclGroup(llvm::SmallVectorImpl<Decl*>& Decls) {
  this->Indent();
  CLDecl::printGroup(Decls.data(), Decls.size(), Out, Policy, Indentation,
                     Context);
  Out << ";\n";
  Decls.clear();

}

void CLDeclPrinter::Print(AccessSpecifier AS) {
  switch(AS) {
  case AS_none:      assert(0 && "No access specifier!"); break;
  case AS_public:    Out << "public"; break;
  case AS_protected: Out << "protected"; break;
  case AS_private:   Out << "private"; break;
  }
}


//----------------------------------------------------------------------------
// Common C declarations
//----------------------------------------------------------------------------

void CLDeclPrinter::VisitDeclContext(DeclContext *DC, bool Indent) {
  if (Indent)
    Indentation += Policy.Indentation;

  llvm::SmallVector<Decl*, 2> Decls;
  for (DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
       D != DEnd; ++D) {

    // Don't print ObjCIvarDecls, as they are printed when visiting the
    // containing ObjCInterfaceDecl.
    if (isa<ObjCIvarDecl>(*D))
      continue;

    if (!Policy.Dump) {
      // Skip over implicit declarations in pretty-printing mode.
      if (D->isImplicit()) continue;
      // FIXME: Ugly hack so we don't pretty-print the builtin declaration
      // of __builtin_va_list or __[u]int128_t.  There should be some other way
      // to check that.
      if (NamedDecl *ND = dyn_cast<NamedDecl>(*D)) {
        if (IdentifierInfo *II = ND->getIdentifier()) {
          if (II->isStr("__builtin_va_list") ||
              II->isStr("__int128_t") || II->isStr("__uint128_t") ||
              II->isStr("__va_list_tag"))
            continue;
        }

        std::string Name = ND->getNameAsString();
        if (CLUtils::IsPredefinedName(Name))
          continue;
      }

      // Skip FunctionDecls that were fully inlined, i.e., there is no
      // invocation of them in the source code.
      if (FunctionDecl *FD = dyn_cast<FunctionDecl>(*D)) {
        if (FD->isFullyInlined()) continue;
      }
    }

    // The next bits of code handles stuff like "struct {int x;} a,b"; we're
    // forced to merge the declarations because there's no other way to
    // refer to the struct in question.  This limited merging is safe without
    // a bunch of other checks because it only merges declarations directly
    // referring to the tag, not typedefs.
    //
    // Check whether the current declaration should be grouped with a previous
    // unnamed struct.
    QualType CurDeclType = getDeclType(*D);
    if (!Decls.empty() && !CurDeclType.isNull()) {
      QualType BaseType = GetBaseType(CurDeclType);
      if (!BaseType.isNull() && isa<TagType>(BaseType) &&
          cast<TagType>(BaseType)->getDecl() == Decls[0]) {
        Decls.push_back(*D);
        continue;
      }
    }

    // If we have a merged group waiting to be handled, handle it now.
    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    // If the current declaration is an unnamed tag type, save it
    // so we can merge it with the subsequent declaration(s) using it.
    if (isa<TagDecl>(*D) && !cast<TagDecl>(*D)->getIdentifier()) {
      Decls.push_back(*D);
      continue;
    }

    if (isa<AccessSpecDecl>(*D)) {
      Indentation -= Policy.Indentation;
      this->Indent();
      Print(D->getAccess());
      Out << ":\n";
      Indentation += Policy.Indentation;
      continue;
    }

    this->Indent();
    Visit(*D);

    // FIXME: Need to be able to tell the CLDeclPrinter when
    const char *Terminator = 0;
    if (isa<FunctionDecl>(*D) &&
        cast<FunctionDecl>(*D)->isThisDeclarationADefinition())
      Terminator = 0;
    else if (isa<ObjCMethodDecl>(*D) && cast<ObjCMethodDecl>(*D)->getBody())
      Terminator = 0;
    else if (isa<NamespaceDecl>(*D) || isa<LinkageSpecDecl>(*D) ||
             isa<ObjCImplementationDecl>(*D) ||
             isa<ObjCInterfaceDecl>(*D) ||
             isa<ObjCProtocolDecl>(*D) ||
             isa<ObjCCategoryImplDecl>(*D) ||
             isa<ObjCCategoryDecl>(*D))
      Terminator = 0;
    else if (isa<EnumConstantDecl>(*D)) {
      DeclContext::decl_iterator Next = D;
      ++Next;
      if (Next != DEnd)
        Terminator = ",";
    } else if (NoTerminator) {
      Terminator = 0;
      NoTerminator = false;
    } else
      Terminator = ";";

    if (Terminator)
      Out << Terminator;
    if (!SkipNewLine)
      Out << "\n";
  }

  if (!Decls.empty())
    ProcessDeclGroup(Decls);

  if (Indent)
    Indentation -= Policy.Indentation;
}

void CLDeclPrinter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDeclContext(D, false);
}

void CLDeclPrinter::VisitTypedefDecl(TypedefDecl *D) {
  std::string S = D->getNameAsString();
  D->getUnderlyingType().getAsStringInternal(S, Policy);
  if (!Policy.SuppressSpecifiers)
    Out << "typedef ";
  Out << S;

  PrintTypeAttrs(D);
}

void CLDeclPrinter::VisitEnumDecl(EnumDecl *D) {
  Out << "enum ";
  if (D->isScoped()) {
    if (D->isScopedUsingClassTag())
      Out << "class ";
    else
      Out << "struct ";
  }
  Out << D;

  if (D->isFixed()) {
    std::string Underlying;
    D->getIntegerType().getAsStringInternal(Underlying, Policy);
    Out << " : " << Underlying;
  }

  if (D->isDefinition()) {
    bool PrvSkipNewLine = SkipNewLine;

    Out << " {";
    SkipNewLine = true;
    VisitDeclContext(D);
    Indent(1) << "}";

    PrintTypeAttrs(D);

    SkipNewLine = PrvSkipNewLine;
  }
}

void CLDeclPrinter::VisitRecordDecl(RecordDecl *D) {
  Out << D->getKindName();
  if (D->getIdentifier())
    Out << ' ' << D;

  if (D->isDefinition()) {
    bool PrvSkipNewLine = SkipNewLine;

    Out << " {";
    SkipNewLine = true;
    VisitDeclContext(D);
    Indent(1) << "}";

    PrintTypeAttrs(D);

    SkipNewLine = PrvSkipNewLine;
  }
}

void CLDeclPrinter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  Out << D;
  if (Expr *Init = D->getInitExpr()) {
    Out << " = ";
    CLStmt::printPretty(Out, Context, 0, Policy, Indentation, Init);
  }
}

void CLDeclPrinter::VisitFunctionDecl(FunctionDecl *D) {
  if (!Policy.SuppressSpecifiers) {
    switch (D->getStorageClass()) {
    case SC_None: break;
    case SC_Extern: Out << "extern "; break;
    case SC_Static: Out << "static "; break;
    case SC_PrivateExtern: Out << "__private_extern__ "; break;
    case SC_Auto: case SC_Register: llvm_unreachable("invalid for functions");
    }

    if (D->isInlineSpecified())           Out << "inline ";
    if (D->isVirtualAsWritten()) Out << "virtual ";
  }

  if (CLUtils::IsKernel(D)) {
    Out << "__kernel ";
  }

  PrintingPolicy SubPolicy(Policy);
  SubPolicy.SuppressSpecifiers = false;
  std::string Proto = D->getNameInfo().getAsString();

  QualType Ty = D->getType();
  while (const ParenType *PT = dyn_cast<ParenType>(Ty)) {
    Proto = '(' + Proto + ')';
    Ty = PT->getInnerType();
  }

  if (isa<FunctionType>(Ty)) {
    const FunctionType *AFT = Ty->getAs<FunctionType>();
    const FunctionProtoType *FT = 0;
    if (D->hasWrittenPrototype())
      FT = dyn_cast<FunctionProtoType>(AFT);

    Proto += "(";
    if (FT) {
      llvm::raw_string_ostream POut(Proto);
      CLDeclPrinter ParamPrinter(POut, Context, SubPolicy, Indentation);
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        if (i) POut << ", ";
        ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
      }

      if (FT->isVariadic()) {
        if (D->getNumParams()) POut << ", ";
        POut << "...";
      }
    } else if (D->isThisDeclarationADefinition() && !D->hasPrototype()) {
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        if (i)
          Proto += ", ";
        Proto += D->getParamDecl(i)->getNameAsString();
      }
    }

    Proto += ")";
    
    if (FT && FT->getTypeQuals()) {
      unsigned TypeQuals = FT->getTypeQuals();
      if (TypeQuals & Qualifiers::Const)
        Proto += " const";
      if (TypeQuals & Qualifiers::Volatile) 
        Proto += " volatile";
      if (TypeQuals & Qualifiers::Restrict)
        Proto += " restrict";
    }
    
    if (FT && FT->hasExceptionSpec()) {
      Proto += " throw(";
      if (FT->hasAnyExceptionSpec())
        Proto += "...";
      else 
        for (unsigned I = 0, N = FT->getNumExceptions(); I != N; ++I) {
          if (I)
            Proto += ", ";
          
          
          std::string ExceptionType;
          FT->getExceptionType(I).getAsStringInternal(ExceptionType, SubPolicy);
          Proto += ExceptionType;
        }
      Proto += ")";
    }

    if (D->hasAttr<NoReturnAttr>())
      Proto += " __attribute((noreturn))";
    if (CXXConstructorDecl *CDecl = dyn_cast<CXXConstructorDecl>(D)) {
      if (CDecl->getNumCtorInitializers() > 0) {
        Proto += " : ";
        Out << Proto;
        Proto.clear();
        for (CXXConstructorDecl::init_const_iterator B = CDecl->init_begin(),
             E = CDecl->init_end();
             B != E; ++B) {
          CXXCtorInitializer * BMInitializer = (*B);
          if (B != CDecl->init_begin())
            Out << ", ";
          if (BMInitializer->isAnyMemberInitializer()) {
            FieldDecl *FD = BMInitializer->getAnyMember();
            Out << FD;
          } else {
            Out << QualType(BMInitializer->getBaseClass(),
                            0).getAsString(Policy);
          }
          
          Out << "(";
          if (!BMInitializer->getInit()) {
            // Nothing to print
          } else {
            Expr *Init = BMInitializer->getInit();
            if (ExprWithCleanups *Tmp = dyn_cast<ExprWithCleanups>(Init))
              Init = Tmp->getSubExpr();
            
            Init = Init->IgnoreParens();
            
            Expr *SimpleInit = 0;
            Expr **Args = 0;
            unsigned NumArgs = 0;
            if (ParenListExpr *ParenList = dyn_cast<ParenListExpr>(Init)) {
              Args = ParenList->getExprs();
              NumArgs = ParenList->getNumExprs();
            } else if (CXXConstructExpr *Construct
                                          = dyn_cast<CXXConstructExpr>(Init)) {
              Args = Construct->getArgs();
              NumArgs = Construct->getNumArgs();
            } else
              SimpleInit = Init;
            
            if (SimpleInit) {
              CLStmt::printPretty(Out, Context, 0, Policy, Indentation,
                                  SimpleInit);
            } else {
              for (unsigned I = 0; I != NumArgs; ++I) {
                if (isa<CXXDefaultArgExpr>(Args[I]))
                  break;
                
                if (I)
                  Out << ", ";
                CLStmt::printPretty(Out, Context, 0, Policy, Indentation,
                                    Args[I]);
              }
            }
          }
          Out << ")";
        }
      }
    }
    else
      AFT->getResultType().getAsStringInternal(Proto, Policy);
  } else {
    Ty.getAsStringInternal(Proto, Policy);
  }

  Out << Proto;

  if (D->isPure())
    Out << " = 0";
  else if (D->isDeleted())
    Out << " = delete";
  else if (D->isThisDeclarationADefinition()) {
    if (!D->hasPrototype() && D->getNumParams()) {
      // This is a K&R function definition, so we need to print the
      // parameters.
      Out << '\n';
      CLDeclPrinter ParamPrinter(Out, Context, SubPolicy, Indentation);
      Indentation += Policy.Indentation;
      for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
        Indent();
        ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
        Out << ";\n";
      }
      Indentation -= Policy.Indentation;
    } else
      Out << ' ';

    CLStmt::printPretty(Out, Context, 0, SubPolicy, Indentation,
                        D->getBody());
    Out << '\n';
  }

  // Print the duplicated FunctionDecl if exists.
  if (FunctionDecl *DupFD = D->getDuplication()) {
    VisitFunctionDecl(DupFD);
  }
}

void CLDeclPrinter::VisitFieldDecl(FieldDecl *D) {
  if (!Policy.SuppressSpecifiers && D->isMutable())
    Out << "mutable ";

  std::string Name = D->getNameAsString();
  D->getType().getAsStringInternal(Name, Policy);
  Out << Name;

  PrintVarAttrs(D);

  if (D->isBitField()) {
    Out << " : ";
    CLStmt::printPretty(Out, Context, 0, Policy, Indentation,
                        D->getBitWidth());
  }
}

void CLDeclPrinter::VisitLabelDecl(LabelDecl *D) {
  Out << D->getNameAsString() << ":";
}


void CLDeclPrinter::VisitVarDecl(VarDecl *D) {
  if (!Policy.SuppressSpecifiers && D->getStorageClass() != SC_None)
    Out << VarDecl::getStorageClassSpecifierString(D->getStorageClass()) << " ";

  if (!Policy.SuppressSpecifiers && D->isThreadSpecified())
    Out << "__thread ";

  std::string Name = D->getNameAsString();
  QualType T = D->getType();
  if (CLUtils::IsCLImageType(T)) {
    Out << D->getAccessQualifierString();
    if (D->getAccessQualifier() != ACQ_None)
      Out << " ";
  } else {
    Out << D->getAddrQualifierString();
    if (D->getAddrQualifier() != AQ_Private)
      Out << " ";
  }

  if (ParmVarDecl *Parm = dyn_cast<ParmVarDecl>(D)) {
    T = Parm->getOriginalType();
  }

  T.getAsStringInternal(Name, Policy);
  Out << Name;

  PrintVarAttrs(D);

  if (Expr *Init = D->getInit()) {
    if (D->hasCXXDirectInitializer())
      Out << "(";
    else {
        CXXConstructExpr *CCE = dyn_cast<CXXConstructExpr>(Init);
        if (!CCE || CCE->getConstructor()->isCopyConstructor())
          Out << " = ";
    }
    CLStmt::printPretty(Out, Context, 0, Policy, Indentation, Init);
    if (D->hasCXXDirectInitializer())
      Out << ")";
  }
}

void CLDeclPrinter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
}

void CLDeclPrinter::PrintTypeAttrs(Decl *D) {
  if (PackedAttr *PA = D->getAttr<PackedAttr>()) {
    Out << ' ' << PA->getAsString();
  }

  if (AlignedAttr *AA = D->getAttr<AlignedAttr>()) {
    Out << ' ' << AA->getAsStringWithContext(Context);
  }
}

void CLDeclPrinter::PrintVarAttrs(Decl *D) {
  if (PackedAttr *PA = D->getAttr<PackedAttr>()) {
    Out << ' ' << PA->getAsString();
  }

  if (AlignedAttr *AA = D->getAttr<AlignedAttr>()) {
    Out << ' ' << AA->getAsStringWithContext(Context);
  }
}

//-------------------------------------------------------------------------//
void CLDecl::print(llvm::raw_ostream &Out, unsigned Indentation,
                   ASTContext &Context, Decl *D) {
  print(Out, Context.PrintingPolicy, Indentation, Context, D);
}

void CLDecl::print(llvm::raw_ostream &Out, const PrintingPolicy &Policy,
                   unsigned Indentation, ASTContext &Context, Decl *D) {
  CLDeclPrinter Printer(Out, Context, Policy, Indentation);
  Printer.Visit(D);
}

void CLDecl::printGroup(Decl** Begin, unsigned NumDecls,
                        llvm::raw_ostream &Out, const PrintingPolicy &Policy,
                        unsigned Indentation, ASTContext &Context) {
  if (NumDecls == 1) {
    print(Out, Policy, Indentation, Context, *Begin);
    return;
  }

  Decl** End = Begin + NumDecls;
  TagDecl* TD = dyn_cast<TagDecl>(*Begin);
  if (TD)
    ++Begin;

  PrintingPolicy SubPolicy(Policy);
  if (TD && TD->isDefinition()) {
    print(Out, Policy, Indentation, Context, TD);
    Out << " ";
    SubPolicy.SuppressTag = true;
  }

  bool isFirst = true;
  for ( ; Begin != End; ++Begin) {
    if (isFirst) {
      SubPolicy.SuppressSpecifiers = false;
      isFirst = false;
    } else {
      if (!isFirst) Out << ", ";
      SubPolicy.SuppressSpecifiers = true;
    }

    print(Out, SubPolicy, Indentation, Context, *Begin);
  }
}

