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

#ifndef SNUCLC_CLDECLPRINTER_H
#define SNUCLC_CLDECLPRINTER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace snuclc {

class CLDeclPrinter : public DeclVisitor<CLDeclPrinter> {
  llvm::raw_ostream &Out;
  ASTContext &Context;
  PrintingPolicy Policy;
  unsigned Indentation;

  bool NoTerminator;
  bool SkipNewLine;

  llvm::raw_ostream& Indent() { return Indent(Indentation); }
  llvm::raw_ostream& Indent(unsigned Indentation);
  void ProcessDeclGroup(llvm::SmallVectorImpl<Decl*>& Decls);

  void Print(AccessSpecifier AS);

public:
  CLDeclPrinter(llvm::raw_ostream &Out, ASTContext &Context,
                const PrintingPolicy &Policy,
                unsigned Indentation = 0)
    : Out(Out), Context(Context), Policy(Policy), Indentation(Indentation),
      NoTerminator(false), SkipNewLine(false) { }

  void VisitDeclContext(DeclContext *DC, bool Indent = true);

  void VisitTranslationUnitDecl(TranslationUnitDecl *D);
  void VisitTypedefDecl(TypedefDecl *D);
  void VisitEnumDecl(EnumDecl *D);
  void VisitRecordDecl(RecordDecl *D);
  void VisitEnumConstantDecl(EnumConstantDecl *D);
  void VisitFunctionDecl(FunctionDecl *D);
  void VisitFieldDecl(FieldDecl *D);
  void VisitVarDecl(VarDecl *D);
  void VisitLabelDecl(LabelDecl *D);
  void VisitParmVarDecl(ParmVarDecl *D);

  void PrintTypeAttrs(Decl *D);
  void PrintVarAttrs(Decl *D);

  void VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {}
  void VisitNamespaceDecl(NamespaceDecl *D) {}
  void VisitUsingDirectiveDecl(UsingDirectiveDecl *D) {}
  void VisitNamespaceAliasDecl(NamespaceAliasDecl *D) {}
  void VisitCXXRecordDecl(CXXRecordDecl *D) {}
  void VisitLinkageSpecDecl(LinkageSpecDecl *D) {}
  void VisitTemplateDecl(TemplateDecl *D) {}
  void VisitObjCMethodDecl(ObjCMethodDecl *D) {}
  void VisitObjCClassDecl(ObjCClassDecl *D) {}
  void VisitObjCImplementationDecl(ObjCImplementationDecl *D) {}
  void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {}
  void VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {}
  void VisitObjCProtocolDecl(ObjCProtocolDecl *D) {}
  void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {}
  void VisitObjCCategoryDecl(ObjCCategoryDecl *D) {}
  void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D) {}
  void VisitObjCPropertyDecl(ObjCPropertyDecl *D) {}
  void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {}
  void VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D) {}
  void VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D) {}
  void VisitUsingDecl(UsingDecl *D) {}
  void VisitUsingShadowDecl(UsingShadowDecl *D) {}
};

class CLDecl {
public:
  static void print(llvm::raw_ostream &Out, unsigned Indentation,
                    ASTContext &Context, Decl *D);
  static void print(llvm::raw_ostream &Out, const PrintingPolicy &Policy,
                    unsigned Indentation, ASTContext &Context, Decl *D);
  static void printGroup(Decl** Begin, unsigned NumDecls,
                         llvm::raw_ostream &Out, const PrintingPolicy &Policy,
                         unsigned Indentation, ASTContext &Context);
};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_CLDECLPRINTER_H
