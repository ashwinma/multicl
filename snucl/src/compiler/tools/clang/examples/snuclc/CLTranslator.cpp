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

//===--- CLTranslator.cpp - CLTranslator implementation -----------------===//
//===--------------------------------------------------------------------===//
//
// CLTranslator - Translate ASTs to C code
//
//===--------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Analysis/AnalysisDiagnostic.h"
#include "Defines.h"
#include "CLTranslator.h"
#include "CLDeclPrinter.h"
#include "FunctionAnalyzer.h"
#include "FunctionInliner.h"
#include "FunctionDuplicator.h"
#include "BarrierMarker.h"
#include "TransformLocalVD.h"
#include "TransformBasic.h"
#include "TransformCoalescing.h"
#include "TransformVector.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

#include <string>
#include <sstream>
using std::string;
using std::stringstream;


CLTranslator::~CLTranslator() {
  if (CLExprs) delete CLExprs;
}

void CLTranslator::Initialize(ASTContext &Context) {
  Policy = &Context.PrintingPolicy;
  Policy->Indentation = 1;

  CLExprs = new CLExpressions(Context, Context.getTranslationUnitDecl());
}

void CLTranslator::HandleTopLevelDecl(DeclGroupRef DG) {
  if (CLOpts.Debug) {
    for (DeclGroupRef::iterator i = DG.begin(), e = DG.end(); i != e; ++i) {
      const Decl *D = *i;
      if (const NamedDecl *ND = dyn_cast<NamedDecl>(D))
        llvm::errs() << "top-level-decl: " << ND->getNameAsString() << "\n";
    }
  }
}

void CLTranslator::HandleTranslationUnit(ASTContext &Context) {
  // Check if there was any error
  if (Context.getDiagnostics().hasErrorOccurred()) return;

  // Save the pointer to ASTContex
  ASTCtx = &Context;

  // Gather kernel informations
  GatherKernelInfo();

  // Transform AST
  TransformAST();

  // Generate C code
  GenerateCode();
}

//===--------------------------------------------------------------------===//
void CLTranslator::GatherKernelInfo() {
  DeclContext *DeclCtx = ASTCtx->getTranslationUnitDecl();
  for (DeclContext::decl_iterator I = DeclCtx->decls_begin(),
                                  E = DeclCtx->decls_end();
       I != E; ++I) {
    FunctionDecl *FD = dyn_cast<FunctionDecl>(*I);
    if (!FD) continue;
      
    // Kernel function needs to print some kernel dependent codes.
    if (!CLUtils::IsKernel(FD) || !FD->hasBody()) continue;

    KernelMap[FD] = KernelInfo(FD);
    KernelInfo &KI = KernelMap[FD];

    for (unsigned i = 0, e = FD->getNumParams(); i != e; ++i) {
      ParmVarDecl *PD = FD->getParamDecl(i);

      // parameter's name
      KI.pushParamName(PD->getNameAsString());

      // address qualifier
      KI.pushParamAddrQualifier(CLUtils::GetAddrQualifier(PD));

      // access qualifier
      KI.pushParamAccessQualifier(CLUtils::GetAccessQualifier(PD));

      // type qualifier
      KI.pushParamTypeQualifier(CLUtils::GetTypeQualifier(PD));

      // type
      KI.pushParamType(PD->getType());
    }

    // attributes
    // __attribute__((reqd_work_group_size(X,Y,Z)))
    if (ReqdWorkGroupSizeAttr *RA = FD->getAttr<ReqdWorkGroupSizeAttr>()) {
      KI.setReqdWorkGroupSize(RA->getXDim(), RA->getYDim(), RA->getZDim());

      // Find the maximum dimension.
      if (RA->getZDim() > 1)      KI.setMaxDim(3);
      else if (RA->getYDim() > 1) KI.setMaxDim(2);
      else if (RA->getXDim() > 1) KI.setMaxDim(1);
      else if (RA->getXDim() == 1 && RA->getYDim() == 1 && RA->getZDim() == 1)
        KI.setMaxDim(KernelInfo::TASKDIM);
    }
    // __attribute__((work_group_size_hint(X,Y,Z)))
    if (WorkGroupSizeHintAttr *WA = FD->getAttr<WorkGroupSizeHintAttr>()) {
      KI.setWorkGroupSizeHint(WA->getXDim(), WA->getYDim(), WA->getZDim());
    }

    // attribute strings
    if (FD->hasAttrs()) {
      AttrVec &Attrs = FD->getAttrs();
      for (AttrVec::iterator AI = Attrs.begin(), AE = Attrs.end();
           AI != AE; ++AI) {
        Attr *A = *AI;
        string AttrStr = A->getAsString();
        if (AttrStr != "") KI.pushAttrStr(AttrStr);
      }
    }

    // estimation of local memory usage
    TransformLocalVD TLV(*ASTCtx, CLOpts, *CLExprs, FD, LocalVDSet);
    KI.setLocalMemSize(TLV.getLocalMemSize());

    // private memory usage
    KI.setPrivateMemSize(TLV.getPrivateMemSize());
  } //end for
}

void CLTranslator::TransformAST() {
  if (CLOpts.NoOpt) {
    TransformLocalVDName();
    return;
  }

  // Analyze all FunctionDecls.
  PrepareTransformation();

  // Duplicate the kernel function's FunctionDecl. 
  DuplicateKernelFunctionDecls();

  //-------------------------------------------------------------------------
  // Main trasformations
  //-------------------------------------------------------------------------
  DeclContext *DeclCtx = ASTCtx->getTranslationUnitDecl();
  for (DeclContext::decl_iterator I = DeclCtx->decls_begin(),
                                  E = DeclCtx->decls_end();
       I != E; ++I) {
    FunctionDecl *FD = dyn_cast<FunctionDecl>(*I);
    if (!FD) {
      if (VarDecl *VD = dyn_cast<VarDecl>(*I)) {
        if (!VD->hasInit()) continue;

        QualType VDTy = VD->getType();
        if (VDTy->isExtVectorType()) {
          TransformVector TV(*ASTCtx, CLOpts, *CLExprs);
          VD->setInit(TV.TransformExpr(VD->getInit()));
        } else if (CLUtils::IsVectorArrayType(VDTy)) {
          TransformVector TV(*ASTCtx, CLOpts, *CLExprs, true);
          VD->setInit(TV.TransformExpr(VD->getInit()));
        }
      }
      continue;
    } else if (!FD->isThisDeclarationADefinition() || !FD->hasBody() ||
               FD->isFullyInlined())
      continue;

    // Basic Transformation
    // - function call specialization
    // - integer division
    // - vector transformation
    TransformBasic TB(*ASTCtx, CLOpts, *CLExprs);
    TB.Transform(FD);

    // Function Duplication
    // Duplicate kernel functions that are called by other functions.
    // Duplicated has to be done before work-item coalescing because the
    // original code needs to be kept.
    if (FD->getDuplication()) {
      FunctionDuplicator FDup(*ASTCtx, *CLExprs);
      FDup.Duplicate(FD);
    }

    // Coalescing
    if (CLUtils::IsKernel(FD)) {
      TransformCoalescing TC(*ASTCtx, *CLExprs, Out);
      TC.Transform(FD, KernelMap[FD]);

      KernelMap[FD].setCoalescingApplied(TC.isTransformed());

      // Add TLB_GET_KEY; to the beginning of the function body.
      // Add TLB_GET_KEY_LOCAL;
      InsertTlbGetKey(FD);
    }
  }

  // Change the name of __local variables.
  TransformLocalVDName();
}

void CLTranslator::GenerateCode() {
  // Print the header code
  PrintHeaderCode();

  // Print the translated code
  CLDeclPrinter Printer(Out, *ASTCtx, *Policy);
  Printer.VisitTranslationUnitDecl(ASTCtx->getTranslationUnitDecl());

  // Print the footer code
  PrintFooterCode();
}
//===--------------------------------------------------------------------===//


//===--------------------------------------------------------------------===//
void CLTranslator::InsertTlbGetKey(FunctionDecl *FD) {
  StmtVector Stmts;

  // TLB_GET_GEY;
  Stmts.push_back(CLExprs->getExpr(CLExpressions::TLB_GET_KEY));

  // TLB_GET_KEY_LOCAL;
  assert(KernelMap.find(FD) != KernelMap.end() && "Not a kernel?");
  if (CLOpts.UseTLBLocal) {
    KernelInfo &KI = KernelMap[FD];
    if (KI.getLocalMemSize() > 0) {
      Stmts.push_back(CLExprs->getExpr(CLExpressions::TLB_GET_KEY_LOCAL));
    }
  }

  CompoundStmt *FDBody = dyn_cast<CompoundStmt>(FD->getBody());
  assert(FDBody && "Function body must be a CompoundStmt");

  CompoundStmt::body_iterator I, E;
  for (I = FDBody->body_begin(), E = FDBody->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }

  FDBody->setStmts(*ASTCtx, Stmts.data(), Stmts.size());
}

void CLTranslator::TransformLocalVDName() {
  if (LocalVDSet.size() == 0) return;

  IdentifierTable &Idents = ASTCtx->Idents;

  unsigned num = 0;
  VarDeclSetTy::iterator it, end;
  for (it = LocalVDSet.begin(), end = LocalVDSet.end(); it != end; ++it) {
    VarDecl *VD = *it;
    stringstream Name;
    Name << OPENCL_LOCAL_VAR_PREFIX << num++ << "_" << VD->getNameAsString();

    string NameStr = Name.str();
    StringRef declName(NameStr);
    IdentifierInfo &declID = Idents.get(declName);

    VD->setDeclName(&declID);
  }
}
//===--------------------------------------------------------------------===//


//===--------------------------------------------------------------------===//
/// Function Analysis
//===--------------------------------------------------------------------===//
void CLTranslator::PrepareTransformation() {
  bool IsGotoUsed = false;
  bool IsBarrierUsed = false;

  // Analyze the barrier() call existence building a call graph.
  DeclContext *DeclCtx = ASTCtx->getTranslationUnitDecl();
  for (DeclContext::decl_iterator I = DeclCtx->decls_begin(),
                                  E = DeclCtx->decls_end();
       I != E; ++I) {
    FunctionDecl *FD = dyn_cast<FunctionDecl>(*I);
    if (!FD || !FD->isThisDeclarationADefinition() || !FD->hasBody())
      continue;

    if (CLUtils::IsKernel(FD)) {
      CallMap[FD].setMaxDim(KernelMap[FD].getMaxDim());
    }

    FunctionAnalyzer FAnalyzer(FD, CallMap[FD]);
    FuncDeclSetTy &Callees = CallMap[FD].Callees;

    for (FuncDeclSetTy::iterator CI = Callees.begin(), CE = Callees.end();
         CI != CE; ++CI) {
      FuncDeclSetTy &Callers = CallMap[*CI].Callers;
      Callers.insert(FD);
    }

    IsGotoUsed = IsGotoUsed || FD->hasGotoStmt();
    IsBarrierUsed = IsBarrierUsed || FD->hasBarrierCall();
  }

  // Check the existence of cycles in the call graph.
  // NOTE: OpenCL does not allow the recursion.
  CheckCyclesInCallGraph();

  // Update the possible max. dimension for each function.
  UpdateAllMaxDimension();

  if (IsGotoUsed) {
    // Update HasGotoStmt of all FunctionDecls.
    UpdateAllHasGotoStmt();
  }

  if (IsBarrierUsed) {
    // Update HasIndirectBarrierCall of all FunctionDecls.
    UpdateAllHasIndirectBarrierCall();

    // Inline functions that have set HasIndirectBarrierCall for work-item
    // coalescing.
    InlineFunctions();
  }
}

/// Since the kernel source code normally does not contain many functions,
/// we use a simple colored DFS algorithm for the cycle detection.
void CLTranslator::CheckCyclesInCallGraph() {
  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionCallInfo &CallInfo = (*I).second;
    CallInfo.ResetMark();
  }

  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionCallInfo &V = (*I).second;
    if (V.Mark == FunctionCallInfo::WHITE) {
      if (VisitInCallGraph(V)) {
        FunctionDecl *FD = (*I).first;
        Diag(diag::err_opencl_recursion) << FD->getName();
        HandleError();
      }
    }
  }
}

bool CLTranslator::VisitInCallGraph(FunctionCallInfo &V) {
  V.Mark = FunctionCallInfo::GREY;

  FuncDeclSetTy &Callees = V.Callees;
  for (FuncDeclSetTy::iterator I = Callees.begin(), E = Callees.end();
       I != E; ++I) {
    FunctionCallInfo &U = CallMap[*I];
    if (U.Mark == FunctionCallInfo::GREY) {
      return true;
    } else if (U.Mark == FunctionCallInfo::WHITE) {
      if (VisitInCallGraph(U)) {
        return true;
      }
    }
  }

  V.Mark = FunctionCallInfo::BLACK;
  return false;
}

void CLTranslator::UpdateAllMaxDimension() {
  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionCallInfo &V = (*I).second;
    FuncDeclSetTy &Callers = V.Callers;

    if (V.getMaxDim() > 0 && Callers.size() > 0) {
      UpdateMaxDimension(Callers, V.getMaxDim());
    }
  }

  // Update KernelMap again.
  for (KernelMapTy::iterator I = KernelMap.begin(), E = KernelMap.end();
       I != E; ++I) {
    FunctionDecl *FD = (*I).first;
    KernelInfo &KInfo = (*I).second;
    KInfo.setMaxDim(CallMap[FD].getMaxDim());
  }
}

void CLTranslator::UpdateMaxDimension(FuncDeclSetTy &Callers,
                                    unsigned CalleeMaxDim) {
  for (FuncDeclSetTy::iterator CI = Callers.begin(), CE = Callers.end();
       CI != CE; ++CI) {
    FunctionCallInfo &U = CallMap[*CI];

    if (U.getMaxDim() < CalleeMaxDim) {
      U.setMaxDim(CalleeMaxDim);

      FuncDeclSetTy &FDCallers = U.Callers;
      if (FDCallers.size() > 0) {
        UpdateMaxDimension(FDCallers, CalleeMaxDim);
      }
    }
  }
}

void CLTranslator::UpdateAllHasGotoStmt() {
  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionCallInfo &CallInfo = (*I).second;
    CallInfo.ResetMark();
  }

  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionDecl *FD = (*I).first;
    FunctionCallInfo &V = (*I).second;
    FuncDeclSetTy &Callers = V.Callers;

    if (FD->hasGotoStmt() && Callers.size() > 0 && 
        V.Mark == FunctionCallInfo::WHITE) {
      UpdateHasGotoStmt(Callers);

      // Mark this node as black because it will not be changed any more.
      V.Mark = FunctionCallInfo::BLACK;
    }
  }
}

void CLTranslator::UpdateHasGotoStmt(FuncDeclSetTy &Callers) {
  for (FuncDeclSetTy::iterator CI = Callers.begin(), CE = Callers.end();
       CI != CE; ++CI) {
    FunctionDecl *FD = *CI;
    FunctionCallInfo &U = CallMap[FD];

    if (U.Mark == FunctionCallInfo::WHITE) {
      // Update all redeclarations of this FunctionDecl.
      for (FunctionDecl::redecl_iterator FI = FD->redecls_begin(), 
           FE = FD->redecls_end(); FI != FE; ++FI) {
        (*FI)->setHasGotoStmt(true);
      }

      FuncDeclSetTy &FDCallers = U.Callers;
      if (FDCallers.size() > 0) {
        UpdateHasGotoStmt(FDCallers);
      }

      U.Mark = FunctionCallInfo::BLACK;
    }
  }
}

/// Set HasIndirectBarrierCall of each FunctionDecl as true if it invokes 
/// a function that has barrier() calls. This setting is propagated to all 
/// callers.
void CLTranslator::UpdateAllHasIndirectBarrierCall() {
  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionCallInfo &CallInfo = (*I).second;
    CallInfo.ResetMark();
  }

  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionDecl *FD = (*I).first;
    FunctionCallInfo &V = (*I).second;
    FuncDeclSetTy &Callers = V.Callers;

    if ((FD->hasBarrierCall() || FD->hasIndirectBarrierCall()) && 
        Callers.size() > 0 && V.Mark == FunctionCallInfo::WHITE) {
      UpdateHasIndirectBarrierCall(Callers);

      // Mark this node as black because it will not be changed any more.
      V.Mark = FunctionCallInfo::BLACK;
    }
  }
}

void CLTranslator::UpdateHasIndirectBarrierCall(FuncDeclSetTy &Callers) {
  for (FuncDeclSetTy::iterator CI = Callers.begin(), CE = Callers.end();
       CI != CE; ++CI) {
    FunctionDecl *FD = *CI;
    FunctionCallInfo &U = CallMap[FD];

    if (U.Mark == FunctionCallInfo::WHITE) {
      // Update all redeclarations of this FunctionDecl.
      for (FunctionDecl::redecl_iterator FI = FD->redecls_begin(), 
           FE = FD->redecls_end(); FI != FE; ++FI) {
        (*FI)->setIndirectBarrierCall(true);
      }

      FuncDeclSetTy &FDCallers = U.Callers;
      if (FDCallers.size() > 0) {
        UpdateHasIndirectBarrierCall(FDCallers);
      }

      U.Mark = FunctionCallInfo::BLACK;
    }
  }
}

/// Function Inlining
void CLTranslator::InlineFunctions() {
  DeclContext *DeclCtx = ASTCtx->getTranslationUnitDecl();
  for (DeclContext::decl_iterator I = DeclCtx->decls_begin(),
                                  E = DeclCtx->decls_end();
       I != E; ++I) {
    FunctionDecl *FD = dyn_cast<FunctionDecl>(*I);
    if (!FD || !FD->isThisDeclarationADefinition() || !FD->hasBody() ||
        !CLUtils::IsKernel(FD))
      continue;

    // Now, FD is a FunctionDecl for the kernel function.
    // Inline functions if FD's HasIndirectBarrierCalls is set.
    if (FD->hasIndirectBarrierCall()) {
      FunctionInliner FIn(*ASTCtx, *CLExprs, CallMap);
      FIn.Inline(FD);
    }
  }

  UpdateFullyInlined();
}

void CLTranslator::UpdateFullyInlined() {
  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionDecl *FD = (*I).first;
    FuncDeclSetTy &Callers = (*I).second.Callers;
    if (!CLUtils::IsKernel(FD) && Callers.size() == 0) {
      // Update all redeclarations of this FunctionDecl.
      for (FunctionDecl::redecl_iterator FI = FD->redecls_begin(), 
           FE = FD->redecls_end(); FI != FE; ++FI) {
        (*FI)->setFullyInlined(true);
      }
    }
  }
}

// Duplicate a FunctionDecl of the kernel function that is called by other 
// functions.
void CLTranslator::DuplicateKernelFunctionDecls() {
  DeclContext *DeclCtx = ASTCtx->getTranslationUnitDecl();
  for (DeclContext::decl_iterator I = DeclCtx->decls_begin(),
                                  E = DeclCtx->decls_end();
       I != E; ++I) {
    FunctionDecl *FD = dyn_cast<FunctionDecl>(*I);
    if (!FD || !FD->isThisDeclarationADefinition() || !FD->hasBody() ||
        !CLUtils::IsKernel(FD))
      continue;

    FunctionCallInfo &V = CallMap[FD];
    if (V.Callers.size() > 0) {
      // The kernel function is called by other functions.
      stringstream Name;
      Name << OPENCL_DUP_FUN_PREFIX << FD->getNameAsString();

      string NameStr = Name.str();
      StringRef DeclName(NameStr);
      IdentifierInfo &DeclID = ASTCtx->Idents.get(DeclName);

      FunctionDecl *DupFD = FunctionDecl::Create(*ASTCtx, DeclCtx,
          FD->getLocation(),
          DeclarationName(&DeclID),
          FD->getType(),
          FD->getTypeSourceInfo(),
          FD->getStorageClass(),
          FD->getStorageClassAsWritten(),
          FD->isInlineSpecified());

      // Update all redeclarations of this FunctionDecl.
      for (FunctionDecl::redecl_iterator FI = FD->redecls_begin(), 
           FE = FD->redecls_end(); FI != FE; ++FI) {
        (*FI)->setDuplication(DupFD);
      }
    }
  }
}
//===--------------------------------------------------------------------===//


//===--------------------------------------------------------------------===//
void CLTranslator::PrintHeaderCode() {
  if (CLOpts.UseTLBLocal)
    Out << "#define USE_TLB_LOCAL\n";

  Out << "#include <cpu_kernel_header.h>\n\n";

  if (CLOpts.UseTLBLocal) {
    // Print __local variable declarations
    PrintLocalVarDecls();

    Out << "TLB_INIT\n\n";
  }
}

void CLTranslator::PrintLocalVarDecls() {
  Out << "TLB_STR_DATA_LOCAL_START\n";

  VarDeclSetTy::iterator i, e;
  for (i = LocalVDSet.begin(), e = LocalVDSet.end(); i != e; ++i) {
    VarDecl *VD = *i;
    string Name = VD->getNameAsString();
    CLUtils::GetTypeStrWithoutCVR(VD->getType(), Name, *Policy);
    Out << "  TLS_STATIC " << Name << ";\n";
  }

  Out << "TLB_STR_DATA_LOCAL_END\n\n";
}
//===--------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
void CLTranslator::PrintFooterCode() {
  // Print kernel dependent code and kernel information
  Out << "#if 0\n";
  Out << KernelMap.size() << "\n";
  if (!KernelMap.empty())
    PrintKernelDependentCode();
  PrintKernelInfoForMerge();
  Out << "#endif\n";
  PrintKernelInfo();
}

/// Kernel dependent code - function info., launch, etc.
void CLTranslator::PrintKernelDependentCode() {
  PrintKernelNamesArray();
  PrintFunctionStructs();
  PrintKernelLaunch();
  PrintKernelSetStack();
  PrintKernelSetArguments();
  PrintKernelFini();
}

void CLTranslator::PrintKernelNamesArray() {
  KernelMapTy::iterator I, E;
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"" << (*I).second.getName() << "\"";
  }
  Out << "\n";
}

void CLTranslator::PrintFunctionStructs() {
  KernelMapTy::iterator I, E;

  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    string StrKernelProto;

    Out << "TLS struct Function_# {";
    for (unsigned p = 0; p < KI.getNumParams(); p++) {
      string StrParamType;
      string StrParamName = KI.getParamName(p);
      QualType Ty = KI.getParamType(p);

      // prototype for function pointer
      if (p) StrKernelProto += ", ";
      string TypeName;
      Ty.getAsStringInternal(TypeName, *Policy);
      StrKernelProto += TypeName;

      // type without CVR + param_name
      Ty.getAsStringInternal(StrParamName, *Policy);
      Out << " " << StrParamName << ";";
    }
    Out << " void(*" << OPENCL_KERNEL_FP << ")(" << StrKernelProto << ");";
    Out << " } func_#;\n";
  }
}

void CLTranslator::PrintKernelLaunch() {
  KernelMapTy::iterator I, E;
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    Out << "    case #: (TLB_GET_FUNC(func_#))." 
        << OPENCL_KERNEL_FP << "(";
    for (unsigned p = 0; p < KI.getNumParams(); p++) {
      if (p) Out << ", ";
      Out << "(TLB_GET_FUNC(func_#))." << KI.getParamName(p);
    }
    Out << "); break;\n";
  }
}

void CLTranslator::PrintKernelSetStack() {
  KernelMapTy::iterator I, E;
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    Out << "    case #:\n";
  }
}

void CLTranslator::PrintKernelSetArguments() {
  KernelMapTy::iterator I, E;
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    Out << "    case #: {"
        << " (TLB_GET_FUNC(func_#))." << OPENCL_KERNEL_FP 
        << " = " << KI.getName() << ";";

    for (unsigned p = 0; p < KI.getNumParams(); p++) {
      QualType ParamTy = KI.getParamType(p);
      string StrParamType;
      ParamTy.getAsStringInternal(StrParamType, *Policy);
      string ParamName = KI.getParamName(p);
      unsigned AQ = KI.getParamAddrQualifier(p);
      if (AQ == OPENCL_N_GLOBAL || AQ == OPENCL_N_CONSTANT)
        Out << " CL_SET_ARG_GLOBAL";
      else if (AQ == OPENCL_N_LOCAL)
        Out << " CL_SET_ARG_LOCAL";
      else 
        Out << " CL_SET_ARG_PRIVATE";
      Out << "(TLB_GET_FUNC(func_#), " << StrParamType << ", " 
          << ParamName << ");";
    }
    Out << " } break;\n";
  }
}

void CLTranslator::PrintKernelFini() {
  KernelMapTy::iterator I, E;
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    Out << "    case #:";
    for (unsigned p = 0; p < KI.getNumParams(); p++) {
      if (KI.getParamAddrQualifier(p) == OPENCL_N_LOCAL)
        Out << " CL_FREE_LOCAL(TLB_GET_FUNC(func_#), " 
            << KI.getParamName(p) << ");";
    }
    Out << " break;\n";
  }
}

void CLTranslator::PrintKernelInfoForMerge() {
  KernelMapTy::iterator I, E;

  ///////////////////////////////////////////////////////////////////////////
  // clGetKernelInfo()
  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_names[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"" << (*I).second.getName() << "\"";
  }
  Out << "\n";

  // _cl_kernel_num_args[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getNumParams();
  }
  Out << "\n";

  // _cl_kernel_attributes[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (int i = 0, e = KI.getNumAttrs(); i < e; i++) {
      if (i) Out << " ";
      Out << KI.getAttrStr(i);
    }
    Out << "\"";
  }
  Out << "\n";

  ///////////////////////////////////////////////////////////////////////////
  // clGetKernelWorkGroupInfo()
  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_work_group_size_hint[][3]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    if (I != KernelMap.begin()) Out << ", ";
    Out << "{" << KI.getWorkDimX() << ", " << KI.getWorkDimY() << ", " 
        << KI.getWorkDimZ() << "}";
  }
  Out << "\n";

  // _cl_kernel_reqd_work_group_size[][3]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    if (I != KernelMap.begin()) Out << ", ";
    Out << "{" << KI.getReqdDimX() << ", " << KI.getReqdDimY() << ", "
        << KI.getReqdDimZ() << "}";
  }
  Out << "\n";

  // _cl_kernel_local_mem_size[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getLocalMemSize();
  }
  Out << "\n";

  // _cl_kernel_private_mem_size[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getPrivateMemSize();
  }
  Out << "\n";

  ///////////////////////////////////////////////////////////////////////////
  // clGetKernelArgInfo()
  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_arg_address_qualifer[]
  // CL_KERNEL_ARG_ADDRESS_GLOBAL  : 3
  // CL_KERNEL_ARG_ADDRESS_CONSTANT: 2
  // CL_KERNEL_ARG_ADDRESS_LOCAL   : 1
  // CL_KERNEL_ARG_ADDRESS_PRIVATE : 0
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      Out << KI.getParamAddrQualifier(p);
    }
    Out << "\"";
  }
  Out << "\n";

  // _cl_kernel_arg_access_qualifier[]
  // CL_KERNEL_ARG_ACCESS_READ_ONLY : 3
  // CL_KERNEL_ARG_ACCESS_WRITE_ONLY: 2
  // CL_KERNEL_ARG_ACCESS_READ_WRITE: 1
  // CL_KERNEL_ARG_ACCESS_NONE      : 0
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      Out << KI.getParamAccessQualifier(p);
    }
    Out << "\"";
  }
  Out << "\n";

  // _cl_kernel_arg_type_name[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      if (p) Out << "\", \"";
      string TypeName;
      CLUtils::GetTypeStrWithoutCVR(KI.getParamType(p), TypeName, *Policy);

      // check modified type names.
      string SubTypeName = TypeName.substr(0, 5);
      if (SubTypeName == "schar" || SubTypeName == "llong")
        Out << TypeName.substr(1);
      else if (TypeName.substr(0, 6) == "ullong")
        Out << "ulong" << TypeName.substr(6);
      else
        Out << TypeName;
    }
    Out << "\"";
  }
  Out << "\n";

  // _cl_kernel_arg_type_qualifier[]
  // CL_KERNEL_ARG_TYPE_NONE        0
  // CL_KERNEL_ARG_TYPE_CONST       (1 << 0)
  // CL_KERNEL_ARG_TYPE_RESTRICT    (1 << 1)
  // CL_KERNEL_ARG_TYPE_VOLATILE    (1 << 2)
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    for (unsigned p = 0; p < ParamNum; p++) {
      if (p) Out << ", ";
      Out << KI.getParamTypeQualifier(p);
    }
  }
  Out << "\n";

  // _cl_kernel_arg_name[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      if (p) Out << "\", \"";
      Out << KI.getParamName(p);
    }
    Out << "\"";
  }
  Out << "\n";

  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_dim[]
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getMaxDim();
  }
  Out << "\n";
}

void CLTranslator::PrintKernelInfo() {
  KernelMapTy::iterator I, E;
  const string PREFIX = "/* SNUCL_KERNEL_COMPILE_INFO */const ";

  Out << "#if 0\n";

  ///////////////////////////////////////////////////////////////////////////
  // clGetKernelInfo()
  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_num
  Out << PREFIX << "unsigned int _cl_kernel_num = " << KernelMap.size()
      << ";\n";

  // _cl_kernel_names[]
  Out << PREFIX << "char* _cl_kernel_names[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"" << (*I).second.getName() << "\"";
  }
  Out << "};\n";

  // _cl_kernel_num_args[]
  Out << PREFIX << "unsigned int _cl_kernel_num_args[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getNumParams();
  }
  Out << "};\n";

  // _cl_kernel_attributes[]
  Out << PREFIX << "char* _cl_kernel_attributes[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (int i = 0, e = KI.getNumAttrs(); i < e; i++) {
      if (i) Out << " ";
      Out << KI.getAttrStr(i);
    }
    Out << "\"";
  }
  Out << "};\n";

  ///////////////////////////////////////////////////////////////////////////
  // clGetKernelWorkGroupInfo()
  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_work_group_size_hint[][3]
  Out << PREFIX << "unsigned int _cl_kernel_work_group_size_hint[][3] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    if (I != KernelMap.begin()) Out << ", ";
    Out << "{" << KI.getWorkDimX() << ", " << KI.getWorkDimY() << ", " 
        << KI.getWorkDimZ() << "}";
  }
  Out << "};\n";

  // _cl_kernel_reqd_work_group_size[][3]
  Out << PREFIX << "unsigned int _cl_kernel_reqd_work_group_size[][3] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    if (I != KernelMap.begin()) Out << ", ";
    Out << "{" << KI.getReqdDimX() << ", " << KI.getReqdDimY() << ", "
        << KI.getReqdDimZ() << "}";
  }
  Out << "};\n";

  // _cl_kernel_local_mem_size[]
  Out << PREFIX << "unsigned long long _cl_kernel_local_mem_size[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getLocalMemSize();
  }
  Out << "};\n";

  // _cl_kernel_private_mem_size[]
  Out << PREFIX << "unsigned long long _cl_kernel_private_mem_size[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getPrivateMemSize();
  }
  Out << "};\n";

  ///////////////////////////////////////////////////////////////////////////
  // clGetKernelArgInfo()
  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_arg_address_qualifer[]
  // CL_KERNEL_ARG_ADDRESS_GLOBAL  : 3
  // CL_KERNEL_ARG_ADDRESS_CONSTANT: 2
  // CL_KERNEL_ARG_ADDRESS_LOCAL   : 1
  // CL_KERNEL_ARG_ADDRESS_PRIVATE : 0
  Out << PREFIX << "char* _cl_kernel_arg_address_qualifier[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      Out << KI.getParamAddrQualifier(p);
    }
    Out << "\"";
  }
  Out << "};\n";

  // _cl_kernel_arg_access_qualifier[]
  // CL_KERNEL_ARG_ACCESS_READ_ONLY : 3
  // CL_KERNEL_ARG_ACCESS_WRITE_ONLY: 2
  // CL_KERNEL_ARG_ACCESS_READ_WRITE: 1
  // CL_KERNEL_ARG_ACCESS_NONE      : 0
  Out << PREFIX << "char* _cl_kernel_arg_access_qualifier[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      Out << KI.getParamAccessQualifier(p);
    }
    Out << "\"";
  }
  Out << "};\n";

  // _cl_kernel_arg_type_name[]
  Out << PREFIX << "char* _cl_kernel_arg_type_name[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      if (p) Out << "\", \"";
      string TypeName;
      CLUtils::GetTypeStrWithoutCVR(KI.getParamType(p), TypeName, *Policy);

      // check modified type names.
      string SubTypeName = TypeName.substr(0, 5);
      if (SubTypeName == "schar" || SubTypeName == "llong")
        Out << TypeName.substr(1);
      else if (TypeName.substr(0, 6) == "ullong")
        Out << "ulong" << TypeName.substr(6);
      else
        Out << TypeName;
    }
    Out << "\"";
  }
  Out << "};\n";

  // _cl_kernel_arg_type_qualifier[]
  // CL_KERNEL_ARG_TYPE_NONE        0
  // CL_KERNEL_ARG_TYPE_CONST       (1 << 0)
  // CL_KERNEL_ARG_TYPE_RESTRICT    (1 << 1)
  // CL_KERNEL_ARG_TYPE_VOLATILE    (1 << 2)
  Out << PREFIX << "unsigned int _cl_kernel_arg_type_qualifier[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    for (unsigned p = 0; p < ParamNum; p++) {
      if (p) Out << ", ";
      Out << KI.getParamTypeQualifier(p);
    }
  }
  Out << "};\n";

  // _cl_kernel_arg_name[]
  Out << PREFIX << "char* _cl_kernel_arg_name[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    KernelInfo &KI = (*I).second;
    unsigned ParamNum = KI.getNumParams();
    if (I != KernelMap.begin()) Out << ", ";
    Out << "\"";
    for (unsigned p = 0; p < ParamNum; p++) {
      if (p) Out << "\", \"";
      Out << KI.getParamName(p);
    }
    Out << "\"";
  }
  Out << "};\n";

  ///////////////////////////////////////////////////////////////////////////
  // _cl_kernel_dim[]
  Out << PREFIX << "unsigned int _cl_kernel_dim[] = {";
  for (I = KernelMap.begin(), E = KernelMap.end(); I != E; ++I) {
    if (I != KernelMap.begin()) Out << ", ";
    Out << (*I).second.getMaxDim();
  }
  Out << "};\n";

  Out << "#endif\n";
}
//===--------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
DiagnosticBuilder CLTranslator::Diag(unsigned DiagID) {
  return ASTCtx->getDiagnostics().Report(DiagID);
}

void CLTranslator::HandleError() {
  exit(-1);
}
//===--------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
// Debugging functions
//===--------------------------------------------------------------------===//
void CLTranslator::printCallMap(CallMapTy &CallMap) {
  for (CallMapTy::iterator I = CallMap.begin(), E = CallMap.end();
       I != E; ++I) {
    FunctionDecl *FD = (*I).first;
    Out << FD->getName() << ":\n";
    
    FuncDeclSetTy &Callers = (*I).second.Callers;
    Out << "  Callers={ ";
    for (FuncDeclSetTy::iterator CI = Callers.begin(), CE = Callers.end();
         CI != CE; ) {
      Out << (*CI)->getName();
      if (++CI != CE) Out << ", ";
    }
    Out << " }\n";

    FuncDeclSetTy &Callees = (*I).second.Callees;
    Out << "  Callees={ "; 
    for (FuncDeclSetTy::iterator CI = Callees.begin(), CE = Callees.end();
         CI != CE; ) {
      Out << (*CI)->getName();
      if (++CI != CE) Out << ", ";
    }
    Out << " }\n";
  }
  Out << "\n\n";
}

