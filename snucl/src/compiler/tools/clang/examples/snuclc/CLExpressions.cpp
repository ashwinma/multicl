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

#include "clang/AST/Decl.h"
#include "CLExpressions.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

#include <algorithm>
#include <cctype>
using std::string;

/// Constructor
CLExpressions::CLExpressions(ASTContext &C, DeclContext *D)
  : ASTCtx(C), DeclCtx(D), Idents(C.Idents) {
  NeedTLBUse = true;
  TmpVarNum = 0;
  DummyFD = 0;
}

Expr *CLExpressions::getExpr(ExprKind kind) {
  Expr *E = NULL;

  switch (kind) {
    default: assert(0 && "Unknown or untreated expr kind"); break;

    // software cache
    case DEV_CACHE_RD: 
      E = NewDeclRefExpr(StringRef("dev_cache_rd"));
      break;
    case DEV_CACHE_WR:
      E = NewDeclRefExpr(StringRef("dev_cache_wr"));
      break;
    case DEV_CACHE_WR_OP:
      E = NewDeclRefExpr(StringRef("dev_cache_wr_op"));
      break;
    case DEV_CACHE_POSTINC:
      E = NewDeclRefExpr(StringRef("dev_cache_postinc"));
      break;
    case DEV_CACHE_POSTDEC:
      E = NewDeclRefExpr(StringRef("dev_cache_postdec"));
      break;
    case DEV_CACHE_PREINC:
      E = NewDeclRefExpr(StringRef("dev_cache_preinc"));
      break;
    case DEV_CACHE_PREDEC:
      E = NewDeclRefExpr(StringRef("dev_cache_predec"));
      break;

    // tlb
    case TLB_GET_KEY:
      E = NewDeclRefExpr(StringRef("TLB_GET_KEY"));
      break;
    case TLB_GET:
      E = NewDeclRefExpr(StringRef("TLB_GET"));
      break;
    case TLB_GET_KEY_LOCAL:
      E = NewDeclRefExpr(StringRef("TLB_GET_KEY_LOCAL"));
      break;
    case TLB_GET_LOCAL:
      E = NewDeclRefExpr(StringRef("TLB_GET_LOCAL"));
      break;

    // operator
    case CL_SAFE_INT_DIV_ZERO: 
      E = NewDeclRefExpr(StringRef("__CL_SAFE_INT_DIV_ZERO"));
      break;
    case CL_SAFE_UINT_DIV_ZERO: 
      E = NewDeclRefExpr(StringRef("__CL_SAFE_UINT_DIV_ZERO"));
      break;
    case DIV:
      E = NewDeclRefExpr(StringRef("/"));
      break;
    case REM:
      E = NewDeclRefExpr(StringRef("%"));
      break;
    case DIV_ASSIGN:
      E = NewDeclRefExpr(StringRef("/="));
      break;
    case REM_ASSIGN:
      E = NewDeclRefExpr(StringRef("%="));
      break;

    // operation
    case OP_MUL: E = NewDeclRefExpr(StringRef("mul")); break;
    case OP_DIV: E = NewDeclRefExpr(StringRef("div")); break;
    case OP_REM: E = NewDeclRefExpr(StringRef("rem")); break;
    case OP_ADD: E = NewDeclRefExpr(StringRef("add")); break;
    case OP_SUB: E = NewDeclRefExpr(StringRef("sub")); break;
    case OP_SHL: E = NewDeclRefExpr(StringRef("shl")); break;
    case OP_SHR: E = NewDeclRefExpr(StringRef("shr")); break;
    case OP_AND: E = NewDeclRefExpr(StringRef("and")); break;
    case OP_XOR: E = NewDeclRefExpr(StringRef("xor")); break;
    case OP_OR:  E = NewDeclRefExpr(StringRef("or"));  break;

    // integer literal
    case ZERO:
    case ONE:
    case TWO:
    case THREE:
    case FOUR:
    case FIVE:
    case SIX:
    case SEVEN:
    case EIGHT:
    case NINE:
    case TEN:
    case ELEVEN:
    case TWELVE:
    case THIRTEEN:
    case FOURTEEN:
    case FIFTEEN: E = NewIntegerLiteral(kind - ZERO); break;

    case SELECT:
      E = NewDeclRefExpr(StringRef("select"));
      break;

    case __LOCAL_SIZE:
      E = NewDeclRefExpr(StringRef("__local_size"));
      break;
    case __LOCAL_SIZE_0:
      E = NewArraySubscriptExpr(getExpr(__LOCAL_SIZE), 
                                getExpr(ZERO),
                                ASTCtx.IntTy);
      break;
    case __LOCAL_SIZE_1:
      E = NewArraySubscriptExpr(getExpr(__LOCAL_SIZE), 
                                getExpr(ONE),
                                ASTCtx.IntTy);
      break;
    case __LOCAL_SIZE_2:
      E = NewArraySubscriptExpr(getExpr(__LOCAL_SIZE), 
                                getExpr(TWO),
                                ASTCtx.IntTy);
      break;

    case LOCAL_SIZE_0: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__LOCAL_SIZE_0);
        } else {
          E = getExpr(__LOCAL_SIZE_0);
        }
        break;
      }
    case LOCAL_SIZE_1: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__LOCAL_SIZE_1);
        } else {
          E = getExpr(__LOCAL_SIZE_1);
        }
        break;
      }
    case LOCAL_SIZE_2: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__LOCAL_SIZE_2);
        } else {
          E = getExpr(__LOCAL_SIZE_2);
        }
        break;
      }

    case __GLOBAL_ID:
      E = NewDeclRefExpr(StringRef("__global_id"));
      break;
    case __GLOBAL_ID_0:
      E = NewArraySubscriptExpr(getExpr(__GLOBAL_ID), 
                                getExpr(ZERO),
                                ASTCtx.IntTy);
      break;
    case __GLOBAL_ID_1:
      E = NewArraySubscriptExpr(getExpr(__GLOBAL_ID), 
                                getExpr(ONE),
                                ASTCtx.IntTy);
      break;
    case __GLOBAL_ID_2:
      E = NewArraySubscriptExpr(getExpr(__GLOBAL_ID), 
                                getExpr(TWO),
                                ASTCtx.IntTy);
      break;
    case GLOBAL_ID_0: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__GLOBAL_ID_0);
        } else {
          E = getExpr(__GLOBAL_ID_0);
        }
        break;
      }
    case GLOBAL_ID_1: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__GLOBAL_ID_1);
        } else {
          E = getExpr(__GLOBAL_ID_1);
        }
        break;
      }
    case GLOBAL_ID_2: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__GLOBAL_ID_2);
        } else {
          E = getExpr(__GLOBAL_ID_2);
        }
        break;
      }

    // For WCL
    case __I: E = NewDeclRefExpr(StringRef("__i")); break;
    case __J: E = NewDeclRefExpr(StringRef("__j")); break;
    case __K: E = NewDeclRefExpr(StringRef("__k")); break;
    case WCL_IDX_I: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__I);
        } else {
          E = getExpr(__I);
        }
        break;
      }
    case WCL_IDX_J: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__J);
        } else {
          E = getExpr(__J);
        }
        break;
      }
    case WCL_IDX_K: {
        if (NeedTLBUse) {
          E = getTLBGETExpr(__K);
        } else {
          E = getExpr(__K);
        }
        break;
      }
    case WCL_IDX_I_INIT:
      E = NewBinaryOperator(getExpr(WCL_IDX_I), getExpr(ZERO),
                            BO_Assign, ASTCtx.IntTy);
      break;
    case WCL_IDX_J_INIT:
      E = NewBinaryOperator(getExpr(WCL_IDX_J), getExpr(ZERO),
                            BO_Assign, ASTCtx.IntTy);
      break;
    case WCL_IDX_K_INIT:
      E = NewBinaryOperator(getExpr(WCL_IDX_K), getExpr(ZERO),
                            BO_Assign, ASTCtx.IntTy);
      break;
    case WCL_IDX_I_COND:
      E = NewBinaryOperator(getExpr(WCL_IDX_I), getExpr(LOCAL_SIZE_0),
                            BO_LT, ASTCtx.IntTy);
      break;
    case WCL_IDX_J_COND:
      E = NewBinaryOperator(getExpr(WCL_IDX_J), getExpr(LOCAL_SIZE_1),
                            BO_LT, ASTCtx.IntTy);
      break;
    case WCL_IDX_K_COND:
      E = NewBinaryOperator(getExpr(WCL_IDX_K), getExpr(LOCAL_SIZE_2),
                            BO_LT, ASTCtx.IntTy);
      break;
    case WCL_IDX_I_INC:
      E = NewUnaryOperator(getExpr(WCL_IDX_I), UO_PostInc, ASTCtx.IntTy);
      break;
    case WCL_IDX_J_INC:
      E = NewUnaryOperator(getExpr(WCL_IDX_J), UO_PostInc, ASTCtx.IntTy);
      break;
    case WCL_IDX_K_INC:
      E = NewUnaryOperator(getExpr(WCL_IDX_K), UO_PostInc, ASTCtx.IntTy);
      break;
    case MALLOC: E = NewDeclRefExpr(StringRef("malloc")); break;
    case FREE:   E = NewDeclRefExpr(StringRef("free")); break;
  }

  return E;
}

Expr *CLExpressions::getExpr(QualType type) {
  ExprKind kind = (ExprKind)getExprTypeKind(type);
  string kindName = getExprKindName(kind);
  return NewDeclRefExpr(StringRef(kindName));
}

Expr *CLExpressions::getConvertExpr(QualType type) {
  ExprKind tKind = (ExprKind)getExprTypeKind(type);
  string kindName = "convert_" + getExprKindName(tKind);
  return NewDeclRefExpr(StringRef(kindName));
}

Expr *CLExpressions::getVectorLiteralExpr(QualType type) {
  ExprKind tKind = (ExprKind)getExprTypeKind(type);
  string typeName = getExprKindName(tKind);
  std::transform(typeName.begin(), typeName.end(), typeName.begin(),
      (int(*)(int))std::toupper); 
  string kindName = "__CL_" + typeName;
  return NewDeclRefExpr(StringRef(kindName));
}

Stmt *CLExpressions::getWCL(CompoundStmt *Body, unsigned MaxDim) {
  // ForStmt (WCL)
  SourceLocation SL;
  ForStmt *WCL_I = new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_I_INIT), getExpr(WCL_IDX_I_COND), NULL,
      getExpr(WCL_IDX_I_INC), Body, SL, SL, SL); 
  if (MaxDim == 1) return WCL_I;

  Stmt *Stmts[1] = { WCL_I };
  Body = new (ASTCtx) CompoundStmt(ASTCtx, Stmts, 1, SL, SL);
  ForStmt *WCL_J = new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_J_INIT), getExpr(WCL_IDX_J_COND), NULL,
      getExpr(WCL_IDX_J_INC), Body, SL, SL, SL); 
  if (MaxDim == 2) return WCL_J;

  Stmts[0] = WCL_J;
  Body = new (ASTCtx) CompoundStmt(ASTCtx, Stmts, 1, SL, SL);
  ForStmt *WCL_K = new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_K_INIT), getExpr(WCL_IDX_K_COND), NULL,
      getExpr(WCL_IDX_K_INC), Body, SL, SL, SL); 

  return WCL_K;
}

Stmt *CLExpressions::getInnerWCL(CompoundStmt *body) {
  // ForStmt (WCL)
  SourceLocation SL;
  ForStmt *wclI = new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_I_INIT), getExpr(WCL_IDX_I_COND), NULL,
      getExpr(WCL_IDX_I_INC), body, SL, SL, SL); 
  return wclI;
}

Stmt *CLExpressions::getLoopK(CompoundStmt *body) {
  SourceLocation SL;
  return new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_K_INIT), getExpr(WCL_IDX_K_COND), NULL,
      getExpr(WCL_IDX_K_INC), body, SL, SL, SL); 
}

Stmt *CLExpressions::getLoopJ(CompoundStmt *body) {
  SourceLocation SL;
  return new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_J_INIT), getExpr(WCL_IDX_J_COND), NULL,
      getExpr(WCL_IDX_J_INC), body, SL, SL, SL); 
}

Stmt *CLExpressions::getLoopKJ(CompoundStmt *body) {
  SourceLocation SL;
  ForStmt *wclJ = new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_J_INIT), getExpr(WCL_IDX_J_COND), NULL,
      getExpr(WCL_IDX_J_INC), body, SL, SL, SL); 

  Stmt *stmts[1] = { wclJ };
  body = new (ASTCtx) CompoundStmt(ASTCtx, stmts, 1, SL, SL);
  ForStmt *wclK = new (ASTCtx) ForStmt(
      ASTCtx, getExpr(WCL_IDX_K_INIT), getExpr(WCL_IDX_K_COND), NULL,
      getExpr(WCL_IDX_K_INC), body, SL, SL, SL); 

  return wclK;
}

/// A dummy CallExpr is used instead of the inlined function call.
Expr *CLExpressions::getDummyCallExpr() {
  if (!DummyFD) {
    StringRef Name("CL_DUMMY_FUNCTION");
    IdentifierInfo &II = Idents.get(Name);
    DeclarationName N(&II);
    DummyFD = FunctionDecl::Create(ASTCtx, DeclCtx, SourceLocation(),
        N, ASTCtx.VoidTy, ASTCtx.getNullTypeSourceInfo());
  }

  DeclRefExpr *Fn = new (ASTCtx) DeclRefExpr(DummyFD, DummyFD->getType(),
      VK_RValue, SourceLocation());
  return NewCallExpr(Fn, NULL, 0, Fn->getType());
}


//---------------------------------------------------------------------------
// private functions
//---------------------------------------------------------------------------
unsigned CLExpressions::getExprTypeKind(QualType type) {
  // size_t, ptrdiff_t
  string typeName = type.getAsString();
  if (typeName == "size_t") return ULONG;
  else if (typeName == "ptrdiff_t") return LONG;

  if (const BuiltinType *BT = type->getAs<BuiltinType>()) {
    switch (BT->getKind()) {
      default: assert(0 && "Unknown builtin type!");
      case BuiltinType::Char_S:
      case BuiltinType::Char_U:
      case BuiltinType::SChar:    return CHAR;
      case BuiltinType::UChar:    return UCHAR;
      case BuiltinType::Short:    return SHORT;
      case BuiltinType::UShort:   return USHORT;
      case BuiltinType::Bool:     return BOOL;
      case BuiltinType::Int:      return INT;
      case BuiltinType::UInt:     return UINT;
      case BuiltinType::Long:     return LLONG;
      case BuiltinType::ULong:    return ULLONG;
      case BuiltinType::Half:     return HALF;
      case BuiltinType::Float:    return FLOAT;
      case BuiltinType::Double:   return DOUBLE;
    }
  } else if (type->isExtVectorType()) {
    const ExtVectorType *EVT = type->getAs<ExtVectorType>();
    QualType elemType = EVT->getElementType();
    if (const BuiltinType *BT = elemType->getAs<BuiltinType>()) {
      ExprKind baseKind;
      switch (BT->getKind()) {
        default: assert(0 && "Unknown vector element type!");
        case BuiltinType::Char_S:
        case BuiltinType::Char_U:
        case BuiltinType::SChar:    baseKind = CHAR2; break;
        case BuiltinType::UChar:    baseKind = UCHAR2; break;
        case BuiltinType::Short:    baseKind = SHORT2; break;
        case BuiltinType::UShort:   baseKind = USHORT2; break;
        case BuiltinType::Int:      baseKind = INT2; break;
        case BuiltinType::UInt:     baseKind = UINT2; break;
        case BuiltinType::Long:     baseKind = LONG2; break;
        case BuiltinType::ULong:    baseKind = ULONG2; break;
        case BuiltinType::Float:    baseKind = FLOAT2; break;
        case BuiltinType::Double:   baseKind = DOUBLE2; break;
      }

      unsigned numElems = EVT->getNumElements();
      if (numElems <= 2)       return baseKind;
      else if (numElems == 3)  return baseKind + 1;
      else if (numElems == 4)  return baseKind + 2;
      else if (numElems <= 8)  return baseKind + 3;
      else if (numElems <= 16) return baseKind + 4;
    }
  }

  return UINT;
}

string CLExpressions::getExprKindName(ExprKind kind) {
  switch (kind) {
    default:        return "";
    case CHAR:      return "schar";
    case UCHAR:     return "uchar";
    case SHORT:     return "short";
    case USHORT:    return "ushort";
    case BOOL:      return "bool";
    case INT:       return "int";
    case UINT:      return "uint";
    case LONG:      return "long";
    case ULONG:     return "ulong";
    case LLONG:     return "llong";
    case ULLONG:    return "ullong";
    case HALF:      return "half";
    case FLOAT:     return "float";
    case DOUBLE:    return "double";
    case CHAR2:     return "char2";
    case CHAR3:     return "char3";
    case CHAR4:     return "char4";
    case CHAR8:     return "char8";
    case CHAR16:    return "char16";
    case UCHAR2:    return "uchar2";
    case UCHAR3:    return "uchar3";
    case UCHAR4:    return "uchar4";
    case UCHAR8:    return "uchar8";
    case UCHAR16:   return "uchar16";
    case SHORT2:    return "short2";
    case SHORT3:    return "short3";
    case SHORT4:    return "short4";
    case SHORT8:    return "short8";
    case SHORT16:   return "short16";
    case USHORT2:   return "ushort2";
    case USHORT3:   return "ushort3";
    case USHORT4:   return "ushort4";
    case USHORT8:   return "ushort8";
    case USHORT16:  return "ushort16";
    case INT2:      return "int2";
    case INT3:      return "int3";
    case INT4:      return "int4";
    case INT8:      return "int8";
    case INT16:     return "int16";
    case UINT2:     return "uint2";
    case UINT3:     return "uint3";
    case UINT4:     return "uint4";
    case UINT8:     return "uint8";
    case UINT16:    return "uint16";
    case LONG2:     return "long2";
    case LONG3:     return "long3";
    case LONG4:     return "long4";
    case LONG8:     return "long8";
    case LONG16:    return "long16";
    case ULONG2:    return "ulong2";
    case ULONG3:    return "ulong3";
    case ULONG4:    return "ulong4";
    case ULONG8:    return "ulong8";
    case ULONG16:   return "ulong16";
    case FLOAT2:    return "float2";
    case FLOAT3:    return "float3";
    case FLOAT4:    return "float4";
    case FLOAT8:    return "float8";
    case FLOAT16:   return "float16";
    case DOUBLE2:   return "double2";
    case DOUBLE3:   return "double3";
    case DOUBLE4:   return "double4";
    case DOUBLE8:   return "double8";
    case DOUBLE16:  return "double16";
  }
  return "";
}

Expr *CLExpressions::NewDeclRefExpr(StringRef Name) {
  SourceLocation  SL;
  IdentifierInfo &id = Idents.get(Name);
  QualType        type = ASTCtx.IntTy;
  TypeSourceInfo *tsi = ASTCtx.getNullTypeSourceInfo();
  StorageClass    sc = SC_None;
  VarDecl *declVD;

  declVD = VarDecl::Create(ASTCtx, DeclCtx, SL, &id, type, tsi, sc, sc);
  return new (ASTCtx) DeclRefExpr(declVD, type, VK_RValue, SL);
}

Expr *CLExpressions::NewUnaryOperator(Expr *input, UnaryOperator::Opcode opc,
                                      QualType t) {
  SourceLocation SL;
  ExprValueKind vk = VK_RValue;
  return new (ASTCtx) UnaryOperator(input, opc, t, vk, OK_Ordinary, SL);
}

Expr *CLExpressions::getTLBGETExpr(ExprKind kind) {
  Expr *kindExpr = getExpr(kind);
  return NewCallExpr(getExpr(TLB_GET), &kindExpr, 1, kindExpr->getType());
}

void CLExpressions::DeleteDeclRefExpr(DeclRefExpr *DRE) {
  VarDecl *declVD = dyn_cast<VarDecl>(DRE->getDecl());
  ASTCtx.Deallocate((void *)declVD);
  ASTCtx.Deallocate((void *)DRE);
}


//---------------------------------------------------------------------------
// Create a new AST node
//---------------------------------------------------------------------------
IntegerLiteral *CLExpressions::NewIntegerLiteral(int val) {
  SourceLocation SL;
  QualType type = ASTCtx.IntTy;
  type.removeLocalFastQualifiers();
  llvm::APInt intVal(32, val);  // 4-byte integer
  return new (ASTCtx) IntegerLiteral(ASTCtx, intVal, type, SL);
}

