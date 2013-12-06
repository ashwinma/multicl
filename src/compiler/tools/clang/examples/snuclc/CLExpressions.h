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

#ifndef SNUCLC_CLEXPRESSIONS_H
#define SNUCLC_CLEXPRESSIONS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "llvm/ADT/StringRef.h"
#include "TypeDefs.h"

namespace clang {
namespace snuclc {

class CLExpressions {
public:
  enum ExprKind {
    DEV_CACHE_RD = 0,
    DEV_CACHE_WR,
    DEV_CACHE_WR_OP,
    DEV_CACHE_POSTINC,
    DEV_CACHE_POSTDEC,
    DEV_CACHE_PREINC,
    DEV_CACHE_PREDEC,

    TLB_GET_KEY,
    TLB_GET,
    TLB_GET_KEY_LOCAL,
    TLB_GET_LOCAL,

    // operator
    CL_SAFE_INT_DIV_ZERO,
    CL_SAFE_UINT_DIV_ZERO,
    DIV,
    REM,
    DIV_ASSIGN,
    REM_ASSIGN,

    // operation
    OP_MUL,
    OP_DIV,
    OP_REM,
    OP_ADD,
    OP_SUB,
    OP_SHL,
    OP_SHR,
    OP_AND,
    OP_XOR,
    OP_OR, 

    // types
    CHAR,
    UCHAR,
    SHORT,
    USHORT,
    BOOL,
    INT,
    UINT,
    LONG,
    ULONG,
    LLONG,
    ULLONG,
    HALF,
    FLOAT,
    DOUBLE,
    CHAR2,
    CHAR3,
    CHAR4,
    CHAR8,
    CHAR16,
    UCHAR2,
    UCHAR3,
    UCHAR4,
    UCHAR8,
    UCHAR16,
    SHORT2,
    SHORT3,
    SHORT4,
    SHORT8,
    SHORT16,
    USHORT2,
    USHORT3,
    USHORT4,
    USHORT8,
    USHORT16,
    INT2,
    INT3,
    INT4,
    INT8,
    INT16,
    UINT2,
    UINT3,
    UINT4,
    UINT8,
    UINT16,
    LONG2,
    LONG3,
    LONG4,
    LONG8,
    LONG16,
    ULONG2,
    ULONG3,
    ULONG4,
    ULONG8,
    ULONG16,
    FLOAT2,
    FLOAT3,
    FLOAT4,
    FLOAT8,
    FLOAT16,
    DOUBLE2,
    DOUBLE3,
    DOUBLE4,
    DOUBLE8,
    DOUBLE16,

    // conversion
    CONVERT_CHAR2,
    CONVERT_CHAR3,
    CONVERT_CHAR4,
    CONVERT_CHAR8,
    CONVERT_CHAR16,
    CONVERT_UCHAR2,
    CONVERT_UCHAR3,
    CONVERT_UCHAR4,
    CONVERT_UCHAR8,
    CONVERT_UCHAR16,
    CONVERT_SHORT2,
    CONVERT_SHORT3,
    CONVERT_SHORT4,
    CONVERT_SHORT8,
    CONVERT_SHORT16,
    CONVERT_USHORT2,
    CONVERT_USHORT3,
    CONVERT_USHORT4,
    CONVERT_USHORT8,
    CONVERT_USHORT16,
    CONVERT_INT2,
    CONVERT_INT3,
    CONVERT_INT4,
    CONVERT_INT8,
    CONVERT_INT16,
    CONVERT_UINT2,
    CONVERT_UINT3,
    CONVERT_UINT4,
    CONVERT_UINT8,
    CONVERT_UINT16,
    CONVERT_LONG2,
    CONVERT_LONG3,
    CONVERT_LONG4,
    CONVERT_LONG8,
    CONVERT_LONG16,
    CONVERT_ULONG2,
    CONVERT_ULONG3,
    CONVERT_ULONG4,
    CONVERT_ULONG8,
    CONVERT_ULONG16,
    CONVERT_FLOAT2,
    CONVERT_FLOAT3,
    CONVERT_FLOAT4,
    CONVERT_FLOAT8,
    CONVERT_FLOAT16,
    CONVERT_DOUBLE2,
    CONVERT_DOUBLE3,
    CONVERT_DOUBLE4,
    CONVERT_DOUBLE8,
    CONVERT_DOUBLE16,

    // vector literal
    VEC_LITERAL_CHAR2,
    VEC_LITERAL_CHAR3,
    VEC_LITERAL_CHAR4,
    VEC_LITERAL_CHAR8,
    VEC_LITERAL_CHAR16,
    VEC_LITERAL_UCHAR2,
    VEC_LITERAL_UCHAR3,
    VEC_LITERAL_UCHAR4,
    VEC_LITERAL_UCHAR8,
    VEC_LITERAL_UCHAR16,
    VEC_LITERAL_SHORT2,
    VEC_LITERAL_SHORT3,
    VEC_LITERAL_SHORT4,
    VEC_LITERAL_SHORT8,
    VEC_LITERAL_SHORT16,
    VEC_LITERAL_USHORT2,
    VEC_LITERAL_USHORT3,
    VEC_LITERAL_USHORT4,
    VEC_LITERAL_USHORT8,
    VEC_LITERAL_USHORT16,
    VEC_LITERAL_INT2,
    VEC_LITERAL_INT3,
    VEC_LITERAL_INT4,
    VEC_LITERAL_INT8,
    VEC_LITERAL_INT16,
    VEC_LITERAL_UINT2,
    VEC_LITERAL_UINT3,
    VEC_LITERAL_UINT4,
    VEC_LITERAL_UINT8,
    VEC_LITERAL_UINT16,
    VEC_LITERAL_LONG2,
    VEC_LITERAL_LONG3,
    VEC_LITERAL_LONG4,
    VEC_LITERAL_LONG8,
    VEC_LITERAL_LONG16,
    VEC_LITERAL_ULONG2,
    VEC_LITERAL_ULONG3,
    VEC_LITERAL_ULONG4,
    VEC_LITERAL_ULONG8,
    VEC_LITERAL_ULONG16,
    VEC_LITERAL_FLOAT2,
    VEC_LITERAL_FLOAT3,
    VEC_LITERAL_FLOAT4,
    VEC_LITERAL_FLOAT8,
    VEC_LITERAL_FLOAT16,
    VEC_LITERAL_DOUBLE2,
    VEC_LITERAL_DOUBLE3,
    VEC_LITERAL_DOUBLE4,
    VEC_LITERAL_DOUBLE8,
    VEC_LITERAL_DOUBLE16,

    // integer literal
    ZERO,
    ONE,
    TWO,
    THREE,
    FOUR,
    FIVE,
    SIX,
    SEVEN,
    EIGHT,
    NINE,
    TEN,
    ELEVEN,
    TWELVE,
    THIRTEEN,
    FOURTEEN,
    FIFTEEN,

    // select
    SELECT,

    __LOCAL_SIZE,
    __LOCAL_SIZE_0,
    __LOCAL_SIZE_1,
    __LOCAL_SIZE_2,
    LOCAL_SIZE_0,
    LOCAL_SIZE_1,
    LOCAL_SIZE_2,

    __GLOBAL_ID,
    __GLOBAL_ID_0,
    __GLOBAL_ID_1,
    __GLOBAL_ID_2,
    GLOBAL_ID_0,
    GLOBAL_ID_1,
    GLOBAL_ID_2,

    // For work-item coalescing loop
    __I,
    __J,
    __K,
    WCL_IDX_I,
    WCL_IDX_J,
    WCL_IDX_K,
    WCL_IDX_I_INIT,
    WCL_IDX_J_INIT,
    WCL_IDX_K_INIT,
    WCL_IDX_I_COND,
    WCL_IDX_J_COND,
    WCL_IDX_K_COND,
    WCL_IDX_I_INC,
    WCL_IDX_J_INC,
    WCL_IDX_K_INC,
    MALLOC,
    FREE,

    END_EXPR
  };

private:
  ASTContext &ASTCtx;
  DeclContext *DeclCtx;
  IdentifierTable &Idents;

  bool NeedTLBUse;            // whether if we need TLB_GET()
  unsigned TmpVarNum;         // counter for temporary variables
  FunctionDecl *DummyFD;      // dummy FunctionDecl for inlined functions

public:
  CLExpressions(ASTContext &C, DeclContext *D);

  void setTLBUse(bool use) { NeedTLBUse = use; }

  Expr *getExpr(ExprKind kind);
  Expr *getExpr(QualType type);
  Expr *getConvertExpr(QualType type);
  Expr *getVectorLiteralExpr(QualType type);

  Stmt *getWCL(CompoundStmt *body, unsigned MaxDim=0);
  Stmt *getInnerWCL(CompoundStmt *body);
  Stmt *getLoopK(CompoundStmt *body);
  Stmt *getLoopJ(CompoundStmt *body);
  Stmt *getLoopKJ(CompoundStmt *body);

  unsigned getTmpVarNum() { return TmpVarNum++; }
  Expr *getDummyCallExpr();

private:
  unsigned getExprTypeKind(QualType type);
  std::string getExprKindName(ExprKind kind);
 
  Expr *NewDeclRefExpr(llvm::StringRef Name);
  Expr *NewUnaryOperator(Expr *input, UnaryOperator::Opcode opc, QualType t);
  Expr *getTLBGETExpr(ExprKind kind);

  void DeleteDeclRefExpr(DeclRefExpr *DRE);

public:
  // AST node functions
  CompoundStmt *NewCompoundStmt(StmtVector &Stmts) {
    return new (ASTCtx) CompoundStmt(ASTCtx, Stmts.data(), Stmts.size(),
                                     SourceLocation(), SourceLocation());
  }

  CompoundStmt *NewStmtList(StmtVector &Stmts) {
    CompoundStmt *CS = NewCompoundStmt(Stmts);
    CS->setIsStmtList(true);
    return CS;
  }

  DeclStmt *NewSingleDeclStmt(Decl *D) {
    DeclGroupRef DG = DeclGroupRef::Create(ASTCtx, &D, 1);
    return new (ASTCtx) DeclStmt(DG, SourceLocation(), SourceLocation());
  }

  DeclRefExpr *NewDeclRefExpr(VarDecl *VD, QualType T) {
    return new (ASTCtx) DeclRefExpr(VD, T, VK_RValue, SourceLocation());
  }

  BinaryOperator *NewBinaryOperator(Expr *LHS, Expr *RHS,
                                    BinaryOperator::Opcode Opc, QualType T) {
    return new (ASTCtx) BinaryOperator(
        LHS, RHS, Opc, T, VK_RValue, OK_Ordinary, SourceLocation());
  }

  CallExpr *NewCallExpr(Expr *fn, Expr **args, unsigned numargs, QualType T) {
    return new (ASTCtx) CallExpr(ASTCtx, fn, args, numargs, T, VK_RValue,
                                 SourceLocation());
  }

  ArraySubscriptExpr *NewArraySubscriptExpr(Expr *LHS, Expr *RHS,
                                            QualType T) {
    return new (ASTCtx) ArraySubscriptExpr(
        LHS, RHS, T, VK_RValue, OK_Ordinary, SourceLocation());
  }

  SizeOfAlignOfExpr *NewSizeOfExpr(QualType T) {
    return new (ASTCtx) SizeOfAlignOfExpr(
        true, ASTCtx.getTrivialTypeSourceInfo(T), ASTCtx.UnsignedIntTy,
        SourceLocation(), SourceLocation());
  }

  CStyleCastExpr *NewCStyleCastExpr(QualType T, Expr *Op, QualType WT) {
    return CStyleCastExpr::Create(
        ASTCtx, T, VK_RValue, CK_BitCast, Op, NULL,
        ASTCtx.getTrivialTypeSourceInfo(WT),
        SourceLocation(), SourceLocation());
  }

  IntegerLiteral *NewIntegerLiteral(int val);

};

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_OPENCLEXPRESSIONS_H
