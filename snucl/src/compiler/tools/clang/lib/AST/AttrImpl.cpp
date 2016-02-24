//===--- AttrImpl.cpp - Classes for representing attributes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains out-of-line virtual methods for Attr classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/AST/Expr.h"
using namespace clang;

Attr::~Attr() { }

#include "clang/AST/AttrImpl.inc"

#ifdef __SNUCL_COMPILER__
#include <sstream>
#include <string>
using std::stringstream;
using std::string;

string AlignedAttr::getAsStringWithContext(ASTContext &C) {
  stringstream AttrStr;
  AttrStr << "__attribute__((aligned";

  if (isAlignmentExpr() && hasAlignmentExpr()) {
    unsigned Alignment = getAlignment(C);
    AttrStr << "(" << (unsigned)(Alignment / C.getCharWidth()) << ")";
  }
  AttrStr << "))";

  return AttrStr.str();
}

string PackedAttr::getAsString() {
  return string("__attribute__((packed))");
}

string ReqdWorkGroupSizeAttr::getAsString() {
  stringstream AttrStr;
  AttrStr << "__attribute__((reqd_work_group_size("
          << getXDim() << "," << getYDim() << "," << getZDim() << ")))";
  return AttrStr.str();
}

string WorkGroupSizeHintAttr::getAsString() {
  stringstream AttrStr;
  AttrStr << "__attribute__((work_group_size_hint("
          << getXDim() << "," << getYDim() << "," << getZDim() << ")))";
  return AttrStr.str();
}

string VecTypeHintAttr::getAsString() {
  stringstream AttrStr;
  AttrStr << "__attribute__((vec_type_hint(" 
          << getHintType().getAsString() << ")))";
  return AttrStr.str();
}

string EndianAttr::getAsString() {
  string EndianType(getEndianType().data());
  return "__attribute__((endian(" + EndianType + ")))";
}
#endif
