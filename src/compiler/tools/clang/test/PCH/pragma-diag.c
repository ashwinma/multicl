// Test this without pch.
// RUN: %clang_cc1 %s -include %s -verify -fsyntax-only

// Test with pch.
// RUN: %clang_cc1 %s -emit-pch -o %t
// RUN: %clang_cc1 %s -include-pch %t -verify -fsyntax-only

#ifndef HEADER
#define HEADER

#pragma clang diagnostic ignored "-Wtautological-compare"

#else

void f() {
  int b = b==b;
}

#endif
