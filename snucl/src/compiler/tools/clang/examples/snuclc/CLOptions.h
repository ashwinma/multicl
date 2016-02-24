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

#ifndef SNUCLC_CLOPTIONS_H
#define SNUCLC_CLOPTIONS_H

namespace clang {
namespace snuclc {

/// CLOptions - Parameter information of __kernel functions
class CLOptions {
public:
  unsigned M64          : 1;    // address space of target machine
  unsigned Debug        : 1;    // 1: debug mode, 0: not
  unsigned NoOpt        : 1;    // 1: without any transformation
  unsigned UseTLBLocal  : 1;    // 1: local needs to accessed by TLB_GET_LOCAL

  unsigned SinglePrecisionConstant        : 1;
  unsigned DenormsAreZero                 : 1;
  unsigned FP32CorrectlyRoundedDivideSqrt : 1;
  unsigned OptDisable                     : 1;
  unsigned MadEnable                      : 1;
  unsigned NoSignedZeros                  : 1;
  unsigned UnsafeMathOptimizations        : 1;
  unsigned FiniteMathOnly                 : 1;
  unsigned FastRelaxedMath                : 1;
  unsigned KernelArgInfo                  : 1;
  unsigned StrictAliasing                 : 1;

  std::string Std;

public:
  CLOptions() {
    M64 = 1;    // default: 64-bit machine
    Debug = 0;  // default: non-debug mode
    NoOpt = 0;  // default: with transformations
    UseTLBLocal = 0;  // default: not use

    SinglePrecisionConstant        = 0;
    DenormsAreZero                 = 0;
    FP32CorrectlyRoundedDivideSqrt = 0;
    OptDisable                     = 0;
    MadEnable                      = 0;
    NoSignedZeros                  = 0;
    UnsafeMathOptimizations        = 0;
    FiniteMathOnly                 = 0;
    FastRelaxedMath                = 0;
    KernelArgInfo                  = 0;
    StrictAliasing                 = 0;
  }
}; //end class CLOptions

} //end namespace snuclc
} //end namespace clang

#endif //SNUCLC_CLOPTIONS_H

