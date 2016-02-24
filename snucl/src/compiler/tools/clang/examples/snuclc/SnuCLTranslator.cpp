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

#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/AST.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/raw_ostream.h"
#include "CLOptions.h"
#include "CLTranslator.h"
using namespace clang;
using namespace clang::snuclc;

namespace {

class SnuCLTranslator : public PluginASTAction {
  CLOptions CLOpts;

protected:
  ASTConsumer *CreateASTConsumer(CompilerInstance &CI,
                                 llvm::StringRef InFile) {
    if (llvm::raw_ostream *OS = CI.createDefaultOutputFile(false, InFile))
      return new CLTranslator(CLOpts, OS);
    else
      return new CLTranslator(CLOpts);
  }

  bool ParseArgs(const CompilerInstance &CI,
                 const std::vector<std::string>& args) {
    if (args.size() == 0) return true;

    for (unsigned i = 0, e = args.size(); i != e; ++i) {
      if (args[i] == "-m64") {
        CLOpts.M64 = 1;
      } else if (args[i] == "-m32") {
        CLOpts.M64 = 0;
      } else if (args[i] == "-debug") {
        CLOpts.Debug = 1;
      } else if (args[i] == "-noopt") {
        CLOpts.NoOpt = 1;
      } else if (args[i] == "-help" || args[i] == "--help") {
        PrintHelp(llvm::errs());
        return false;
      }

      // 5.6.4.2 Math Intrinsics Options
      else if (args[i] == "-cl-single-precision-constant") {
        CLOpts.SinglePrecisionConstant = 1;
      } else if (args[i] == "-cl-denorms-are-zero") {
        CLOpts.DenormsAreZero = 1;
      } else if (args[i] == "-cl-fp32-correctly-rounded-divide-sqrt") {
        CLOpts.FP32CorrectlyRoundedDivideSqrt = 1;
      }

      // 5.6.4.3 Optimization Options
      else if (args[i] == "-cl-opt-disable") {
        CLOpts.OptDisable = 1;
      } else if (args[i] == "-cl-mad-enable") {
        CLOpts.MadEnable = 1;
      } else if (args[i] == "-cl-no-signed-zeros") {
        CLOpts.NoSignedZeros = 1;
      } else if (args[i] == "-cl-unsafe-math-optimizations") {
        CLOpts.UnsafeMathOptimizations = 1;
      } else if (args[i] == "-cl-finite-math-only") {
        CLOpts.FiniteMathOnly = 1;
      } else if (args[i] == "-cl-fast-relaxed-math") {
        CLOpts.FastRelaxedMath = 1;
      }

      // 5.6.4.5 Options Controlling the OpenCL C version
      else if (args[i].substr(0, 8) == "-cl-std=") {
        CLOpts.Std = args[i].substr(8);
      }

      // 5.6.4.6 Options for Querying Kernel Argument Information
      else if (args[i] == "-cl-kernel-arg-info") {
        CLOpts.KernelArgInfo = 1;
      }

      // Deprecated options
      else if (args[i] == "-cl-strict-aliasing") {
        CLOpts.StrictAliasing = 1;
      }
     
      else {
        Diagnostic &D = CI.getDiagnostics();
        unsigned DiagID = D.getCustomDiagID(
            Diagnostic::Error, "invalid argument '" + args[i] + "'");
        D.Report(DiagID);
        return false;
      }
    }

    return true;
  }

  void PrintHelp(llvm::raw_ostream& ros) {
    ros << "Help for snuclc\n";
  }
};

}

static FrontendPluginRegistry::Add<SnuCLTranslator>
X("snuclc", "SnuCL Translator");
