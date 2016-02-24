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

#include "llvm/ADT/StringRef.h"
#include "TransformFlowKeyword.h"
#include "Defines.h"
using namespace llvm;
using namespace clang;
using namespace clang::snuclc;

#include <string>
#include <sstream>
using std::string;
using std::stringstream;


void TransformFlowKeyword::Transform() {
  for (TransformWCR::iterator I = TWCR.begin(), E = TWCR.end(); I != E; ++I) {
    CompoundStmt *wcr = *I;
    if (wcr->size() == 0) continue;
    
    ReplaceFlowKeyword(wcr);

    if (NextLabel != NULL) {
      InsertNextLabel(wcr);
    }
  }
}


Stmt *TransformFlowKeyword::ReplaceFlowKeyword(Stmt *S) {
  switch (S->getStmtClass()) {
  default: //assert(0 && "Unknown or unsupported stmt kind!");
           return S;

  // statements
  case Stmt::GotoStmtClass:     return S;
  case Stmt::ContinueStmtClass: return S;
  case Stmt::BreakStmtClass: {
    ASTCtx.Deallocate(S);
    return new (ASTCtx) ContinueStmt(SourceLocation());
  }
  case Stmt::ReturnStmtClass: {
    ReturnStmt *Node = static_cast<ReturnStmt*>(S);
    if (!Node->getRetValue()) {
      ASTCtx.Deallocate(S);
      return new (ASTCtx) ContinueStmt(SourceLocation());
    }
    return Node;
  }

  case Stmt::IndirectGotoStmtClass: return S;

  case Stmt::CompoundStmtClass: {
    CompoundStmt *Node = static_cast<CompoundStmt*>(S);
    CompoundStmt::body_iterator I, E;
    for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
      *I = ReplaceFlowKeyword(*I);
    }
    return Node;
  }
  case Stmt::IfStmtClass: {
    IfStmt *Node = static_cast<IfStmt*>(S);
    Node->setThen(ReplaceFlowKeyword(Node->getThen()));
    if (Stmt *Else = Node->getElse()) {
      Node->setElse(ReplaceFlowKeyword(Else));
    }
    return Node;
  }

  case Stmt::SwitchStmtClass:  //FIXME:
  case Stmt::CaseStmtClass: 
  case Stmt::DefaultStmtClass: 
  case Stmt::LabelStmtClass:   return S;

  case Stmt::WhileStmtClass: {
    WhileStmt *Node = static_cast<WhileStmt*>(S);
    Node->setBody(SubReplace(Node->getBody()));
    return Node;
  }
  case Stmt::DoStmtClass: {
    DoStmt *Node = static_cast<DoStmt*>(S);
    Node->setBody(SubReplace(Node->getBody()));
    return Node;
  }
  case Stmt::ForStmtClass: {
    ForStmt *Node = static_cast<ForStmt*>(S);
    Node->setBody(SubReplace(Node->getBody()));
    return Node;
  }

  case Stmt::DeclStmtClass: return S;
  } //end switch

  return S;
}


Stmt *TransformFlowKeyword::SubReplace(Stmt *S) {
  switch (S->getStmtClass()) {
  default: return S;

  case Stmt::GotoStmtClass: 
  case Stmt::ContinueStmtClass:
  case Stmt::BreakStmtClass:    return S;

  case Stmt::ReturnStmtClass: {
    ReturnStmt *Node = static_cast<ReturnStmt*>(S);
    if (!Node->getRetValue()) {
      ASTCtx.Deallocate(S);
      if (NextLabel == NULL) {
        NextLabel = NewNextLabel();
      }
      return new (ASTCtx) GotoStmt(NextLabel->getDecl(), 
                                   SourceLocation(), SourceLocation());
    }
    return Node;
  }

  case Stmt::IndirectGotoStmtClass: return S;

  case Stmt::CompoundStmtClass: {
    CompoundStmt *Node = static_cast<CompoundStmt*>(S);
    CompoundStmt::body_iterator I, E;
    for (I = Node->body_begin(), E = Node->body_end(); I != E; ++I) {
      *I = SubReplace(*I);
    }
    return Node;
  }

  case Stmt::IfStmtClass: {
    IfStmt *Node = static_cast<IfStmt*>(S);
    Node->setThen(SubReplace(Node->getThen()));
    if (Stmt *Else = Node->getElse()) {
      Node->setElse(SubReplace(Else));
    }
    return Node;
  }

  case Stmt::SwitchStmtClass:  //FIXME:
  case Stmt::CaseStmtClass: 
  case Stmt::DefaultStmtClass: 
  case Stmt::LabelStmtClass:   return S;

  case Stmt::WhileStmtClass: {
    WhileStmt *Node = static_cast<WhileStmt*>(S);
    Node->setBody(SubReplace(Node->getBody()));
    return Node;
  }
  case Stmt::DoStmtClass: {
    DoStmt *Node = static_cast<DoStmt*>(S);
    Node->setBody(SubReplace(Node->getBody()));
    return Node;
  }
  case Stmt::ForStmtClass: {
    ForStmt *Node = static_cast<ForStmt*>(S);
    Node->setBody(SubReplace(Node->getBody()));
    return Node;
  }

  case Stmt::DeclStmtClass: return S;
  } //end switch

  return S;
}


void TransformFlowKeyword::InsertNextLabel(CompoundStmt *CS) {
  StmtVector Stmts;

  CompoundStmt::body_iterator I, E;
  for (I = CS->body_begin(), E = CS->body_end(); I != E; ++I) {
    Stmts.push_back(*I);
  }
  Stmts.push_back(NextLabel);

  NextLabel = NULL;

  CS->setStmts(ASTCtx, Stmts.data(), Stmts.size());
}


LabelStmt *TransformFlowKeyword::NewNextLabel() {
  stringstream Name;
  Name << OPENCL_NEXT_LABEL_PREFIX << LabelCount++;

  string NameStr = Name.str();
  StringRef DeclName(NameStr);
  IdentifierInfo &DeclID = ASTCtx.Idents.get(DeclName);

  DeclContext *DeclCtx = ASTCtx.getTranslationUnitDecl();
  LabelDecl *LD = LabelDecl::Create(ASTCtx, DeclCtx, SourceLocation(), 
                                    &DeclID);

  return new (ASTCtx) LabelStmt(SourceLocation(), LD, NULL);
}

