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

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <cstdlib>
using namespace std;

static int num_inputs = 0;
static int total_kernel_num = 0;
static int *num_kernels;
static ifstream **in_file;

const char *code_header[] = {
  "TLB_STR_FUNC_START\n",

  "inline void kernel_launch() {\n"
  "  TLB_GET_KEY;\n"
  "  TLB_GET_KEY_FUNC;\n"
  "  switch (TLB_GET(kernel_idx)) {\n",

  "inline void kernel_set_stack(unsigned int* num, unsigned int* size) {\n"
  "  TLB_GET_KEY;\n"
  "  switch (TLB_GET(kernel_idx)) {\n",

  "inline void kernel_set_arguments(void) {\n"
  "  TLB_GET_KEY;\n"
  "  TLB_GET_KEY_FUNC;\n"
  "  CL_SET_ARG_INIT();\n"
  "  switch (TLB_GET(kernel_idx)) {\n",

  "inline void kernel_fini(void) {\n"
  "  TLB_GET_KEY;\n"
  "  TLB_GET_KEY_FUNC;\n"
  "   switch (TLB_GET(kernel_idx)) {\n"
};

const char *code_footer[] = {
  "TLB_STR_FUNC_END\n\n",

  "  }\n"
  "}\n\n",

  "       *num = __CL_KERNEL_DEFAULT_STACK_NUM;\n"
  "       *size = __CL_KERNEL_DEFAULT_STACK_SIZE;\n"
  "       break;\n"
  "    default: CL_DEV_ERR(\"[S%d] error %d\\n\", TLB_GET(cu_id),__LINE__);\n"
  "  }\n"
  "}\n\n",

  "    default: CL_DEV_ERR(\"[S%d] error %d\\n\", TLB_GET(cu_id),__LINE__);\n"
  "  }\n"
  "}\n\n",

  "    default: CL_DEV_ERR(\"[S%d] error %d\\n\", TLB_GET(cu_id),__LINE__);\n"
  "  }\n"
  "}\n\n"
};

const char *kernel_info[] = {
  "const char* _cl_kernel_names[] = {",
  "const unsigned int _cl_kernel_num_args[] = {",
  "const char* _cl_kernel_attributes[] = {",

  "const unsigned int _cl_kernel_work_group_size_hint[][3] = {",
  "const unsigned int _cl_kernel_reqd_work_group_size[][3] = {",
  "const unsigned long long _cl_kernel_local_mem_size[] = {",
  "const unsigned long long _cl_kernel_private_mem_size[] = {",

  "/* CL_KERNEL_ARG_ADDRESS_GLOBAL  : 3\n"
  "   CL_KERNEL_ARG_ADDRESS_CONSTANT: 2\n"
  "   CL_KERNEL_ARG_ADDRESS_LOCAL   : 1\n"
  "   CL_KERNEL_ARG_ADDRESS_PRIVATE : 0 */\n"
  "const char* _cl_kernel_arg_address_qualifier[] = {",

  "/* CL_KERNEL_ARG_ACCESS_READ_ONLY : 3\n"
  "   CL_KERNEL_ARG_ACCESS_WRITE_ONLY: 2\n"
  "   CL_KERNEL_ARG_ACCESS_READ_WRITE: 1\n"
  "   CL_KERNEL_ARG_ACCESS_NONE      : 0 */\n"
  "const char* _cl_kernel_arg_access_qualifier[] = {",

  "const char* _cl_kernel_arg_type_name[] = {",

  "/* CL_KERNEL_ARG_TYPE_NONE    : 0x0\n"
  "   CL_KERNEL_ARG_TYPE_CONST   : 0x1\n"
  "   CL_KERNEL_ARG_TYPE_RESTRICT: 0x2\n"
  "   CL_KERNEL_ARG_TYPE_VOLATILE: 0x4 */\n"
  "const unsigned int _cl_kernel_arg_type_qualifier[] = {",
  "const char* _cl_kernel_arg_name[] = {",

  "const unsigned int _cl_kernel_dim[] = {"
};


void replace_with_id(string &line, int id);
void print_stmts(ofstream &out_file);
void generate_kernel_info_file(const char *filename, int total_kernel_num);
void print_info_stmts(ofstream &info_file);


int main(int argc, char **argv) {
  const int NUM_OUTPUTS = 2;

  if (argc < (NUM_OUTPUTS + 2)) {
    cerr << "Usage: " << argv[0]
         << " <launch code file> <kernel info file> <input files>" << endl;
    return EXIT_FAILURE;
  }

  num_inputs = argc - NUM_OUTPUTS - 1;
  num_kernels = (int *)malloc(sizeof(int) * num_inputs);

  // Open the kernel launch code file.
  const char *out_filename = argv[1];
  ofstream out_file(out_filename);
  if (!out_file.is_open()) {
    cerr << "ERROR: cannot open '" << out_filename << "'" << endl;
    return EXIT_FAILURE;
  }

  // Open input files.
  in_file = (ifstream **)malloc(sizeof(ifstream *) * num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    const char *in_filename = argv[i + NUM_OUTPUTS + 1];
    in_file[i] = new ifstream(in_filename);
    if (!in_file[i]->is_open()) {
      cerr << "ERROR: cannot open '" << in_filename << "'" << endl;
      return EXIT_FAILURE;
    }
  }

  // Print the header code.
  out_file << "#include <cpu_launch_header.h>\n\n";

  // typedef and kernel declarations
  set<string> GlobalDefs;
  string line;
  for (int i = 0; i < num_inputs; i++) {
    while (in_file[i]->good()) {
      getline(*in_file[i], line);
      if (line.substr(0, 7) == "typedef" ||
          line.substr(0, 6) == "struct" ||
          line.substr(0, 5) == "union" ||
          line.substr(0, 4) == "enum") {
        if (line[line.size() - 1] == ';') {
          if (GlobalDefs.find(line) == GlobalDefs.end()) {
            GlobalDefs.insert(line);
            out_file << line << "\n";
          }
        }
      } else if (line.substr(0, 8) == "__kernel") {
        size_t pos = line.find(" {");
        if (pos != string::npos) {
          out_file << line.substr(0, pos) << ";\n";
        }
      } else if (line == "#if 0") {
        break;
      }
    }
  }
  out_file << "\n";

  // the number of kernels
  for (int i = 0; i < num_inputs; i++) {
    getline(*in_file[i], line);
    num_kernels[i] = atoi(line.c_str());
    total_kernel_num += num_kernels[i];
  }

  if (total_kernel_num > 0) {
    // kernel names
    out_file << "static const char* KernelNames[] = {\n  ";
    for (int i = 0; i < num_inputs; i++) {
      if (num_kernels[i] > 0) {
        getline(*in_file[i], line);
        out_file << line;
        if (i + 1 < num_inputs) out_file << ", ";
      }
    }
    out_file << "\n};\n\n";

    int num_code = (int)(sizeof(code_header) / sizeof(char *));
    for (int i = 0; i < num_code; i++) {
      out_file << code_header[i];
      print_stmts(out_file);
      out_file << code_footer[i];
    }
  }

  // Print the footer code.
  out_file << "#include <cpu_launch_footer.h>\n";

  // Close the output file.
  out_file.close();

  // Generate the kernel info file.
  generate_kernel_info_file(argv[2], total_kernel_num);

  // free and close input files
  free(num_kernels);
  for (int i = 0; i < num_inputs; i++) {
    in_file[i]->close();
    delete in_file[i];
  }
  free(in_file);

  return EXIT_SUCCESS;
}


// Replace # with id.
void replace_with_id(string &line, int id) {
  stringstream ss;
  ss << id;
  string id_str = ss.str();

  size_t pos = 0;
  size_t found;
  while ((found = line.find("#", pos)) != string::npos) {
    line.replace(found, 1, id_str);
    pos = found + 1;
  }
}


void print_stmts(ofstream &out_file) {
  int kernel_id = 0 ;
  string line;

  for (int i = 0; i < num_inputs; i++) {
    for (int k = 0; k < num_kernels[i]; k++) {
      getline(*in_file[i], line);
      replace_with_id(line, kernel_id++);
      out_file << line << "\n";
    }
  }
}


void generate_kernel_info_file(const char *filename, int total_kernel_num) {
  // Open the kernel info file.
  ofstream info_file(filename);
  if (!info_file.is_open()) {
    cerr << "ERROR: cannot open '" << filename << "'" << endl;
    return;
  }

  info_file << "const unsigned int _cl_kernel_num = " 
            << total_kernel_num << ";\n\n";

  int num_kernel_info = (int)(sizeof(kernel_info) / sizeof(char *));
  for (int i = 0; i < num_kernel_info; i++) {
    info_file << kernel_info[i];
    print_info_stmts(info_file);
    info_file << "};\n\n";
  }

  info_file.close();
}


void print_info_stmts(ofstream &info_file) {
  string line;
  int cur_kernel_num = 0;

  for (int i = 0; i < num_inputs; i++) {
    cur_kernel_num += num_kernels[i];

    getline(*in_file[i], line);
    if (num_kernels[i] > 0) {
      info_file << line;
      if (i + 1 < num_inputs && cur_kernel_num < total_kernel_num)
        info_file << ", ";
    }
  }
}
