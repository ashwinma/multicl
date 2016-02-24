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
/*   Department of Computer Science and Engineering                          */
/*   Seoul National University, Seoul 151-744, Korea                         */
/*   http://aces.snu.ac.kr                                                   */
/*                                                                           */
/* Contributors:                                                             */
/*   Jungwon Kim, Sangmin Seo, Gangwon Jo, Jun Lee, Jeongho Nah,             */
/*   Jungho Park, Junghyun Kim, and Jaejin Lee                               */
/*                                                                           */
/*****************************************************************************/

/*****************************************************************************/
/* This file is based on the SNU-SAMSUNG OpenCL Compiler and is distributed  */
/* under GNU General Public License.                                         */
/* See LICENSE.SNU-SAMSUNG_OpenCL_C_Compiler.TXT for details.                */
/*****************************************************************************/

#ifndef __CL_KERNEL_CONSTANTS_H__
#define __CL_KERNEL_CONSTANTS_H__

#define FLT_DIG          6
#define FLT_MANT_DIG     24
#define FLT_MAX_10_EXP   +38
#define FLT_MAX_EXP      +128
#define FLT_MIN_10_EXP   -37
#define FLT_MIN_EXP      -125
#define FLT_RADIX        2
#ifdef TARGET_MACH_C64XP
#define FLT_MAX          340282346638528859811704183484516925440.0f
#define FLT_MIN          0.000000000000000000000000000000000000011754943508222875079687365372222456778186655567720875215087517f
#define FLT_EPSILON      0.000000119209289550781250f
#else
#define FLT_MAX          0x1.fffffep127f
#define FLT_MIN          0x1.0p-126f
#define FLT_EPSILON      0x1.0p-23f
#endif

#define M_E_F            2.71828174591064f
#define M_LOG2E_F        1.44269502162933f
#define M_LOG10E_F       0.43429449200630f
#define M_LN2_F          0.69314718246460f
#define M_LN10_F         2.30258512496948f
#define M_PI_F           3.14159274101257f
#define M_PI_2_F         1.57079637050629f
#define M_PI_4_F         0.78539818525314f
#define M_1_PI_F         0.31830987334251f
#define M_2_PI_F         0.63661974668503f
#define M_2_SQRTPI_F     1.12837922573090f
#define M_SQRT2_F        1.41421353816986f
#define M_SQRT1_2_F      0.70710676908493f

#define DBL_DIG          15
#define DBL_MANT_DIG     53
#define DBL_MAX_10_EXP   +308
#define DBL_MAX_EXP      +1024
#define DBL_MIN_10_EXP   -307
#define DBL_MIN_EXP      -1021
#define DBL_RADIX        2
#define DBL_MAX          0x1.fffffffffffffp1023
#define DBL_MIN          0x1.0p-1022
#define DBL_EPSILON      0x1.0p-52

#define M_E              2.718281828459045090796
#define M_LOG2E          1.442695040888963387005
#define M_LOG10E         0.434294481903251816668
#define M_LN2            0.693147180559945286227
#define M_LN10           2.302585092994045901094
#define M_PI             3.141592653589793115998
#define M_PI_2           1.570796326794896557999
#define M_PI_4           0.785398163397448278999
#define M_1_PI           0.318309886183790691216
#define M_2_PI           0.636619772367581382433
#define M_2_SQRTPI       1.128379167095512558561
#define M_SQRT2          1.414213562373095145475
#define M_SQRT1_2        0.707106781186547572737

#define MAXFLOAT         FLT_MAX
#define HUGE_VALF        ((float) 1e50)
#define INFINITY         HUGE_VALF
#define NAN              (INFINITY - INFINITY)
#define HUGE_VAL         ((double) 1e500)

#define CHAR_BIT         8
#define SCHAR_MAX        127
#define SCHAR_MIN        (-127-1)
#define CHAR_MAX         SCHAR_MAX
#define CHAR_MIN         SCHAR_MIN
#define UCHAR_MAX        255
#define SHRT_MAX         32767
#define SHRT_MIN         (-32767-1)
#define USHRT_MAX        65535
#define INT_MAX          2147483647
#define INT_MIN          (-2147483647-1)
#define UINT_MAX         0xffffffffU
#define LONG_MAX         0x7FFFFFFFFFFFFFFFLL
#define LONG_MIN         (-0x7FFFFFFFFFFFFFFFLL - 1LL)
#define ULONG_MAX        0xFFFFFFFFFFFFFFFFULL

#undef FP_ILOGB0
#define FP_ILOGB0        INT_MIN
#undef FP_ILOGBNAN
#define FP_ILOGBNAN      INT_MAX

#endif //__CL_KERNEL_CONSTANTS_H__
