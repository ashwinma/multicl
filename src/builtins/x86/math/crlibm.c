/*****************************************************************************/
/* Copyright (C) 2010, 2011 Seoul National University                        */
/* and Samsung Electronics Co., Ltd.                                         */
/*                                                                           */
/* Contributed by Sangmin Seo <sangmin@aces.snu.ac.kr>, Jungwon Kim          */
/* <jungwon@aces.snu.ac.kr>, Jaejin Lee <jlee@cse.snu.ac.kr>, Seungkyun Kim  */
/* <seungkyun@aces.snu.ac.kr>, Jungho Park <jungho@aces.snu.ac.kr>,          */
/* Honggyu Kim <honggyu@aces.snu.ac.kr>, Jeongho Nah                         */
/* <jeongho@aces.snu.ac.kr>, Sung Jong Seo <sj1557.seo@samsung.com>,         */
/* Seung Hak Lee <s.hak.lee@samsung.com>, Seung Mo Cho                       */
/* <seungm.cho@samsung.com>, Hyo Jung Song <hjsong@samsung.com>,             */
/* Sang-Bum Suh <sbuk.suh@samsung.com>, and Jong-Deok Choi                   */
/* <jd11.choi@samsung.com>                                                   */
/*                                                                           */
/* All rights reserved.                                                      */
/*                                                                           */
/* This file is part of the SNU-SAMSUNG OpenCL runtime.                      */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is free software: you can redistribute it  */
/* and/or modify it under the terms of the GNU Lesser General Public License */
/* as published by the Free Software Foundation, either version 3 of the     */
/* License, or (at your option) any later version.                           */
/*                                                                           */
/* The SNU-SAMSUNG OpenCL runtime is distributed in the hope that it will be */
/* useful, but WITHOUT ANY WARRANTY; without even the implied warranty of    */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  */
/* Public License for more details.                                          */
/*                                                                           */
/* You should have received a copy of the GNU Lesser General Public License  */
/* along with the SNU-SAMSUNG OpenCL runtime. If not, see                    */
/* <http://www.gnu.org/licenses/>.                                           */
/*****************************************************************************/

/*===---             Correctly Rounded mathematical library             ---===*/
/*                                                                            */
/** version: crlibm-0.10beta                                                  */
/** website: http://lipforge.ens-lyon.fr/www/crlibm                           */
/*                                                                            */
/* Copyright (C) 2002  David Defour and Florent de Dinechin

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   This library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this library; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA  */

#include "math_util.h"

int snu_payne_hanek(scs_ptr result, const scs_ptr x){
  uint64_t r[SCS_NB_WORDS+3], tmp;
  unsigned int N;
  /* result r[0],...,r[10] could store till 300 bits of precision */
  /* that is really enough for computing the reduced argument */
  int sign, i, j, ind;
  int *two_over_pi_pt;

  if ((X_EXP != 1)||(X_IND < -1)){
    scs_set(result, x);
    return 0;
  }

  /* Compute the product |x| * 2/Pi */
  if ((X_IND == -1)){
    /* In this case we consider number between ]-1,+1[    */
    /* we may use simpler algorithm such as Cody And Waite */
    r[0] =  0;
    r[1] =  0;
    r[2] =  (uint64_t)(two_over_pi[0]) * X_HW[0];
    r[3] = ((uint64_t)(two_over_pi[0]) * X_HW[1]
        +(uint64_t)(two_over_pi[1]) * X_HW[0]);
    if(X_HW[2] == 0){
      for(i=4; i<(SCS_NB_WORDS+3); i++){
        r[i] = ((uint64_t)(two_over_pi[i-3]) * X_HW[1]
            +(uint64_t)(two_over_pi[i-2]) * X_HW[0]);
      }}else {
        for(i=4; i<(SCS_NB_WORDS+3); i++){
          r[i] = ((uint64_t)(two_over_pi[i-4]) * X_HW[2]
              +(uint64_t)(two_over_pi[i-3]) * X_HW[1]
              +(uint64_t)(two_over_pi[i-2]) * X_HW[0]);
        }
      }
  }else {
    if (X_IND == 0){
      r[0] =  0;
      r[1] =  (uint64_t)(two_over_pi[0]) * X_HW[0];
      r[2] = ((uint64_t)(two_over_pi[0]) * X_HW[1]
          +(uint64_t)(two_over_pi[1]) * X_HW[0]);
      if(X_HW[2] == 0){
        for(i=3; i<(SCS_NB_WORDS+3); i++){
          r[i] = ((uint64_t)(two_over_pi[i-2]) * X_HW[1]
              +(uint64_t)(two_over_pi[i-1]) * X_HW[0]);
        }}else {
          for(i=3; i<(SCS_NB_WORDS+3); i++){
            r[i] = ((uint64_t)(two_over_pi[i-3]) * X_HW[2]
                +(uint64_t)(two_over_pi[i-2]) * X_HW[1]
                +(uint64_t)(two_over_pi[i-1]) * X_HW[0]);
          }}
    }else {
      if (X_IND == 1){
        r[0] =  (uint64_t)(two_over_pi[0]) * X_HW[0];
        r[1] = ((uint64_t)(two_over_pi[0]) * X_HW[1]
            +(uint64_t)(two_over_pi[1]) * X_HW[0]);
        if(X_HW[2] == 0){
          for(i=2; i<(SCS_NB_WORDS+3); i++){
            r[i] = ((uint64_t)(two_over_pi[i-1]) * X_HW[1]
                +(uint64_t)(two_over_pi[ i ]) * X_HW[0]);
          }}else {
            for(i=2; i<(SCS_NB_WORDS+3); i++){
              r[i] = ((uint64_t)(two_over_pi[i-2]) * X_HW[2]
                  +(uint64_t)(two_over_pi[i-1]) * X_HW[1]
                  +(uint64_t)(two_over_pi[ i ]) * X_HW[0]);
            }}
      }else {
        if (X_IND == 2){
          r[0] = ((uint64_t)(two_over_pi[0]) * X_HW[1]
              +(uint64_t)(two_over_pi[1]) * X_HW[0]);
          if(X_HW[2] == 0){
            for(i=1; i<(SCS_NB_WORDS+3); i++){
              r[i] = ((uint64_t)(two_over_pi[ i ]) * X_HW[1]
                  +(uint64_t)(two_over_pi[i+1]) * X_HW[0]);
            }}else {
              for(i=1; i<(SCS_NB_WORDS+3); i++){
                r[i] = ((uint64_t)(two_over_pi[i-1]) * X_HW[2]
                    +(uint64_t)(two_over_pi[ i ]) * X_HW[1]
                    +(uint64_t)(two_over_pi[i+1]) * X_HW[0]);
              }}
        }else {
          ind = (X_IND - 3);
          two_over_pi_pt = (int*)&(two_over_pi[ind]);
          if(X_HW[2] == 0){
            for(i=0; i<(SCS_NB_WORDS+3); i++){
              r[i] = ((uint64_t)(two_over_pi_pt[i+1]) * X_HW[1]
                  +(uint64_t)(two_over_pi_pt[i+2]) * X_HW[0]);
            }}else {
              for(i=0; i<(SCS_NB_WORDS+3); i++){
                r[i] = ((uint64_t)(two_over_pi_pt[ i ]) * X_HW[2]
                    +(uint64_t)(two_over_pi_pt[i+1]) * X_HW[1]
                    +(uint64_t)(two_over_pi_pt[i+2]) * X_HW[0]);
              }
            }
        }
      }
    }
  }

  /* Carry propagate */
  r[SCS_NB_WORDS+1] += r[SCS_NB_WORDS+2]>>30;
  for(i=(SCS_NB_WORDS+1); i>0; i--) {tmp=r[i]>>30;   r[i-1] += tmp;  r[i] -= (tmp<<30);}

  /* The integer part is in r[0] */
  N = (unsigned int)(r[0]);
#if 0
  printf("r[0] = %d\n", N);
#endif

  /* test if the reduced part is bigger than Pi/4 */
  if (r[1] > (uint64_t)(SCS_RADIX)/2){
    N += 1;
    sign = -1;
    for(i=1; i<(SCS_NB_WORDS+3); i++) { r[i]=((~(unsigned int)(r[i])) & 0x3fffffff);}
  }
  else
    sign = 1;

  /* Now we get the reduce argument and check for possible
   * cancellation By Kahan algorithm we will have at most 2 digits
   * of cancellations r[1] and r[2] in the worst case.
   */
  if (r[1] == 0)
    if (r[2] == 0) i = 3;
    else           i = 2;
  else             i = 1;

  for(j=0; j<SCS_NB_WORDS; j++) { R_HW[j] = (unsigned int)(r[i+j]);}

  R_EXP   = 1;
  R_IND   = -i;
  R_SGN   = sign*X_SGN;

  /* Last step :
   *   Multiplication by pi/2
   */
  scs_mul(result, Pio2_ptr, result);
  return X_SGN*N;
}

/**
  This function copies a result into another. There is an unrolled
  version for the case SCS_NB_WORDS==8.
 */
void scs_set(scs_ptr result, scs_ptr x){
  /* unsigned int i;*/

#if (SCS_NB_WORDS==8)
  R_HW[0] = X_HW[0]; R_HW[1] = X_HW[1];
  R_HW[2] = X_HW[2]; R_HW[3] = X_HW[3];
  R_HW[4] = X_HW[4]; R_HW[5] = X_HW[5];
  R_HW[6] = X_HW[6]; R_HW[7] = X_HW[7];
#else
  for(i=0; i<SCS_NB_WORDS; i++)
    R_HW[i] = X_HW[i];
#endif
  R_EXP = X_EXP;
  R_IND = X_IND;
  R_SGN = X_SGN;
}

void scs_mul(scs_ptr result, scs_ptr x, scs_ptr y){
  uint64_t     val, tmp;
  uint64_t     r0,r1,r2,r3,r4,r5,r6,r7,r8;
  uint64_t     x0,x1,x2,x3,x4,x5,x6,x7;
  int                    y0,y1,y2,y3,y4,y5,y6,y7;

  R_EXP = X_EXP * Y_EXP;
  R_SGN = X_SGN * Y_SGN;
  R_IND = X_IND + Y_IND;

  /* Partial products computation */
  x7=X_HW[7];  y7=Y_HW[7];  x6=X_HW[6];  y6=Y_HW[6];
  x5=X_HW[5];  y5=Y_HW[5];  x4=X_HW[4];  y4=Y_HW[4];
  x3=X_HW[3];  y3=Y_HW[3];  x2=X_HW[2];  y2=Y_HW[2];
  x1=X_HW[1];  y1=Y_HW[1];  x0=X_HW[0];  y0=Y_HW[0];

  r8 = x7*y1 + x6*y2 + x5*y3 + x4*y4 + x3*y5 + x2*y6 + x1*y7;
  r7 = x7*y0 + x6*y1 + x5*y2 + x4*y3 + x3*y4 + x2*y5 + x1*y6 + x0*y7;
  r6 = x6*y0 + x5*y1 + x4*y2 + x3*y3 + x2*y4 + x1*y5 + x0*y6;
  r5 = x5*y0 + x4*y1 + x3*y2 + x2*y3 + x1*y4 + x0*y5;
  r4 = x4*y0 + x3*y1 + x2*y2 + x1*y3 + x0*y4 ;
  r3 = x3*y0 + x2*y1 + x1*y2 + x0*y3;
  r2 = x2*y0 + x1*y1 + x0*y2;
  r1 = x1*y0 + x0*y1 ;
  r0 = x0*y0 ;

  val= 0;
  /* Carry Propagate */
  SCS_CARRY_PROPAGATE(r8,r7,tmp)
    SCS_CARRY_PROPAGATE(r7,r6,tmp)
    SCS_CARRY_PROPAGATE(r6,r5,tmp)
    SCS_CARRY_PROPAGATE(r5,r4,tmp)
    SCS_CARRY_PROPAGATE(r4,r3,tmp)
    SCS_CARRY_PROPAGATE(r3,r2,tmp)
    SCS_CARRY_PROPAGATE(r2,r1,tmp)
    SCS_CARRY_PROPAGATE(r1,r0,tmp)
    SCS_CARRY_PROPAGATE(r0,val,tmp)

    if(val != 0){
      /* shift all the digits ! */
      R_HW[0] = val; R_HW[1] = r0; R_HW[2] = r1;  R_HW[3] = r2;
      R_HW[4] = r3;  R_HW[5] = r4; R_HW[6] = r5;  R_HW[7] = r6;
      R_IND += 1;
    }
    else {
      R_HW[0] = r0; R_HW[1] = r1; R_HW[2] = r2; R_HW[3] = r3;
      R_HW[4] = r4; R_HW[5] = r5; R_HW[6] = r6; R_HW[7] = r7;
    }
}

/** Convert a double precision number in it SCS multiprecision
  representation
 */
void scs_set_d(scs_ptr result, double x){
  db_number nb, mantissa;
  int exponent, exponent_remainder;
  int ind, i;

  if(x>=0){R_SGN = 1;    nb.d = x;}
  else    {R_SGN = -1;   nb.d = -x;}

  exponent = nb.i[HI] & 0x7ff00000 ;

  if (exponent == 0x7ff00000)  {
    /*
     * x = +/- Inf, s/qNAN
     */
    R_EXP = x;
    for(i=0; i<SCS_NB_WORDS; i++)
      R_HW[i] = 0;

    R_IND = 0;
    R_SGN = 1;
  }

  else {    /* Normals,  denormals, +/- 0.  */

    /* This number is not an exception */
    R_EXP = 1;

#if 1

    if (exponent == 0){
      /* x is a denormal number : bring it back to the normal range */
      nb.d = nb.d * SCS_RADIX_TWO_DOUBLE;      /* 2^(2.SCS_NB_BITS) */
      exponent = nb.i[HI] & 0x7ff00000 ;
      R_IND = -2;
    }else {
      R_IND = 0;
    }

    exponent = exponent >> 20;  /* get the actual value */

    ind = ((exponent +((100*SCS_NB_BITS)-1023))/SCS_NB_BITS) - 100 ;
    /* means : = (exponent -1023 + 100*SCS_NB_BITS)/SCS_NB_BITS -100
       The business with 100*SCS_NB_BITS is to stay within the positive
       range for exponent_remainder between 1 and SCS_NB_BITS */

    exponent_remainder = exponent - 1022 - (SCS_NB_BITS*ind);

    R_IND += ind;

    /* now get the mantissa and add the implicit 1 in fp. format*/
    mantissa.l = (nb.l & ULL(000fffffffffffff)) | ULL(0010000000000000);


    /* and spread it over the structure
       Everything here is 64-bit arithmetic */
    R_HW[0] = (unsigned int) (mantissa.l >> (53 - exponent_remainder) );

    /* 11 = 64-53 */
    mantissa.l =  (mantissa.l << (exponent_remainder+11));
    R_HW[1] = (mantissa.i[HI] >> (32 - SCS_NB_BITS))& SCS_MASK_RADIX ;
    mantissa.l =  (mantissa.l << SCS_NB_BITS);
    R_HW[2] = (mantissa.i[HI] >> (32 - SCS_NB_BITS))& SCS_MASK_RADIX ;
#if SCS_NB_BITS < 27
    mantissa.l =  (mantissa.l << SCS_NB_BITS);
    R_HW[3] = (mantissa.i[HI] >> (32 - SCS_NB_BITS))& SCS_MASK_RADIX ;
#else
    R_HW[3] = 0 ;
#endif

#if (SCS_NB_WORDS==8)
    R_HW[4] = 0; R_HW[5] = 0; R_HW[6] = 0; R_HW[7] = 0;
#else
    for(i=4; i<SCS_NB_WORDS; i++)
      R_HW[i] = 0;
#endif

#else /* Other algorithm as in the research report. Slower */
    R_IND = 0;

    while(nb.d>SCS_RADIX_ONE_DOUBLE) {
      R_IND++;
      nb.d *= SCS_RADIX_MONE_DOUBLE;
    }

    while(nb.d<1)  {
      R_IND--;
      nb.d *= SCS_RADIX_ONE_DOUBLE;
    }

    i=0;
    while(nb.d != 0){
      R_HW[i] = (unsigned int) nb.d;
      nb.d = (nb.d - (double)R_HW[i]) * SCS_RADIX_ONE_DOUBLE;
      i++;
    }
    for(; i<SCS_NB_WORDS; i++)
      R_HW[i] = 0;
#endif
  } /* end if test NaN etc */
  return;
}

/*  computes the exponent from the index */
/* in principle an inline function would be cleaner, but
   this leads to faster and smaller code
 */
void scs_get_d(double *result, scs_ptr x){
  db_number nb, rndcorr;
  uint64_t lowpart, t1;
  int expo, expofinal;
  double res;

  /* convert the MSB digit into a double, and store it in nb.d */
  nb.d = (double)X_HW[0];

  /* place the two next digits in lowpart */
  t1   = X_HW[1];
  lowpart  = (t1 << SCS_NB_BITS) + X_HW[2];
  /* there is at least one significant bit in nb,
     and at least 2*SCS_NB_BITS in lowpart,
     so provided SCS_NB_BITS >= 27
     together they include the 53+ guard bits to decide rounding
   */

  /* test for  s/qNan, +/- Inf, +/- 0, placed here for obscure performance reasons */
  if (X_EXP != 1){
    *result = X_EXP;
    return;
  }

  /* take the exponent of nb.d (will be in [0:SCS_NB_BITS])*/
  expo = ((nb.i[HI] & 0x7ff00000)>>20) - 1023;

  /* compute the exponent of the result */
  expofinal = expo + SCS_NB_BITS*X_IND;

  /* Is the SCS number not too large for the IEEE exponent range ? */
  if (expofinal >  1023) {
    /* return an infinity */
    res = SCS_RADIX_RNG_DOUBLE*SCS_RADIX_RNG_DOUBLE;
  }

  /* Is our SCS number a denormal  ? */
  else if (expofinal >= -1022){
    /* x is in the normal range */

    /* align the rest of the mantissa to nb : shift by (2*SCS_NB_BITS)-53-exp */
    lowpart = lowpart >> (expo+(2*SCS_NB_BITS)-53);
    /* Look at the last bit to decide rounding */
    if (lowpart & ULL(0000000000000001)){
      /* need to add an half-ulp */
      rndcorr.i[LO] = 0;
      rndcorr.i[HI] = (expo-52+1023)<<20;    /* 2^(exp-52) */
    }else{
      /* need to add nothing*/
      rndcorr.d = 0.0;
    }

    lowpart = lowpart >> 1;
    nb.l = nb.l | lowpart;    /* Finish to fill the mantissa */
    res  = nb.d + rndcorr.d;  /* rounded to nearest   */

    /* now compute the exponent from the index :
       we need to multiply res by 2^(X_IND*SCS_NB_BITS)
       First check this number won't be a denormal itself */
    if((X_IND)*SCS_NB_BITS +1023 > 0) {
      /* build the double 2^(X_IND*SCS_NB_BITS)   */
      nb.i[HI] = ((X_IND)*SCS_NB_BITS +1023)  << 20;
      nb.i[LO] = 0;
      res *= nb.d;     /* exact multiplication */
    }
    else { /*offset the previous computation by 2^(2*SCS_NB_BITS) */
      /* build the double 2^(X_IND*SCS_NB_BITS)   */
      nb.i[HI] = ((X_IND)*SCS_NB_BITS +1023 + 2*SCS_NB_BITS)  << 20;
      nb.i[LO] = 0;
      res *= SCS_RADIX_MTWO_DOUBLE;  /* exact multiplication */
      res *= nb.d;                  /* exact multiplication */
    }
  }

  else {
    /* the final number is a denormal with 52-(expfinal+1022)
       significant bits. */
    if (expofinal < -1022 - 53 ) {
      res = 0.0;
    }
    else {
      /* align the rest of the mantissa to nb */
      lowpart = lowpart >> (expo+(2*SCS_NB_BITS)-52);
      /* Finish to fill the mantissa */
      nb.l = nb.l | lowpart;

      /* this is still a normal number.
         Now remove its exponent and add back the implicit one */
      nb.l = (nb.l & ULL(000FFFFFFFFFFFFF)) | ULL(0010000000000000);

      /* keep only the significant bits */
      nb.l = nb.l >> (-1023 - expofinal);
      /* Look at the last bit to decide rounding */
      if (nb.i[LO] & 0x00000001){
        /* need to add an half-ulp */
        rndcorr.l = 1;    /* this is a full ulp but we multiply by 0.5 in the end */
      }else{
        /* need to add nothing*/
        rndcorr.d = 0.0;

      }
      res  = 0.5*(nb.d + rndcorr.d);  /* rounded to nearest   */

      /* the exponent field is already set to zero so that's all */
    }
  }
  /* sign management */
  if (X_SGN < 0)
    *result = - res;
  else
    *result = res;
}
