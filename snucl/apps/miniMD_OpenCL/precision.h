/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National 
   Laboratories ( http://www.mantevo.org ). The primary 
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier 
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you 
   can redistribute it and/or modify it under the terms of the GNU Lesser 
   General Public License as published by the Free Software Foundation; 
   either version 3 of the License, or (at your option) any later 
   version.
  
   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
   Lesser General Public License for more details.
    
   You should have received a copy of the GNU Lesser General Public 
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov). 

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#ifndef PRECISION_H_
#define PRECISION_H_

#ifndef IAMONDEVICE
#include <CL/cl.h>
#endif
#ifndef MDPREC
#define MDPREC 2
#endif
#if MDPREC == 2
#pragma OPENCL EXTENSION cl_khr_fp64 : disable
#ifndef double3
typedef struct sdouble3{
 double x;
 double y;
 double z;
} double3;
//typedef struct sdouble3 double3;
#endif

#ifndef double4
struct sdouble4{
 double x;
 double y;
 double z;
 double w;
};
typedef struct sdouble4 double4;
#endif

typedef cl_double3 MMD_float3;
typedef cl_double4 MMD_float4;


typedef double3 MMD_floatK3;
typedef double4 MMD_floatK4;
typedef double MMD_float;
#define F(a) a

#endif //MDPREC == 2

#if MDPREC == 1

#ifndef float3
struct sfloat3{
 float x;
 float y;
 float z;
};
#ifndef IAMONDEVICE
typedef struct sfloat3 float3;
#endif
#endif

#ifndef float4
struct sfloat4{
 float x;
 float y;
 float z;
 float w;
};
#ifndef IAMONDEVICE
typedef struct sfloat4 float4;
#endif
#endif

#ifndef IAMONDEVICE
typedef cl_float3 MMD_float3;
typedef cl_float4 MMD_float4;
#endif
typedef float3 MMD_floatK3 ;
typedef float4 MMD_floatK4 ;
typedef float MMD_float;
#define F(a) a.f

#endif //MDPREC == 1

#ifndef PRECMPI
#define PRECMPI MPI_DOUBLE
#endif

#endif /* PRECISION_H_ */
