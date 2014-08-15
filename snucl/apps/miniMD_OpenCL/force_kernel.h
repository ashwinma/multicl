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

/*
 * force_kernel.h
 *
 *  Created on: Jan 19, 2012
 *      Author: ctrott
 */

#include "precision.h"
//#define MMD_floatK3 float3;
//#define MMD_float float;
const sampler_t TEXMODE = CLK_ADDRESS_NONE|CLK_NORMALIZED_COORDS_FALSE;
__inline float4 fetch_tex(__read_only image2d_t I,int i,int size) {return read_imagef(I,TEXMODE,(int2)(i%size,i/size));};

__kernel void force_compute(__global MMD_floatK3* x, __global MMD_floatK3* f, __global int* numneigh,
		  	  	  	  	  	  __global int* neighbors, int maxneighs, int nlocal, MMD_float cutforcesq)
{
  int i = get_global_id(0);
  if(i<nlocal)
  {

  	__global int* neighs = neighbors + i;
    MMD_floatK3 ftmp;
    MMD_floatK3 xi = x[i];
    MMD_floatK3 fi = {0.0f,0.0f,0.0f};

    for (int k = 0; k < numneigh[i]; k++) {
      int j = neighs[k*nlocal];
      MMD_floatK3 delx = xi - x[j];

      MMD_float rsq = delx.x*delx.x + delx.y*delx.y + delx.z*delx.z;
      if (rsq < cutforcesq) {
	MMD_float sr2 = 1.0f/rsq;
	MMD_float sr6 = sr2*sr2*sr2;
	MMD_float force = 48.0f*sr6*(sr6-0.5f)*sr2;
	  fi += force * delx;
      }

    }


    f[i] = fi;

 }

}

__kernel void force_compute_tex(__read_only image2d_t x, __global MMD_floatK3* f, __global int* numneigh,
		  	  	  	  	  	  __global int* neighbors, int maxneighs, int nlocal, MMD_float cutforcesq,int imagesize)
{
  int i = get_global_id(0);
  if(i<nlocal)
  {

  	__global int* neighs = neighbors + i;
    float4 xi = fetch_tex(x,i,imagesize);
    MMD_floatK4 fi = {0.0f,0.0f,0.0f,0.0f};

    for (int k = 0; k < numneigh[i]; k++) {
      int j = neighs[k*nlocal];
      MMD_floatK4 delx = xi - fetch_tex(x,j,imagesize);

      MMD_float rsq = delx.x*delx.x + delx.y*delx.y + delx.z*delx.z;
      if (rsq < cutforcesq) {
	MMD_float sr2 = 1.0f/rsq;
	MMD_float sr6 = sr2*sr2*sr2;
	MMD_float force = 48.0f*sr6*(sr6-0.5f)*sr2;
	  fi += force * delx;
      }

    }

    MMD_floatK3 ftmp = {fi.x,fi.y,fi.z};

    f[i] = ftmp;

 }

}

__kernel void force_compute_loop(__global MMD_floatK3* x, __global MMD_floatK3* f, __global int* numneigh,
		  	  	  	  	  	  __global int* neighbors, int maxneighs, int nlocal, MMD_float cutforcesq, int atoms_per_thread)
{
  int ii = get_global_id(0);
  for(int i=ii;i<nlocal;i+=get_global_size(0))
  if(i<nlocal)
  {

  	__global int* neighs = neighbors + i;
    MMD_floatK3 ftmp;
    MMD_floatK3 xi = x[i];
    MMD_floatK3 fi = {0.0f,0.0f,0.0f};

    for (int k = 0; k < numneigh[i]; k++) {
      int j = neighs[k*nlocal];
      MMD_floatK3 delx = xi - x[j];

      MMD_float rsq = delx.x*delx.x + delx.y*delx.y + delx.z*delx.z;
      if (rsq < cutforcesq) {
	MMD_float sr2 = 1.0f/rsq;
	MMD_float sr6 = sr2*sr2*sr2;
	MMD_float force = 48.0f*sr6*(sr6-0.5f)*sr2;
	  fi += force * delx;

      }

    }


    f[i] = fi;

 }

}

__kernel void force_compute_split(__global MMD_floatK3* x, __global MMD_floatK3* f, __global int* numneigh,
		  	  	  	  	  	  __global int* neighbors, int maxneighs, int nlocal, MMD_float cutforcesq, int threads_per_atom,
		  	  	  	  	  	  __local MMD_floatK3* sf)
{
  int ii = get_global_id(0);
  int k = get_local_id(0);
  int nk = get_local_size(0);
  int i = ii/threads_per_atom;
  int jl = ii%threads_per_atom;

  if(i<nlocal)
  {

  	__global int* neighs = neighbors + i;
    MMD_floatK3 ftmp;
    MMD_floatK3 xi = x[i];
    MMD_floatK3 fi = {0.0f,0.0f,0.0f};

    for (int jj = jl; jj < numneigh[i]; jj+=threads_per_atom) {
      int j = neighs[jj*nlocal];
      MMD_floatK3 delx = xi - x[j];

      MMD_float rsq = delx.x*delx.x + delx.y*delx.y + delx.z*delx.z;
      if (rsq < cutforcesq) {
	MMD_float sr2 = 1.0f/rsq;
	MMD_float sr6 = sr2*sr2*sr2;
	MMD_float force = 48.0f*sr6*(sr6-0.5f)*sr2;
	  fi += force * delx;

      }

    }

    sf[k] = fi;
    for(int m=threads_per_atom/2;m>0;m/=2)
    {
    	if(jl<m)
    	sf[k]+=sf[k+m];
    }
    if(jl==0)
    f[i] = sf[k];

 }

}
