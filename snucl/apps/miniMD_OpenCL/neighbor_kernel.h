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

#include "precision.h"
//#define MMD_floatK3 float3;
//#define MMD_float float;
/*
int coord2bin(MMD_floatK3 &x,MMD_floatK3 &bininv, MMD_floatK3 &prd, int3 &mbinlo, int3 &nbin, int3 &mbin)
{
	int ix = (int) ((x.x-(x.x>=prd.x)*prd.x)*bininv.x) + nbin.x - mbinlo.x - (x.x<0.0):
    int iy = (int) ((x.y-(x.y>=prd.y)*prd.y)*bininv.y) + nbin.y - mbinlo.y - (x.y<0.0):
    int iz = (int) ((x.z-(x.z>=prd.z)*prd.z)*bininv.z) + nbin.z - mbinlo.z - (x.z<0.0):
						  ;
	return (iz*mbin.y*mbin.x + iy*mbin.x + ix + 1);
}*/

__kernel void neighbor_build(__global MMD_floatK3* x, __global int* numneigh, __global int* neighbors,
		__global int* bincount, __global int* bins, __global int* ibins, __global int* flag,
		__global int* stencil, int nstencil, MMD_float cutneighsq, int atoms_per_bin, int maxneighs, int nlocal)//,MMD_floatK3 &bininv, MMD_floatK3 prd, int3 &mbinlo, int3 nbin, int3 mbin)
{

	int i = get_global_id(0);
	if(i>=nlocal) return;
	int ibin = ibins[i];
	MMD_floatK3 xtmp = x[i];
	int n = 0;
	for(int k = 0; k < nstencil; k++)
	{
		int jbin = ibin + stencil[k];
	    for(int m=0;m<bincount[jbin];m++)
	    {
	      int j = bins[jbin*atoms_per_bin+m];
	      MMD_floatK3 del = xtmp - x[j];
	      MMD_float rsq = del.x*del.x + del.y*del.y + del.z*del.z;
	      if ((rsq <= cutneighsq)&&(j!=i)) neighbors[i+n++*nlocal] = j;

	    }

	}

	numneigh[i] = n;
	if(n>maxneighs)
	  	flag[0] = 1;

}


__kernel void neighbor_bin(__global MMD_floatK3* xg, __global int* bincount, __global int* bins,
		__global int* ibins, __global int* flag, int atoms_per_bin, int nall,
		MMD_floatK3 bininv, MMD_floatK3 prd, int3 mbinlo, int3 nbin, int3 mbin)
{
	int i = get_global_id(0);
	if(i>=nall) return;
	MMD_floatK3 x = xg[i];
	/*  int ix;
	  if (x.x >= prd.x)
	    ix = (int) ((x.x-prd.x)*bininv.x) + nbin.x - mbinlo.x;
	  else if (x.x >= 0.0)
	    ix = (int) (x.x*bininv.x) - mbinlo.x;
	  else
	    ix = (int) (x.x*bininv.x) - mbinlo.x - 1;
	  int iy;
	  if (x.y >= prd.y)
	    iy = (int) ((x.y-prd.y)*bininv.y) + nbin.y - mbinlo.y;
	  else if (x.y >= 0.0)
	    iy = (int) (x.y*bininv.y) - mbinlo.y;
	  else
	    iy = (int) (x.y*bininv.y) - mbinlo.y - 1;
	  int iz;
	  if (x.z >= prd.z)
	    iz = (int) ((x.z-prd.z)*bininv.z) + nbin.z - mbinlo.z;
	  else if (x.z >= 0.0)
	    iz = (int) (x.z*bininv.z) - mbinlo.z;
	  else
	    iz = (int) (x.z*bininv.z) - mbinlo.z - 1;*/
	int3 doit = x>=prd;
	int ix = (int) ((x.x-(x.x>=prd.x)*prd.x)*bininv.x) - nbin.x*doit.x - mbinlo.x - (x.x<0.0);
    int iy = (int) ((x.y-(x.y>=prd.y)*prd.y)*bininv.y) - nbin.y*doit.y - mbinlo.y - (x.y<0.0);
    int iz = (int) ((x.z-(x.z>=prd.z)*prd.z)*bininv.z) - nbin.z*doit.z - mbinlo.z - (x.z<0.0);
						  ;
	int ibin = (iz*mbin.y*mbin.x + iy*mbin.x + ix + 1);
	//int ibin = coord2bin(x[i],bininv,prd,mbinlo,nbin,mbin);
	ibins[i] = ibin;
	int pos = atomic_add(&bincount[ibin],1);
	if(pos<atoms_per_bin) bins[ibin*atoms_per_bin+pos]=i;
	else flag[0]=1;


}

