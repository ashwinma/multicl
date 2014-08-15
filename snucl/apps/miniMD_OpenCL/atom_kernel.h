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

__kernel void atom_pack_comm(__global MMD_floatK3* x, __global MMD_float* buf, __global int* list, int offset, MMD_floatK3 pbc, int n)
{
	  list += offset;
	  int j = get_global_id(0);
	  if(j<n)
	  {
		  int i=list[j];
		  MMD_floatK3 xi=x[i]+pbc;
		  buf[3*j]=xi.x;
		  buf[3*j+1]=xi.y;
		  buf[3*j+2]=xi.z;
	  }
}

__kernel void atom_unpack_comm(__global MMD_floatK3* x, __global MMD_float* buf, int first, int n)
{
	  int i = get_global_id(0);
	  if(i<n)
	  {
		  MMD_floatK3 xi;
		  xi.x=buf[3*i];
		  xi.y=buf[3*i+1];
		  xi.z=buf[3*i+2];
		  x[i+first]=xi;

	  }

}

__kernel void atom_comm_self(__global MMD_floatK3* x, __global int* list, int offset, MMD_floatK3 pbc, int first, int n)
{
	  list += offset;
	  int j = get_global_id(0);
	  if(j<n)
	  {
		  int i=list[j];
		  x[j+first] = x[i]+pbc;

	  }


}

