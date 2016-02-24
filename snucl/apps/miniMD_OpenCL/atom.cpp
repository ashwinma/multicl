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

#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "mpi.h"
#include "atom.h"

#define DELTA 20000

Atom::Atom()
{
  natoms = 0;
  nlocal = 0;
  nghost = 0;
  nmax = 0;

  x = v = f = vold = NULL;

  comm_size = 3;
  reverse_size = 3;
  border_size = 3;

  threads_per_atom = 1;
  use_tex = 0;
}

Atom::~Atom()
{
  if (nmax) {
	  delete d_x;
	  delete d_v;
	  delete d_f;
	  delete d_vold;
  }
}

void Atom::growarray()
{
  int nold = nmax;
  if(nmax==0) nmax=300000;
  nmax += DELTA;

  d_x = new cOpenCLData<MMD_float3,MMD_float3,xx>(opencl,nmax,0,0,true,true);
  d_v = new cOpenCLData<MMD_float3,MMD_float3,xx>(opencl,nmax,0,0,true);
  d_f = new cOpenCLData<MMD_float3,MMD_float3,xx>(opencl,nmax,0,0,true);
  d_vold = new cOpenCLData<MMD_float3,MMD_float3,xx>(opencl,nmax,0,0,true);
  x = d_x->hostData();
  v = d_v->hostData();
  f = d_f->hostData();
  vold = d_vold->hostData();


  if (x == NULL || v == NULL || f == NULL || vold == NULL) {
    printf("ERROR: No memory for atoms\n");
  }
}

void Atom::addatom(MMD_float x_in, MMD_float y_in, MMD_float z_in, 
		   MMD_float vx_in, MMD_float vy_in, MMD_float vz_in)
{
  if (nlocal == nmax) growarray();

  x[nlocal].x = x_in;
  x[nlocal].y = y_in;
  x[nlocal].z = z_in;
  v[nlocal].x = vx_in;
  v[nlocal].y = vy_in;
  v[nlocal].z = vz_in;

  nlocal++;
}

/* enforce PBC
   order of 2 tests is important to insure lo-bound <= coord < hi-bound
   even with round-off errors where (coord +/- epsilon) +/- period = bound */

void Atom::pbc()
{
  for (int i = 0; i < nlocal; i++) {
    if (x[i].x < 0.0) x[i].x += box.xprd;
    if (x[i].x >= box.xprd) x[i].x -= box.xprd;
    if (x[i].y < 0.0) x[i].y += box.yprd;
    if (x[i].y >= box.yprd) x[i].y -= box.yprd;
    if (x[i].z < 0.0) x[i].z += box.zprd;
    if (x[i].z >= box.zprd) x[i].z -= box.zprd;
  }
}

void Atom::copy(int i, int j)
{
  x[j].x = x[i].x;
  x[j].y = x[i].y;
  x[j].z = x[i].z;
  v[j].x = v[i].x;
  v[j].y = v[i].y;
  v[j].z = v[i].z;
}

void Atom::pack_comm(int n, int *list, MMD_float *buf, int *pbc_flags)
{
  int i,j,m;

  m = 0;
  if (pbc_flags[0] == 0) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j].x;
      buf[m++] = x[j].y;
      buf[m++] = x[j].z;
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = x[j].x + pbc_flags[1]*box.xprd;
      buf[m++] = x[j].y + pbc_flags[2]*box.yprd;
      buf[m++] = x[j].z + pbc_flags[3]*box.zprd;
    }
  }
}

void Atom::unpack_comm(int n, int first, MMD_float *buf)
{
  int i,j,m;
  m = 0;
  j = first;
  for (i = 0; i < n; i++, j++) {
    x[j].x = buf[m++];
    x[j].y = buf[m++];
    x[j].z = buf[m++];
  }
}

void Atom::pack_reverse(int n, int first, MMD_float *buf)
{
  int i,j,m;
  m = 0;
  j = first;
  for (i = 0; i < n; i++, j++) {
    buf[m++] = f[j].x;
    buf[m++] = f[j].y;
    buf[m++] = f[j].z;
  }
}

void Atom::unpack_reverse(int n, int *list, MMD_float *buf)
{
  int i,j,m;
  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j].x += buf[m++];
    f[j].y += buf[m++];
    f[j].z += buf[m++];
  }
}

int Atom::pack_border(int i, MMD_float *buf, int *pbc_flags)
{
  int m = 0;
  if (pbc_flags[0] == 0) {
    buf[m++] = x[i].x;
    buf[m++] = x[i].y;
    buf[m++] = x[i].z;
  } else {
    buf[m++] = x[i].x + pbc_flags[1]*box.xprd;
    buf[m++] = x[i].y + pbc_flags[2]*box.yprd;
    buf[m++] = x[i].z + pbc_flags[3]*box.zprd;
  }
  return m;
}

int Atom::unpack_border(int i, MMD_float *buf)
{
  if (i == nmax) growarray();

  int m = 0;
  x[i].x = buf[m++];
  x[i].y = buf[m++];
  x[i].z = buf[m++];
  return m;
}

int Atom::pack_exchange(int i, MMD_float *buf)
{
  int m = 0;
  buf[m++] = x[i].x;
  buf[m++] = x[i].y;
  buf[m++] = x[i].z;
  buf[m++] = v[i].x;
  buf[m++] = v[i].y;
  buf[m++] = v[i].z;
  return m;
}

int Atom::unpack_exchange(int i, MMD_float *buf)
{
  if (i == nmax) growarray();

  int m = 0;
  x[i].x = buf[m++];
  x[i].y = buf[m++];
  x[i].z = buf[m++];
  v[i].x = buf[m++];
  v[i].y = buf[m++];
  v[i].z = buf[m++];
  return m;
}

int Atom::skip_exchange(MMD_float *buf)
{
  return 6;
}

/* realloc a 2-d MMD_float array */

MMD_float **Atom::realloc_2d_MMD_float_array(MMD_float **array, 
				       int n1, int n2, int nold)

{
  MMD_float **newarray;

  newarray = create_2d_MMD_float_array(n1,n2);
  if (nold) memcpy(newarray[0],array[0],nold*sizeof(MMD_float));
  destroy_2d_MMD_float_array(array);

  return newarray;
}

/* create a 2-d MMD_float array */

MMD_float **Atom::create_2d_MMD_float_array(int n1, int n2)

{
  MMD_float **array;
  MMD_float *data;
  int i,n;

  if (n1*n2 == 0) return NULL;

  array = (MMD_float **) malloc(n1*sizeof(MMD_float *));
  data = (MMD_float *) malloc(n1*n2*sizeof(MMD_float));

  n = 0;
  for (i = 0; i < n1; i++) {
    array[i] = &data[n];
    n += n2;
  }

  return array;
}

/* free memory of a 2-d MMD_float array */

void Atom::destroy_2d_MMD_float_array(MMD_float **array)

{
  if (array != NULL) {
    free(array[0]);
    free(array);
  }
}
