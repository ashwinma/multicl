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
#include "math.h"
#include "force.h"

Force::Force() {}
Force::~Force() {}

void Force::setup()
{
  cutforcesq = cutforce*cutforce;
}

void Force::compute(Atom &atom, Neighbor &neighbor, int me)
{
	if(atom.threads_per_atom<0)
	    opencl->LaunchKernel("force_compute_loop",-(atom.nlocal-atom.threads_per_atom-1)/atom.threads_per_atom,8,
	    		atom.d_x->devDataRef(),sizeof(atom.d_x->devDataRef()),
	    		atom.d_f->devDataRef(),sizeof(atom.d_f->devDataRef()),
	    		neighbor.d_numneigh->devDataRef(),sizeof(neighbor.d_numneigh->devDataRef()),
	    		neighbor.d_neighbors->devDataRef(),sizeof(neighbor.d_neighbors->devDataRef()),
	    		&neighbor.maxneighs,sizeof(neighbor.maxneighs),&atom.nlocal,sizeof(atom.nlocal),
	    		&cutforcesq,sizeof(cutforcesq),&atom.threads_per_atom,sizeof(atom.threads_per_atom));
	else if(atom.threads_per_atom>1)
	    opencl->LaunchKernel("force_compute_split",atom.nlocal*atom.threads_per_atom,9,
	    		atom.d_x->devDataRef(),sizeof(atom.d_x->devDataRef()),
	    		atom.d_f->devDataRef(),sizeof(atom.d_f->devDataRef()),
	    		neighbor.d_numneigh->devDataRef(),sizeof(neighbor.d_numneigh->devDataRef()),
	    		neighbor.d_neighbors->devDataRef(),sizeof(neighbor.d_neighbors->devDataRef()),
	    		&neighbor.maxneighs,sizeof(neighbor.maxneighs),&atom.nlocal,sizeof(atom.nlocal),
	    		&cutforcesq,sizeof(cutforcesq),&atom.threads_per_atom,sizeof(atom.threads_per_atom),
	    		NULL,sizeof(MMD_float3)*opencl->blockdim);
	else if(atom.use_tex)
    opencl->LaunchKernel("force_compute_tex",atom.nlocal,8,
    		atom.d_x->devImageRef(),sizeof(atom.d_x->devData()),
    		atom.d_f->devDataRef(),sizeof(atom.d_f->devData()),
    		neighbor.d_numneigh->devDataRef(),sizeof(neighbor.d_numneigh->devDataRef()),
    		neighbor.d_neighbors->devDataRef(),sizeof(neighbor.d_neighbors->devDataRef()),
    		&neighbor.maxneighs,sizeof(neighbor.maxneighs),&atom.nlocal,sizeof(atom.nlocal),
    		&cutforcesq,sizeof(cutforcesq),&atom.d_x->imagesize,sizeof(atom.d_x->imagesize));
    else
	opencl->LaunchKernel("force_compute",atom.nlocal,7,
	    		atom.d_x->devDataRef(),sizeof(atom.d_x->devDataRef()),
	    		atom.d_f->devDataRef(),sizeof(atom.d_f->devDataRef()),
	    		neighbor.d_numneigh->devDataRef(),sizeof(neighbor.d_numneigh->devDataRef()),
	    		neighbor.d_neighbors->devDataRef(),sizeof(neighbor.d_neighbors->devDataRef()),
	    		&neighbor.maxneighs,sizeof(neighbor.maxneighs),&atom.nlocal,sizeof(atom.nlocal),
	    		&cutforcesq,sizeof(cutforcesq));
	    return;
  int i,j,k,nlocal,nall,numneigh;
  MMD_float xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  MMD_float sr2,sr6,force;
  int *neighs;
  MMD_float3 *x,*f;

  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;
  x = atom.x;
  f = atom.f;

  // clear force on own and ghost atoms

  for (i = 0; i < nall; i++) {
    f[i].x = 0.0;
    f[i].y = 0.0;
    f[i].z = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j
  
  for (i = 0; i < nlocal; i++) {
    neighs = &neighbor.neighbors[i*neighbor.maxneighs];
    numneigh = neighbor.numneigh[i];
    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    for (k = 0; k < numneigh; k++) {
      j = neighs[k];
      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < cutforcesq) {
	sr2 = 1.0/rsq;
	sr6 = sr2*sr2*sr2;
	force = sr6*(sr6-0.5)*sr2;
	f[i].x += delx*force;
	f[i].y += dely*force;
	f[i].z += delz*force;
      }
    }
 }
}
