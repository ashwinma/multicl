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
#include "stdlib.h"
#include "mpi.h"
#include "integrate.h"
#include "thermo.h"

Thermo::Thermo() {}
Thermo::~Thermo() {}

void Thermo::setup(MMD_float rho_in, Integrate &integrate, Atom &atom,int units)
{
  rho = rho_in;
  ntimes = integrate.ntimes;

  int maxstat;
  if (nstat == 0) maxstat = 2;
  else maxstat = ntimes/nstat + 1;
  steparr = (int *) malloc(maxstat*sizeof(int));
  tmparr = (MMD_float *) malloc(maxstat*sizeof(MMD_float));
  engarr = (MMD_float *) malloc(maxstat*sizeof(MMD_float));
  prsarr = (MMD_float *) malloc(maxstat*sizeof(MMD_float));

  if(units == LJ) {
    mvv2e = 1.0;
    dof_boltz = (atom.natoms * 3 - 3);
    t_scale = mvv2e / dof_boltz;
    p_scale = 1.0 / 3 / atom.box.xprd / atom.box.yprd / atom.box.zprd;
    e_scale = 0.5;
  } else if(units == METAL) {
    mvv2e = 1.036427e-04;
    dof_boltz = (atom.natoms * 3 - 3) * 8.617343e-05;
    t_scale = mvv2e / dof_boltz;
    p_scale = 1.602176e+06 / 3 / atom.box.xprd / atom.box.yprd / atom.box.zprd;
    e_scale = 524287.985533;//16.0;
    integrate.dtforce /= mvv2e;

  }

}

void Thermo::compute(int iflag, Atom &atom, Neighbor &neighbor, Force &force, Timer &timer, Comm &comm)
{
  MMD_float t,eng,p;

  if (iflag > 0 && iflag % nstat) return;
  if (iflag == -1 && nstat > 0 && ntimes % nstat == 0) return;

  t_act=0;
  e_act=0;
  p_act=0;

  atom.d_vold->download();
  atom.d_v->download();
  atom.d_x->download();
  neighbor.d_numneigh->download();
  neighbor.d_neighbors->download();
  t = temperature(atom);
  MMD_float2 ev = energy_virial(atom,neighbor,force);
  eng = 0.5*ev.x;
  p = (t * dof_boltz + 0.5*ev.y) * p_scale;

  int istep = iflag;
  if (iflag == -1) istep = ntimes;

  if (iflag == 0) mstat = 0;

  steparr[mstat] = istep;
  tmparr[mstat] = t;
  engarr[mstat] = eng;
  prsarr[mstat] = p;

  mstat++;

  double oldtime=timer.array[TIME_TOTAL];
  timer.barrier_stop(TIME_TOTAL);
  if(threads->mpi_me == 0)
  {
    fprintf(stdout,"%i %e %e %e %6.3lf\n",istep,t,eng,p,istep==0?0.0:timer.array[TIME_TOTAL]);
  }
  timer.array[TIME_TOTAL]=oldtime;
}

/* reduced potential energy */

MMD_float2 Thermo::energy_virial(Atom &atom, Neighbor &neighbor, Force &force)
{
  int i,j,k,numneigh;
  MMD_float delx,dely,delz,rsq,sr2,sr6,phi,pair;
  int *neighs;

  MMD_float2 ev = {0.0, 0.0};
  for (i = 0; i < atom.nlocal; i++) {
    neighs = &neighbor.neighbors[i];
    numneigh = neighbor.numneigh[i];
    for (k = 0; k < numneigh; k++) {
      j = neighs[k*atom.nlocal];
      delx = atom.x[i].x - atom.x[j].x;
      dely = atom.x[i].y - atom.x[j].y;
      delz = atom.x[i].z - atom.x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < force.cutforcesq) {
	sr2 = 1.0/rsq;
	sr6 = sr2*sr2*sr2;
	phi = sr6*(sr6-1.0);
	pair = 48.0 * sr6 * (sr6 - 0.5) * sr2;
	ev.x += 4.0*phi;
	ev.y += (delx * delx + dely * dely + delz * delz) * pair;

      }
    }
  }

  MMD_float tmp = ev.x/atom.natoms;
  MPI_Allreduce(&tmp,&ev.x,1,PRECMPI,MPI_SUM,MPI_COMM_WORLD);
  tmp = ev.y;
  MPI_Allreduce(&tmp,&ev.y,1,PRECMPI,MPI_SUM,MPI_COMM_WORLD);
  return ev;
}

/*  reduced temperature */

MMD_float Thermo::temperature(Atom &atom)
{
  int i;
  MMD_float vx,vy,vz;

  MMD_float t = 0.0;
  for (i = 0; i < atom.nlocal; i++) {
    vx = atom.v[i].x;
    vy = atom.v[i].y;
    vz = atom.v[i].z;
    t += vx*vx + vy*vy + vz*vz;
  }

  MMD_float t1;
  MPI_Allreduce(&t,&t1,1,PRECMPI,MPI_SUM,MPI_COMM_WORLD);
  return t1 * t_scale;
}
      
      
/* reduced pressure from virial
   virial = Fi dot Ri summed over own and ghost atoms, since PBC info is
   stored correctly in force array before reverse_communicate is performed */

MMD_float Thermo::pressure(MMD_float t, Atom &atom)
{
  int i;

  MMD_float virial = 0.0;
  for (i = 0; i < atom.nlocal; i++)
    virial += atom.f[i].x*atom.x[i].x + atom.f[i].y*atom.x[i].y +
      atom.f[i].z*atom.x[i].z;

  MMD_float virtmp = 48.0*virial;
  MPI_Allreduce(&virtmp,&virial,1,PRECMPI,MPI_SUM,MPI_COMM_WORLD);
  return (t*rho + rho/3.0/atom.natoms * virial*1.0);
}

