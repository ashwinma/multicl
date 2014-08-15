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
#include "integrate.h"

Integrate::Integrate() {}
Integrate::~Integrate() {}

void Integrate::setup()
{
  dtforce = 0.5*dt;
}

void Integrate::run(Atom &atom, Force &force, Neighbor &neighbor,
		    Comm &comm, Thermo &thermo, Timer &timer)
{
  timer.array[TIME_TEST]=0.0;

  atom.d_x->upload();
  atom.d_v->upload();
  comm.d_sendlist->upload();
  for (int n = 0; n < ntimes; n++) {
	    //atom.d_x->download();
	    //atom.d_v->download();
	    //atom.d_f->download();
	    //for(int i=0;i<atom.nlocal;i++) printf("%i %e %e %e // %e %e %e // %e %e %e // %e %e\n",i,atom.x[i].x,atom.x[i].y,atom.x[i].z,atom.v[i].x,atom.v[i].y,atom.v[i].z,atom.f[i].x,atom.f[i].y,atom.f[i].z,dt,dtforce);
    opencl->LaunchKernel("integrate_initial",atom.nlocal,7,
    		atom.d_x->devDataRef(),sizeof(atom.d_x->devDataRef()),
    		atom.d_v->devDataRef(),sizeof(atom.d_v->devDataRef()),
    		atom.d_f->devDataRef(),sizeof(atom.d_f->devDataRef()),
    		&atom.nlocal,sizeof(atom.nlocal),&dt,sizeof(dt),&dtforce,sizeof(dtforce),&atom.nmax,sizeof(atom.nmax));
    //neighbor.d_flag->download();
    clFinish ( 	opencl->defaultQueue);
    timer.stamp();

    if ((n+1) % neighbor.every) {
      comm.communicate(atom);
      clFinish ( 	opencl->defaultQueue);
      timer.stamp(TIME_COMM);
    } else {
      atom.d_x->download();
      atom.d_v->download();

      comm.exchange(atom);

      comm.borders(atom);
      atom.d_v->upload();
      atom.d_x->upload();
      comm.d_sendlist->upload();
      clFinish ( 	opencl->defaultQueue);

      timer.stamp(TIME_COMM);
      neighbor.build(atom);
      clFinish (opencl->defaultQueue);
    timer.stamp(TIME_NEIGH);
   }
    if(atom.use_tex)
    atom.d_x->syncImage();
    force.compute(atom,neighbor,comm.me);
    //neighbor.d_flag->download();
    clFinish ( 	opencl->defaultQueue);
    timer.stamp(TIME_FORCE);

    opencl->LaunchKernel("integrate_final",atom.nlocal,5,
    		atom.d_v->devDataRef(),sizeof(atom.d_v->devDataRef()),
    		atom.d_f->devDataRef(),sizeof(atom.d_f->devDataRef()),
    		&atom.nlocal,sizeof(atom.nlocal),&dtforce,sizeof(dtforce),&atom.nmax,sizeof(atom.nmax));
    if(thermo.nstat)
    {
      thermo.compute(n+1,atom,neighbor,force,timer,comm);
    }

  }
  atom.d_x->download();
  atom.d_v->download();
  atom.d_f->download();
}
