#include "switches.h"
!-----------------------------------------------------------------------
subroutine read_inp(myid_world,numprocs,fileInp,fileMod,fileRec, &
                    fileInf,fileOut,nd2_pml,RF,ierr)
! Input all the parameters need for calculation
 use mpi
 use comput_input_param
 implicit NONE
 integer, intent(IN):: myid_world,numprocs
 character(len=*), intent(IN):: fileInp
 integer, intent(OUT):: nd2_pml,ierr
 real, intent(OUT):: RF
 character(len=*), intent(OUT):: fileMod,fileRec,fileInf,fileOut
 integer:: nsubx,nsuby,getlen,ncount,i
 integer, dimension(5)::intemp
 real:: dxsub,dysub,cxp,cyp
 real, dimension(8):: rltemp
 character(len=72):: file_sou_list
!
 rltemp=0.0
 intemp=0
 if(myid_world  == 0) then
   open(unit=8,file=fileInp,status='old', position='rewind')
!
!*********************************************************************
! Enter the name of file to store the information
!*********************************************************************
   read(8,'(1a)') fileInf
   open(unit=13,file=fileInf,status='replace')
   write(13,*) ' Enter the name of file for store the information'
   write(13,'(1a)') fileInf
!
!*********************************************************************
! Read and print time stepping data
!*********************************************************************
   write(13,*) 'Enter computing duration (sec) and print interval'
   read(8,*)   tdura, intprt
   write(13,*) tdura, intprt
!
!*********************************************************************
! Read and print X-Axis direction of FD system
!*********************************************************************
   write(13,*) 'Enter the angle (degree) from North to X-Axis ', &
               '(clockwise) of FD coordinate system'
   read(8,*)  angle_north_to_x
   write(13,*) angle_north_to_x
!
!*********************************************************************
! Read and print the coordinates of the given reference location
! in the FD coordinate system
!*********************************************************************
   write(13,*) 'Enter the (x,y) value of given reference point', &
               'in the FD coordinate system'
   read(8,*)   xref_fdm,yref_fdm
   write(13,*) xref_fdm,yref_fdm
!
!*********************************************************************
! Read and print Partition number of FD zone
!*********************************************************************
   write(13,*) 'Partition number of FD zone in X, Y direction'
   read(8,*)  nproc_x,nproc_y
   write(13,*) nproc_x,nproc_y
!
!*********************************************************************
! Read and print the nodes number of PML zone
!*********************************************************************
   write(13,*) 'Set the nodes number of PML zone in Region II'
   write(13,*) 'and the theoretical reflect factor'
   read(8,*)  nd2_pml, RF
   write(13,*) nd2_pml, RF
!
!*********************************************************************
! Read and print the name of file with velocity structure 
!*********************************************************************
   write(13,*) ' Enter the name of file with velocity structure'
   read(8,'(1a)') fileMod
   write(13,'(1a)') fileMod
!
!*********************************************************************
! Read and print the name of file with source parameters 
!*********************************************************************
   write(13,*) ' Enter the name of file list source file names'
   read(8,'(1a)') file_sou_list
   write(13,'(1a)') file_sou_list
!
!*********************************************************************
! Read and print the name of file with receiver locations 
!*********************************************************************
   write(13,*) ' Enter the name of file with receiver locations'
   read(8,'(1a)') fileRec
   write(13,'(1a)') fileRec
!*********************************************************************
! Read and print name of file for store the outputing Velocity 
!*********************************************************************
   write(13,*) ' Give the file name for outputing the Velocity.'
   read(8,'(1a)')  fileOut
   write(13,'(1a)') fileOut
   close(8)
!======================== END of INPUT ==============================
!
!
!======================================================================
! ------- Input the name of source files  -----------------------
!======================================================================
!
!
   open(unit=8, file=file_sou_list, status='old', position='rewind')
!   read(8,*) num_src_model,nsubx,nsuby,dxsub,dysub,cxp,cyp
   read(8,*) num_src_model,nsubx,nsuby
   read(8,*) xref_src,yref_src,afa_src
!
! xref_src,yref_src are the (x, y) value of the given reference point
! in the coordinate system used for describing the point sources locations
! afa_src is the angle from North to the X-axis of this coordinate system
!
   do i=1,num_src_model
     read(8,'(1a)') source_name_list(i)
   enddo
   close(8)
!======================== END of INPUT ==============================
   intemp(1) = intprt
   intemp(2) = nproc_x
   intemp(3) = nproc_y
   intemp(4) = num_src_model
   intemp(5) = nd2_pml
   rltemp(1)=tdura
   rltemp(2)=xref_fdm
   rltemp(3)=yref_fdm
   rltemp(4)=angle_north_to_x
   rltemp(5)=xref_src
   rltemp(6)=yref_src
   rltemp(7)=afa_src
   rltemp(8)=RF
 else
   fileInf=' '
   fileRec=' '
   fileOut=' '
   fileMod=' '
   source_name_list = ' '
 endif
!
 ncount=70
 call MPI_BCAST (fileMod,ncount,MPI_CHARACTER,0,MPI_COMM_WORLD, ierr)
 ncount=70
 call MPI_BCAST (fileOut,ncount,MPI_CHARACTER,0,MPI_COMM_WORLD, ierr)
 ncount=5
 call MPI_BCAST (intemp, ncount,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
 ncount=8
 call MPI_BCAST (rltemp, ncount,MPI_REAL,0,MPI_COMM_WORLD,ierr)
 intprt        = intemp(1)
 nproc_x       = intemp(2)
 nproc_y       = intemp(3)
 num_src_model = intemp(4)
 nd2_pml       = intemp(5)
 tdura             =  rltemp(1)
 xref_fdm          =  rltemp(2)
 yref_fdm          =  rltemp(3)
 angle_north_to_x  =  rltemp(4)
 xref_src          =  rltemp(5)
 yref_src          =  rltemp(6)
 afa_src           =  rltemp(7)
 RF                =  rltemp(8)
!
 if(ierr/=0) return
 if(nproc_x*nproc_y > numprocs) then
   ierr=1
   if(myid_world ==0) then
     write(13,*) ' ' 
     write(13,*) 'ERROR !'
     write(13,*) 'Multiplying (the partition number in X-axis)',nproc_x
     write(13,*) 'by (the partition number in Y-axis)', &
                  nproc_y, '  (=',nproc_x*nproc_y,')'
     write(13,*) 'is greater than the number of processors', &
                  numprocs
   endif
 endif
 
end subroutine read_inp
!
!----------------------------------------------------------------------
subroutine input_material(fileMod,comm_worker,myid,nprx,npry, &
                          nd2_pml,RF0,dt,ierr)
!  get the earth model and index of material type for each cell
!  Note: The inputing velocity structure values are given at the point
!        where Txx, Tyy, and Tzz of each cell are located
!
 use mpi
 use boundMpi_comm
 use grid_node_comm
 use, intrinsic :: iso_c_binding, ONLY: c_f_pointer
 implicit NONE
 character(len=*), intent(IN):: fileMod
 integer, intent(IN):: comm_worker,myid,nprx,npry,nd2_pml
 integer, intent(OUT):: ierr
 real, intent(IN):: RF0
 real, intent(OUT):: dt
 integer:: i,j,k,k1,k2,ii,jj,nsum,ncount,mmat
! integer:: nxb0,nxe1,nyb0,nye1,nx_all,nxy_all,nwrt,nll
 integer:: nxb0,nxe1,nyb0,nye1,nx_all,nxy_all,nwrt
 integer, dimension(MPI_STATUS_SIZE):: status
 integer, allocatable, dimension(:):: node_all,idmat_tmp
 integer(kind=MPI_ADDRESS_KIND) size 
 integer :: real_itemsize, ierror
!
 RF=RF0
 mw2_pml=nd2_pml
 mw1_pml=mw2_pml*3
 mw1_pml1=mw1_pml+1
 mw2_pml1=mw2_pml+1
 np_x=nprx
 np_y=npry
 nprc1=np_x*np_y+1
 allocate(bound_x(nprc1),bound_y(nprc1))
 allocate(npat_x(nprc1), npat_y(nprc1))
 allocate(node_all(5))
!
 if(myid == 0) then
!  Open the materil file
   open(unit=15, file=fileMod, status='old', action='read', &
        form='unformatted', position='rewind')
   read(15) node_all
!              nx2_all,ny2_all,nz2,nz1,nwrt
  !debug--------------------------------------------------------
   write(13,*) 'nx2_all=',node_all(1),'ny2_all=',node_all(2), &
        'nz2=',node_all(3),'nz1=',node_all(4),'nwrt=',node_all(5)
   !------------------------------------------------------------
 endif
! 
 call MPI_BCAST (node_all,5,MPI_INTEGER,0,comm_worker,ierr)
 allocate(dx_all(node_all(1)),dy_all(node_all(2)))
 allocate(dz_all(node_all(3)+node_all(4)))
 nwrt=node_all(5)
!
 if(myid == 0) then
! Inputting Gridding Spaces
  read(15) dx_all
  read(15) dy_all
  read(15) dz_all
  !debug---------------------------------
  write(13,*), 'dx_all=', dx_all
  write(13,*), 'dy_all=', dy_all
  write(13,*), 'dz_all=', dz_all
 endif
!
 ncount=node_all(1)
 call MPI_BCAST (dx_all,ncount, MPI_REAL, 0,comm_worker,ierr)
 ncount=node_all(2)
 call MPI_BCAST (dy_all,ncount, MPI_REAL, 0,comm_worker,ierr)
 ncount=node_all(3)+node_all(4)
 call MPI_BCAST (dz_all,ncount, MPI_REAL, 0,comm_worker,ierr)
!
! Partition the FD modeling area among the processors
 call partition(myid,node_all,ierr)
! write(*,*) ' after call partition(myid,node_all,ierr)'

!
!  Inputing idmat -- the index of material type
 deallocate(node_all)
 allocate(node_all(nwrt))
! write(*,*) ' after deallocate(node_all) allocate(node_all(nwrt))'
 nxb0=npat_x(myid_x)*3
 nxe1=nxtop+2
 if(myid_x == np_x) nxe1=nxtop+1
 nyb0=npat_y(myid_y)*3
 nye1=nytop+2
 if(myid_y == np_y) nye1=nytop+1
 nx_all=3*nx2_all-2
 nxy_all=(3*nx2_all-2)*(3*ny2_all-2)
 nsum=0;  k1=1;  k2=0;   node_all=0
 do while(node_all(1) /= -999)
   if(myid==0) read(15) node_all
   call MPI_BCAST(node_all,nwrt,MPI_INTEGER,0,comm_worker,ierr)
   do ii=1, nwrt-1, 2
   do jj=1, node_all(ii)
     nsum=nsum+1
     if(nsum>nxy_all) then
       nsum=nsum-nxy_all
       k1=k1+1
       k2=k1-nztop
       if(k2 == 1) then
         nxb0=npat_x(myid_x)
         nxe1=nxbtm+2
         if(myid_x == np_x) nxe1=nxbtm+1
         nyb0=npat_y(myid_y)
         nye1=nybtm+2
         if(myid_y == np_y) nye1=nybtm+1
         nx_all =nx2_all
         nxy_all=nx2_all*ny2_all
       endif
     endif
     j=(nsum-1)/nx_all+1-nyb0
     i=nsum-(j-1+nyb0)*nx_all-nxb0
     if((j>0 .and. j<nye1) .and. (i>0 .and. i<nxe1)) then
       if(k2 < 1) then
         idmat1(k1,i,j)=node_all(ii+1)
       else
         idmat2(k2,i,j)=node_all(ii+1)
       endif
     endif
   enddo
   enddo
 enddo
!
!  Inputting the numbere of material, nll,  and time step
 if(myid==0) read(15) mmat, nll, dt
!
 call MPI_BCAST (mmat, 1, MPI_INTEGER, 0, comm_worker, ierr)
 call MPI_BCAST (nll,  1, MPI_INTEGER, 0, comm_worker, ierr)
 call MPI_BCAST (dt,   1, MPI_REAL,    0, comm_worker, ierr)
!
 deallocate(node_all)
 allocate(idmat_tmp(mmat))
 call mat_type_myid(mmat,idmat_tmp)
! write(*,*) 'before allocate(clamda(nmat),cmu(nmat),qwp(nmat),qws(nmat))'
! allocate(clamda(nmat),cmu(nmat),rho(nmat),qwp(nmat),qws(nmat))

! write(*,*) 'after allocate(clamda(nmat),cmu(nmat),qwp(nmat),qws(nmat))'
! call MPI_Type_extent(MPI_REAL, real_itemsize, ierror)
real_itemsize=4
 size=nmat*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rho, ierror)
! write(*,*) 'after MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rho, ierror)'
 call c_f_pointer(cptr_rho, rho, (/ nmat /))
! write(*,*) 'after  c_f_pointer(cptr_rho, rho, (/ nmat /))'

 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_clamda, ierror)
 call c_f_pointer(cptr_clamda, clamda, (/ nmat /))

 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_cmu, ierror)
 call c_f_pointer(cptr_cmu, cmu, (/ nmat /))

 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qwp, ierror)
 call c_f_pointer(cptr_qwp, qwp, (/ nmat /))

 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qws, ierror)
 call c_f_pointer(cptr_qws, qws, (/ nmat /))

 size = nll * real_itemsize
! allocate(epdt(nll), qwt1(nll), qwt2(nll))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_epdt, ierror)
 call c_f_pointer(cptr_epdt, epdt, (/ nll /))

 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qwt1, ierror)
 call c_f_pointer(cptr_qwt1, qwt1, (/ nll /))

 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qwt2, ierror)
 call c_f_pointer(cptr_qwt2, qwt2, (/ nll /))

!
! Inputing Vp, Vs, rho, and weight coefficients for Qp, and Qs
! Vp
 call read_mvl(mmat,nmat,comm_worker,myid,idmat_tmp,clamda)
! Vs
 call read_mvl(mmat,nmat,comm_worker,myid,idmat_tmp,cmu)
! rho
 call read_mvl(mmat,nmat,comm_worker,myid,idmat_tmp,rho)
! qwp
 call read_mvl(mmat,nmat,comm_worker,myid,idmat_tmp,qwp)
! qws
 call read_mvl(mmat,nmat,comm_worker,myid,idmat_tmp,qws)
!
! Inputing epdt,qwt1,qwt2
 if(myid==0) then
   read(15) epdt 
   read(15) qwt1 
   read(15) qwt2 
 endif
!
 ncount=nll
 call MPI_BCAST (epdt, ncount, MPI_REAL, 0, comm_worker, ierr)
 call MPI_BCAST (qwt1, ncount, MPI_REAL, 0, comm_worker, ierr)
 call MPI_BCAST (qwt2, ncount, MPI_REAL, 0, comm_worker, ierr)
! close the file of 15
 if(myid==0) close(15)
 deallocate(idmat_tmp)
!++++++++++++++++  End input of 3D velocity structer  +++++++++++++++++
!
 call consts
!
 call operator_diff(dt)
!
 if(myid==0) then
   write(13,*) ' '
   write(13,*) 'Type number of material=',mmat
   write(13,*) ' '
   write(13,*) 'Time step=',dt
   write(13,*) ' '
   write(13,*) '    PID, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm'
   do j=1,np_y
   do i=1,np_x
     k=(j-1)*np_x+i
     nxe1=npat_x(i+1)-npat_x(i)
     nxb0=3*nxe1
     if(i==np_x) nxb0=3*nxe1-2
     nye1=npat_y(j+1)-npat_y(j)
     nyb0=3*nye1
     if(j==np_y) nyb0=3*nye1-2
     write(13,'(7i7)') k,nxb0,nyb0,nztop,nxe1,nye1,nzbtm
   enddo
   enddo
 endif
!
 call MPI_Barrier(comm_worker, ierr)

end subroutine input_material
!
!-----------------------------------------------------------------------
subroutine partition(myid,node_all,ierr)
 use grid_node_comm
 use boundMpi_comm
 use plane_wave_bd
 implicit NONE
 integer, intent(IN):: myid
 integer, dimension(5), intent(IN):: node_all
 integer, intent(OUT):: ierr
 integer:: i,k,nn1,nn2,nb
 real:: delt,bound_z
! X direcction
 npat_x=0
 if(np_x>2) then
   nb=max0(node_all(1)/np_x*9/10, mw2_pml+2)
   do k=1,node_all(1)-2*nb
     i=mod(k,np_x-2)+3
     npat_x(i)=npat_x(i)+1
   enddo
   npat_x(2)=nb
   npat_x(np_x+1)=nb
 else
   npat_x(2)=node_all(1)/np_x
   if(np_x==2) npat_x(3)=node_all(1)-npat_x(2)
 endif
! Y direcction
 npat_y=0
 if(np_y>2) then
   nb=max0(node_all(2)/np_y*9/10, mw2_pml+2)
   do k=1,node_all(2)-2*nb
     i=mod(k,np_y-2)+3
     npat_y(i)=npat_y(i)+1
   enddo
   npat_y(2)=nb
   npat_y(np_y+1)=nb
 else
   npat_y(2)=node_all(2)/np_y
   if(np_y==2) npat_y(3)=node_all(2)-npat_y(2)
 endif
! index of nodes
 do i=2,np_x+1
   npat_x(i)=npat_x(i)+npat_x(i-1)
 enddo
 do i=2,np_y+1
   npat_y(i)=npat_y(i)+npat_y(i-1)
 enddo
! Determining the id-number of processor around myid
 myid_y=myid/np_x+1
 myid_x=myid-(myid_y-1)*np_x+1
 neighb=-1
 if(myid_x.ne.1   ) neighb(1) = myid-1
 if(myid_x.ne.np_x) neighb(2) = myid+1
 if(myid_y.ne.1   ) neighb(3) = myid-np_x
 if(myid_y.ne.np_y) neighb(4) = myid+np_x
! Determining the node numbers and grid spaces for myid
 nxbtm=npat_x(myid_x+1)-npat_x(myid_x)
 nybtm=npat_y(myid_y+1)-npat_y(myid_y)
 nzbtm=node_all(3)
 nztop=node_all(4)
 nztol=nzbtm+nztop
 nxtop=nxbtm*3
 if(myid_x == np_x) nxtop=nxbtm*3-2
 nytop=nybtm*3
 if(myid_y == np_y) nytop=nybtm*3-2
 nx2_all=node_all(1)
 ny2_all=node_all(2)
 nxb2=npat_x(myid_x)
 nyb2=npat_y(myid_y)
 nxb1=3*nxb2
 nyb1=3*nyb2
!
 call set_node_grid
! write (*,*) 'before call  alloc_memory(myid)'
 call  alloc_memory(myid)
! write (*,*) 'after call  alloc_memory(myid)'
! Set grid space for each processor
 do i=-1,nxbtm+4
   k=min0(max0(i+nxb2,1),nx2_all)
   gridx2(i)=dx_all(k)
 enddo 
 do i=-1,nybtm+4
   k=min0(max0(i+nyb2,1),ny2_all)
   gridy2(i)=dy_all(k)
 enddo  
 do i=-1,nztol+4
   gridz(i)=dz_all(min0(max0(i,1),nztol))
 enddo  
 do i=-1,nxtop+4
   gridx1(i)=gridx2( (i+2)/3 )/3.
 enddo
 do i=-1,nytop+4
   gridy1(i)=gridy2( (i+2)/3 )/3.
 enddo
! Determining the bound of the zone for each process
 bound_x=0.0
 do i=2,np_x+1
   k=npat_x(i)
   bound_x(i)=sum(dx_all(1:k))
 enddo
 bound_y=0.0
 do i=2,np_y+1
   k=npat_y(i)
   bound_y(i)=sum(dy_all(1:k))
 enddo
! set the taper range for inputting plane wave
 nn1=3
 nn2=nn1+10
 delt=dx_all(1)
 xplwv(1)=bound_x(myid_x)
 xplwv(2)=nn1*delt
 xplwv(3)=nn2*delt
 xplwv(4)=bound_x(np_x+1)-xplwv(3)
 xplwv(5)=bound_x(np_x+1)-xplwv(2)
 yplwv(1)=bound_y(myid_y)
 yplwv(2)=nn1*delt
 yplwv(3)=nn2*delt
 yplwv(4)=bound_y(np_y+1)-yplwv(3)
 yplwv(5)=bound_y(np_y+1)-yplwv(2)
 bound_z=sum(dz_all(1:nztol))
 zplwv(1)=bound_z-nn2*delt
 zplwv(2)=bound_z-nn1*delt
 return
end subroutine partition
!
!-----------------------------------------------------------------------
subroutine mat_type_myid(mmat,idmat_tmp)
 use boundMpi_comm
 use grid_node_comm
 implicit NONE
 integer, intent(IN):: mmat
 integer, dimension(mmat), intent(OUT):: idmat_tmp 
 integer:: i,j,k,ii,jj,nx00,ny00,nz00
!
 do j=1,nytop+1
 do i=1,nxtop+1
   idmat1(0,i,j)=idmat1(1,i,j)
   jj=(j-1)/mrgsp+1
   ii=(i-1)/mrgsp+1
   idmat1(nztop+1,i,j)=idmat2(1,ii,jj)
 enddo
 enddo
 do  jj=1,nybtm+1
 do  ii=1,nxbtm+1
   j=min0((jj-1)*mrgsp+1,nytop+1)
   i=min0((ii-1)*mrgsp+1,nxtop+1)
   idmat2(0,ii,jj)=idmat1(nztop,i,j)
 enddo
 enddo
 if(myid_x==1) then
   do j=1,nytop+1
   do i=1,mw1_pml
   do k=0,nztop+1
    idmat1(k,i,j)=idmat1(k,mw1_pml1,j)
   enddo
   enddo
   enddo
   do j=1,nybtm+1
   do i=1,mw2_pml
   do k=0,nzbtm
    idmat2(k,i,j)=idmat2(k,mw2_pml1,j)
   enddo
   enddo
   enddo
 endif
 if(myid_x==np_x) then
   nx00=nxtop-mw1_pml
   do j=1,nytop+1
   do i=nx00+1,nxtop
   do k=0,nztop+1
    idmat1(k,i,j)=idmat1(k,nx00,j)
   enddo
   enddo
   enddo
   nx00=nxbtm-mw2_pml
   do j=1,nybtm+1
   do i=nx00+1,nxbtm
   do k=0,nzbtm
    idmat2(k,i,j)=idmat2(k,nx00,j)
   enddo
   enddo
   enddo
 endif
 if(myid_y==1) then
   do j=1,mw1_pml
   do i=1,nxtop+1
   do k=0,nztop+1
    idmat1(k,i,j)=idmat1(k,i,mw1_pml1)
   enddo
   enddo
   enddo
   do j=1,mw2_pml
   do i=1,nxbtm
   do k=0,nzbtm
    idmat2(k,i,j)=idmat2(k,i,mw2_pml1)
   enddo
   enddo
   enddo
 endif
 if(myid_y==np_y) then
   ny00=nytop-mw1_pml
   do j=ny00+1,nytop
   do i=1,nxtop+1
   do k=0,nztop+1
    idmat1(k,i,j)=idmat1(k,i,ny00)
   enddo
   enddo
   enddo
   ny00=nybtm-mw2_pml
   do j=ny00+1,nybtm
   do i=1,nxbtm+1
   do k=0,nzbtm
    idmat2(k,i,j)=idmat2(k,i,ny00)
   enddo
   enddo
   enddo
 endif
 nz00=nzbtm-mw2_pml
 do j=1,nybtm+1
 do i=1,nxbtm+1
 do k=nz00+1,nzbtm
    idmat2(k,i,j)=idmat2(nz00,i,j)
 enddo
 enddo
 enddo
!
 idmat_tmp=0
 nmat=0
 do j=1,nytop+1
 do i=1,nxtop+1
 do k=0,nztop+1
   ii=idmat1(k,i,j)
   if(ii <=0 ) cycle
   if(idmat_tmp(ii) ==0) then
     nmat=nmat+1
     idmat_tmp(ii)=nmat
   endif
   idmat1(k,i,j) = idmat_tmp(ii)
 enddo
 enddo
 enddo
 do j=1,nybtm+1
 do i=1,nxbtm+1
 do k=0,nzbtm
   ii=idmat2(k,i,j)
   if(ii <=0 ) cycle
   if(idmat_tmp(ii) == 0) then
     nmat=nmat+1
     idmat_tmp(ii)=nmat
   endif
   idmat2(k,i,j) = idmat_tmp(ii)
 enddo
 enddo
 enddo
 return
end subroutine mat_type_myid
!
!-----------------------------------------------------------------------
subroutine read_mvl(mmat,nmat,comm_worker,myid,idmat_tmp,cmat)
 use mpi
 implicit NONE
 integer, intent(IN):: mmat,nmat,comm_worker,myid
 integer, dimension(mmat), intent(In):: idmat_tmp 
 real, dimension(nmat), intent(OUT):: cmat 
 real, dimension(mmat):: cmat_tmp
 integer:: k,k2,ierr

 if(myid==0)  read(15) cmat_tmp
!
 call MPI_BCAST (cmat_tmp, mmat, MPI_REAL, 0, comm_worker, ierr)
 do k=1,mmat
   k2=idmat_tmp(k)
   if(k2 == 0) cycle
   cmat(k2)=cmat_tmp(k)
 enddo 
 call MPI_Barrier(comm_worker, ierr)

end subroutine read_mvl
!
!-----------------------------------------------------------------------
subroutine input_Receiver_location(fileRec,comm_worker,myid,nr_all, &
                        nrs,xref_fdm,yref_fdm,angle_north_to_x,ierr0)
! read  receiver parameters 
! ix=jy=kz=1 is the original point of Cartesian coordinates
! [z,x,y]=[ (kz-1)*gridsp, (ix-1)*gridsp, (jy-1)*gridsp ]
 use mpi
 use station_comm
 use boundMpi_comm
! use itface_comm
 implicit NONE
 integer, intent(IN):: comm_worker,myid 
 integer, intent(OUT):: nr_all,nrs,ierr0
 real, intent(IN):: xref_fdm,yref_fdm,angle_north_to_x
 character(len=*), intent(IN):: fileRec
 character(len=72):: stat_name
 integer:: j,k,ncount,ierr
 real:: dr,xcr,ycr,zcr,afa_stn,xref_stn,yref_stn,csaf,snaf
 character(len=72), allocatable, dimension(:):: rname_tmp
 real, allocatable, dimension(:,:):: recev_tmp
!
 if(myid == 0) then
   open(unit=19, file=fileRec,status='old', position='rewind')
   read(19,*) nr_all,xref_stn,yref_stn,afa_stn,recv_type
 endif
 call MPI_BCAST (nr_all,   1,MPI_INTEGER,0,comm_worker, ierr)
 call MPI_BCAST (recv_type,1,MPI_INTEGER,0,comm_worker, ierr)
!
 if(recv_type == 2 ) then
!
! Input the bundaries of each block to output wavefield 
   nblock=2*nr_all
   allocate(ipox(3,nblock),ipoy(3,nblock),ipoz(3,nblock))
   ipox=0;   ipoy=0;  ipoz=0
   if(myid ==0) then
     allocate(recev_tmp(9,nr_all))
     do j=1,nr_all
       read(19,'(1a)') stat_name
! X-direction rx_start, rx_end, drx
       read(19,*) (recev_tmp(k,j),k=1,3)
! Y-direction ry_start, ry_end, dry
       read(19,*) (recev_tmp(k,j),k=4,6)
! Z-direction rz_start, rz_end, drz
       read(19,*) (recev_tmp(k,j),k=7,9)
     enddo
     call Type2_Recv_int(nr_all,recev_tmp)
     deallocate(recev_tmp)
   endif
   ncount=nblock*3
   call MPI_BCAST(ipox,ncount,MPI_INTEGER,0,comm_worker, ierr)
   call MPI_BCAST(ipoy,ncount,MPI_INTEGER,0,comm_worker, ierr)
   call MPI_BCAST(ipoz,ncount,MPI_INTEGER,0,comm_worker, ierr)
   call Assign_Type2_Rev_2_Proc(nrs)
   return   
 endif
!
! Input x, y, z (coordinates of receiver, z positive downward) 
 allocate(recev_tmp(4,nr_all), rname_tmp(nr_all))
 ierr0=0
 if(myid ==0) then
   zcr=atan(1.0)/45.0
   csaf=cos((afa_stn-angle_north_to_x)*zcr)
   snaf=sin((afa_stn-angle_north_to_x)*zcr)
   do j=1,nr_all
     read(19,'(1a)') rname_tmp(j)
     read(19,*) (recev_tmp(k,j),k=1,3)
     xcr=recev_tmp(1,j)-xref_stn
     ycr=recev_tmp(2,j)-yref_stn
     recev_tmp(1,j)=xcr*csaf-ycr*snaf+xref_fdm
     recev_tmp(2,j)=xcr*snaf+ycr*csaf+yref_fdm
     if(recev_tmp(1,j) <= bound_x(1) .or.    & 
        recev_tmp(1,j) >= bound_x(np_x+1) .or. &
        recev_tmp(2,j) <= bound_y(1) .or.    &
        recev_tmp(2,j) >= bound_y(np_y+1) ) then
        write(13,*) 'Receiver out of the bounds of FD:', j
        ierr0=1
      endif
   enddo
   close(19)
 endif
 call MPI_BCAST (ierr0,1,MPI_INTEGER,0, comm_worker, ierr)
 if(ierr0 /=0) return
 do j=1,nr_all
   stat_name=rname_tmp(j)
   call MPI_BCAST (stat_name,71,MPI_CHARACTER,0, comm_worker, ierr)
   rname_tmp(j)=stat_name(1:70)
 enddo
 if(myid==0) call Assign_Rev_2_Proc(nr_all,recev_tmp)
 ncount=nr_all*4
 call MPI_BCAST (recev_tmp,ncount,MPI_REAL,0,comm_worker,ierr)
!
 nrecs=0
 do j=1,nr_all
   if(int(recev_tmp(4,j)) == myid) nrecs=nrecs+1
 enddo
 if(nrecs > 0) then
   allocate(fname_stn(nrecs)) 
   allocate(ipox(2,nrecs),ipoy(2,nrecs),ipoz(2,nrecs))
   allocate(xcor(2,nrecs),ycor(2,nrecs),zcor(2,nrecs))
 endif
 nrs=0
 do j=1,nr_all
   if(int(recev_tmp(4,j)) == myid) then
     stat_name=rname_tmp(j)
     nrs=nrs+1
     xcr=recev_tmp(1,j)-bound_x(myid_x)
     ycr=recev_tmp(2,j)-bound_y(myid_y)
     zcr=recev_tmp(3,j)
     call set_receiver(stat_name, nrs, xcr, ycr, zcr)
   endif
 enddo
 deallocate(recev_tmp,rname_tmp)
 return
end subroutine input_Receiver_location
!
!-----------------------------------------------------------------------
subroutine Assign_Rev_2_Proc(nrcv,recev_tmp)
 use boundMpi_comm
 use grid_node_comm
 implicit NONE
 integer, intent(IN):: nrcv
 real, dimension(4,nrcv), intent(INOUT):: recev_tmp
 integer:: k,i1,i2,ii,j1,j2,jj
 real:: xs,ys
!
 do k=1,nrcv
   xs=recev_tmp(1,k)
   ys=recev_tmp(2,k)
   i1=1
   i2=np_x+1
   do 
     ii=(i1+i2)/2
     if(xs > bound_x(ii) .and. xs <= bound_x(ii+1)) exit
     if(xs > bound_x(ii) ) then
       i1=ii
     else
       i2=ii
     endif
   enddo
!
   j1=1
   j2=np_y+1
   do 
     jj=(j1+j2)/2
     if(ys > bound_y(jj) .and. ys <= bound_y(jj+1)) exit
     if(ys >  bound_y(jj) ) then
       j1=jj
     else
       j2=jj
     endif
   enddo
   recev_tmp(4,k) = float((jj-1)*np_x+ii-1)+0.1
 enddo
 return
end subroutine Assign_Rev_2_Proc


!
!-----------------------------------------------------------------------
subroutine Type2_Recv_Int(nrcv,recev_tmp)
 use station_comm
 use boundMpi_comm
 use grid_node_comm
 implicit NONE
 integer, intent(IN):: nrcv
 real, dimension(9,nrcv), intent(INOUT):: recev_tmp
 integer:: kc,i1,i2,i3,ni,nx1b,ny1b
 real:: xs,dx1,dy1,dx2,dy2,dz1,dz2,zbtm0
!
 nx1b=nx2_all*3-2
 ny1b=ny2_all*3-2
 dx2=maxval( dx_all(1: nx2_all) )
 dy2=maxval( dy_all(1: ny2_all) )
 dz1=maxval( dz_all(1: nztop) )
 dz2=maxval( dz_all(nztop+1: nztol) )
 zbtm0=sum(dz_all(1:nztop))
 dx1=dx2/3.0
 dy1=dy2/3.0
!
 do kc=1,nrcv
! Region I
   if( recev_tmp(7,kc) < zbtm0-0.49*dz1 ) then
!--x
     i3=max0(1, int(recev_tmp(3,kc)/dx1+0.5))
     i1=1;  i2=1;  xs=0.0
     do
       xs=xs+dx_all((i2-1)/3+1)/3.0
       if(xs >  recev_tmp(2,kc) .or. i2==nx1b) exit
       if(xs <= recev_tmp(1,kc)) i1=i1+1
       i2=i2+1
     enddo
     ni=(i2-i1)/i3+1
     if(ni*i3+i1>=nx1b .or. (ni-1)*i3+i1==i2) ni=ni-1
     ipox(1, kc*2-1)=i1
     ipox(2, kc*2-1)=ni*i3+i1
     ipox(3, kc*2-1)=i3
!--y
     i3=max0(1, int(recev_tmp(6,kc)/dy1+0.5))
     i1=1;  i2=1;  xs=0.0
     do
       xs=xs+dy_all((i2-1)/3+1)/3.0
       if(xs >  recev_tmp(5,kc) .or. i2==ny1b) exit
       if(xs <= recev_tmp(4,kc)) i1=i1+1
       i2=i2+1
     enddo
     ni=(i2-i1)/i3+1
     if(ni*i3+i1>=ny1b .or. (ni-1)*i3+i1==i2) ni=ni-1
     ipoy(1, kc*2-1)=i1
     ipoy(2, kc*2-1)=ni*i3+i1
     ipoy(3, kc*2-1)=i3
!--z
     i3=max0(1, int(recev_tmp(9,kc)/dz1+0.5))
     i1=1;  i2=1;  xs=0.0
     do
       xs=xs+dz_all(i2)
       if(xs >  recev_tmp(8,kc) .or. i2==nztop) exit
       if(xs <= recev_tmp(7,kc)) i1=i1+1
       i2=i2+1
     enddo
     ni=(i2-i1)/i3+1
     if(ni*i3+i1>nztop .or. (ni-1)*i3+i1==i2) ni=ni-1
     ipoz(1, kc*2-1)=i1
     ipoz(2, kc*2-1)=ni*i3+i1
     ipoz(3, kc*2-1)=i3
!
   endif
! Region II
   if( recev_tmp(8,kc) > zbtm0-0.51*dz1 ) then
!--x
     i3=max0(1, int(recev_tmp(3,kc)/dx2+0.5))
     i1=1;  i2=1;  xs=0.0
     do
       xs=xs+dx_all(i2)
       if(xs >  recev_tmp(2,kc) .or. i2==nx2_all) exit
       if(xs <= recev_tmp(1,kc)) i1=i1+1
       i2=i2+1
     enddo
     ni=(i2-i1)/i3+1
     if(ni*i3+i1>=nx2_all .or. (ni-1)*i3+i1==i2) ni=ni-1
     ipox(1, kc*2)=i1
     ipox(2, kc*2)=ni*i3+i1
     ipox(3, kc*2)=i3
!--y
     i3=max0(1, int(recev_tmp(6,kc)/dy2+0.5))
     i1=1;  i2=1;  xs=0.0
     do
       xs=xs+dy_all(i2)
       if(xs >  recev_tmp(5,kc) .or. i2==ny2_all) exit
       if(xs <= recev_tmp(4,kc)) i1=i1+1
       i2=i2+1
     enddo
     ni=(i2-i1)/i3+1
     if(ni*i3+i1>=ny2_all .or. (ni-1)*i3+i1==i2) ni=ni-1
     ipoy(1, kc*2)=i1
     ipoy(2, kc*2)=ni*i3+i1
     ipoy(3, kc*2)=i3
!--z
     i3=max0(1, int(recev_tmp(9,kc)/dz2+0.5))
     i1=1;  i2=1;  xs=zbtm0
     print *, 'the numbers are'
     print *,dz_all
     print *, 'xs is'
     print *,xs
     print *, 'i2 is'
     print *,i2
     print *, 'nztop is'
     print *,nztop
     do
       xs=xs+dz_all(i2+nztop)
       if(xs >  recev_tmp(8,kc) .or. i2==nzbtm-10) exit
       if(xs <= recev_tmp(7,kc)) i1=i1+1
       i2=i2+1
     enddo
     ni=(i2-i1)/i3+1
     if(ni*i3+i1>=nzbtm .or. (ni-1)*i3+i1==i2) ni=ni-1
     ipoz(1, kc*2)=i1
     ipoz(2, kc*2)=ni*i3+i1
     ipoz(3, kc*2)=i3
   endif
 enddo
 
end subroutine Type2_Recv_Int
!
!-----------------------------------------------------------------------
subroutine Assign_Type2_Rev_2_Proc(nrs)
 use station_comm
 use boundMpi_comm
 use grid_node_comm
 implicit NONE
 integer, intent(OUT):: nrs
 integer:: kc,ni,nxx,nyy,nzz
!
 allocate(nrxyz(6,nblock))
 nrxyz=0;   nrecs=0
 do kc=1,nblock
   if(mod(kc, 2) == 1) then
     ipox(1:2, kc)=ipox(1:2, kc)-npat_x(myid_x)*3
     ipoy(1:2, kc)=ipoy(1:2, kc)-npat_y(myid_y)*3
     nxx=nxtop;  nyy=nytop;  nzz=nztop
   else
     ipox(1:2, kc)=ipox(1:2, kc)-npat_x(myid_x)
     ipoy(1:2, kc)=ipoy(1:2, kc)-npat_y(myid_y)
     nxx=nxbtm;  nyy=nybtm;  nzz=nzbtm
   endif
   if( ipox(1,kc)<=nxx .and. ipox(2,kc)>0 .and. &
       ipoy(1,kc)<=nyy .and. ipoy(2,kc)>0 .and. ipoz(2,kc)>0 ) then
      if(ipox(1,kc)<1) nrxyz(1,kc)=(-ipox(1,kc))/ipox(3,kc)+1
      ipox(1,kc)=ipox(1,kc)+ipox(3,kc)*nrxyz(1,kc)
      ni=0
      if(ipox(2,kc)>nxx) ni=(ipox(2,kc)-nxx-1)/ipox(3,kc)+1
      ipox(2,kc)=ipox(2,kc)-ipox(3,kc)*ni
      nrxyz(2,kc)=(ipox(2,kc)-ipox(1,kc))/ipox(3,kc)+1

      if(ipoy(1,kc)<1) nrxyz(3,kc)=(-ipoy(1,kc))/ipoy(3,kc)+1
      ipoy(1,kc)=ipoy(1,kc)+ipoy(3,kc)*nrxyz(3,kc)
      ni=0
      if(ipoy(2,kc)>nyy) ni=(ipoy(2,kc)-nyy-1)/ipoy(3,kc)+1
      ipoy(2,kc)=ipoy(2,kc)-ipoy(3,kc)*ni
      nrxyz(4,kc)=(ipoy(2,kc)-ipoy(1,kc))/ipoy(3,kc)+1
      nrxyz(6,kc)=(ipoz(2,kc)-ipoz(1,kc))/ipoz(3,kc)+1
      nrecs=nrecs+nrxyz(2,kc)*nrxyz(4,kc)*nrxyz(6,kc)
   else
      ipox(:, kc)=1;   ipox(2, kc)=-1
      ipoy(:, kc)=1;   ipoy(2, kc)=-1
      ipoz(:, kc)=1;   ipoz(2, kc)=-1
   endif
 enddo
 nrs=nrecs
 
end subroutine Assign_Type2_Rev_2_Proc
!
!-----------------------------------------------------------------------
subroutine Type2_Rev_Location(nrc3,recv_xyz)
 use station_comm
 use boundMpi_comm
 use grid_node_comm
 implicit NONE
 integer, intent(IN):: nrc3
 real, dimension(nrc3), intent(OUT):: recv_xyz
 integer:: kc,k2,i,j,k,ir,nx2b,ny2b
!
 allocate(xcor(nxtop,2), ycor(nytop,2), zcor(nztol,1))
 nx2b=npat_x(myid_x)
 ny2b=npat_y(myid_y)
 xcor(1,1)=bound_x(myid_x)
 do i=2,nxtop
   xcor(i, 1)=xcor(i-1, 1) + dx_all((i-2)/3+1+nx2b)/3.0
 enddo
 do i=1,nxbtm
   xcor(i, 2)=xcor(i*3-2, 1)
 enddo
!
 ycor(1,1)=bound_y(myid_y)
 do j=2,nytop
   ycor(j,1)=ycor(j-1,1)+dy_all((j-2)/3+1+ny2b)/3.0
 enddo
 do j=1,nybtm
   ycor(j, 2)=ycor(3*j-2, 1)
 enddo
!
 zcor(1,1)=0.0
 do k=2,nztol
   zcor(k,1)=zcor(k-1,1)+dz_all(k-1)
 enddo
!
 ir=0
 do kc=1,nblock-1,2
   if(ipoz(2,kc)>0 .and. ipoy(2,kc)>0 .and. ipox(2,kc)>0 )then
   do k=ipoz(1, kc), ipoz(2, kc), ipoz(3,kc) 
   do j=ipoy(1, kc), ipoy(2, kc), ipoy(3,kc) 
   do i=ipox(1, kc), ipox(2, kc), ipox(3,kc) 
     ir=ir+3
     recv_xyz(ir-2)=xcor(i,1) 
     recv_xyz(ir-1)=ycor(j,1) 
     recv_xyz(ir  )=zcor(k,1)
   enddo
   enddo
   enddo
   endif
   k2=kc+1
   if(ipoz(2,k2)>0 .and. ipoy(2,k2)>0 .and. ipox(2,k2)>0 )then
   do k=nztop+ipoz(1, k2), nztop+ipoz(2, k2), ipoz(3,k2) 
   do j=ipoy(1, k2), ipoy(2, k2), ipoy(3,k2) 
   do i=ipox(1, k2), ipox(2, k2), ipox(3,k2) 
     ir=ir+3
     recv_xyz(ir-2)=xcor(i,2) 
     recv_xyz(ir-1)=ycor(j,2) 
     recv_xyz(ir  )=zcor(k,1)
   enddo
   enddo
   enddo
   endif
 enddo
 deallocate(xcor, ycor, zcor)
 
end subroutine Type2_Rev_Location
!
!-----------------------------------------------------------------------
subroutine input_source_param(fileSou,comm_worker,myid,dt,xref_fdm, &
                              yref_fdm,angle_north_to_x,xref_src, &
                              yref_src,afa_src,is_moment,ierr0)
! read and set the source parameters of double couple force
 use mpi
 use boundMpi_comm
 use grid_node_comm
 use source_parm_comm
 implicit NONE
 character(len=*), intent(IN):: fileSou
 integer, intent(IN):: comm_worker,myid
 integer, intent(OUT):: is_moment,ierr0
 real, intent(IN):: dt,xref_fdm,yref_fdm,angle_north_to_x, &
                    xref_src,yref_src,afa_src
 integer, parameter::  maxSou=3000
 integer:: nsub_fault,ndim,ncount,ierr,nseg,id_pvh,i,j,k
 integer:: status(MPI_STATUS_SIZE),node_all(3)
 integer, allocatable, dimension(:,:,:):: itmp
 real:: xb1,xb2,yb1,yb2,xs,ys,csaf,snaf
 real(kind=8)::  pi
 real:: sourc_tmp(10,maxSou), sp(10)
 real, allocatable, dimension(:,:):: rtmp
!
 pi=4.d0*datan(1.d0)
 csaf=cos((afa_src-angle_north_to_x)*sngl(pi)/180.0)
 snaf=sin((afa_src-angle_north_to_x)*sngl(pi)/180.0)
 xb1=bound_x(myid_x)-(gridx2(1)-gridx2(0))
 xb2=bound_x(myid_x+1)+(gridx2(nxbtm+2)-gridx2(nxbtm+1))
 yb1=bound_y(myid_y)-(gridy2(1)-gridy2(0))
 yb2=bound_y(myid_y+1)+(gridy2(nybtm+2)-gridy2(nybtm+1))
!
 if(myid==0) then
   open(unit=19, file=fileSou,status='old', position='rewind')
   read(19,*) is_moment,nsub_fault,id_sf_type
   node_all(1) = is_moment
   node_all(2) = nsub_fault
   node_all(3) = id_sf_type
 endif
 call MPI_BCAST (node_all,3,MPI_INTEGER,0, comm_worker, ierr)
 ierr0=0
 is_moment =node_all(1)
 nsub_fault=node_all(2)
 id_sf_type=node_all(3)
!
 nfadd=0
 ndim= min0(maxSou,72*nsub_fault)
 if(is_moment == 9) ndim=nsub_fault
 if(allocated(famp)) then
   deallocate(index_xyz_source, famp, ruptm, riset, sparam2)
 endif
 allocate(index_xyz_source(3,ndim,4))
 allocate(famp(ndim,6), ruptm(ndim), riset(ndim), sparam2(ndim))
!
 nseg = (nsub_fault-1)/maxSou+1
 do i=1,nseg
   ncount=maxSou
   if(i==nseg) ncount= nsub_fault-(nseg-1)*maxSou
   if(myid==0) then
     do j=1,ncount
       read(19,*) (sourc_tmp(k,j), k=1,10)
       xs=sourc_tmp(1,j)-xref_src
       ys=sourc_tmp(2,j)-yref_src
       sourc_tmp(1,j)=xs*csaf-ys*snaf+xref_fdm
       sourc_tmp(2,j)=xs*snaf+ys*csaf+yref_fdm
     enddo
   endif
! >>>>>>>>>>>>>>  read in all the point source parameters  <<<<<<<<<<<<<
! NOTE:  sourc_tmp(k,j), for k= 
!  1-3,  are xs,ys,zs (the coornidates of each point source)
!  4  ,  is the moment of each point source, if is_moment=1;     
!        is the slip-amplitute * area-of-subfault, if is_moment=2;
!  5  ,  is rupture starting time (second)  of this point source;
!  6  ,  is rise-time (second) of this point source;
!  7  ,  is another parameter for describing the source function;
!  8-10, are the angles (degree) of faulting strike, dip, and rake.
!^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   ncount=10*ncount   
   call MPI_BCAST (sourc_tmp,ncount,MPI_REAL,0, comm_worker, ierr)
   ncount=ncount/10   
   j=1
   do j=1,ncount
     sp=sourc_tmp(:,j)
     xs=sp(1)
     ys=sp(2)
     sp(1)=sp(1)-bound_x(myid_x)
     sp(2)=sp(2)-bound_y(myid_y)
     sp(8)=sp(8)-angle_north_to_x
     if(is_moment == 9) then
! id_pvh = 1, P-wave; =2, SV-wave; =3, SH-wave
       nfadd = 0
       id_pvh=j
       famp(id_pvh,1:4)=sp(1:4)
       famp(id_pvh,5:6)=sp(8:9)
       riset(id_pvh)   =sp(6)
     else if(xs>xb1 .and. xs<xb2 .and. ys>yb1 .and. ys<yb2) then
       if( (ndim-nfadd)<72 ) then   
         allocate(itmp(3,nfadd,4), rtmp(nfadd,9))
         itmp=index_xyz_source(1:3,1:nfadd,1:4)
         rtmp(:,1:6)= famp(1:nfadd,1:6)
         rtmp(:,7)  =ruptm(1:nfadd)
         rtmp(:,8)  =riset(1:nfadd)
         rtmp(:,9)=sparam2(1:nfadd)
         ndim= min0(nfadd+maxSou,72*nsub_fault)
         deallocate(index_xyz_source, famp, ruptm, riset, sparam2)
         allocate(index_xyz_source(3,ndim,4))
         allocate(famp(ndim,6), ruptm(ndim), riset(ndim), sparam2(ndim))
         index_xyz_source(1:3,1:nfadd,1:4) = itmp
         famp(1:nfadd,1:6)=rtmp(:,1:6)
         ruptm(1:nfadd)   =rtmp(:,7) 
         riset(1:nfadd)   =rtmp(:,8) 
         sparam2(1:nfadd) =rtmp(:,9)
         deallocate(itmp, rtmp)
       endif
       call source_coef(is_moment,dt,pi,sp,ierr0)
     endif
   enddo
 enddo
 if(myid==0) close(19)
 if(is_moment /= 9) then
   if(nfadd == 0) then
     deallocate(index_xyz_source, famp, ruptm, riset, sparam2)
   else
     allocate(itmp(3,nfadd,4), rtmp(nfadd,9))
     itmp=index_xyz_source(1:3,1:nfadd,1:4)
     rtmp(:,1:6)= famp(1:nfadd,1:6)
     rtmp(:,7)  =ruptm(1:nfadd)
     rtmp(:,8)  =riset(1:nfadd)
     rtmp(:,9)=sparam2(1:nfadd)
     ndim= nfadd
     deallocate(index_xyz_source, famp, ruptm, riset, sparam2)
     allocate(index_xyz_source(3,nfadd,4))
     allocate(famp(nfadd,6), ruptm(nfadd), riset(nfadd), sparam2(nfadd))
     index_xyz_source = itmp
     famp   =rtmp(:,1:6)
     ruptm  =rtmp(:,7) 
     riset  =rtmp(:,8) 
     sparam2=rtmp(:,9)
     deallocate(itmp, rtmp)
   endif
 endif
 call MPI_BCAST (ierr0,1,MPI_INTEGER,0,comm_worker,ierr)
 return
end subroutine input_source_param
!
!-----------------------------------------------------------------------
subroutine source_coef(is_moment,dt,pi,sp,ierr)
! set the source parameters of double couple force
 use grid_node_comm
 use source_parm_comm
 implicit NONE
 integer, intent(IN)::is_moment
 integer, intent(out)::ierr
 real, intent(IN):: dt
 real(kind=8), intent(IN):: pi
 real, dimension(10), intent(IN):: sp
 integer:: ix,jy,kz,ii,ijk,kzb,isu,nsou
 integer, dimension(72,4):: ix2,jy2,kz2
 real :: arsp,cof,rmoment,volume
 real(kind=8):: st,dp,rk,csst,snst,csdp,sndp,csrk,snrk
 real:: vslip(3),vnorm(3),tensor(6),cf8(72,4)
!
 arsp=sp(4)
 st=dble(sp(8) )*pi/180.0d0
 dp=dble(sp(9) )*pi/180.0d0
 rk=dble(sp(10))*pi/180.0d0
 csst=dcos(st)
 snst=dsin(st)
 csdp=dcos(dp)
 sndp=dsin(dp)
 csrk=dcos(rk)
 snrk=dsin(rk) 
 vslip(1)=csrk*csst+csdp*snrk*snst
 vslip(2)=csrk*snst-csdp*snrk*csst
 vslip(3)=         -sndp*snrk
 vnorm(1)=-sndp*snst
 vnorm(2)= sndp*csst
 vnorm(3)=-csdp
!
!    Mpq=M0*(vnorm(p)*vslip(q)+vnorm(q)*vslip(p))
!
 if(is_moment==1) rmoment=arsp
 tensor(1)=2.*vnorm(1)*vslip(1)
 tensor(2)=2.*vnorm(2)*vslip(2)
 tensor(3)=2.*vnorm(3)*vslip(3)
 tensor(4)=(vnorm(1)*vslip(2)+vnorm(2)*vslip(1))
 tensor(5)=(vnorm(1)*vslip(3)+vnorm(3)*vslip(1))
 tensor(6)=(vnorm(2)*vslip(3)+vnorm(3)*vslip(2))
 call assign_source(sp(1),sp(2),sp(3),ix2,jy2,kz2,cf8,nsou)
 do isu=1,nsou
   ii=nfadd+isu
   ruptm(ii)=sp(5)
   riset(ii)=sp(6)
   sparam2(ii)=sp(7)
   do ijk=1,4
     ix=ix2(isu,ijk)
     jy=jy2(isu,ijk)
     kz=kz2(isu,ijk)
     index_xyz_source(1,ii,ijk)=ix
     index_xyz_source(2,ii,ijk)=jy
     index_xyz_source(3,ii,ijk)=kz
     if(kz<=nzrg1(ijk)) then
       if(ijk==1) then
         if(is_moment==2) rmoment=arsp*cmu(idmat1(kz,ix,jy))
         volume=0.25*(gridx1(ix+1)-gridx1(ix-1))* &
               (gridy1(jy+1)-gridy1(jy-1))*(gridz(kz+1)-gridz(kz))
       else if(ijk==2) then
         if(is_moment==2) rmoment=arsp*0.25* &
               (cmu(idmat1(kz,ix,jy))+cmu(idmat1(kz,ix+1,jy))+ &
                cmu(idmat1(kz,ix,jy+1))+ cmu(idmat1(kz,ix+1,jy+1)))
         volume=(gridx1(ix+1)-gridx1(ix))* &
                (gridy1(jy+1)-gridy1(jy))*(gridz(kz+1)-gridz(kz))
       else if(ijk==3) then
         if(is_moment==2) rmoment=arsp*0.25* &
               (cmu(idmat1(kz,ix,jy)) + cmu(idmat1(kz,ix+1,jy))+ &
                cmu(idmat1(kz-1,ix,jy))+cmu(idmat1(kz-1,ix+1,jy)))
         volume=0.25*(gridx1(ix+1)-gridx1(ix))* &
               (gridy1(jy+1)-gridy1(jy-1))*(gridz(kz+1)-gridz(kz-1))
       else 
         if(is_moment==2) rmoment=arsp*0.25* &
               (cmu(idmat1(kz,ix,jy  ))+cmu(idmat1(kz,ix,jy+1))+ &
                cmu(idmat1(kz-1,ix,jy))+cmu(idmat1(kz-1,ix,jy+1)))
         volume=0.25*(gridx1(ix+1)-gridx1(ix-1))* &
               (gridy1(jy+1)-gridy1(jy))*(gridz(kz+1)-gridz(kz-1))
       endif
     else
       kzb=kz-nztop
       if(ijk==1) then
         if(is_moment==2) rmoment=arsp*cmu(idmat2(kzb,ix,jy))
         volume=0.25*(gridx2(ix+1)-gridx2(ix-1))* &
                 (gridy2(jy+1)-gridy2(jy-1))*(gridz(kz+1)-gridz(kz))
       else if(ijk==2) then
         if(is_moment==2) rmoment=arsp*0.25* &
               (cmu(idmat2(kzb,ix,jy))  +cmu(idmat2(kzb,ix+1,jy))+ &
                cmu(idmat2(kzb,ix,jy+1))+cmu(idmat2(kzb,ix+1,jy+1)))
         volume=(gridx2(ix+1)-gridx2(ix))* &
                (gridy2(jy+1)-gridy2(jy))*(gridz(kz+1)-gridz(kz))
       else if(ijk==3) then
         if(is_moment==2) rmoment=arsp*0.25* &
               (cmu(idmat2(kzb,ix,jy))  +cmu(idmat2(kzb,ix+1,jy))+ &
                cmu(idmat2(kzb-1,ix,jy))+cmu(idmat2(kzb-1,ix+1,jy)))
         volume=0.25*(gridx2(ix+1)-gridx2(ix))* &
               (gridy2(jy+1)-gridy2(jy-1))*(gridz(kz+1)-gridz(kz-1))
       else
         if(is_moment==2) rmoment=arsp*0.25* &
               (cmu(idmat2(kzb,ix,jy))  +cmu(idmat2(kzb,ix,jy+1))+ &
                cmu(idmat2(kzb-1,ix,jy))+cmu(idmat2(kzb-1,ix,jy+1)))
         volume=0.25*(gridx2(ix+1)-gridx2(ix-1))* &
               (gridy2(jy+1)-gridy2(jy))*(gridz(kz+1)-gridz(kz-1))
       endif
     endif
     cof=rmoment*dt/volume
     if(ijk==1) then
       famp(ii,1)=cof*tensor(1)*cf8(isu,ijk)
       famp(ii,2)=cof*tensor(2)*cf8(isu,ijk)
       famp(ii,3)=cof*tensor(3)*cf8(isu,ijk)
     else
       famp(ii,ijk+2)=cof*tensor(ijk+2)*cf8(isu,ijk)
     endif
   enddo
   call normalize(ii,ierr)
 enddo
 nfadd=nfadd+nsou
 return
end subroutine source_coef
!
!-----------------------------------------------------------------------
subroutine normalize(ii,ierr)
 use source_parm_comm
 implicit NONE
 integer, intent(IN):: ii
 integer, intent(OUT):: ierr
 real, parameter:: pi=3.1415926, pi2=2.*pi
 real, parameter:: cft4=0.20,c41=0.60,c42=1.-c41
 real, parameter:: cft5=0.20,c51=0.50,c52=1.-c51
 integer:: k
 real:: tt1,tt2,coef_norm,gammln
!
 coef_norm=1.0
 select case (id_sf_type )
 case (1) 
!  Brune'' Slip Rate
   riset(ii)=riset(ii)/pi2
   coef_norm = 1./riset(ii)
 case (2) 
! t**sp2*(1-t)**(5-sp2)
!   sparam2(ii)=amax1(0.5,amin1(4.5,sparam2(ii)))
!   coef_norm=720.0*exp(gammln(6.-sparam2(ii)))* &
!                   exp(gammln(1.+sparam2(ii))) / riset(ii)
! sp2=1
   sparam2(ii)=1.0
   coef_norm=30./riset(ii)
 case (3) 
   sparam2(ii)=0.2
   tt1=sparam2(ii)*riset(ii)
   tt2=(riset(ii)-tt1)
   coef_norm=pi2/(4.*tt1+pi*tt2) 
 case (4) 
   tt1=cft4*riset(ii)
   tt2=riset(ii)-tt1
   coef_norm=pi2/(c41*2.*pi*tt1+c42*4.*tt1+c42*pi*tt2)
   sparam2(ii)=tt1/tt2
 case (5) 
   tt1=cft5*riset(ii)
   tt2=riset(ii)-tt1
   coef_norm=pi2/(4.*tt1+c51*pi*1.5*tt1+c52*pi*tt2)
   sparam2(ii)=tt1/tt2
 case (6) 
   tt1=riset(ii)
   tt2=sparam2(ii)
   coef_norm=pi/(2.*tt1+0.5*pi*tt2)
   riset(ii)=tt1+tt2
 case default
!  No definition for inputing type of source fun.
   ierr=1
 end select
 do k=1,6
   famp(ii,k)=famp(ii,k)*coef_norm
 enddo
 return
end subroutine normalize
!
!-----------------------------------------------------------------------
subroutine initial(myid,is_moment,dt)
 use grid_node_comm
 use wave_field_comm
 use random_initial
 implicit NONE
 integer, intent(IN):: myid,is_moment
 real, intent(IN):: dt
 integer:: i,j,k,iseed
! real:: pertV=1.e-28, pertS=1.e-25, pertQ=1.e-26
 real:: pertV=0.0, pertS=0.0, pertQ=0.0
!
 call coef_interp
 iseed=3
! Region I
 call assign_rand_ini(iseed,pertV,v1x)
 call assign_rand_ini(iseed,pertV,v1y)
 call assign_rand_ini(iseed,pertV,v1z)
 call assign_rand_ini(iseed,pertS,t1xx)
 call assign_rand_ini(iseed,pertS,t1yy)
 call assign_rand_ini(iseed,pertS,t1zz)
 call assign_rand_ini(iseed,pertS,t1xz)
 call assign_rand_ini(iseed,pertS,t1yz)
 call assign_rand_ini(iseed,pertS,t1xy)
 call assign_rand_ini(iseed,pertQ,qt1xx)
 call assign_rand_ini(iseed,pertQ,qt1yy)
 call assign_rand_ini(iseed,pertQ,qt1zz)
 call assign_rand_ini(iseed,pertQ,qt1xz)
 call assign_rand_ini(iseed,pertQ,qt1yz)
 call assign_rand_ini(iseed,pertQ,qt1xy)
! Region II
 call assign_rand_ini(iseed,pertV,v2x)
 call assign_rand_ini(iseed,pertV,v2y)
 call assign_rand_ini(iseed,pertV,v2z)
 call assign_rand_ini(iseed,pertS,t2xx)
 call assign_rand_ini(iseed,pertS,t2yy)
 call assign_rand_ini(iseed,pertS,t2zz)
 call assign_rand_ini(iseed,pertS,t2xz)
 call assign_rand_ini(iseed,pertS,t2yz)
 call assign_rand_ini(iseed,pertS,t2xy)
 call assign_rand_ini(iseed,pertQ,qt2xx)
 call assign_rand_ini(iseed,pertQ,qt2yy)
 call assign_rand_ini(iseed,pertQ,qt2zz)
 call assign_rand_ini(iseed,pertQ,qt2xz)
 call assign_rand_ini(iseed,pertQ,qt2yz)
 call assign_rand_ini(iseed,pertQ,qt2xy)
!PML
 if(lbx(2) >=lbx(1)) then
   call assign_rand_ini(iseed,pertV,v1x_px)
   call assign_rand_ini(iseed,pertV,v1y_px)
   call assign_rand_ini(iseed,pertV,v1z_px)
   call assign_rand_ini(iseed,pertS,t1xx_px)
   call assign_rand_ini(iseed,pertS,t1yy_px)
   call assign_rand_ini(iseed,pertS,t1xz_px)
   call assign_rand_ini(iseed,pertS,t1xy_px)
   call assign_rand_ini(iseed,pertQ,qt1xx_px)
   call assign_rand_ini(iseed,pertQ,qt1yy_px)
   call assign_rand_ini(iseed,pertQ,qt1xz_px)
   call assign_rand_ini(iseed,pertQ,qt1xy_px)
   call assign_rand_ini(iseed,pertV,v2x_px)
   call assign_rand_ini(iseed,pertV,v2y_px)
   call assign_rand_ini(iseed,pertV,v2z_px)
   call assign_rand_ini(iseed,pertS,t2xx_px)
   call assign_rand_ini(iseed,pertS,t2yy_px)
   call assign_rand_ini(iseed,pertS,t2xz_px)
   call assign_rand_ini(iseed,pertS,t2xy_px)
   call assign_rand_ini(iseed,pertQ,qt2xx_px)
   call assign_rand_ini(iseed,pertQ,qt2yy_px)
   call assign_rand_ini(iseed,pertQ,qt2xz_px)
   call assign_rand_ini(iseed,pertQ,qt2xy_px)
 endif
!
 if(lby(2)>=lby(1)) then
   call assign_rand_ini(iseed,pertV,v1x_py)
   call assign_rand_ini(iseed,pertV,v1y_py)
   call assign_rand_ini(iseed,pertV,v1z_py)
   call assign_rand_ini(iseed,pertS,t1xx_py)
   call assign_rand_ini(iseed,pertS,t1yy_py)
   call assign_rand_ini(iseed,pertS,t1yz_py)
   call assign_rand_ini(iseed,pertS,t1xy_py)
   call assign_rand_ini(iseed,pertQ,qt1xx_py)
   call assign_rand_ini(iseed,pertQ,qt1yy_py)
   call assign_rand_ini(iseed,pertQ,qt1yz_py)
   call assign_rand_ini(iseed,pertQ,qt1xy_py)
   call assign_rand_ini(iseed,pertV,v2x_py)
   call assign_rand_ini(iseed,pertV,v2y_py)
   call assign_rand_ini(iseed,pertV,v2z_py)
   call assign_rand_ini(iseed,pertS,t2xx_py)
   call assign_rand_ini(iseed,pertS,t2yy_py)
   call assign_rand_ini(iseed,pertS,t2yz_py)
   call assign_rand_ini(iseed,pertS,t2xy_py)
   call assign_rand_ini(iseed,pertQ,qt2xx_py)
   call assign_rand_ini(iseed,pertQ,qt2yy_py)
   call assign_rand_ini(iseed,pertQ,qt2yz_py)
   call assign_rand_ini(iseed,pertQ,qt2xy_py)
 endif
!
 call assign_rand_ini(iseed,pertV,v2x_pz)
 call assign_rand_ini(iseed,pertV,v2y_pz)
 call assign_rand_ini(iseed,pertV,v2z_pz)
 call assign_rand_ini(iseed,pertS,t2xx_pz)
 call assign_rand_ini(iseed,pertS,t2zz_pz)
 call assign_rand_ini(iseed,pertS,t2xz_pz)
 call assign_rand_ini(iseed,pertS,t2yz_pz)
 call assign_rand_ini(iseed,pertQ,qt2xx_pz)
 call assign_rand_ini(iseed,pertQ,qt2zz_pz)
 call assign_rand_ini(iseed,pertQ,qt2xz_pz)
 call assign_rand_ini(iseed,pertQ,qt2yz_pz)
!
 call Pml_coef(myid,dt)
 if(is_moment==9) call ini_plane_wave(myid,dt)
 return
end subroutine initial
!
!-----------------------------------------------------------------------
subroutine alloc_memory(myid)
 use mpi
 use grid_node_comm
 use wave_field_comm
 use itface_comm
 use boundMpi_comm
 use pointer_remapping
! use remapping_interface
 use, intrinsic :: iso_c_binding, ONLY: c_f_pointer, c_loc
 implicit NONE
 integer, intent(IN):: myid
 integer:: i,j,nx6,ny6,nzt1,nv2,nth,nti
 integer(kind=MPI_ADDRESS_KIND) size 
 integer :: real_itemsize, integer_itemsize, ierror
!
 order_imsr=-1
 pid_ims   =-1
 pid_imr   =-1
 if(myid_x<np_x .and. myid_y<np_y) then
   order_imsr= mod(myid_y,2)+2
   if(myid_x==1 .or. myid_y==1) order_imsr=1
   pid_imr   = myid+np_x+1
 endif
 if(myid_x >1 .and. myid_y >1) then
   order_imsr= mod(myid_y,2)+2
   if(myid_x==np_x .or. myid_y==np_y) order_imsr=4
   pid_ims   = myid-np_x-1
 endif
!
 nzt1=nztop+1 
!
! Use MPI memory allocation with C pointers and
! define Fortran pointers for Fortran calcs
 !call MPI_Type_extent(MPI_REAL, real_itemsize, ierror)
 !call MPI_Type_extent(MPI_INTEGER, integer_itemsize, ierror)
real_itemsize=4
integer_itemsize=4
! Grid space
! allocate(gridx2(-1:nxbtm+4), gridy2(-1:nybtm+4) )
! allocate(gridx1(-1:nxtop+4), gridy1(-1:nytop+4) )
! allocate(gridz(-1:nztol+4) )

 size=(nxbtm+6)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_gridx2, ierror)
 call c_f_pointer(cptr_gridx2, gridx2_init, (/ nxbtm+6 /))
 call assign1d_real( gridx2_init, -1, gridx2 )
 size=(nybtm+6)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_gridy2, ierror)
 call c_f_pointer(cptr_gridy2, gridy2_init, (/ nybtm+6 /))
 call assign1d_real( gridy2_init, -1, gridy2 )
 size=(nxtop+6)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_gridx1, ierror)
 call c_f_pointer(cptr_gridx1, gridx1_init, (/ nxtop+6 /))
 call assign1d_real( gridx1_init, -1, gridx1 )
 size=(nytop+6)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_gridy1, ierror)
 call c_f_pointer(cptr_gridy1, gridy1_init, (/ nytop+6 /))
 call assign1d_real( gridy1_init, -1, gridy1 )
 size=(nztol+6)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_gridz, ierror)
 call c_f_pointer(cptr_gridz, gridz_init, (/ nztol+6 /))
 call assign1d_real( gridz_init, -1, gridz )

! allocate(dxh1(4,nxtop), dxi1(4,nxtop), dyh1(4,nytop))
! allocate(dyi1(4,nytop), dzh1(4,nztop+1), dzi1(4,nztop+1) )
 size=4*nxtop*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dxh1, ierror)
 call c_f_pointer(cptr_dxh1, dxh1, (/ 4,nxtop /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dxi1, ierror)
 call c_f_pointer(cptr_dxi1, dxi1, (/ 4,nxtop /))
 size=4*nytop*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dyh1, ierror)
 call c_f_pointer(cptr_dyh1, dyh1, (/ 4,nytop /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dyi1, ierror)
 call c_f_pointer(cptr_dyi1, dyi1, (/ 4,nytop /))
 size=4*(nztop+1)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_dzh1, ierror)
 call c_f_pointer(cptr_dzh1, dzh1, (/ 4,nztop+1/))
 call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_dzi1, ierror)
 call c_f_pointer(cptr_dzi1, dzi1, (/ 4,nztop+1/))

! allocate(dxh2(4,nxbtm), dxi2(4,nxbtm), dyh2(4,nybtm) )
! allocate(dyi2(4,nybtm), dzh2(4,nzbtm), dzi2(4,nzbtm) )
 size=4*nxbtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dxh2, ierror)
 call c_f_pointer(cptr_dxh2, dxh2, (/ 4,nxbtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dxi2, ierror)
 call c_f_pointer(cptr_dxi2, dxi2, (/ 4,nxbtm /))
 size=4*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dyh2, ierror)
 call c_f_pointer(cptr_dyh2, dyh2, (/ 4,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dyi2, ierror)
 call c_f_pointer(cptr_dyi2, dyi2, (/ 4,nybtm /))
 size=4*nzbtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dzh2, ierror)
 call c_f_pointer(cptr_dzh2, dzh2, (/ 4,nzbtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_dzi2, ierror)
 call c_f_pointer(cptr_dzi2, dzi2, (/ 4,nzbtm /))

! material index
! allocate(idmat1(0:nztop+1,nxtop+1,nytop+1))
 size=(nztop+2)*(nxtop+1)*(nytop+1)*integer_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_idmat1, ierror)
 call c_f_pointer(cptr_idmat1, idmat1_init, (/ nztop+2, nxtop+1, nytop+1 /))
 call assign3d_integer(idmat1_init, 0, 1, 1, idmat1)

! allocate(idmat2(0:nzbtm,  nxbtm+1,nybtm+1))
 size=(nzbtm+1)*(nxbtm+1)*(nybtm+1)*integer_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_idmat2, ierror)
 call c_f_pointer(cptr_idmat2, idmat2_init, (/ nzbtm+1, nxbtm+1, nybtm+1 /))
 call assign3d_integer(idmat2_init, 0, 1, 1, idmat2)
 idmat1=0
 idmat2=0
! On the interface between Region I and II
 nx6=nxtop+6
 ny6=nytop+6
 allocate(sdy1(nx6), sdy2(nx6), rcy1(nx6), rcy2(nx6) )
 allocate(sdx1(ny6), sdx2(ny6), rcx1(ny6), rcx2(ny6) )
! size=nx6*real_itemsize
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_sdy1, ierror)
! call c_f_pointer(cptr_sdy1, sdy1, (/ nx6 /))
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_sdy2, ierror)
! call c_f_pointer(cptr_sdy2, sdy2, (/ nx6 /))
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_rcy1, ierror)
! call c_f_pointer(cptr_rcy1, rcy1, (/ nx6 /))
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_rcy2, ierror)
! call c_f_pointer(cptr_rcy2, rcy2, (/ nx6 /))
! size=ny6*real_itemsize
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_sdx1, ierror)
! call c_f_pointer(cptr_sdx1, sdx1, (/ ny6 /))
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_sdy2, ierror)
! call c_f_pointer(cptr_sdx2, sdx2, (/ ny6 /))
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_rcx1, ierror)
! call c_f_pointer(cptr_rcx1, rcx1, (/ ny6 /))
! call MPI_Alloc_mem(size, MPI_INFO_NULL,cptr_rcx2, ierror)
! call c_f_pointer(cptr_rcx2, rcx2, (/ ny6 /))

 nx6=nxbtm+1
 ny6=nybtm+1
 allocate(chx(8,0:nx6), cix(8,0:nx6), chy(8,0:ny6), ciy(8,0:ny6))
! size=(nx6+1)*8*real_itemsize
! call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_chx, ierror)
! call c_f_pointer(cptr_chx, chx_init, (/ 8,nx6+1 /))
! call assign2d_real(chx_init, 1, 0, chx)
! call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_cix, ierror)
! call c_f_pointer(cptr_cix, cix_init, (/ 8,nx6+1 /))
! call assign2d_real(cix_init, 1, 0, cix)
! size=(ny6+1)*8*real_itemsize
! call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_chy, ierror)
! call c_f_pointer(cptr_chy, chy_init, (/ 8,ny6+1 /))
! call assign2d_real(chy_init, 1, 0, chy)
! call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_ciy, ierror)
! call c_f_pointer(cptr_ciy, ciy_init, (/ 8,ny6+1 /))
! call assign2d_real(ciy_init, 1, 0, ciy)
!
! Varibales for PML
 i=lbx(1)
 j=lbx(2)
 if(j >= i) then
!   allocate(damp1_x(nzt1,nytop,i:j),damp2_x(nzbtm,nybtm,i:j))
   size=nzt1*nytop*(j-i+1)*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_damp1_x, ierror)
   call c_f_pointer(cptr_damp1_x, damp1_x_init, (/ nzt1,nytop,j-i+1 /))
   call assign3d_real( damp1_x_init, 1, 1, i, damp1_x )
   size=nzbtm*nybtm*(j-i+1)*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_damp2_x, ierror)
   call c_f_pointer(cptr_damp2_x, damp2_x_init, (/ nzbtm,nybtm,j-i+1 /))
   call assign3d_real( damp2_x_init, 1, 1, i, damp2_x )

   nv2=(j-i+1)*mw1_pml
   nth=nv2+1-i
   nti=nv2+j
!   allocate(v1x_px(nztop,nv2,nytop), v1y_px(nztop,nv2,nytop))
!   allocate(v1z_px(nztop,nv2,nytop))
!   allocate(t1xx_px(nztop,nti,nytop), qt1xx_px(nztop,nti,nytop))
!   allocate(t1yy_px(nztop,nti,nytop), qt1yy_px(nztop,nti,nytop))
!   allocate(t1xz_px(nzt1, nth,nytop), qt1xz_px(nzt1, nth,nytop))
!   allocate(t1xy_px(nztop,nth,nytop), qt1xy_px(nztop,nth,nytop))
   size=nztop*nv2*nytop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1x_px, ierror)
   call c_f_pointer(cptr_v1x_px, v1x_px, (/ nztop,nv2,nytop /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1y_px, ierror)
   call c_f_pointer(cptr_v1y_px, v1y_px, (/ nztop,nv2,nytop /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1z_px, ierror)
   call c_f_pointer(cptr_v1z_px, v1z_px, (/ nztop,nv2,nytop /))
   size=nztop*nti*nytop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xx_px, ierror)
   call c_f_pointer(cptr_t1xx_px, t1xx_px, (/ nztop,nti,nytop /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xx_px, ierror)
   call c_f_pointer(cptr_qt1xx_px, qt1xx_px, (/ nztop,nti,nytop /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1yy_px, ierror)
   call c_f_pointer(cptr_t1yy_px, t1yy_px, (/ nztop,nti,nytop /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1yy_px, ierror)
   call c_f_pointer(cptr_qt1yy_px, qt1yy_px, (/ nztop,nti,nytop /))
   size=nzt1*nth*nytop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xz_px, ierror)
   call c_f_pointer(cptr_t1xz_px, t1xz_px, (/ nzt1, nth,nytop /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xz_px, ierror)
   call c_f_pointer(cptr_qt1xz_px,qt1xz_px, (/ nzt1, nth,nytop /))
   size=nztop*nth*nytop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xy_px, ierror)
   call c_f_pointer(cptr_t1xy_px, t1xy_px, (/ nztop,nth,nytop /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xy_px, ierror)
   call c_f_pointer(cptr_qt1xy_px, qt1xy_px, (/ nztop,nth,nytop /))
   nv2=(j-i+1)*mw2_pml
   nth=nv2+1-i
   nti=nv2+j
!   allocate(v2x_px(nzbtm,nv2,nybtm), v2y_px(nzbtm,nv2,nybtm))
!   allocate(v2z_px(nzbtm,nv2,nybtm))
!   allocate(t2xx_px(nzbtm,nti,nybtm), qt2xx_px(nzbtm,nti,nybtm))
!   allocate(t2yy_px(nzbtm,nti,nybtm), qt2yy_px(nzbtm,nti,nybtm))
!   allocate(t2xz_px(nzbtm,nth,nybtm), qt2xz_px(nzbtm,nth,nybtm))
!   allocate(t2xy_px(nzbtm,nth,nybtm), qt2xy_px(nzbtm,nth,nybtm))
   size=nzbtm*nv2*nybtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2x_px, ierror)
   call c_f_pointer(cptr_v2x_px, v2x_px, (/ nzbtm,nv2,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2y_px, ierror)
   call c_f_pointer(cptr_v2y_px, v2y_px, (/ nzbtm,nv2,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2z_px, ierror)
   call c_f_pointer(cptr_v2z_px, v2z_px, (/ nzbtm,nv2,nybtm /))
   size=nzbtm*nti*nybtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xx_px, ierror)
   call c_f_pointer(cptr_t2xx_px, t2xx_px, (/ nzbtm,nti,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xx_px, ierror)
   call c_f_pointer(cptr_qt2xx_px, qt2xx_px, (/ nzbtm,nti,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2yy_px, ierror)
   call c_f_pointer(cptr_t2yy_px, t2yy_px, (/ nzbtm,nti,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2yy_px, ierror)
   call c_f_pointer(cptr_qt2yy_px, qt2yy_px, (/ nzbtm,nti,nybtm /))
   size=nzbtm*nth*nybtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xz_px, ierror)
   call c_f_pointer(cptr_t2xz_px, t2xz_px, (/ nzbtm,nth,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xz_px, ierror)
   call c_f_pointer(cptr_qt2xz_px,qt2xz_px, (/ nzbtm,nth,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xy_px, ierror)
   call c_f_pointer(cptr_t2xy_px, t2xy_px, (/ nzbtm,nth,nybtm /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xy_px, ierror)
   call c_f_pointer(cptr_qt2xy_px, qt2xy_px, (/ nzbtm,nth,nybtm /))

   nx6=i*4+1
   ny6=j*4+4
!   allocate(abcx1(nx6:ny6,nztop,nytop), abcx2(nx6:ny6,nzbtm,nybtm))
   size=(ny6-nx6+1)*nztop*nytop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_abcx1, ierror)
   call c_f_pointer(cptr_abcx1, abcx1_init, (/ ny6-nx6+1, nztop, nytop /))
   call assign3d_real( abcx1_init, nx6, 1, 1, abcx1 )
   size=(ny6-nx6+1)*nzbtm*nybtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_abcx2, ierror)
   call c_f_pointer(cptr_abcx2, abcx2_init, (/ ny6-nx6+1, nzbtm, nybtm /))
   call assign3d_real( abcx2_init, nx6, 1, 1, abcx2 )
 endif
! Boundary y
 i=lby(1)
 j=lby(2)
 if(j >= i) then
!   allocate(damp1_y(nzt1,nxtop,i:j),damp2_y(nzbtm,nxbtm,i:j))
   size=nzt1*nxtop*(j-i+1)*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_damp1_y, ierror)
   call c_f_pointer(cptr_damp1_y, damp1_y_init, (/ nzt1,nxtop,j-i+1 /))
   call assign3d_real( damp1_y_init, 1, 1, i, damp1_y )
   size=nzbtm*nxbtm*(j-i+1)*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_damp2_y, ierror)
   call c_f_pointer(cptr_damp2_y, damp2_y_init, (/ nzbtm,nxbtm,j-i+1 /))
   call assign3d_real( damp2_y_init, 1, 1, i, damp2_y)
   nv2=(j-i+1)*mw1_pml
   nth=nv2+1-i
   nti=nv2+j
!   allocate(v1x_py(nztop,nxtop,nv2), v1y_py(nztop,nxtop,nv2))
!   allocate(v1z_py(nztop,nxtop,nv2))
!   allocate(t1xx_py(nztop,nxtop,nti), qt1xx_py(nztop,nxtop,nti))
!   allocate(t1yy_py(nztop,nxtop,nti), qt1yy_py(nztop,nxtop,nti))
!   allocate(t1yz_py(nzt1, nxtop,nth), qt1yz_py(nzt1, nxtop,nth))
!   allocate(t1xy_py(nztop,nxtop,nth), qt1xy_py(nztop,nxtop,nth))
   size=nztop*nv2*nxtop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1x_py, ierror)
   call c_f_pointer(cptr_v1x_py, v1x_py, (/ nztop,nxtop,nv2 /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1y_py, ierror)
   call c_f_pointer(cptr_v1y_py, v1y_py, (/ nztop,nxtop,nv2 /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1z_py, ierror)
   call c_f_pointer(cptr_v1z_py, v1z_py, (/ nztop,nxtop,nv2 /))
   size=nztop*nxtop*nti*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xx_py, ierror)
   call c_f_pointer(cptr_t1xx_py, t1xx_py, (/ nztop,nxtop,nti /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xx_py, ierror)
   call c_f_pointer(cptr_qt1xx_py, qt1xx_py, (/ nztop,nxtop,nti /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1yy_py, ierror)
   call c_f_pointer(cptr_t1yy_py, t1yy_py, (/ nztop,nxtop,nti /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1yy_py, ierror)
   call c_f_pointer(cptr_qt1yy_py, qt1yy_py, (/ nztop,nxtop,nti /))
   size=nzt1*nth*nxtop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1yz_py, ierror)
   call c_f_pointer(cptr_t1yz_py, t1yz_py, (/ nzt1, nxtop,nth /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1yz_py, ierror)
   call c_f_pointer(cptr_qt1yz_py,qt1yz_py, (/ nzt1, nxtop,nth /))
   size=nztop*nth*nxtop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xy_py, ierror)
   call c_f_pointer(cptr_t1xy_py, t1xy_py, (/ nztop,nxtop,nth /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xy_py, ierror)
   call c_f_pointer(cptr_qt1xy_py, qt1xy_py, (/ nztop,nxtop,nth /))

   nv2=(j-i+1)*mw2_pml
   nth=nv2+1-i
   nti=nv2+j
!   allocate(v2x_py(nzbtm,nxbtm,nv2), v2y_py(nzbtm,nxbtm,nv2))
!   allocate(v2z_py(nzbtm,nxbtm,nv2))
!   allocate(t2xx_py(nzbtm,nxbtm,nti), qt2xx_py(nzbtm,nxbtm,nti))
!   allocate(t2yy_py(nzbtm,nxbtm,nti), qt2yy_py(nzbtm,nxbtm,nti))
!   allocate(t2yz_py(nzbtm,nxbtm,nth), qt2yz_py(nzbtm,nxbtm,nth))
!   allocate(t2xy_py(nzbtm,nxbtm,nth), qt2xy_py(nzbtm,nxbtm,nth))
   size=nzbtm*nv2*nxbtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2x_py, ierror)
   call c_f_pointer(cptr_v2x_py, v2x_py, (/ nzbtm,nxbtm,nv2 /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2y_py, ierror)
   call c_f_pointer(cptr_v2y_py, v2y_py, (/ nzbtm,nxbtm,nv2 /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2z_py, ierror)
   call c_f_pointer(cptr_v2z_py, v2z_py, (/ nzbtm,nxbtm,nv2 /))
   size=nzbtm*nti*nxbtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xx_py, ierror)
   call c_f_pointer(cptr_t2xx_py, t2xx_py, (/ nzbtm,nxbtm,nti /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xx_py, ierror)
   call c_f_pointer(cptr_qt2xx_py, qt2xx_py, (/ nzbtm,nxbtm,nti /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2yy_py, ierror)
   call c_f_pointer(cptr_t2yy_py, t2yy_py, (/ nzbtm,nxbtm,nti /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2yy_py, ierror)
   call c_f_pointer(cptr_qt2yy_py, qt2yy_py, (/ nzbtm,nxbtm,nti /))
   size=nzbtm*nth*nxbtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2yz_py, ierror)
   call c_f_pointer(cptr_t2yz_py, t2yz_py, (/ nzbtm,nxbtm,nth /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2yz_py, ierror)
   call c_f_pointer(cptr_qt2yz_py, qt2yz_py, (/ nzbtm,nxbtm,nth /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xy_py, ierror)
   call c_f_pointer(cptr_t2xy_py, t2xy_py, (/ nzbtm,nxbtm,nth /))
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xy_py, ierror)
   call c_f_pointer(cptr_qt2xy_py, qt2xy_py, (/ nzbtm,nxbtm,nth /))
   nx6=i*4+1
   ny6=j*4+4
!   allocate(abcy1(nx6:ny6,nztop,nxtop), abcy2(nx6:ny6,nzbtm,nxbtm))
   size=(ny6-nx6+1)*nztop*nxtop*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_abcy1, ierror)
   call c_f_pointer(cptr_abcy1, abcy1_init, (/ ny6-nx6+1, nztop, nxtop /))
   call assign3d_real( abcy1_init, nx6, 1, 1, abcy1 )
   size=(ny6-nx6+1)*nzbtm*nxbtm*real_itemsize
   call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_abcy2, ierror)
   call c_f_pointer(cptr_abcy2, abcy2_init, (/ ny6-nx6+1, nzbtm, nxbtm /))
   call assign3d_real( abcy2_init, nx6, 1, 1, abcy2 )
 endif
! allocate(drti1(mw1_pml1,0:1), drth1(mw1_pml1,0:1) )
! allocate(drti2(mw2_pml1,0:1), drth2(mw2_pml1,0:1) )
! allocate(drvh1(mw1_pml1,0:1), drvh2(mw2_pml1,0:1) )
 size=mw1_pml1*2*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_drti1, ierror)
 call c_f_pointer(cptr_drti1, drti1_init, (/ mw1_pml1, 2 /))
 call assign2d_real( drti1_init, 1, 0, drti1 )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_drth1, ierror)
 call c_f_pointer(cptr_drth1, drth1_init, (/ mw1_pml1, 2 /))
 call assign2d_real( drth1_init, 1, 0, drth1 )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_drvh1, ierror)
 call c_f_pointer(cptr_drvh1, drvh1_init, (/ mw1_pml1, 2 /))
 call assign2d_real( drvh1_init, 1, 0, drvh1 )
 size=mw2_pml1*2*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_drti2, ierror)
 call c_f_pointer(cptr_drti2, drti2_init, (/ mw2_pml1, 2 /))
 call assign2d_real( drti2_init, 1, 0, drti2 )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_drth2, ierror)
 call c_f_pointer(cptr_drth2, drth2_init, (/ mw2_pml1, 2 /))
 call assign2d_real( drth2_init, 1, 0, drth2 )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_drvh2, ierror)
 call c_f_pointer(cptr_drvh2, drvh2_init, (/ mw2_pml1, 2 /))
 call assign2d_real( drvh2_init, 1, 0, drvh2 )

! allocate(damp2_z(nxbtm,nybtm))
 size=nxbtm*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_damp2_z, ierror)
 call c_f_pointer(cptr_damp2_z, damp2_z, (/ nxbtm,nybtm /))

! allocate(v2x_pz(mw2_pml,nxbtm,nybtm),v2y_pz(mw2_pml,nxbtm,nybtm))
! allocate(v2z_pz(mw2_pml,nxbtm,nybtm))
 size=mw2_pml*nxbtm*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2x_pz, ierror)
 call c_f_pointer(cptr_v2x_pz, v2x_pz, (/ mw2_pml,nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2y_pz, ierror)
 call c_f_pointer(cptr_v2y_pz, v2y_pz, (/ mw2_pml,nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2z_pz, ierror)
 call c_f_pointer(cptr_v2z_pz, v2z_pz, (/ mw2_pml,nxbtm,nybtm /))

! allocate(t2xx_pz(mw2_pml, nxbtm,nybtm), qt2xx_pz(mw2_pml, nxbtm,nybtm))
! allocate(t2zz_pz(mw2_pml, nxbtm,nybtm), qt2zz_pz(mw2_pml, nxbtm,nybtm))
! allocate(t2xz_pz(mw2_pml1,nxbtm,nybtm), qt2xz_pz(mw2_pml1,nxbtm,nybtm))
! allocate(t2yz_pz(mw2_pml1,nxbtm,nybtm), qt2yz_pz(mw2_pml1,nxbtm,nybtm))
! allocate(abcz(4,nxbtm,nybtm))
 size=mw2_pml*nxbtm*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xx_pz, ierror)
 call c_f_pointer(cptr_t2xx_pz, t2xx_pz, (/ mw2_pml, nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xx_pz, ierror)
 call c_f_pointer(cptr_qt2xx_pz, qt2xx_pz, (/ mw2_pml, nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2zz_pz, ierror)
 call c_f_pointer(cptr_t2zz_pz, t2zz_pz, (/ mw2_pml, nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2zz_pz, ierror)
 call c_f_pointer(cptr_qt2zz_pz, qt2zz_pz, (/ mw2_pml, nxbtm,nybtm /))
 size=mw2_pml1*nxbtm*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xz_pz, ierror)
 call c_f_pointer(cptr_t2xz_pz, t2xz_pz, (/ mw2_pml1, nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xz_pz, ierror)
 call c_f_pointer(cptr_qt2xz_pz, qt2xz_pz, (/ mw2_pml1, nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2yz_pz, ierror)
 call c_f_pointer(cptr_t2yz_pz, t2yz_pz, (/ mw2_pml1, nxbtm,nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2yz_pz, ierror)
 call c_f_pointer(cptr_qt2yz_pz, qt2yz_pz, (/ mw2_pml1, nxbtm,nybtm /))
 size=4*nxbtm*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_abcz, ierror)
 call c_f_pointer(cptr_abcz, abcz, (/ 4,nxbtm,nybtm /))

! variables in Region I
! allocate(v1x(0:nztop+1,-1:nxtop+1, 0:nytop+2))
! allocate(v1y(0:nztop+1, 0:nxtop+2,-1:nytop+1))
! allocate(v1z(0:nztop+1, 0:nxtop+2, 0:nytop+2))

 size=(nztop+2)*(nxtop+3)*(nytop+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1x, ierror)
 call c_f_pointer(cptr_v1x, v1x_init, (/ nztop+2, nxtop+3, nytop+3 /))
 call assign3d_real( v1x_init, 0, -1, 0, v1x )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1y, ierror)
 call c_f_pointer(cptr_v1y, v1y_init, (/ nztop+2, nxtop+3, nytop+3 /))
 call assign3d_real( v1y_init, 0, 0, -1, v1y )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v1z, ierror)
 call c_f_pointer(cptr_v1z, v1z_init, (/ nztop+2, nxtop+3, nytop+3 /))
 call assign3d_real( v1z_init, 0, 0, 0, v1z )

! allocate(t1xx(nztop,   0:nxtop+2,   nytop))
! allocate(t1xy(nztop,  -1:nxtop+1,-1:nytop+1))
! allocate(t1xz(nztop+1,-1:nxtop+1,   nytop))
! allocate(t1yy(nztop,     nxtop,   0:nytop+2))
! allocate(t1yz(nztop+1,   nxtop,  -1:nytop+1))
! allocate(t1zz(nztop,     nxtop,     nytop))
 size=nztop*(nxtop+3)*nytop*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xx, ierror)
 call c_f_pointer(cptr_t1xx, t1xx_init, (/ nztop, nxtop+3, nytop /))
 call assign3d_real( t1xx_init, 1, 0, 1, t1xx )
 size=nztop*(nxtop+3)*(nytop+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xy, ierror)
 call c_f_pointer(cptr_t1xy, t1xy_init, (/ nztop, nxtop+3, nytop+3 /))
 call assign3d_real( t1xy_init, 1, -1, -1, t1xy )
 size=(nztop+1)*(nxtop+3)*nytop*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1xz, ierror)
 call c_f_pointer(cptr_t1xz, t1xz_init, (/ nztop+1, nxtop+3, nytop /))
 call assign3d_real( t1xz_init, 1, -1, 1, t1xz )
 size=nztop*nxtop*(nytop+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1yy, ierror)
 call c_f_pointer(cptr_t1yy, t1yy_init, (/ nztop, nxtop, nytop+3 /))
 call assign3d_real( t1yy_init, 1, 1, 0, t1yy )
 size=(nztop+1)*nxtop*(nytop+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1yz, ierror)
 call c_f_pointer(cptr_t1yz, t1yz_init, (/ nztop+1, nxtop, nytop+3 /))
 call assign3d_real( t1yz_init, 1, 1, -1, t1yz )
 size=nztop*nxtop*nytop*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t1zz, ierror)
 call c_f_pointer(cptr_t1zz, t1zz, (/ nztop, nxtop, nytop /))

! allocate(qt1xx(nztop,nxtop,nytop), qt1xy(nztop,nxtop,nytop))
! allocate(qt1yy(nztop,nxtop,nytop), qt1zz(nztop,nxtop,nytop))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xx, ierror)
 call c_f_pointer(cptr_qt1xx, qt1xx, (/ nztop, nxtop, nytop /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xy, ierror)
 call c_f_pointer(cptr_qt1xy, qt1xy, (/ nztop, nxtop, nytop /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1yy, ierror)
 call c_f_pointer(cptr_qt1yy, qt1yy, (/ nztop, nxtop, nytop /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1zz, ierror)
 call c_f_pointer(cptr_qt1zz, qt1zz, (/ nztop, nxtop, nytop /))

! allocate(qt1xz(nztop+1,nxtop,nytop), qt1yz(nztop+1,nxtop,nytop))
 size=(nztop+1)*nxtop*nytop*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1xz, ierror)
 call c_f_pointer(cptr_qt1xz, qt1xz, (/ nztop+1, nxtop, nytop /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt1yz, ierror)
 call c_f_pointer(cptr_qt1yz, qt1yz, (/ nztop+1, nxtop, nytop /))
!
! variables in Region II
! allocate(v2x(0:nzbtm,-1:nxbtm+1, 0:nybtm+2))
! allocate(v2y(0:nzbtm, 0:nxbtm+2,-1:nybtm+1))
! allocate(v2z(0:nzbtm, 0:nxbtm+2, 0:nybtm+2))
 size=(nzbtm+1)*(nxbtm+3)*(nybtm+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2x, ierror)
 call c_f_pointer(cptr_v2x, v2x_init, (/ nzbtm+1, nxbtm+3, nybtm+3 /))
 call assign3d_real( v2x_init, 0, -1, 0, v2x )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2y , ierror)
 call c_f_pointer(cptr_v2y, v2y_init, (/ nzbtm+1, nxbtm+3, nybtm+3 /))
 call assign3d_real( v2y_init, 0, 0, -1, v2y )
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_v2z , ierror)
 call c_f_pointer(cptr_v2z, v2z_init, (/ nzbtm+1, nxbtm+3, nybtm+3 /))
 call assign3d_real( v2z_init, 0, 0, 0, v2z )

! allocate(t2xx(  nzbtm, 0:nxbtm+2,   nybtm))
! allocate(t2xy(  nzbtm,-1:nxbtm+1,-1:nybtm+1))
! allocate(t2xz(0:nzbtm,-1:nxbtm+1,   nybtm))
! allocate(t2yy(  nzbtm,   nxbtm,   0:nybtm+2))
! allocate(t2yz(0:nzbtm,   nxbtm,  -1:nybtm+1))
! allocate(t2zz(0:nzbtm,   nxbtm,     nybtm))
 size=nzbtm*(nxbtm+3)*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xx , ierror)
 call c_f_pointer(cptr_t2xx, t2xx_init, (/ nzbtm, nxbtm+3, nybtm /))
 call assign3d_real( t2xx_init, 1, 0, 1, t2xx )
 size=nzbtm*(nxbtm+3)*(nybtm+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xy , ierror)
 call c_f_pointer(cptr_t2xy, t2xy_init, (/ nzbtm, nxbtm+3, nybtm+3 /))
 call assign3d_real( t2xy_init, 1, -1, -1, t2xy )
 size=(nzbtm+1)*(nxbtm+3)*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2xz , ierror)
 call c_f_pointer(cptr_t2xz, t2xz_init, (/ nzbtm+1, nxbtm+3, nybtm /))
 call assign3d_real( t2xz_init, 0, -1, 1, t2xz )
 size=nzbtm*nxbtm*(nybtm+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2yy , ierror)
 call c_f_pointer(cptr_t2yy, t2yy_init, (/ nzbtm, nxbtm, nybtm+3 /))
 call assign3d_real( t2yy_init, 1, 1, 0, t2yy )
 size=(nzbtm+1)*nxbtm*(nybtm+3)*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2yz , ierror)
 call c_f_pointer(cptr_t2yz, t2yz_init, (/ nzbtm+1, nxbtm, nybtm+3 /))
 call assign3d_real( t2yz_init, 0, 1, -1, t2yz )
 size=(nzbtm+1)*nxbtm*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_t2zz , ierror)
 call c_f_pointer(cptr_t2zz, t2zz_init, (/ nzbtm+1, nxbtm, nybtm /))
 call assign3d_real( t2zz_init, 0, 1, 1, t2zz )

!allocate(qt2xx(nzbtm,nxbtm,nybtm), qt2xy(nzbtm,nxbtm,nybtm))
!allocate(qt2xz(nzbtm,nxbtm,nybtm), qt2yy(nzbtm,nxbtm,nybtm))
!allocate(qt2yz(nzbtm,nxbtm,nybtm), qt2zz(nzbtm,nxbtm,nybtm))
 size=nzbtm*nxbtm*nybtm*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xx , ierror)
 call c_f_pointer(cptr_qt2xx, qt2xx, (/ nzbtm, nxbtm, nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xy , ierror)
 call c_f_pointer(cptr_qt2xy, qt2xy, (/ nzbtm, nxbtm, nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2xz , ierror)
 call c_f_pointer(cptr_qt2xz, qt2xz, (/ nzbtm, nxbtm, nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2yy , ierror)
 call c_f_pointer(cptr_qt2yy, qt2yy, (/ nzbtm, nxbtm, nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2yz , ierror)
 call c_f_pointer(cptr_qt2yz, qt2yz, (/ nzbtm, nxbtm, nybtm /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_qt2zz , ierror)
 call c_f_pointer(cptr_qt2zz, qt2zz, (/ nzbtm, nxbtm, nybtm /))
! variable for MPI
! allocate(sdx51(nztop,nytop,5),sdx41(nztop,nytop,4))
! allocate(sdy51(nztop,nxtop,5),sdy41(nztop,nxtop,4))
! allocate(rcx51(nztop,nytop,5),rcx41(nztop,nytop,4))
! allocate(rcy51(nztop,nxtop,5),rcy41(nztop,nxtop,4))
! allocate(sdx52(nzbtm,nybtm,5),sdx42(nzbtm,nybtm,4))
! allocate(sdy52(nzbtm,nxbtm,5),sdy42(nzbtm,nxbtm,4))
! allocate(rcx52(nzbtm,nybtm,5),rcx42(nzbtm,nybtm,4))
! allocate(rcy52(nzbtm,nxbtm,5),rcy42(nzbtm,nxbtm,4))
 size=nztop*nytop*5*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdx51 , ierror)
 call c_f_pointer(cptr_sdx51, sdx51, (/ nztop,nytop,5 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcx51 , ierror)
 call c_f_pointer(cptr_rcx51, rcx51, (/ nztop,nytop,5 /))
 size=nztop*nxtop*5*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdy51 , ierror)
 call c_f_pointer(cptr_sdy51, sdy51, (/ nztop,nxtop,5 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcy51 , ierror)
 call c_f_pointer(cptr_rcy51, rcy51, (/ nztop,nxtop,5 /))
 size=nzbtm*nybtm*5*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdx52 , ierror)
 call c_f_pointer(cptr_sdx52, sdx52, (/ nzbtm,nybtm,5 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcx52 , ierror)
 call c_f_pointer(cptr_rcx52, rcx52, (/ nzbtm,nybtm,5 /))
 size=nzbtm*nxbtm*5*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdy52 , ierror)
 call c_f_pointer(cptr_sdy52, sdy52, (/ nzbtm,nxbtm,5 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcy52 , ierror)
 call c_f_pointer(cptr_rcy52, rcy52, (/ nzbtm,nxbtm,5 /))
 size=nztop*nytop*4*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdx41 , ierror)
 call c_f_pointer(cptr_sdx41, sdx41, (/ nztop,nytop,4 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcx41 , ierror)
 call c_f_pointer(cptr_rcx41, rcx41, (/ nztop,nytop,4 /))
 size=nztop*nxtop*4*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdy41 , ierror)
 call c_f_pointer(cptr_sdy41, sdy41, (/ nztop,nxtop,4 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcy41 , ierror)
 call c_f_pointer(cptr_rcy41, rcy41, (/ nztop,nxtop,4 /))
 size=nzbtm*nybtm*4*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdx42 , ierror)
 call c_f_pointer(cptr_sdx42, sdx42, (/ nzbtm,nybtm,4 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcx42 , ierror)
 call c_f_pointer(cptr_rcx42, rcx42, (/ nzbtm,nybtm,4 /))
 size=nzbtm*nxbtm*4*real_itemsize
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_sdy42 , ierror)
 call c_f_pointer(cptr_sdy42, sdy42, (/ nzbtm,nxbtm,4 /))
 call MPI_Alloc_mem(size, MPI_INFO_NULL, cptr_rcy42 , ierror)
 call c_f_pointer(cptr_rcy42, rcy42, (/ nzbtm,nxbtm,4 /))
 return
end subroutine 
!
!-----------------------------------------------------------------------
subroutine set_node_grid
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i
!
 nxtm1=nxtop-1
 nytm1=nytop-1
 nztm1=nztop-1
 nxbm1=nxbtm-1
 nybm1=nybtm-1
 nzbm1=nzbtm-1
 nz1p1=nztop+1
 ny1p1=nytop+1
 ny1p2=nytop+2
 ny2p1=nybtm+1
 ny2p2=nybtm+2
 nx1p1=nxtop+1
 nx1p2=nxtop+2
 nx2p1=nxbtm+1
 nx2p2=nxbtm+2
 nzrg1(1)=nztop
 nzrg1(2)=nztop
 nzrg1(3)=nztop+1
 nzrg1(4)=nztop+1
!
 nd1_vel(1)  =1
 nd1_vel(2)  =0
 nd1_vel(3)  =1
 nd1_vel(4)  = nytop
 nd1_vel(5)  = nytop+1
 nd1_vel(6)  = nytop
 nd1_vel(7)  = 1
 nd1_vel(8)  = 0
 nd1_vel(9)  = 1
 nd1_vel(10) = nxtop
 nd1_vel(11) = nxtop+1
 nd1_vel(12) = nxtop
 nd1_vel(13) = 1
 nd1_vel(14) = 0
 nd1_vel(15) = 1
 nd1_vel(16) = nztop
 nd1_vel(17) = nztop+1
 nd1_vel(18) = nztop

 nd2_vel(1)  = 1
 nd2_vel(2)  = 0
 nd2_vel(3)  = 1
 nd2_vel(4)  = nybtm
 nd2_vel(5)  = nybtm+1
 nd2_vel(6)  = nybtm
 nd2_vel(7)  = 1
 nd2_vel(8)  = 0
 nd2_vel(9)  = 1
 nd2_vel(10) = nxbtm
 nd2_vel(11) = nxbtm+1
 nd2_vel(12) = nxbtm
 nd2_vel(13) = 1
 nd2_vel(14) = 0
 nd2_vel(15) = 1
 nd2_vel(16) = nzbtm-mw2_pml1
 nd2_vel(17) = nd2_vel(16)+1
 nd2_vel(18) = nzbtm-1
 do i=1,18
   nd1_tyy(i)=nd1_vel(i)
   nd1_txy(i)=nd1_vel(i)
   nd1_txz(i)=nd1_vel(i)
   nd1_tyz(i)=nd1_vel(i)
   nd2_tyy(i)=nd2_vel(i)
   nd2_txy(i)=nd2_vel(i)
   nd2_txz(i)=nd2_vel(i)
   nd2_tyz(i)=nd2_vel(i)
 enddo
 nd1_txz(13)=2
 nd1_tyz(13)=2
 nd1_txz(18)=nztop+1
 nd1_tyz(18)=nztop+1
 nd2_txz(13)=2
 nd2_tyz(13)=2
 nd2_txz(18)=nzbtm
 nd2_tyz(18)=nzbtm
 if(neighb(1) < 0) then
   nd1_vel(7) =2
   nd1_tyy(7) =2
   nd1_txy(7) =1
   nd1_txz(7) =1
   nd1_tyz(7) =2
   nd2_vel(7) =2
   nd2_tyy(7) =2
   nd2_txy(7) =1
   nd2_txz(7) =1
   nd2_tyz(7) =2
   nd1_vel(8) = mw1_pml1
   nd1_vel(9) = mw1_pml+2
   nd2_vel(8) = mw2_pml1
   nd2_vel(9) = mw2_pml+2
 endif
 if(neighb(2) < 0) then
   nd1_vel(10) =nxtop-mw1_pml1
   nd1_vel(11) =nxtop-mw1_pml
   nd2_vel(10) =nxbtm-mw2_pml1
   nd2_vel(11) =nxbtm-mw2_pml
   nd1_vel(12) =nxtm1
   nd1_tyy(12) =nxtop
   nd1_txy(12) =nxtm1
   nd1_txz(12) =nxtm1
   nd1_tyz(12) =nxtm1
   nd2_vel(12) =nxbm1
   nd2_tyy(12) =nxbtm
   nd2_txy(12) =nxbm1
   nd2_txz(12) =nxbm1
   nd2_tyz(12) =nxbm1
 endif
 if(neighb(3) < 0) then
   nd1_vel(1) =2
   nd1_tyy(1) =2
   nd1_txy(1) =1
   nd1_txz(1) =2
   nd1_tyz(1) =1
   nd2_vel(1) =2
   nd2_tyy(1) =2
   nd2_txy(1) =1
   nd2_txz(1) =2
   nd2_tyz(1) =1
   nd1_vel(2) = mw1_pml1
   nd1_vel(3) = mw1_pml+2
   nd2_vel(2) = mw2_pml1
   nd2_vel(3) = mw2_pml+2
 endif
 if(neighb(4) < 0) then
   nd1_vel(4) = nytop-mw1_pml1
   nd1_vel(5) = nytop-mw1_pml
   nd2_vel(4) = nybtm-mw2_pml1
   nd2_vel(5) = nybtm-mw2_pml
   nd1_vel(6) = nytm1
   nd1_tyy(6) = nytop
   nd1_txy(6) = nytm1
   nd1_txz(6) = nytm1
   nd1_tyz(6) = nytm1
   nd2_vel(6) = nybm1
   nd2_tyy(6) = nybtm
   nd2_txy(6) = nybm1
   nd2_txz(6) = nybm1
   nd2_tyz(6) = nybm1
 endif
 do i=2,11
   if(i<6 .or. i>7) then
     nd1_tyy(i) = nd1_vel(i)
     nd1_txy(i) = nd1_vel(i)
     nd1_txz(i) = nd1_vel(i)
     nd1_tyz(i) = nd1_vel(i)
     nd2_tyy(i) = nd2_vel(i)
     nd2_txy(i) = nd2_vel(i)
     nd2_txz(i) = nd2_vel(i)
     nd2_tyz(i) = nd2_vel(i)
   endif
 enddo
 lbx(1)=0
 lbx(2)=-1
 if(neighb(1)<0 .and. neighb(2)<0) then
   lbx(2)=1
 else if(neighb(1)<0 ) then
   lbx(2)=0
 else if(neighb(2)<0 ) then
   lbx(1)=1
   lbx(2)=1
 endif

 lby(1)=0
 lby(2)=-1
 if(neighb(3)<0 .and. neighb(4)<0) then
   lby(2)=1
 else if(neighb(3)<0 ) then
   lby(2)=0
 else if(neighb(4)<0 ) then
   lby(1)=1
   lby(2)=1
 endif
 return
end subroutine set_node_grid
!
!-----------------------------------------------------------------------
subroutine Pml_coef(myid,dt)
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer, intent(IN):: myid
 real, intent(IN):: dt
 integer:: i,j,k,idx,ij,ia,nm,b1,b2,nabc=4, npow=2
 real:: vp,vs,gridsp,damp0
 real, dimension(4):: cfabc
!
! PML-X-I
 b1=lbx(1)
 b2=lbx(2)
 if(b2 >= b1) then
   do i=b1,b2
     gridsp=gridx2(i*(nxbtm-2)+2)-gridx2(i*(nxbtm-2)+1)
     damp0=0.5*dt*alog(1./RF)*(npow+1.)/2./(mw2_pml*gridsp)
     do j=1,nytop
     do k=1,nztop+1
       idx=idmat1(k,i*nxtm1+1,j)
       vp=sqrt((clamda(idx)+2.*cmu(idx))*rho(idx))
       vs=sqrt(cmu(idx)*rho(idx))
       damp1_x(k,j,i)=damp0*vp
       if(k > nztop) cycle
       call coef_abc(vs,gridsp/3.0,dt,damp0*vp,cfabc)
       abcx1(i*4+1:i*4+4, k, j)=cfabc
     enddo
     enddo
     do j=1,nybtm
     do k=1,nzbtm
       idx=idmat2(k,i*nxbm1+1,j)
       vp=sqrt((clamda(idx)+2.*cmu(idx))*rho(idx))
       vs=sqrt(cmu(idx)*rho(idx))
       damp2_x(k,j,i)=damp0*vp
       call coef_abc(vs,gridsp,dt,damp0*vp,cfabc)
       abcx2(i*4+1:i*4+4, k, j)=cfabc
     enddo
     enddo
   enddo
 endif
! PML-Y-I
 b1=lby(1)
 b2=lby(2)
 if(b2 >= b1) then
   do j=b1,b2
     gridsp=gridy2(j*(nybtm-2)+2)-gridy2(j*(nybtm-2)+1)
     damp0=0.5*dt*alog(1./RF)*(npow+1.)/2./(mw2_pml*gridsp)
     do i=1,nxtop
     do k=1,nztop+1
       idx=idmat1(k,i,j*nytm1+1)
       vp=sqrt((clamda(idx)+2.*cmu(idx))*rho(idx))
       vs=sqrt(cmu(idx)*rho(idx))
       damp1_y(k,i,j)=damp0*vp
       if(k > nztop) cycle
       call coef_abc(vs,gridsp/3.0,dt,damp0*vp,cfabc)
       abcy1(j*4+1:j*4+4, k, i)=cfabc
     enddo
     enddo
     do i=1,nxbtm
     do k=1,nzbtm
       idx=idmat2(k,i,j*nybm1+1)
       vp=sqrt((clamda(idx)+2.*cmu(idx))*rho(idx))
       vs=sqrt(cmu(idx)*rho(idx))
       damp2_y(k,i,j)=damp0*vp
       call coef_abc(vs,gridsp,dt,damp0*vp,cfabc)
       abcy2(j*4+1:j*4+4, k, i)=cfabc
     enddo
     enddo
   enddo
 endif
! PML-Z-II
 gridsp=gridz(nzbtm)-gridz(nzbtm-1)
 damp0=0.5*dt*alog(1./RF)*(npow+1.)/2./(mw2_pml*gridsp)
 do j=1,nybtm
 do i=1,nxbtm
   idx=idmat2(nzbm1,i,j)
   vp=sqrt((clamda(idx)+2.*cmu(idx))*rho(idx))
   vs=sqrt(cmu(idx)*rho(idx))
   damp2_z(i,j)=damp0*vp
   call coef_abc(vs,gridsp,dt,damp0*vp,cfabc)
   abcz(:, i, j)=cfabc
 enddo
 enddo
!
 drti1=0.0
 drth1=0.0
 drvh1=0.0
 drti2=0.0
 drth2=0.0
 drvh2=0.0
 do i=1,mw1_pml-2
   drti1(i,0)=(1.0-      i/(mw1_pml-1.5))**npow
   drth1(i,0)=(1.0-(i-0.5)/(mw1_pml-1.5))**npow
   drvh1(i,0)=(1.0-(i+0.5)/(mw1_pml-1.5))**npow
 enddo
 do i=1,mw2_pml-1
   drti2(i,0)=(1.0-      i/(mw2_pml-0.5))**npow
   drth2(i,0)=(1.0-(i-0.5)/(mw2_pml-0.5))**npow
   drvh2(i,0)=(1.0-(i+0.5)/(mw2_pml-0.5))**npow
 enddo
 do i=4,mw1_pml1
   drti1(i,1)=((i-4.0)/(mw1_pml-3.))**npow
   drth1(i,1)=((i-3.5)/(mw1_pml-3.))**npow
   drvh1(i,1)=((i-3.5)/(mw1_pml-3.))**npow
 enddo   
 do i=2,mw2_pml1
   drti2(i,1)=((i-2.0)/(mw2_pml-1.))**npow
   drth2(i,1)=((i-1.5)/(mw2_pml-1.))**npow
   drvh2(i,1)=((i-1.5)/(mw2_pml-1.))**npow
 enddo   
!   determine the parameters of absorbing boundary condition
 do i=1,15,2
   nxyabc(i) =1
   nxyabc(i+1) =0
 enddo

 if(neighb(3) < 0) then
   nxyabc(6)  = nxtop
   nxyabc(14) = nxbtm
 endif
 if(neighb(4) < 0) then
   nxyabc(8)  = nxtop
   nxyabc(16) = nxbtm
 endif

 if(neighb(1) < 0) then
   nxyabc(2)  =nytop
   nxyabc(10) =nybtm
   nxyabc(5)  =2
   nxyabc(7)  =2
   nxyabc(13) =2
   nxyabc(15) =2   
 endif
 if(neighb(2) < 0) then
   nxyabc(4)  =nytop
   nxyabc(12)  =nybtm
   nxyabc(6)  =nxyabc(6)-1
   nxyabc(8)  =nxyabc(8)-1
   nxyabc(14) =nxyabc(14)-1
   nxyabc(16) =nxyabc(16)-1
 endif
 return
end subroutine Pml_coef
!

!!-----------------------------------------------------------------------
!subroutine update_velocity(comm_worker,myid)
!! Updating the velocity field with 2-nd and 4th-order FD scheme
! use grid_node_comm
! use wave_field_comm
! implicit NONE
! integer, intent(IN):: comm_worker,myid
!!
!! call pre_abc
!!  Compute the velocity field in the region I
! call velocity_inner_I
! call vel_PmlX_I
! call vel_PmlY_I
!!  Compute the velocity field in the region II
! call velocity_inner_II
! call vel_PmlX_II
! call vel_PmlY_II
! call vel_PmlZ_II
!! Interpolating velocity on the interface betwee Region I and II
! call ISEND_IRECV_Velocity(comm_worker)
! call velocity_interp(comm_worker)
! call vxy_image_layer(comm_worker)
! return
!end subroutine update_velocity
!
!XSC--------------------------------------------------------------------
subroutine compute_velocity
 use grid_node_comm
 use wave_field_comm
 implicit NONE
!
! call pre_abc
!  Compute the velocity field in the region I
 call velocity_inner_I
 call vel_PmlX_I
 call vel_PmlY_I
!  Compute the velocity field in the region II
 call velocity_inner_II
 call vel_PmlX_II
 call vel_PmlY_II
 call vel_PmlZ_II
 return
end subroutine compute_velocity

subroutine comm_velocity(comm_worker, myid)
 use grid_node_comm
 use wave_field_comm
 use iso_c_binding
 use ctimer
 implicit NONE
 integer, intent(IN):: comm_worker, myid
 real(c_double) :: tstart, tend

! Interpolating velocity on the interface betwee Region I and II
 call record_time(tstart)
 call ISEND_IRECV_Velocity(comm_worker)
 call record_time(tend)
 write(*,*) "TIME :: ISEND_IRECV_Velocity :", tend-tstart
! write(*,*) ' after call call ISEND_IRECV_Velocity(comm_worker)'

 call record_time(tstart)
 call velocity_interp(comm_worker)
 call record_time(tend)
 write(*,*) "TIME :: velocity_interp :", tend-tstart

! write(*,*) ' after call velocity_interp(comm_worker)'

 call record_time(tstart)
 call vxy_image_layer(comm_worker)
 call record_time(tend)
 write(*,*) "TIME :: vxy_image_layer :", tend-tstart

! write(*,*) ' after call vxy_image_layer(comm_worker)'
 return
end subroutine comm_velocity

!-----------------------------------------------------------------------
subroutine vxy_image_layer(comm_worker)
 use mpi
 use grid_node_comm
 use wave_field_comm
 use itface_comm
 use:: iso_c_binding
! use logging
 use metadata
 use ctimer
 implicit NONE
 interface
!     subroutine vxy_image_layer_vel1 (v1x, v1y, v1z, v2x, v2y, v2z, &
!                                        neighb1, neighb2, neighb3, neighb4, &
!                                        nxtm1, nytm1, nxtop, nytop, nztop,&
!                                         nxbtm, nybtm, nzbtm) &
!       bind(C, name="vxy_image_layer_vel1")
!        use:: iso_c_binding
!        type(c_ptr), intent(in), value :: v1x, v1y, v1z, v2x, v2y, v2z
!        integer(c_int) :: nxtm1, nytm1, nxtop, nytop, nztop
!        integer(c_int), intent(in), value:: neighb1, neighb2, neighb3, neighb4 
!     end subroutine vxy_image_layer_vel1

     subroutine vxy_image_layer_vel1 ( nd1_vel, i, dzdx, &
                                       nxbtm, nybtm, nzbtm, &
                                       nxtop, nytop, nztop)&
       bind(C, name="vxy_image_layer_vel1")
        use:: iso_c_binding
        integer(c_int), intent(in), value :: i, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm
        real(c_float), intent(in), value:: dzdx
        integer(c_int), dimension(*), intent(in) :: nd1_vel
     end subroutine vxy_image_layer_vel1
 
     subroutine vxy_image_layer_vel2 ( nd1_vel, v1x, iix, dzdt, &
                                       nxbtm, nybtm, nzbtm, &
                                       nxtop, nytop, nztop)&
       bind(C, name="vxy_image_layer_vel2")
        use:: iso_c_binding
        integer(c_int) :: iix, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm
        real(c_float) :: dzdt
        integer(c_int), dimension(*), intent(in) :: nd1_vel
        type(c_ptr), intent(in), value :: v1x
     end subroutine vxy_image_layer_vel2
 
     subroutine vxy_image_layer_vel3 ( nd1_vel, j, dzdy, &
                                       nxbtm, nybtm, nzbtm, &
                                       nxtop, nytop, nztop)&
       bind(C, name="vxy_image_layer_vel3")
        use:: iso_c_binding
        integer(c_int) :: j, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm
        real(c_float) :: dzdy
        integer(c_int), dimension(*), intent(in) :: nd1_vel
     end subroutine vxy_image_layer_vel3
 
     subroutine vxy_image_layer_vel4 ( nd1_vel, v1y, jjy, dzdt, &
                                       nxbtm, nybtm, nzbtm, &
                                       nxtop, nytop, nztop)&
       bind(C, name="vxy_image_layer_vel4")
        use:: iso_c_binding
        integer(c_int) :: jjy, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm
        real(c_float) :: dzdt
        integer(c_int), dimension(*), intent(in) :: nd1_vel
        type(c_ptr), intent(in), value :: v1y
     end subroutine vxy_image_layer_vel4
  
     subroutine vxy_image_layer_sdx_vel(sdx1, sdx2, nxtop, nytop, nztop) &
         bind (C, name="vxy_image_layer_sdx_vel")
         use:: iso_c_binding
        type(c_ptr), intent(in), value:: sdx1, sdx2
        integer(c_int) , intent(in) :: nytop, nxtop, nztop
     end subroutine vxy_image_layer_sdx_vel

     subroutine vxy_image_layer_sdy_vel(sdy1, sdy2, nxtop, nytop, nztop) &
         bind (C, name="vxy_image_layer_sdy_vel")
         use:: iso_c_binding
        type(c_ptr), intent(in), value:: sdy1, sdy2
        integer(c_int) , intent(in) :: nxtop, nztop, nytop
     end subroutine vxy_image_layer_sdy_vel

     subroutine vxy_image_layer_rcx_vel(rcx1, rcx2, nxtop, nytop, nztop, nx1p1) &
         bind (C, name="vxy_image_layer_rcx_vel")
         use:: iso_c_binding
        type(c_ptr), intent(in), value::rcx1, rcx2
        integer(c_int), intent(in) :: nytop, nx1p1, nxtop, nztop
     end subroutine vxy_image_layer_rcx_vel

     subroutine vxy_image_layer_rcx2_vel(rcy1, rcy2, nxtop, nytop, nztop, ny1p1) &
         bind (C, name="vxy_image_layer_rcx2_vel")
         use:: iso_c_binding
        type(c_ptr), intent(in), value:: rcy1, rcy2
        integer(c_int) , intent(in) :: nxtop, ny1p1, nytop, nztop
     end subroutine vxy_image_layer_rcx2_vel

     subroutine getsizeof_float(sizeof_) bind(C, name='getsizeof_float')
        use::iso_c_binding
        integer(c_int), intent(out) :: sizeof_
     end subroutine getsizeof_float

    include "mpiacc_wrappers.h"
 end interface

 integer, intent(IN):: comm_worker
 integer:: i,j,k,iix,jjy,nnum,ierr
 integer, dimension(MPI_STATUS_SIZE):: status
 integer:: req(8),istatus(MPI_STATUS_SIZE,8),nr
 integer:: reqX(8),istatusX(MPI_STATUS_SIZE,8)
 integer::sizeof_float
 real:: time_marsh=0.0
 real(c_double) :: gpu_tstart, gpu_tend, cpu_tstart, cpu_tend
 real(c_double):: total_gpu_marsh_time=0.0, total_cpu_marsh_time=0.0
 real:: dzdt,dzdx,dzdy
 real, dimension(5):: sendv,recev
 total_gpu_marsh_time =0.0
 total_cpu_marsh_time =0.0
! finding X- and Y-componet velocity on the image layer.
 dzdt=ca/dzh1(3,1)
!--Vx
 iix=nd1_vel(12)
 if(neighb(1) < 0 .or. neighb(2) < 0) then
   i=1
   if(neighb(2) < 0) then
     iix=nxtop-2
     i=nxtm1
   endif
   dzdx=dzdt*dxi1(3,i)/ca
#ifdef DISFD_GPU_MARSHALING
 if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call vxy_image_layer_vel1(nd1_vel, i, dzdx, nxbtm, nybtm, nzbtm, nxtop, nytop, nztop);
 if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vel1_vxy_image_layer", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=nd1_vel(1),nd1_vel(6)
     v1x(0,i,j)=v1x(1,i,j)+dzdx*(v1z(1,i+1,j)-v1z(1,i,j))
   enddo
#endif
 endif
#ifdef DISFD_GPU_MARSHALING
 if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call vxy_image_layer_vel2(nd1_vel, cptr_v1x, iix, dzdt, nxbtm, nybtm, nzbtm, nxtop, nytop, nztop);
 if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vel2_vxy_image_layer", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
 do j=nd1_vel(1),nd1_vel(6)
 do i=nd1_vel(7),iix
     v1x(0,i,j)=v1x(1,i,j)+dzdt* &
               (dxi1(1,i)*v1z(1,i-1,j)+dxi1(2,i)*v1z(1,i,  j)+ &
                dxi1(3,i)*v1z(1,i+1,j)+dxi1(4,i)*v1z(1,i+2,j))
 enddo
 enddo
#endif
!---Vy
 jjy=nd1_vel(6)
 if(neighb(3) < 0 .or. neighb(4) < 0) then
   j=1
   if(neighb(4) < 0) then
     jjy=nytop-2
     j=nytm1
   endif
   dzdy=dzdt*dyi1(3,j)/ca
#ifdef DISFD_GPU_MARSHALING
 if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call vxy_image_layer_vel3(nd1_vel, j, dzdy, nxbtm, nybtm, nzbtm, nxtop, nytop, nztop);
 if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vel3_vxy_image_layer", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=nd1_vel(7),nd1_vel(12)
     v1y(0,i,j)=v1y(1,i,j)+dzdy*(v1z(1,i,j+1)-v1z(1,i,j))
   enddo
#endif
 endif
#ifdef DISFD_GPU_MARSHALING
 if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call vxy_image_layer_vel4(nd1_vel, cptr_v1y, jjy, dzdt, nxbtm, nybtm, nzbtm, nxtop, nytop, nztop);
 if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vxy_image_layer_vel4", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
 do j=nd1_vel(1),jjy
 do i=nd1_vel(7),nd1_vel(12)
     v1y(0,i,j)=v1y(1,i,j)+dzdt* &
               (dyi1(1,j)*v1z(1,i,j-1)+dyi1(2,j)*v1z(1,i,j)+ &
                dyi1(3,j)*v1z(1,i,j+1)+dyi1(4,j)*v1z(1,i,j+2))
 enddo
 enddo
#endif
!
 nnum=nztop+2

#define v1xD(i, j, k) c_v1x_id,((i) + (nztop + 2) * ((j) + 1 + (k) * (nxtop + 3)))
#define v1yD(i, j, k) c_v1y_id,((i) + (nztop + 2) * ((j) + ((k) + 1) * (nxtop + 3)))
#define v1zD(i, j, k) c_v1z_id,((i) + (nztop + 2) * ((j) + (k) * (nxtop + 3)))
#define v2xD(i, j, k) c_v2x_id,((i) + (nzbtm + 1) * ((j) + 1 + (nxbtm + 3) * (k)))
#define v2yD(i, j, k) c_v2y_id,((i) + (nzbtm + 1) * ((j) + (nxbtm + 3) * ((k) + 1)))
#define v2zD(i, j, k) c_v2z_id,((i) + (nzbtm + 1) * ((j) + (nxbtm + 3) * (k)))


 select case(order_imsr) 
 case(1) 
#if USE_MPIX == 1
   call MPIX_RECV_offset_C(v1xD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,211,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v1yD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,221,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v1zD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,231,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v2xD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,241,comm_worker,status(1),ierr, 1)
   call MPIX_RECV_offset_C(v2yD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,251,comm_worker,status(1),ierr, 1)
   call MPIX_RECV_offset_C(v2zD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                pid_imr,261,comm_worker,status(1),ierr, 1)
#else
   call MPI_RECV(v1x(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,21,comm_worker,status,ierr)
   call MPI_RECV(v1y(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,22,comm_worker,status,ierr)
   call MPI_RECV(v1z(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,23,comm_worker,status,ierr)
   call MPI_RECV(v2x(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,24,comm_worker,status,ierr)
   call MPI_RECV(v2y(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,25,comm_worker,status,ierr)
   call MPI_RECV(v2z(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,26,comm_worker,status,ierr)
#endif
 case(2)
#if USE_MPIX ==1
   call MPIX_RECV_offset_C(v1xD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,211,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v1yD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,221,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v1zD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,231,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v2xD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,241,comm_worker,status(1),ierr, 1)
   call MPIX_RECV_offset_C(v2yD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,251,comm_worker,status(1),ierr, 1)
   call MPIX_RECV_offset_C(v2zD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,261,comm_worker,status(1),ierr, 1)
   call MPIX_SEND_offset_C(v1xD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 211, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v1yD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 221, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v1zD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 231, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v2xD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 241, comm_worker, ierr, 1)
   call MPIX_SEND_offset_C(v2yD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 251, comm_worker, ierr, 1)
   call MPIX_SEND_offset_C(v2zD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 261, comm_worker, ierr, 1)
#else
   call MPI_RECV(v1x(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,21,comm_worker,status,ierr)
   call MPI_RECV(v1y(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,22,comm_worker,status,ierr)
   call MPI_RECV(v1z(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,23,comm_worker,status,ierr)
   call MPI_RECV(v2x(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,24,comm_worker,status,ierr)
   call MPI_RECV(v2y(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,25,comm_worker,status,ierr)
   call MPI_RECV(v2z(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,26,comm_worker,status,ierr)
   call MPI_SEND(v1x(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 21, comm_worker, ierr)
   call MPI_SEND(v1y(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 22, comm_worker, ierr)
   call MPI_SEND(v1z(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 23, comm_worker, ierr)
   call MPI_SEND(v2x(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 24, comm_worker, ierr)
   call MPI_SEND(v2y(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 25, comm_worker, ierr)
   call MPI_SEND(v2z(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 26, comm_worker, ierr)
#endif
 case(3) 
#if USE_MPIX ==1 
   call MPIX_SEND_offset_C(v1xD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 211, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v1yD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 221, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v1zD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 231, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v2xD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 241, comm_worker, ierr, 1)
   call MPIX_SEND_offset_C(v2yD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 251, comm_worker, ierr, 1)
   call MPIX_SEND_offset_C(v2zD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 261, comm_worker, ierr, 1)
   call MPIX_RECV_offset_C(v1xD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,211,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v1yD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,221,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v1zD(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,231,comm_worker,status(1),ierr, 0)
   call MPIX_RECV_offset_C(v2xD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,241,comm_worker,status(1),ierr, 1)
   call MPIX_RECV_offset_C(v2yD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,251,comm_worker,status(1),ierr, 1)
   call MPIX_RECV_offset_C(v2zD(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,261,comm_worker,status(1),ierr, 1)
#else
   call MPI_SEND(v1x(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 21, comm_worker, ierr)
   call MPI_SEND(v1y(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 22, comm_worker, ierr)
   call MPI_SEND(v1z(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 23, comm_worker, ierr)
   call MPI_SEND(v2x(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 24, comm_worker, ierr)
   call MPI_SEND(v2y(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 25, comm_worker, ierr)
   call MPI_SEND(v2z(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 26, comm_worker, ierr)
   call MPI_RECV(v1x(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,21,comm_worker,status,ierr)
   call MPI_RECV(v1y(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,22,comm_worker,status,ierr)
   call MPI_RECV(v1z(0,nx1p1,ny1p1),nnum, MPI_REAL, &
                 pid_imr,23,comm_worker,status,ierr)
   call MPI_RECV(v2x(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,24,comm_worker,status,ierr)
   call MPI_RECV(v2y(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,25,comm_worker,status,ierr)
   call MPI_RECV(v2z(0,nx2p1,ny2p1),nzbtm, MPI_REAL, &
                 pid_imr,26,comm_worker,status,ierr)
#endif
 case(4) 
#if USE_MPIX ==1
   call MPIX_SEND_offset_C(v1xD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 211, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v1yD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 221, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v1zD(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 231, comm_worker, ierr, 0)
   call MPIX_SEND_offset_C(v2xD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 241, comm_worker, ierr, 1)
   call MPIX_SEND_offset_C(v2yD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 251, comm_worker, ierr, 1)
   call MPIX_SEND_offset_C(v2zD(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 261, comm_worker, ierr, 1)
#else
    call MPI_SEND(v1x(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 21, comm_worker, ierr)
   call MPI_SEND(v1y(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 22, comm_worker, ierr)
   call MPI_SEND(v1z(0,1,1), nnum, MPI_REAL, &
                 pid_ims, 23, comm_worker, ierr)
   call MPI_SEND(v2x(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 24, comm_worker, ierr)
   call MPI_SEND(v2y(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 25, comm_worker, ierr)
   call MPI_SEND(v2z(0,1,1), nzbtm, MPI_REAL, &
                 pid_ims, 26, comm_worker, ierr)
#endif
 end select
!
 nr=0
! SEND
 if(neighb(1) > -1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call vxy_image_layer_sdx_vel(c_loc(sdx1), c_loc(sdx2), nxtop, nytop, nztop)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vxy_image_layer_sdx_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdx1(1:nytop)=v1x(0,1,1:nytop)
   sdx2(1:nytop)=v1y(0,1,1:nytop)
#endif
   nr=nr+1
#if USE_MPIX==1
  call MPIX_ISEND_C(c_sdx1_id, nytop, MPI_REAL, &
                  neighb(1), 111, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdx1(1), nytop, MPI_REAL, &
                  neighb(1), 11, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
#if USE_MPIX==1
   call MPIX_ISEND_C(c_sdx2_id, nytop, MPI_REAL, &
                  neighb(1), 121, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdx2(1), nytop, MPI_REAL, &
                  neighb(1), 12, comm_worker, req(nr), ierr)
#endif
 endif
 if(neighb(3) > -1) then
#ifdef DISFD_GPU_MARSHALING
    if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call vxy_image_layer_sdy_vel(c_loc(sdy1), c_loc(sdy2), nxtop, nytop, nztop)
    if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vxy_image_layer_sdy_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdy1(1:nxtop)=v1x(0,1:nxtop,1)
   sdy2(1:nxtop)=v1y(0,1:nxtop,1)
#endif
   nr=nr+1
#if USE_MPIX==1
   call MPIX_ISEND_C(c_sdy1_id, nxtop, MPI_REAL, &
                  neighb(3), 331, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdy1(1), nxtop, MPI_REAL, &
                  neighb(3), 33, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
#if USE_MPIX==1
   call MPIX_ISEND_C(c_sdy2_id, nxtop, MPI_REAL, &
                  neighb(3), 341, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdy2(1), nxtop, MPI_REAL, &
                  neighb(3), 34, comm_worker, req(nr), ierr)
#endif
 endif
! RECV
 if(neighb(2) > -1) then
   nr=nr+1
#if USE_MPIX==1
   call MPIX_IRECV_C(c_rcx1_id, nytop, MPI_REAL, &
                  neighb(2), 111, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcx1(1), nytop, MPI_REAL, &
                  neighb(2), 11, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
#if USE_MPIX==1
   call MPIX_IRECV_C(c_rcx2_id, nytop, MPI_REAL, &
                  neighb(2), 121, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcx2(1), nytop, MPI_REAL, &
                  neighb(2), 12, comm_worker, req(nr), ierr)
#endif
 endif
 if(neighb(4) > -1) then
   nr=nr+1
#if USE_MPIX==1
   call MPIX_IRECV_C(c_rcy1_id, nxtop, MPI_REAL, &
                  neighb(4), 331, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcy1(1), nxtop, MPI_REAL, &
                  neighb(4), 33, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
#if USE_MPIX==1
   call MPIX_IRECV_C(c_rcy2_id, nxtop, MPI_REAL, &
                  neighb(4), 341, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcy2(1), nxtop, MPI_REAL, &
                  neighb(4), 34, comm_worker, req(nr), ierr)
#endif
 endif
#if USE_MPIX==1
 call MPI_WAITALL(nr,reqX,istatusX,ierr)
#else
 call MPI_WAITALL(nr,req,istatus,ierr)
#endif
 if(neighb(2) > -1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call vxy_image_layer_rcx_vel(c_loc(rcx1), c_loc(rcx2), nxtop, nytop, nztop, nx1p1)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vxy_image_layer_rcx_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

#else
   v1x(0,nx1p1,1:nytop)=rcx1(1:nytop)
   v1y(0,nx1p1,1:nytop)=rcx2(1:nytop)
#endif
 endif
 if(neighb(4) > -1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call vxy_image_layer_rcx2_vel(c_loc(rcx1), c_loc(rcx2), nxtop, nytop, nztop, ny1p1)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU vxy_image_layer_rcx2_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

#else
   v1x(0,1:nxtop,ny1p1)=rcx1(1:nxtop)
   v1y(0,1:nxtop,ny1p1)=rcx2(1:nxtop)
#endif
 endif
 return
end subroutine vxy_image_layer
!
!-----------------------------------------------------------------------
subroutine velocity_inner_I
!  Compute the velocity at region I
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,k3
 real:: dtxz,dtyz,dtzz
!
 do j=nd1_vel(3),nd1_vel(4)
 do i=nd1_vel(9),nd1_vel(10)
   do k3=1,3
     k=k3
     if(k3==3) k=nztop
     if(k==1) then
       dtxz=(dzi1(3,k)-dzi1(1,k))*t1xz(2,i,j)+dzi1(4,k)*t1xz(3,i,j)
       dtyz=(dzi1(3,k)-dzi1(1,k))*t1yz(2,i,j)+dzi1(4,k)*t1yz(3,i,j)
       dtzz=dzh1(3,k)/ca*(35./8.*t1zz(k,i,j) -35./24.*t1zz(k+1,i,j)+ &
                  21./40.*t1zz(k+2,i,j)-5./56.*t1zz(k+3,i,j))
     else if(k==2) then
       dtxz=dzi1(2,k)*t1xz(2,i,j)+dzi1(3,k)*t1xz(3,i,j)+dzi1(4,k)*t1xz(4,i,j)
       dtyz=dzi1(2,k)*t1yz(2,i,j)+dzi1(3,k)*t1yz(3,i,j)+dzi1(4,k)*t1yz(4,i,j)
       dtzz=dzh1(3,k)/ca*(-31./24.*t1zz(k-1,i,j) +29./24.*t1zz(k,i,j)- &
                   3./40.*t1zz(k+1,i,j)+1./168.*t1zz(k+2,i,j))
     else
       dtxz=dzi1(2,k)/ca*(t1xz(k,i,j)-t1xz(k+1,i,j))
       dtyz=dzi1(2,k)/ca*(t1yz(k,i,j)-t1yz(k+1,i,j))
       dtzz=dzh1(2,k)/ca*(t1zz(k-1,i,j)-t1zz(k,i,j))
     endif
     v1x(k,i,j)=v1x(k,i,j)+ &
                0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i+1,j)))* &
               (dxi1(1,i)*t1xx(k,i-1,j)+dxi1(2,i)*t1xx(k,i,  j)+ &
                dxi1(3,i)*t1xx(k,i+1,j)+dxi1(4,i)*t1xx(k,i+2,j)+ &
                dyh1(1,j)*t1xy(k,i,j-2)+dyh1(2,j)*t1xy(k,i,j-1)+ &
                dyh1(3,j)*t1xy(k,i,j  )+dyh1(4,j)*t1xy(k,i,j+1)+dtxz)
     v1y(k,i,j)=v1y(k,i,j)+ &
                0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i,j+1)))* &
               (dxh1(1,i)*t1xy(k,i-2,j)+dxh1(2,i)*t1xy(k,i-1,j)+ &
                dxh1(3,i)*t1xy(k,i,  j)+dxh1(4,i)*t1xy(k,i+1,j)+ &
                dyi1(1,j)*t1yy(k,i,j-1)+dyi1(2,j)*t1yy(k,i,j  )+ &
                dyi1(3,j)*t1yy(k,i,j+1)+dyi1(4,j)*t1yy(k,i,j+2)+dtyz)
     v1z(k,i,j)=v1z(k,i,j)+ &
                0.5*(rho(idmat1(k,i,j))+rho(idmat1(k-1,i,j)))* &
               (dxh1(1,i)*t1xz(k,i-2,j)+dxh1(2,i)*t1xz(k,i-1,j)+ &
                dxh1(3,i)*t1xz(k,i,  j)+dxh1(4,i)*t1xz(k,i+1,j)+ &
                dyh1(1,j)*t1yz(k,i,j-2)+dyh1(2,j)*t1yz(k,i,j-1)+ &
                dyh1(3,j)*t1yz(k,i,j  )+dyh1(4,j)*t1yz(k,i,j+1)+dtzz)
   enddo
   do k=3,nztm1
     v1x(k,i,j)=v1x(k,i,j)+ &
                0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i+1,j)))* &
               (dxi1(1,i)*t1xx(k,i-1,j)+dxi1(2,i)*t1xx(k,i,  j)+ &
                dxi1(3,i)*t1xx(k,i+1,j)+dxi1(4,i)*t1xx(k,i+2,j)+ &
                dyh1(1,j)*t1xy(k,i,j-2)+dyh1(2,j)*t1xy(k,i,j-1)+ &
                dyh1(3,j)*t1xy(k,i,j  )+dyh1(4,j)*t1xy(k,i,j+1)+ &
                dzi1(1,k)*t1xz(k-1,i,j)+dzi1(2,k)*t1xz(k,  i,j)+ &
                dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j))
     v1y(k,i,j)=v1y(k,i,j)+ &
                0.5*(rho(idmat1(k,i,j))+rho(idmat1(k,i,j+1)))* &
               (dxh1(1,i)*t1xy(k,i-2,j)+dxh1(2,i)*t1xy(k,i-1,j)+ &
                dxh1(3,i)*t1xy(k,i,  j)+dxh1(4,i)*t1xy(k,i+1,j)+ &
                dyi1(1,j)*t1yy(k,i,j-1)+dyi1(2,j)*t1yy(k,i,j  )+ &
                dyi1(3,j)*t1yy(k,i,j+1)+dyi1(4,j)*t1yy(k,i,j+2)+ &
                dzi1(1,k)*t1yz(k-1,i,j)+dzi1(2,k)*t1yz(k  ,i,j)+ &
                dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j)) 
     v1z(k,i,j)=v1z(k,i,j)+ &
                0.5*(rho(idmat1(k,i,j))+rho(idmat1(k-1,i,j)))* &
               (dxh1(1,i)*t1xz(k,i-2,j)+dxh1(2,i)*t1xz(k,i-1,j)+ &
                dxh1(3,i)*t1xz(k,i,  j)+dxh1(4,i)*t1xz(k,i+1,j)+ &
                dyh1(1,j)*t1yz(k,i,j-2)+dyh1(2,j)*t1yz(k,i,j-1)+ &
                dyh1(3,j)*t1yz(k,i,j  )+dyh1(4,j)*t1yz(k,i,j+1)+ &
                dzh1(1,k)*t1zz(k-2,i,j)+dzh1(2,k)*t1zz(k-1,i,j)+ &
                dzh1(3,k)*t1zz(k  ,i,j)+dzh1(4,k)*t1zz(k+1,i,j))
   enddo
 enddo
 enddo
 return
end subroutine velocity_inner_I
!
!-----------------------------------------------------------------------
subroutine velocity_inner_II
! Compute the velocity at Region II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k
!
 do j=nd2_vel(3),nd2_vel(4)
 do i=nd2_vel(9),nd2_vel(10)
   k=1
   v2x(k,i,j)=v2x(k,i,j)+ &
             0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i+1,j)))* &
             (dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+ &
              dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j)+ &
              dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+ &
              dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1)+ &
              dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j  )+ &
              dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j))
   v2y(k,i,j)=v2y(k,i,j)+ &
             0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i,j+1)))* &
             (dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+ &
              dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j)+ &
              dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+ &
              dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2)+ &
              dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+ &
              dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j))
   v2z(k,i,j)=v2z(k,i,j)+ &
             0.5*(rho(idmat2(k,i,j))+rho(idmat2(k-1,i,j)))* &
             (dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+ &
              dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j)+ &
              dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+ &
              dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1)+ &
              dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j)))
   do k=2,nd2_vel(16)
     v2x(k,i,j)=v2x(k,i,j)+ &
             0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i+1,j)))* &
             (dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+ &
              dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j)+ &
              dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+ &
              dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1)+ &
              dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j  )+ &
              dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j))
     v2y(k,i,j)=v2y(k,i,j)+ &
               0.5*(rho(idmat2(k,i,j))+rho(idmat2(k,i,j+1)))* &
              (dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+ &
               dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j)+ &
               dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+ &
               dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2)+ &
               dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+ &
               dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j))
     v2z(k,i,j)=v2z(k,i,j)+ &
               0.5*(rho(idmat2(k,i,j))+rho(idmat2(k-1,i,j)))* &
              (dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+ &
               dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j)+ &
               dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+ &
               dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1)+ &
               dzh2(1,k)*t2zz(k-2,i,j)+dzh2(2,k)*t2zz(k-1,i,j)+ &
               dzh2(3,k)*t2zz(k,  i,j)+dzh2(4,k)*t2zz(k+1,i,j))
   enddo
 enddo
 enddo
 return
end subroutine velocity_inner_II
!
!-----------------------------------------------------------------------
subroutine vel_PmlX_I
!  Compute the velocities in region of PML-x-I
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb
 real:: rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz, &
        vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz,dtxy,dtyy,dtzy
!
 if ( lbx(1)>lbx(2) ) return
 do j=nd1_vel(1),nd1_vel(6)
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd1_vel(7+4*lb),nd1_vel(8+4*lb)
       kb=kb+1
       ib=ib+1
       rth=drvh1(kb,lb)
       rti=drti1(kb,lb)
       do k=1,nztop
         damp0=damp1_x(k,j,lb)
         dmpx2=1./(1.+rth*damp0)
         dmpx1=dmpx2*2.-1.
         dmpyz2=1./(1.+rti*damp0)
         dmpyz1=dmpyz2*2.-1.
         ro1=rho(idmat1(k,i,j))
         rox=0.5*(ro1+rho(idmat1(k,i+1,j)))
         roy=0.5*(ro1+rho(idmat1(k,i,j+1)))
         roz=0.5*(ro1+rho(idmat1(k-1,i,j)))
         vtmpx=v1x(k,i,j)-v1x_px(k,ib,j)
         vtmpy=v1y(k,i,j)-v1y_px(k,ib,j)
         vtmpz=v1z(k,i,j)-v1z_px(k,ib,j)
         if(j>nd1_vel(2) .and. j<nd1_vel(5)) then
           dtxy=dyh1(1,j)*t1xy(k,i,j-2)+dyh1(2,j)*t1xy(k,i,j-1)+ &
                dyh1(3,j)*t1xy(k,i,j  )+dyh1(4,j)*t1xy(k,i,j+1)
           dtyy=dyi1(1,j)*t1yy(k,i,j-1)+dyi1(2,j)*t1yy(k,i,j  )+ &
                dyi1(3,j)*t1yy(k,i,j+1)+dyi1(4,j)*t1yy(k,i,j+2)
           dtzy=dyh1(1,j)*t1yz(k,i,j-2)+dyh1(2,j)*t1yz(k,i,j-1)+ &
                dyh1(3,j)*t1yz(k,i,j  )+dyh1(4,j)*t1yz(k,i,j+1)
           if(k==1) then
             dtxz=(dzi1(3,k)-dzi1(1,k))*t1xz(2,i,j)+dzi1(4,k)*t1xz(3,i,j)
             dtyz=(dzi1(3,k)-dzi1(1,k))*t1yz(2,i,j)+dzi1(4,k)*t1yz(3,i,j)
             dtzz=dzh1(3,k)/ca*(35./8.*t1zz(k,i,j)-35./24.*t1zz(k+1,i,j)+ &
                  21./40.*t1zz(k+2,i,j)-5./56.*t1zz(k+3,i,j))
           else if(k==2) then
             dtxz=dzi1(2,k)*t1xz(k,i,j)+ &
                  dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j)
             dtyz=dzi1(2,k)*t1yz(k,i,j)+ &
                  dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j)
             dtzz=dzh1(3,k)/ca*(-31./24.*t1zz(k-1,i,j)+29./24.*t1zz(k,i,j)- &
                  3./40.*t1zz(k+1,i,j)+1./168.*t1zz(k+2,i,j))
           else if(k==nztop) then
             dtxz=dzi1(2,k)/ca*(t1xz(k,i,j)-t1xz(k+1,i,j))
             dtyz=dzi1(2,k)/ca*(t1yz(k,i,j)-t1yz(k+1,i,j))
             dtzz=dzh1(2,k)/ca*(t1zz(k-1,i,j)-t1zz(k,i,j))
           else
             dtxz=dzi1(1,k)*t1xz(k-1,i,j)+dzi1(2,k)*t1xz(k,  i,j)+ &
                  dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j)
             dtyz=dzi1(1,k)*t1yz(k-1,i,j)+dzi1(2,k)*t1yz(k  ,i,j)+ &
                  dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j)
             dtzz=dzh1(1,k)*t1zz(k-2,i,j)+dzh1(2,k)*t1zz(k-1,i,j)+ &
                  dzh1(3,k)*t1zz(k  ,i,j)+dzh1(4,k)*t1zz(k+1,i,j)
           endif
           vtmpx=vtmpx+(dtxy+dtxz)*rox
           vtmpy=vtmpy+(dtyy+dtyz)*roy
           vtmpz=vtmpz+(dtzy+dtzz)*roz
         endif
         v1x_px(k,ib,j)=v1x_px(k,ib,j)*dmpx1+dmpx2*rox* &
                        dxi1(2,i)/ca*(t1xx(k,i,j)-t1xx(k,i+1,j))
         v1x(k,i,j)=vtmpx+v1x_px(k,ib,j)
         v1y_px(k,ib,j)=v1y_px(k,ib,j)*dmpyz1+dmpyz2*roy* &
                        dxh1(2,i)/ca*(t1xy(k,i-1,j)-t1xy(k,i,j))
         v1y(k,i,j)=vtmpy+v1y_px(k,ib,j)
         v1z_px(k,ib,j)=v1z_px(k,ib,j)*dmpyz1+dmpyz2*roz* &
                        dxh1(2,i)/ca*(t1xz(k,i-1,j)-t1xz(k,i,j))
         v1z(k,i,j)=vtmpz+v1z_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine vel_PmlX_I
!
!-----------------------------------------------------------------------
subroutine vel_PmlY_I
!   Compute the velocities in region of PML-y-I
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb
 real:: rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz, &
        dtxz,dtyz,dtzz,vtmpx,vtmpy,vtmpz
!
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1),lby(2)
   kb=0
   do j=nd1_vel(1+4*lb),nd1_vel(2+4*lb)
     kb=kb+1
     jb=jb+1
     rth=drvh1(kb,lb)
     rti=drti1(kb,lb)
     do i=nd1_vel(7),nd1_vel(12)
     do k=1,nztop
       damp0=damp1_y(k,i,lb)
       dmpy2=1./(1.+rth*damp0)
       dmpy1=dmpy2*2.-1.
       dmpxz2=1./(1.+rti*damp0)
       dmpxz1=dmpxz2*2.-1.
       ro1=rho(idmat1(k,i,j))
       rox=0.5*(ro1+rho(idmat1(k,i+1,j)))
       roy=0.5*(ro1+rho(idmat1(k,i,j+1)))
       roz=0.5*(ro1+rho(idmat1(k-1,i,j)))
       if(k==1) then
         dtxz=(dzi1(3,k)-dzi1(1,k))*t1xz(2,i,j)+dzi1(4,k)*t1xz(3,i,j)
         dtyz=(dzi1(3,k)-dzi1(1,k))*t1yz(2,i,j)+dzi1(4,k)*t1yz(3,i,j)
         dtzz=dzh1(3,k)/ca*(35./8.*t1zz(k,i,j)-35./24.*t1zz(k+1,i,j)+ &
              21./40.*t1zz(k+2,i,j)-5./56.*t1zz(k+3,i,j))
       else if(k==2) then
         dtxz=dzi1(2,k)*t1xz(k,i,j)+ &
              dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j)
         dtyz=dzi1(2,k)*t1yz(k,i,j)+ &
              dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j)
         dtzz=dzh1(3,k)/ca*(-31./24.*t1zz(k-1,i,j)+29./24.*t1zz(k,i,j)- &
              3./40.*t1zz(k+1,i,j)+1./168.*t1zz(k+2,i,j))
       else if(k==nztop) then
         dtxz=dzi1(2,k)/ca*(t1xz(k,i,j)-t1xz(k+1,i,j))
         dtyz=dzi1(2,k)/ca*(t1yz(k,i,j)-t1yz(k+1,i,j))
         dtzz=dzh1(2,k)/ca*(t1zz(k-1,i,j)-t1zz(k,i,j))
       else
         dtxz=dzi1(1,k)*t1xz(k-1,i,j)+dzi1(2,k)*t1xz(k,  i,j)+ &
              dzi1(3,k)*t1xz(k+1,i,j)+dzi1(4,k)*t1xz(k+2,i,j)
         dtyz=dzi1(1,k)*t1yz(k-1,i,j)+dzi1(2,k)*t1yz(k  ,i,j)+ &
              dzi1(3,k)*t1yz(k+1,i,j)+dzi1(4,k)*t1yz(k+2,i,j)
         dtzz=dzh1(1,k)*t1zz(k-2,i,j)+dzh1(2,k)*t1zz(k-1,i,j)+ &
              dzh1(3,k)*t1zz(k  ,i,j)+dzh1(4,k)*t1zz(k+1,i,j)
       endif
       vtmpx=v1x(k,i,j)-v1x_py(k,i,jb)+dtxz*rox
       vtmpy=v1y(k,i,j)-v1y_py(k,i,jb)+dtyz*roy
       vtmpz=v1z(k,i,j)-v1z_py(k,i,jb)+dtzz*roz
       if(i>nd1_vel(8) .and. i<nd1_vel(11)) then
         vtmpx=vtmpx+ &
             rox*(dxi1(1,i)*t1xx(k,i-1,j)+dxi1(2,i)*t1xx(k,i,  j)+ &
                  dxi1(3,i)*t1xx(k,i+1,j)+dxi1(4,i)*t1xx(k,i+2,j))
         vtmpy=vtmpy+ &
             roy*(dxh1(1,i)*t1xy(k,i-2,j)+dxh1(2,i)*t1xy(k,i-1,j)+ &
                  dxh1(3,i)*t1xy(k,i,  j)+dxh1(4,i)*t1xy(k,i+1,j))
         vtmpz=vtmpz+ &
             roz*(dxh1(1,i)*t1xz(k,i-2,j)+dxh1(2,i)*t1xz(k,i-1,j)+ &
                  dxh1(3,i)*t1xz(k,i,  j)+dxh1(4,i)*t1xz(k,i+1,j))
       endif
       v1x_py(k,i,jb)=v1x_py(k,i,jb)*dmpxz1+dmpxz2* &
                 rox*dyh1(2,j)/ca*(t1xy(k,i,j-1)-t1xy(k,i,j))
       v1x(k,i,j)=vtmpx+v1x_py(k,i,jb)
       v1y_py(k,i,jb)=v1y_py(k,i,jb)*dmpy1+dmpy2* &
                 roy*dyi1(2,j)/ca*(t1yy(k,i,j)-t1yy(k,i,j+1))
       v1y(k,i,j)=vtmpy+v1y_py(k,i,jb)
       v1z_py(k,i,jb)=v1z_py(k,i,jb)*dmpxz1+dmpxz2* &
                 roz*dyh1(2,j)/ca*(t1yz(k,i,j-1)-t1yz(k,i,j))
       v1z(k,i,j)=vtmpz+v1z_py(k,i,jb)
     enddo
     enddo
   enddo
 enddo
 return
end subroutine vel_PmlY_I
!
!-----------------------------------------------------------------------
subroutine vel_PmlX_II
!  Compute the velocities in Region of PML-x-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb
 real:: rth,rti,damp0,dmpx2,dmpx1,dmpyz2,dmpyz1,ro1,rox,roy,roz, &
        vtmpx,vtmpy,vtmpz,dtxy,dtyy,dtzy,dtxz,dtyz,dtzz
!
 if ( lbx(1)>lbx(2) ) return
 do j=nd2_vel(1),nd2_vel(6)
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd2_vel(7+4*lb),nd2_vel(8+4*lb)
       kb=kb+1
       ib=ib+1
       rth=drvh2(kb,lb)
       rti=drti2(kb,lb)
       do k=1,nzbm1
         damp0=damp2_x(k,j,lb)
         dmpx2=1./(1.+rth*damp0)
         dmpx1=dmpx2*2.-1.
         dmpyz2=1./(1.+rti*damp0)
         dmpyz1=dmpyz2*2.-1.
         ro1=rho(idmat2(k,i,j))
         rox=0.5*(ro1+rho(idmat2(k,i+1,j)))
         roy=0.5*(ro1+rho(idmat2(k,i,j+1)))
         roz=0.5*(ro1+rho(idmat2(k-1,i,j)))
         vtmpx=v2x(k,i,j)-v2x_px(k,ib,j)
         vtmpy=v2y(k,i,j)-v2y_px(k,ib,j)
         vtmpz=v2z(k,i,j)-v2z_px(k,ib,j)
         if(j>nd2_vel(2) .and. j<nd2_vel(5)) then
           dtxy=dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+ &
                dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1)
           dtyy=dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+ &
                dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2)
           dtzy=dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+ &
                dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1)
           if(k==1) then
             dtxz=dzi2(2,k)/ca*(t2xz(k,i,j)-t2xz(k+1,i,j))
             dtyz=dzi2(2,k)/ca*(t2yz(k,i,j)-t2yz(k+1,i,j))
             dtzz=dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j))
           else if(k<nd2_vel(17)) then
             dtxz=dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j)+ &
                  dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j)
             dtyz=dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+ &
                  dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j)
             dtzz=dzh2(1,k)*t2zz(k-2,i,j)+dzh2(2,k)*t2zz(k-1,i,j)+ &
                  dzh2(3,k)*t2zz(k,  i,j)+dzh2(4,k)*t2zz(k+1,i,j)
           else
             dtxz=0.0
             dtyz=0.0
             dtzz=0.0
           endif
           vtmpx=vtmpx+(dtxy+dtxz)*rox
           vtmpy=vtmpy+(dtyy+dtyz)*roy
           vtmpz=vtmpz+(dtzy+dtzz)*roz
         endif
         v2x_px(k,ib,j)=v2x_px(k,ib,j)*dmpx1+dmpx2* &
                     rox*dxi2(2,i)/ca*(t2xx(k,i,j)-t2xx(k,i+1,j))
         v2x(k,i,j)=vtmpx+v2x_px(k,ib,j)
         v2y_px(k,ib,j)=v2y_px(k,ib,j)*dmpyz1+dmpyz2* &
                     roy*dxh2(2,i)/ca*(t2xy(k,i-1,j)-t2xy(k,i,j))
         v2y(k,i,j)=vtmpy+v2y_px(k,ib,j)
         v2z_px(k,ib,j)=v2z_px(k,ib,j)*dmpyz1+dmpyz2* &
                     roz*dxh2(2,i)/ca*(t2xz(k,i-1,j)-t2xz(k,i,j))
         v2z(k,i,j)=vtmpz+v2z_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine vel_PmlX_II
!
!-----------------------------------------------------------------------
subroutine vel_PmlY_II
! Compute the velocities in region of PML-y-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb
 real:: rth,rti,damp0,dmpy2,dmpy1,dmpxz2,dmpxz1,ro1,rox,roy,roz, &
        vtmpx,vtmpy,vtmpz,dtxz,dtyz,dtzz
!
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1),lby(2)
   kb=0
   do j=nd2_vel(1+4*lb),nd2_vel(2+4*lb)
     kb=kb+1
     jb=jb+1
     rth=drvh2(kb,lb)
     rti=drti2(kb,lb)
     do i=nd2_vel(7),nd2_vel(12)
     do k=1,nzbm1
       damp0=damp2_y(k,i,lb)
       dmpy2=1./(1.+rth*damp0)
       dmpy1=dmpy2*2.-1.0
       dmpxz2=1./(1.+rti*damp0)
       dmpxz1=dmpxz2*2.-1.
       ro1=rho(idmat2(k,i,j))
       rox=0.5*(ro1+rho(idmat2(k,i+1,j)))
       roy=0.5*(ro1+rho(idmat2(k,i,j+1)))
       roz=0.5*(ro1+rho(idmat2(k-1,i,j)))
       vtmpx=v2x(k,i,j)-v2x_py(k,i,jb)
       vtmpy=v2y(k,i,j)-v2y_py(k,i,jb)
       vtmpz=v2z(k,i,j)-v2z_py(k,i,jb)
       if(k<nd2_vel(17)) then
         if(k>1) then
           dtxz=dzi2(1,k)*t2xz(k-1,i,j)+dzi2(2,k)*t2xz(k,i,j)+ &
                dzi2(3,k)*t2xz(k+1,i,j)+dzi2(4,k)*t2xz(k+2,i,j)
           dtyz=dzi2(1,k)*t2yz(k-1,i,j)+dzi2(2,k)*t2yz(k,i,j)+ &
                dzi2(3,k)*t2yz(k+1,i,j)+dzi2(4,k)*t2yz(k+2,i,j)
           dtzz=dzh2(1,k)*t2zz(k-2,i,j)+dzh2(2,k)*t2zz(k-1,i,j)+ &
                dzh2(3,k)*t2zz(k,  i,j)+dzh2(4,k)*t2zz(k+1,i,j)
         else
           dtxz=dzi2(2,k)/ca*(t2xz(k,i,j)-t2xz(k+1,i,j))
           dtyz=dzi2(2,k)/ca*(t2yz(k,i,j)-t2yz(k+1,i,j))
           dtzz=dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j))
         endif
         if(i>nd2_vel(8) .and. i<nd2_vel(11)) then
           vtmpx=vtmpx+rox*(dtxz+ &
               dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+ &
               dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j))
           vtmpy=vtmpy+roy*(dtyz+ &
               dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+ &
               dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j))
           vtmpz=vtmpz+roz*(dtzz+ &
               dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+ &
               dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j))
         else
           vtmpx=vtmpx+rox*dtxz
           vtmpy=vtmpy+roy*dtyz
           vtmpz=vtmpz+roz*dtzz
         endif
       else
         if(i>nd2_vel(8) .and. i<nd2_vel(11)) then
           vtmpx=vtmpx+rox* &
              (dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+ &
               dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j))
           vtmpy=vtmpy+ roy* &
              (dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+ &
               dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j))
           vtmpz=vtmpz+ roz* &
              (dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+ &
               dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j))
         endif
       endif
       v2x_py(k,i,jb)=v2x_py(k,i,jb)*dmpxz1+dmpxz2*rox* &
                    dyh2(2,j)/ca*(t2xy(k,i,j-1)-t2xy(k,i,j))
       v2x(k,i,j)=vtmpx+v2x_py(k,i,jb)
       v2y_py(k,i,jb)=v2y_py(k,i,jb)*dmpy1+dmpy2*roy* &
                    dyi2(2,j)/ca*(t2yy(k,i,j)-t2yy(k,i,j+1))
       v2y(k,i,j)=vtmpy+v2y_py(k,i,jb)
       v2z_py(k,i,jb)=v2z_py(k,i,jb)*dmpxz1+dmpxz2*roz* &
                    dyh2(2,j)/ca*(t2yz(k,i,j-1)-t2yz(k,i,j))
       v2z(k,i,j)=vtmpz+v2z_py(k,i,jb)
     enddo
     enddo
   enddo
 enddo
 return
end subroutine vel_PmlY_II
!
!-----------------------------------------------------------------------
subroutine vel_PmlZ_II
! Compute the velocities in region of PML-z-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,kb
 real:: damp0,dmpz2,dmpz1,dmpxy2,dmpxy1,ro1,rox,roy,roz, &
        vtmpx,vtmpy,vtmpz
!
 do j=nd2_vel(1),nd2_vel(6)
 do i=nd2_vel(7),nd2_vel(12)
   kb=0
   damp0=damp2_z(i,j)
   do k=nd2_vel(17),nzbm1
     kb=kb+1
     dmpz2=1./(1.+damp0*drti2(kb,1))
     dmpz1=dmpz2*2.-1.
     dmpxy2=1./(1.+damp0*drvh2(kb,1))
     dmpxy1=dmpxy2*2.-1.
     ro1=rho(idmat2(k,i,j))
     rox=0.5*(ro1+rho(idmat2(k,i+1,j)))
     roy=0.5*(ro1+rho(idmat2(k,i,j+1)))
     roz=0.5*(ro1+rho(idmat2(k-1,i,j)))
     vtmpx=v2x(k,i,j)-v2x_pz(kb,i,j)
     vtmpy=v2y(k,i,j)-v2y_pz(kb,i,j)
     vtmpz=v2z(k,i,j)-v2z_pz(kb,i,j)
     if(j>nd2_vel(2) .and. j<nd2_vel(5) .and. &
        i>nd2_vel(8) .and. i<nd2_vel(11)) then
       vtmpx=vtmpx+rox* &
            (dxi2(1,i)*t2xx(k,i-1,j)+dxi2(2,i)*t2xx(k,i,  j)+ &
             dxi2(3,i)*t2xx(k,i+1,j)+dxi2(4,i)*t2xx(k,i+2,j)+ &
             dyh2(1,j)*t2xy(k,i,j-2)+dyh2(2,j)*t2xy(k,i,j-1)+ &
             dyh2(3,j)*t2xy(k,i,j  )+dyh2(4,j)*t2xy(k,i,j+1))
       vtmpy=vtmpy+roy* &
            (dxh2(1,i)*t2xy(k,i-2,j)+dxh2(2,i)*t2xy(k,i-1,j)+ &
             dxh2(3,i)*t2xy(k,i,  j)+dxh2(4,i)*t2xy(k,i+1,j)+ &
             dyi2(1,j)*t2yy(k,i,j-1)+dyi2(2,j)*t2yy(k,i,j)+ &
             dyi2(3,j)*t2yy(k,i,j+1)+dyi2(4,j)*t2yy(k,i,j+2))
       vtmpz=vtmpz+roz* &
            (dxh2(1,i)*t2xz(k,i-2,j)+dxh2(2,i)*t2xz(k,i-1,j)+ &
             dxh2(3,i)*t2xz(k,i,  j)+dxh2(4,i)*t2xz(k,i+1,j)+ &
             dyh2(1,j)*t2yz(k,i,j-2)+dyh2(2,j)*t2yz(k,i,j-1)+ &
             dyh2(3,j)*t2yz(k,i,j  )+dyh2(4,j)*t2yz(k,i,j+1))
     endif
     v2x_pz(kb,i,j)=v2x_pz(kb,i,j)*dmpxy1+dmpxy2*rox* &
                   dzi2(2,k)/ca*(t2xz(k,i,j)-t2xz(k+1,i,j))
     v2x(k,i,j)=vtmpx+v2x_pz(kb,i,j)
     v2y_pz(kb,i,j)=v2y_pz(kb,i,j)*dmpxy1+dmpxy2*roy* &
                    dzi2(2,k)/ca*(t2yz(k,i,j)-t2yz(k+1,i,j))
     v2y(k,i,j)=vtmpy+v2y_pz(kb,i,j)
     v2z_pz(kb,i,j)=v2z_pz(kb,i,j)*dmpz1+dmpz2*roz* &
                    dzh2(2,k)/ca*(t2zz(k-1,i,j)-t2zz(k,i,j))
     v2z(k,i,j)=vtmpz+v2z_pz(kb,i,j)
   enddo
 enddo
 enddo
 return
end subroutine vel_PmlZ_II
!
!-----------------------------------------------------------------------
subroutine update_stress(comm_worker,myid,tim)
! Updating the stress field with 2-nd and 4th-order FD scheme
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer, intent(IN):: comm_worker,myid
 real, intent(IN):: tim
! Adding double couple sources (stress)
! call add_dcs(tim)
! Computing the stree in the inner of region I
 call stress_norm_xy_I
 call stress_xz_yz_I
! Implement PML boundary conditions in the region I
 call stress_norm_PmlX_I
 call stress_norm_PmlY_I
 call stress_xy_PmlX_I
 call stress_xy_PmlY_I
 call stress_xz_PmlX_I
 call stress_xz_PmlY_I
 call stress_yz_PmlX_I
 call stress_yz_PmlY_I
! Computing the stree in the inner of region II
 call stress_norm_xy_II
 call stress_xz_yz_II
!  Implement PML boundary conditions in region II
 call stress_norm_PmlX_II
 call stress_norm_PmlY_II
 call stress_norm_PmlZ_II
 call stress_xy_PmlX_II
 call stress_xy_PmlY_II
 call stress_xy_PmlZ_II
 call stress_xz_PmlX_II
 call stress_xz_PmlY_II
 call stress_xz_PmlZ_II
 call stress_yz_PmlX_II
 call stress_yz_PmlY_II
 call stress_yz_PmlZ_II
! Interpolating Stress on the interface betwee Region I and II
 call stress_interp
 call ISEND_IRECV_Stress(comm_worker)
 return
end subroutine update_stress
!
!XSC-----------------------------------------------------------
subroutine compute_stress(myid, tim)
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer, intent(IN):: myid
 real, intent(IN):: tim

! Adding double couple sources (stress)
! call add_dcs(tim)
! Computing the stree in the inner of region I
 call stress_norm_xy_I
 call stress_xz_yz_I
! Implement PML boundary conditions in the region I
 call stress_norm_PmlX_I
 call stress_norm_PmlY_I
 call stress_xy_PmlX_I
 call stress_xy_PmlY_I
 call stress_xz_PmlX_I
 call stress_xz_PmlY_I
 call stress_yz_PmlX_I
 call stress_yz_PmlY_I
! Computing the stree in the inner of region II
 call stress_norm_xy_II
 call stress_xz_yz_II
!  Implement PML boundary conditions in region II
 call stress_norm_PmlX_II
 call stress_norm_PmlY_II
 call stress_norm_PmlZ_II
 call stress_xy_PmlX_II
 call stress_xy_PmlY_II
 call stress_xy_PmlZ_II
 call stress_xz_PmlX_II
 call stress_xz_PmlY_II
 call stress_xz_PmlZ_II
 call stress_yz_PmlX_II
 call stress_yz_PmlY_II
 call stress_yz_PmlZ_II

 return

end subroutine compute_stress

subroutine comm_stress(comm_worker)
 use grid_node_comm
 use wave_field_comm
 use iso_c_binding
 use ctimer
 implicit NONE

 integer, intent(IN):: comm_worker
 real(c_double) :: tend, tstart
! Interpolating Stress on the interface betwee Region I and II
 call record_time(tstart)
 call stress_interp
 call record_time(tend)
 write(*,*) "TIME :: stress_interp :", tend-tstart

 call record_time(tstart)
 call ISEND_IRECV_Stress(comm_worker)
 call record_time(tend)
 write(*,*) "TIME :: ISEND_IRECV_Stress :", tend-tstart

 return
end subroutine comm_stress
!end XSC=======================================================

subroutine velocity_one_time_data

 use grid_node_comm
 use wave_field_comm
 use source_parm_comm
 use :: iso_c_binding
 implicit NONE


    interface
        subroutine one_time_data_vel(index_xyz_source, ixsX, ixsY, ixsZ, &
                                    famp, fampX, fampY, &
                                    ruptm, ruptmX, riset, risetX, sparam2, sparam2X) &
            bind (C, name="one_time_data_vel")
            use :: iso_c_binding
            type(c_ptr), intent(in), value :: index_xyz_source, famp, ruptm, riset, sparam2
            integer(c_int),intent(in), value:: ixsX, ixsY, ixsZ, fampX, fampY, ruptmX, risetX, sparam2X
        end subroutine one_time_data_vel
    end interface
    call one_time_data_vel(c_loc(index_xyz_source), size(index_xyz_source,1),&
                size(index_xyz_source,2), size(index_xyz_source,3), &
                c_loc(famp), size(famp,1), size(famp,2), &
                c_loc(ruptm), size(ruptm, 1), &
                c_loc(riset), size(riset, 1), &
                c_loc(sparam2), size(sparam2, 1)) 
end subroutine velocity_one_time_data



!-----------------------------------------------------------------------
subroutine add_dcs(tim)
! Adding double couple sources 
 use grid_node_comm
 use wave_field_comm
 use source_parm_comm
 use :: iso_c_binding
! use logging
 use metadata
 use ctimer
 !use iso_c_binding
 implicit NONE

 interface
    subroutine add_dcs_vel(sutmArr, nfadd, ixsX, ixsY, ixsZ, fampX, fampY, ruptmX, &
    risetX, sparam2X, nzrg11, nzrg12, nzrg13, nzrg14, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm) &
        bind(C, name="add_dcs_vel")
        use :: iso_c_binding
        integer(c_int), value, intent(in) :: nfadd, nzrg11, nzrg12, nzrg13, nzrg14, nxtop, nytop, nztop, nxbtm, nybtm, nzbtm, &
                                            ixsX, ixsY, ixsZ, fampX, fampY, ruptmX, risetX, sparam2X
        type(c_ptr),value, intent(in) :: sutmArr
    end subroutine add_dcs_vel
 end interface

 real, intent(IN):: tim
 integer:: i,j,k,kap
 real::sutm,fsource
 real, allocatable, target, dimension(:) :: sutmArr
 real(c_double) :: tstart, tend
 real(c_double) :: gpu_tstart, gpu_tend, cpu_tstart, cpu_tend
 real(c_double):: total_gpu_marsh_time=0.0, total_cpu_marsh_time=0.0
 total_gpu_marsh_time =0.0
 total_cpu_marsh_time =0.0
 
 if(DETAILED_TIMING .eq. 1) then; call log_timing ("GPU ADD_DCS NFADD ", DBLE(nfadd)); end if
 allocate(sutmArr(nfadd))
 if (nfadd .eq. 0) then
    return
 endif    

 do  kap=1,nfadd
   sutmArr(kap)=-fsource(tim-ruptm(kap),riset(kap),sparam2(kap),id_sf_type)
 enddo
#ifdef DISFD_GPU_MARSHALING
 if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
 call add_dcs_vel(c_loc(sutmArr), nfadd, size(index_xyz_source,1),&
                size(index_xyz_source,2), size(index_xyz_source,3), &
                size(famp,1), size(famp,2), &
                size(ruptm, 1), &
                size(riset, 1), &
                size(sparam2, 1), nzrg1(1), nzrg1(2), nzrg1(3), nzrg1(4), &
                nxtop, nytop, nztop, nxbtm, nybtm , nzbtm)
 if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU add_dcs", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

 if(DETAILED_TIMING .eq. 1) then; call log_timing ("GPU Marshaling add_dcs", total_gpu_marsh_time); end if
#else
 call record_time(tstart)
! Mxx, Myy, Mzz:
 do  kap=1,nfadd
   sutm=-fsource(tim-ruptm(kap),riset(kap),sparam2(kap),id_sf_type)
   i=index_xyz_source(1,kap,1)
   j=index_xyz_source(2,kap,1)
   k=index_xyz_source(3,kap,1)
   if(k>nzrg1(1)) then
     k=k-nztop
     t2xx(k,i,j)=t2xx(k,i,j)+famp(kap,1)*sutm
     t2yy(k,i,j)=t2yy(k,i,j)+famp(kap,2)*sutm
     t2zz(k,i,j)=t2zz(k,i,j)+famp(kap,3)*sutm
   else
     t1xx(k,i,j)=t1xx(k,i,j)+famp(kap,1)*sutm
     t1yy(k,i,j)=t1yy(k,i,j)+famp(kap,2)*sutm
     t1zz(k,i,j)=t1zz(k,i,j)+famp(kap,3)*sutm
   endif
 enddo
! Mxy
 do  kap=1,nfadd
   sutm=-fsource(tim-ruptm(kap),riset(kap),sparam2(kap),id_sf_type)
   i=index_xyz_source(1,kap,2)
   j=index_xyz_source(2,kap,2)
   k=index_xyz_source(3,kap,2)
   if(k>nzrg1(2)) then
     k=k-nztop
     t2xy(k,i,j)=t2xy(k,i,j)+famp(kap,4)*sutm
   else
     t1xy(k,i,j)=t1xy(k,i,j)+famp(kap,4)*sutm
   endif
 enddo
! Mxz
 do  kap=1,nfadd
   sutm=-fsource(tim-ruptm(kap),riset(kap),sparam2(kap),id_sf_type)
   i=index_xyz_source(1,kap,3)
   j=index_xyz_source(2,kap,3)
   k=index_xyz_source(3,kap,3)
   if(k>nzrg1(3)) then
     k=k-nztop
     t2xz(k,i,j)=t2xz(k,i,j)+famp(kap,5)*sutm
   else
     t1xz(k,i,j)=t1xz(k,i,j)+famp(kap,5)*sutm
   endif
 enddo
! Myz
 do  kap=1,nfadd
   sutm=-fsource(tim-ruptm(kap),riset(kap),sparam2(kap),id_sf_type)
   i=index_xyz_source(1,kap,4)
   j=index_xyz_source(2,kap,4)
   k=index_xyz_source(3,kap,4)
   if(k>nzrg1(4)) then
     k=k-nztop
     t2yz(k,i,j)=t2yz(k,i,j)+famp(kap,6)*sutm
   else
     t1yz(k,i,j)=t1yz(k,i,j)+famp(kap,6)*sutm
   endif
 enddo

 call record_time(tend)
 write(*,*) "TIME :: add_dcs :", tend-tstart
#endif
 return
end subroutine add_dcs
!
!-----------------------------------------------------------------------
function fsource(t,dura,sp2,id_sf_type)
!  Source time function
!  When you modify this function please check normalizinf coefticient
 implicit NONE
 real, parameter:: pi=3.1415926, pi2=2.*pi, pi25=2.5*pi
 real, parameter:: cft4=0.20, c41=0.60*0.5, c42=0.5-c41, c43=2.*c42
 real, parameter:: cft5=0.20, c51=0.50*0.5, c52=0.5-c51
 integer, intent(IN):: id_sf_type
 real, intent(IN):: t,dura,sp2
 real:: tt1,tt2, fsource
! 
 fsource= 0.0
 if( t<0.0 .or. (id_sf_type/=1 .and. t>=dura) ) return 
!
 select case (id_sf_type)
 case(1)
   fsource=(t/dura)*exp(-t/dura)
 case(2)
   fsource=(t/dura)**sp2*(1.-t/dura)**(5.-sp2)
 case(3)
   tt1=sp2*dura
   tt2=(dura-tt1)
   if(t < tt1) then 
      fsource=sin(0.5*pi*t/tt1)
   else
      fsource=0.5*(1.+cos(pi*(t-tt1)/tt2))
   endif
 case(4)
   tt1=pi*t/(cft4*dura)
   if(tt1 < pi) then
     fsource=c43*sin(0.5*tt1)
   else
     fsource=c42*(1+cos((tt1-pi)*sp2))
   endif
   if(tt1 < pi2) then
     fsource=fsource+c41*(1.-cos(tt1))
   endif
 case(5)
   tt1=pi*t/(cft5*dura)
   if(tt1 < pi) then
     fsource=sin(0.5*tt1)
   else
     fsource=c52*(1+cos((tt1-pi)*sp2))
!     if(tt1 < pi25) fsource=fsource+c51*(1+cos((tt1-pi)/1.5))
     if(tt1 < pi2) fsource=fsource+c51*(1+cos(tt1-pi))
   endif
 case(6)
   tt1=dura-sp2
   if(t < tt1) then
     fsource=sin(0.5*pi*t/tt1)
   else
     fsource=0.5*(1+cos((t-tt1)*pi/sp2))
   endif
 case default
   fsource=0.0
 end select 
 return
end function fsource
!
!-----------------------------------------------------------------------
subroutine stress_norm_xy_I
! Compute stress-Norm and XY component in Region I
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,jkq,kodd,inod,irw
 real:: sxx,syy,szz,sxy,qxx,qyy,qzz,qxy,cusxy,sss, &
        cl,sm2,pm,et,et1,wtp,wts
!
 do j=nd1_tyy(3),nd1_tyy(4)
   kodd=2*mod(j+nyb1,2)+1
   do i=nd1_tyy(9),nd1_tyy(10)
     jkq=mod(i+nxb1,2)+kodd
     do k=nd1_tyy(13),nd1_tyy(18)
       sxx=dxh1(1,i)*v1x(k,i-2,j)+dxh1(2,i)*v1x(k,i-1,j)+ &
           dxh1(3,i)*v1x(k,i  ,j)+dxh1(4,i)*v1x(k,i+1,j)
       syy=dyh1(1,j)*v1y(k,i,j-2)+dyh1(2,j)*v1y(k,i,j-1)+ &
           dyh1(3,j)*v1y(k,i  ,j)+dyh1(4,j)*v1y(k,i,j+1)
       sxy=dxi1(1,i)*v1y(k,i-1,j)+dxi1(2,i)*v1y(k,i,  j)+ &
           dxi1(3,i)*v1y(k,i+1,j)+dxi1(4,i)*v1y(k,i+2,j)+ &
           dyi1(1,j)*v1x(k,i,j-1)+dyi1(2,j)*v1x(k,i,j  )+ &
           dyi1(3,j)*v1x(k,i,j+1)+dyi1(4,j)*v1x(k,i,j+2)
       if(k==1) then
         szz=dzi1(2,k)/ca*(22.*v1z(k,i,j)-17.*v1z(k+1,i,j)- &
             9.*v1z(k+2,i,j)+5.*v1z(k+3,i,j)-v1z(k+4,i,j))/24.
       else if(k==nztop) then
         szz=dzi1(2,k)/ca*(v1z(k,i,j)-v1z(k+1,i,j))
       else
         szz=dzi1(1,k)*v1z(k-1,i,j)+dzi1(2,k)*v1z(k,  i,j)+ &
             dzi1(3,k)*v1z(k+1,i,j)+dzi1(4,k)*v1z(k+2,i,j)
       endif
       inod=idmat1(k,i,j)
       cl=clamda(inod)
       sm2=2.*cmu(inod)
       pm=cl+sm2
       cusxy=sxy/(1./sm2+.5/cmu(idmat1(k,i+1,j+1)))
       sss=sxx+syy+szz
       irw=jkq+4*mod(k,2)
       et=epdt(irw)
       et1=1.0-et
       wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw))
       wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       qxx=qt1xx(k,i,j)
       qt1xx(k,i,j)=qxx*et+(wtp*sss-wts*(syy+szz))*et1
       t1xx(k,i,j)=t1xx(k,i,j)+sm2*sxx+cl*sss-qxx-qt1xx(k,i,j)
       qyy=qt1yy(k,i,j)
       qt1yy(k,i,j)=qyy*et+(wtp*sss-wts*(sxx+szz))*et1
       t1yy(k,i,j)=t1yy(k,i,j)+sm2*syy+cl*sss-qyy-qt1yy(k,i,j)
       qzz=qt1zz(k,i,j)
       qt1zz(k,i,j)=qzz*et+(wtp*sss-wts*(sxx+syy))*et1
       t1zz(k,i,j)=t1zz(k,i,j)+sm2*szz+cl*sss-qzz-qt1zz(k,i,j)
       qxy=qt1xy(k,i,j)
       qt1xy(k,i,j)=qxy*et+wts/sm2*cusxy*et1
       t1xy(k,i,j)=t1xy(k,i,j)+cusxy-qxy-qt1xy(k,i,j)
     enddo
   enddo
 enddo
 return
end subroutine stress_norm_xy_I
!
!-----------------------------------------------------------------------
subroutine stress_xz_yz_I
! Compute stress-XZand YZ component in Region I
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 real, parameter:: tfr1=-577./528./ca,tfr2=201./176./ca, &
                   tfr3=-9./176./ca,  tfr4=1./528./ca
 integer:: i,j,k,kodd,inod,jkq,irw
 real:: dvzx,dvzy,dvxz,dvyz,sm,cusxz,cusyz,et,et1,dmws,qxz,qyz
!
 do j=nd1_tyz(3),nd1_tyz(4)
   kodd=2*mod(j+nyb1,2)+1
   do i=nd1_tyz(9),nd1_tyz(10)
     jkq=mod(i+nxb1,2)+kodd
     do k=nd1_tyz(13),nd1_tyz(18)
       dvzx=dxi1(1,i)*v1z(k,i-1,j)+dxi1(2,i)*v1z(k,i,  j)+ &
            dxi1(3,i)*v1z(k,i+1,j)+dxi1(4,i)*v1z(k,i+2,j)
       dvzy=dyi1(1,j)*v1z(k,i,j-1)+dyi1(2,j)*v1z(k,i,j  )+ &
            dyi1(3,j)*v1z(k,i,j+1)+dyi1(4,j)*v1z(k,i,j+2)
       if(k<nztop) then 
         dvxz=dzh1(1,k)*v1x(k-2,i,j)+dzh1(2,k)*v1x(k-1,i,j)+ &
              dzh1(3,k)*v1x(k,  i,j)+dzh1(4,k)*v1x(k+1,i,j)
         dvyz=dzh1(1,k)*v1y(k-2,i,j)+dzh1(2,k)*v1y(k-1,i,j)+ &
              dzh1(3,k)*v1y(k,  i,j)+dzh1(4,k)*v1y(k+1,i,j)
       else
         dvxz=dzh1(2,k)/ca*(v1x(k-1,i,j)-v1x(k,i,j))
         dvyz=dzh1(2,k)/ca*(v1y(k-1,i,j)-v1y(k,i,j))
       endif
       inod=idmat1(k,i,j)
       sm=cmu(inod)
       cusxz=(dvzx+dvxz)/(.5/sm+.5/cmu(idmat1(k-1,i+1,j)))
       cusyz=(dvzy+dvyz)/(.5/sm+.5/cmu(idmat1(k-1,i,j+1)))
       irw=jkq+4*mod(k,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       qxz=qt1xz(k,i,j)
       qt1xz(k,i,j)=qxz*et+dmws*cusxz*et1
       t1xz(k,i,j)=t1xz(k,i,j)+cusxz-qxz-qt1xz(k,i,j)
       qyz=qt1yz(k,i,j)
       qt1yz(k,i,j)=qyz*et+dmws*cusyz*et1
       t1yz(k,i,j)=t1yz(k,i,j)+cusyz-qyz-qt1yz(k,i,j)
     enddo
   enddo
 enddo
 do j=-1,ny1p1
 do i=1,nxtop
   t1yz(1,i,j)=0.0
 enddo
 enddo
 do j=1,nytop
 do i=-1,nx1p1
   t1xz(1,i,j)=0.0
 enddo
 enddo
 return
end subroutine stress_xz_yz_I
!
!-----------------------------------------------------------------------
subroutine stress_norm_xy_II
! Compute stress-Norm and XY component in Region II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,kodd,inod,jkq,irw
 real:: sxx,syy,szz,sxy,sss,qxx,qyy,qzz,qxy,cusxy, &
        cl,sm2,et,et1,dmws,pm,wtp,wts
!
 do j=nd2_tyy(3),nd2_tyy(4)
   kodd=2*mod(j+nyb2,2)+1
   do i=nd2_tyy(9),nd2_tyy(10)
     jkq=mod(i+nxb2,2)+kodd
     do k=nd2_tyy(13),nd2_tyy(16)
       sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+ &
           dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j)
       syy=dyh2(1,j)*v2y(k,i,j-2)+dyh2(2,j)*v2y(k,i,j-1)+ &
           dyh2(3,j)*v2y(k,i  ,j)+dyh2(4,j)*v2y(k,i,j+1)
       sxy=dxi2(1,i)*v2y(k,i-1,j)+dxi2(2,i)*v2y(k,i,  j)+ &
           dxi2(3,i)*v2y(k,i+1,j)+dxi2(4,i)*v2y(k,i+2,j)+ &
           dyi2(1,j)*v2x(k,i,j-1)+dyi2(2,j)*v2x(k,i,j  )+ &
           dyi2(3,j)*v2x(k,i,j+1)+dyi2(4,j)*v2x(k,i,j+2)
       szz=dzi2(1,k)*v2z(k-1,i,j)+dzi2(2,k)*v2z(k,  i,j)+ &
           dzi2(3,k)*v2z(k+1,i,j)+dzi2(4,k)*v2z(k+2,i,j)
       sss=sxx+syy+szz
       inod=idmat2(k,i,j)
       cl=clamda(inod)
       sm2=2.*cmu(inod)
       pm=cl+sm2
       cusxy=sxy/(1./sm2+.5/cmu(idmat2(k,i+1,j+1)))
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw))
       wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       qxx=qt2xx(k,i,j)
       qt2xx(k,i,j)=qxx*et+(wtp*sss-wts*(syy+szz))*et1
       t2xx(k,i,j)=t2xx(k,i,j)+sm2*sxx+cl*sss-qxx-qt2xx(k,i,j)
       qyy=qt2yy(k,i,j)
       qt2yy(k,i,j)=qyy*et+(wtp*sss-wts*(sxx+szz))*et1
       t2yy(k,i,j)=t2yy(k,i,j)+sm2*syy+cl*sss-qyy-qt2yy(k,i,j)
       qzz=qt2zz(k,i,j)
       qt2zz(k,i,j)=qzz*et+(wtp*sss-wts*(sxx+syy))*et1
       t2zz(k,i,j)=t2zz(k,i,j)+sm2*szz+cl*sss-qzz-qt2zz(k,i,j)
       qxy=qt2xy(k,i,j)
       qt2xy(k,i,j)=qxy*et+wts/sm2*cusxy*et1
       t2xy(k,i,j)=t2xy(k,i,j)+cusxy-qxy-qt2xy(k,i,j)
     enddo
   enddo
 enddo
 return
end subroutine stress_norm_xy_II
!
!-----------------------------------------------------------------------
subroutine stress_xz_yz_II
! Compute stress-XZ and YZ component in the Region II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,kodd,inod,jkq,irw
 real:: qxz,qyz,cusxz,cusyz,sm,et,et1,dmws
!
 do j=nd2_tyz(3),nd2_tyz(4)
   kodd=2*mod(j+nyb2,2)+1
   do i=nd2_tyz(9),nd2_tyz(10)
     jkq=mod(i+nxb2,2)+kodd
     do k=nd2_tyz(13),nd2_tyz(16)
       inod=idmat2(k,i,j)
       sm=cmu(inod)
       cusxz=(dxi2(1,i)*v2z(k,i-1,j)+dxi2(2,i)*v2z(k,i,  j)+ &
              dxi2(3,i)*v2z(k,i+1,j)+dxi2(4,i)*v2z(k,i+2,j)+ &
              dzh2(1,k)*v2x(k-2,i,j)+dzh2(2,k)*v2x(k-1,i,j)+ &
              dzh2(3,k)*v2x(k,  i,j)+dzh2(4,k)*v2x(k+1,i,j))/ &
              (.5/sm+.5/cmu(idmat2(k-1,i+1,j)))
       cusyz=(dyi2(1,j)*v2z(k,i,j-1)+dyi2(2,j)*v2z(k,i,j  )+ &
              dyi2(3,j)*v2z(k,i,j+1)+dyi2(4,j)*v2z(k,i,j+2)+ &
              dzh2(1,k)*v2y(k-2,i,j)+dzh2(2,k)*v2y(k-1,i,j)+ &
              dzh2(3,k)*v2y(k,  i,j)+dzh2(4,k)*v2y(k+1,i,j))/ &
              (.5/sm+.5/cmu(idmat2(k-1,i,j+1)))
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       qxz=qt2xz(k,i,j)
       qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1
       t2xz(k,i,j)=t2xz(k,i,j)+cusxz-qxz-qt2xz(k,i,j)
       qyz=qt2yz(k,i,j)
       qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1
       t2yz(k,i,j)=t2yz(k,i,j)+cusyz-qyz-qt2yz(k,i,j)
     enddo
   enddo
 enddo
 return
end subroutine stress_xz_yz_II
!
!-----------------------------------------------------------------------
subroutine stress_norm_PmlX_I
! Compute the velocity of PML-x-I region
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
 real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
!
 if ( lbx(1)>lbx(2) ) return
 do j=nd1_tyy(1),nd1_tyy(6)
   kodd=2*mod(j+nyb1,2)+1
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd1_tyy(7+4*lb),nd1_tyy(8+4*lb)
       kb=kb+1
       ib=ib+1
       rti=drti1(kb,lb)
       jkq=mod(i+nxb1,2)+kodd
       do k=nd1_tyy(13),nd1_tyy(18)
         damp2=1./(1.+damp1_x(k,j,lb)*rti)
         damp1=damp2*2.0-1.
         inod=idmat1(k,i,j)
         cl=clamda(inod)
         sm2=2.*cmu(inod)
         pm=cl+sm2
         irw=jkq+4*mod(k,2)
         et=epdt(irw)
         et1=1.0-et
         wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw))
         wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxx=t1xx(k,i,j)-t1xx_px(k,ib,j)
         taoyy=t1yy(k,i,j)-t1yy_px(k,ib,j)
         taozz=t1zz(k,i,j)-t1yy_px(k,ib,j)
         if(j>nd1_tyy(2) .and. j<nd1_tyy(5)) then
           syy=dyh1(1,j)*v1y(k,i,j-2)+dyh1(2,j)*v1y(k,i,j-1)+ &
               dyh1(3,j)*v1y(k,i  ,j)+dyh1(4,j)*v1y(k,i,j+1)
           if(k==1) then
           szz=dzi1(2,k)/ca*(22.*v1z(k,i,j)-17.*v1z(k+1,i,j)- &
               9.*v1z(k+2,i,j)+5.*v1z(k+3,i,j)-v1z(k+4,i,j))/24.
           else if(k==nztop) then
             szz=dzi1(2,k)/ca*(v1z(k,i,j)-v1z(k+1,i,j))
           else
             szz=dzi1(1,k)*v1z(k-1,i,j)+dzi1(2,k)*v1z(k,  i,j)+ &
                 dzi1(3,k)*v1z(k+1,i,j)+dzi1(4,k)*v1z(k+2,i,j)
           endif
           sss=syy+szz
           qxx=qt1xx(k,i,j)
           qt1xx(k,i,j)=qxx*et+(wtp-wts)*sss*et1
           taoxx=taoxx+cl*sss-qxx-qt1xx(k,i,j)
           qyy=qt1yy(k,i,j)
           qt1yy(k,i,j)=qyy*et+(wtp*sss-wts*szz)*et1
           taoyy=taoyy+sm2*syy+cl*sss-qyy-qt1yy(k,i,j)
           qzz=qt1zz(k,i,j)
           qt1zz(k,i,j)=qzz*et+(wtp*sss-wts*syy)*et1
           taozz=taozz+sm2*szz+cl*sss-qzz-qt1zz(k,i,j)
         endif
         sxx=dxh1(2,i)/ca*(v1x(k,i-1,j)-v1x(k,i,j))
         qxx=qt1xx_px(k,ib,j)
         qt1xx_px(k,ib,j)=qxx*et+wtp*sxx*et1
         t1xx_px(k,ib,j)=damp1*t1xx_px(k,ib,j)+ &
                         damp2*(pm*sxx-qxx-qt1xx_px(k,ib,j))
         t1xx(k,i,j)=taoxx+t1xx_px(k,ib,j)
         qyy=qt1yy_px(k,ib,j)
         qt1yy_px(k,ib,j)=qyy*et+(wtp-wts)*sxx*et1
         t1yy_px(k,ib,j)=damp1*t1yy_px(k,ib,j)+ &
                         damp2*(cl*sxx-qyy-qt1yy_px(k,ib,j))
         t1yy(k,i,j)=taoyy+t1yy_px(k,ib,j)
         t1zz(k,i,j)=taozz+t1yy_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_norm_PmlX_I
!
!-----------------------------------------------------------------------
subroutine stress_norm_PmlY_I
! Compute the velocity of PML-x-I region
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
 real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
!
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1),lby(2)
   kb=0
   do j=nd1_tyy(1+4*lb),nd1_tyy(2+4*lb)
     kb=kb+1
     jb=jb+1
     rti=drti1(kb,lb)
     kodd=2*mod(j+nyb1,2)+1
     do i=nd1_tyy(7),nd1_tyy(12)
       jkq=mod(i+nxb1,2)+kodd
       do k=nd1_tyy(13),nd1_tyy(18)
         damp2=1./(1.+damp1_y(k,i,lb)*rti)
         damp1=damp2*2.-1.
         inod=idmat1(k,i,j)
         cl=clamda(inod)
         sm2=2.*cmu(inod)
         pm=cl+sm2
         irw=jkq+4*mod(k,2)
         et=epdt(irw)
         et1=1.0-et
         wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw))
         wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         if(i>nd1_tyy(8) .and. i<nd1_tyy(11)) then
           sxx=dxh1(1,i)*v1x(k,i-2,j)+dxh1(2,i)*v1x(k,i-1,j)+ &
               dxh1(3,i)*v1x(k,i  ,j)+dxh1(4,i)*v1x(k,i+1,j)
         else
           sxx=0.0
         endif
         if(k==1) then
           szz=dzi1(2,k)/ca*(22.*v1z(k,i,j)-17.*v1z(k+1,i,j)- &
               9.*v1z(k+2,i,j)+5.*v1z(k+3,i,j)-v1z(k+4,i,j))/24.
         else if(k==nztop) then
           szz=dzi1(2,k)/ca*(v1z(k,i,j)-v1z(k+1,i,j))
         else
           szz=dzi1(1,k)*v1z(k-1,i,j)+dzi1(2,k)*v1z(k,  i,j)+ &
               dzi1(3,k)*v1z(k+1,i,j)+dzi1(4,k)*v1z(k+2,i,j)
         endif
         sss=sxx+szz
         qxx=qt1xx(k,i,j)
         qt1xx(k,i,j)=qxx*et+(wtp*sss-wts*szz)*et1
         taoxx=t1xx(k,i,j)-t1xx_py(k,i,jb)+sm2*sxx+cl*sss-qxx-qt1xx(k,i,j)
         qyy=qt1yy(k,i,j)
         qt1yy(k,i,j)=qyy*et+(wtp-wts)*sss*et1
         taoyy=t1yy(k,i,j)-t1yy_py(k,i,jb)+cl*sss-qyy-qt1yy(k,i,j)
         qzz=qt1zz(k,i,j)
         qt1zz(k,i,j)=qzz*et+(wtp*sss-wts*sxx)*et1
         taozz=t1zz(k,i,j)-t1xx_py(k,i,jb)+sm2*szz+cl*sss-qzz-qt1zz(k,i,j)
         syy=dyh1(2,j)/ca*(v1y(k,i,j-1)-v1y(k,i,j))
         qxx=qt1xx_py(k,i,jb)
         qt1xx_py(k,i,jb)=qxx*et+(wtp-wts)*syy*et1
         t1xx_py(k,i,jb)=damp1*t1xx_py(k,i,jb)+ &
                         damp2*(cl*syy-qxx-qt1xx_py(k,i,jb))
         t1xx(k,i,j)=taoxx+t1xx_py(k,i,jb)
         t1zz(k,i,j)=taozz+t1xx_py(k,i,jb)
         qyy=qt1yy_py(k,i,jb)
         qt1yy_py(k,i,jb)=qyy*et+wtp*syy*et1
         t1yy_py(k,i,jb)=damp1*t1yy_py(k,i,jb)+ &
                         damp2*(pm*syy-qyy-qt1yy_py(k,i,jb))
         t1yy(k,i,j)=taoyy+t1yy_py(k,i,jb)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_norm_PmlY_I
!
!-----------------------------------------------------------------------
subroutine stress_xy_PmlX_I
! Compute the Stress-xy at region of PML-x-I
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
 real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
!
 if ( lbx(1)>lbx(2) ) return
 do j=nd1_txy(1),nd1_txy(6)
   kodd=2*mod(j+nyb1,2)+1
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd1_txy(7+4*lb),nd1_txy(8+4*lb)
       kb=kb+1
       ib=ib+1
       rth=drth1(kb,lb)
       jkq=mod(i+nxb1,2)+kodd
       do k=nd1_txy(13),nd1_txy(18)
         damp2=1./(1.+damp1_x(k,j,lb)*rth)
         damp1=damp2*2.-1.
         inod=idmat1(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat1(k,i+1,j+1)))
         irw=jkq+4*mod(k,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxy=t1xy(k,i,j)-t1xy_px(k,ib,j)
         if(j>nd1_txy(2) .and. j<nd1_txy(5)) then
           cusxy=(dyi1(1,j)*v1x(k,i,j-1)+dyi1(2,j)*v1x(k,i,j)+ &
                  dyi1(3,j)*v1x(k,i,j+1)+dyi1(4,j)*v1x(k,i,j+2))*sm
           qxy=qt1xy(k,i,j)
           qt1xy(k,i,j)=qxy*et+dmws*cusxy*et1
           taoxy=taoxy+cusxy-qxy-qt1xy(k,i,j)
         endif
         cusxy=sm*dxi1(2,i)/ca*(v1y(k,i,j)-v1y(k,i+1,j))
         qxy=qt1xy_px(k,ib,j)
         qt1xy_px(k,ib,j)=qxy*et+dmws*cusxy*et1
         t1xy_px(k,ib,j)=damp1*t1xy_px(k,ib,j)+ &
                         damp2*(cusxy-qxy-qt1xy_px(k,ib,j))
         t1xy(k,i,j)=taoxy+t1xy_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_xy_PmlX_I
!
!-----------------------------------------------------------------------
subroutine stress_xy_PmlY_I
! Compute the Stress-xy at region of PML-y-I
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
 real:: taoxy,cusyx,qxy,rth,damp2,damp1,sm,dmws,et,et1
!
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1), lby(2)
   kb=0
   do j=nd1_txy(1+4*lb),nd1_txy(2+4*lb)
     kb=kb+1
     jb=jb+1
     rth=drth1(kb,lb)
     kodd=2*mod(j+nyb1,2)+1
     do i=nd1_txy(7),nd1_txy(12)
       jkq=mod(i+nxb1,2)+kodd
       do k=nd1_txy(13),nd1_txy(18)
         damp2=1./(1.+damp1_y(k,i,lb)*rth)
         damp1=damp2*2.-1.
         inod=idmat1(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat1(k,i+1,j+1)))
         irw=jkq+4*mod(k,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxy=t1xy(k,i,j)-t1xy_py(k,i,jb)
         if(i>nd1_txy(8) .and. i<nd1_txy(11)) then
           cusyx=(dxi1(1,i)*v1y(k,i-1,j)+dxi1(2,i)*v1y(k,i,j)+ &
                  dxi1(3,i)*v1y(k,i+1,j)+dxi1(4,i)*v1y(k,i+2,j))*sm
           qxy=qt1xy(k,i,j)
           qt1xy(k,i,j)=qxy*et+dmws*cusyx*et1
           taoxy=taoxy+cusyx-qxy-qt1xy(k,i,j)
         endif
         cusyx=sm*dyi1(2,j)/ca*(v1x(k,i,j)-v1x(k,i,j+1))
         qxy=qt1xy_py(k,i,jb)
         qt1xy_py(k,i,jb)=qxy*et+dmws*cusyx*et1
         t1xy_py(k,i,jb)=damp1*t1xy_py(k,i,jb)+ &
                         damp2*(cusyx-qxy-qt1xy_py(k,i,jb))
         t1xy(k,i,j)=taoxy+t1xy_py(k,i,jb)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_xy_PmlY_I
!
!-----------------------------------------------------------------------
subroutine stress_xz_PmlX_I
! Compute the stress-xz at PML-x-I region
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
 real:: taoxz,cusxz,dvxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
!
 if ( lbx(1)>lbx(2) ) return
 do j=nd1_txz(1),nd1_txz(6)
   kodd=2*mod(j+nyb1,2)+1
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd1_txz(7+4*lb),nd1_txz(8+4*lb)
       kb=kb+1
       ib=ib+1
       rth=drth1(kb,lb)
       jkq=mod(i+nxb1,2)+kodd
       do k=nd1_txz(13),nd1_txz(18)
         damp2=1./(1.+damp1_x(k,j,lb)*rth)
         damp1=damp2*2.-1.
         inod=idmat1(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i+1,j)))
         irw=jkq+4*mod(k,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         if(k<nztop) then 
           dvxz=dzh1(1,k)*v1x(k-2,i,j)+dzh1(2,k)*v1x(k-1,i,j)+ &
                dzh1(3,k)*v1x(k,  i,j)+dzh1(4,k)*v1x(k+1,i,j)
         else 
           dvxz=dzh1(2,k)/ca*(v1x(k-1,i,j)-v1x(k,i,j))
         endif
         cusxz=dvxz*sm
         qxz=qt1xz(k,i,j)
         qt1xz(k,i,j)=qxz*et+dmws*cusxz*et1
         taoxz=t1xz(k,i,j)-t1xz_px(k,ib,j)+cusxz-qxz-qt1xz(k,i,j)
         cusxz=sm*dxi1(2,i)/ca*(v1z(k,i,j)-v1z(k,i+1,j))
         qxz=qt1xz_px(k,ib,j)
         qt1xz_px(k,ib,j)=qxz*et+dmws*cusxz*et1
         t1xz_px(k,ib,j)=damp1*t1xz_px(k,ib,j)+ &
                         damp2*(cusxz-qxz-qt1xz_px(k,ib,j))
         t1xz(k,i,j)=taoxz+t1xz_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_xz_PmlX_I
!
!-----------------------------------------------------------------------
subroutine stress_xz_PmlY_I
! Compute the stress-xz at PML-y-I region
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kodd,jkq,inod,irw
 real:: cusxz,dvxz,dvzx,qxz,sm,dmws,et,et1
!
 if( lby(1)>lby(2) ) return
 do lb=lby(1),lby(2)
 do j=nd1_txz(1+4*lb),nd1_txz(2+4*lb)
   kodd=2*mod(j+nyb1,2)+1
   do i=nd1_txz(9),nd1_txz(10)
     jkq=mod(i+nxb1,2)+kodd
     do k=nd1_txz(13),nd1_txz(18)
       inod=idmat1(k,i,j)
       sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i+1,j)))
       irw=jkq+4*mod(k,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       dvzx=dxi1(1,i)*v1z(k,i-1,j)+dxi1(2,i)*v1z(k,i,  j)+ &
            dxi1(3,i)*v1z(k,i+1,j)+dxi1(4,i)*v1z(k,i+2,j)
       if(k<nztop) then 
         dvxz=dzh1(1,k)*v1x(k-2,i,j)+dzh1(2,k)*v1x(k-1,i,j)+ &
              dzh1(3,k)*v1x(k,  i,j)+dzh1(4,k)*v1x(k+1,i,j)
       else
         dvxz=dzh1(2,k)/ca*(v1x(k-1,i,j)-v1x(k,i,j))
       endif
       cusxz=(dvzx+dvxz)*sm
       qxz=qt1xz(k,i,j)
       qt1xz(k,i,j)=qxz*et+dmws*cusxz*et1
       t1xz(k,i,j)=t1xz(k,i,j)+cusxz-qxz-qt1xz(k,i,j)
     enddo
   enddo
   enddo
 enddo
 return
end subroutine stress_xz_PmlY_I
!
!-----------------------------------------------------------------------
subroutine stress_yz_PmlX_I
! Compute the stress-yz at PML-x-I region
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kodd,jkq,inod,irw
 real:: cusyz,dvyz,dvzy,qyz,sm,dmws,et,et1
!
 if( lbx(1)>lbx(2) ) return
 do j=nd1_tyz(3),nd1_tyz(4)
   kodd=2*mod(j+nyb1,2)+1
   do lb=lbx(1),lbx(2)
   do i=nd1_tyz(7+4*lb),nd1_tyz(8+4*lb)
     jkq=mod(i+nxb1,2)+kodd
     do k=nd1_tyz(13),nd1_tyz(18)
       inod=idmat1(k,i,j)
       sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i,j+1)))
       irw=jkq+4*mod(k,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       dvzy=dyi1(1,j)*v1z(k,i,j-1)+dyi1(2,j)*v1z(k,i,j  )+ &
            dyi1(3,j)*v1z(k,i,j+1)+dyi1(4,j)*v1z(k,i,j+2)
       if(k<nztop) then
         dvyz=dzh1(1,k)*v1y(k-2,i,j)+dzh1(2,k)*v1y(k-1,i,j)+ &
              dzh1(3,k)*v1y(k,  i,j)+dzh1(4,k)*v1y(k+1,i,j)
       else 
         dvyz=dzh1(2,k)/ca*(v1y(k-1,i,j)-v1y(k,i,j))
       endif
       cusyz=(dvzy+dvyz)*sm
       qyz=qt1yz(k,i,j)
       qt1yz(k,i,j)=qyz*et+dmws*cusyz*et1
       t1yz(k,i,j)=t1yz(k,i,j)+cusyz-qyz-qt1yz(k,i,j)
     enddo
   enddo
   enddo
 enddo
 return
end subroutine stress_yz_PmlX_I
!
!-----------------------------------------------------------------------
subroutine stress_yz_PmlY_I
! Compute the stress-yz at PML-y-I region
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
 real:: taoyz,cusyz,dvyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
!
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1),lby(2)
   kb=0
   do j=nd1_tyz(1+4*lb),nd1_tyz(2+4*lb)
     kb=kb+1
     jb=jb+1
     rth=drth1(kb,lb)
     kodd=2*mod(j+nyb1,2)+1
     do i=nd1_tyz(7),nd1_tyz(12)
       jkq=mod(i+nxb1,2)+kodd
       do k=nd1_tyz(13),nd1_tyz(18)
         damp2=1./(1.+damp1_y(k,i,lb)*rth)
         damp1=damp2*2.-1.
         inod=idmat1(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat1(k-1,i,j+1)))
         irw=jkq+4*mod(k,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         if(k<nztop) then
           dvyz=dzh1(1,k)*v1y(k-2,i,j)+dzh1(2,k)*v1y(k-1,i,j)+ &
                dzh1(3,k)*v1y(k,  i,j)+dzh1(4,k)*v1y(k+1,i,j)
         else
           dvyz=dzh1(2,k)/ca*(v1y(k-1,i,j)-v1y(k,i,j))
         endif
         cusyz=dvyz*sm
         qyz=qt1yz(k,i,j)
         qt1yz(k,i,j)=qyz*et+dmws*cusyz*et1
         taoyz=t1yz(k,i,j)-t1yz_py(k,i,jb)+cusyz-qyz-qt1yz(k,i,j)
         cusyz=sm*dyi1(2,j)/ca*(v1z(k,i,j)-v1z(k,i,j+1))
         qyz=qt1yz_py(k,i,jb)
         qt1yz_py(k,i,jb)=qyz*et+dmws*cusyz*et1
         t1yz_py(k,i,jb)=damp1*t1yz_py(k,i,jb)+ &
                         damp2*(cusyz-qyz-qt1yz_py(k,i,jb))
         t1yz(k,i,j)=taoyz+t1yz_py(k,i,jb)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_yz_PmlY_I
!
!-----------------------------------------------------------------------
subroutine stress_norm_PmlX_II
! Compute the Stress-norm at region of PML-x-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
 real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
!
 if( lbx(1)>lbx(2) ) return
 do j=nd2_tyy(1),nd2_tyy(6)
   kodd=2*mod(j+nyb2,2)+1
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd2_tyy(7+4*lb),nd2_tyy(8+4*lb)
       kb=kb+1
       ib=ib+1
       rti=drti2(kb,lb)
       jkq=mod(i+nxb2,2)+kodd
       do k=nd2_tyy(13),nd2_tyy(18)
         damp2=1./(1.+damp2_x(k,j,lb)*rti)
         damp1=damp2*2.0-1.0
         inod=idmat2(k,i,j)
         cl=clamda(inod)
         sm2=2.*cmu(inod)
         pm=cl+sm2
         irw=jkq+4*mod(k+nztop,2)
         et=epdt(irw)
         et1=1.0-et
         wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw))
         wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxx=t2xx(k,i,j)-t2xx_px(k,ib,j)
         taoyy=t2yy(k,i,j)-t2yy_px(k,ib,j)
         taozz=t2zz(k,i,j)-t2yy_px(k,ib,j)
         if(j>nd2_tyy(2) .and. j<nd2_tyy(5)) then
           syy=dyh2(1,j)*v2y(k,i,j-2)+dyh2(2,j)*v2y(k,i,j-1)+ &
               dyh2(3,j)*v2y(k,i  ,j)+dyh2(4,j)*v2y(k,i,j+1)
           if(k<nd2_tyy(17)) then
             szz=dzi2(1,k)*v2z(k-1,i,j)+dzi2(2,k)*v2z(k,  i,j)+ &
                 dzi2(3,k)*v2z(k+1,i,j)+dzi2(4,k)*v2z(k+2,i,j)
           else
             szz=0.0
           endif
           sss=syy+szz
           qxx=qt2xx(k,i,j)
           qt2xx(k,i,j)=qxx*et+(wtp-wts)*sss*et1
           taoxx=taoxx+cl*sss-qxx-qt2xx(k,i,j)
           qyy=qt2yy(k,i,j)
           qt2yy(k,i,j)=qyy*et+(wtp*sss-wts*szz)*et1
           taoyy=taoyy+sm2*syy+cl*sss-qyy-qt2yy(k,i,j)
           qzz=qt2zz(k,i,j)
           qt2zz(k,i,j)=qzz*et+(wtp*sss-wts*syy)*et1
           taozz=taozz+sm2*szz+cl*sss-qzz-qt2zz(k,i,j)
         endif
         sxx=dxh2(2,i)/ca*(v2x(k,i-1,j)-v2x(k,i,j))
         qxx=qt2xx_px(k,ib,j)
         qt2xx_px(k,ib,j)=qxx*et+wtp*sxx*et1
         t2xx_px(k,ib,j)=damp1*t2xx_px(k,ib,j)+ &
                         damp2*(pm*sxx-qxx-qt2xx_px(k,ib,j))
         t2xx(k,i,j)=taoxx+t2xx_px(k,ib,j)
         qyy=qt2yy_px(k,ib,j)
         qt2yy_px(k,ib,j)=qyy*et+(wtp-wts)*sxx*et1
         t2yy_px(k,ib,j)=damp1*t2yy_px(k,ib,j)+ &
                         damp2*(cl*sxx-qyy-qt2yy_px(k,ib,j))
         t2yy(k,i,j)=taoyy+t2yy_px(k,ib,j)
         t2zz(k,i,j)=taozz+t2yy_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_norm_PmlX_II
!
!-----------------------------------------------------------------------
subroutine stress_norm_PmlY_II
! Compute the stress-norm at region of PML-y-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
 real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
        rti,damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
!
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1),lby(2)
   kb=0
   do j=nd2_tyy(1+4*lb),nd2_tyy(2+4*lb)
     kb=kb+1
     jb=jb+1
     rti=drti2(kb,lb)
     kodd=2*mod(j+nyb2,2)+1
     do i=nd2_tyy(7),nd2_tyy(12)
       jkq=mod(i+nxb2,2)+kodd
       do k=nd2_tyy(13),nd2_tyy(18)
         damp2=1./(1.+damp2_y(k,i,lb)*rti)
         damp1=damp2*2.0-1.
         inod=idmat2(k,i,j)
         cl=clamda(inod)
         sm2=2.*cmu(inod)
         pm=cl+sm2
         irw=jkq+4*mod(k+nztop,2)
         et=epdt(irw)
         et1=1.0-et
         wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw))
         wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxx=t2xx(k,i,j)-t2xx_py(k,i,jb)
         taoyy=t2yy(k,i,j)-t2yy_py(k,i,jb)
         taozz=t2zz(k,i,j)-t2xx_py(k,i,jb)
         if(k<nd2_tyy(17)) then
           szz=dzi2(1,k)*v2z(k-1,i,j)+dzi2(2,k)*v2z(k,  i,j)+ &
               dzi2(3,k)*v2z(k+1,i,j)+dzi2(4,k)*v2z(k+2,i,j)
           if(i>nd2_tyy(8) .and. i<nd2_tyy(11)) then
             sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+ &
                 dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j)
           else
             sxx=0.0
           endif
           sss=sxx+szz
           qxx=qt2xx(k,i,j)
           qt2xx(k,i,j)=qxx*et+(wtp*sss-wts*szz)*et1
           taoxx=taoxx+sm2*sxx+cl*sss-qxx-qt2xx(k,i,j)
           qyy=qt2yy(k,i,j)
           qt2yy(k,i,j)=qyy*et+(wtp-wts)*sss*et1
           taoyy=taoyy+cl*sss-qyy-qt2yy(k,i,j)
           qzz=qt2zz(k,i,j)
           qt2zz(k,i,j)=qzz*et+(wtp*sss-wts*sxx)*et1
           taozz=taozz+sm2*szz+cl*sss-qzz-qt2zz(k,i,j)
         else
           if(i>nd2_tyy(8) .and. i<nd2_tyy(11)) then
             sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+ &
                 dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j)
             qxx=qt2xx(k,i,j)
             qt2xx(k,i,j)=qxx*et+wtp*sxx*et1
             taoxx=taoxx+pm*sxx-qxx-qt2xx(k,i,j)
             qyy=qt2yy(k,i,j)
             qt2yy(k,i,j)=qyy*et+(wtp-wts)*sxx*et1
             taoyy=taoyy+cl*sxx-qyy-qt2yy(k,i,j)
             qzz=qt2zz(k,i,j)
             qt2zz(k,i,j)=qzz*et+(wtp-wts)*sxx*et1
             taozz=taozz+cl*sxx-qzz-qt2zz(k,i,j)
           endif
         endif
         syy=dyh2(2,j)/ca*(v2y(k,i,j-1)-v2y(k,i,j))
         qxx=qt2xx_py(k,i,jb)
         qt2xx_py(k,i,jb)=qxx*et+(wtp-wts)*syy*et1
         t2xx_py(k,i,jb)=damp1*t2xx_py(k,i,jb)+ &
                         damp2*(cl*syy-qxx-qt2xx_py(k,i,jb))
         t2xx(k,i,j)=taoxx+t2xx_py(k,i,jb)
         t2zz(k,i,j)=taozz+t2xx_py(k,i,jb)
         qyy=qt2yy_py(k,i,jb)
         qt2yy_py(k,i,jb)=qyy*et+wtp*syy*et1
         t2yy_py(k,i,jb)=damp1*t2yy_py(k,i,jb)+ &
                         damp2*(pm*syy-qyy-qt2yy_py(k,i,jb))
         t2yy(k,i,j)=taoyy+t2yy_py(k,i,jb)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_norm_PmlY_II
!
!-----------------------------------------------------------------------
subroutine stress_norm_PmlZ_II
! Compute the stress-norm at region of PML-z-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
 real:: taoxx,taoyy,taozz,sxx,syy,szz,sss,qxx,qyy,qzz, &
        damp2,damp1,cl,sm2,pm,et,et1,wtp,wts
 do j=nd2_tyy(1),nd2_tyy(6)
   kodd=2*mod(j+nyb2,2)+1
   do i=nd2_tyy(7),nd2_tyy(12)
     jkq=mod(i+nxb2,2)+kodd
     kb=0
     do k=nd2_tyy(17),nd2_tyy(18)
       kb=kb+1
       damp2=1./(1.+damp2_z(i,j)*drth2(kb,1))
       damp1=damp2*2.-1.
       inod=idmat2(k,i,j)
       cl=clamda(inod)
       sm2=2.*cmu(inod)
       pm=cl+sm2
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       wtp= pm*qwp(inod)*(qwp(inod)*qwt1(irw)+qwt2(irw))
       wts=sm2*qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       taoxx=t2xx(k,i,j)-t2xx_pz(kb,i,j)
       taoyy=t2yy(k,i,j)-t2xx_pz(kb,i,j)
       taozz=t2zz(k,i,j)-t2zz_pz(kb,i,j)
       if(i>nd2_tyy(8) .and. i<nd2_tyy(11) .and. &
          j>nd2_tyy(2) .and. j<nd2_tyy(5)) then
         sxx=dxh2(1,i)*v2x(k,i-2,j)+dxh2(2,i)*v2x(k,i-1,j)+ &
             dxh2(3,i)*v2x(k,i  ,j)+dxh2(4,i)*v2x(k,i+1,j)
         syy=dyh2(1,j)*v2y(k,i,j-2)+dyh2(2,j)*v2y(k,i,j-1)+ &
             dyh2(3,j)*v2y(k,i  ,j)+dyh2(4,j)*v2y(k,i,j+1)
         sss=sxx+syy
         qxx=qt2xx(k,i,j)
         qt2xx(k,i,j)=qxx*et+(wtp*sss-wts*syy)*et1
         taoxx=taoxx+sm2*sxx+cl*sss-qxx-qt2xx(k,i,j)
         qyy=qt2yy(k,i,j)
         qt2yy(k,i,j)=qyy*et+(wtp*sss-wts*sxx)*et1
         taoyy=taoyy+sm2*syy+cl*sss-qyy-qt2yy(k,i,j)
         qzz=qt2zz(k,i,j)
         qt2zz(k,i,j)=qzz*et+(wtp-wts)*sss*et1
         taozz=taozz+cl*sss-qzz-qt2zz(k,i,j)
       endif
       szz=dzi2(2,k)/ca*(v2z(k,i,j)-v2z(k+1,i,j))
       qxx=qt2xx_pz(kb,i,j)
       qt2xx_pz(kb,i,j)=qxx*et+(wtp-wts)*szz*et1
       t2xx_pz(kb,i,j)=damp1*t2xx_pz(kb,i,j)+ &
                       damp2*(cl*szz-qxx-qt2xx_pz(kb,i,j))
       t2xx(k,i,j)=taoxx+t2xx_pz(kb,i,j)
       t2yy(k,i,j)=taoyy+t2xx_pz(kb,i,j)
       qzz=qt2zz_pz(kb,i,j)
       qt2zz_pz(kb,i,j)=qzz*et+wtp*szz*et1
       t2zz_pz(kb,i,j)=damp1*t2zz_pz(kb,i,j)+ &
                       damp2*(pm*szz-qzz-qt2zz_pz(kb,i,j))
       t2zz(k,i,j)=taozz+t2zz_pz(kb,i,j)
     enddo
   enddo
 enddo
 return
end subroutine stress_norm_PmlZ_II
!
!-----------------------------------------------------------------------
subroutine stress_xy_PmlX_II
! Compute the Stress-xy at region of PML-x-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
 real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
!
 if( lbx(1)>lbx(2) ) return
 do j=nd2_txy(1),nd2_txy(6)
   kodd=2*mod(j+nyb2,2)+1
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd2_txy(7+4*lb),nd2_txy(8+4*lb)
       kb=kb+1
       ib=ib+1
       rth=drth2(kb,lb)
       jkq=mod(i+nxb2,2)+kodd
       do k=nd2_txy(13),nd2_txy(18)
         damp2=1./(1.+damp2_x(k,j,lb)*rth)
         damp1=damp2*2.0-1.
         inod=idmat2(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat2(k,i+1,j+1)))
         irw=jkq+4*mod(k+nztop,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxy=t2xy(k,i,j)-t2xy_px(k,ib,j)
         if(j>nd2_txy(2) .and. j<nd2_txy(5)) then
           cusxy=(dyi2(1,j)*v2x(k,i,j-1)+dyi2(2,j)*v2x(k,i,j)+ &
                  dyi2(3,j)*v2x(k,i,j+1)+dyi2(4,j)*v2x(k,i,j+2))*sm
           qxy=qt2xy(k,i,j)
           qt2xy(k,i,j)=qxy*et+dmws*cusxy*et1
           taoxy=taoxy+cusxy-qxy-qt2xy(k,i,j)
         endif
         cusxy=sm*dxi2(2,i)/ca*(v2y(k,i,j)-v2y(k,i+1,j))
         qxy=qt2xy_px(k,ib,j)
         qt2xy_px(k,ib,j)=qxy*et+dmws*cusxy*et1
         t2xy_px(k,ib,j)=damp1*t2xy_px(k,ib,j)+ &
                         damp2*(cusxy-qxy-qt2xy_px(k,ib,j))
         t2xy(k,i,j)=taoxy+t2xy_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_xy_PmlX_II
!
!-----------------------------------------------------------------------
subroutine stress_xy_PmlY_II
! Compute the Stress-xy at region of PML-y-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
 real:: taoxy,cusxy,qxy,rth,damp2,damp1,sm,dmws,et,et1
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1),lby(2)
   kb=0
   do j=nd2_txy(1+4*lb),nd2_txy(2+4*lb)
     kb=kb+1
     jb=jb+1
     rth=drth2(kb,lb)
     kodd=2*mod(j+nyb2,2)+1
     do i=nd2_txy(7),nd2_txy(12)
       jkq=mod(i+nxb2,2)+kodd
       do k=nd2_txy(13),nd2_txy(18)
         damp2=1./(1.+damp2_y(k,i,lb)*rth)
         damp1=damp2*2.-1.
         inod=idmat2(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat2(k,i+1,j+1)))
         irw=jkq+4*mod(k+nztop,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxy=t2xy(k,i,j)-t2xy_py(k,i,jb)
         if(i>nd2_txy(8) .and. i<nd2_txy(11)) then
           cusxy=(dxi2(1,i)*v2y(k,i-1,j)+dxi2(2,i)*v2y(k,i,j)+ &
                  dxi2(3,i)*v2y(k,i+1,j)+dxi2(4,i)*v2y(k,i+2,j))*sm
           qxy=qt2xy(k,i,j)
           qt2xy(k,i,j)=qxy*et+dmws*cusxy*et1
           taoxy=taoxy+cusxy-qxy-qt2xy(k,i,j)
         endif
         cusxy=sm*dyi2(2,j)/ca*(v2x(k,i,j)-v2x(k,i,j+1))
         qxy=qt2xy_py(k,i,jb)
         qt2xy_py(k,i,jb)=qxy*et+dmws*cusxy*et1
         t2xy_py(k,i,jb)=damp1*t2xy_py(k,i,jb)+ &
                         damp2*(cusxy-qxy-qt2xy_py(k,i,jb))
         t2xy(k,i,j)=taoxy+t2xy_py(k,i,jb)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_xy_PmlY_II
!
!-----------------------------------------------------------------------
subroutine stress_xy_PmlZ_II
!  Compute the Stress-xy at region of PML-z-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kodd,jkq,inod,irw
 real:: cusxy,qxy,sm,dmws,et,et1
 do j=nd2_txy(3),nd2_txy(4)
   kodd=2*mod(j+nyb2,2)+1
   do i=nd2_txy(9),nd2_txy(10)
     jkq=mod(i+nxb2,2)+kodd
     do k=nd2_txy(17),nd2_txy(18)
       inod=idmat2(k,i,j)
       sm=2./(1./cmu(inod)+1./cmu(idmat2(k,i+1,j+1)))
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       cusxy=(dxi2(1,i)*v2y(k,i-1,j)+dxi2(2,i)*v2y(k,i,  j)+ &
              dxi2(3,i)*v2y(k,i+1,j)+dxi2(4,i)*v2y(k,i+2,j)+ &
              dyi2(1,j)*v2x(k,i,j-1)+dyi2(2,j)*v2x(k,i,j  )+ &
              dyi2(3,j)*v2x(k,i,j+1)+dyi2(4,j)*v2x(k,i,j+2))*sm
       qxy=qt2xy(k,i,j)
       qt2xy(k,i,j)=qxy*et+dmws*cusxy*et1
       t2xy(k,i,j)=t2xy(k,i,j)+cusxy-qxy-qt2xy(k,i,j)
     enddo
   enddo
 enddo
 return
end subroutine stress_xy_PmlZ_II
!
!-----------------------------------------------------------------------
subroutine stress_xz_PmlX_II
! Compute the stress-xz at region of PML-x-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,ib,kb,kodd,jkq,inod,irw
 real:: taoxz,cusxz,qxz,rth,damp2,damp1,sm,dmws,et,et1
!
 if( lbx(1)>lbx(2) ) return
 do j=nd2_txz(1),nd2_txz(6)
   kodd=2*mod(j+nyb2,2)+1
   ib=0
   do lb=lbx(1),lbx(2)
     kb=0
     do i=nd2_txz(7+4*lb),nd2_txz(8+4*lb)
       kb=kb+1
       ib=ib+1
       rth=drth2(kb,lb)
       jkq=mod(i+nxb2,2)+kodd
       do k=nd2_txz(13),nd2_txz(18)
         damp2=1./(1.+damp2_x(k,j,lb)*rth)
         damp1=damp2*2.-1.
         inod=idmat2(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i+1,j)))
         irw=jkq+4*mod(k+nztop,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoxz=t2xz(k,i,j)-t2xz_px(k,ib,j)
         if(k<nd2_txz(17)) then
           cusxz=(dzh2(1,k)*v2x(k-2,i,j)+dzh2(2,k)*v2x(k-1,i,j)+ &
                  dzh2(3,k)*v2x(k,i,j)+dzh2(4,k)*v2x(k+1,i,j))*sm
           qxz=qt2xz(k,i,j)
           qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1
           taoxz=taoxz+cusxz-qxz-qt2xz(k,i,j)
         endif
         cusxz=sm*dxi2(2,i)/ca*(v2z(k,i,j)-v2z(k,i+1,j))
         qxz=qt2xz_px(k,ib,j)
         qt2xz_px(k,ib,j)=qxz*et+dmws*cusxz*et1
         t2xz_px(k,ib,j)=damp1*t2xz_px(k,ib,j)+ &
                         damp2*(cusxz-qxz-qt2xz_px(k,ib,j))
         t2xz(k,i,j)=taoxz+t2xz_px(k,ib,j)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_xz_PmlX_II
!
!-----------------------------------------------------------------------
subroutine stress_xz_PmlY_II
! Compute the stress-xz at region of PML-y-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kodd,jkq,inod,irw
 real:: dvxz,dvzx,cusxz,qxz,sm,dmws,et,et1
!
 if( lby(1)>lby(2) ) return
 do lb=lby(1),lby(2)
 do j=nd2_txz(1+4*lb),nd2_txz(2+4*lb)
   kodd=2*mod(j+nyb2,2)+1
   do i=nd2_txz(9),nd2_txz(10)
     jkq=mod(i+nxb2,2)+kodd
     do k=nd2_txz(13),nd2_txz(16)
       inod=idmat2(k,i,j)
       sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i+1,j)))
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       dvzx=dxi2(1,i)*v2z(k,i-1,j)+dxi2(2,i)*v2z(k,i,  j)+ &
            dxi2(3,i)*v2z(k,i+1,j)+dxi2(4,i)*v2z(k,i+2,j)
       dvxz=dzh2(1,k)*v2x(k-2,i,j)+dzh2(2,k)*v2x(k-1,i,j)+ &
            dzh2(3,k)*v2x(k,  i,j)+dzh2(4,k)*v2x(k+1,i,j)
       cusxz=(dvzx+dvxz)*sm
       qxz=qt2xz(k,i,j)
       qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1
       t2xz(k,i,j)=t2xz(k,i,j)+cusxz-qxz-qt2xz(k,i,j)
     enddo
   enddo
 enddo
 enddo
 return
end subroutine stress_xz_PmlY_II
!-----------------------------------------------------------------------
subroutine stress_xz_PmlZ_II
! Compute the stress-xz at region of PML-z-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
 real:: taoxz,cusxz,qxz,damp2,damp1,sm,dmws,et,et1
!
 do j=nd2_txz(1),nd2_txz(6)
   kodd=2*mod(j+nyb2,2)+1
   do i=nd2_txz(7),nd2_txz(12)
     jkq=mod(i+nxb2,2)+kodd
     kb=0
     do k=nd2_txz(17),nd2_txz(18)
       kb=kb+1
       damp2=1./(1.+damp2_z(i,j)*drti2(kb,1))
       damp1=damp2*2.-1.
       inod=idmat2(k,i,j)
       sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i+1,j)))
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       taoxz=t2xz(k,i,j)-t2xz_pz(kb,i,j)
       if(i>nd2_txz(8) .and. i<nd2_txz(11)) then
         cusxz=(dxi2(1,i)*v2z(k,i-1,j)+dxi2(2,i)*v2z(k,i,  j)+ &
                dxi2(3,i)*v2z(k,i+1,j)+dxi2(4,i)*v2z(k,i+2,j))*sm
         qxz=qt2xz(k,i,j)
         qt2xz(k,i,j)=qxz*et+dmws*cusxz*et1
         taoxz=taoxz+cusxz-qxz-qt2xz(k,i,j)
       endif
       cusxz=sm*dzh2(2,k)/ca*(v2x(k-1,i,j)-v2x(k,i,j))
       qxz=qt2xz_pz(kb,i,j)
       qt2xz_pz(kb,i,j)=qxz*et+dmws*cusxz*et1
       t2xz_pz(kb,i,j)=damp1*t2xz_pz(kb,i,j)+ &
                       damp2*(cusxz-qxz-qt2xz_pz(kb,i,j))
       t2xz(k,i,j)=taoxz+t2xz_pz(kb,i,j)
     enddo
   enddo
 enddo
 return
end subroutine stress_xz_PmlZ_II
!
!-----------------------------------------------------------------------
subroutine stress_yz_PmlX_II
! Compute the stress-yz at region of PML-x-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kodd,jkq,inod,irw
 real:: cusyz,qyz,sm,dmws,et,et1
!
 if( lbx(1)>lbx(2) ) return
 do j=nd2_tyz(3),nd2_tyz(4)
   kodd=2*mod(j+nyb2,2)+1
   do lb=lbx(1),lbx(2)
   do i=nd2_tyz(7+4*lb),nd2_tyz(8+4*lb)
     jkq=mod(i+nxb2,2)+kodd
     do k=nd2_tyz(13),nd2_tyz(16)
       inod=idmat2(k,i,j)
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       cusyz=(dyi2(1,j)*v2z(k,i,j-1)+dyi2(2,j)*v2z(k,i,j  )+ &
              dyi2(3,j)*v2z(k,i,j+1)+dyi2(4,j)*v2z(k,i,j+2)+ &
              dzh2(1,k)*v2y(k-2,i,j)+dzh2(2,k)*v2y(k-1,i,j)+ &
              dzh2(3,k)*v2y(k,  i,j)+dzh2(4,k)*v2y(k+1,i,j))/ &
              (.5/cmu(inod)+.5/cmu(idmat2(k-1,i,j+1)))
       qyz=qt2yz(k,i,j)
       qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1
       t2yz(k,i,j)=t2yz(k,i,j)+cusyz-qyz-qt2yz(k,i,j)
     enddo
   enddo
   enddo
 enddo
 return
end subroutine stress_yz_PmlX_II
!
!-----------------------------------------------------------------------
subroutine stress_yz_PmlY_II
! Compute the stress-yz at region of PML-y-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,jb,kb,kodd,jkq,inod,irw
 real:: taoyz,cusyz,qyz,rth,damp2,damp1,sm,dmws,et,et1
!
 if( lby(1)>lby(2) ) return
 jb=0
 do lb=lby(1),lby(2)
   kb=0
   do j=nd2_tyz(1+4*lb),nd2_tyz(2+4*lb)
     kb=kb+1
     jb=jb+1
     rth=drth2(kb,lb)
     kodd=2*mod(j+nyb2,2)+1
     do i=nd2_tyz(7),nd2_tyz(12)
       jkq=mod(i+nxb2,2)+kodd
       do k=nd2_tyz(13),nd2_tyz(18)
         damp2=1./(1.+damp2_y(k,i,lb)*rth)
         damp1=damp2*2.-1.
         inod=idmat2(k,i,j)
         sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i,j+1)))
         irw=jkq+4*mod(k+nztop,2)
         et=epdt(irw)
         et1=1.0-et
         dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
         taoyz=t2yz(k,i,j)-t2yz_py(k,i,jb)
         if(k<nd2_tyz(17)) then
           cusyz=(dzh2(1,k)*v2y(k-2,i,j)+dzh2(2,k)*v2y(k-1,i,j)+ &
                  dzh2(3,k)*v2y(k,  i,j)+dzh2(4,k)*v2y(k+1,i,j))*sm
           qyz=qt2yz(k,i,j)
           qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1
           taoyz=taoyz+cusyz-qyz-qt2yz(k,i,j)
         endif
         cusyz=sm*dyi2(2,j)/ca*(v2z(k,i,j)-v2z(k,i,j+1))
         qyz=qt2yz_py(k,i,jb)
         qt2yz_py(k,i,jb)=qyz*et+dmws*cusyz*et1
         t2yz_py(k,i,jb)=damp1*t2yz_py(k,i,jb)+ &
                         damp2*(cusyz-qyz-qt2yz_py(k,i,jb))
         t2yz(k,i,j)=taoyz+t2yz_py(k,i,jb)
       enddo
     enddo
   enddo
 enddo
 return
end subroutine stress_yz_PmlY_II
!
!-----------------------------------------------------------------------
subroutine stress_yz_PmlZ_II
! Compute the stress-yz at region of PML-y-II
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,lb,kb,kodd,jkq,inod,irw
 real:: taoyz,cusyz,qyz,damp2,damp1,sm,dmws,et,et1
!
 do j=nd2_tyz(1),nd2_tyz(6)
   kodd=2*mod(j+nyb2,2)+1
   do i=nd2_tyz(7),nd2_tyz(12)
     jkq=mod(i+nxb2,2)+kodd
     kb=0
     do k=nd2_tyz(17),nd2_tyz(18)
       kb=kb+1
       damp2=1./(1.+damp2_z(i,j)*drti2(kb,1))
       damp1=damp2*2.-1.
       inod=idmat2(k,i,j)
       sm=2./(1./cmu(inod)+1./cmu(idmat2(k-1,i,j+1)))
       irw=jkq+4*mod(k+nztop,2)
       et=epdt(irw)
       et1=1.0-et
       dmws=qws(inod)*(qws(inod)*qwt1(irw)+qwt2(irw))
       taoyz=t2yz(k,i,j)-t2yz_pz(kb,i,j)
       if(j>nd2_tyz(2) .and. j<nd2_tyz(5)) then
         cusyz=(dyi2(1,j)*v2z(k,i,j-1)+dyi2(2,j)*v2z(k,i,j)+ &
                dyi2(3,j)*v2z(k,i,j+1)+dyi2(4,j)*v2z(k,i,j+2))*sm
         qyz=qt2yz(k,i,j)
         qt2yz(k,i,j)=qyz*et+dmws*cusyz*et1
         taoyz=taoyz+cusyz-qyz-qt2yz(k,i,j)
       endif
       cusyz=sm*dzh2(2,k)/ca*(v2y(k-1,i,j)-v2y(k,i,j))
       qyz=qt2yz_pz(kb,i,j)
       qt2yz_pz(kb,i,j)=qyz*et+dmws*cusyz*et1
       t2yz_pz(kb,i,j)=damp1*t2yz_pz(kb,i,j)+ &
                       damp2*(cusyz-qyz-qt2yz_pz(kb,i,j))
       t2yz(k,i,j)=taoyz+t2yz_pz(kb,i,j)
     enddo
   enddo
 enddo
 return
end subroutine stress_yz_PmlZ_II
!
!-----------------------------------------------------------------------
subroutine stress_interp
 use grid_node_comm
 use wave_field_comm
! use logging
 use metadata
 use ctimer
 implicit NONE
 interface
     subroutine interp_stress (neighb1, neighb2, neighb3, neighb4,&
                    nxbm1, nybm1, nxbtm, nybtm, nzbtm, nxtop, nytop, nztop, nz1p1) &
                    bind(C, name="interp_stress")
            use iso_c_binding
            integer(c_int), value, intent(in) :: nxbtm, nybtm, nzbtm , &
                            nxtop, nytop, nztop, nz1p1, &
                            neighb1, neighb2, neighb3, neighb4, nxbm1, nybm1 
     end subroutine interp_stress

     subroutine interp_stress1 (ntx1, nz1p1, nxbtm, nybtm, nzbtm, &
                                nxtop, nytop, nztop) bind (C, name="interp_stress1")
            use iso_c_binding
            integer(c_int), value, intent(in) :: ntx1, nz1p1, &
                                                nxbtm, nybtm, nzbtm, &
                                                nxtop, nytop, nztop 
     end subroutine interp_stress1

     subroutine interp_stress2 (nty1, nz1p1, nxbtm, nybtm, nzbtm, &
                                nxtop, nytop, nztop) bind (C, name="interp_stress2")
            use iso_c_binding
            integer(c_int), value, intent(in) :: nty1, nz1p1, &
                                                nxbtm, nybtm, nzbtm, &
                                                nxtop, nytop, nztop 
     end subroutine interp_stress2
 
     subroutine interp_stress3 (nxbtm, nybtm, nzbtm, &
                                nxtop, nytop, nztop) bind (C, name="interp_stress3")
            use iso_c_binding
            integer(c_int), value, intent(in) :: nxbtm, nybtm, nzbtm, &
                                                nxtop, nytop, nztop 
     end subroutine interp_stress3
 
 end interface
 integer:: i,j,ii,jj,ntx1,nty1
 real(c_double) :: gpu_tstart, gpu_tend, cpu_tstart, cpu_tend
 real(c_double):: total_gpu_marsh_time=0.0, total_cpu_marsh_time=0.0
 total_gpu_marsh_time =0.0
 total_cpu_marsh_time =0.0

!
 ntx1=nxbtm
 if(neighb(2) < 0) ntx1=nxbm1
 nty1=nybtm
 if(neighb(4) < 0) nty1=nybm1
!
#ifdef DISFD_GPU_MARSHALING
 if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
 call interp_stress1(ntx1, nz1p1, nxbtm, nybtm, nzbtm, nxtop, nytop, nztop);
! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU stress_interp1", gpu_tend-gpu_tstart); &
!        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
 call interp_stress2(nty1, nz1p1, nxbtm, nybtm, nzbtm, nxtop, nytop, nztop);
! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU stress_interp2", gpu_tend-gpu_tstart); &
!        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
 
! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
 call interp_stress3(nxbtm, nybtm, nzbtm, nxtop, nytop, nztop);
 if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU stress_interp3", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
 if(DETAILED_TIMING .eq. 1) then; call log_timing ("GPU Marshaling stress_interp", total_gpu_marsh_time); end if
#else
 j=-2
 do jj=1,nybtm
   j=j+3
   i=-1
   do ii=1,ntx1
     i=i+3
     t2xz(0,ii,jj)=t1xz(nztop,i,j)
     t2xz(1,ii,jj)=t1xz(nz1p1,i,j)
   enddo
 enddo
!
 j=-1
 do jj=1,nty1
   j=j+3
   i=-2
   do ii=1,nxbtm
     i=i+3
     t2yz(0,ii,jj)=t1yz(nztop,i,j)
     t2yz(1,ii,jj)=t1yz(nz1p1,i,j)
   enddo
 enddo
!
 j=-2
 do jj=1,nybtm
   j=j+3
   i=-2
   do ii=1,nxbtm
     i=i+3
     t2zz(0,ii,jj)=t1zz(nztop,i,j)
   enddo
 enddo
#endif
 return
end subroutine stress_interp
!
!-----------------------------------------------------------------------
subroutine pre_abc
! prepare for finding the velocity on the artifical boundary
 use grid_node_comm
 use wave_field_comm
 implicit NONE
 integer:: i,j,k,i1,j1,k1
!
! On the side of negative X-direction
 i=1
 i1=i+1
 do j=nxyabc(1),nxyabc(2)
   do k=1,nztop
     v1x(k,i,j)=abcx1(1,k,j)*v1x(k,i,j)+abcx1(2,k,j)*v1x(k,i1,j)
     v1y(k,i,j)=abcx1(3,k,j)*v1y(k,i,j)+abcx1(4,k,j)*v1y(k,i1,j)
     v1z(k,i,j)=abcx1(3,k,j)*v1z(k,i,j)+abcx1(4,k,j)*v1z(k,i1,j)
   enddo 
 enddo 
!
! On the side of possitive X-direction
 i=nxtop
 i1=i-1
 do j=nxyabc(3),nxyabc(4)
   do k=1,nztop
     v1x(k,i,j)=abcx1(5,k,j)*v1x(k,i,j)+abcx1(6,k,j)*v1x(k,i1,j)
     v1y(k,i,j)=abcx1(7,k,j)*v1y(k,i,j)+abcx1(8,k,j)*v1y(k,i1,j)
     v1z(k,i,j)=abcx1(7,k,j)*v1z(k,i,j)+abcx1(8,k,j)*v1z(k,i1,j)
   enddo 
 enddo
!
! On the side of  negative Y-direction
 j=1
 j1=j+1
 do i=nxyabc(5),nxyabc(6)
   do k=1,nztop
      v1x(k,i,j)=abcy1(3,k,i)*v1x(k,i,j)+abcy1(4,k,i)*v1x(k,i,j1)
      v1y(k,i,j)=abcy1(1,k,i)*v1y(k,i,j)+abcy1(2,k,i)*v1y(k,i,j1)
      v1z(k,i,j)=abcy1(3,k,i)*v1z(k,i,j)+abcy1(4,k,i)*v1z(k,i,j1)
   enddo
 enddo
!
! On the side of positive Y-direction
 j=nytop
 j1=j-1
 do i=nxyabc(7),nxyabc(8)
   do k=1,nztop
     v1x(k,i,j)=abcy1(7,k,i)*v1x(k,i,j)+abcy1(8,k,i)*v1x(k,i,j1)
     v1y(k,i,j)=abcy1(5,k,i)*v1y(k,i,j)+abcy1(6,k,i)*v1y(k,i,j1)
     v1z(k,i,j)=abcy1(7,k,i)*v1z(k,i,j)+abcy1(8,k,i)*v1z(k,i,j1)
   enddo
 enddo
!
! On the side of possitive Z-direction
 k=nzbtm
 k1=k-1
 do j=1,nybtm
   do i=1,nxbtm
     v2x(k,i,j)=abcz(3,i,j)*v2x(k,i,j)+abcz(4,i,j)*v2x(k1,i,j)
     v2y(k,i,j)=abcz(3,i,j)*v2y(k,i,j)+abcz(4,i,j)*v2y(k1,i,j)
     v2z(k,i,j)=abcz(1,i,j)*v2z(k,i,j)+abcz(2,i,j)*v2z(k1,i,j)
   enddo
 enddo
!
! On the side of negative X-direction
 i=1
 i1=i+1
 do j=nxyabc(9),nxyabc(10)
   do k=1,nzbm1
     v2x(k,i,j)=abcx2(1,k,j)*v2x(k,i,j)+abcx2(2,k,j)*v2x(k,i1,j)
     v2y(k,i,j)=abcx2(3,k,j)*v2y(k,i,j)+abcx2(4,k,j)*v2y(k,i1,j)
     v2z(k,i,j)=abcx2(3,k,j)*v2z(k,i,j)+abcx2(4,k,j)*v2z(k,i1,j)
   enddo
 enddo
!
! On the side of possitive X-direction
 i=nxbtm
 i1=i-1
 do j=nxyabc(11),nxyabc(12)
   do k=1,nzbm1
     v2x(k,i,j)=abcx2(5,k,j)*v2x(k,i,j)+abcx2(6,k,j)*v2x(k,i1,j)
     v2y(k,i,j)=abcx2(7,k,j)*v2y(k,i,j)+abcx2(8,k,j)*v2y(k,i1,j)
     v2z(k,i,j)=abcx2(7,k,j)*v2z(k,i,j)+abcx2(8,k,j)*v2z(k,i1,j)
   enddo 
 enddo
!
! On the side of negative Y-direction
 j=1
 j1=j+1
 do i=nxyabc(13),nxyabc(14)
   do k=1,nzbm1
      v2x(k,i,j)=abcy2(3,k,i)*v2x(k,i,j)+abcy2(4,k,i)*v2x(k,i,j1)
      v2y(k,i,j)=abcy2(1,k,i)*v2y(k,i,j)+abcy2(2,k,i)*v2y(k,i,j1)
      v2z(k,i,j)=abcy2(3,k,i)*v2z(k,i,j)+abcy2(4,k,i)*v2z(k,i,j1)
   enddo
 enddo
!
! On the side of positive Y-direction
 j=nybtm
 j1=j-1
 do i=nxyabc(15),nxyabc(16)
   do k=1,nzbm1
     v2x(k,i,j)=abcy2(7,k,i)*v2x(k,i,j)+abcy2(8,k,i)*v2x(k,i,j1)
     v2y(k,i,j)=abcy2(5,k,i)*v2y(k,i,j)+abcy2(6,k,i)*v2y(k,i,j1)
     v2z(k,i,j)=abcy2(7,k,i)*v2z(k,i,j)+abcy2(8,k,i)*v2z(k,i,j1)
   enddo
 enddo
 return
end subroutine pre_abc
!
!-----------------------------------------------------------------------
subroutine output_1(nrev3,vout)
! output the simulation at each time step
 use grid_node_comm
 use wave_field_comm
 use station_comm
 implicit NONE
 integer, intent(IN):: nrev3
 real, dimension(nrev3), intent(OUT):: vout
 integer:: ir,io,i,j,k
 real::rx1,rx11,rx2,rx21,ry1,ry11,ry2,ry21,rz1,rz11,rz2,rz21 
!
 do ir=1,nrecs
   rx1=xcor(1,ir)
   rx11=1.-rx1
   rx2=xcor(2,ir)
   rx21=1.-rx2
   ry1=ycor(1,ir)
   ry11=1.-ry1
   ry2=ycor(2,ir)
   ry21=1.-ry2
   rz1=zcor(1,ir)
   rz11=1.-rz1
   rz2=zcor(2,ir)
   rz21=1.-rz2
   io=(ir-1)*3   
   if(ipoz(1,ir) > nztop ) then
!- outputing Vx
     i=ipox(2,ir)
     j=ipoy(1,ir)
     k=ipoz(2,ir)-nztop
     vout(io+1)=((v2x(k,i,j)*rx2+v2x(k,i+1,j)*rx21)*ry1+ &
             (v2x(k,i,j+1  )*rx2+v2x(k,i+1,j+1  )*rx21)*ry11)*rz2+ &
            ((v2x(k+1,i,j  )*rx2+v2x(k+1,i+1,j  )*rx21)*ry1+ &
             (v2x(k+1,i,j+1)*rx2+v2x(k+1,i+1,j+1)*rx21)*ry11)*rz21
!- outputing Vy
     i=ipox(1,ir)
     j=ipoy(2,ir)
     vout(io+2)=((v2y(k,i,j)*rx1+v2y(k,i+1,j)*rx11)*ry2+ &
             (v2y(k,i,j+1  )*rx1+v2y(k,i+1,j+1  )*rx11)*ry21)*rz2+ &
            ((v2y(k+1,i,j  )*rx1+v2y(k+1,i+1,j  )*rx11)*ry2+ &
             (v2y(k+1,i,j+1)*rx1+v2y(k+1,i+1,j+1)*rx11)*ry21)*rz21
!- outputing Vz
     j=ipoy(1,ir)
     k=ipoz(1,ir)-nztop
     vout(io+3)=((v2z(k,i,j)*rx1+v2z(k,i+1,j)*rx11)*ry1+ &
             (v2z(k,i,j+1  )*rx1+v2z(k,i+1,j+1  )*rx11)*ry11)*rz1+ &
            ((v2z(k+1,i,j  )*rx1+v2z(k+1,i+1,j  )*rx11)*ry1+ &
             (v2z(k+1,i,j+1)*rx1+v2z(k+1,i+1,j+1)*rx11)*ry11)*rz11
   else
!- outputing Vx
     i=ipox(2,ir)
     j=ipoy(1,ir)
     k=ipoz(2,ir)
     vout(io+1)=((v1x(k,i,j)*rx2+v1x(k,i+1,j)*rx21)*ry1+ &
             (v1x(k,i,j+1  )*rx2+v1x(k,i+1,j+1  )*rx21)*ry11)*rz2+ &
            ((v1x(k+1,i,j  )*rx2+v1x(k+1,i+1,j  )*rx21)*ry1+ &
             (v1x(k+1,i,j+1)*rx2+v1x(k+1,i+1,j+1)*rx21)*ry11)*rz21
!- outputing Vy
     i=ipox(1,ir)
     j=ipoy(2,ir)
     vout(io+2)=((v1y(k,i,j)*rx1+v1y(k,i+1,j)*rx11)*ry2+ &
             (v1y(k,i,j+1  )*rx1+v1y(k,i+1,j+1  )*rx11)*ry21)*rz2+ &
            ((v1y(k+1,i,j  )*rx1+v1y(k+1,i+1,j  )*rx11)*ry2+ &
             (v1y(k+1,i,j+1)*rx1+v1y(k+1,i+1,j+1)*rx11)*ry21)*rz21
!- outputing Vz
     j=ipoy(1,ir)
     k=ipoz(1,ir)
     vout(io+3)=((v1z(k,i,j)*rx1+v1z(k,i+1,j)*rx11)*ry1+ &
             (v1z(k,i,j+1  )*rx1+v1z(k,i+1,j+1  )*rx11)*ry11)*rz1+ &
            ((v1z(k+1,i,j  )*rx1+v1z(k+1,i+1,j  )*rx11)*ry1+ &
             (v1z(k+1,i,j+1)*rx1+v1z(k+1,i+1,j+1)*rx11)*ry11)*rz11
   endif
 enddo
 return
end subroutine output_1
!
!-----------------------------------------------------------------------
subroutine output_2(nrev3,vout)
! output the simulation at each time step
 use grid_node_comm
 use wave_field_comm
 use station_comm
 implicit NONE
 integer, intent(IN):: nrev3
 real, dimension(nrev3), intent(OUT):: vout
 integer:: io,i,j,k,k1,kc,k2
!
 io=0
 do kc=1,nblock-1,2
   if(ipoz(2,kc)>0 .and. ipoy(2,kc)>0 .and. ipox(2,kc)>0) then
   do k=ipoz(1, kc), ipoz(2, kc), ipoz(3,kc) 
   do j=ipoy(1, kc), ipoy(2, kc), ipoy(3,kc) 
   do i=ipox(1, kc), ipox(2, kc), ipox(3,kc) 
     io=io+3;   k1=k-1
     vout(io-2)=0.25*(v1x(k1,i-1,j)+v1x(k,i-1,j)+v1x(k1,i,j)+v1x(k,i,j))
     vout(io-1)=0.25*(v1y(k1,i,j-1)+v1y(k,i,j-1)+v1y(k1,i,j)+v1y(k,i,j))
     vout(io  )=v1z(k,i,j)
   enddo
   enddo
   enddo
   endif
! Region 2
   k2=kc+1
   if(ipoz(2,k2)>0 .and. ipoy(2,k2)>0 .and. ipox(2,k2)>0) then
   do k=ipoz(1, k2), ipoz(2, k2), ipoz(3,k2) 
   do j=ipoy(1, k2), ipoy(2, k2), ipoy(3,k2) 
   do i=ipox(1, k2), ipox(2, k2), ipox(3,k2) 
     io=io+3;  k1=k-1
     vout(io-2)=0.25*(v2x(k1,i-1,j)+v2x(k,i-1,j)+v2x(k1,i,j)+v2x(k,i,j))
     vout(io-1)=0.25*(v2y(k1,i,j-1)+v2y(k,i,j-1)+v2y(k1,i,j)+v2y(k,i,j))
     vout(io  )=v2z(k,i,j)
   enddo
   enddo
   enddo
   endif
 enddo
 
end subroutine output_2
!
!-----------------------------------------------------------------------
subroutine consts
! compute the constants parameters
 use grid_node_comm
 implicit NONE
 integer:: nm
 real:: vp,vs,dens,sm,cl,prym,bm
 do nm=1,nmat
   vp=clamda(nm)
   vs=cmu(nm)
   dens=rho(nm)
   sm=dens*vs**2
   cl=dens*vp**2-2.*sm
!   pr=0.5*cl/(cl+sm)
!   ym=2.*(1.+pr)*sm
!   bm=cl+2.*sm/3.
! rho=1/density (buoyancy)
   rho(nm)=1/dens
! cmu=(shear mudules)
   cmu(nm)=sm
! clamda=(Lambda: One of the Lame constants)
   clamda(nm)=cl
 enddo
 return
end subroutine consts
!
!-----------------------------------------------------------------------
subroutine coef_abc(vs,gridsp,dt,damp0,cfabc)
! Determine the parameters of absorbing boundary condition
 implicit NONE
 real, intent(IN):: vs,gridsp,dt,damp0
 real, dimension(4), intent(OUT):: cfabc
 real:: vp,ss,ddd,damp
!
 damp=0.1*damp0
 vp=1.73*vs
 vp=vs
 ddd=1.0/(1.+damp)
 ss=vp*dt/gridsp
 cfabc(1)=(1.0-ss-damp)*ddd
 cfabc(2)=ss*ddd
 ss=vs*dt/gridsp
 cfabc(3)=(1-ss-damp)*ddd
 cfabc(4)=ss*ddd
 return
end subroutine coef_abc
!
!-----------------------------------------------------------------------
subroutine assign_source(xs,ys,zs,ix2,jy2,kz2,cf8,nsou)
 use grid_node_comm
 implicit NONE
 integer, intent(OUT):: ix2(72,4),jy2(72,4),kz2(72,4),nsou
 real, intent(IN):: xs,ys,zs
 real, dimension(72,4), intent(OUT):: cf8
 integer:: i,j,k,n1,n2,nx,ny,nsi,ijk,kzdf,knd,khf,ind,ihf,jnd,jhf
 integer, dimension(4):: nss
 real, dimension(2):: xrt,yrt,zrt,xrh,yrh,zrh
!
 kzdf=nztop
 call source_index(kzdf,nztol,gridz,zs,knd,zrt(1),khf,zrh(1))
 zrt(2)=1.-zrt(1)
 zrh(2)=1.-zrh(1)
! ijk=1 (Mxx, Myy, Mzz) or ijk=2, Mxy
 kzdf=khf-nztop
 if(kzdf > 0 ) then
   call source_index(kzdf,nxbtm,gridx2,xs,ind,xrt(1),ihf,xrh(1))
   call source_index(kzdf,nybtm,gridy2,ys,jnd,yrt(1),jhf,yrh(1))
 else
   call source_index(kzdf,nxtop,gridx1,xs,ind,xrt(1),ihf,xrh(1))
   call source_index(kzdf,nytop,gridy1,ys,jnd,yrt(1),jhf,yrh(1))
 endif
 xrt(2)=1.-xrt(1)
 xrh(2)=1.-xrh(1)
 yrt(2)=1.-yrt(1)
 yrh(2)=1.-yrh(1)
 ijk=1
 call distrib(kzdf,1,ind,jnd,khf,xrt,yrt,zrh, &
              ix2(1,ijk),jy2(1,ijk),kz2(1,ijk),cf8(1,ijk),nss(ijk))
 ijk=2
 call distrib(kzdf,2,ihf,jhf,khf,xrh,yrh,zrh, &
              ix2(1,ijk),jy2(1,ijk),kz2(1,ijk),cf8(1,ijk),nss(ijk))
! ijk=3 Mxz or ijk=4, Myz
 kzdf=knd-nztop
 if(kzdf>0) then
   call source_index(kzdf,nxbtm,gridx2,xs,ind,xrt(1),ihf,xrh(1))
   call source_index(kzdf,nybtm,gridy2,ys,jnd,yrt(1),jhf,yrh(1))
 else
   call source_index(kzdf,nxtop,gridx1,xs,ind,xrt(1),ihf,xrh(1))
   call source_index(kzdf,nytop,gridy1,ys,jnd,yrt(1),jhf,yrh(1))
 endif
 xrt(2)=1.-xrt(1)
 xrh(2)=1.-xrh(1)
 yrt(2)=1.-yrt(1)
 yrh(2)=1.-yrh(1)
 ijk=3
 call distrib(kzdf,3,ihf,jnd,knd,xrh,yrt,zrt, &
              ix2(1,ijk),jy2(1,ijk),kz2(1,ijk),cf8(1,ijk),nss(ijk))
 ijk=4 
 call distrib(kzdf,4,ind,jhf,knd,xrt,yrh,zrt, &
              ix2(1,ijk),jy2(1,ijk),kz2(1,ijk),cf8(1,ijk),nss(ijk))
 do ijk=1,4
   n1=0
   do nsi=1,nss(ijk)
     i=ix2(nsi,ijk)
     j=jy2(nsi,ijk)
     k=kz2(nsi,ijk)
     if(k>nzrg1(ijk)) then
       nx=nxbtm
       ny=nybtm
     else
       nx=nxtop
       ny=nytop
     endif
     if(i<1.or.i>nx.or.j<1.or.j>ny) then
       n1=n1+1
     else if(n1 > 0) then
       n2=nsi-n1
       ix2(n2,ijk)=i
       jy2(n2,ijk)=j
       kz2(n2,ijk)=k
       cf8(n2,ijk)=cf8(nsi,ijk)
     endif
   enddo   
   nss(ijk) =nss(ijk) - n1
 enddo
 nsou=0
 do ijk=1,4
   nsou=max0(nsou,nss(ijk))
 enddo
 do ijk=1,4
   do nsi=nss(ijk)+1,nsou
     ix2(nsi,ijk)=3
     jy2(nsi,ijk)=3
     kz2(nsi,ijk)=3
     cf8(nsi,ijk)=0.0
   enddo
 enddo
 return
end subroutine assign_source
!
!-----------------------------------------------------------------------
subroutine distrib(kzdf,ijk,inh,jnh,knh,xrt,yrt,zrt,ix2,jy2,kz2,cf8,nss)
! ijk=1, normal; =2, Txy; =3, Txz; =4 Tyz;
 implicit NONE
 integer, intent(IN):: kzdf,ijk,inh,jnh,knh
 integer, intent(OUT):: ix2(72),jy2(72),kz2(72),nss
 real, dimension(2), intent(IN):: xrt,yrt,zrt
 real, dimension(72), intent(OUT)::cf8 
 integer:: i,j,k,ix,jy,kz,i3,j3
 real:: cof
!
 nss=0
 do k=1,2
   kz=knh+k-1
   do j=1,2
     jy=jnh+j-1
     do i=1,2
       ix=inh+i-1
       cof=zrt(k)*yrt(j)*xrt(i)
       if(cof>0.00001) then
         nss=nss+1
         ix2(nss)=ix
         jy2(nss)=jy
         kz2(nss)=kz
         cf8(nss)=cof
         if(kzdf==-1 .or. kzdf==0) then
           if(kzdf==-1 .or. k==1 .or. ijk>2 ) then
             nss=nss-1
             do j3=-1,1
             do i3=-1,1
               nss=nss+1
               ix2(nss)=ix+(i-1)*2+i3
               jy2(nss)=jy+(j-1)*2+j3
               kz2(nss)=kz
               cf8(nss)=cof/9.0
             enddo
             enddo
           else
             ix2(nss)=(ix+(i-1)*2)/3+1
             jy2(nss)=(jy+(j-1)*2)/3+1
             cf8(nss)=cof
           endif
         elseif(kzdf==1 .and.k==1 .and. ijk>2) then
           nss=nss-1
           do j3=-1,1
           do i3=-1,1
             nss=nss+1
             ix2(nss)=3*ix+2-ijk+i3
             jy2(nss)=3*jy-5+ijk+j3
             kz2(nss)=kz
             cf8(nss)=cof/9.0
           enddo
           enddo
         endif
       endif
     enddo
   enddo
 enddo
 return
end subroutine distrib
!
!-----------------------------------------------------------------------
subroutine source_index(kzdf,nn,dist,xs,ind,rnd,ihf,rhf)
 implicit NONE
 integer, intent(IN):: kzdf,nn
 integer, intent(OUT):: ind,ihf
 real, intent(IN):: xs,dist(nn+6)
 real, intent(OUT):: rnd,rhf
 integer:: n1,n2,nm,ii
! 
 n1=2
 n2=nn+4
 do while(n2-n1>1)
   nm=(n1+n2)/2
   if(xs>dist(nm)) then
     n1=nm
   else
     n2=nm
   endif
 enddo
 ind=n1
 rnd=(dist(ind+1)-xs)/(dist(ind+1)-dist(ind))
!
 ihf=ind 
 if(rnd>0.5) ihf=ind-1
 rhf=(dist(ihf+1)+dist(ihf+2)-2.*xs)/(dist(ihf+2)-dist(ihf))
 if(kzdf==-1 .or. kzdf==0) then
   ii=ind/3
   ind=ii*3
   rnd=(dist(ind+3)-xs)/(dist(ind+2)-dist(ind+1))/3.
   ii=(ihf-1)/3
   ihf=ii*3+1
   rhf=(dist(ihf+3)+dist(ihf+4)-2.*xs)/(dist(ihf+3)-dist(ihf+1))/3.0
 endif
 ind=ind-2
 ihf=ihf-2
 if(rnd<0.05) rnd=0.0
 if(rnd>0.95) rnd=1.0
 if(rhf<0.05) rhf=0.0
 if(rhf>0.95) rhf=1.0
 return
end subroutine source_index
!
!-----------------------------------------------------------------------
subroutine alloc_station(ifx,nr_all)
 use station_comm
 implicit NONE
 integer, intent(IN) :: ifx,nr_all
 integer, allocatable, dimension(:,:):: itmp
 real, allocatable, dimension(:,:):: rtmp
! 
 if(nrecs>0) then
   allocate(itmp(2,nrecs)) 
   allocate(rtmp(2,nrecs)) 
   itmp=ipox(:,1:nrecs)
   deallocate(ipox) 
   allocate(ipox(2,nrecs)) 
   ipox=itmp
   itmp=ipoy(:,1:nrecs)
   deallocate(ipoy) 
   allocate(ipoy(2,nrecs)) 
   ipoy=itmp
   itmp=ipoz(:,1:nrecs)
   deallocate(ipoz) 
   allocate(ipoz(2,nrecs)) 
   ipoz=itmp
 else
   deallocate(ipox,ipoy,ipoz)
   deallocate(xcor,ycor,zcor)
 endif
 return
end subroutine alloc_station
!
!-----------------------------------------------------------------------
subroutine set_receiver(name_stn,ir,xr,yr,zr)
 use station_comm
 use grid_node_comm
 implicit NONE
 character(len=72), intent(IN):: name_stn
 integer, intent(IN):: ir
 real, intent(IN):: xr,yr,zr
!
 fname_stn(ir)=name_stn
 call receiver_index(nztol,gridz,zr,ipoz(1,ir),zcor(1,ir))
 if(ipoz(1,ir) > nztop) then
   call receiver_index(nxbtm,gridx2,xr,ipox(1,ir),xcor(1,ir))
   call receiver_index(nybtm,gridy2,yr,ipoy(1,ir),ycor(1,ir))
 else
   call receiver_index(nxtop,gridx1,xr,ipox(1,ir),xcor(1,ir))
   call receiver_index(nytop,gridy1,yr,ipoy(1,ir),ycor(1,ir))
 endif
 return
end subroutine set_receiver
!
!-----------------------------------------------------------------------
subroutine receiver_index(nn,dist,xs,nr2,xr2)
 implicit NONE
 integer, intent(IN):: nn
 integer, intent(OUT):: nr2(2)
 real, intent(IN):: dist(nn+6),xs
 real, intent(OUT):: xr2(2)
 integer:: n1,n2,nm,ihf
!
 n1=2
 n2=nn+3
 do while(n2-n1>1) 
   nm=(n1+n2)/2
   if(xs>dist(nm)) then
     n1=nm
   else
     n2=nm
   endif
 enddo
 xr2(1)=(dist(n1+1)-xs)/(dist(n1+1)-dist(n1))
 nr2(1)=n1
!
 ihf=nr2(1) 
 if(xr2(1)>0.5) ihf=nr2(1)-1
 xr2(2)=(dist(ihf+1)+dist(ihf+2)-2.*xs)/(dist(ihf+2)-dist(ihf))
 nr2(1)=nr2(1)-2
 nr2(2)=ihf-2
 return
end subroutine receiver_index
!
!-----------------------------------------------------------------------
subroutine operator_diff(dt)
 use grid_node_comm
 implicit NONE
 real, intent(IN):: dt
 integer:: i,j,k,kk,ii
 real:: x1,x2
!
 do i=1,nxtop
   call eq_linear(i,0,nxtop+6,gridx1,dt,dxi1(1,i)) 
   call eq_linear(i,1,nxtop+6,gridx1,dt,dxh1(1,i)) 
 enddo
 do i=1,nxbtm
   call eq_linear(i,0,nxbtm+6,gridx2,dt,dxi2(1,i)) 
   call eq_linear(i,1,nxbtm+6,gridx2,dt,dxh2(1,i)) 
 enddo

 do j=1,nytop
   call eq_linear(j,0,nytop+6,gridy1,dt,dyi1(1,j)) 
   call eq_linear(j,1,nytop+6,gridy1,dt,dyh1(1,j)) 
 enddo
 do j=1,nybtm
   call eq_linear(j,0,nybtm+6,gridy2,dt,dyi2(1,j)) 
   call eq_linear(j,1,nybtm+6,gridy2,dt,dyh2(1,j)) 
 enddo

 do k=1,nztol
   if(k>nztop) then
     kk=k-nztop
     call eq_linear(k,0,nztol+6,gridz,dt,dzi2(1,kk)) 
     call eq_linear(k,1,nztol+6,gridz,dt,dzh2(1,kk)) 
   else
     call eq_linear(k,0,nztol+6,gridz,dt,dzi1(1,k)) 
     call eq_linear(k,1,nztol+6,gridz,dt,dzh1(1,k)) 
   endif
 enddo
 do i=1,4
   dzi1(i,nztop+1) = dzi2(i,1)
   dzh1(i,nztop+1) = dzh2(i,1)
 enddo
!
 x1=-gridx1(0)-gridx1(-1)
 do ii=-1,nxtop+4
   x2=gridx1(ii)
   gridx1(ii)=x1
   x1=x1+x2
 enddo
 x1=-gridx2(0)-gridx2(-1)
 do ii=-1,nxbtm+4
   x2=gridx2(ii)
   gridx2(ii)=x1
   x1=x1+x2
 enddo
 x1=-gridy1(0)-gridy1(-1)
 do ii=-1,nytop+4
   x2=gridy1(ii)
   gridy1(ii)=x1
   x1=x1+x2
 enddo
 x1=-gridy2(0)-gridy2(-1)
 do ii=-1,nybtm+4
   x2=gridy2(ii)
   gridy2(ii)=x1
   x1=x1+x2
 enddo
 x1=-gridz(0)-gridz(-1)
 do ii=-1,nztol+4
   x2=gridz(ii)
   gridz(ii)=x1
   x1=x1+x2
 enddo
 return
end subroutine operator_diff
!
!-----------------------------------------------------------------------
subroutine eq_linear(imd,is_hf,ndm,dist,dt,diff_op)
 implicit NONE
 integer, intent(IN)::imd,is_hf,ndm 
 real, intent(IN):: dt
 real, dimension(ndm), intent(IN):: dist
 real, dimension(4), intent(OUT):: diff_op
 integer:: i,j,k,n,ii,imax
 real(kind=8):: cff(4),aa(4,4),bb(4),amx,summ,pc
!
 if(is_hf==0) then
   ii=imd+2
   cff(1)=dble(-0.5*dist(ii)-dist(ii-1))
   cff(2)=dble(-0.5*dist(ii))
   cff(3)=-cff(2)
   cff(4)= cff(3)+dble(dist(ii+1))
 else
   ii=imd+2
   cff(1)=dble(-0.5*dist(ii-2)-dist(ii-1))
   cff(2)=dble(-0.5*dist(ii-1))
   cff(3)=dble( 0.5*dist(ii))
   cff(4)=dble( 0.5*dist(ii+1)+dist(ii))
 endif
!
 n=4
 do j=1,n
   bb(j)=0.0d0
   aa(1,j)=1.0d0
 enddo
 bb(2)=1.0d0
 amx=dabs(cff(1))
 do j=2,n
   if(amx<dabs(cff(j))) amx=cff(j)
 enddo
 bb(2)=bb(2)/amx
 do j=1,N
   cff(j)=cff(j)/amx
 enddo
 do i=2,n
   pc=1.0d0
   do j=1,n
     aa(i,j)=pc*aa(i-1,j)*cff(j)
   enddo
   pc=-pc
 enddo 
 do k=1,n-1
   imax=k
   amx=dabs(aa(k,k))
          do j=k+1,n
     if(dabs(aa(j,k))>amx) then
       amx=dabs(aa(j,k))
       imax=j
     endif
   enddo
   if(imax.ne.k) then
     do j=k,n
       amx=aa(k,j)
       aa(k,j)=aa(imax,j)
       aa(imax,j)=amx
     enddo
     amx=bb(k)
     bb(k)=bb(imax)
     bb(imax)=amx
   endif
   do i=k+1,n
     amx=aa(i,k)/aa(k,k)
     do j=k+1,n
       aa(i,j)=aa(i,j)-aa(k,j)*amx
     enddo
     bb(i)=bb(i)-bb(k)*amx
   enddo
 enddo
 do i=n,1,-1
   summ=0.0d0
   do j=i+1,n
     summ=summ+aa(i,j)*cff(j)
   enddo
   cff(i)=(bb(i)-summ)/aa(i,i)
 enddo
 do i=1,n
   diff_op(i)=dt*sngl(cff(i))
 enddo
 return
end subroutine eq_linear
!
!-----------------------------------------------------------------------
subroutine coef_interp
 use grid_node_comm
 use wave_field_comm
 use itface_comm
 implicit NONE
 integer:: i,j
 real:: xi,hh(3)
!
 nxvy=nxbtm+2
 if(neighb(2)<0) nxvy=nxbtm
 nxvy1=max0(nxvy-2,nxbtm-1)
 nyvx=nybtm+2
 if(neighb(4) < 0) nyvx =nybtm
 nyvx1=max0(nyvx-2,nybtm-1)
 do i=0,nxbtm+1
   hh(1)=gridx2(i)  -gridx2(i-1)
   hh(2)=gridx2(i+1)-gridx2(i)
   hh(3)=gridx2(i+2)-gridx2(i+1)
   xi=hh(2)/3.0
   call BSL_interp(xi,hh,cix(1,i))
   xi=hh(2)*2./3.0
   call BSL_interp(xi,hh,cix(5,i))
   if(i <= nxbtm) then
     hh(1)=(gridx2(i+1)-gridx2(i-1))/2.0
     hh(2)=(gridx2(i+2)-gridx2(i  ))/2.0
     hh(3)=(gridx2(i+3)-gridx2(i+1))/2.0
     xi=(gridx2(i+1)-gridx2(i))/3.0
     call BSL_interp(xi,hh,chx(5,i))
     xi=hh(2)-(gridx2(i+2)-gridx2(i+1))/3.0
     call BSL_interp(xi,hh,chx(1,i+1))
   endif
 enddo
 do i=0,nybtm+1
   hh(1)=gridy2(i)  -gridy2(i-1)
   hh(2)=gridy2(i+1)-gridy2(i)
   hh(3)=gridy2(i+2)-gridy2(i+1)
   xi=hh(2)/3.0
   call BSL_interp(xi,hh,ciy(1,i))
   xi=hh(2)*2./3.0
   call BSL_interp(xi,hh,ciy(5,i))
   if(i <= nybtm) then
     hh(1)=(gridy2(i+1)-gridy2(i-1))/2.0
     hh(2)=(gridy2(i+2)-gridy2(i  ))/2.0
     hh(3)=(gridy2(i+3)-gridy2(i+1))/2.0
     xi=(gridy2(i+1)-gridy2(i))/3.0
     call BSL_interp(xi,hh,chy(5,i))
     xi=hh(2)-(gridy2(i+2)-gridy2(i+1))/3.0
     call BSL_interp(xi,hh,chy(1,i+1))
   endif
 enddo
 if(neighb(1) < 0) then
   do j=0,1
   do i=1,8
     cix(i,j)=0.0
     chx(i,j)=0.0
   enddo
   enddo
   cix(2,1)=2./3.
   cix(3,1)=1./3.
   cix(6,1)=1./3.
   cix(7,1)=2./3.
   chx(6,1)=2./3.
   chx(7,1)=1./3.
   chx(1,2)=0.0
   chx(2,2)=1./3.
   chx(3,2)=2./3.
   chx(4,2)=0.0
 endif
!
 if(neighb(2) < 0) then
   do j=nxbm1,nx2p1
   do i=1,8
     cix(i,j)=0.0
     if(j>nxbm1.or.(j==nxbm1.and.i>4)) chx(i,j)=0.0
   enddo
   enddo
   cix(2,nxbm1)=2./3.
   cix(3,nxbm1)=1./3.
   cix(6,nxbm1)=1./3.
   cix(7,nxbm1)=2./3.
   chx(6,nxbm1)=2./3.
   chx(7,nxbm1)=1./3.
 endif
!
 if(neighb(3) < 0) then
   do j=0,1
   do i=1,8
     ciy(i,j)=0.0
     chy(i,j)=0.0
   enddo
   enddo
   ciy(2,1)=2./3.
   ciy(3,1)=1./3.
   ciy(6,1)=1./3.
   ciy(7,1)=2./3.
   chy(6,1)=2./3.
   chy(7,1)=1./3.
   chy(1,2)=0.0
   chy(2,2)=1./3.
   chy(3,2)=2./3.
   chy(4,2)=0.0
 endif
!
 if(neighb(4) < 0) then
   do j=nybm1,ny2p1
   do i=1,8
     ciy(i,j)=0.0
     if(j>nybm1.or.(j==nybm1.and.i>4)) chy(i,j)=0.0
   enddo
   enddo
   ciy(2,nybm1)=2./3.
   ciy(3,nybm1)=1./3.
   ciy(6,nybm1)=1./3.
   ciy(7,nybm1)=2./3.
   chy(6,nybm1)=2./3.
   chy(7,nybm1)=1./3.
 endif
 return
end subroutine coef_interp
!
!-----------------------------------------------------------------------
subroutine BSL_interp(x,h,cf)
 implicit NONE
 real,intent(IN):: x
 real,dimension(3), intent(IN):: h
 real, dimension(4), intent(OUT):: cf
 integer:: i,j
 real:: y(4),rh(2)
 real:: dy21,dy32,dy43,d2y1,d2y2,ss,rx
!
 rx=x/h(2)
 rh(1)=h(1)/h(2)
 rh(2)=h(3)/h(2)
 do i=1,4
   do j=1,4
     y(j)=0.0
   enddo
   y(i)=1.0
   dy21=(y(2)-y(1))/rh(1)
   dy32= y(3)-y(2)
   dy43=(y(4)-y(3))/rh(2)
   d2y1=(dy32-dy21)/(1.+rh(1))
   d2y2=(dy43-dy32)/(1.+rh(2))
   ss=(1.+rh(2))/(2.+rh(1)+rh(2))
   ss=0.5
   cf(i)=(1.-rx)*y(2)+rx*y(3)+rx*(rx-1.)*(ss*d2y1+(1.-ss)*d2y2)
 enddo
 return
end subroutine BSL_interp
!-----------------------------------------------------------------------
subroutine ISEND_IRECV_Stress(comm_worker)
 use mpi
 use grid_node_comm
 use wave_field_comm
! use logging
 use metadata
 use ctimer
 implicit NONE
 interface
     subroutine sdx41_stress (c_sdx41, nxtop, nytop, nztop) bind(C, name="sdx41_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdx41
        integer(c_int), intent(in) :: nxtop, nytop, nztop
     end subroutine sdx41_stress

     subroutine sdx42_stress (c_sdx42, nxbtm, nybtm, nzbtm) bind(C, name="sdx42_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdx42
        integer(c_int), intent(in) :: nxbtm,nybtm, nzbtm
     end subroutine sdx42_stress

     subroutine sdx51_stress (c_sdx51, nxtop, nytop, nztop, nxtm1) bind(C, name="sdx51_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdx51
        integer(c_int), intent(in) :: nxtop, nytop, nztop, nxtm1
     end subroutine sdx51_stress

     subroutine sdx52_stress (c_sdx52, nxbtm, nybtm, nzbtm, nxbm1) bind(C, name="sdx52_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdx52
        integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm, nxbm1
     end subroutine sdx52_stress

     subroutine sdy41_stress (c_sdy41, nxtop, nytop, nztop) bind(C, name="sdy41_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdy41
        integer(c_int), intent(in) :: nxtop, nytop, nztop
     end subroutine sdy41_stress

     subroutine sdy42_stress (c_sdy42, nxbtm, nybtm, nzbtm) bind(C, name="sdy42_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdy42
        integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm
     end subroutine sdy42_stress

     subroutine sdy51_stress (c_sdy51, nxtop, nytop, nztop, nytm1) bind(C, name="sdy51_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdy51
        integer(c_int), intent(in) :: nxtop, nytop, nztop, nytm1
     end subroutine sdy51_stress

     subroutine sdy52_stress (c_sdy52, nxbtm, nybtm, nzbtm, nybm1) bind(C, name="sdy52_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_sdy52
        integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm, nybm1
     end subroutine sdy52_stress

     subroutine rcx41_stress (c_rcx41, nxtop, nytop, nztop, nx1p1, nx1p2) bind(C, name="rcx41_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcx41
        integer(c_int), intent(in) :: nxtop, nytop, nztop, nx1p1, nx1p2
     end subroutine rcx41_stress

     subroutine rcx42_stress (c_rcx42, nxbtm, nybtm, nzbtm, nx2p1, nx2p2) bind(C, name="rcx42_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcx42
        integer(c_int), intent(in) :: nxbtm,nybtm, nzbtm, nx2p1, nx2p2
     end subroutine rcx42_stress

     subroutine rcx51_stress (c_rcx51, nxtop, nytop, nztop) bind(C, name="rcx51_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcx51
        integer(c_int), intent(in) :: nxtop, nytop, nztop
     end subroutine rcx51_stress

     subroutine rcx52_stress (c_rcx52, nxbtm, nybtm, nzbtm) bind(C, name="rcx52_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcx52
        integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm
     end subroutine rcx52_stress

     subroutine rcy41_stress (c_rcy41, nxtop, nytop, nztop, ny1p1, ny1p2) bind(C, name="rcy41_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcy41
        integer(c_int), intent(in) :: nxtop, nytop, nztop, ny1p1, ny1p2
     end subroutine rcy41_stress

     subroutine rcy42_stress (c_rcy42, nxbtm, nybtm, nzbtm, ny2p1, ny2p2) bind(C, name="rcy42_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcy42
        integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm, ny2p1, ny2p2
     end subroutine rcy42_stress

     subroutine rcy51_stress (c_rcy51, nxtop, nytop, nztop) bind(C, name="rcy51_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcy51
        integer(c_int), intent(in) :: nxtop, nytop, nztop
     end subroutine rcy51_stress

     subroutine rcy52_stress (c_rcy52, nxbtm, nybtm, nzbtm) bind(C, name="rcy52_stress")
        use :: iso_c_binding
        type(c_ptr), intent(inout) :: c_rcy52
        integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm
     end subroutine rcy52_stress

     include "mpiacc_wrappers.h"
 end interface

 integer, intent(IN):: comm_worker
 integer:: req(16),ncout(16),status(MPI_STATUS_SIZE,16),ierr,nr
 integer:: reqX(16),statusX(MPI_STATUS_SIZE,16)
 integer:: i,j,k
 real(c_double) :: gpu_tstart, gpu_tend, cpu_tstart, cpu_tend
 real(c_double):: total_gpu_marsh_time=0.0, total_cpu_marsh_time=0.0

 total_gpu_marsh_time =0.0
 total_cpu_marsh_time =0.0
!
 nr=0
! Send 
!-from 1 to 2
 if(neighb(1)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdx41_stress(cptr_sdx41, nxtop, nytop, nztop);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx41_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdx41(:, :, 1)=t1xx(1:nztop, 1, 1:nytop)
   sdx41(:, :, 2)=t1xx(1:nztop, 2, 1:nytop)
   sdx41(:, :, 3)=t1xy(1:nztop, 1, 1:nytop)
   sdx41(:, :, 4)=t1xz(1:nztop, 1, 1:nytop)
#endif
   nr=nr+1
   ncout(nr)=4*nztop*nytop
#if USE_MPIX == 1
   call MPIX_ISEND_C(c_sdx41_id, ncout(nr), MPI_REAL, &
                  neighb(1), 5011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdx41, ncout(nr), MPI_REAL, &
                  neighb(1), 501, comm_worker, req(nr), ierr)
#endif

#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdx42_stress(cptr_sdx42, nxbtm, nybtm, nzbtm);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx42_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdx42(:, :, 1)=t2xx(1:nzbtm, 1, 1:nybtm)
   sdx42(:, :, 2)=t2xx(1:nzbtm, 2, 1:nybtm)
   sdx42(:, :, 3)=t2xy(1:nzbtm, 1, 1:nybtm)
   sdx42(:, :, 4)=t2xz(1:nzbtm, 1, 1:nybtm)
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nybtm
#if USE_MPIX == 1
   call MPIX_ISEND_C(c_sdx42_id, ncout(nr), MPI_REAL, & 
                  neighb(1), 5021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdx42, ncout(nr), MPI_REAL, & 
                  neighb(1), 502, comm_worker, req(nr), ierr)
#endif
 endif
!-from 2 to 1
 if(neighb(2)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdx51_stress(cptr_sdx51, nxtop, nytop, nztop, nxtm1);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx51_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdx51(:, :, 1)=t1xx(1:nztop, nxtop, 1:nytop)
   sdx51(:, :, 2)=t1xy(1:nztop, nxtm1, 1:nytop)
   sdx51(:, :, 3)=t1xy(1:nztop, nxtop, 1:nytop)
   sdx51(:, :, 4)=t1xz(1:nztop, nxtm1, 1:nytop)
   sdx51(:, :, 5)=t1xz(1:nztop, nxtop, 1:nytop)
#endif
   nr=nr+1
   ncout(nr)=5*nztop*nytop
#if USE_MPIX == 1
   call MPIX_ISEND_C(c_sdx51_id, ncout(nr), MPI_REAL, & 
                  neighb(2), 6011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdx51, ncout(nr), MPI_REAL, & 
                  neighb(2), 601, comm_worker, req(nr), ierr)
#endif
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdx52_stress(cptr_sdx52, nxbtm, nybtm, nzbtm, nxbm1);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx52_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdx52(:, :, 1)=t2xx(1:nzbtm, nxbtm, 1:nybtm)
   sdx52(:, :, 2)=t2xy(1:nzbtm, nxbm1, 1:nybtm)
   sdx52(:, :, 3)=t2xy(1:nzbtm, nxbtm, 1:nybtm)
   sdx52(:, :, 4)=t2xz(1:nzbtm, nxbm1, 1:nybtm)
   sdx52(:, :, 5)=t2xz(1:nzbtm, nxbtm, 1:nybtm)
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nybtm
#if USE_MPIX == 1
  call MPIX_ISEND_C(c_sdx52_id, ncout(nr), MPI_REAL, &
                  neighb(2), 6021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdx52, ncout(nr), MPI_REAL, &
                  neighb(2), 602, comm_worker, req(nr), ierr)
#endif
 endif
!-from 3 to 4
 if(neighb(3)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdy41_stress(cptr_sdy41, nxtop, nytop, nztop);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy41_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdy41(:, :, 1)=t1yy(1:nztop, 1:nxtop, 1)
   sdy41(:, :, 2)=t1yy(1:nztop, 1:nxtop, 2)
   sdy41(:, :, 3)=t1xy(1:nztop, 1:nxtop, 1)
   sdy41(:, :, 4)=t1yz(1:nztop, 1:nxtop, 1)
#endif
   nr=nr+1
   ncout(nr)=4*nztop*nxtop
#if USE_MPIX == 1
 call MPIX_ISEND_C(c_sdy41_id, ncout(nr), MPI_REAL, &
                  neighb(3), 7011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdy41, ncout(nr), MPI_REAL, &
                  neighb(3), 701, comm_worker, req(nr), ierr)
#endif
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdy42_stress(cptr_sdy42, nxbtm, nybtm, nzbtm);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy42_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdy42(:, :, 1)=t2yy(1:nzbtm, 1:nxbtm, 1)
   sdy42(:, :, 2)=t2yy(1:nzbtm, 1:nxbtm, 2)
   sdy42(:, :, 3)=t2xy(1:nzbtm, 1:nxbtm, 1)
   sdy42(:, :, 4)=t2yz(1:nzbtm, 1:nxbtm, 1)
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nxbtm
#if USE_MPIX == 1
    call MPIX_ISEND_C(c_sdy42_id, ncout(nr), MPI_REAL, &
                  neighb(3), 7021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdy42, ncout(nr), MPI_REAL, &
                  neighb(3), 702, comm_worker, req(nr), ierr)
#endif
 endif
!-from 4 to 3
 if(neighb(4)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdy51_stress(cptr_sdy51, nxtop, nytop, nztop, nytm1)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy51_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdy51(:, :, 1)=t1yy(1:nztop, 1:nxtop, nytop)
   sdy51(:, :, 2)=t1xy(1:nztop, 1:nxtop, nytm1)
   sdy51(:, :, 3)=t1xy(1:nztop, 1:nxtop, nytop)
   sdy51(:, :, 4)=t1yz(1:nztop, 1:nxtop, nytm1)
   sdy51(:, :, 5)=t1yz(1:nztop, 1:nxtop, nytop)
#endif
   nr=nr+1
   ncout(nr)=5*nztop*nxtop
#if USE_MPIX == 1
    call MPIX_ISEND_C(c_sdy51_id, ncout(nr), MPI_REAL, &
                  neighb(4), 8011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdy51, ncout(nr), MPI_REAL, &
                  neighb(4), 801, comm_worker, req(nr), ierr)
#endif
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call sdy52_stress(cptr_sdy52, nxbtm, nybtm, nzbtm, nybm1)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy52_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   sdy52(:, :, 1)=t2yy(1:nzbtm, 1:nxbtm, nybtm)
   sdy52(:, :, 2)=t2xy(1:nzbtm, 1:nxbtm, nybm1)
   sdy52(:, :, 3)=t2xy(1:nzbtm, 1:nxbtm, nybtm)
   sdy52(:, :, 4)=t2yz(1:nzbtm, 1:nxbtm, nybm1)
   sdy52(:, :, 5)=t2yz(1:nzbtm, 1:nxbtm, nybtm)
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nxbtm
#if USE_MPIX == 1
 call MPIX_ISEND_C(c_sdy52_id, ncout(nr), MPI_REAL, &
                  neighb(4), 8021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdy52, ncout(nr), MPI_REAL, &
                  neighb(4), 802, comm_worker, req(nr), ierr)
#endif
 endif
!
! Receive
!-from 1 to 2
 if(neighb(2)>-1) then
   nr=nr+1
   ncout(nr)=4*nztop*nytop
#if USE_MPIX == 1
  call MPIX_IRECV_C(c_rcx41_id, ncout(nr), MPI_REAL, &
                  neighb(2), 5011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcx41, ncout(nr), MPI_REAL, &
                  neighb(2), 501, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nybtm
#if USE_MPIX == 1
 call MPIX_IRECV_C(c_rcx42_id, ncout(nr), MPI_REAL, &
                  neighb(2), 5021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcx42, ncout(nr), MPI_REAL, &
                  neighb(2), 502, comm_worker, req(nr), ierr)
#endif
 endif
!-from 2 to 1
 if(neighb(1)>-1) then
   nr=nr+1
   ncout(nr)=5*nztop*nytop
#if USE_MPIX == 1
    call MPIX_IRECV_C(c_rcx51_id, ncout(nr), MPI_REAL, &
                  neighb(1), 6011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcx51, ncout(nr), MPI_REAL, &
                  neighb(1), 601, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nybtm
#if USE_MPIX == 1
    call MPIX_IRECV_C(c_rcx52_id, ncout(nr), MPI_REAL, &
                  neighb(1), 6021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcx52, ncout(nr), MPI_REAL, &
                  neighb(1), 602, comm_worker, req(nr), ierr)
#endif
 endif
!-from 3 to 4
 if(neighb(4)>-1) then
   nr=nr+1
   ncout(nr)=4*nztop*nxtop
#if USE_MPIX == 1
    call MPIX_IRECV_C(c_rcy41_id, ncout(nr), MPI_REAL, &
                  neighb(4), 7011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcy41, ncout(nr), MPI_REAL, &
                  neighb(4), 701, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nxbtm
#if USE_MPIX == 1
    call MPIX_IRECV_C(c_rcy42_id, ncout(nr), MPI_REAL, &
                  neighb(4), 7021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcy42, ncout(nr), MPI_REAL, &
                  neighb(4), 702, comm_worker, req(nr), ierr)
#endif
 endif
!-from 4 to 3
 if(neighb(3)>-1) then
   nr=nr+1
   ncout(nr)=5*nztop*nxtop
#if USE_MPIX == 1
    call MPIX_IRECV_C(c_rcy51_id, ncout(nr), MPI_REAL, &
                  neighb(3), 8011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcy51, ncout(nr), MPI_REAL, &
                  neighb(3), 801, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nxbtm
#if USE_MPIX == 1
    call MPIX_IRECV_C(c_rcy52_id, ncout(nr), MPI_REAL, &
                  neighb(3), 8021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcy52, ncout(nr), MPI_REAL, &
                  neighb(3), 802, comm_worker, req(nr), ierr)
#endif
 endif
#if USE_MPIX == 1
 call MPI_WAITALL(nr,reqX,statusX,ierr)
#else
 call MPI_WAITALL(nr,req,status,ierr)
#endif
!-from 1 to 2
 if(neighb(2)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call rcx41_stress(cptr_rcx41, nxtop, nytop, nztop, nx1p1, nx1p2)
        call rcx42_stress(cptr_rcx42, nxbtm, nybtm, nzbtm, nx2p1, nx2p2)   
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx41/2_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   t1xx(1:nztop, nx1p1, 1:nytop) = rcx41(:, :, 1)
   t1xx(1:nztop, nx1p2, 1:nytop) = rcx41(:, :, 2)
   t1xy(1:nztop, nx1p1, 1:nytop) = rcx41(:, :, 3)
   t1xz(1:nztop, nx1p1, 1:nytop) = rcx41(:, :, 4)
   t2xx(1:nzbtm, nx2p1, 1:nybtm) = rcx42(:, :, 1)
   t2xx(1:nzbtm, nx2p2, 1:nybtm) = rcx42(:, :, 2)
   t2xy(1:nzbtm, nx2p1, 1:nybtm) = rcx42(:, :, 3)
   t2xz(1:nzbtm, nx2p1, 1:nybtm) = rcx42(:, :, 4)
#endif
 endif
!-from 2 to 1
 if(neighb(1)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
           call rcx51_stress(cptr_rcx51, nxtop, nytop, nztop)
           call rcx52_stress(cptr_rcx52, nxbtm, nybtm, nzbtm)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx51/2_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   t1xx(1:nztop,  0, 1:nytop) = rcx51(:, :, 1)
   t1xy(1:nztop, -1, 1:nytop) = rcx51(:, :, 2)
   t1xy(1:nztop,  0, 1:nytop) = rcx51(:, :, 3)
   t1xz(1:nztop, -1, 1:nytop) = rcx51(:, :, 4)
   t1xz(1:nztop,  0, 1:nytop) = rcx51(:, :, 5)
   t2xx(1:nzbtm,  0, 1:nybtm) = rcx52(:, :, 1)
   t2xy(1:nzbtm, -1, 1:nybtm) = rcx52(:, :, 2)
   t2xy(1:nzbtm,  0, 1:nybtm) = rcx52(:, :, 3)
   t2xz(1:nzbtm, -1, 1:nybtm) = rcx52(:, :, 4)
   t2xz(1:nzbtm,  0, 1:nybtm) = rcx52(:, :, 5)
#endif
 endif
!------------------ from 3 to 4
 if(neighb(4)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call rcy41_stress(cptr_rcy41, nxtop, nytop, nztop, ny1p1, ny1p2)
        call rcy42_stress(cptr_rcy42, nxbtm, nybtm, nzbtm, ny2p1, ny2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy41/2_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   t1yy(1:nztop, 1:nxtop, ny1p1) = rcy41(:, :, 1)
   t1yy(1:nztop, 1:nxtop, ny1p2) = rcy41(:, :, 2)
   t1xy(1:nztop, 1:nxtop, ny1p1) = rcy41(:, :, 3)
   t1yz(1:nztop, 1:nxtop, ny1p1) = rcy41(:, :, 4)
   t2yy(1:nzbtm, 1:nxbtm, ny2p1) = rcy42(:, :, 1)
   t2yy(1:nzbtm, 1:nxbtm, ny2p2) = rcy42(:, :, 2)
   t2xy(1:nzbtm, 1:nxbtm, ny2p1) = rcy42(:, :, 3)
   t2yz(1:nzbtm, 1:nxbtm, ny2p1) = rcy42(:, :, 4)
#endif
 endif
!-from 4 to 3
 if(neighb(3)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
        call rcy51_stress(cptr_rcy51, nxtop, nytop, nztop)
        call rcy52_stress(cptr_rcy52, nxbtm, nybtm, nzbtm)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy51/2_stress", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   t1yy(1:nztop, 1:nxtop,  0) = rcy51(:, :, 1)
   t1xy(1:nztop, 1:nxtop, -1) = rcy51(:, :, 2)
   t1xy(1:nztop, 1:nxtop,  0) = rcy51(:, :, 3)
   t1yz(1:nztop, 1:nxtop, -1) = rcy51(:, :, 4)
   t1yz(1:nztop, 1:nxtop,  0) = rcy51(:, :, 5)
   t2yy(1:nzbtm, 1:nxbtm,  0) = rcy52(:, :, 1)
   t2xy(1:nzbtm, 1:nxbtm, -1) = rcy52(:, :, 2)
   t2xy(1:nzbtm, 1:nxbtm,  0) = rcy52(:, :, 3)
   t2yz(1:nzbtm, 1:nxbtm, -1) = rcy52(:, :, 4)
   t2yz(1:nzbtm, 1:nxbtm,  0) = rcy52(:, :, 5)
#endif
 endif
 return
end subroutine ISEND_IRECV_Stress
!
!-----------------------------------------------------------------------
subroutine ISEND_IRECV_Velocity(comm_worker)
 use mpi
 use grid_node_comm
 use ctimer
 use wave_field_comm
 use, intrinsic :: iso_c_binding
 implicit NONE

 interface
     subroutine sdx51_vel(c_ptr_sdx51, c_nytop, c_nztop, c_nxtop) &
         bind(C, name="sdx51_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdx51
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop
     end subroutine sdx51_vel

     subroutine sdx52_vel(c_ptr_sdx52, c_nybtm, c_nzbtm, c_nxbtm) &
         bind(C, name="sdx52_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdx52
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm
     end subroutine sdx52_vel

     subroutine sdx41_vel(c_ptr_sdx41, c_nxtop, c_nytop, c_nztop,c_nxtm1) &
         bind(C, name="sdx41_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdx41
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop, c_nxtm1
     end subroutine sdx41_vel
    
    subroutine sdx42_vel(c_ptr_sdx42, c_nxbtm, c_nybtm, c_nzbtm, c_nxbm1) &
         bind(C, name="sdx42_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdx42
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm, c_nxbm1
     end subroutine sdx42_vel

    subroutine sdy51_vel(c_ptr_sdy51, c_nxtop, c_nytop, c_nztop) &
         bind(C, name="sdy51_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdy51
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop
     end subroutine sdy51_vel

     subroutine sdy52_vel(c_ptr_sdy52, c_nxbtm, c_nybtm, c_nzbtm) &
         bind(C, name="sdy52_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdy52
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm
     end subroutine sdy52_vel

     subroutine sdy41_vel(c_ptr_sdy41, c_nxtop, c_nytop, c_nztop,c_nytm1) &
         bind(C, name="sdy41_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdy41
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop, c_nytm1
     end subroutine sdy41_vel

    subroutine sdy42_vel(c_ptr_sdy42, c_nxbtm, c_nybtm, c_nzbtm, c_nybm1) &
         bind(C, name="sdy42_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_sdy42
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm, c_nybm1
     end subroutine sdy42_vel

     subroutine rcx51_vel(c_ptr_rcx51, c_nxtop, c_nytop, c_nztop, nx1p1, nx1p2) &
         bind(C, name="rcx51_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcx51
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop, nx1p1, nx1p2
     end subroutine rcx51_vel

     subroutine rcx52_vel(c_ptr_rcx52, c_nxbtm, c_nybtm, c_nzbtm, nx2p1, nx2p2) &
         bind(C, name="rcx52_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcx52
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm, nx2p1, nx2p2
     end subroutine rcx52_vel

     subroutine rcx41_vel(c_ptr_rcx41, c_nxtop, c_nytop, c_nztop) &
         bind(C, name="rcx41_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcx41
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop
     end subroutine rcx41_vel

    subroutine rcx42_vel(c_ptr_rcx42, c_nxbtm, c_nybtm, c_nzbtm) &
         bind(C, name="rcx42_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcx42
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm
     end subroutine rcx42_vel

    subroutine rcy51_vel(c_ptr_rcy51, c_nxtop, c_nytop, c_nztop, ny1p1, ny1p2) &
         bind(C, name="rcy51_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcy51
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop, ny1p1, ny1p2
     end subroutine rcy51_vel

     subroutine rcy52_vel(c_ptr_rcy52, c_nxbtm, c_nybtm, c_nzbtm, ny2p1, ny2p2) &
         bind(C, name="rcy52_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcy52
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm, ny2p1, ny2p2
     end subroutine rcy52_vel

     subroutine rcy41_vel(c_ptr_rcy41, c_nxtop, c_nytop, c_nztop) &
         bind(C, name="rcy41_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcy41
         integer(c_int), intent(in):: c_nytop, c_nztop, c_nxtop
     end subroutine rcy41_vel

    subroutine rcy42_vel(c_ptr_rcy42, c_nxbtm, c_nybtm, c_nzbtm) &
         bind(C, name="rcy42_vel")
         use, intrinsic :: iso_c_binding, ONLY: C_INT, C_PTR, C_F_POINTER
         type(c_ptr), intent(inout) :: c_ptr_rcy42
         integer(c_int), intent(in):: c_nybtm, c_nzbtm, c_nxbtm
     end subroutine rcy42_vel


    subroutine print_3d_array(array_name, array_ptr, size_x, size_y, size_z)
        use  :: iso_c_binding
        integer , intent(in):: size_x, size_y, size_z
        real(c_float), pointer, dimension(:,:,:) :: array_ptr
        integer:: i,j,k
        character(len=10):: array_name
    end subroutine print_3d_array

    include "mpiacc_wrappers.h"
 end interface

 integer, intent(IN):: comm_worker
 integer:: req(54),ncout(54),status(MPI_STATUS_SIZE,54),ierr,nr,i,j,k
 integer:: reqX(54), statusX(MPI_STATUS_SIZE,54)
 real:: time_marsh_send_ISRV=0.0,time_marsh_recv_ISRV=0.0
 real(c_double) :: gpu_tstart, gpu_tend, cpu_tstart, cpu_tend
 real(c_double) :: total_gpu_marsh_time=0.0, total_cpu_marsh_time=0.0
 type(c_ptr) :: temp_rcx41C
 real(c_float), pointer, dimension(:,:,:) :: rcx41_tmp
 integer(kind=MPI_ADDRESS_KIND) size 
 integer :: real_itemsize, ierror
 total_gpu_marsh_time =0.0
 total_cpu_marsh_time =0.0
!
 nr=0
! Send
!-from 1 to 2
 if(neighb(1)>-1) then
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call sdx51_vel(cptr_sdx51, nytop, nztop, nxtop);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx51_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=1,nytop
   do k=1,nztop
     sdx51(k,j,1)=v1x(k,1,j)
     sdx51(k,j,2)=v1y(k,1,j)
     sdx51(k,j,3)=v1y(k,2,j)
     sdx51(k,j,4)=v1z(k,1,j)
     sdx51(k,j,5)=v1z(k,2,j)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=5*nztop*nytop
#if USE_MPIX == 1
   call MPIX_ISEND_C(c_sdx51_id, ncout(nr), MPI_REAL, &
                neighb(1), 1011, comm_worker, reqX(nr), ierr, 0)
#else 
   call MPI_ISEND(sdx51, ncout(nr), MPI_REAL, &
                  neighb(1), 101, comm_worker, req(nr), ierr)
#endif
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call sdx52_vel(cptr_sdx52, nybtm, nzbtm, nxbtm);
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx52_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=1,nybtm
   do k=1,nzbtm
     sdx52(k,j,1)=v2x(k,1,j)
     sdx52(k,j,2)=v2y(k,1,j)
     sdx52(k,j,3)=v2y(k,2,j)
     sdx52(k,j,4)=v2z(k,1,j)
     sdx52(k,j,5)=v2z(k,2,j)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nybtm
#if USE_MPIX == 1
  call MPIX_ISEND_C(c_sdx52_id, ncout(nr), MPI_REAL, &
                  neighb(1), 1021, comm_worker, reqX(nr), ierr, 1)
#else
  call MPI_ISEND(sdx52, ncout(nr), MPI_REAL, &
                  neighb(1), 102, comm_worker, req(nr), ierr)
#endif
 endif
!-from 2 to 1
 if(neighb(2)>-1) then
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call sdx41_vel(cptr_sdx41, nxtop,  nytop, nztop, nxtm1)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx41_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=1,nytop
   do k=1,nztop
     sdx41(k,j,1)=v1x(k,nxtm1,j)
     sdx41(k,j,2)=v1x(k,nxtop,j)
     sdx41(k,j,3)=v1y(k,nxtop,j)
     sdx41(k,j,4)=v1z(k,nxtop,j)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=4*nztop*nytop
#if USE_MPIX == 1
     call MPIX_ISEND_C(c_sdx41_id, ncout(nr), MPI_REAL, &
                  neighb(2), 2011, comm_worker, reqX(nr), ierr, 0)
#else
     call MPI_ISEND(sdx41, ncout(nr), MPI_REAL, &
                  neighb(2), 201, comm_worker, req(nr), ierr)
#endif
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if    
   call sdx42_vel(cptr_sdx42, nxbtm, nybtm, nzbtm, nxbm1)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx42_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=1,nybtm
   do k=1,nzbtm
     sdx42(k,j,1)=v2x(k,nxbm1,j)
     sdx42(k,j,2)=v2x(k,nxbtm,j)
     sdx42(k,j,3)=v2y(k,nxbtm,j)
     sdx42(k,j,4)=v2z(k,nxbtm,j)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nybtm
#if USE_MPIX == 1
    call MPIX_ISEND_C(c_sdx42_id, ncout(nr), MPI_REAL, &
                  neighb(2), 2021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdx42, ncout(nr), MPI_REAL, &
                  neighb(2), 202, comm_worker, req(nr), ierr)
#endif
 endif
!-from 3 to 4
 if(neighb(3)>-1) then
#ifdef DISFD_GPU_MARSHALING   
    if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call sdy51_vel(cptr_sdy51, nxtop, nytop, nztop)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy51_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=1,nxtop
   do k=1,nztop
     sdy51(k,i,1)=v1x(k,i,1)
     sdy51(k,i,2)=v1x(k,i,2)
     sdy51(k,i,3)=v1y(k,i,1)
     sdy51(k,i,4)=v1z(k,i,1)
     sdy51(k,i,5)=v1z(k,i,2)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=5*nztop*nxtop
#if USE_MPIX == 1
    call MPIX_ISEND_C(c_sdy51_id, ncout(nr), MPI_REAL, &
                  neighb(3), 3011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdy51, ncout(nr), MPI_REAL, &
                  neighb(3), 301, comm_worker, req(nr), ierr)
#endif
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call sdy52_vel(cptr_sdy52, nxbtm, nybtm, nzbtm)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy52_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=1,nxbtm
   do k=1,nzbtm
     sdy52(k,i,1)=v2x(k,i,1)
     sdy52(k,i,2)=v2x(k,i,2)
     sdy52(k,i,3)=v2y(k,i,1)
     sdy52(k,i,4)=v2z(k,i,1)
     sdy52(k,i,5)=v2z(k,i,2)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nxbtm
#if USE_MPIX == 1
  call MPIX_ISEND_C(c_sdy52_id, ncout(nr), MPI_REAL, &
                  neighb(3), 3021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdy52, ncout(nr), MPI_REAL, &
                  neighb(3), 302, comm_worker, req(nr), ierr)
#endif
 endif
!-from 4 to 3
 if(neighb(4)>-1) then
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call sdy41_vel(cptr_sdy41, nxtop, nytop, nztop, nytm1)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy41_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=1,nxtop
   do k=1,nztop
     sdy41(k,i,1)=v1x(k,i,nytop)
     sdy41(k,i,2)=v1y(k,i,nytm1)
     sdy41(k,i,3)=v1y(k,i,nytop)
     sdy41(k,i,4)=v1z(k,i,nytop)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=4*nztop*nxtop
#if USE_MPIX == 1
  call MPIX_ISEND_C(c_sdy41_id, ncout(nr), MPI_REAL, &
                  neighb(4), 4011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdy41, ncout(nr), MPI_REAL, &
                  neighb(4), 401, comm_worker, req(nr), ierr)
#endif
#ifdef DISFD_GPU_MARSHALING   
  if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call sdy42_vel(cptr_sdy42, nxbtm, nybtm, nzbtm, nybm1)
  if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy42_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=1,nxbtm
   do k=1,nzbtm
     sdy42(k,i,1)=v2x(k,i,nybtm)
     sdy42(k,i,2)=v2y(k,i,nybm1)
     sdy42(k,i,3)=v2y(k,i,nybtm)
     sdy42(k,i,4)=v2z(k,i,nybtm)
   enddo
   enddo
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nxbtm
#if USE_MPIX == 1
  call MPIX_ISEND_C(c_sdy42_id, ncout(nr), MPI_REAL, &
                  neighb(4), 4021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdy42, ncout(nr), MPI_REAL, &
                  neighb(4), 402, comm_worker, req(nr), ierr)
#endif
 endif
!
! Receive
!-from 1 to 2
 if(neighb(2)>-1) then
   nr=nr+1
   ncout(nr)=5*nztop*nytop
#if USE_MPIX == 1
  call MPIX_IRECV_C(c_rcx51_id, ncout(nr), MPI_REAL, &
                  neighb(2), 1011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcx51, ncout(nr), MPI_REAL, &
                  neighb(2), 101, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nybtm
#if USE_MPIX == 1
      call MPIX_IRECV_C(c_rcx52_id, ncout(nr), MPI_REAL, &
                  neighb(2), 1021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcx52, ncout(nr), MPI_REAL, &
                  neighb(2), 102, comm_worker, req(nr), ierr)
#endif
 endif
!-from 2 to 1
 if(neighb(1)>-1) then
   nr=nr+1
   ncout(nr)=4*nztop*nytop
#if USE_MPIX == 1
      call MPIX_IRECV_C(c_rcx41_id, ncout(nr), MPI_REAL, &
                  neighb(1), 2011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcx41, ncout(nr), MPI_REAL, &
                  neighb(1), 201, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nybtm
#if USE_MPIX == 1
  call MPIX_IRECV_C(c_rcx42_id, ncout(nr), MPI_REAL, &
                  neighb(1), 2021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcx42, ncout(nr), MPI_REAL, &
                  neighb(1), 202, comm_worker, req(nr), ierr)
#endif
 endif
!-from 3 to 4
 if(neighb(4)>-1) then
   nr=nr+1
   ncout(nr)=5*nztop*nxtop
#if USE_MPIX == 1
  call MPIX_IRECV_C(c_rcy51_id, ncout(nr), MPI_REAL, &
                  neighb(4), 3011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcy51, ncout(nr), MPI_REAL, &
                  neighb(4), 301, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=5*nzbtm*nxbtm
#if USE_MPIX == 1
  call MPIX_IRECV_C(c_rcy52_id, ncout(nr), MPI_REAL, &
                  neighb(4), 3021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcy52, ncout(nr), MPI_REAL, &
                  neighb(4), 302, comm_worker, req(nr), ierr)
#endif
 endif
!-from 4 to 3
 if(neighb(3)>-1) then
   nr=nr+1
   ncout(nr)=4*nztop*nxtop
#if USE_MPIX == 1
      call MPIX_IRECV_C(c_rcy41_id, ncout(nr), MPI_REAL, &
                  neighb(3), 4011, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcy41, ncout(nr), MPI_REAL, &
                  neighb(3), 401, comm_worker, req(nr), ierr)
#endif
   nr=nr+1
   ncout(nr)=4*nzbtm*nxbtm
#if USE_MPIX == 1
      call MPIX_IRECV_C(c_rcy42_id, ncout(nr), MPI_REAL, &
                  neighb(3), 4021, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcy42, ncout(nr), MPI_REAL, &
                  neighb(3), 402, comm_worker, req(nr), ierr)
#endif
 endif
#if USE_MPIX==1
 call MPI_WAITALL(nr,reqX,statusX,ierr)
#else
 call MPI_WAITALL(nr,req,status,ierr)
#endif
!-from 1 to 2
 if(neighb(2)>-1) then
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call rcx51_vel (cptr_rcx51, nxtop, nytop, nztop, nx1p1, nx1p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx51_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call rcx52_vel(cptr_rcx52, nxbtm, nybtm, nzbtm, nx2p1, nx2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx52_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=1,nytop
   do k=1,nztop
     v1x(k,nx1p1,j) = rcx51(k,j,1)
     v1y(k,nx1p1,j) = rcx51(k,j,2)
     v1y(k,nx1p2,j) = rcx51(k,j,3)
     v1z(k,nx1p1,j) = rcx51(k,j,4)
     v1z(k,nx1p2,j) = rcx51(k,j,5)
   enddo
   enddo
   do j=1,nybtm
   do k=1,nzbtm
     v2x(k,nx2p1,j) = rcx52(k,j,1)
     v2y(k,nx2p1,j) = rcx52(k,j,2)
     v2y(k,nx2p2,j) = rcx52(k,j,3)
     v2z(k,nx2p1,j) = rcx52(k,j,4)
     v2z(k,nx2p2,j) = rcx52(k,j,5)
   enddo
   enddo
#endif
 endif
!-from 2 to 1
 if(neighb(1)>-1) then
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call rcx41_vel(cptr_rcx41, nxtop, nytop, nztop)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx41_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call rcx42_vel(cptr_rcx42, nxbtm, nybtm, nzbtm)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx42_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=1,nytop
   do k=1,nztop
     v1x(k,-1,j) = rcx41(k,j,1)
     v1x(k, 0,j) = rcx41(k,j,2)
     v1y(k, 0,j) = rcx41(k,j,3)
     v1z(k, 0,j) = rcx41(k,j,4)
   enddo
   enddo
   do j=1,nybtm
   do k=1,nzbtm
     v2x(k,-1,j) = rcx42(k,j,1)
     v2x(k, 0,j) = rcx42(k,j,2)
     v2y(k, 0,j) = rcx42(k,j,3)
     v2z(k, 0,j) = rcx42(k,j,4)
   enddo
   enddo
#endif
 endif
!-from 3 to 4
 if(neighb(4)>-1) then
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call rcy51_vel(cptr_rcy51, nxtop, nytop, nztop, ny1p1, ny1p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy51_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call rcy52_vel(cptr_rcy52, nxbtm, nybtm, nzbtm, ny2p1, ny2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy52_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=1,nxtop
   do k=1,nztop
     v1x(k,i,ny1p1) = rcy51(k,i,1)
     v1x(k,i,ny1p2) = rcy51(k,i,2)
     v1y(k,i,ny1p1) = rcy51(k,i,3)
     v1z(k,i,ny1p1) = rcy51(k,i,4)
     v1z(k,i,ny1p2) = rcy51(k,i,5)
   enddo
   enddo
   do i=1,nxbtm
   do k=1,nzbtm
     v2x(k,i,ny2p1) = rcy52(k,i,1)
     v2x(k,i,ny2p2) = rcy52(k,i,2)
     v2y(k,i,ny2p1) = rcy52(k,i,3)
     v2z(k,i,ny2p1) = rcy52(k,i,4)
     v2z(k,i,ny2p2) = rcy52(k,i,5)
   enddo
   enddo
#endif
 endif
!-from 4 to 3
 if(neighb(3)>-1) then
#ifdef DISFD_GPU_MARSHALING   
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call rcy41_vel(cptr_rcy41, nxtop, nytop, nztop)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy41_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if    
    call rcy42_vel(cptr_rcy42, nxbtm, nybtm, nzbtm)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy42_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=1,nxtop
   do k=1,nztop
     v1x(k,i, 0) = rcy41(k,i,1)
     v1y(k,i,-1) = rcy41(k,i,2)
     v1y(k,i, 0) = rcy41(k,i,3)
     v1z(k,i, 0) = rcy41(k,i,4)
   enddo
   enddo
   do i=1,nxbtm
   do k=1,nzbtm
     v2x(k,i, 0) = rcy42(k,i,1)
     v2y(k,i,-1) = rcy42(k,i,2)
     v2y(k,i, 0) = rcy42(k,i,3)
     v2z(k,i, 0) = rcy42(k,i,4)
   enddo
   enddo
#endif
 endif
 return
end subroutine ISEND_IRECV_Velocity
!
!-----------------------------------------------------------------------
subroutine velocity_interp(comm_worker)
 use mpi
 use grid_node_comm
 use wave_field_comm
 use itface_comm
 use ctimer
 use iso_c_binding
 implicit NONE

 interface
    subroutine sdx1_vel(sdx1, nxtop, nytop , nxbtm, nzbtm, ny2p1, ny2p2) &
    bind(C, name="sdx1_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: sdx1 
        integer(c_int), intent(in)::nxtop, nytop, ny2p1, ny2p2, nxbtm, nzbtm 
    end subroutine sdx1_vel

    subroutine sdy1_vel(sdy1, nxtop, nytop ,nxbtm, nzbtm, nx2p1, nx2p2) &
    bind(C, name="sdy1_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: sdy1
        integer(c_int), intent(in)::nxtop, nytop, nx2p1, nx2p2, nxbtm, nzbtm
    end subroutine sdy1_vel

    subroutine sdx2_vel(sdx2, nxtop, nytop , nxbm1, nxbtm, nzbtm, ny2p1, ny2p2) &
    bind(C, name="sdx2_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: sdx2
        integer(c_int), intent(in) :: nxtop, nytop, nxbm1, nxbtm, ny2p1, ny2p2, nzbtm
    end subroutine sdx2_vel

    subroutine sdy2_vel(sdy2, nxtop, nytop , nybm1, nybtm , nxbtm, nzbtm, nx2p1, nx2p2) &
    bind(C, name="sdy2_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: sdy2
        integer(c_int), intent(in) :: nxtop, nytop, nybm1, nybtm, nxbtm, nzbtm, nx2p1, nx2p2
    end subroutine sdy2_vel

    subroutine rcx1_vel(rcx1, nxtop, nytop ,nxbtm , nzbtm,ny2p1, ny2p2) &
    bind(C, name="rcx1_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: rcx1
        integer(c_int), intent(in) :: nxtop, nytop, ny2p1, ny2p2, nxbtm, nzbtm
    end subroutine rcx1_vel

    subroutine rcy1_vel(rcy1, nxtop, nytop ,nx2p1, nxbtm, nzbtm, nx2p2) &
    bind(C, name="rcy1_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: rcy1
        integer(c_int), intent(in) :: nxtop, nytop, nx2p1, nx2p2, nxbtm, nzbtm
    end subroutine rcy1_vel

    subroutine rcx2_vel(rcx2, nxtop, nytop , nxbtm, nzbtm, nx2p1, nx2p2, ny2p1, ny2p2) &
    bind(C, name="rcx2_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: rcx2
        integer(c_int), intent(in) :: nxtop, nytop, nx2p1, nx2p2, ny2p1, ny2p2, nxbtm, nzbtm
    end subroutine rcx2_vel

    subroutine rcy2_vel(rcy2, nxtop, nytop , nxbtm, nzbtm, nx2p1, nx2p2, ny2p1, ny2p2) &
    bind(C, name="rcy2_vel")
    use:: iso_c_binding
        type(c_ptr), intent(in), value:: rcy2
        integer(c_int), intent(in) ::nxtop, nytop, ny2p1, ny2p2,  nx2p1, nx2p2, nxbtm,nzbtm
    end subroutine rcy2_vel
    
    include "mpiacc_wrappers.h"
 end interface
 integer, intent(IN):: comm_worker
 integer:: req(54),ncout(54),status(MPI_STATUS_SIZE,54),nr,ierr
 integer:: reqX(54),statusX(MPI_STATUS_SIZE,54)
 integer:: i0,i,j0,j
 real(c_double) :: gpu_tstart, gpu_tend, cpu_tstart, cpu_tend
 real(c_double) :: tend, tstart
 real(c_double):: total_gpu_marsh_time=0.0, total_cpu_marsh_time=0.0
 total_gpu_marsh_time =0.0
 total_cpu_marsh_time =0.0

 nr=0
!
! SEND 
!-from 1 to 2
 if(neighb(1)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if    
   call sdx1_vel(c_loc(sdx1), nxtop, nytop, nxbtm, nzbtm,  ny2p1, ny2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx1_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   j0=ny2p2+2
   do j=0,ny2p2
     sdx1(j+1 ) = v2z(1,3,j)
     sdx1(j+j0) = v2x(1,2,j)
   enddo
   j0=2*(ny2p2+1)
   sdx1(j0+1) = v2z(1,1,ny2p1)
   sdx1(j0+2) = v2z(1,2,ny2p1)
   sdx1(j0+3) = v2z(1,1,ny2p2)
   sdx1(j0+4) = v2z(1,2,ny2p2)
#endif
   nr=nr+1
   ncout(nr) = j0+4
#if USE_MPIX == 1
   call MPIX_ISEND_C(c_sdx1_id, ncout(nr), MPI_REAL, &
                  neighb(1), 111, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdx1(1), ncout(nr), MPI_REAL, &
                  neighb(1), 11, comm_worker, req(nr), ierr)
#endif 
 endif
!-from 3 to 4
 if(neighb(3)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
    call sdy1_vel(c_loc(sdy1), nxtop, nytop, nxbtm, nzbtm, nx2p1, nx2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy1_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   i0=nx2p2+2
   do i=0,nx2p2
     sdy1(i+1 ) = v2z(1,i,3)
     sdy1(i+i0) = v2y(1,i,2)
   enddo
   i0=2*(nx2p2+1)
   sdy1(i0+1) = v2x(1,nx2p1,1)
   sdy1(i0+2) = v2x(1,nx2p1,2)
   sdy1(i0+3) = v2y(1,nx2p1,1)
   sdy1(i0+4) = v2y(1,nx2p2,1)
#endif
   nr=nr+1
   ncout(nr) = i0 + 4
#if USE_MPIX == 1
    call MPIX_ISEND_C(c_sdy1_id, ncout(nr), MPI_REAL, &
          neighb(3), 331, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_ISEND(sdy1(1), ncout(nr), MPI_REAL, &
                  neighb(3), 33, comm_worker, req(nr), ierr)
#endif 
 endif
!-from 2 to 1 
 if(neighb(2)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call sdx2_vel (c_loc(sdx2), nxtop, nytop, nxbm1, nxbtm, nzbtm, ny2p1, ny2p2)
    if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdx2_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do j=0,ny2p2
     sdx2(j+1 ) = v2z(1,nxbm1,j)
   enddo
   j0=ny2p2+1
   sdx2(j0+1 ) = v2x(1,nxbm1,ny2p1)
   sdx2(j0+2 ) = v2x(1,nxbtm,ny2p1)
   sdx2(j0+3 ) = v2x(1,nxbm1,ny2p2)
   sdx2(j0+4 ) = v2x(1,nxbtm,ny2p2)
   sdx2(j0+5 ) = v2y(1,nxbtm,ny2p1)
   sdx2(j0+6 ) = v2z(1,nxbtm,ny2p1)
   sdx2(j0+7 ) = v2z(1,nxbtm,ny2p2)
#endif
   nr=nr+1
   ncout(nr) = j0 + 7
#if USE_MPIX == 1
   call MPIX_ISEND_C(c_sdx2_id, ncout(nr), MPI_REAL, &
                  neighb(2), 221, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdx2(1), ncout(nr), MPI_REAL, &
                  neighb(2), 22, comm_worker, req(nr), ierr)
#endif 
 endif
!-from 4 to 3 
 if(neighb(4)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call sdy2_vel(c_loc(sdy2), nxtop, nytop, nybm1, nybtm, nxbtm, nzbtm, nx2p1, nx2p2 )
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU sdy2_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   do i=0,nx2p2
     sdy2(i+1 ) = v2z(1,i,nybm1)
   enddo
   i0=nx2p2+1
   sdy2(i0+1) = v2x(1,   -1,nybtm)
   sdy2(i0+2) = v2x(1,    0,nybtm)
   sdy2(i0+3) = v2x(1,nx2p1,nybtm)
   sdy2(i0+4) = v2y(1,    0,nybm1)
   sdy2(i0+5) = v2y(1,nx2p1,nybm1)
   sdy2(i0+6) = v2y(1,nx2p2,nybm1)
   sdy2(i0+7) = v2y(1,    0,nybtm)
   sdy2(i0+8) = v2y(1,nx2p1,nybtm)
   sdy2(i0+9) = v2y(1,nx2p2,nybtm)
   sdy2(i0+10)= v2z(1,    0,nybtm)
   sdy2(i0+11)= v2z(1,nx2p1,nybtm)
   sdy2(i0+12)= v2z(1,nx2p2,nybtm)
#endif
   nr=nr+1
   ncout(nr) = i0 + 12
#if USE_MPIX == 1
   call MPIX_ISEND_C(c_sdy2_id, ncout(nr), MPI_REAL, &
                  neighb(4), 441, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_ISEND(sdy2(1), ncout(nr), MPI_REAL, &
                  neighb(4), 44, comm_worker, req(nr), ierr)
#endif 
 endif
!
! RECEIVE
!-from 1 to 2
 if(neighb(2)>-1) then
   nr=nr+1
   ncout(nr) = 2*(ny2p2+1)+4
#if USE_MPIX == 1
   call MPIX_IRECV_C(c_rcx2_id, ncout(nr), MPI_REAL, &
    neighb(2), 111, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcx2(1), ncout(nr), MPI_REAL, &
                  neighb(2), 11, comm_worker, req(nr), ierr)
#endif
 endif
!-from 3 to 4
 if(neighb(4)>-1) then
   nr=nr+1
   ncout(nr) = 2*(nx2p2+1) + 4
#if USE_MPIX == 1
   call MPIX_IRECV_C(c_rcy2_id, ncout(nr), MPI_REAL, &
                  neighb(4), 331, comm_worker, reqX(nr), ierr, 1)
#else
   call MPI_IRECV(rcy2(1), ncout(nr), MPI_REAL, &
                  neighb(4), 33, comm_worker, req(nr), ierr)
#endif   
 endif
!-from 2 to 1 
 if(neighb(1)>-1) then
   nr=nr+1
   ncout(nr) = ny2p2+1 + 7
#if USE_MPIX == 1
   call MPIX_IRECV_C(c_rcx1_id, ncout(nr), MPI_REAL, &
                  neighb(1), 221, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcx1(1), ncout(nr), MPI_REAL, &
                  neighb(1), 22, comm_worker, req(nr), ierr)
#endif   
 endif
!-from 4 to 3 
 if(neighb(3)>-1) then
   nr=nr+1
   ncout(nr) = nx2p2+1 + 12
#if USE_MPIX == 1
   call MPIX_IRECV_C(c_rcy1_id, ncout(nr), MPI_REAL, &
                  neighb(3), 441, comm_worker, reqX(nr), ierr, 0)
#else
   call MPI_IRECV(rcy1(1), ncout(nr), MPI_REAL, &
                  neighb(3), 44, comm_worker, req(nr), ierr)
#endif 
 endif
#if USE_MPIX == 1
 call MPI_WAITALL(nr,reqX,statusX,ierr)
#else
 call MPI_WAITALL(nr,req,status,ierr)
#endif
!-from 1 to 2
 if(neighb(2)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call rcx2_vel(c_loc(rcx2), nxtop, nytop, nxbtm, nzbtm, nx2p1, nx2p2, ny2p1, ny2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx2_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   j0=2*(ny2p2+1)
   v2z(1,nx2p1,ny2p1) = rcx2(j0+1)
   v2z(1,nx2p2,ny2p1) = rcx2(j0+2)
   v2z(1,nx2p1,ny2p2) = rcx2(j0+3)
   v2z(1,nx2p2,ny2p2) = rcx2(j0+4)
#endif
 endif
!-from 3 to 4
 if(neighb(4)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call rcy2_vel(c_loc(rcy2), nxtop, nytop, nxbtm, nzbtm, nx2p1, nx2p2, ny2p1, ny2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy2_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   i0=2*(nx2p2+1)
   v2x(1,nx2p1,ny2p1) = rcy2(i0+1)
   v2x(1,nx2p1,ny2p2) = rcy2(i0+2)
   v2y(1,nx2p1,ny2p1) = rcy2(i0+3)
   v2y(1,nx2p2,ny2p1) = rcy2(i0+4)
#endif
 endif
!-from 2 to 1
 if(neighb(1)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call rcx1_vel(c_loc(rcx1), nxtop, nytop, nxbtm, nzbtm, ny2p1, ny2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcx1_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   j0=ny2p2+1
   v2x(1,-1,ny2p1) = rcx1(j0+1)
   v2x(1, 0,ny2p1) = rcx1(j0+2)
   v2x(1,-1,ny2p2) = rcx1(j0+3)
   v2x(1, 0,ny2p2) = rcx1(j0+4)
   v2y(1, 0,ny2p1) = rcx1(j0+5)
   v2z(1, 0,ny2p1) = rcx1(j0+6)
   v2z(1, 0,ny2p2) = rcx1(j0+7)
#endif
 endif
!-from 4 to 3
 if(neighb(3)>-1) then
#ifdef DISFD_GPU_MARSHALING
   if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
   call rcy1_vel(c_loc(rcy1), nxtop, nytop, nxbtm, nzbtm, nx2p1, nx2p2)
   if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU rcy1_vel", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
   i0=nx2p2+1
   v2x(1,   -1, 0) = rcy1(i0+1 )
   v2x(1,    0, 0) = rcy1(i0+2 )
   v2x(1,nx2p1, 0) = rcy1(i0+3 )
   v2y(1,    0,-1) = rcy1(i0+4 )
   v2y(1,nx2p1,-1) = rcy1(i0+5 )
   v2y(1,nx2p2,-1) = rcy1(i0+6 )
   v2y(1,    0, 0) = rcy1(i0+7 )
   v2y(1,nx2p1, 0) = rcy1(i0+8 )
   v2y(1,nx2p2, 0) = rcy1(i0+9 )
   v2z(1,    0, 0) = rcy1(i0+10)
   v2z(1,nx2p1, 0) = rcy1(i0+11)
   v2z(1,nx2p2, 0) = rcy1(i0+12)
#endif
 endif

 call record_time(tstart)
 call interpl_3vbtm
 call record_time(tend)
 write(*,*) "TIME :: interpl_3vbtm :", tend-tstart

 return
end subroutine velocity_interp
!
!-----------------------------------------------------------------------
subroutine interpl_3vbtm
 use grid_node_comm
 use wave_field_comm
 use itface_comm
 use:: iso_c_binding
! use logging
 use metadata
 use ctimer
 implicit NONE
 
 interface
      subroutine interpl_3vbtm_vel1( ny1p2, ny2p2, nz1p1, nyvx, nxbm1, nxbtm, &
                                     nzbtm, nxtop, nztop, neighb2, & 
                                    rcx2) bind (C, name="interpl_3vbtm_vel1")
                use :: iso_c_binding
                integer(c_int), intent(in) :: ny1p2, ny2p2, nz1p1, nyvx, nxbm1, &
                                              nxbtm, nzbtm, nxtop, nztop, neighb2
                type(c_ptr), intent(in), value :: rcx2
      end subroutine interpl_3vbtm_vel1               

      subroutine interpl_3vbtm_vel3( ny1p2, nz1p1, nyvx1, nxbm1, nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel3")
                use :: iso_c_binding
                integer(c_int), intent(in) :: ny1p2, nz1p1, nyvx1, nxbm1, &
                                              nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel3

      subroutine interpl_3vbtm_vel4( nx1p2, ny2p2, nz1p1, nxvy, nybm1, nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel4")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nx1p2, ny2p2, nz1p1, nxvy, nybm1, &
                                              nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel4

      subroutine interpl_3vbtm_vel5( nx1p2, nx2p2, nz1p1, nxvy, nybm1, nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel5")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nx1p2, nx2p2, nz1p1, nxvy, nybm1, &
                                              nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel5

      subroutine interpl_3vbtm_vel6( nx1p2, nz1p1, nxvy1, nybm1, nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel6")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nx1p2, nz1p1, nxvy1, nybm1, &
                                              nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel6

      subroutine interpl_3vbtm_vel7( nxbtm, nybtm, nzbtm, nxtop, nytop, nztop, sdx1) & 
                                     bind (C, name="interpl_3vbtm_vel7")
                use :: iso_c_binding
                integer(c_int), intent(in) ::nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
                type(c_ptr), intent(in), value :: sdx1
      end subroutine interpl_3vbtm_vel7

      subroutine interpl_3vbtm_vel8( nxbtm, nybtm, nzbtm, nxtop, nytop, nztop, sdx2) & 
                                     bind (C, name="interpl_3vbtm_vel8")
                use :: iso_c_binding
                integer(c_int), intent(in) ::nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
                type(c_ptr), intent(in), value :: sdx2

      end subroutine interpl_3vbtm_vel8

      subroutine interpl_3vbtm_vel9( nx1p2, ny2p1, nz1p1, nxvy, nybm1, nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop, neighb4) & 
                                     bind (C, name="interpl_3vbtm_vel9")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nx1p2, ny2p1, nz1p1, nxvy, nybm1, &
                                              nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop, neighb4
      end subroutine interpl_3vbtm_vel9

      subroutine interpl_3vbtm_vel11( nx1p2, nx2p1, ny1p1, nz1p1, nxvy1, nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel11")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nx1p2, nx2p1, ny1p1, nz1p1, nxvy1, &
                                              nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel11

      subroutine interpl_3vbtm_vel13( nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel13")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel13

      subroutine interpl_3vbtm_vel14( nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel14")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel14

      subroutine interpl_3vbtm_vel15( nxbtm, nybtm, &
                                     nzbtm, nxtop, nytop, nztop) & 
                                     bind (C, name="interpl_3vbtm_vel15")
                use :: iso_c_binding
                integer(c_int), intent(in) :: nxbtm, nybtm, nzbtm, &
                                              nxtop, nytop, nztop
      end subroutine interpl_3vbtm_vel15
 end interface
 integer:: i,j,i0,j0,ii,jj
 real::vv1,vv2,vv3,vv4 
 real(c_double)::gpu_tstart, gpu_tend, cpu_tstart, cpu_tend
 real(c_double):: total_gpu_marsh_time=0.0, total_cpu_marsh_time=0.0
 total_gpu_marsh_time =0.0
 total_cpu_marsh_time =0.0
#ifdef DISFD_GPU_MARSHALING
 if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel1( ny1p2, ny2p2, nz1p1, nyvx, nxbm1, nxbtm, nzbtm, &
                          nxtop, nztop, neighb(2), c_loc(rcx2))
! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_vel1", gpu_tend-gpu_tstart); &
!        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel3( ny1p2, nz1p1, nyvx1, nxbm1, nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop)
 ! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_3", gpu_tend-gpu_tstart); &
  !       total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
 
 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel4( nx1p2, ny2p2, nz1p1, nxvy, nybm1, nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop)
  if (neighb(4) > -1 ) then
        call interpl_3vbtm_vel5 (nx1p2, nx2p2, nz1p1, nxvy, nybm1, nxbtm, nybtm, &
                                nzbtm, nxtop, nytop, nztop)
  end if
 ! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_4/5", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel6( nx1p2, nz1p1, nxvy1, nybm1, nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop)
 ! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_6", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
 
 if(neighb(1) > -1) then 
 !  if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
     call interpl_3vbtm_vel7( nxbtm, nybtm, nzbtm, nxtop, nytop, nztop,c_loc(sdx1))
 !  if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_7", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
 end if

 if(neighb(2) > -1) then 
 !  if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
     call interpl_3vbtm_vel8( nxbtm, nybtm, nzbtm, nxtop, nytop, nztop, c_loc(sdx2))
 !  if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_8", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
 end if

 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel9( nx1p2, ny2p1, nz1p1, nxvy, nybm1, nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop, neighb(4))
 ! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_9", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel11( nx1p2, nx2p1, ny1p1, nz1p1, nxvy1, nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop)
 ! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_11", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel13( nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop)
 ! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_13", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel14( nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop)
 ! if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_14", gpu_tend-gpu_tstart); &
 !        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if

 ! if(DETAILED_TIMING .eq. 1) then;  call record_time(gpu_tstart) ; end if
  call interpl_3vbtm_vel15( nxbtm, nybtm, nzbtm, &
                          nxtop, nytop, nztop)
 if(DETAILED_TIMING .eq. 1) then; call record_time(gpu_tend); call log_timing ("GPU interpl_3vbtm_15", gpu_tend-gpu_tstart); &
        total_gpu_marsh_time = total_gpu_marsh_time + (gpu_tend-gpu_tstart); end if
#else
!---Vx----
 j0=ny2p2+2
 j=0
 do jj=0,nyvx
   vv1=v2x(1,-1,jj)
   vv2=v2x(1, 0,jj)
   vv3=v2x(1, 1,jj)
   vv4=v2x(1, 2,jj)
   i=-2
   do ii=1,nxbm1
     i=i+3
     v1x(nz1p1,i,j)=vv1*chx(1,ii)+vv2*chx(2,ii)+ &
                    vv3*chx(3,ii)+vv4*chx(4,ii)
     vv1=vv2
     vv2=vv3
     vv3=vv4
     vv4=v2x(1,ii+2,jj)
     v1x(nz1p1,i+1,j)=vv2
     v1x(nz1p1,i+2,j)=vv1*chx(5,ii)+vv2*chx(6,ii)+ &
                      vv3*chx(7,ii)+vv4*chx(8,ii)
   enddo
   if(neighb(2) > -1) then
     i=i+3
     v1x(nz1p1,i,j  )=vv1*chx(1,nxbtm)+vv2*chx(2,nxbtm)+ &
                      vv3*chx(3,nxbtm)+vv4*chx(4,nxbtm)
     v1x(nz1p1,i+1,j)=vv3
     v1x(nz1p1,i+2,j)=vv2*chx(5,nxbtm)+vv3*chx(6,nxbtm)+ &
                      rcx2(jj+j0)*chx(8,nxbtm)+vv4*chx(7,nxbtm)
   endif
   j=min0(jj*3+1,ny1p2)
 enddo
!
 do i=1,nxtop
   vv1=v1x(nz1p1,i,0)
   vv2=v1x(nz1p1,i,1)
   vv3=v1x(nz1p1,i,4)
   j=2
   do jj=1,nyvx1
     vv4=v1x(nz1p1,i,min0(j+5,ny1p2))
     v1x(nz1p1,i,j  )=vv1*ciy(1,jj)+vv2*ciy(2,jj)+ &
                      vv3*ciy(3,jj)+vv4*ciy(4,jj)
     v1x(nz1p1,i,j+1)=vv1*ciy(5,jj)+vv2*ciy(6,jj)+ &
                      vv3*ciy(7,jj)+vv4*ciy(8,jj)
     vv1=vv2
     vv2=vv3
     vv3=vv4
     j=j+3
   enddo
 enddo
!---Vy----
 i0=nx2p2+2
 i=0
 do ii=0,nxvy
   vv1=v2y(1,ii,-1)
   vv2=v2y(1,ii, 0)
   vv3=v2y(1,ii, 1)
   vv4=v2y(1,ii, 2)
   j=-2
   do jj=1,nybm1
     j=j+3
     v1y(nz1p1,i,j)=vv1*chy(1,jj)+vv2*chy(2,jj)+ &
                    vv3*chy(3,jj)+vv4*chy(4,jj)
     vv1=vv2
     vv2=vv3
     vv3=vv4
     vv4=v2y(1,ii,jj+2)
     v1y(nz1p1,i,j+1)=vv2
     v1y(nz1p1,i,j+2)=vv1*chy(5,jj)+vv2*chy(6,jj)+ &
                      vv3*chy(7,jj)+vv4*chy(8,jj)
   enddo
   if(neighb(4) > -1) then
     j=j+3
     v1y(nz1p1,i,j)=vv1*chy(1,nybtm)+vv2*chy(2,nybtm)+ &
                    vv3*chy(3,nybtm)+vv4*chy(4,nybtm)
     v1y(nz1p1,i,j+1)=vv3
     v1y(nz1p1,i,j+2)=vv2*chy(5,nybtm)+vv3*chy(6,nybtm)+ &
                      rcy2(ii+i0)*chy(8,nybtm)+vv4*chy(7,nybtm)
   endif
   i=min0(ii*3+1,nx1p2)
 enddo
!
 do j=1,nytop
   vv1=v1y(nz1p1,0,j)
   vv2=v1y(nz1p1,1,j)
   vv3=v1y(nz1p1,4,j)
   i=2
   do ii=1,nxvy1
     vv4=v1y(nz1p1,min0(i+5,nx1p2),j)
     v1y(nz1p1,i  ,j)=vv1*cix(1,ii)+vv2*cix(2,ii)+ &
                      vv3*cix(3,ii)+vv4*cix(4,ii)
     v1y(nz1p1,i+1,j)=vv1*cix(5,ii)+vv2*cix(6,ii)+ &
                      vv3*cix(7,ii)+vv4*cix(8,ii)
     vv1=vv2
     vv2=vv3
     vv3=vv4
     i=i+3
   enddo
 enddo
!---Vz----
 if(neighb(1) > -1) then
   j=2
   vv1 = rcx1(1)
   vv2 = rcx1(2)
   vv3 = rcx1(3)
   sdx1(1)=(vv1+vv2*2.)/3.0
   do jj=1,nybtm
     vv4 = rcx1(jj+3)
     sdx1(j)=vv2  
     sdx1(j+1)=vv1*ciy(1,jj)+vv2*ciy(2,jj)+vv3*ciy(3,jj)+vv4*ciy(4,jj)
     sdx1(j+2)=vv1*ciy(5,jj)+vv2*ciy(6,jj)+vv3*ciy(7,jj)+vv4*ciy(8,jj)
     vv1=vv2
     vv2=vv3
     vv3=vv4
     j=j+3
   enddo     
   sdx1(j  )=vv2
   sdx1(j+1)=(vv2*2+vv3)/3.0
 endif
 if(neighb(2) > -1) then
   j=2
   vv1 = rcx2(1)
   vv2 = rcx2(2)
   vv3 = rcx2(3)
   sdx2(1)=(vv1+vv2*2.)/3.0
   do jj=1,nybtm
     vv4 = rcx2(jj+3)
     sdx2(j)=vv2  
     sdx2(j+1)=vv1*ciy(1,jj)+vv2*ciy(2,jj)+vv3*ciy(3,jj)+vv4*ciy(4,jj)
     sdx2(j+2)=vv1*ciy(5,jj)+vv2*ciy(6,jj)+vv3*ciy(7,jj)+vv4*ciy(8,jj)
     vv1=vv2
     vv2=vv3
     vv3=vv4
     j=j+3
   enddo     
   sdx2(j  )=vv2
   sdx2(j+1)=(vv3+vv2*2.)/3.0
 endif
 i=0
 do ii=0,nxvy
   vv1=rcy1(ii+1)
   vv2=v2z(1,ii,0)
   vv3=v2z(1,ii,1)
   vv4=v2z(1,ii,2)
   v1z(nz1p1,i,0)=vv1*ciy(5,0)+vv2*ciy(6,0)+vv3*ciy(7,0)+vv4*ciy(8,0)
   j=1
   do jj=1,nybtm
     vv1=vv2
     vv2=vv3
     vv3=vv4
     vv4=v2z(1,ii,jj+2)
     v1z(nz1p1,i,j  )=vv2
     v1z(nz1p1,i,j+1)=vv1*ciy(1,jj)+vv2*ciy(2,jj)+ &
                      vv3*ciy(3,jj)+vv4*ciy(4,jj)
     v1z(nz1p1,i,j+2)=vv1*ciy(5,jj)+vv2*ciy(6,jj)+ &
                      vv3*ciy(7,jj)+vv4*ciy(8,jj)
     j=j+3
   enddo
   if(neighb(4) > -1) then
     v1z(nz1p1,i,j)=vv3
     v1z(nz1p1,i,j+1)=vv2*ciy(1,ny2p1)+vv3*ciy(2,ny2p1)+ &
                      rcy2(ii+1)*ciy(4,ny2p1)+vv4*ciy(3,ny2p1)
   endif
   i=min0(ii*3+1,nx1p2)
 enddo

 do j=0,ny1p1
   vv1=sdx1(j+1)
   vv2=v1z(nz1p1,0,j)
   vv3=v1z(nz1p1,1,j)
   vv4=v1z(nz1p1,4,j)
   v1z(nz1p1,0,j)=vv1*cix(5,0)+vv2*cix(6,0)+vv3*cix(7,0)+vv4*cix(8,0)
   i=2
   do ii=1,nxvy1
     vv1=vv2
     vv2=vv3
     vv3=vv4
     vv4=v1z(nz1p1,min0(i+5,nx1p2),j)
     v1z(nz1p1,i  ,j)=vv1*cix(1,ii)+vv2*cix(2,ii)+ &
                      vv3*cix(3,ii)+vv4*cix(4,ii)
     v1z(nz1p1,i+1,j)=vv1*cix(5,ii)+vv2*cix(6,ii)+ &
                      vv3*cix(7,ii)+vv4*cix(8,ii)
     i=i+3
   enddo
   v1z(nz1p1,i,j)=vv2*cix(1,nx2p1)+vv3*cix(2,nx2p1)+ &
                  sdx2(j+1)*cix(4,nx2p1)+vv4*cix(3,nx2p1)
 enddo
!---Region II---
!
 do jj=1,nybtm
   j=jj*3-2
   do ii=1,nxbtm
     v2x(0,ii,jj)=v1x(nztop,ii*3-1,j)
   enddo
 enddo
!
 do jj=1,nybtm
   j=jj*3-1
   do ii=1,nxbtm
     v2y(0,ii,jj)=v1y(nztop,ii*3-2,j)
   enddo
 enddo
!
 do jj=1,nybtm
   j=jj*3-2
   do ii=1,nxbtm
     v2z(0,ii,jj)=v1z(nztop,ii*3-2,j)
   enddo
 enddo
#endif
 if(DETAILED_TIMING .eq. 1) then; call log_timing ("GPU Marshaling interpl_3vbtm", total_gpu_marsh_time); end if
 return
end  subroutine interpl_3vbtm
!-----------------------------------------------------------------------
FUNCTION gammln(xx)
 implicit NONE
 real, intent(IN):: xx
 INTEGER:: j
 real:: gammln
 real(kind=8):: cof(6), ser,tmp,x,y, stp=2.5066282746310005d0
 cof=(/ 76.18009172947146d0,-86.50532032941677d0,24.01409824083091d0,&
       -1.231739572450155d0,.1208650973866179d-2,-.5395239384953d-5 /)
!
 x=dble(xx);     y=x
 tmp=x+5.5d0;    tmp=(x+0.5d0)*log(tmp)-tmp
 ser=1.000000000190015d0
 do j=1,6
   y=y+1.d0
   ser=ser+cof(j)/y
 enddo
 gammln=sngl(tmp+dlog(stp*ser/x))
 return
END FUNCTION gammln
!
!-----------------------------------------------------------------------
subroutine ini_plane_wave(myid,dt)
 use grid_node_comm
 use wave_field_comm
 use source_parm_comm
 use plane_wave_bd
 implicit NONE
 integer, intent(IN):: myid
 real, intent(IN):: dt
 integer:: im1,im2,jm1,jm2,km1,km2,nx22,ny22,nz22,nm,i,j,k
 integer:: ix2(72,4),jy2(72,4),kz2(72,4),ierr,id_pvh
 real:: psvsh_vector(3,3),rvps(3)
 real:: pi,st,dp,csst,snst,csdp,sndp,dt05
 real:: xs,ys,zs,x0,y0,z0,dsdx,dsdy,dsdz,rstim,cfamp,strike,dip
 real:: vxmm,vymm,vzmm,uxmm,uymm,uzmm,tpmm,t1mm,t2mm,t3mm,t4mm,t5mm,t6mm
 real:: v11,v22,xi,yi,zi,xhf,yhf,zhf,smu,clm,pl2m,rv,xr,yr,zr
 real:: tpw,tpx,tpy,tpz,ux,uy,uz,trav,taper,arg,vel_plan
!
 pi=atan(1.0)*4.0
 nfadd=0
 dt05=0.5*dt
 do id_pvh=1,3
! Using Brune far field slip rate function
   xs=famp(id_pvh,1)
   ys=famp(id_pvh,2)
   zs=famp(id_pvh,3)
   rstim=riset(id_pvh)
   cfamp=famp(id_pvh,4)*0.5
   strike=famp(id_pvh,5)
   dip=famp(id_pvh,6)
   st=strike*pi/180.0
   dp=dip*pi/180.0
   csst=cos(st)
   snst=sin(st)
   csdp=cos(dp)
   sndp=sin(dp)
   psvsh_vector(1,1)= sndp*snst
   psvsh_vector(2,1)=-sndp*csst
   psvsh_vector(3,1)= csdp
   psvsh_vector(1,2)=-csdp*snst
   psvsh_vector(2,2)= csdp*csst
   psvsh_vector(3,2)= sndp
   psvsh_vector(1,3)=-csst
   psvsh_vector(2,3)=-snst
   psvsh_vector(3,3)= 0.0
   x0=xs+gridx2(1)
   y0=ys+gridy2(1)
   z0=zs+gridz(1)
   dsdx=psvsh_vector(1,1)
   dsdy=psvsh_vector(2,1)
   dsdz=psvsh_vector(3,1)
   vxmm=0.0
   vymm=0.0
   vzmm=0.0
   uxmm=0.0
   uymm=0.0
   uzmm=0.0
   tpmm=0.0
   t1mm=0.0
   t2mm=0.0
   t3mm=0.0
   t4mm=0.0
   t5mm=0.0
   t6mm=0.0
   v11=50000
   v22=0.0
   im1=1000000
   im2=0
   jm1=1000000
   jm2=0
   km1=1000000
   km2=0
   nx22=nxbtm+2
   ny22=nybtm+2
   do j=-1,ny22
   do i=-1,nx22
   do k= 1,nzbtm-mw2_pml
     xi=gridx2(i)-x0
     yi=gridy2(j)-y0
     zi=gridz(k+nztop)- z0
     xhf=xi+0.5*(gridx2(i+1)-gridx2(i))
     yhf=yi+0.5*(gridy2(j+1)-gridy2(j))
     zhf=zi+0.5*(gridz(k+1+nztop)-gridz(k+nztop))
     nm=idmat2(k,min0(max0(i,1),nxbtm), min0(max0(j,1),nybtm))
     smu=cmu(nm)
     clm=clamda(nm)
     pl2m=clm+2.*smu
     if(id_pvh == 1) then
       rv=1./sqrt(pl2m*rho(nm))
     else
       rv=1./sqrt(smu*rho(nm))
     endif 
     v11=amin1(v11,1./rv)
     v22=amax1(v22,1./rv)
     xr=xplwv(1)+gridx2(i)-gridx2(1)
     yr=yplwv(1)+gridy2(j)-gridy2(1)
     zr=gridz(k+nztop)-gridz(1)
     tpw=pi/(xplwv(3)-xplwv(2))
     tpx=1.0
     if(xr<xplwv(3)) then
       tpx=0.5-0.5*cos(tpw*amax1(0.0,xr-xplwv(2)))
     endif
     if(xr>xplwv(4)) then
       tpx=0.5-0.5*cos(tpw*amax1(0.0,xplwv(5)-xr))
     endif
     tpy=1.0
     if(yr<yplwv(3)) then
       tpy=0.5-0.5*cos(tpw*amax1(0.0,yr-yplwv(2)))
     endif
     if(yr>yplwv(4)) then
       tpy=0.5-0.5*cos(tpw*amax1(0.0,yplwv(5)-yr))
     endif
     tpz=1.0
     if(zr>zplwv(1)) then
       tpz=0.5-0.5*cos(tpw*amax1(0.0,zplwv(2)-zr))
     endif
     taper=tpx*tpy*tpz
     ux=cfamp*psvsh_vector(1,id_pvh)*taper
     uy=cfamp*psvsh_vector(2,id_pvh)*taper
     uz=cfamp*psvsh_vector(3,id_pvh)*taper
! Vx
     trav=(xhf*dsdx + yi*dsdy + zhf*dsdz)*rv
     if(trav>0.0 .and. i<nx22 .and. j>=0 ) then
       arg=-trav/rstim
       v2x(k,i,j) = v2x(k,i,j) + vel_plan(trav,rstim)*ux
       vxmm=amax1(vxmm,abs(v2x(k,i,j)))
     endif
! Vy
     trav=(xi*dsdx + yhf*dsdy + zhf*dsdz)*rv
     if(trav>0.0 .and. j<ny22 .and. i>=0 ) then
       arg=-trav/rstim
       v2y(k,i,j) = v2y(k,i,j) + vel_plan(trav,rstim)*uy
       vymm=amax1(vymm,abs(v2y(k,i,j)))
     endif
! Vz
     trav=(xi*dsdx + yi*dsdy + zi*dsdz)*rv
     if(trav>0.0 .and. i>=0 .and. j>=0 ) then
       arg=-trav/rstim
       v2z(k,i,j) = v2z(k,i,j) + vel_plan(trav,rstim)*uz
       vzmm=amax1(vzmm,abs(v2z(k,i,j)))
     endif
     uxmm=amax1(uxmm,abs(ux))
     uymm=amax1(uymm,abs(uy))
     uzmm=amax1(uzmm,abs(uz))
     tpmm=amax1(tpmm,abs(taper))
! Normal Stress
     trav=(xi*dsdx + yi*dsdy + zhf*dsdz)*rv+dt05
     arg=-trav/rstim
     if(trav>0.0 .and. i>=0 .and. j>0 .and. j<=nybtm) then
       t2xx(k,i,j) =t2xx(k,i,j) + vel_plan(trav,rstim)*rv* &
                    (pl2m*dsdx*ux+clm*(dsdy*uy+dsdz*uz))
       t1mm=amax1(t1mm,abs(t2xx(k,i,j)))
       im1=min0(im1,i)
       im2=max0(im2,i)
       jm1=min0(jm1,j)
       jm2=max0(jm2,j)
       km1=min0(km1,k)
       km2=max0(km2,k)
     endif
     if(trav>0.0 .and. i>0 .and. i<=nxbtm .and. j>=0 ) then
       t2yy(k,i,j) =t2yy(k,i,j)+vel_plan(trav,rstim)*rv* &
                    (pl2m*dsdy*uy+clm*(dsdx*ux+dsdz*uz))
       t2mm=amax1(t2mm,abs(t2yy(k,i,j)))
     endif
     if(trav>0.0 .and. i>0 .and. i<=nxbtm .and. j>0 .and. j<=nybtm ) then
       t2zz(k,i,j) = t2zz(k,i,j)+vel_plan(trav,rstim)*rv* &
                    (pl2m*dsdz*uz+clm*(dsdx*ux+dsdy*uy))
       t3mm=amax1(t3mm,abs(t2zz(k,i,j)))
     endif
! Txy 
     trav=(xhf*dsdx + yhf*dsdy + zhf*dsdz)*rv+dt05
     if(trav>0.0 .and. i<nx22 .and. j<ny22 ) then
       arg=-trav/rstim
       t2xy(k,i,j) = t2xy(k,i,j)+vel_plan(trav,rstim)*rv* &
                     smu*(dsdy*ux+dsdx*uy)
       t4mm=amax1(t4mm,abs(t2xy(k,i,j)))
     endif
! Txz 
     trav=(xhf*dsdx + yi*dsdy + zi*dsdz)*rv+dt05
     if(trav>0.0 .and. i<nx22 .and. j>0 .and. j<=nybtm ) then
       arg=-trav/rstim
       t2xz(k,i,j) = t2xz(k,i,j)+vel_plan(trav,rstim)*rv* &
                     smu*(dsdz*ux+dsdx*uz)
       t5mm=amax1(t5mm,abs(t2xz(k,i,j)))
     endif
! Tyz 
     trav=(xi*dsdx + yhf*dsdy + zi*dsdz)*rv+dt05
     if(trav>0.0 .and. i>0 .and. i<=nxbtm .and. j<ny22 ) then
       arg=-trav/rstim
       t2yz(k,i,j) =t2yz(k,i,j)+vel_plan(trav,rstim)*rv* &
                    smu*(dsdy*uz+dsdz*uy)
       t6mm=amax1(t6mm,abs(t2yz(k,i,j)))
     endif
   enddo
   enddo
   enddo
 enddo
 
end subroutine ini_plane_wave
!
!-----------------------------------------------------------------------
function vel_plan(t,rstim)
 implicit NONE
 real, intent(IN):: t,rstim
 real:: arg,vel_plan
!  Brune's (t/rstim*exp(-t/rstim)
! if(t>0.0) then
!   arg=-t/rstim
!   vel_plan=exp(arg)*(1.+arg)/rstim
! else
!   vel_plan=0.0
! endif
! 1-cos(2*pi*t/rstim)
 if(t>0.0 .and. t<rstim) then
   arg=2.*3.1415926/rstim
   vel_plan=-arg*sin(arg*t)
 else
   vel_plan=0.0
 endif
 return
end function vel_plan
!
!-----------------------------------------------------------------------
subroutine dig2ch(n,nch,ch)
 implicit NONE
 character(len=*), intent(OUT):: ch
 integer, intent(IN):: n,nch
 integer:: nzero,ntmp,n1,n2,i
!
 nzero=ichar('0')
 ntmp=n
 do i=nch,1,-1
   n1=ntmp/10
   n2=ntmp-n1*10
   ch(i:i)=char(nzero+n2)
   ntmp=n1
 enddo
 return
end subroutine dig2ch
!
!-----------------------------------------------------------------------
function getlen(string)
 implicit NONE
 character(len=*), intent(INOUT):: string
 character(len=len(string)):: chatmp
 integer i,nch, getlen
 chatmp=adjustl(string)
 nch=len_trim(chatmp)
 string=''
 string=chatmp(1:nch)
 getlen=nch
 return
end function getlen
