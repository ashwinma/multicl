module fd_model_comm
 implicit NONE
 save
 integer, parameter:: nll=8
 real, parameter:: ar_ha=-1.0, errm=0.001
! vm4fd 
 integer:: ntype,mtype
 integer, allocatable, dimension(:):: idmat
 real, allocatable, dimension(:):: evp,evs,erh,eqp,eqs
! viscty
 real, dimension(nll):: epdt,qwt1,qwt2,relaxt
! 2D interpolation 
 integer, allocatable, dimension(:,:,:):: idfxy
 real, allocatable, dimension(:,:,:):: cefxy,rbkly,rmuly, & 
                                       rholy,qply,qsly
! inorder
 integer, allocatable, dimension(:):: ntyp_order
end module fd_model_comm
!
module grid_sp_num
 implicit NONE
 save
! bottom
 integer:: nx_fd0,ny_fd0,nz_fd0,nz1_fd,nz2_fd
 real, allocatable, dimension(:):: dxx0,dyy0,dzz0
! fdnode
 integer :: nx_fd,ny_fd,nz_fd
 real, allocatable, dimension(:):: dxx,dyy,dzz
end module grid_sp_num
!
module vmodel_inp
 implicit NONE
 save
 character(len=72), allocatable, dimension(:):: fileall
 character(len=72):: is_1Dmodel
 integer:: num_file,nx_vm,ny_vm,nz_vm,modlz
 real:: dx_vm,dy_vm,dz_vm
 integer, allocatable, dimension(:,:):: nodexyz
 real, allocatable, dimension(:,:):: gdsp_vm,rbkmd,rmumd,roth,qpmd,qsmd
 real, allocatable, dimension(:):: top_depth
 real:: vpmin,vsmin,rohmin,qpmin,qsmin,qqmax
end module vmodel_inp
!
Program vm4dfd_v3
!------------------------------------------------------------------------|
!   This code construct velocity strucure parameters for FD code         |
!                                                                        |
!   Note: The velocity structure model and FD grid system may have       |
!   different  x-direction or y-direction.                               |
!------------------------------------------------------------------------|
 use vmodel_inp
 use grid_sp_num
 use fd_model_comm
 implicit NONE
 character(len=72):: fileout
 integer, parameter:: nwrt=20
 integer:: initxdr,ixdrimat,ixdrrmat,ixdrint,ixdrreal,ixdrclose
 integer:: i,ix,jy,kz,ncx,ncy,ncz,modlx,modly,ncount,nof,myfh,ierr
 integer:: k,kup,kbm,kzsub_vm,nchk,is_refresh,ndis,kdis,inds,itype
 integer:: iwrt,nnode(nwrt)
 real:: depth_I_II,fmax,freq_ref,anglex,angley,xmd0,ymd0
 real:: tstep,gstep,ztb1,ztb2,z2_top
 real:: depth_bm,depth_up,z0,zz,dzbm,zr2,dt
 real:: vpmx1,vpmn1,vsmx1,vsmn1,rhmx1,rhmn1,qpmx1,qpmn1,qsmx1,qsmn1
 real, allocatable, dimension(:):: dxseg,distx,dyseg,disty,dzseg,distz
!
 open(9,file='vm4dfd_v3.inf')
 open(8,file='vm4dfd_v3.inp')
!
!  Here begin the inputting
!  

 write(9,*) 'Enter the depth of interface if you know the value.'
! If you did not know the depth of interface, enter a negative value
 read(8,*) depth_I_II
 write(9,*) depth_I_II
!
 write(9,*) 'Enter the Max. frequency (fmax) of FD computation '
 write(9,*) 'and the Reference frequency of Velocity Structure'
 read(8,*) fmax,freq_ref
 write(9,*) fmax,freq_ref
!
!??????????????????????????????????????????????????????????????????????
! Following parameters are used to generate the node number and grid
! space in X-axis, Y-axis, and Z-axis (nx_fd,dxx,ny_fd,dyy,nz_fd,dzz)
!
! ncx, ncy, ncz are the numbers of control points used in X-axis, 
! Y-axis, Z-axis, respectively.
! distx(*), disty(*), and distz(*) are the locations of control points
! on X-axis,  Y-axis, Z-axis, respectively.
! Normally, distx(1), disty(1), or distz(1) equals to zero
! distx(ncx), disty(ncy), and distz(ncz) equal to the length of
! FD system in X-axis, Y-axis, Z-axis, respectively.
! dxseg(i), dyseg(i) and  dzseg(i) are the grid space at the point of
! distx(i), disty(i), distz(i), respectively.
!
! The grid space between distx(i) and distx(i+1) is calculated by
! linear-interpolating the values of dxseg(i) and dxseg(i+1).
!
! Assumeing the depth of interface of Region I and Region II is Hi
!       Hi = (depth of interface given by Vm4dfd)
!           +(3--5)*(minimum grid space suggested by Vm4dfd)
!       
!  In the region above the depth Hi+5*dz1, it is better to set vertical 
!    grid space dz of dz1 (dz1 should be the minimum grid space suggested 
!    by Vm4nufd); between Hi+5*dz1 and Hi+(15--20)*dz1, the gird space 
!    gradually increase from dz1 to dz2 (dz2=3*dz1). Below this region, 
!    set gird space dz=dz2=3*dz1.
! 
! The value of inputting horizotal grid space of control point should be
! for Region II. If you use regular horizotal grid space, in Region I 
! and II, respectively, you can input
! distx(ix)=3*(minimum grid space suggested by Vm4dfd)
! disty(iy)=3*(minimum grid space suggested by Vm4dfd)
!??????????????????????????????????????????????????????????????????????
! 
 write(9,*) 'Enter the number of control points used in X-axis of FD'
 read(8,*) ncx
 write(9,*) ncx
!
 allocate(distx(ncx),dxseg(ncx))
 write(9,*) 'Enter the location and grid sapce of control points ', &
            'in X-axis for Region II' 
 do ix=1,ncx
   read(8,*) distx(ix),dxseg(ix)
   write(9,*) distx(ix),dxseg(ix)
 enddo
!
 write(9,*) 'Enter the number of control points used in Y-axis of FD '
 read(8,*) ncy
 write(9,*) ncy
!
 allocate(disty(ncy),dyseg(ncy))
 write(9,*) 'Enter the location and grid sapce of control points ', &
            'in Y-axis for Region II'
 do jy=1,ncy
   read(8,*) disty(jy),dyseg(jy)
   write(9,*) disty(jy),dyseg(jy)
 enddo
!
 write(9,*) 'Enter the number of control points  used in Z-axis of FD '
 read(8,*) ncz
 write(9,*) ncz
!
 allocate(distz(ncz),dzseg(ncz))
 write(9,*) 'Enter the location and grid sapce of control points in Z-axis'
 do kz=1,ncz
   read(8,*) distz(kz),dzseg(kz)
   write(9,*) distz(kz),dzseg(kz)
 enddo
!
 write(9,*) 'Enter min. Vp, Vs, roh, Qp, and Qs used in FD model' 
 read(8,*) vpmin,vsmin,rohmin,qpmin,qsmin
 write(9,*) vpmin,vsmin,rohmin,qpmin,qsmin
!
! ================== Begin Enter Velocity Model =====================
 write(9,*) 'The velocity structure is 1D model(y) or not(n)'
 read(8,'(1a)') is_1Dmodel
 write(9,'(1a)') is_1Dmodel
!
 write(9,*) 'How many files are used to store whole velocity structure model'
! For 1D structure, num_file can only be 1
 read(8,*) num_file
 write(9,*) num_file
!
 allocate(fileall(num_file), top_depth(num_file))
 allocate(nodexyz(3,num_file), gdsp_vm(3,num_file))
 do nof=1,num_file
   write(9,*) 'Enter the file name of velocity model'
   read(8,'(1a)') fileall(nof)
   write(9,'(1a)') fileall(nof)
   write(9,*) 'Enter the depth (m) of the top of this structure model '
   read(8,*) top_depth(nof)
   write(9,*) top_depth(nof)
   write(9,*) 'Enter node number (nx,ny,nz) of each part of Structure'
   read(8,*)  nodexyz(1,nof),nodexyz(2,nof),nodexyz(3,nof)
   write(9,*) nodexyz(1,nof),nodexyz(2,nof),nodexyz(3,nof)
   write(9,*) 'Enter grid spaces (dx, dy,dz) of each part of Structure'
   read(8,*)  gdsp_vm(1,nof),gdsp_vm(2,nof),gdsp_vm(3,nof)
   write(9,*) gdsp_vm(1,nof),gdsp_vm(2,nof),gdsp_vm(3,nof)
 enddo
!
! ================== End Enter Velocity Model =====================
 write(9,*) 'Enter angles (degree) clockwise counting from X-axis of FD system '
 write(9,*) 'to X and Y-axis of Structure model, respectively'
 read(8,*)  anglex,angley
 write(9,*) anglex,angley
!
 write(9,*) 'Enter the (x,y) of the first point of structure model'
 write(9,*) 'in the of FD coordinate system'
 read(8,*) xmd0, ymd0
 write(9,*) xmd0, ymd0
!
 write(9,*) 'Give the name of outputing velocity model of FD'
 read(8,'(1a)') fileout
 write(9,'(1a)') fileout
 close(8)
!######################################################################
!
 call set_vm_dim(ncx,ncy,ncz,anglex,angley,xmd0,ymd0,distx,disty,distz)
! Determining the interface between region I and region II
 if(depth_I_II <= 0.0) then
   call inter_face(fmax,depth_I_II)
 endif
!
! Determining the grid space and node number
 call grid_space(1,ncx,distx,dxseg)
 call grid_space(2,ncy,disty,dyseg)
 call grid_space(3,ncz,distz,dzseg)
!
 tstep=-1.0;   gstep=-1.0
 nz1_fd=1;     z2_top=dzz0(1)
 do while(z2_top < depth_I_II)
    nz1_fd=nz1_fd+1
    z2_top=z2_top+dzz0(nz1_fd)    
 enddo
 ztb1=z2_top-dzz0(nz1_fd)*0.5
 ztb2=z2_top+dzz0(nz1_fd+1)*0.5
 ndis=1;     nz2_fd=nz_fd0
 if(nz1_fd < nz_fd0) then
   ndis=2;   nz2_fd=nz_fd0-nz1_fd
 endif
!
 myfh=15 
 open(unit=15,file=fileout,status='replace',form='unformatted')
! open(unit=15,file=fileout)
 !open(42,file='fd_north.inf')
 write(15) nx_fd0, ny_fd0, nz2_fd, nz1_fd, nwrt
 write(15) (dxx0(i),i=1,nx_fd0)
 write(15) (dyy0(i),i=1,ny_fd0)
 write(15) (dzz0(i),i=1,nz_fd0)
!=====
! Construct 3-D Velocity of model for FD
 itype=0;  inds=0; iwrt=0; nnode=-999
 ntype=0;  kup=1;  kbm=2;  kzsub_vm=0;  nchk=0;  depth_bm=-1.e+10
 do kdis=1,ndis
   is_refresh=kdis-1
   call switch(kdis)
   z0 =0.0+(kdis-1.0)*z2_top
   do k=1,nz_fd
     zz=z0+0.5*dzz(k)
     z0=z0+dzz(k)
     if(is_1Dmodel(1:1)=='Y') zz=z0
     if(nchk<num_file .or. kzsub_vm>0) then
       do while(zz >= depth_bm)
         if(nodexyz(3,nchk+1) == 0) then
           nchk=nchk+1
           cycle
         endif
         kup=kbm;  depth_up=depth_bm;  kbm=3-kup
! Reading one layer from the velocity structure model 
         call readvel(myfh,nchk,kzsub_vm,depth_bm)
         dzbm=depth_bm-depth_up
! Dtermining the coeficients for doing the bi-linear interpolation
         if(kzsub_vm==1 .or. is_refresh>0) then
           call coef_interp(anglex,angley,xmd0,ymd0)
           is_refresh=0
         endif
! Bi-linear interpolation
         call h2d_interp(kbm,nx_fd,ny_fd)
!         call smooth(nchk,kzsub_vm,kbm,nx_fd,ny_fd,ztb1,ztb2)
       enddo
     endif
!
     if(depth_up <0.0 .or. dzbm<0.001*dzz(k)) then
       kup=kbm;  zr2=0.0
     else
       zr2=(zz-depth_up)/dzbm
     endif
     if(is_1Dmodel(1:1)=='Y') zr2=(zz-depth_up)/dzz(k)
     zr2=amax1(amin1(zr2,1.0), 0.0)
     call v1d_interp(zr2,k,kup,kbm,tstep,gstep)
!
     ncount=ny_fd*nx_fd
     if(itype==0) itype=idmat(1)
     do i=1,ncount
       if(itype /= idmat(i) ) then
         iwrt=iwrt+2
         nnode(iwrt-1)=inds
         nnode(iwrt)  =itype
         inds=0
         itype=idmat(i)
         if(iwrt == nwrt) then
           write(15) nnode
           iwrt=0
           nnode=-999
         endif
       endif
       inds=inds+1
     enddo
   enddo
 enddo
 nnode(iwrt+1)=inds
 nnode(iwrt+2)=itype
 write(15) nnode
 nnode=-999
 write(15) nnode
! 
! Determining the coefficient for modeling Q
 dt = 0.48 * tstep
 call format_dt(dt)
 call smq_coef(1, dt,freq_ref,qsmn1,qsmx1,qpmn1,qpmx1)
!
! Outputing ntype and dt
  write(15) ntype, nll, dt
!
! Outputing Vp, Vs, rho, Qp, and Qs
 write(15) (evp(i), i=1,ntype)
 write(15) (evs(i), i=1,ntype)
 write(15) (erh(i), i=1,ntype)
 write(15) (eqp(i), i=1,ntype)
 write(15) (eqs(i), i=1,ntype)
!
! Outputing epdt,qwt1,qwt2
  write(15) epdt
  write(15) qwt1
  write(15) qwt2
!
! close the file of myth
 close(15)
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 write(9,*)
 write(9,*) '--------------------------------------------------'
 write(9,*) 'The node number of X,Y,Z-axis in Region II'
        write(9,*) nx_fd0, ny_fd0, nz2_fd, nz1_fd
 write(9,*)
 write(9,*) 'The number of different material: ',ntype
 write(9,*)
 vpmx1=0.0
 vpmn1=evp(1)
 vsmx1=0.0
 vsmn1=evs(1)
 rhmx1=0.0
 rhmn1=erh(1)
 do k=1,ntype
   vpmn1=amin1(vpmn1,evp(k))
   vpmx1=amax1(vpmx1,evp(k))
   vsmn1=amin1(vsmn1,evs(k))
   vsmx1=amax1(vsmx1,evs(k))
   rhmn1=amin1(rhmn1,erh(k))
   rhmx1=amax1(rhmx1,erh(k))
 enddo
 write(9,*) 'max. and min. P-velocity: ',vpmx1,vpmn1
 write(9,*) 'max. and min. S-velocity: ',vsmx1,vsmn1
 write(9,*) 'max. and min. Density: ',rhmx1,rhmn1
 write(9,*) 'max. and min. Qp: ',qpmx1,qpmn1
 write(9,*) 'max. and min. Qs: ',qsmx1,qsmn1
 write(9,*) 
 write(9,*) 'exp(-dt/relaxt),    qwt1,      qwt2'
 do k=1,8
   write(9,711) epdt(k), qwt1(k), qwt2(k)
 enddo
 711 format(3e14.5) 
 write(9,*) '  '
 gstep=gstep/fmax
 write(9,*) 'The time step is ', dt
 write(9,*) 'The minimum node-number per wavelength is ',gstep
 write(9,*) ' '
 stop
end program vm4dfd_v3

!=======================================================================
subroutine set_vm_dim(ncx,ncy,ncz,anglex,angley,xmd0,ymd0, &
                      distx,disty,distz)
 use vmodel_inp
 implicit NONE
 character(len=72):: chtmp
 integer, intent(IN):: ncx,ncy,ncz
 real, intent(INOUT):: anglex,angley,xmd0,ymd0
 real, intent(IN):: distx(ncx),disty(ncy),distz(ncz)
 integer:: nof,kup,k,modlx,modly,ntmp(3)
 real:: z0, zz, rtmp(3) 
!
 if(is_1Dmodel(1:1)=='Y' .or. is_1Dmodel(1:1)=='y') then
   is_1Dmodel(1:1)='Y'
   num_file=1
   anglex =0.0
   angley =90.0
   xmd0 =0.0
   ymd0 =0.0
   top_depth=0.0
   nodexyz =2
   gdsp_vm(1,1)=distx(ncx)-distx(1)+1.e+4
   gdsp_vm(2,1)=disty(ncy)-disty(1)+1.e+4
   gdsp_vm(3,1)=distz(ncz)-distz(1)+1.e+4
 endif
!
! From top to bottom
 do nof=1,num_file-1
 do k=nof+1, num_file
   if(top_depth(nof) > top_depth(k)) then
     chtmp=fileall(nof)
     z0=top_depth(nof)   
     ntmp=nodexyz(:,nof)
     rtmp=gdsp_vm(:,nof)
     fileall(nof)=fileall(k)
     fileall(k)=chtmp
     top_depth(nof)=top_depth(k)
     top_depth(k)=z0   
     nodexyz(:,nof)=nodexyz(:,k)
     nodexyz(:,k)=ntmp
     gdsp_vm(:,nof)=gdsp_vm(:,k)
     gdsp_vm(:,k)=rtmp
   endif
 enddo
 enddo
! Remove the overlap
 do nof=1,num_file-1
   kup=0
   zz=top_depth(nof)
   z0=top_depth(nof+1)-0.01*gdsp_vm(3,nof)
   do k=1,nodexyz(3,nof)
     if(zz >= z0) exit
     kup=k
     zz=zz+gdsp_vm(3,nof)
   enddo
   nodexyz(3,nof)=kup
 enddo
 modlx=0;   modly=0;   modlz=1
 do nof=1,num_file
   modlx=max0(modlx,nodexyz(1,nof))
   modly=max0(modly,nodexyz(2,nof))
   modlz=modlz+nodexyz(3,nof)
 enddo 
 if(is_1Dmodel(1:1)=='Y') modlz=1000
 allocate(roth(modlx,modly))
 allocate(rbkmd(modlx,modly), rmumd(modlx,modly))
 allocate(qpmd(modlx,modly),  qsmd(modlx,modly))
end subroutine set_vm_dim
!=======================================================================
subroutine readvel(myfh,nfile,kzsub_vm,depth_bm)
 use vmodel_inp
 implicit NONE
 integer, intent(IN):: myfh
 integer, intent(INOUT):: nfile,kzsub_vm
 real, intent(INOUT):: depth_bm
 integer:: id_vm,ix,jy
 real:: xxp,yyp,depth,thick,vpmd,vsmd,den,qpm,qsm,cmu
! checking if it is necessary to open a new file
 id_vm=19;  if(id_vm == myfh) id_vm=17
 if(kzsub_vm == 0) then
! Open a new file
   nfile=nfile+1
   nx_vm=nodexyz(1,nfile)
   ny_vm=nodexyz(2,nfile)
   nz_vm=nodexyz(3,nfile)
   dx_vm=gdsp_vm(1,nfile)
   dy_vm=gdsp_vm(2,nfile)
   dz_vm=gdsp_vm(3,nfile)
!
   if(is_1Dmodel(1:1)=='Y') then
     open(unit=id_vm,file=fileall(1),status='old',position='rewind')
     read(id_vm,*) nz_vm
     depth_bm=0.0
   else
!     open(unit=id_vm,file=fileall(nfile),form='unformatted')
     open(unit=id_vm,file=fileall(nfile),status='old',position='rewind')
   endif
 endif
! read the material value from a file
 do jy=1,ny_vm
 do ix=1,nx_vm
   if(is_1Dmodel(1:1)=='Y') then
!1D
     if(jy==1 .and. ix==1) read(id_vm,*) thick,vpmd,vsmd,den,qpm,qsm
   else
!3D
     read(id_vm,*) xxp,yyp,depth,vpmd,vsmd,den,qpm,qsm
! xxp, yyp, depth no useful
   endif

!================================================================
   den =amax1(den, rohmin)  
   vsmd=amax1(vsmd,vsmin)
   vpmd=amax1(vpmd,vpmin)
   if(vpmd < 1.5*vsmd) vpmd=1.5*vsmd
   if(vsmd <= 1000.0) then
     qsm=0.06*vsmd
   else
     qsm=0.16*vsmd
     if(vsmd <= 2000.0) qsm=0.14*vsmd
   endif
   qpm=1.5*qsm
   qpm=amax1(qpm, qpmin)
   qsm=amax1(qsm, qsmin)
! density
   roth(ix,jy)=den 
! shear modulus
   cmu=den*vsmd**2
   rmumd(ix,jy)=cmu 
! bulk muduls (lamada+2/3*mu)
   rbkmd(ix,jy)=den*vpmd**2-4./3.*cmu       
! Q value
   qpmd(ix,jy)=qpm 
   qsmd(ix,jy)=qsm 
 enddo
 enddo
!
 kzsub_vm=kzsub_vm+1
 if(is_1Dmodel(1:1)=='Y') then
   depth_bm=depth_bm+thick 
   if(kzsub_vm==nz_vm) depth_bm=dz_vm 
 else
   depth_bm=top_depth(nfile)+(kzsub_vm-1)*dz_vm
 endif
 if(kzsub_vm==nz_vm) then
   kzsub_vm=0
   close(id_vm)
 endif
end subroutine readvel
!=======================================================================
subroutine inter_face(fmax,dzbtm)
 use vmodel_inp
 implicit NONE
 real, intent(IN):: fmax
 real, intent(OUT):: dzbtm
 integer:: ix,jy,kz,nfile,nzz
 real:: dx,dx1,vs1,vs2,vsmd
 real, dimension(modlz):: vhm,hhh
!
 kz=0;  nfile=0;  hhh(1)=0.0
 do while(nfile < num_file)
   nzz=0
   if(nodexyz(3,nfile+1) == 0) then
     nfile=nfile+1
     cycle
   endif
   do
     call readvel(0,nfile,nzz,dzbtm)
     vs1=1.e+10
     do jy=1,ny_vm
     do ix=1,nx_vm
       vsmd=sqrt(rmumd(ix,jy)/roth(ix,jy))
       vs1=amin1(vs1,vsmd)
     enddo
     enddo
     kz=kz+1
     vhm(kz)=vs1
     if(is_1Dmodel(1:1)=='Y') then
       hhh(kz+1)=dzbtm
     else
       hhh(kz)=dzbtm
     endif
     if(nzz <=0 ) exit
   enddo
 enddo
! find interface
 nzz=kz; vs1=vhm(1)
 do ix=1,kz-1
   vs1=amin1(vs1,vhm(ix))
   vs2=vhm(ix+1)
   do jy=ix+1, kz
     vs2=amin1(vs2, vhm(jy))
   enddo
   if(vs2 >= 3.0*vs1) then
     nzz=ix+1
     exit
   endif
 enddo
!
! grid space
 dzbtm=hhh(nzz)
 vsmd=fmax*5.5
 dx1=vhm(1)/vsmd
 do ix=1,nzz
   dx1=amin1(dx1, vhm(ix)/vsmd)
 enddo
 dx=3.0*dx1
 do ix=nzz+1, kz
   dx=amin1(dx, vhm(ix)/vsmd)
 enddo
!
 write(*,*) 'The top of region II will locat at: ',dzbtm
 write(*,*) 'The minimum grid space should be or less than: ',dx/3.0
 write(*,*) 'Please modify the input file and re-run this code'
 write(*,*) 
 stop
end subroutine inter_face
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine h2d_interp(kbm,nx_fd,ny_fd)
 use vmodel_inp
 use fd_model_comm
 implicit NONE
 integer, intent(IN):: kbm,nx_fd,ny_fd
 integer:: i,j,i1,j1,i2,j2
 real:: rrx,rry,rr1,rr2,rr3,rr4
! do bi-linear interpolution for each layer
 do j=1,ny_fd
 do i=1,nx_fd
   i1 = idfxy(1,i,j);      j1 = idfxy(2,i,j)
   i2 = i1+1;              j2 = j1+1
   rrx=cefxy(1,i,j);       rry=cefxy(2,i,j)
   rr1=(1.-rrx)*(1.-rry);  rr2=rrx *(1.-rry)
   rr3=(1.-rrx)*rry;       rr4=rrx *rry
   rholy(kbm,i,j)=rr1*roth(i1,j1)+rr2*roth(i2,j1) &
                 +rr3*roth(i1,j2)+rr4*roth(i2,j2)
   rbkly(kbm,i,j)=1./(rr1/rbkmd(i1,j1)+rr2/rbkmd(i2,j1) &
                 +rr3/rbkmd(i1,j2)+rr4/rbkmd(i2,j2))
   rmuly(kbm,i,j)=1./(rr1/rmumd(i1,j1)+rr2/rmumd(i2,j1) &
                 +rr3/rmumd(i1,j2)+rr4/rmumd(i2,j2))
   qply(kbm,i,j) =1./(rr1/qpmd(i1,j1)+rr2/qpmd(i2,j1) &
                 +rr3/qpmd(i1,j2)+rr4/qpmd(i2,j2))
   qsly(kbm,i,j) =1./(rr1/qsmd(i1,j1)+rr2/qsmd(i2,j1) &
                 +rr3/qsmd(i1,j2)+rr4/qsmd(i2,j2))
 enddo
 enddo
 return
end subroutine h2d_interp
!=======================================================================
subroutine smooth(nfile,kz_sub,kbm,nx_fd,ny_fd,ztb1,ztb2)
 use vmodel_inp
 use fd_model_comm
 implicit NONE
 integer, intent(IN):: nfile,kz_sub,kbm,nx_fd,ny_fd
 real, intent(IN):: ztb1,ztb2
 integer, save:: nlsm,nsmf(4),nsmkz(4)
 integer:: i,j,k,kk,ix1,jy1,ix2,jy2,nkz,nck,icheck
 real::  zz1,zz2,sm1,sm2
 real, dimension(nx_fd, ny_fd):: vmtp
!
 if(nfile == 1 .and. kz_sub == 1) then
   nlsm=0;  zz2=top_depth(1)
   do nck=1,num_file
     nkz= nodexyz(3,nck)
     do k=1,nkz
       zz1=zz2;  zz2=zz2+gdsp_vm(3,nck)
       if(k==nkz .and. nck<num_file) zz2=top_depth(nck+1)
       if((ztb1>=zz1.and.ztb1<zz2).or.(ztb2>=zz1.and.ztb2<zz2))then
         nlsm=nlsm+1;  nsmf(nlsm)=nck;  nsmkz(nlsm)=k
         nlsm=nlsm+1;  nsmf(nlsm)=nck;  nsmkz(nlsm)=k+1
         if(k==nkz) then
           nsmf(nlsm)=nck+1;  nsmkz(nlsm)=1
           if(nck==num_file)  nlsm=nlsm-1
         endif
       endif
     enddo
   enddo
   if(nlsm > 4) then
     write(*,*) 'Error in subroutine smooth'
     stop
   endif
   do icheck=1,2
     if(nlsm>2.and.nsmf(2)==nsmf(3).and.nsmkz(2)==nsmkz(3)) then
       nsmf(3) =nsmf(nlsm);  nsmkz(3)=nsmkz(nlsm); nlsm=nlsm-1
     endif
   enddo
 endif
!
! Smoothing
 sm1=0.4;  sm2=(1.0-sm1)/8.0
 do kk=1,nlsm
   if(nsmf(kk)==nfile .and. nsmkz(kk)==kz_sub) then
     ix1=2;  jy1=2;  ix2=nx_fd-ix1+1;  jy2=ny_fd-jy1+1
     vmtp=rholy(kbm, 1:nx_fd, 1:ny_fd)
     do j=jy1,jy2
     do i=ix1,ix2
       rholy(kbm,i,j)=sm1*vmtp(i,j)+sm2*(vmtp(i-1,j-1)+ &
               vmtp(i,j-1)+vmtp(i+1,j-1)+vmtp(i-1,j)+vmtp(i+1,j)+ &
               vmtp(i-1,j+1)+vmtp(i,j+1)+vmtp(i+1,j+1))
     enddo
     enddo
     vmtp=rbkly(kbm, 1:nx_fd, 1:ny_fd)
     do j=jy1,jy2
     do i=ix1,ix2
       rbkly(kbm,i,j)=1./(sm1*vmtp(i,j)+sm2*(vmtp(i-1,j-1)+ &
               vmtp(i,j-1)+vmtp(i+1,j-1)+vmtp(i-1,j)+vmtp(i+1,j)+ &
               vmtp(i-1,j+1)+vmtp(i,j+1)+vmtp(i+1,j+1)))
     enddo
     enddo
     vmtp=rmuly(kbm, 1:nx_fd, 1:ny_fd)
     do j=jy1,jy2
     do i=ix1,ix2
       rmuly(kbm,i,j)=1./(sm1*vmtp(i,j)+sm2*(vmtp(i-1,j-1)+ &
              vmtp(i,j-1)+vmtp(i+1,j-1)+vmtp(i-1,j)+vmtp(i+1,j)+ &
              vmtp(i-1,j+1)+vmtp(i,j+1)+vmtp(i+1,j+1)))
     enddo
     enddo
     vmtp=qply(kbm, 1:nx_fd, 1:ny_fd)
     do j=jy1,jy2
     do i=ix1,ix2
       qply(kbm,i,j)=1./(sm1*vmtp(i,j)+sm2*(vmtp(i-1,j-1)+ &
              vmtp(i,j-1)+vmtp(i+1,j-1)+vmtp(i-1,j)+vmtp(i+1,j)+ &
              vmtp(i-1,j+1)+vmtp(i,j+1)+vmtp(i+1,j+1)))
     enddo
     enddo
     vmtp=qsly(kbm, 1:nx_fd, 1:ny_fd)
     do j=jy1,jy2
     do i=ix1,ix2
       qsly(kbm,i,j)=1./(sm1*vmtp(i,j)+sm2*(vmtp(i-1,j-1)+ &
              vmtp(i,j-1)+vmtp(i+1,j-1)+vmtp(i-1,j)+vmtp(i+1,j)+ &
              vmtp(i-1,j+1)+vmtp(i,j+1)+vmtp(i+1,j+1)))
     enddo
     enddo
   endif
 enddo
 return
end subroutine smooth
!********************************************************************
subroutine v1d_interp(zr2,k,kup,kbm,tstep,gstep)
 use grid_sp_num
 use fd_model_comm
 implicit NONE 
 integer, intent(IN):: k,kup,kbm
 real, intent(IN):: zr2
 real, intent(INOUT):: tstep,gstep
 integer:: ntp1,ntp2,i,j,ii,lls,iselc
 real:: zr1,ro,cbk,cmu,qp,qs,vp,vs,delt
! do interpolution between up and bottom layer
 zr1=1.-zr2
 do j=1,ny_fd
 do i=1,nx_fd
   ro=zr1*rholy(kup,i,j)+zr2*rholy(kbm,i,j)
   cbk=1.0/(zr1/rbkly(kup,i,j)+zr2/rbkly(kbm,i,j))
   cmu=1.0/(zr1/rmuly(kup,i,j)+zr2/rmuly(kbm,i,j))
   qp=1./(zr1/qply(kup,i,j)+zr2/qply(kbm,i,j))
   qs=1./(zr1/qsly(kup,i,j)+zr2/qsly(kbm,i,j))
   vs=sqrt(cmu/ro)
   vp=sqrt((cbk+4./3.*cmu)/ro)
   call search(vs, ntp1,ntp2)
   iselc=0
   do lls=ntp1,ntp2
     ii=ntyp_order(lls)
     if(abs(1.-evp(ii)/vp)<errm .and. abs(1.-evs(ii)/vs)<errm &
       .and. abs(1.-erh(ii)/ro)<errm) iselc=ii 
   enddo
   if(iselc==0) then 
     ntype=ntype+1;  iselc=ntype
     evp(ntype)=vp;  evs(ntype)=vs;  erh(ntype)=ro
     eqp(ntype)=qp;  eqs(ntype)=qs
     call re_order
   endif
   idmat((j-1)*nx_fd+i)=iselc
! checking the time step and minimum node number per wavelength
   delt=amin1(dxx(i),dyy(j),dzz(k))
   if(tstep.lt.0.0) then
     tstep=delt/evp(iselc)
   else
     tstep=amin1(tstep,delt/evp(iselc))
   endif
   delt=amax1(dxx(i),dyy(j),dzz(k))
   if(gstep.lt.0.0) then
     gstep=evs(iselc)/delt
   else
     gstep=amin1(gstep,evs(iselc)/delt)
   endif
 enddo
 enddo
 
end subroutine v1d_interp
!********************************************************************
subroutine search(vvs,ntp1,ntp2)
 use fd_model_comm
 implicit NONE
 integer, intent(out):: ntp1,ntp2
 real, intent(IN):: vvs
 integer:: nn1,nn2,nnn
 real:: vv1,vv2,vs0
!
 vv1=(1.0-errm)*vvs;    vv2=(1.0+errm)*vvs
 ntp1=1;   nn2=ntype;   nnn=(ntp1+nn2)/2
 do while(nnn > ntp1) 
   vs0=evs(ntyp_order(nnn))
   if(vv1 > vs0) then
      ntp1=nnn
   else
      nn2=nnn
   endif
   nnn=(ntp1+nn2)/2
 enddo
 nn1=ntp1;   ntp2=ntype;   nnn=(nn1+ntp2)/2
 do while(nnn > nn1) 
   vs0=evs(ntyp_order(nnn))
   if(vv2 < vs0) then
      ntp2=nnn
   else
      nn1=nnn
   endif
   nnn=(nn1+ntp2)/2
 enddo
 return
end subroutine search
!********************************************************************
subroutine re_order
 use fd_model_comm
 implicit NONE
 integer:: i,nn1,nn2,nnn
 integer, allocatable, dimension(:):: itmp
 real, allocatable, dimension(:):: rtmp
 real:: vs0,vs1
!
 ntyp_order(ntype)=ntype
 if(ntype == mtype) then
   allocate(itmp(ntype), rtmp(ntype))
   itmp=ntyp_order;  mtype=3*mtype/2
   deallocate(ntyp_order);  allocate(ntyp_order(mtype))
   ntyp_order(1:ntype)=itmp;  rtmp=evp
   deallocate(evp);  allocate(eqp(mtype))
   evp(1:ntype)=rtmp;         rtmp=evs
   deallocate(evs);  allocate(evs(mtype))
   evs(1:ntype)=rtmp;         rtmp=erh
   deallocate(erh);  allocate(erh(mtype))
   erh(1:ntype)=rtmp;         rtmp=eqp
   deallocate(eqp);  allocate(eqp(mtype))
   eqp(1:ntype)=rtmp;         rtmp=eqs
   deallocate(eqs);  allocate(eqs(mtype))
   eqs(1:ntype)=rtmp;         deallocate(itmp, rtmp)
 endif
!
 nn1=1;  nn2=ntype;  nnn=(nn1+nn2)/2;  vs1=evs(ntype)
 do 
   vs0=evs(ntyp_order(nnn))
   if(vs1 >= vs0) then
     nn1=nnn
   else
     nn2=nnn
   endif
   nnn=(nn1+nn2)/2
   if(nnn <= nn1) exit
 enddo
 do i=ntype-1,nn2,-1
   ntyp_order(i+1)=ntyp_order(i)
 enddo
 ntyp_order(nn2)=ntype
 
end subroutine re_order
!**********************************************************************
subroutine switch(ks)
 use vmodel_inp
 use grid_sp_num
 use fd_model_comm
 implicit NONE
 integer, intent(IN):: ks
 integer:: i,j,k,i3,j3
 real:: dx,vs1,vs2,vsmd,depth_bm
! Which region, I or II ?
 if(ks == 1) then
   nx_fd=(nx_fd0-1)*3+1;  ny_fd=(ny_fd0-1)*3+1; nz_fd= nz1_fd
   allocate(dxx(nx_fd+2), dyy(ny_fd+2), dzz(nz_fd))
   allocate(rbkly(2,nx_fd,ny_fd), rmuly(2,nx_fd,ny_fd))
   allocate(rholy(2,nx_fd,ny_fd), qply(2,nx_fd,ny_fd))
   allocate(qsly(2,nx_fd,ny_fd),  cefxy(2,nx_fd,ny_fd))
   allocate(idfxy(2,nx_fd,ny_fd), idmat(nx_fd*ny_fd))
   mtype=nx_fd*ny_fd;     allocate(ntyp_order(mtype))
   allocate(evp(mtype),evs(mtype),erh(mtype),eqp(mtype),eqs(mtype))
!
   do i=1,nx_fd0
     i3=(i-1)*3+1;  dxx(i3 : i3+2)=dxx0(i)/3.0
   enddo
   do j=1, ny_fd0
     j3=(j-1)*3+1;  dyy(j3 : j3+2)=dyy0(j)/3.0
   enddo
   dzz=dzz0(1:nz_fd)
 else
   deallocate(dxx, dyy, dzz)
   nx_fd=nx_fd0;  ny_fd=ny_fd0;  nz_fd=nz_fd0-nz1_fd
   allocate(dxx(nx_fd), dyy(ny_fd), dzz(nz_fd))
   dxx=dxx0(1 : nx_fd);   dyy=dyy0(1 : ny_fd)
   dzz=dzz0(nz1_fd+1 : nz_fd0)
   do j=1,ny_fd
     j3=(j-1)*3+1
     do i=1,nx_fd
       i3=(i-1)*3+1
       rbkly(:,i,j)=rbkly(:,i3,j3)
       rmuly(:,i,j)=rmuly(:,i3,j3)
       rholy(:,i,j)=rholy(:,i3,j3)
        qply(:,i,j)= qply(:,i3,j3)
        qsly(:,i,j)= qsly(:,i3,j3)
     enddo
   enddo
 endif
 
end subroutine switch
!**********************************************************************
subroutine coef_interp(anglex,angley,xmd0,ymd0)
 use vmodel_inp
 use grid_sp_num
 use fd_model_comm
 implicit NONE
 real, intent(IN):: anglex,angley,xmd0,ymd0
 integer:: i,j,ixm,jym 
 real:: dc,crx,snx,cry,sny,xf,dxf,yf,dyf,xm,ym,xr,yr
 dc=atan(1.)/45.0
 crx=cos(anglex*dc);   snx=sin(anglex*dc)
 cry=cos(angley*dc);   sny=sin(angley*dc)
 yf =-ymd0;             dyf=0.0
 do j=1,ny_fd
   yf=yf+dyf;    dyf=dyy(j)
   xf=-xmd0;     dxf=0.0
   do i=1,nx_fd
     xf=xf+dxf;   dxf=dxx(i)
     xm=xf*crx+yf*snx
     ym=xf*cry+yf*sny
     ixm=min0(max0(int(xm/dx_vm)+1, 1), nx_vm-1)
     jym=min0(max0(int(ym/dy_vm)+1, 1), ny_vm-1)
     xr=amin1(amax1(xm/dx_vm-ixm+1, 0.0),  1.0)
     yr=amin1(amax1(ym/dy_vm-jym+1, 0.0),  1.0)
     idfxy(1,i,j)=ixm;   idfxy(2,i,j)=jym
     cefxy(1,i,j)=xr;    cefxy(2,i,j)=yr
   enddo
 enddo
 return
end subroutine coef_interp
!**********************************************************************
subroutine grid_space(id_xyz,nseg,xx,dx)
 use grid_sp_num
 implicit NONE
 integer, intent(IN):: id_xyz,nseg
 real, dimension(nseg), intent(IN):: dx
 real, dimension(nseg), intent(INOUT):: xx
 real, allocatable, dimension(:):: gsp
 integer:: i,ks,nnode
 real:: dseg,slop,af1,afa,sss,x0,xi
!
 Type:: xyz_list
   real:: coor 
   Type (xyz_list), pointer:: p
 End Type xyz_list
 type(xyz_list), pointer:: xyz_head,xyz_tail,ptr
!
 allocate(xyz_head)
 xyz_tail => xyz_head
 nullify(xyz_tail%p);  xyz_tail%coor=xx(1)
 nnode=1
 do ks=2,nseg
   dseg=xx(ks)-xx(ks-1);   slop=(dx(ks)-dx(ks-1))/dseg
   af1=(1.-0.5*slop);      afa=(1.+0.5*slop)/af1
   sss=dx(ks-1)/af1;       x0=xx(ks-1);    xi=0.0
   print *, 'af1 afa sss'
   print *, af1, afa, sss
   print *, 'old xx'
   print *, xx
   do while(xi < dseg)
     xi=xi*afa+sss;          nnode=nnode+1
     allocate(xyz_tail%p);   xyz_tail => xyz_tail%p
     nullify(xyz_tail%p);    xyz_tail%coor=xi+x0
   enddo
   xx(ks)=xi + x0
   print *, 'new xx'
   print *, xx
   print *, 'nnode'
   print *, nnode
 enddo
 allocate(gsp(nnode+1))
 nnode=0;  ptr => xyz_head
 do while (associated(ptr)) 
   nnode=nnode+1;  gsp(nnode)=ptr%coor;  ptr=>ptr%p
 enddo
 do i=nnode,2,-1
   gsp(i)=gsp(i)-gsp(i-1)
 enddo
 nnode=nnode+1;  gsp(1)=gsp(2);   gsp(nnode)=gsp(nnode-1)
 select case(id_xyz)
 case (1)
   allocate(dxx0(nnode))
   nx_fd0=nnode; dxx0=gsp
 case (2)
   allocate(dyy0(nnode))
   ny_fd0=nnode; dyy0=gsp
 case (3)
   allocate(dzz0(nnode))
   nz_fd0=nnode; dzz0=gsp
 endselect

 print *, 'gsp array'
 print *, gsp
! 
 deallocate(gsp)
 do 
   if(.NOT. associated(xyz_head)) exit
   if(associated(xyz_head%p)) then
     ptr=>xyz_head%p
     deallocate(xyz_head);  xyz_head=>ptr
   else
     deallocate(xyz_head)
   endif
 enddo

end subroutine grid_space
!-----------------------------------------------------------------------
subroutine format_dt(dt)
 implicit NONE
 real, intent(INOUT):: dt
 integer:: nn
 real:: dtsetp,dd,dti
!
 dtsetp=dt
 if(dtsetp > 1.0 ) then
   nn=int(dtsetp*100.0)
   dt=nn*0.01
 else
   dd=1.0
   do 
     dd=dd*10.
     dti=dtsetp*dd
     nn=int(dti)
     if(nn >= 100)  exit
   enddo
   dt=nn/dd
 endif
 return
end subroutine format_dt
!-----------------------------------------------------------------------
subroutine smq_coef(id_smq,dt,freq_ref,qsmin,qsmax,qpmin,qpmax)
 use fd_model_comm
 implicit NONE
 integer, intent(IN):: id_smq
 real, intent(IN):: dt,freq_ref
 real, intent(OUT):: qsmin,qsmax,qpmin,qpmax
 integer:: i,j,nm
 real, dimension(nll):: qwt,rt_tmp,wt1_tmp,wt2_tmp
 real:: unrelaxed_mu,real_mu,cmur,qq,deltm
!
 qsmin=eqs(1);  qsmax=eqs(1);  qpmin=eqp(1);  qpmax=eqp(1)
 do i=2,ntype
   qsmin=amin1(qsmin,eqs(i));  qsmax=amax1(qsmax,eqs(i))
   qpmin=amin1(qpmin,eqp(i));  qpmax=amax1(qpmax,eqp(i))
 enddo
! The relaxation, weight coeficients and correccting  mudulus
 do  nm=1,ntype
   qq=eqp(nm)
   call relx_wt_coef(id_smq,qq,deltm,relaxt,qwt1,qwt2,qwt)
   eqp(nm)=deltm
   real_mu=erh(nm)*evp(nm)**2
   cmur=unrelaxed_mu(freq_ref,real_mu,ar_ha,nll,relaxt,qwt)
   evp(nm)=sqrt(cmur/erh(nm))
   qq=eqs(nm)
   call relx_wt_coef(id_smq,qq,deltm,relaxt,qwt1,qwt2,qwt)
   eqs(nm)=deltm
   real_mu=erh(nm)*evs(nm)**2
   cmur=unrelaxed_mu(freq_ref,real_mu,ar_ha,nll,relaxt,qwt)
   evs(nm)=sqrt(cmur/erh(nm))
 enddo
!
 do j=1,nll
   rt_tmp(j)=relaxt(j)
   wt1_tmp(j)=qwt1(j)
   wt2_tmp(j)=qwt2(j)
 enddo
 do i=1,nll
   epdt(i)=exp(-dt/rt_tmp(i))
   epdt(i)=(1.-0.5*dt/rt_tmp(i))/(1.+0.5*dt/rt_tmp(i))
   qwt1(i)=wt1_tmp(i)*0.5
   qwt2(i)=wt2_tmp(i)*0.5
 enddo
 
end 
!-----------------------------------------------------------------------
subroutine relx_wt_coef(id_smq,qq,deltm,relaxt,qwt1,qwt2,qwt)
 implicit NONE
 integer, intent(IN):: id_smq
 real, intent(IN):: qq
 real, intent(OUT):: deltm
 real, dimension(8), intent(OUT):: relaxt,qwt1,qwt2,qwt
!
 select case(id_smq)
 case (1)
   call rw_ha_10hz1(qq,deltm,relaxt,qwt1,qwt2,qwt)
 case (2)
   call rw_ha_10hz2(qq,deltm,relaxt,qwt1,qwt2,qwt)
 return
 case (3)
   call rw_ha_3hz1(qq,deltm,relaxt,qwt1,qwt2,qwt)
 end select
 return
end subroutine relx_wt_coef
!-----------------------------------------------------------------------
subroutine rw_ha_3hz1(qq,deltm,relaxt,qwt1,qwt2,qwt)
! f=0-3 Hz Qs are constant (>=10)
! f> 3Hz Qs like Qo*(f/3)**(-0.7)
 implicit NONE
 integer, parameter:: np=8
 real, parameter:: Qmin=5, Qmax=5000
 real, intent(IN):: qq
 real, intent(OUT):: deltm
 real, dimension(np), intent(out):: relaxt,qwt1,qwt2,qwt
 real :: qr, a1=6.348, a2=-2.166, a3=-0.724, a4=0.094
 relaxt=(/ 0.795776E-02, 0.795782E-02, 0.795786E-02, 0.954593E-01, &
          0.238935E+00, 0.102562E+01, 0.358557E+01, 0.144682E+02 /)
 qwt1= (/ 0.110478E-01,-0.720167E-02,-0.161613E-01, 0.288380E-01, &
          0.130184E-01, 0.178227E-01, 0.286709E-01, 0.164743E-01 /)
 qwt2= (/ 0.108291E+00, 0.167838E+00, 0.197520E+00, 0.489398E-01, &
          0.112535E+00, 0.107674E+00, 0.792218E-01, 0.175334E+00 /)
 qr=alog(qq/Qmin);  deltm=(a1+a2*qr*qq**a3)/(1.0+a4*qq)
 qwt=deltm*(deltm*qwt1+qwt2)
 return
end subroutine rw_ha_3hz1
!-----------------------------------------------------------------------
subroutine rw_ha_10hz2(qqi,deltm,relaxt,qwt1,qwt2,qwt)
 implicit NONE
 integer, parameter:: np=8
 real, parameter:: Qmin=5, Qmax=5000
 real, intent(IN):: qqi
 real, intent(OUT):: deltm
 real, dimension(np), intent(out):: relaxt,qwt1,qwt2,qwt
 real:: q1,q2,qq, a1=1.0, a2=-0.0044, a3=0.788927, a4=0.018050
 relaxt=(/ 0.132630E-01, 0.132631E-01, 0.641349E-01, 0.208257E+00, &
           0.678804E+00, 0.254317E+01, 0.144614E+02, 0.144684E+02 /)
 qwt1  =(/ 0.372039E+00, 0.374809E+00, 0.420333E+00, 0.450612E+00, &
           0.488465E+00, 0.505551E+00, 0.512479E+00, 0.507892E+00 /)
 qwt2  =(/ 0.101519E+00, 0.103131E+00, 0.113124E+00, 0.100308E+00, &
           0.110382E+00, 0.126741E+00, 0.982004E-01, 0.115844E+00 /)
 qq=amin1(4999.0,amax1(5.01,qqi))
 q1=alog(qq/Qmin);  q2=alog(Qmax/qq)
 deltm=(a1+a2*q1**1.28*q2**2.10)/(a3+a4*qq)
 qwt   =deltm*(deltm*qwt1+qwt2)
 return
end subroutine rw_ha_10hz2
!-----------------------------------------------------------------------
subroutine rw_ha_10hz1(qq,deltm,relaxt,qwt1,qwt2,qwt)
! f=0-10 Hz, Qs are constant
 implicit NONE
 integer, parameter:: np=8
 real, parameter:: Qmin=5, Qmax=5000
 real, intent(IN):: qq
 real, intent(OUT):: deltm
 real, dimension(np), intent(out):: relaxt,qwt1,qwt2,qwt
 real:: qr, a1= 5.649, a2=-0.477, a3=-0.530, a4= 0.109
 relaxt=(/ 0.795929E-02, 0.796067E-02, 0.328661E-01, 0.138501E+00, &
           0.541466E+00, 0.197729E+01, 0.122426E+02, 0.122427E+02 /)
 qwt1  =(/ 0.189599E-01, 0.131638E-01, 0.155379E-01, 0.157445E-01, &
           0.230073E-01, 0.209506E-01, 0.342489E-01, 0.238454E-01 /)
 qwt2  =(/ 0.850577E-01, 0.113676E+00, 0.136285E+00, 0.143302E+00, &
           0.125200E+00, 0.143108E+00, 0.104466E+00, 0.145809E+00 /)
!
 qr=alog(amin1(amax1(qq,Qmin), Qmax)/Qmin)
 deltm=(a1+a2*qr*qq**a3)/(1.0+a4*qq)
 qwt   =deltm*(deltm*qwt1+qwt2)
end subroutine rw_ha_10hz1
!-----------------------------------------------------------------------
function unrelaxed_mu(fr,rmu_vm,ar_ha,nparam,relaxt,qwt)
! ar_ha=-1., Harmonic Average of Modulus
! ar_ha=1.0, Arithmetic Average of Modulus
 implicit NONE
 integer, intent(IN):: nparam
 real, intent(IN):: fr,rmu_vm,ar_ha 
 real, dimension(nparam), intent(IN):: relaxt,qwt
 integer:: i
 complex:: cxlnf
 real:: unrelaxed_mu,ww,rmus,rtmp
 ww=fr*atan(1.0)*8.0
 cxlnf=(0.0, 0.0)
 rmus=0.0
 if(ar_ha > 0.5) then
   do i=1,nparam
     cxlnf=cxlnf+1.0-qwt(i)/cmplx(1.,relaxt(i)*ww)
   enddo
   cxlnf=cxlnf/nparam
 else
   do i=1,nparam
     cxlnf=cxlnf+1./(1.0-qwt(i)/cmplx(1.,relaxt(i)*ww))
     rmus=rmus+cabs(1.0-qwt(i)/cmplx(1.,relaxt(i)*ww))
   enddo
   cxlnf=nparam/cxlnf
 endif
 rtmp=cabs(cxlnf)
 unrelaxed_mu=rmu_vm/rtmp*(1.+real(cxlnf)/rtmp)/2.
! unrelaxed_mu=rmu_vm/rtmp
 return
end function unrelaxed_mu
!-----------------------------------------------------------------------
function cmu_freq(cmu_ref,ww,ar_ha,nparam,relaxt,qwt)
 implicit NONE
 real, parameter:: pi=3.1415926
 integer, intent(IN):: nparam
 real, intent(IN):: cmu_ref,ww,ar_ha 
 real, dimension(nparam), intent(IN):: relaxt,qwt
 integer:: i
 complex:: xlnf, cmu_freq
!
 xlnf=(0.0, 0.0)
 if(ar_ha > 0.5) then
   do i=1,nparam
     xlnf=xlnf+1.0-qwt(i)/cmplx(1.,relaxt(i)*ww)
   enddo
   xlnf=xlnf/nparam
 else
   do i=1,nparam
     xlnf=xlnf+1./(1.0-qwt(i)/cmplx(1.,relaxt(i)*ww))
   enddo
   xlnf=nparam/xlnf
 endif
 cmu_freq=cmu_ref*xlnf
 return
end function cmu_freq
!-----------------------------------------------------------------------
function smq(ww,ar_ha,nparam,relaxt,qwt)
! ar_ha=-1., Harmonic Average of Modulus
! ar_ha=1.0, Arithmetic Average of Modulus
 implicit NONE
 integer, intent(IN):: nparam
 real, intent(IN):: ww,ar_ha 
 real, dimension(nparam), intent(IN):: relaxt,qwt
 integer:: i
 real:: smq,sum1,sum2,wti,wt2,aa,bb,ab2
 sum1=0.0
 sum2=0.0
 do i=1,nparam
   wti=ww*relaxt(i)
   wt2=wti**2
   aa=1.0-qwt(i)/(1.+wt2)
   bb=wti*qwt(i)/(1.+wt2)
   ab2=1.0
   if(ar_ha <= 0.5) ab2=aa*aa+bb*bb
   sum1=sum1+aa/ab2
   sum2=sum2+bb/ab2
 enddo
 smq=sum1/sum2
 return
end function smq
