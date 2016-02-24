!================================================
! 1. Original version
! 2. Change the output format to text
!==================================================
Program read_dfd
 implicit NONE
 character(len=72), allocatable, dimension(:):: stan_name,fd_out
 integer,allocatable, dimension(:,:):: nbbf
 integer,allocatable, dimension(:,:,:):: nrxyz_all
 real, allocatable, dimension(:):: vel,xyz,xyz_all
 real, allocatable, dimension(:,:):: vel_tmp
 real:: dt,dt_new,rtmp,fc_lp,fc_hp
 integer:: id_format,isrc,kf,ks,nch1,nch2,npt,nt_new,nr3,i,j,k,i2,j2
 integer:: num_src_model,num_fout,recv_type,nr_all,nr_inp
 integer:: nblock,nrbx,nrby,nrbz,iblk,nt0,ii0,iib,iia,it,last_blk
 character (len=70):: fd_out_list,fname_vel,fout,fblock,ch1,ch3
!---------------------------------------------------------------
 open(7,file='read_dfd.inp',form='formatted',status='old')

! Enter the new time step
 read(7,*) dt_new

 !debug---------------------------------------------
 write (*, *) 'dt_new=', dt_new
 !--------------------------------------------------

! Enter the corner frequencies of high- and low-pass
 read(7,*) fc_hp, fc_lp
 !debug---------------------------------------------
 write (*, *) 'fc_hp=', fc_hp, 'fc_lp=', fc_lp
 !--------------------------------------------------

! Enter  1 for SAC;  2 TXT; 3 Binary' 
 read(7, *) id_format
 !debug---------------------------------------------
 write (*, *) 'id_format=', id_format
 !--------------------------------------------------

! Enter the name of file to store a block outputting
 read(7,'(1a)') fblock 
 !debug---------------------------------------------
 write (*, *) 'fblock=', fblock
 !--------------------------------------------------
 close (7)
!
!---------------------------------------------------------------
!
 fd_out_list='FD_output_file_list'
 open(11, file=fd_out_list, status='old', position='rewind')
 read(11,*) num_fout,num_src_model,recv_type
 !debug---------------------------------------------
 write (*, *) 'num_fout, num_src_model, recv_type=', num_fout,num_src_model,recv_type
 !--------------------------------------------------
 allocate(fd_out(num_fout))
!
 read(11,'(1a)')  ch1
 do kf=1,num_fout
   read(11, '(1a)') fd_out(kf)
   !debug---------------------------------------------
   write (*, *) 'fd_out=', fd_out(kf)
   !--------------------------------------------------
 enddo
 close(11)
!
!---------------------------------------------------------------
!
 do isrc=1,num_src_model
   call dig2ch(isrc,3,ch3)
   last_blk=0
   do kf=1,num_fout
     fname_vel=adjustl(fd_out(kf))
     nch2=len_trim(fname_vel)
     fname_vel(nch2-2:nch2)=ch3(1:3)
!
! read FD synthetics
     open(13, file=fname_vel,status='old',form='unformatted')
     read(13) recv_type,nr_all,nr_inp,npt,dt
!
     nr3=nr_inp*3
     allocate(vel_tmp(nr3,npt),xyz(nr3),vel(npt),stan_name(nr_inp))
!
     if(recv_type == 1 ) then
       do j=1,nr_inp
         read(13) stan_name(j)
       enddo
     else
       read(13) nblock
       if(kf==1) allocate(nrxyz_all(6,nblock,num_fout))
       do k=1,nblock
         read(13) nrxyz_all(1:6, k, kf)
         if(nrxyz_all(6,k,kf)>0) last_blk=max0(last_blk, k)
       enddo
       read(13) xyz
     endif
     do i=1, npt
       read(13) (vel_tmp(j,i), j=1,nr3)
     enddo
     close(13)
!
! Interpolation
     dt_new=amax1(dt_new, dt)
     nt_new=ifix((npt*dt-dt)/dt_new)+1
     if(abs(dt_new-dt)< 0.01*dt) then
       nt_new=npt
       dt_new=dt
     else
       fc_lp=amin1(fc_lp, 0.45/dt_new)
     endif
     if(kf==1) nt0=nt_new
     nt0=min0(nt0, nt_new)
!     
     do j=1, nr3-2, 3
     do k=0,2
       do i=1,npt
         vel(i)=vel_tmp(j+k,i)
       enddo
       if(fc_hp > 1./(npt*dt) .and. fc_lp < 0.5/dt_new) then
         call xapiir( vel, npt, 'BU', 0.0, 0.0, 4, &
                      'BP', fc_hp, fc_lp, dt, 2, npt) 
       else if( fc_lp < 0.5/dt_new) then
         call xapiir( vel, npt, 'BU', 0.0, 0.0, 4, &
                      'LP', fc_hp, fc_lp, dt, 2, npt) 
       else if(fc_hp > 1./(npt*dt) ) then 
         call xapiir( vel, npt, 'BU', 0.0, 0.0, 4, &
                      'HP', fc_hp, fc_lp, dt, 2, npt) 
       endif
       if(npt > nt_new) then
         call interpol(0.0,dt,npt,vel,dt_new,nt_new)
       endif
       do i=1,nt_new
         vel_tmp(j+k,i) = vel(i)
       enddo
     enddo
     enddo
!
! Outputting 
     if(recv_type == 1) then
! Opition-1 
       k=0
       do j=1, nr3-2, 3
         k=k+1
         fout=adjustl(stan_name(k))
         nch2=len_trim(fout)
         fout(nch2+1:nch2+8)='_fdX.'//ch3(1:3)
         do i=1,nt_new
           vel(i)=vel_tmp(j,i)
         enddo
         call output3(fout,id_format,nt_new,dt_new,vel)
         fout(nch2+1:nch2+8)='_fdY.'//ch3(1:3)
         do i=1,nt_new
           vel(i)=vel_tmp(j+1,i)
         enddo
         call output3(fout,id_format,nt_new,dt_new,vel)
         fout(nch2+1:nch2+8)='_fdZ.'//ch3(1:3)
         do i=1,nt_new
           vel(i)=vel_tmp(j+2,i)
         enddo
         call output3(fout,id_format,nt_new,dt_new,vel)
       enddo
     else
! Opition-2
       fname_vel(nch2-2:nch2)='tmp'
       open(9, file=fname_vel,status='replace',form='unformatted')
       write(9) recv_type,nr_all,nr_inp,nt_new,dt_new
       !debug-------------------------------------------------
       write(*, *) recv_type,nr_all,nr_inp,nt_new,dt_new
       !--------------------------------------------------
       write(9) nblock
       !debug---------------------------------------------
       write(*, *) nblock
       !--------------------------------------------------
       do k=1,nblock
         write(9) nrxyz_all(1:6, k, kf)
         !debug--------------------------------------------------
         write(*, *) nrxyz_all(1:6, k, kf)
         !--------------------------------------------------
       enddo
       write(9) xyz
       !debug---------------------------------------
       write(*,*) xyz
       !--------------------------------------------
       do i=1, nt_new
         write(9) (vel_tmp(j,i), j=1,nr3)
         !debug--------------------------------------
         write(*,*) (vel_tmp(j,i), j=1,nr3)
         !---------------------------------------------
       enddo
       close(9)
     endif
     deallocate(vel_tmp, vel, xyz, stan_name)
   enddo
   if(recv_type==1) cycle
!
! Put the opition-2 synthetics into one file
   do iblk=1,nblock
     nrbx=0;  nrby=0;  nrbz=0
     do kf=1,num_fout
       if(nrxyz_all(2,iblk,kf) >0 .and. & 
          nrxyz_all(4,iblk,kf) >0 .and. &
          nrxyz_all(6,iblk,kf) >0 ) then
         nrbx=max0(nrbx,nrxyz_all(1,iblk,kf)+nrxyz_all(2,iblk,kf)) 
         nrby=max0(nrby,nrxyz_all(3,iblk,kf)+nrxyz_all(4,iblk,kf)) 
         nrbz=nrxyz_all(6,iblk,kf)
       endif
     enddo
     nr3=3*nrbx*nrby*nrbz
     if(nr3 == 0) cycle
     allocate(xyz_all(nr3), vel_tmp(nr3,nt0))
! 
     do kf=1,num_fout
       fname_vel=adjustl(fd_out(kf))
       nch2=len_trim(fname_vel)
       fname_vel(nch2-2:nch2)='tmp'
! read FD synthetics from temp. file
       open(13, file=fname_vel,status='old',form='unformatted')
       read(13) recv_type,nr_all,nr_inp,npt,dt
!
       nr3=nr_inp*3
       allocate(xyz(nr3))
       ii0=0
       do k=1,iblk-1
         ii0=ii0+nrxyz_all(2,k,kf)*nrxyz_all(4,k,kf)*nrxyz_all(6,k,kf)
       enddo
       i2=nrxyz_all(1,iblk,kf)+nrxyz_all(2,iblk,kf)
       j2=nrxyz_all(3,iblk,kf)+nrxyz_all(4,iblk,kf)
!
       read(13) nblock
       do k=1,nblock
         read(13) nrxyz_all(1:6, k, kf)
       enddo
       read(13) xyz
!
       iib=ii0
       do k=1, nrxyz_all(6,iblk,kf)
       do j=nrxyz_all(3,iblk,kf)+1, j2 
       do i=nrxyz_all(1,iblk,kf)+1, i2 
         iib=iib+3
         iia=(((k-1)*nrby+j-1)*nrbx+i)*3
         xyz_all(iia-2:iia)= xyz(iib-2:iib)
       enddo
       enddo
       enddo 
!         
       do it=1, nt0
         read(13) xyz 
         iib=ii0
         do k=1, nrxyz_all(6,iblk,kf)
         do j=nrxyz_all(3,iblk,kf)+1, j2 
         do i=nrxyz_all(1,iblk,kf)+1, i2 
           iib=iib+3
           iia=(((k-1)*nrby+j-1)*nrbx+i)*3
           vel_tmp(iia-2:iia, it)= xyz(iib-2:iib)
         enddo
         enddo
         enddo 
       enddo
       deallocate(xyz)
       if(iblk==last_blk) then
         close(unit=13, status='delete')
       else
         close(unit=13)
       endif
     enddo
!
!  output collected data
     nr3=3*nrbx*nrby*nrbz
     call dig2ch(iblk,2,ch1)
     fout=adjustl(fblock)
     nch1=len_trim(fout)+1
     fout(nch1:nch1+10)='.blk_'//ch1(1:2)//'.'//ch3(1:3)
     open(19, file=fout,status='replace',form='unformatted')
     write(19) nrbx,nrby,nrbz,nt0,dt
     !debug----------------------------------
     write(*,*) nrbx,nrby,nrbz,nt0,dt
     !---------------------------------------
     write(19) xyz_all
     !--------------------------------------
     write(*,*) xyz_all
     !--------------------------------------
     do it=1, nt0
       write(19) vel_tmp(:, it)
       !debug----------------------------
       write(*,*) vel_tmp(:, it)
       !-----------------------------------
     enddo
     close(19)
     write(*,*) 'The output file name of block ', iblk
     write(*,'(1a)') fout
!
!
!     do it=1, nt0
!     ii0=0
!     do k=1,nrbz
!     do j=1,nrby
!     do i=1,nrbx
!       ii0=ii0+3
!       vx(i,j,k, it) = vel_tmp(ii0-2, it)
!       vy(i,j,k, it) = vel_tmp(ii0-1, it)
!       vz(i,j,k, it) = vel_tmp(ii0-0, it)
!     enddo
!     enddo
!     enddo
!
!----------------------  check  ----------------------------------
   if(iblk==3 .and.isrc==2) then
     open(22, file=fout,status='old',form='unformatted')
     read(22) nrbx,nrby,nrbz,nt0,dt
     read(22) xyz_all
     do it=1, nt0
       read(22) vel_tmp(:, it)
     enddo
     close(22)
!
     k=1;   j=3;  i=2
     iia=(((k-1)*nrby+j-1)*nrbx+i-1)*3
     write(*,*) nrbx,nrby,nrbz,nt0,dt
     write(*,*) xyz_all(iia+1:iia+3)
     call output3('gri_tmpx',id_format,nt0,dt,vel_tmp(iia+1,1:nt0))
     call output3('gri_tmpy',id_format,nt0,dt,vel_tmp(iia+2,1:nt0))
     call output3('gri_tmpz',id_format,nt0,dt,vel_tmp(iia+3,1:nt0))
   endif
!--------------------------------------------------------------------
!
     deallocate(xyz_all, vel_tmp)
   enddo
   deallocate(nrxyz_all)
 enddo
 deallocate(fd_out)

end Program read_dfd
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine output3(tmfile,id_format,nump,dt,velc)
 implicit NONE
 character(len=72):: tmfile,file_tmp
 integer, intent(IN):: nump,id_format
 real, intent(IN):: dt
 real, dimension(nump), intent(IN):: velc
 integer:: nerr,myfh,getlen,nlch,j
 integer:: initxdr,ixdrint,ixdrreal,ixdrrmat,ixdrclose
 real:: b0
!
 if(id_format == 1) then
!  SAC
   b0=0.0
   file_tmp=tmfile
   call wsac1(file_tmp,velc,nump,b0,dt,nerr)

 else if(id_format == 2) then
!  TXT
   open(19, file=tmfile, status='unknown', position='rewind')
   write(19,*) nump,dt
   write(19,925) (velc(j), j=1,nump)
   close(19)
   925  format(5e14.5)

 else if(id_format == 3) then
!  Binary
   open(19, file=tmfile,status='unknown',form='unformatted')
   write(19) nump,dt
   write(19) (velc(j), j=1,nump)
   close(19)
 else
   write(*,*) 'No Format can find for given index'
 endif  
end  subroutine output3
!
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
subroutine interpol(tbg,dt0,n0,y0,dti,ni)
!    This interpolation subroutine is based on
!    Wiggins, 1976, BSSA, Vol.66, P. 2077, 
!    and written by Pengcheng Liu
!  
!    yo   REAL ARRAY CONTAINING SEQUENCE TO BE INTERPOLATED               
!         ORIGINAL DATA DESTROYED, REPLACED BY INTERPOLATED DATA 
!    n0   NUMBER OF SAMPLES IN y0
!    dt0  ORIGINAL POINT INTERVAL OF y0 
!    dti  POINT INTERVAL OF y0 AFTER INTERPOLATION
!    ni   NUMBER OF SAMPLES of y0 AFTER INTERPOLATION

 real, parameter :: etol=1.e-22
 INTEGER, INTENT(IN) :: n0,ni
 REAL, INTENT(IN) :: tbg,dt0,dti
 REAL, DIMENSION(n0), INTENT(INOUT) :: y0
 REAL, DIMENSION(n0) :: yi,ydif
 INTEGER :: i,j,j0,j1
 REAL :: difu1,difu2,tt,rr,r2,r3,ww1,ww2
 
 do j=1,n0
   if(j==1) then
     difu1=y0(1)
     difu2=y0(2)-y0(1)
   elseif(j==n0) then
     difu1=y0(n0)-y0(n0-1)
     difu2=-y0(n0)
   else
     difu1=y0(j)-y0(j-1)
     difu2=y0(j+1)-y0(j)
   endif
   ww1=1./amax1(abs(difu1/dt0),etol)
   ww2=1./amax1(abs(difu2/dt0),etol)
   ydif(j)=(ww1*difu1+ww2*difu2)/(ww1+ww2)
 enddo

 do i=1,ni
   tt=amax1(0.0,((i-1.0)*dti+tbg)/dt0)
   j0=int(tt)+1
   if(j0>=n0) then
     j0=n0-1
     tt=real(j0)
   endif
   j1=j0+1
   rr=1-float(j0)+tt
   r2=rr*rr
   r3=r2*rr
   yi(i)=(ydif(j1)+ydif(j0)-2*(y0(j1)-y0(j0)))*r3+ &
         (3*(y0(j1)-y0(j0))-(ydif(j1)+2.*ydif(j0)))*r2+ &
          ydif(j0)*rr+y0(j0)
 enddo 

 do i=1,ni
  y0(i)=yi(i)
 enddo
END subroutine interpol

