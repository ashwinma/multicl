
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE comput_input_param
 implicit NONE
 save
 integer :: num_src_model,nproc_x,nproc_y,intprt,npt_fd,npt_out,&
            nr_all,ndata,num_fout
 integer :: nproc,group_id,myid
 real :: dt_fd,dt_out,tdura,angle_north_to_x,xref_fdm,yref_fdm, &
         xref_src,yref_src,afa_src
 integer, allocatable, dimension(:) :: nfiles
 real, allocatable, dimension(:) :: syn_dti
 character(len=72) :: fd_out_root,fd_out_file
 character(len=72), dimension(200) :: source_name_list
END MODULE comput_input_param
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE station_comm
 implicit NONE
 save
 integer:: nrecs,recv_type,nblock
 character(len=72), allocatable, dimension(:)::fname_stn
 integer, allocatable, dimension(:,:):: ipox,ipoy,ipoz,nrxyz
 real, allocatable, dimension(:,:):: xcor,ycor,zcor
END MODULE station_comm
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE boundMpi_comm
 implicit NONE
 save
 integer:: nprc1,np_x,np_y,myid_x,myid_y,nx2_all,ny2_all
 integer, allocatable, dimension(:):: npat_x,npat_y
 real, allocatable, dimension(:):: bound_x,bound_y,dx_all,dy_all,dz_all
END MODULE boundMpi_comm
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE plane_wave_bd 
 implicit NONE
 save
 real, dimension(5):: xplwv,yplwv
 real, dimension(2):: zplwv
END MODULE plane_wave_bd 
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE itface_comm
 use, intrinsic :: iso_c_binding, ONLY: C_FLOAT, C_PTR
 implicit NONE
 save
 integer:: nyvx,nyvx1,nxvy,nxvy1,order_imsr,pid_ims,pid_imr
!
! DRHO WARNING: Using the C pointer approach with these variables and MPI_alloc_mem
! causes seg faults in ISEND_IRECV_Velocity. 
!
! These are the only allocatable variables used with MPI_ISEND and MPI_IREC (non-blocking
! send and receive calls), so that may be the reason.
!
! Have to leave these as Fortran allocatable for now
!
real, allocatable, dimension(:):: sdy1,sdy2,sdx1,sdx2,rcy1,rcy2,rcx1,rcx2
! real(c_float), pointer, dimension(:):: sdy1,sdy2,sdx1,sdx2,rcy1,rcy2,rcx1,rcx2
! type (c_ptr), target :: cptr_sdy1,cptr_sdy2,cptr_sdx1,cptr_sdx2,cptr_rcy1, &
!      cptr_rcy2,cptr_rcx1,cptr_rcx2
!
! DRHO WARNING: Using the C pointer approach with these variables and MPI_alloc_mem
! causes seg faults in ISEND_IRECV_Velocity
!
! Have to leave these as Fortran allocatable for now
!
 real, allocatable, dimension(:,:):: chx,chy,cix,ciy
! real(c_float), pointer, dimension(:,:):: chx,chy,cix,ciy
! real(c_float), pointer, dimension(:,:):: chx_init,chy_init,cix_init,ciy_init
! type (c_ptr), target :: cptr_chx,cptr_chy,cptr_cix,cptr_ciy
END MODULE itface_comm

MODULE remapping_interface
  interface
     subroutine assign1d_real( old_pointer, lb1, new_pointer )
       implicit none
       integer, intent(in) :: lb1
       real, pointer, intent(inout):: new_pointer(:)
       real, intent(in), target :: old_pointer(lb1:)
     end subroutine assign1d_real
     subroutine assign2d_real( old_pointer, lb1, lb2, new_pointer )
       implicit none
       integer, intent(in) :: lb1, lb2
       real, pointer, intent(inout):: new_pointer(:,:)
       real, intent(in), target :: old_pointer(lb1:,lb2:)
     end subroutine assign2d_real
     subroutine assign3d_real( old_pointer, lb1, lb2, lb3, new_pointer )
       implicit none
       integer, intent(in) :: lb1, lb2, lb3
       real, pointer, intent(inout):: new_pointer(:,:,:)
       real, intent(in), target :: old_pointer(lb1:,lb2:,lb3:)
     end subroutine assign3d_real
     subroutine assign1d_integer( old_pointer, lb1, new_pointer )
       implicit none
       integer, intent(in) :: lb1
       integer, pointer, intent(inout):: new_pointer(:)
       integer, intent(in), target :: old_pointer(lb1:)
     end subroutine assign1d_integer
     subroutine assign2d_integer( old_pointer, lb1, lb2, new_pointer )
       implicit none
       integer, intent(in) :: lb1, lb2
       integer, pointer, intent(inout):: new_pointer(:,:)
       integer, intent(in), target :: old_pointer(lb1:,lb2:)
     end subroutine assign2d_integer
     subroutine assign3d_integer( old_pointer, lb1, lb2, lb3, new_pointer )
       implicit none
       integer, intent(in) :: lb1, lb2, lb3
       integer, pointer, intent(inout):: new_pointer(:,:,:)
       integer, intent(in), target :: old_pointer(lb1:,lb2:,lb3:)
     end subroutine assign3d_integer
  end interface
END MODULE remapping_interface
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE pointer_remapping
contains
  subroutine assign1d_real( old_pointer, lb1, new_pointer )
    implicit none
    integer, intent(in) :: lb1
    real, pointer, intent(inout):: new_pointer(:)
    real, intent(in), target :: old_pointer(lb1:)
    new_pointer => old_pointer
  end subroutine assign1d_real
  subroutine assign2d_real( old_pointer, lb1, lb2, new_pointer )
    implicit none
    integer, intent(in) :: lb1, lb2
    real, pointer, intent(inout):: new_pointer(:,:)
    real, intent(in), target :: old_pointer(lb1:,lb2:)
    new_pointer => old_pointer
  end subroutine assign2d_real
  subroutine assign3d_real( old_pointer, lb1, lb2, lb3, new_pointer )
    implicit none
    integer, intent(in) :: lb1, lb2, lb3
    real, pointer, intent(inout):: new_pointer(:,:,:)
    real, intent(in), target :: old_pointer(lb1:,lb2:,lb3:)
    new_pointer => old_pointer
  end subroutine assign3d_real
  subroutine assign1d_integer( old_pointer, lb1, new_pointer )
    implicit none
    integer, intent(in) :: lb1
    integer, pointer, intent(inout):: new_pointer(:)
    integer, intent(in), target :: old_pointer(lb1:)
    new_pointer => old_pointer
  end subroutine assign1d_integer
  subroutine assign2d_integer( old_pointer, lb1, lb2, new_pointer )
    implicit none
    integer, intent(in) :: lb1, lb2
    integer, pointer, intent(inout):: new_pointer(:,:)
    integer, intent(in), target :: old_pointer(lb1:,lb2:)
    new_pointer => old_pointer
  end subroutine assign2d_integer
  subroutine assign3d_integer( old_pointer, lb1, lb2, lb3, new_pointer )
    implicit none
    integer, intent(in) :: lb1, lb2, lb3
    integer, pointer, intent(inout):: new_pointer(:,:,:)
    integer, intent(in), target :: old_pointer(lb1:,lb2:,lb3:)
    new_pointer => old_pointer
  end subroutine assign3d_integer
END MODULE pointer_remapping

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE grid_node_comm
 use, intrinsic :: iso_c_binding, ONLY: C_INT, C_FLOAT, C_PTR
 implicit NONE
 save
 integer, parameter:: mrgsp=3, msh1=(mrgsp-1)/2+1
! real, parameter:: ca=9.0/8.0, cb=-1./24.0
 real(c_float), parameter :: ca=9.0/8.0, cb=-1./24.0
! PML
! integer:: mw2_pml,mw2_pml1,mw1_pml,mw1_pml1
 integer(c_int) :: mw2_pml,mw2_pml1,mw1_pml,mw1_pml1
 real:: RF
! Material
 integer:: nmat, nll
! integer, allocatable, dimension(:,:,:):: idmat1,idmat2
 integer(c_int), pointer, dimension(:,:,:):: idmat1, idmat1_init
! integer, allocatable, dimension(:,:,:):: idmat2
 integer(c_int), pointer, dimension(:,:,:):: idmat2, idmat2_init
 type (c_ptr), target :: cptr_idmat1, cptr_idmat2
! real, allocatable, dimension(:):: clamda,cmu,rho,qwp,qws,epdt,qwt1,qwt2
 real(c_float), pointer, dimension(:) :: rho,clamda,cmu,qwp,qws,epdt,qwt1,qwt2
 type (c_ptr), target :: cptr_clamda,cptr_cmu,cptr_rho,cptr_qwp,cptr_qws,cptr_epdt,cptr_qwt1,cptr_qwt2
!
! Grid nodes
! integer:: nxbtm,nybtm,nzbtm,nxtop,nytop,nztop,nztol
! integer:: nxtm1,nytm1,nztm1,nxbm1,nybm1,nzbm1,nx1p1,nx1p2,nz1p1, &
!           ny1p1,ny1p2,nx2p1,nx2p2,ny2p1,ny2p2,nxb1,nyb1,nxb2,nyb2
! integer, dimension(2):: lbx,lby
 integer, dimension(4):: nzrg1,neighb
 integer(c_int), target :: nxbtm,nybtm,nzbtm,nxtop,nytop,nztop,nztol
 integer(c_int), target :: nxtm1,nytm1,nztm1,nxbm1,nybm1,nzbm1,nx1p1,nx1p2,nz1p1, &
           ny1p1,ny1p2,nx2p1,nx2p2,ny2p1,ny2p2,nxb1,nyb1,nxb2,nyb2
 integer(c_int), target, dimension(2):: lbx,lby
! integer, dimension(18)::nd1_vel,nd1_tyy,nd1_txy,nd1_txz,nd1_tyz, &
!                         nd2_vel,nd2_tyy,nd2_txy,nd2_txz,nd2_tyz
 integer(c_int), target, dimension(18)::nd1_vel,nd1_tyy,nd1_txy,nd1_txz,nd1_tyz, &
                         nd2_vel,nd2_tyy,nd2_txy,nd2_txz,nd2_tyz
 type (c_ptr) :: cptr_nd1_vel,cptr_nd1_tyy,cptr_nd1_txy,cptr_nd1_txz,cptr_nd1_tyz, &
                         cptr_nd2_vel,cptr_nd2_tyy,cptr_nd2_txy,cptr_nd2_txz,cptr_nd2_tyz
! real, allocatable, dimension(:):: gridx1,gridy1,gridx2,gridy2,gridz
 real(c_float), pointer, dimension(:):: gridx1,gridy1,gridx2,gridy2,gridz
 real(c_float), pointer, dimension(:):: gridx1_init,gridy1_init,gridx2_init,gridy2_init,gridz_init
 type (c_ptr), target :: cptr_gridx1,cptr_gridy1,cptr_gridx2,cptr_gridy2,cptr_gridz
! real, allocatable, dimension(:,:):: dxh1,dxi1,dyh1,dyi1,dzh1,dzi1
 real(c_float), pointer, dimension(:,:) :: dxh1,dxi1,dyh1,dyi1,dzh1,dzi1
 type (c_ptr), target :: cptr_dxh1,cptr_dxi1,cptr_dyh1,cptr_dyi1,cptr_dzh1,cptr_dzi1
! real, allocatable, dimension(:,:):: dxh2,dxi2,dyh2,dyi2,dzh2,dzi2
 real(c_float), pointer, dimension(:,:):: dxh2,dxi2,dyh2,dyi2,dzh2,dzi2
 type (c_ptr), target :: cptr_dxh2,cptr_dxi2,cptr_dyh2,cptr_dyi2,cptr_dzh2,cptr_dzi2
! ABC
 integer, dimension(16):: nxyabc
! real, allocatable, dimension(:,:,:):: abcx1,abcy1,abcx2,abcy2,abcz
 real(c_float), pointer, dimension(:,:,:):: abcx1,abcy1,abcx2,abcy2,abcz
 real(c_float), pointer, dimension(:,:,:):: abcx1_init,abcy1_init,abcx2_init,abcy2_init
 type (c_ptr), target :: cptr_abcx1,cptr_abcy1,cptr_abcx2,cptr_abcy2,cptr_abcz
END MODULE grid_node_comm
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE source_parm_comm
 implicit NONE
 save
 integer:: nfadd,id_sf_type
 integer, allocatable, dimension(:,:,:):: index_xyz_source
 real, allocatable, dimension(:,:):: famp
 real, allocatable, dimension(:):: ruptm,riset,sparam2
END MODULE source_parm_comm
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE wave_field_comm
 use mpi
 use, intrinsic :: iso_c_binding, ONLY: C_INT, C_FLOAT, C_PTR
 implicit NONE
 save
! PML
! real, allocatable, dimension(:):: drti1,drth1,drti2,drth2,drvh1,drvh2
! real, allocatable, dimension(:,:):: drti1,drth1,drti2,drth2,drvh1,drvh2
 real(c_float), pointer,  dimension(:,:):: drti1,drth1,drti2,drth2,drvh1,drvh2
 real(c_float), pointer,  dimension(:,:):: drti1_init,drth1_init,drti2_init, &
      drth2_init,drvh1_init,drvh2_init
 type (c_ptr), target :: cptr_drti1,cptr_drth1,cptr_drti2,cptr_drth2,cptr_drvh1,cptr_drvh2

! real, allocatable, dimension(:,:):: damp2_z
 real(c_float), pointer,  dimension(:,:):: damp2_z
 type (c_ptr), target :: cptr_damp2_z

! real, allocatable, dimension(:,:,:):: damp1_x,damp1_y,damp2_x,damp2_y
 real(c_float), pointer, dimension(:,:,:):: damp1_x,damp1_y,damp2_x,damp2_y
 real(c_float), pointer, dimension(:,:,:):: damp1_x_init,damp1_y_init, &
      damp2_x_init,damp2_y_init
 type (c_ptr), target :: cptr_damp1_x,cptr_damp1_y,cptr_damp2_x,cptr_damp2_y

! real, allocatable, dimension(:,:,:):: v1x_px,v1y_px,v1z_px, &
!                                       v2x_px,v2y_px,v2z_px
 real(c_float), pointer, dimension(:,:,:):: v1x_px,v1y_px,v1z_px, &
                                       v2x_px,v2y_px,v2z_px
 type (c_ptr), target :: cptr_v1x_px,cptr_v1y_px,cptr_v1z_px, &
      cptr_v2x_px,cptr_v2y_px,cptr_v2z_px

! real, allocatable, dimension(:,:,:):: v1x_py,v1y_py,v1z_py, &
!                                       v2x_py,v2y_py,v2z_py
 real(c_float), pointer,  dimension(:,:,:):: v1x_py,v1y_py,v1z_py, &
                                       v2x_py,v2y_py,v2z_py
 type (c_ptr), target :: cptr_v1x_py,cptr_v1y_py,cptr_v1z_py, &
      cptr_v2x_py,cptr_v2y_py,cptr_v2z_py

! real, allocatable, dimension(:,:,:):: v2x_pz,v2y_pz,v2z_pz
 real(c_float), pointer, dimension(:,:,:):: v2x_pz,v2y_pz,v2z_pz
 type (c_ptr), target :: cptr_v2x_pz,cptr_v2y_pz,cptr_v2z_pz

! real, allocatable, dimension(:,:,:):: t1xx_px,t1yy_px,t1xy_px,t1xz_px, &
!                                       t2xx_px,t2yy_px,t2xy_px,t2xz_px
 real(c_float), pointer, dimension(:,:,:):: t1xx_px,t1yy_px,t1xy_px,t1xz_px, &
                                       t2xx_px,t2yy_px,t2xy_px,t2xz_px
 type (c_ptr), target :: cptr_t1xx_px,cptr_t1yy_px,cptr_t1xy_px,cptr_t1xz_px, &
      cptr_t2xx_px,cptr_t2yy_px,cptr_t2xy_px,cptr_t2xz_px

! real, allocatable, dimension(:,:,:):: t1xx_py,t1yy_py,t1xy_py,t1yz_py, &
!                                       t2xx_py,t2yy_py,t2xy_py,t2yz_py
 real(c_float), pointer, dimension(:,:,:):: t1xx_py,t1yy_py,t1xy_py,t1yz_py, &
                                       t2xx_py,t2yy_py,t2xy_py,t2yz_py
 type (c_ptr), target :: cptr_t1xx_py,cptr_t1yy_py,cptr_t1xy_py,cptr_t1yz_py, &
      cptr_t2xx_py,cptr_t2yy_py,cptr_t2xy_py,cptr_t2yz_py

! real, allocatable, dimension(:,:,:):: t2xx_pz,t2zz_pz,t2xz_pz,t2yz_pz
 real(c_float), pointer, dimension(:,:,:):: t2xx_pz,t2zz_pz,t2xz_pz,t2yz_pz
 type (c_ptr), target :: cptr_t2xx_pz,cptr_t2zz_pz,cptr_t2xz_pz,cptr_t2yz_pz

! real, allocatable, dimension(:,:,:):: qt1xx_px,qt1yy_px,qt1xy_px,qt1xz_px, &
!                                       qt2xx_px,qt2yy_px,qt2xy_px,qt2xz_px
 real(c_float), pointer, dimension(:,:,:):: qt1xx_px,qt1yy_px,qt1xy_px,qt1xz_px, &
                                       qt2xx_px,qt2yy_px,qt2xy_px,qt2xz_px
 type (c_ptr), target :: cptr_qt1xx_px,cptr_qt1yy_px,cptr_qt1xy_px,cptr_qt1xz_px, &
                         cptr_qt2xx_px,cptr_qt2yy_px,cptr_qt2xy_px,cptr_qt2xz_px

! real, allocatable, dimension(:,:,:):: qt1xx_py,qt1yy_py,qt1xy_py,qt1yz_py, &
!                                       qt2xx_py,qt2yy_py,qt2xy_py,qt2yz_py
 real(c_float), pointer, dimension(:,:,:):: qt1xx_py,qt1yy_py,qt1xy_py,qt1yz_py, &
                                       qt2xx_py,qt2yy_py,qt2xy_py,qt2yz_py
 type (c_ptr), target :: cptr_qt1xx_py,cptr_qt1yy_py,cptr_qt1xy_py,cptr_qt1yz_py, &
      cptr_qt2xx_py,cptr_qt2yy_py,cptr_qt2xy_py,cptr_qt2yz_py

! real, allocatable, dimension(:,:,:):: qt2xx_pz,qt2zz_pz,qt2xz_pz,qt2yz_pz
 real(c_float), pointer, dimension(:,:,:):: qt2xx_pz,qt2zz_pz,qt2xz_pz,qt2yz_pz
 type (c_ptr), target :: cptr_qt2xx_pz,cptr_qt2zz_pz,cptr_qt2xz_pz,cptr_qt2yz_pz

! waves in Region I
! real, allocatable, dimension(:,:,:):: v1x,v1y,v1z
 real(c_float), pointer,  dimension(:,:,:) :: v1x,v1y,v1z, v1x_init,v1y_init,v1z_init
 type (c_ptr), target :: cptr_v1x,cptr_v1y,cptr_v1z
! real, allocatable, dimension(:,:,:):: t1xx,t1xy,t1xz,t1yy,t1yz,t1zz
 real(c_float), pointer,  dimension(:,:,:) :: t1xx,t1xy,t1xz,t1yy,t1yz,t1zz
 real(c_float), pointer,  dimension(:,:,:) :: t1xx_init,t1xy_init,t1xz_init, &
      t1yy_init,t1yz_init
 type (c_ptr), target :: cptr_t1xx,cptr_t1xy,cptr_t1xz,cptr_t1yy,cptr_t1yz,cptr_t1zz

! real, allocatable, dimension(:,:,:):: qt1xx,qt1xy,qt1xz,qt1yy,qt1yz,qt1zz
 real(c_float), pointer, dimension(:,:,:):: qt1xx,qt1xy,qt1xz,qt1yy,qt1yz,qt1zz
 type (c_ptr), target :: cptr_qt1xx,cptr_qt1xy,cptr_qt1xz,cptr_qt1yy,cptr_qt1yz,cptr_qt1zz

! waves in Region II
! real, allocatable, dimension(:,:,:):: v2x,v2y,v2z
 real(c_float), pointer, dimension(:,:,:):: v2x,v2y,v2z,v2x_init,v2y_init,v2z_init
 type (c_ptr), target :: cptr_v2x,cptr_v2y,cptr_v2z

! real, allocatable, dimension(:,:,:):: t2xx,t2xy,t2xz,t2yy,t2yz,t2zz
 real(c_float), pointer, dimension(:,:,:):: t2xx,t2xy,t2xz,t2yy,t2yz,t2zz
 real(c_float), pointer, dimension(:,:,:):: t2xx_init,t2xy_init,t2xz_init, &
                                            t2yy_init,t2yz_init,t2zz_init
 type (c_ptr), target :: cptr_t2xx,cptr_t2xy,cptr_t2xz,cptr_t2yy,cptr_t2yz,cptr_t2zz

! real, allocatable, dimension(:,:,:):: qt2xx,qt2xy,qt2xz,qt2yy,qt2yz,qt2zz
 real(c_float), pointer, dimension(:,:,:):: qt2xx,qt2xy,qt2xz,qt2yy,qt2yz,qt2zz
 type (c_ptr), target :: cptr_qt2xx,cptr_qt2xy,cptr_qt2xz,cptr_qt2yy,cptr_qt2yz,cptr_qt2zz
! MPI
! real, allocatable, dimension(:,:,:):: sdx51,sdx41,sdy51,sdy41,rcx51,rcx41, &
!                                       rcy51,rcy41,sdx52,sdx42,sdy52,sdy42, &
!                                       rcx52,rcx42,rcy52,rcy42
 real(c_float), pointer, dimension(:,:,:):: sdx51,sdx41,sdy51,sdy41,rcx51,rcx41, &
                                       rcy51,rcy41,sdx52,sdx42,sdy52,sdy42, &
                                       rcx52,rcx42,rcy52,rcy42
 type (c_ptr), target :: cptr_sdx51,cptr_sdx41,cptr_sdy51,cptr_sdy41,cptr_rcx51,cptr_rcx41, &
                                       cptr_rcy51,cptr_rcy41,cptr_sdx52,cptr_sdx42,cptr_sdy52, &
                                       cptr_sdy42,cptr_rcx52,cptr_rcx42,cptr_rcy52,cptr_rcy42
END MODULE wave_field_comm
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MODULE random_initial
 implicit NONE
 contains 
!-----------------------------------------------------------------------
  function ran1(idum)
! generate a uniformly distributed random number on the interval [0,1].
   implicit NONE
   integer, intent(INOUT):: idum
   integer, parameter:: im=139968,ia=3877,ic=29573
   real:: ran1
!
   idum=mod(idum*ia+ic,im)
   ran1=float(idum)/float(im)
   return
  end function ran1
!-----------------------------------------------------------------------
  subroutine assign_rand_ini(iseed,coef,arr)
   implicit NONE
   integer, intent(INOUT):: iseed
   real, intent(IN):: coef
   real, dimension(:,:,:):: arr
   integer:: i,j,k,nl(3),nu(3)
   nl=lbound(arr)
   nu=ubound(arr)
   do j=nl(3),nu(3)
   do i=nl(2),nu(2)
   do k=nl(1),nu(1)
      arr(k,i,j) = coef*(ran1(iseed)-0.5)
     !arr(k,i,j) = (ran1(iseed))
   enddo
   enddo
   enddo
   return
  end subroutine assign_rand_ini
END MODULE random_initial

! 
! A timer which calls C gettimeofday function
! and stores current time (from epoch) in 'tim'
module ctimer
  interface 
    subroutine record_time (tim) bind(C, name="record_time")
      use iso_c_binding
      real(c_double), intent(out)::tim
    end subroutine
  end interface
end module ctimer
