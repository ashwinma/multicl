!
! © 2013.  Virginia Polytechnic Institute & State University
! 
! This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
!
   subroutine MPI_Init_C (ierr)&
        bind(C, name="MPI_Init_C")
        use::iso_c_binding
        integer(c_int), intent(inout):: ierr
    end subroutine MPI_Init_C
   
