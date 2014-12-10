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
   
    subroutine MPIX_ISEND_C (buf_id, n, datatype, dest, tag, comm, req, ierr, cmdq_id)&
        bind(C, name="MPIX_Isend_C")
        use::iso_c_binding
        integer(c_int), intent(in), value:: buf_id        
        integer(c_int), intent(in), value :: n, dest, tag
        integer(c_int), intent(in) :: comm, datatype ! need to be passed as pointers
        integer(c_int), intent(inout):: req, ierr
        integer(c_int), intent(in), value:: cmdq_id
    end subroutine MPIX_ISEND_C

    subroutine MPIX_IRECV_C (buf_id, n, datatype, src, tag, comm, req, ierr, cmdq_id)&
        bind(C, name="MPIX_Irecv_C")
        use::iso_c_binding
        integer(c_int), intent(in), value:: buf_id        
        integer(c_int), intent(in), value :: n, src, tag
        integer(c_int), intent(in) :: comm, datatype !need to be passed as pointers
        integer(c_int), intent(inout):: req, ierr
        integer(c_int), intent(in), value:: cmdq_id
    end subroutine MPIX_IRECV_C

    subroutine MPIX_IRECV_C1 (buf, n, datatype, src, tag, comm, req, ierr)&
        bind(C, name="MPIX_Irecv_C1")
        use::iso_c_binding
        type(c_ptr), intent(in), value:: buf
        integer(c_int), intent(in), value :: n, src, tag
        integer(c_int), intent(in) :: comm, datatype !need to be passed as pointers
        integer(c_int), intent(inout):: req, ierr
    end subroutine MPIX_IRECV_C1

    subroutine MPIX_SEND_C (buf_id, n, datatype, dest, tag, comm, ierr, cmdq_id)&
        bind(C, name="MPIX_Send_C")
        use::iso_c_binding
        integer(c_int), intent(in), value:: buf_id        
        integer(c_int), intent(in), value :: n, dest, tag
        integer(c_int), intent(in) :: comm, datatype ! need to be passed as pointers
        integer(c_int), intent(inout):: ierr
        integer(c_int), intent(in), value:: cmdq_id
    end subroutine MPIX_SEND_C

    subroutine MPIX_RECV_C (buf_id, n, datatype, src, tag, comm, status, ierr, cmdq_id)&
        bind(C, name="MPIX_Recv_C")
        use::iso_c_binding
        integer(c_int), intent(in), value:: buf_id        
        integer(c_int), intent(in), value :: n, src, tag
        integer(c_int), intent(in) :: comm, datatype !need to be passed as pointers
        integer(c_int), intent(inout):: status, ierr
        integer(c_int), intent(in), value:: cmdq_id
    end subroutine MPIX_RECV_C
    subroutine MPIX_RECV_C1 (buf_id, n, datatype, src, tag, comm, ierr)&
        bind(C, name="MPIX_Recv_C1")
        use::iso_c_binding
        integer(c_int), intent(in), value:: buf_id        
        integer(c_int), intent(in), value :: n, src, tag
        integer(c_int), intent(in) :: comm, datatype !need to be passed as pointers
        integer(c_int), intent(inout):: ierr
    end subroutine MPIX_RECV_C1

    subroutine MPIX_SEND_offset_C (buf_id, offset, n, datatype, dest, tag, comm, ierr, cmdq_id)&
        bind(C, name="MPIX_Send_offset_C")
        use::iso_c_binding
        integer(c_int), intent(in), value:: buf_id        
        integer(c_int), intent(in), value :: offset, n, dest, tag
        integer(c_int), intent(in) :: comm, datatype ! need to be passed as pointers
        integer(c_int), intent(inout):: ierr
        integer(c_int), intent(in), value:: cmdq_id
    end subroutine MPIX_SEND_offset_C

    subroutine MPIX_RECV_offset_C (buf_id,offset, n, datatype, src, tag, comm, status, ierr, cmdq_id)&
        bind(C, name="MPIX_Recv_offset_C")
        use::iso_c_binding
        integer(c_int), intent(in), value:: buf_id        
        integer(c_int), intent(in), value :: offset, n, src, tag
        integer(c_int), intent(in) :: comm, datatype !need to be passed as pointers
        integer(c_int), intent(inout):: status, ierr
        integer(c_int), intent(in), value:: cmdq_id
    end subroutine MPIX_RECV_offset_C

