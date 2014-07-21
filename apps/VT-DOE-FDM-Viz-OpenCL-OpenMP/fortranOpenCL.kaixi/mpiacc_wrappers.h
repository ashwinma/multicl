   subroutine MPI_Init_C (ierr)&
        bind(C, name="MPI_Init_C")
        use::iso_c_binding
        integer(c_int), intent(inout):: ierr
    end subroutine MPI_Init_C
   
