!
! © 2013.  Virginia Polytechnic Institute & State University
! 
! This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
!

call allocate_gpu_memC_opencl(lbx, lby, nmat, mw1_pml1, mw2_pml1, nxtop, nytop, &
	nztop, mw1_pml, mw2_pml, nxbtm, nybtm, nzbtm, nzbm1, nll)

