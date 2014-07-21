!
! © 2013.  Virginia Polytechnic Institute & State University
! 
! This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
!

call cpy_d2h_velocityOutputsC(lbx, &
                          lby, &
                          cptr_v1x, &
                          cptr_v1y, &
                          cptr_v1z, &
                          cptr_v1x_px, &
                          cptr_v1y_px, &
                          cptr_v1z_px, &
                          cptr_v1x_py, &
                          cptr_v1y_py, &
                          cptr_v1z_py, &
                          cptr_v2x, &
                          cptr_v2y, &
                          cptr_v2z, &
                          cptr_v2x_px, &
                          cptr_v2y_px, &
                          cptr_v2z_px, &
                          cptr_v2x_py, &
                          cptr_v2y_py, &
                          cptr_v2z_py, &
                          cptr_v2x_pz, &
                          cptr_v2y_pz, &
                          cptr_v2z_pz, &
                          mw1_pml1, &
                          mw2_pml1, &
                          nxtop, &
                          nytop, &
                          nztop, &
                          mw1_pml, &
                          mw2_pml, &
                          nxbtm, &
                          nybtm, &
                          nzbtm)

