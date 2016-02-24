!
! © 2013.  Virginia Polytechnic Institute & State University
! 
! This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
!

     call compute_velocityC_opencl(nztop, & 
                            nztm1, &
                            ca, &
                            lbx, &
                            lby, &
                            nd1_vel, &
                            cptr_rho, &
                            cptr_drvh1, &
                            cptr_drti1, &
                            cptr_damp1_x, &
                            cptr_damp1_y, &
                            cptr_idmat1, &
                            cptr_dxi1, &
                            cptr_dyi1, &
                            cptr_dzi1, &
                            cptr_dxh1, &
                            cptr_dyh1, &
                            cptr_dzh1, &
                            cptr_t1xx, &
                            cptr_t1xy, &
                            cptr_t1xz, &
                            cptr_t1yy, &
                            cptr_t1yz, &
                            cptr_t1zz, &
                            cptr_v1x, &
                            cptr_v1y, &
                            cptr_v1z, &
                            cptr_v1x_px, &
                            cptr_v1y_px, &
                            cptr_v1z_px, &
                            cptr_v1x_py, &
                            cptr_v1y_py, &
                            cptr_v1z_py, &
                            nzbm1, & 
                            nd2_vel, &
                            cptr_drvh2, &
                            cptr_drti2, &
                            cptr_idmat2, &
                            cptr_damp2_x, &
                            cptr_damp2_y, &
                            cptr_damp2_z, &
                            cptr_dxi2, & 
                            cptr_dyi2, &
                            cptr_dzi2, &
                            cptr_dxh2, &
                            cptr_dyh2, &
                            cptr_dzh2, &
                            cptr_t2xx, &
                            cptr_t2xy, &
                            cptr_t2xz, &
                            cptr_t2yy, &
                            cptr_t2yz, &
                            cptr_t2zz, &
                            cptr_v2x, & 
                            cptr_v2y, &
                            cptr_v2z, &
                            cptr_v2x_px,& 
                            cptr_v2y_px,&
                            cptr_v2z_px,&
                            cptr_v2x_py,&
                            cptr_v2y_py,&
                            cptr_v2z_py,&
                            cptr_v2x_pz,&
                            cptr_v2y_pz,&
                            cptr_v2z_pz,&
                            nmat, &
                            mw1_pml1, &
                            mw2_pml1, &
                            nxtop, &
                            nytop, &
                            mw1_pml, &
                            mw2_pml, &
                            nxbtm, &
                            nybtm, &
                            nzbtm, &
							myid)
