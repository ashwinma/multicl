!    call compute_stressC(nxb1, nyb1, nx1p1, ny1p1, nxtop, nytop, nztop, mw1_pml, mw1_pml1, nmat, nll, &
!                   lbx, lby, nd1_txy, nd1_txz, nd1_tyy, nd1_tyz, cptr_idmat1, ca, cptr_drti1, &
!                   cptr_drth1, cptr_damp1_x, cptr_damp1_y, cptr_clamda, cptr_cmu, cptr_epdt, &
!                   cptr_qwp, cptr_qws, cptr_qwt1, cptr_qwt2, cptr_dxh1, cptr_dyh1, cptr_dzh1, &
!                   cptr_dxi1, cptr_dyi1, cptr_dzi1, cptr_t1xx, cptr_t1xy, cptr_t1xz, cptr_t1yy, &
!                   cptr_t1yz, cptr_t1zz, cptr_qt1xx, cptr_qt1xy, cptr_qt1xz, cptr_qt1yy, cptr_qt1yz, &
!                   cptr_qt1zz, cptr_t1xx_px, cptr_t1xy_px, cptr_t1xz_px, cptr_t1yy_px, cptr_qt1xx_px, &
!                   cptr_qt1xy_px, cptr_qt1xz_px, cptr_qt1yy_px, cptr_t1xx_py, cptr_t1xy_py, cptr_t1yy_py, &
!                   cptr_t1yz_py, cptr_qt1xx_py, cptr_qt1xy_py, cptr_qt1yy_py, cptr_qt1yz_py, cptr_v1x, &
!                   cptr_v1y, cptr_v1z,&
!                   nxb2, nyb2, nxbtm, nybtm, nzbtm, mw2_pml, mw2_pml1, nd2_txy, nd2_txz, nd2_tyy, nd2_tyz, &
!                   cptr_idmat2, cptr_drti2, cptr_drth2, cptr_damp2_x, cptr_damp2_y, cptr_damp2_z, &
!                   cptr_t2xx, cptr_t2xy, cptr_t2xz, cptr_t2yy, cptr_t2yz, cptr_t2zz, cptr_qt2xx, cptr_qt2xy, cptr_qt2xz, cptr_qt2yy, &
!                   cptr_qt2yz, cptr_qt2zz, cptr_dxh2, cptr_dyh2, cptr_dzh2, cptr_dxi2, cptr_dyi2, cptr_dzi2, cptr_t2xx_px, &
!                   cptr_t2xy_px, cptr_t2xz_px, cptr_t2yy_px, cptr_t2xx_py, cptr_t2xy_py, cptr_t2yy_py, cptr_t2yz_py, &
!                   cptr_t2xx_pz, cptr_t2xz_pz, cptr_t2yz_pz, cptr_t2zz_pz, cptr_qt2xx_px, cptr_qt2xy_px, cptr_qt2xz_px, &
!                   cptr_qt2yy_px, cptr_qt2xx_py, cptr_qt2xy_py, cptr_qt2yy_py, cptr_qt2yz_py, cptr_qt2xx_pz, cptr_qt2xz_pz, &
!                   cptr_qt2yz_pz, cptr_qt2zz_pz, cptr_v2x, cptr_v2y, cptr_v2z, myid)

    call compute_stressCm(nxb1, nyb1, nx1p1, ny1p1, nxtop, nytop, nztop, mw1_pml, mw1_pml1, &
                   lbx, lby, nd1_txy, nd1_txz, nd1_tyy, nd1_tyz, cptr_idmat1, ca, cptr_drti1, &
                   cptr_drth1, cptr_damp1_x, cptr_damp1_y, cptr_clamda, cptr_cmu, cptr_epdt, &
                   cptr_qwp, cptr_qws, cptr_qwt1, cptr_qwt2, cptr_dxh1, cptr_dyh1, cptr_dzh1, &
                   cptr_dxi1, cptr_dyi1, cptr_dzi1, cptr_t1xx, cptr_t1xy, cptr_t1xz, cptr_t1yy, &
                   cptr_t1yz, cptr_t1zz, cptr_qt1xx, cptr_qt1xy, cptr_qt1xz, cptr_qt1yy, cptr_qt1yz, &
                   cptr_qt1zz, cptr_t1xx_px, cptr_t1xy_px, cptr_t1xz_px, cptr_t1yy_px, cptr_qt1xx_px, &
                   cptr_qt1xy_px, cptr_qt1xz_px, cptr_qt1yy_px, cptr_t1xx_py, cptr_t1xy_py, cptr_t1yy_py, &
                   cptr_t1yz_py, cptr_qt1xx_py, cptr_qt1xy_py, cptr_qt1yy_py, cptr_qt1yz_py, cptr_v1x, &
                   cptr_v1y, cptr_v1z,&
                   nxb2, nyb2, nxbtm, nybtm, nzbtm, mw2_pml, mw2_pml1, nd2_txy, nd2_txz, nd2_tyy, nd2_tyz, &
                   cptr_idmat2, cptr_drti2, cptr_drth2, cptr_damp2_x, cptr_damp2_y, cptr_damp2_z, &
                   cptr_t2xx, cptr_t2xy, cptr_t2xz, cptr_t2yy, cptr_t2yz, cptr_t2zz, &
				   cptr_qt2xx, cptr_qt2xy, cptr_qt2xz, cptr_qt2yy, &
                   cptr_qt2yz, cptr_qt2zz, cptr_dxh2, cptr_dyh2, cptr_dzh2, cptr_dxi2, cptr_dyi2, cptr_dzi2, cptr_t2xx_px, &
                   cptr_t2xy_px, cptr_t2xz_px, cptr_t2yy_px, cptr_t2xx_py, cptr_t2xy_py, cptr_t2yy_py, cptr_t2yz_py, &
                   cptr_t2xx_pz, cptr_t2xz_pz, cptr_t2yz_pz, cptr_t2zz_pz, cptr_qt2xx_px, cptr_qt2xy_px, cptr_qt2xz_px, &
                   cptr_qt2yy_px, cptr_qt2xx_py, cptr_qt2xy_py, cptr_qt2yy_py, cptr_qt2yz_py, cptr_qt2xx_pz, cptr_qt2xz_pz, &
                   cptr_qt2yz_pz, cptr_qt2zz_pz, cptr_v2x, cptr_v2y, cptr_v2z, myid)

