//!XSC--------------------------------------------------------------------
#define drvh1(i, j) drvh1M[(i) - 1 + (j) * mw1_pml1]
#define drti1(i, j) drti1M[(i) - 1 + (j) * mw1_pml1]
#define drth1(i, j) drth1M[(i) - 1 + (j) * mw1_pml1]

#define damp1_x(i, j, k) damp1_xM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) - lbx0) * nytop)]
#define damp1_y(i, j, k) damp1_yM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) - lby0) * nxtop)]

#define idmat1(i, j, k) idmat1M[(i) + (nztop + 2) * ((j) - 1 + ((k) - 1) * (nxtop + 1))]

#define v1x(i, j, k) v1xM[(i) + (nztop + 2) * ((j) + 1 + (k) * (nxtop + 3))]
#define v1y(i, j, k) v1yM[(i) + (nztop + 2) * ((j) + ((k) + 1) * (nxtop + 3))]
#define v1z(i, j, k) v1zM[(i) + (nztop + 2) * ((j) + (k) * (nxtop + 3))]
//nv2x=(lbx(2) - lbx(1) + 1) * mw1_pml
#define v1x_px(i, j, k) v1x_pxM[(i) - 1 + nztop * ((j) - 1 + nv2x * ((k) - 1))]
#define v1y_px(i, j, k) v1y_pxM[(i) - 1 + nztop * ((j) - 1 + nv2x * ((k) - 1))]
#define v1z_px(i, j, k) v1z_pxM[(i) - 1 + nztop * ((j) - 1 + nv2x * ((k) - 1))]

#define v1x_py(i, j, k) v1x_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define v1y_py(i, j, k) v1y_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define v1z_py(i, j, k) v1z_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]

#define dxi1(i, j) dxi1M[((j) - 1) * 4 + (i) - 1]
#define dyi1(i, j) dyi1M[((j) - 1) * 4 + (i) - 1]
#define dzi1(i, j) dzi1M[((j) - 1) * 4 + (i) - 1]
#define dxh1(i, j) dxh1M[((j) - 1) * 4 + (i) - 1]
#define dyh1(i, j) dyh1M[((j) - 1) * 4 + (i) - 1]
#define dzh1(i, j) dzh1M[((j) - 1) * 4 + (i) - 1]

#define t1xx(i, j, k) t1xxM[(i) - 1 + nztop * ((j) + ((k) - 1) * (nxtop + 3))]
#define t1xy(i, j, k) t1xyM[(i) - 1 + nztop * ((j) + 1 + ((k) + 1) * (nxtop + 3))]
#define t1xz(i, j, k) t1xzM[(i) - 1 + (nztop + 1) * ((j) + 1 + ((k) - 1) * (nxtop + 3))]
#define t1yy(i, j, k) t1yyM[(i) - 1 + nztop * (((j) - 1) + (k) * nxtop)]
#define t1yz(i, j, k) t1yzM[(i) - 1 + (nztop + 1) * ((j) - 1 + ((k) + 1) * nxtop)]
#define t1zz(i, j, k) t1zzM[(i) - 1 + nztop * ((j) - 1 + ((k) - 1) * nxtop)]

//nti = (lbx(2) - lbx(1) + 1) * mw1_pml + lbx(2)
//nth = (lbx(2) - lbx(1) + 1) * mw1_pml + 1 - lbx(1)
#define t1xx_px(i, j, k)   t1xx_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]
#define t1xy_px(i, j, k)   t1xy_pxM[(i) - 1 + nztop * ((j) - 1 + nth * ((k) - 1))]
#define t1xz_px(i, j, k)   t1xz_pxM[(i) - 1 + (nztop + 1) * ((j) - 1 + nth * ((k) - 1))]
#define t1yy_px(i, j, k)   t1yy_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]
#define qt1xx_px(i, j, k) qt1xx_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]
#define qt1xy_px(i, j, k) qt1xy_pxM[(i) - 1 + nztop * ((j) - 1 + nth * ((k) - 1))]
#define qt1xz_px(i, j, k) qt1xz_pxM[(i) - 1 + (nztop + 1) * ((j) - 1 + nth * ((k) - 1))]
#define qt1yy_px(i, j, k) qt1yy_pxM[(i) - 1 + nztop * ((j) - 1 + nti * ((k) - 1))]

//nti = (lby(2) - lby(1) + 1) * mw1_pml + lby(2)
//nth = (lby(2) - lby(1) + 1) * mw1_pml + 1 - lby(1)
#define t1xx_py(i, j, k)   t1xx_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define t1xy_py(i, j, k)   t1xy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define t1yy_py(i, j, k)   t1yy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define t1yz_py(i, j, k)   t1yz_pyM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xx_py(i, j, k) qt1xx_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xy_py(i, j, k) qt1xy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yy_py(i, j, k) qt1yy_pyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yz_py(i, j, k) qt1yz_pyM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]

#define qt1xx(i, j, k) qt1xxM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xy(i, j, k) qt1xyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1xz(i, j, k) qt1xzM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yy(i, j, k) qt1yyM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1yz(i, j, k) qt1yzM[(i) - 1 + (nztop + 1) * ((j) - 1 + nxtop * ((k) - 1))]
#define qt1zz(i, j, k) qt1zzM[(i) - 1 + nztop * ((j) - 1 + nxtop * ((k) - 1))]

#define rho(i)     rhoM[(i) - 1]
#define clamda(i) clamdaM[(i) - 1]
#define cmu(i)     cmuM[(i) - 1]
#define epdt(i)    epdtM[(i) - 1]
#define qwp(i)     qwpM[(i) - 1]
#define qws(i)     qwsM[(i) - 1]
#define qwt1(i)    qwt1M[(i) - 1]
#define qwt2(i)    qwt2M[(i) - 1]

//for inner_II
#define drvh2(i, j) drvh2M[(i) - 1 + (j) * mw2_pml1]
#define drti2(i, j) drti2M[(i) - 1 + (j) * mw2_pml1]
#define drth2(i, j) drth2M[(i) - 1 + (j) * mw2_pml1]

#define idmat2(i, j, k) idmat2M[(i) + (nzbtm + 1) * ((j) - 1 + ((k) - 1) * (nxbtm + 1))]

#define damp2_x(i, j, k) damp2_xM[(i) - 1 + nzbtm * ((j) - 1 + ((k) - lbx0) * nybtm)]
#define damp2_y(i, j, k) damp2_yM[(i) - 1 + nzbtm * ((j) - 1 + ((k) - lby0) * nxbtm)]
#define damp2_z(i, j) 	 damp2_zM[(i) - 1 + nxbtm * ((j) - 1)]

#define dxi2(i, j) dxi2M[(i) - 1 + 4 * ((j) - 1)]
#define dyi2(i, j) dyi2M[(i) - 1 + 4 * ((j) - 1)]
#define dzi2(i, j) dzi2M[(i) - 1 + 4 * ((j) - 1)]

#define dxh2(i, j) dxh2M[(i) - 1 + 4 * ((j) - 1)]
#define dyh2(i, j) dyh2M[(i) - 1 + 4 * ((j) - 1)]
#define dzh2(i, j) dzh2M[(i) - 1 + 4 * ((j) - 1)]

#define t2xx(i, j, k) t2xxM[(i) - 1 + nzbtm * ((j) + ((k) - 1) * (nxbtm + 3))]
#define t2xy(i, j, k) t2xyM[(i) - 1 + nzbtm * ((j) + 1 + ((k) + 1) * (nxbtm + 3))]
#define t2xz(i, j, k) t2xzM[(i) + (nzbtm + 1) * ((j) + 1 + ((k) - 1) * (nxbtm + 3))]
#define t2yy(i, j, k) t2yyM[(i) - 1 + nzbtm * (((j) - 1) + (k) * nxbtm)]
#define t2yz(i, j, k) t2yzM[(i) + (nzbtm + 1) * ((j) - 1 + ((k) + 1) * nxbtm)]
#define t2zz(i, j, k) t2zzM[(i) + (nzbtm + 1) * ((j) - 1 + ((k) - 1) * nxbtm)]
#define qt2xx(i, j, k) qt2xxM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xy(i, j, k) qt2xyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xz(i, j, k) qt2xzM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yy(i, j, k) qt2yyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yz(i, j, k) qt2yzM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2zz(i, j, k) qt2zzM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

//nti = (lbx(2) - lbx(1) + 1) * mw2_pml + lbx(2)
//nth = (lbx(2) - lbx(1) + 1) * mw2_pml + 1 - lbx(1)
#define t2xx_px(i, j, k) t2xx_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]
#define t2xy_px(i, j, k) t2xy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define t2xz_px(i, j, k) t2xz_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define t2yy_px(i, j, k) t2yy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]

#define t2xx_py(i, j, k) t2xx_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2xy_py(i, j, k) t2xy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2yy_py(i, j, k) t2yy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2yz_py(i, j, k) t2yz_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

#define t2xx_pz(i, j, k) t2xx_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2xz_pz(i, j, k) t2xz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2yz_pz(i, j, k) t2yz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define t2zz_pz(i, j, k) t2zz_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]

#define qt2xx_px(i, j, k) qt2xx_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]
#define qt2xy_px(i, j, k) qt2xy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define qt2xz_px(i, j, k) qt2xz_pxM[(i) - 1 + nzbtm * ((j) - 1 + nth * ((k) - 1))]
#define qt2yy_px(i, j, k) qt2yy_pxM[(i) - 1 + nzbtm * ((j) - 1 + nti * ((k) - 1))]

#define qt2xx_py(i, j, k) qt2xx_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xy_py(i, j, k) qt2xy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yy_py(i, j, k) qt2yy_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yz_py(i, j, k) qt2yz_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

#define qt2xx_pz(i, j, k) qt2xx_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2xz_pz(i, j, k) qt2xz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2yz_pz(i, j, k) qt2yz_pzM[(i) - 1 + mw2_pml1 * ((j) - 1 + nxbtm * ((k) - 1))]
#define qt2zz_pz(i, j, k) qt2zz_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]

#define v2x(i, j, k) v2xM[(i) + (nzbtm + 1) * ((j) + 1 + (nxbtm + 3) * (k))]
#define v2y(i, j, k) v2yM[(i) + (nzbtm + 1) * ((j) + (nxbtm + 3) * ((k) + 1))]
#define v2z(i, j, k) v2zM[(i) + (nzbtm + 1) * ((j) + (nxbtm + 3) * (k))]

//nv2y = (lbx(2) - lbx(1) + 1) * mw2_pml
#define v2x_px(i, j, k) v2x_pxM[(i) - 1 + nzbtm * ((j) - 1 + nv2y * ((k) - 1))]
#define v2y_px(i, j, k) v2y_pxM[(i) - 1 + nzbtm * ((j) - 1 + nv2y * ((k) - 1))]
#define v2z_px(i, j, k) v2z_pxM[(i) - 1 + nzbtm * ((j) - 1 + nv2y * ((k) - 1))]

#define v2x_py(i, j, k) v2x_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2y_py(i, j, k) v2y_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2z_py(i, j, k) v2z_pyM[(i) - 1 + nzbtm * ((j) - 1 + nxbtm * ((k) - 1))]

#define v2x_pz(i, j, k) v2x_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2y_pz(i, j, k) v2y_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]
#define v2z_pz(i, j, k) v2z_pzM[(i) - 1 + mw2_pml * ((j) - 1 + nxbtm * ((k) - 1))]

//MPI-ACC
#define sdx51(i, j, k) sdx51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define sdx52(i, j, k) sdx52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]
#define sdx41(i, j, k) sdx41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define sdx42(i, j, k) sdx42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]

#define sdy51(i, j, k) sdy51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define sdy52(i, j, k) sdy52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]
#define sdy41(i, j, k) sdy41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define sdy42(i, j, k) sdy42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]

#define rcx51(i, j, k) rcx51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define rcx52(i, j, k) rcx52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]
#define rcx41(i, j, k) rcx41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nytop))]
#define rcx42(i, j, k) rcx42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nybtm))]

#define rcy51(i, j, k) rcy51M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define rcy52(i, j, k) rcy52M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]
#define rcy41(i, j, k) rcy41M[(i-1) + (nztop) * ((j-1)  + (k-1) * (nxtop))]
#define rcy42(i, j, k) rcy42M[(i-1) + (nzbtm) * ((j-1)  + (k-1) * (nxbtm))]

#define cix(i,j) cixM[i-1 + j*8 ]
#define ciy(i,j) ciyM[i-1 + j*8 ]
#define chx(i,j) chxM[i-1 + j*8 ]
#define chy(i,j) chyM[i-1 + j*8 ]

