//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
//
#include<sys/time.h>
#include<stdlib.h>
void record_time(double* time) {
    struct timeval tim;
    gettimeofday(&tim, NULL);
    *time = (double)((tim.tv_sec)*1000.0 + (tim.tv_usec)/1000.0);
}
