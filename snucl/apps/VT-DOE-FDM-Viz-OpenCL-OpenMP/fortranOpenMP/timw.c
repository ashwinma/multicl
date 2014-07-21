//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This OpenMP-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
//
#include <stdio.h>
#include <sys/time.h>


#define REAL float

void wtime_(REAL *tim, int *ierr2)
{
	struct timeval time1;
	int ierr;
	REAL elap;
	ierr = gettimeofday(&time1, NULL);
	*ierr2 = ierr;

	if (ierr != 0)
	{
		printf("bad return of gettimeofday, ierr = %d\n", ierr);
	}

	tim[0] = time1.tv_sec * 1.0;
	tim[1] = time1.tv_usec * 1.0;
}

void wdiff_(float *tim, int *ierr2, float *elapsedTime)
{
	struct timeval time1;
	int ierr;
	REAL wdiff;
	ierr = gettimeofday(&time1, NULL);

	*ierr2 = ierr;

	tim[0] = time1.tv_sec * 1.0 - tim[0];
	tim[1] = time1.tv_usec * 1.0 - tim[1];

	wdiff = (tim[0] * 1000.0) + (tim[1] / 1000.0);

	*elapsedTime = wdiff;

	//return (wdiff / 1.e6);
}
