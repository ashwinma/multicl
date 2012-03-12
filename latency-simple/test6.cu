#include <iostream>
# include <stdio.h>
# include "repeat.h"

#define TYPE float
#define INSTR +
#define INSTR2 *

__global__ void add (TYPE* a, TYPE* b, TYPE *c, clock_t* t_start, clock_t* t0, clock_t* t1, clock_t* t2, clock_t* t3, TYPE operand, TYPE operand2){
	volatile clock_t start = clock();
	TYPE op = operand;
	TYPE op2 = operand2;
	int t = threadIdx.x;
	t_start[t] = start;

        volatile TYPE c_;
        clock_t t1_, t2_, t3_, t4_;
	volatile TYPE a_ = a[t];
	TYPE b_ = b[t];

	volatile clock_t start_time, end_time;
	c_= b_;
	float d=1;	
	volatile TYPE d_ = b_;
        volatile TYPE e_ = d_;
        volatile TYPE f_ = d_;
	for (int i =0; i<2; i++) {
		start_time = clock();
        	//repeat128(c_ = c_ INSTR2  b_;  b_ = b_ INSTR d_; a_ = c_ INSTR2 a_; d_ = c_ INSTR b_;)
        	//repeat171(c_ = c_ INSTR2  b_;  b_ = b_ INSTR a_; a_ = c_ INSTR a_;)
        	//repeat128( a_ = a_ INSTR b_ ;c_ = c_ + operand; d_ = d_+operand; b_ = b_ INSTR a_;)
        	//repeat64( a_ = a_ INSTR b_ ;c_ = c_ + d_; d_ = d_+c_; b_ = b_ INSTR a_;)
        	repeat64( a_ = a_ INSTR b_ ; c_ = c_ INSTR operand; d_ = d_ INSTR operand; e_ = e_ INSTR operand2; f_ = f_ INSTR operand;)
        	//repeat2048( a_ = a_ INSTR operand ; b_ = b_ INSTR operand; c_ = c_ INSTR operand; d = d INSTR operand;)
        	//repeat171(c_ = c_ INSTR  b_;  b_ = b_ INSTR a_; a_ = c_ INSTR a_;)
        	//repeat512(c_ = c_ *  b_; c_ = c_ + b_;) // Fused Multiply Add - FFMA
		end_time = clock ();		
	}
	a[t] = d_;
        a[t] = e_;
        a[t] = f_;
	a[t] = a_;
	b[t] = b_;
	c[t] = c_;
	t0[t] = end_time - start_time;
        t1_ = clock();
        t2_ = clock();
        t3_ = clock();
        t4_ = clock();
	t1[t] = t2_-t1_;
	t2[t] = t3_-t2_;
        t3[t] = t4_-t3_;        
}

int main(int argc, char* argv[]) {
	if (argc != 2) {
		printf ("Usage : arg- NUM THREADS \n EXITING \n");
		return 0;
	}
	
	
	TYPE* c;
	
	const int THREADS = atoi(argv[1]);
 	std::cout << "Num of threads: " << THREADS<< std::endl;
	size_t size = sizeof(TYPE);
	c = (TYPE*) malloc(size*THREADS);
	TYPE* d_c;
	cudaMalloc (&d_c,size*THREADS);

	clock_t *t, *t_start, *t1, *t2, *t3; 		
	t = (clock_t*) malloc(sizeof(clock_t)*THREADS);
	t_start = (clock_t*) malloc(sizeof(clock_t)*THREADS);
	t1 = (clock_t*) malloc(sizeof(clock_t)*THREADS);
	t2 = (clock_t*) malloc(sizeof(clock_t)*THREADS);
	t3 = (clock_t*) malloc(sizeof(clock_t)*THREADS);
	
	clock_t *d_t,  *d_tstart, *d_t1, *d_t2, *d_t3;
	cudaMalloc (&d_t, sizeof(clock_t)*THREADS);
	cudaMalloc (&d_tstart, sizeof(clock_t)*THREADS);
	cudaMalloc (&d_t1, sizeof(clock_t)*THREADS);
	cudaMalloc (&d_t2, sizeof(clock_t)*THREADS);
	cudaMalloc (&d_t3, sizeof(clock_t)*THREADS);
        
	TYPE* a  = (TYPE*) malloc(size*THREADS);
	TYPE* b  = (TYPE*) malloc(size*THREADS);
	for (int i =0 ; i < THREADS; i++) {	
		a[i] = 1;
		b[i] = 2;		
	}
	
	TYPE* d_a;
	TYPE* d_b;
	cudaMalloc(&d_a, size*THREADS);
	cudaMalloc(&d_b, size*THREADS);
	cudaMemcpy(d_a, a , size*THREADS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b , size*THREADS, cudaMemcpyHostToDevice);
	
	//add <<<1,THREADS>>> (d_a, d_b, d_c, d_t, d_t1, d_t2, d_t3);
	add <<<1,THREADS>>> (d_a, d_b, d_c, d_tstart, d_t, d_t1, d_t2, d_t3, 2, 2);
	cudaThreadSynchronize();
//        add <<<1,1>>> (d_c);
	cudaMemcpy (c, d_c, size*THREADS, cudaMemcpyDeviceToHost);
	cudaMemcpy (t, d_t, sizeof(clock_t)*THREADS, cudaMemcpyDeviceToHost);
	cudaMemcpy (t_start, d_tstart, sizeof(clock_t)*THREADS, cudaMemcpyDeviceToHost);
	cudaMemcpy (t1, d_t1, sizeof(clock_t)*THREADS, cudaMemcpyDeviceToHost);
	cudaMemcpy (t2, d_t2, sizeof(clock_t)*THREADS, cudaMemcpyDeviceToHost);
	cudaMemcpy (t3, d_t3, sizeof(clock_t)*THREADS, cudaMemcpyDeviceToHost);

	for(int i=0; i<THREADS; i++) {
		printf ("THREAD %d: \n", i);
		printf ("%f\n", c[i]);
		printf ("Time: %ld \n", t[i]);
		printf ("Start Time: %ld \n", t_start[i]);
		printf ("Time t1: %ld \n", t1[i]);
		printf ("Time t2: %ld \n", t2[i]);
		printf ("Time t3: %ld \n", t3[i]);
		printf("\n");
	}
	cudaFree(d_t);
	cudaFree(d_t1);
	cudaFree(d_t2);
	cudaFree(d_t3);
}
