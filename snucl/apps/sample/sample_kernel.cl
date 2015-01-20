//#include "sample.h"
// v = 10.0
#define ADD1_OP   s=v-s;
#define ADD2_OP   ADD1_OP s2=v-s2;
#define ADD4_OP   ADD2_OP s3=v-s3; s4=v-s4;
#define ADD8_OP   ADD4_OP s5=v-s5; s6=v-s6; s7=v-s7; s8=v-s8;

// v = 1.01
#define MUL1_OP   s=s*s*v;
#define MUL2_OP   MUL1_OP s2=s2*s2*v;
#define MUL4_OP   MUL2_OP s3=s3*s3*v; s4=s4*s4*v;
#define MUL8_OP   MUL4_OP s5=s5*s5*v; s6=s6*s6*v; s7=s7*s7*v; s8=s8*s8*v;

// v1 = 10.0, v2 = 0.9899
#define MADD1_OP  s=v1-s*v2;
#define MADD2_OP  MADD1_OP s2=v1-s2*v2;
#define MADD4_OP  MADD2_OP s3=v1-s3*v2; s4=v1-s4*v2;
#define MADD8_OP  MADD4_OP s5=v1-s5*v2; s6=v1-s6*v2; s7=v1-s7*v2; s8=v1-s8*v2;

// v1 = 3.75, v2 = 0.355
#define MULMADD1_OP  s=(v1-v2*s)*s;
#define MULMADD2_OP  MULMADD1_OP s2=(v1-v2*s2)*s2;
#define MULMADD4_OP  MULMADD2_OP s3=(v1-v2*s3)*s3; s4=(v1-v2*s4)*s4;
#define MULMADD8_OP  MULMADD4_OP s5=(v1-v2*s5)*s5; s6=(v1-v2*s6)*s6; s7=(v1-v2*s7)*s7; s8=(v1-v2*s8)*s8;

#define ADD1_MOP20  \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP \
     ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP ADD1_OP
#define ADD2_MOP20  \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP \
     ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP ADD2_OP
#define ADD4_MOP10  \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP \
     ADD4_OP ADD4_OP ADD4_OP ADD4_OP ADD4_OP
#define ADD8_MOP5  \
     ADD8_OP ADD8_OP ADD8_OP ADD8_OP ADD8_OP

#define MUL1_MOP20  \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP \
     MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP MUL1_OP
#define MUL2_MOP20  \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP \
     MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP MUL2_OP
#define MUL4_MOP10  \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP \
     MUL4_OP MUL4_OP MUL4_OP MUL4_OP MUL4_OP
#define MUL8_MOP5  \
     MUL8_OP MUL8_OP MUL8_OP MUL8_OP MUL8_OP

#define MADD1_MOP20  \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP \
     MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP MADD1_OP
#define MADD2_MOP20  \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP \
     MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP MADD2_OP
#define MADD4_MOP10  \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP \
     MADD4_OP MADD4_OP MADD4_OP MADD4_OP MADD4_OP
#define MADD8_MOP5  \
     MADD8_OP MADD8_OP MADD8_OP MADD8_OP MADD8_OP

#define MULMADD1_MOP20  \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP
#define MULMADD2_MOP20  \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP
#define MULMADD4_MOP10  \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP \
     MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP MULMADD4_OP
#define MULMADD8_MOP4  \
     MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP
#define MULMADD8_MOP2  \
     MULMADD8_OP MULMADD8_OP 
#define MULMADD8_MOP1  \
     MULMADD8_OP 
#define MULMADD8_MOP5  \
     MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP MULMADD8_OP
#define MULMADD1_MOP5  \
     MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP MULMADD1_OP 
#define MULMADD2_MOP16  \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP \
     MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP MULMADD2_OP 
#define MULMADD2_MOP1  \
     MULMADD2_OP 

__kernel void marshal_kernel(__global float *d_a, __global float* other_d_a, int nIters, float v1, float v2, long int d_size)
{
	int globalThreadId = (get_group_id(0) * get_local_size(0)) + get_local_id(0);
	int localThreadId = get_local_id(0);
	int blockId = get_group_id(0);
	int globalThreadCount = get_num_groups(0) * get_local_size(0);
	int localThreadCount = get_local_size(0);
	int blockCount = get_num_groups(0);

	for(int i = globalThreadId; i < d_size; i += globalThreadCount)
	{
		//register 
		float s = d_a[i], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2, s5=8.0f-s, s6=8.0f-s2, s7=7.0f-s, s8=7.0f-s2;
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		for (int j=0 ; j<nIters ; ++j) {
			/* Each macro op has 5 operations. 
			   Unroll 4 more times for 20 operations total.
			   */
			MULMADD8_MOP1
		}
		d_a[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
	}
}

__kernel void compute_kernel(__global float *d_a, int nIters, float v1, float v2, long int d_size)
{
	int globalThreadId = (get_group_id(0) * get_local_size(0)) + get_local_id(0);
	int localThreadId = get_local_id(0);
	int blockId = get_group_id(0);
	int globalThreadCount = get_num_groups(0) * get_local_size(0);
	int localThreadCount = get_local_size(0);
	int blockCount = get_num_groups(0);

	for(int i = globalThreadId; i < d_size; i += globalThreadCount)
	{
		//register 
		float s = d_a[i], s2=10.0f-s, s3=9.0f-s, s4=9.0f-s2, s5=8.0f-s, s6=8.0f-s2, s7=7.0f-s, s8=7.0f-s2;
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		MULMADD8_MOP1
		for (int j=0 ; j<nIters ; ++j) {
			/* Each macro op has 5 operations. 
			   Unroll 4 more times for 20 operations total.
			   */
			MULMADD8_MOP1
		}
		d_a[i] = ((s+s2)+(s3+s4))+((s5+s6)+(s7+s8));
	}
}
#if 0
void addKernel(cl_mem d_a, const size_t d_size, cl_command_queue d_queue, const int iter_scale)
{
	/* d_a should have 4194304 floats */
	int blocks = 14 * 8;
	int threads = 128;
	int nIters = iter_scale;
	//int nIters = (int)round(3.21571*iter_scale);
	__addKernel<float><<< blocks, threads, 0, d_stream>>> (d_a, nIters, 3.75, 0.355, d_size);
}
#endif
