//
// © 2013.  Virginia Polytechnic Institute & State University
// 
// This GPU-accelerated code is based on the MPI code supplied by Pengcheng Liu of USBR.
//
/*
 * Set to 0 or 1 to turn off/on the use of MPIX calls (MPI-ACC)
 * Use of MPIX calls makes GPUs 1st class citizens
 * with MPI communication happening directly from GPU buffers
 */
#define USE_MPIX 0
/*
 * If set to 1, turns ON the comprehensive correctness testing.
 * The baseline correct code is the CUDA code where all the data marshalling
 * and MPI communication takes place on CPU
 * Can test correctness of both MPIX enabled/disabled code
 * CAUTION: The output timings of execution of various functions would be
         much larger because of correctness validation
*/ 
#define VALIDATE_MODE 0

/*
 * If enabled to 1, it will start storing the computation arrays
 * in the node-{rank}.log file
 * Each node has its own .log file.
 * Used for deep digging/debugging
 * CAUTION: The output timings of execution of various functions would be
         much larger because of logging and writing to disk
 */
#define LOGGING_ENABLED 0


/*
 * If enabled to 1, the marshalling computation (both on GPU & CPU,
 *  on CPU only if VALIDATE_MODE is 1) will be timed 
 * Logging in the node-{rank}.log file
 * Each node has its own .log file.
 */
#define DETAILED_TIMING 0

/*
 * If enabled to 1, the velocity and stress computation results
 * are validated with C version
 */
#define COMPUTATION_CORRECTNESS_MODE 0

/*
 * Switch bw optimised and originial(unoptimised) kernels
 */
#define USE_Optimized_velocity_inner_IC 1
#define USE_Optimized_vel_PmlX_IC 1
#define USE_Optimized_vel_PmlY_IC 1
#define USE_Optimized_velocity_inner_IIC 1
#define USE_Optimized_vel_PmlX_IIC 1
#define USE_Optimized_vel_PmlY_IIC 1
#define USE_Optimized_vel_PmlZ_IIC 1

#define USE_Optimized_stress_norm_xy_IC 1
#define USE_Optimized_stress_xz_yz_IC 1
#define USE_Optimized_stress_norm_PmlX_IC 1
#define USE_Optimized_stress_norm_PmlY_IC 1 
#define USE_Optimized_stress_xy_PmlX_IC 1
#define USE_Optimized_stress_xy_PmlY_IC 1
#define USE_Optimized_stress_xz_PmlX_IC 1
#define USE_Optimized_stress_xz_PmlY_IC 1
#define USE_Optimized_stress_yz_PmlX_IC 1 
#define USE_Optimized_stress_yz_PmlY_IC 1 
#define USE_Optimized_stress_norm_xy_II 1
#define USE_Optimized_stress_xz_yz_IIC 1
#define USE_Optimized_stress_norm_PmlX_IIC 1
#define USE_Optimized_stress_norm_PmlY_II 1
#define USE_Optimized_stress_norm_PmlZ_IIC 1
#define USE_Optimized_stress_xy_PmlX_IIC 1
#define USE_Optimized_stress_xy_PmlY_IIC 1
#define USE_Optimized_stress_xy_PmlZ_II 1
#define USE_Optimized_stress_xz_PmlX_IIC 1
#define USE_Optimized_stress_xz_PmlY_IIC 1
#define USE_Optimized_stress_xz_PmlZ_IIC 1
#define USE_Optimized_stress_yz_PmlX_IIC 1
#define USE_Optimized_stress_yz_PmlY_IIC 1
#define USE_Optimized_stress_yz_PmlZ_IIC 1
