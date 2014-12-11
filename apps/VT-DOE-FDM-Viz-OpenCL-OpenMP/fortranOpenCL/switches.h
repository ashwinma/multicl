#ifndef  __SWITCHES__
#define __SWITCHES__
/*
 * Set to 0 or 1 to turn off/on the use of MPIX calls (MPI-ACC)
 * Use of MPIX calls makes GPUs 1st class citizens
 * with MPI communication happening directly from GPU buffers
 */
#define USE_MPIX 1
#if USE_MPIX == 1
#define DISFD_GPU_MARSHALING
#endif
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


#endif // __SWITCHES__
