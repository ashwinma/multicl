#!/bin/bash


# Example qsub script for HokieSpeed

# NOTE: You will need to edit the Walltime, Node and Processor Per Node (ppn), Queue, and Module lines
# to suit the requirements of your job. You will also, of course have to replace the example job
# commands below with those that run your job.
 
# Set the walltime, which is the maximum time your job can run in HH:MM:SS
# Note that if your job exceeds the walltime estimated during submission, the scheduler
# will kill it. So it is important to be conservative (i.e., to err on the high side)
# with the walltime that you include in your submission script. 
#PBS -l walltime=12:00:00

# Set the number of nodes, and the number of processors per node (generally should be 12)
#PBS -l nodes=compute-0-2:ppn=16

# Access group, queue, and accounting project
# Queue name. Replace normal_q with long_q to submit a job to the long queue.
# See HokieSpeed documentation for details on queue parameters.
#PBS -q c2050
#PBS -A fire

# Uncomment and add your email address to get an email when your job starts, completes, or aborts
##PBS -M pid@vt.edu
##PBS -m bea

# Add any modules you might require. Use the module avail command to see a list of available modules.
# This example removes all modules, adds the GCC, OpenMPI, and CUDA modules, then loads the FFTW module.
cd $PBS_O_WORKDIR

# Simple single process examples:
time mpirun -np 1 -ppn 1 -prepend-rank  ./disfd

exit;
