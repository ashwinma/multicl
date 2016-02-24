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
#PBS -l nodes=4:ppn=12

# Access group, queue, and accounting project
#PBS -W group_list=hokiespeed
# Queue name. Replace normal_q with long_q to submit a job to the long queue.
# See HokieSpeed documentation for details on queue parameters.
#PBS -q normal_q
#PBS -A hokiespeed

# Uncomment and add your email address to get an email when your job starts, completes, or aborts
##PBS -M pid@vt.edu
##PBS -m bea

# Add any modules you might require. Use the module avail command to see a list of available modules.
# This example removes all modules, adds the GCC, OpenMPI, and CUDA modules, then loads the FFTW module.
# . ~/.bashrc
# # PBS -t 0-1
cd $PBS_O_WORKDIR

NUM_NODES=4

# Simple single process examples:
#mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d2d2.txt
SNUCL_DEV_0=1 SNUCL_DEV_1=1 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/a_out_columnmajor.${NUM_NODES}.d1d1.txt
# SNUCL_DEV_0=1 SNUCL_DEV_1=${PBS_ARRAYID} mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_columnmajor.${NUM_NODES}.d1d${PBS_ARRAYID}.txt

#SNUCL_DEV_0=0 SNUCL_DEV_1=1 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d0d1.txt

#SNUCL_DEV_0=0 SNUCL_DEV_1=2 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d0d2.txt

#SNUCL_DEV_0=1 SNUCL_DEV_1=0 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d1d0.txt

#SNUCL_DEV_0=1 SNUCL_DEV_1=1 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d1d1.txt

#SNUCL_DEV_0=1 SNUCL_DEV_1=2 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d1d2.txt

#SNUCL_DEV_0=2 SNUCL_DEV_1=0 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d2d0.txt

#SNUCL_DEV_0=2 SNUCL_DEV_1=1 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d2d1.txt

#SNUCL_DEV_0=2 SNUCL_DEV_1=2 mpirun -binding user:0 -np ${NUM_NODES} -ppn 1 -prepend-rank -print-rank-map ./disfd >& output/out_noopt.${NUM_NODES}.d2d2.txt

#sprofile ./disfd >& out_noopt.${NUM_NODES}.d2d2.profile.txt

exit;
