#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Maverick2 GTX nodes
#----------------------------------------------------

#SBATCH -J HW3_g0                        # Job name
#SBATCH -o HW3_g0.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e HW3_g0.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gtx                           # Queue (partition) name
#SBATCH -N 1                             # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                             # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                      # Run time (hh:mm:ss)
#SBATCH --mail-user=<UT_MAIL_INSERT_HERE>@utexas.edu
#SBATCH --mail-type=all                  # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A EdgeAI-EE379K                 # Allocation name

# Other commands must follow all #SBATCH directives...

module load intel/18.0.2 python3/3.7.0
module load cuda/10.1 cudnn/7.6.5 nccl/2.5.6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/10.1/lib64
source $WORK/HW3_virtualenv/bin/activate

# Launch code...

# ---------------------------------------------------

