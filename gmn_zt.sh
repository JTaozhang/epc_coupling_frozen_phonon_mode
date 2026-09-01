#!/bin/sh
#SBATCH --job-name=gmn
#SBATCH --nodes=1              #节点数
#SBATCH --ntasks-per-node=1   #每个节点的核数
#SBATCH --cpus-per-task=4 
#SBATCH --output=%j.log
#SBATCH --error=%j.err
#SBATCH --partition=regular

# load the environment
module purge
module load oneapi/2024.1/mkl
module load oneapi/2024.1/mpi
module load oneapi/2024.1/compiler

export I_MPI_PMI_LIBRARY=/opt/gridview/slurm/lib/libpmi.so


export MKL_DEBUG_CPU_TYPE=5
export MKL_CBWR=AVX2
export I_MPI_PIN_DOMAIN=numa
export I_MPI_FABRICS=shm:ofi
export OMP_NUM_THREADS=3
export UCX_TLS=self,sm,ud
ulimit -s unlimited


#########################command to run###################

python calculate_epc_gamma_v2.py --hr-plus plus/tb_hr.dat  --hr-minus minus/tb_hr.dat --wavefunction equi/tb_wavef.dat  --kpoints KPOINTS  --metadata phonon_q0000_b023_metadata.json   --bands 5580 5581 5582 5583 5584 5585 5586

#这行 Shell 代码使用 printf 命令将一组变量格式化输出，并追加到文件 /public/home/zhangtao/work/job.list 中。
printf "%-10s %-15s %-12s %-90s %-25s\n" $SLURM_JOB_ID $SLURM_JOB_NAME $(date +%Y%m%d%H) $SLURM_SUBMIT_DIR 'RUNNING FOR TB model' >> /public/home/tzhang/job.list