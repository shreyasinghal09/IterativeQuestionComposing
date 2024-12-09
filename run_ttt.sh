#! /bin/bash
#SBATCH --job-name=ttt
#SBATCH --nodes=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=amittur@cs.cmu.edu
#SBATCH --output=ttt-%j.out

eval "$(conda shell.bash hook)"
conda activate iqc

# echo $1

srun python test_time_tuning_eval.py $1
# srun python test_time_tuning_eval.py 

