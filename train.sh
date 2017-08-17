#!/bin/bash

#SBATCH -p cox
#SBATCH --gres=gpu:1
#SBATCH --constraint=titanx
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine                        
#SBATCH --mem=48000
#SBATCH -t 0-168:00:00
#SBATCH --output=./output/train_runet_halfres.out

module load python/2.7.12-fasrc01
source deactivate
source activate keras
module load cuda/7.5-fasrc01
module load cudnn/7.0-fasrc01

# For Batch
THEANO_FLAGS=device=gpu,exception_verbosity=high,floatX=float32 python -u train.py
# For RC
#THEANO_FLAGS=device=gpu,exception_verbosity=high srun -p cox -n 1 -N 1 --mem=32000 -t 0-2:00:00 python -u train.py
#THEANO_FLAGS=device=gpu,exception_verbosity=high,floatX=float32,lib.cnmem=0.75,optimizer=None srun -p cox --gres=gpu:1 -n 1 -N 1 --mem=48000 -t 0-2:00:00 python -u train3d_same.py
# For Local
#THEANO_FLAGS=exception_verbosity=high python -u train.py
# end of program
exit 0;