#!/bin/sh
#SBATCH --partition=gpu --gres=gpu:1 --mem=96GB --time=5:00:00 -J refine_4 -o /scratch/ttrine/speed/experiments/refine_4/log.out
cd /scratch/ttrine/speed/
module load python/2.7.12/b2 blas hdf5/1.8.14 cudnn/7.5 git opencv
module unload python/2.7.10
source /scratch/ttrine/speed/venv/bin/activate
PYTHONPATH=/scratch/ttrine/speed/venv/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH
export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high
python -m experiments.refine_4.train 1000 10 2500
