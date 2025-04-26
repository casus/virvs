#!/bin/bash
#SBATCH -p casus_a100
#SBATCH -A casus
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=10G

module purge
module load git
module load cuda/12.1
module load intel/19.0
module load openblas/0.3.10

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate virvs
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/vacv/weights/model_100000_8ebf_20250422_080943.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed_43/test" --virus "vacv"