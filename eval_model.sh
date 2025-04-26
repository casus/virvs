#!/bin/bash
#SBATCH -p casus_a100
#SBATCH -A casus
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1             # Keeps 2 GPUs, but consider reducing to 1 if your code doesn't need multi-GPU
#SBATCH --cpus-per-task=24       # Reduced from 48 to prevent memory overallocation
#SBATCH --mem=240G               # Changed to total memory instead of per-cpu
#SBATCH --nodes=1                # Explicitly request 1 node
#SBATCH --ntasks-per-node=1      # Run 1 task per node
#SBATCH --hint=nomultithread     # Disable hyperthreading for better memory usage

module purge
module load git
module load cuda/12.1
module load intel/19.0
module load openblas/0.3.10

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate virvs

# HAdV 1ch
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hadv_1ch/weights/model_100000_1651_20250422_080927.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed_43/test" --virus "hadv_1ch"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hadv_1ch/weights/model_100000_a07c_20250422_080929.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed_44/test" --virus "hadv_1ch"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hadv_1ch/weights/model_100000_0ea3_20250422_080940.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed_45/test" --virus "hadv_1ch"

HAdV 2ch
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hadv_2ch/weights/model_100000_d493_20250422_080941.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed_43/test" --virus "hadv_2ch"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hadv_2ch/weights/model_100000_d225_20250422_081002.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed_44/test" --virus "hadv_2ch"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hadv_2ch/weights/model_100000_3d6d_20250422_081037.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HADV/processed_45/test" --virus "hadv_2ch"

# HSV
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hsv/weights/model_100000_a7f0_20250422_080942.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed_43/test" --virus "hsv"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hsv/weights/model_100000_a42c_20250422_080942.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed_44/test" --virus "hsv"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/hsv/weights/model_100000_6758_20250422_081037.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/HSV/processed_45/test" --virus "hsv"

# IAV
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/iav/weights/model_100000_58e4_20250422_080939.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed_43/test" --virus "iav"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/iav/weights/model_100000_b99a_20250422_080941.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed_44/test" --virus "iav"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/iav/weights/model_100000_cb70_20250422_080959.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/IAV/processed_45/test" --virus "iav"

# RV
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/rhv/weights/model_100000_f90c_20250422_080944.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed_43/test" --virus "rv"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/rhv/weights/model_100000_7df5_20250422_080946.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed_44/test" --virus "rv"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/rhv/weights/model_100000_13bd_20250422_080950.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/RV/processed_45/test" --virus "rv"

# VACV
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/vacv/weights/model_100000_c1e6_20250422_080942.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed_43/test" --virus "vacv"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/vacv/weights/model_100000_fa4c_20250422_080943.h5" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed_44/test" --virus "vacv"
python scripts/testing/evaluate.py --model unet --weights "/bigdata/casus/MLID/maria/outputs/vacv/weights/model_100000_8ebf_20250422_080943.h" --dataset "/bigdata/casus/MLID/maria/VIRVS_data/VACV/processed_45/test" --virus "vacv"
