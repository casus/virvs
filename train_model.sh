#!/bin/bash

# Array of config files
configs=(
    "staining_hadv_1ch.yaml"
    "staining_hadv_2ch.yaml"
    "staining_hsv.yaml"
    "staining_iav.yaml"
    "staining_rhv.yaml"
    "staining_vacv.yaml"
)

# Array of random seeds
seeds=(43 44 45)

for config in "${configs[@]}"; do
    for seed in "${seeds[@]}"; do
        sbatch -J "train_${config%.*}_$seed" <<EOF
#!/bin/bash
#SBATCH -p casus
#SBATCH -A casus
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G

module purge
module load git
module load cuda/12.1

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate virvs

export RANDOM_SEED=$seed
python scripts/training/train_unet.py \
    --neptune-token MY_TOKEN \
    --config-path "configs/$config"
EOF
    done
done