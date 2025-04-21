#!/bin/bash

# Define viruses and their corresponding preprocessing scripts
declare -A virus_scripts=(
  ["VACV"]="preprocess_vacv.py"
  ["HADV"]="preprocess_hadv.py"
  ["HSV"]="preprocess_other.py"
  ["RV"]="preprocess_other.py"
  ["IAV"]="preprocess_other.py"
)

# Random seeds to iterate over
seeds=(43 44 45)

# Loop through each virus and run preprocessing + to_npy.py
for virus in "${!virus_scripts[@]}"; do
  script="${virus_scripts[$virus]}"
  
  echo "Processing $virus with $script"
  
  for seed in "${seeds[@]}"; do
    echo "Running with RANDOM_SEED=$seed"
    
    # Run preprocessing script
    VIRUS="$virus" RANDOM_SEED="$seed" python "scripts/data_processing/$script"
    
    # Run to_npy.py
    VIRUS="$virus" RANDOM_SEED="$seed" python "scripts/data_processing/to_npy.py"
  done
done

echo "All processing complete."