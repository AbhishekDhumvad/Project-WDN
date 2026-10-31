#!/bin/bash
# Run the study as a batch job on a Slurm cluster (CPU cores for EPANET; add a GPU line only if one is available).
#   sbatch scripts/run_slurm.sh            -> all 6 networks
#   sbatch scripts/run_slurm.sh 5          -> Hanoi only
# Adapt account / partition / time to your cluster.
#SBATCH --job-name=leak_study
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
##SBATCH --gres=gpu:1            # optional: encoder-decoder training on the GPU

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
chmod +x dist/main.exe
./dist/main.exe "${@:-all}"
