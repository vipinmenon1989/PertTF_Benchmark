#!/bin/bash
#SBATCH --job-name=pertTF_benchmark
#SBATCH --account=ihc
#SBATCH --partition=ihc
#SBATCH --nodelist=ihc-h200-1
#SBATCH --mem=400G
#SBATCH --time=600
#SBATCH --gres=gpu:1
#SBATCH --output=snakemake_pipeline_%j.log
#SBATCH --error=snakemake_pipeline_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pertTF_bench

snakemake --cores "${SLURM_CPUS_ON_NODE:-8}" --rerun-incomplete
