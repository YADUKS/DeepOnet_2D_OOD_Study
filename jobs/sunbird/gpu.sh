#!/bin/bash
#SBATCH --job-name=ood_gpu
#SBATCH --partition=accel_ai
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%j.out

# Partition/GPU defaults follow the reference repo; override for your allocation.
set -euo pipefail
PROJECT_DIR="${STUDY_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" ]]; then
    PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_DIR"
if [[ -n "${STUDY_MODULE:-}" ]]; then
    module load "$STUDY_MODULE"
fi
PYTHON="${STUDY_PYTHON:-${STUDY_ENV:-$PROJECT_DIR/.venv}/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment missing: $PYTHON. Set STUDY_ENV or STUDY_PYTHON." >&2
    exit 2
fi
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname): $PYTHON -u run.py $*"
exec "$PYTHON" -u run.py "$@" --device cuda --cpu-threads "$OMP_NUM_THREADS"
