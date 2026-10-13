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
# Explicit settings take priority; discover the documented Conda env on scratch.
if [[ -n "${STUDY_PYTHON:-}" ]]; then
    PYTHON="$STUDY_PYTHON"
elif [[ -n "${STUDY_ENV:-}" ]]; then
    PYTHON="$STUDY_ENV/bin/python"
elif [[ -x "$PROJECT_DIR/env/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/env/bin/python"
else
    PYTHON="$PROJECT_DIR/.venv/bin/python"
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment missing: $PYTHON. Set STUDY_ENV or STUDY_PYTHON." >&2
    exit 2
fi
# Pip's native extensions may otherwise load Sunbird's older system C++ runtime.
# Select libraries from the chosen interpreter's own environment for this job.
PYTHON_PREFIX="$(cd -- "$(dirname -- "$PYTHON")/.." && pwd)"
NATIVE_LIBS=""
for runtime_library in libgcc_s.so.1 libstdc++.so.6; do
    if [[ -f "$PYTHON_PREFIX/lib/$runtime_library" ]]; then
        NATIVE_LIBS="${NATIVE_LIBS:+$NATIVE_LIBS:}$PYTHON_PREFIX/lib/$runtime_library"
    fi
done
if [[ -n "$NATIVE_LIBS" ]]; then
    echo "Native runtime: $NATIVE_LIBS"
    export LD_PRELOAD="$NATIVE_LIBS${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname): $PYTHON -u run.py $*"
exec "$PYTHON" -u run.py "$@" --device cuda --cpu-threads "$OMP_NUM_THREADS"
