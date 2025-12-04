#!/bin/bash
# ==============================================================================
# Submit All MovieLens Experiments
# 
# This script submits training jobs. Evaluation jobs should be submitted
# AFTER training completes to run comprehensive ranking metrics.
#
# Strategy:
#   1. Training jobs: Fast (4h), no ranking eval
#   2. Evaluation jobs: Comprehensive ranking metrics (6h)
# ==============================================================================

cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

echo "=" * 60
echo "MM-TGN MovieLens Experiment Suite"
echo "=" * 60
echo ""

# Check if jobs directory exists
if [ ! -d "jobs" ]; then
    echo "❌ jobs/ directory not found!"
    exit 1
fi

# ==============================================================================
# PHASE 1: TRAINING JOBS
# ==============================================================================

echo "📦 PHASE 1: Submitting TRAINING jobs..."
echo ""

# Vanilla baseline (random features)
echo "1/4 Submitting: Vanilla (Random Features)"
JOB_VANILLA=$(sbatch jobs/train_ml_vanilla.sh | awk '{print $4}')
echo "    Job ID: $JOB_VANILLA"

# SOTA + MLP fusion
echo "2/4 Submitting: SOTA + MLP Fusion"
JOB_SOTA=$(sbatch jobs/train_ml_sota.sh | awk '{print $4}')
echo "    Job ID: $JOB_SOTA"

# SOTA + FiLM fusion
echo "3/4 Submitting: SOTA + FiLM Fusion"
JOB_FILM=$(sbatch jobs/train_ml_sota_film.sh | awk '{print $4}')
echo "    Job ID: $JOB_FILM"

# SOTA + Gated fusion
echo "4/4 Submitting: SOTA + Gated Fusion"
JOB_GATED=$(sbatch jobs/train_ml_sota_gated.sh | awk '{print $4}')
echo "    Job ID: $JOB_GATED"

echo ""
echo "✅ Training jobs submitted!"
echo ""
echo "=" * 60
echo "TRAINING JOB IDs:"
echo "=" * 60
echo "  Vanilla:     $JOB_VANILLA"
echo "  SOTA+MLP:    $JOB_SOTA"
echo "  SOTA+FiLM:   $JOB_FILM"
echo "  SOTA+Gated:  $JOB_GATED"
echo ""

# ==============================================================================
# PHASE 2: EVALUATION JOBS (submit after training)
# ==============================================================================

echo "=" * 60
echo "PHASE 2: EVALUATION JOBS"
echo "=" * 60
echo ""
echo "⚠️  Submit evaluation jobs AFTER training completes:"
echo ""
echo "    sbatch --dependency=afterok:$JOB_VANILLA jobs/eval_ml_vanilla.sh"
echo "    sbatch --dependency=afterok:$JOB_SOTA jobs/eval_ml_sota.sh"
echo "    sbatch --dependency=afterok:$JOB_FILM jobs/eval_ml_sota_film.sh"
echo "    sbatch --dependency=afterok:$JOB_GATED jobs/eval_ml_sota_gated.sh"
echo ""
echo "Or submit all evaluation jobs after all training completes:"
echo ""
echo "    sbatch --dependency=afterok:$JOB_VANILLA:$JOB_SOTA:$JOB_FILM:$JOB_GATED jobs/eval_ml_vanilla.sh"
echo "    sbatch --dependency=afterok:$JOB_VANILLA:$JOB_SOTA:$JOB_FILM:$JOB_GATED jobs/eval_ml_sota.sh"
echo "    sbatch --dependency=afterok:$JOB_VANILLA:$JOB_SOTA:$JOB_FILM:$JOB_GATED jobs/eval_ml_sota_film.sh"
echo "    sbatch --dependency=afterok:$JOB_VANILLA:$JOB_SOTA:$JOB_FILM:$JOB_GATED jobs/eval_ml_sota_gated.sh"
echo ""

# ==============================================================================
# MONITORING
# ==============================================================================

echo "=" * 60
echo "MONITORING"
echo "=" * 60
echo ""
echo "Check job status:  squeue -u \$USER"
echo "View train logs:   tail -f logs/train_ml_*_<jobid>.out"
echo "View eval logs:    tail -f logs/eval_ml_*_<jobid>.out"
echo "Cancel all:        scancel $JOB_VANILLA $JOB_SOTA $JOB_FILM $JOB_GATED"
echo ""
