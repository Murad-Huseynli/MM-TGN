# 🕸️ MM-TGN: Multimodal Temporal Graph Network

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

A multimodal extension of Temporal Graph Networks for recommendation systems, addressing **cold-start** and **concept drift** problems through SOTA vision-language features.

---

## 🎯 Research Goal

> **Hypothesis**: Multimodal features (text + image) modulated by temporal context solve the Cold Start problem better than ID-only collaborative filtering.

**Key Innovation**: Combine TGN's temporal memory with SOTA multimodal embeddings (Qwen2-1.5B + SigLIP-SO400M).

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Navigate to project
cd /scratch/cse576f25s001_class_root/cse576f25s001_class/huseynli/mm-tgn

# Activate conda environment
conda activate mmtgn

# Verify GPU
nvidia-smi
```

### 2. Run Smoke Test (Quick Verification)

```bash
# Submit smoke test job (~45 min total: 25min training + 15min eval + 5min splits)
sbatch jobs/smoke_test.sh

# Or run interactively
srun --account cse576f25s001_class --partition gpu --gpus 1 --mem 32G --time 01:00:00 \
    python train_mmtgn.py --data-dir data/processed --dataset ml-modern --epochs 1
```

### 3. Run Full Ablation Study (Two-Phase)

```bash
# PHASE 1: Submit ALL training experiments (4 jobs, ~4h each)
./jobs/submit_all_ml.sh
# Jobs: Vanilla, SOTA+MLP, SOTA+FiLM, SOTA+Gated

# PHASE 2: After training completes, submit evaluation jobs
sbatch --dependency=afterok:<TRAIN_JOB_ID> jobs/eval_ml_vanilla.sh
sbatch --dependency=afterok:<TRAIN_JOB_ID> jobs/eval_ml_sota.sh
# etc. (see submit_all_ml.sh output for exact commands)
```

### 4. Monitor Progress

```bash
# Check job status
squeue -u $USER

# View training output
tail -f logs/train_ml_*.out

# View evaluation output (after Phase 2)
tail -f logs/eval_ml_*.out

# Results saved to:
# checkpoints/<run_name>/best_model.pt        # Model checkpoint
# checkpoints/<run_name>/results_partial.json # Link pred metrics (after train)
# checkpoints/<run_name>/results_full.json    # All metrics (after eval)
```

---

## 📊 Evaluation Protocol (FOR BASELINE ALIGNMENT)

### Data Split: **Chronological 70/15/15**

| Split | Ratio | Description |
|-------|-------|-------------|
| Train | 70% | Oldest interactions |
| Validation | 15% | Middle interactions |
| Test | 15% | Newest interactions |

**⚠️ IMPORTANT**: Split is by TIMESTAMP, not random. This prevents future data leakage and is required for TGN's temporal memory.

### Canonical Splits (For Team Alignment)

Pre-exported splits for all teammates:
```
data/splits/
├── ml-modern/      # MovieLens (1M interactions)
├── amazon-cloth/   # Amazon Clothing (510K)
└── amazon-sports/  # Amazon Sports (218K)
```

Each contains: `train.csv`, `val.csv`, `test.csv`, `splits_metadata.json`

### Three-Group Evaluation

We report metrics in **THREE groups** for fair comparison:

| Group | Description | Purpose |
|-------|-------------|---------|
| **Overall** | All test interactions | Headline metric |
| **Transductive** | Users seen in training | Fair comparison with LOO baselines |
| **Inductive** | Cold-start users (new in test) | MM-TGN's key advantage |

### Ranking Strategy: **Negative Sampling (N=100)**

For each positive test edge:
1. Sample 100 random negative items
2. Rank positive among 101 candidates
3. Compute metrics based on rank

### Metrics Reported

| Metric | Description |
|--------|-------------|
| **Recall@10** | % of positives ranked in top-10 |
| **Recall@20** | % of positives ranked in top-20 |
| **NDCG@10** | Normalized DCG at rank 10 |
| **NDCG@20** | Normalized DCG at rank 20 |
| **MRR** | Mean Reciprocal Rank |
| **AUC** | Area Under ROC Curve |
| **AP** | Average Precision |

---

## 📁 Project Structure

```
mm-tgn/
├── train_mmtgn.py          # Main training script
├── mmtgn.py                # MMTGN model architecture
├── dataset.py              # Data loading & splits
├── modules/
│   ├── embedding.py        # HybridNodeFeatures, FiLM, Projectors
│   ├── memory.py           # TGN Memory
│   └── ...
├── utils/
│   ├── metrics.py          # Recall@K, NDCG@K, MRR
│   └── utils.py            # NeighborFinder, Samplers
├── data/
│   ├── processed/          # TGN-formatted data
│   ├── splits/             # Canonical train/val/test splits
│   ├── datasets/           # Raw datasets
│   └── script/             # Data processing scripts
├── jobs/                   # SLURM job scripts ⭐
│   ├── smoke_test.sh       # Quick verification
│   ├── train_ml_vanilla.sh # Vanilla baseline
│   ├── train_ml_sota.sh    # SOTA + MLP fusion
│   ├── train_ml_sota_film.sh    # SOTA + FiLM fusion
│   ├── train_ml_sota_gated.sh   # SOTA + Gated fusion
│   └── submit_all_ml.sh    # Submit all experiments
├── checkpoints/            # Saved models
├── runs/                   # TensorBoard logs
├── logs/                   # Job output logs
├── ARCHITECTURE.md         # Detailed technical documentation
└── README.md               # This file
```

---

## 🧪 Ablation Studies

### Feature Source Ablation

| Experiment | Command | Purpose |
|------------|---------|---------|
| Vanilla | `sbatch jobs/train_ml_vanilla.sh` | Lower bound (no content) |
| SOTA | `sbatch jobs/train_ml_sota.sh` | Full multimodal |

### Multimodal Fusion Ablation

| Experiment | Command | Description |
|------------|---------|-------------|
| **MLP** | `--mm-fusion mlp` | Concatenate text+image, then 2-layer MLP projection: `MLP(concat(text, image))` |
| **FiLM** | `--mm-fusion film` | Text modulates image features: `γ(text) ⊙ proj(image) + β(text)` |
| **Gated** | `--mm-fusion gated` | Learned attention weights: `gate ⊙ proj(text) + (1-gate) ⊙ proj(image)` |

**Note**: All fusion methods output 172-dim embeddings (TGN working dimension). Input: pre-concatenated 2688-dim features (1536 text + 1152 image).

---

## 📈 Training Results (December 5, 2025)

### MovieLens (ML-Modern)

| Model | Val AP | Val AUC | Val MRR | Status |
|-------|--------|---------|---------|--------|
| **SOTA (Qwen2+SigLIP)** | **0.849** | **0.872** | **0.934** | ✅ Running |
| Vanilla (random) | 0.473 | 0.518 | 0.757 | ✅ Completed |

### Key Finding: **+79% improvement** from multimodal features!

| Comparison | Result | Interpretation |
|------------|--------|----------------|
| SOTA AP > Vanilla AP | 0.849 > 0.473 | ✅ Multimodal features help significantly |
| SOTA AUC > Vanilla AUC | 0.872 > 0.518 | ✅ Better discrimination |
| SOTA MRR > Vanilla MRR | 0.934 > 0.757 | ✅ Positive items ranked higher |

**All experiments use BPR (Bayesian Personalized Ranking) loss** - standard for recommender systems.

**Next**: Full ranking metrics (Recall@K, NDCG@K) will be computed by separate evaluation jobs.

---

## 🤝 For Teammates Running Baselines

### Required Settings

```python
SPLIT_TYPE = "chronological"  # NOT random or LOO
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
N_NEGATIVES = 100  # Per positive for ranking
EVAL_SEED = 42     # For reproducible negative sampling
METRICS = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "MRR"]
```

### Use Canonical Splits

```bash
# Pre-exported for everyone:
data/splits/ml-modern/train.csv    # 700K rows
data/splits/ml-modern/val.csv      # 150K rows
data/splits/ml-modern/test.csv     # 150K rows
```

### ⚠️ Fixed Evaluation Samples (CRITICAL for Fair Comparison)

Full test set (150K) is too slow. Use the **same fixed sample** for all models:

```bash
# Pre-generated sample (5,000 interactions, seed=42):
data/eval_samples/ml-modern_eval_sample.csv    # USE THIS
data/eval_samples/ml-modern_eval_metadata.json # Stats & seed info
```

**All baselines (LightGCN, SASRec, MMGCN) must evaluate on this exact sample!**

### Results Location

```
checkpoints/<run_name>/
├── best_model.pt           # Model checkpoint
├── train.log               # Training log
├── results_partial.json    # Link prediction metrics
└── results.json            # All metrics (if ranking eval completes)
```

---

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed technical documentation
  - System architecture diagrams
  - Complete CLI reference
  - API documentation
  - Known issues & solutions

---

## 🔧 TensorBoard

```bash
# On compute node (where your job is running)
tensorboard --logdir=runs --port=6006 --host=0.0.0.0 &
# Check node: squeue -u $USER

# On local machine (SSH tunnel - replace gl1013 with your compute node)
ssh -L 6006:gl1013.arc-ts.umich.edu:6006 huseynli@greatlakes.arc-ts.umich.edu

# Open browser: http://localhost:6006
```

**What you'll see in TensorBoard:**
- **Scalars**: Train Loss, Val AP, Val AUC per epoch
- **Final Metrics**: test_AP, test_Recall@10, test_NDCG@10
- **Split Metrics**: trans_AP, induct_AP, trans_Recall@10, induct_Recall@10

---

## 📝 Citation

```bibtex
@misc{mmtgn2025,
  title={MM-TGN: Multimodal Temporal Graph Networks for Cold-Start Recommendation},
  author={CSE576 Team},
  year={2025},
  institution={University of Michigan}
}
```

---

## 📄 License

MIT License - see LICENSE file for details.
