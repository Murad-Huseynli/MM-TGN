# Training Strategies Summary

## Complete Training Configuration

### ✅ **Currently Used Strategies**

| Strategy | Type | Value | Default? | Notes |
|----------|------|-------|----------|-------|
| **Optimizer** | Adam | `lr=1e-4`, `weight_decay=1e-5` | ✅ | Standard Adam |
| **Learning Rate Scheduler** | ReduceLROnPlateau | `mode='max'`, `factor=0.5`, `patience=2` | ✅ | Reduces LR when val AP plateaus |
| **Early Stopping** | EarlyStopMonitor | `patience=5` | ✅ | Stops if no improvement for 5 epochs |
| **Gradient Clipping** | Clip norm | `max_norm=1.0` | ✅ | Prevents gradient explosion |
| **Dropout** | Dropout layers | `0.1` | ✅ | Applied in attention/MLP layers |
| **Loss Function** | BPR | `bpr_loss` | ✅ | Bayesian Personalized Ranking |
| **Batch Size** | Fixed | `200` | ✅ | Standard for TGN |
| **Epochs** | Max | `50` | ✅ | Early stopping can stop earlier |

---

## Detailed Breakdown

### 1. Optimizer: **Adam**

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,              # Learning rate
    weight_decay=1e-5     # L2 regularization
)
```

**Configuration:**
- Learning Rate: `1e-4` (0.0001)
- Weight Decay: `1e-5` (0.00001) - L2 regularization
- Beta1/Beta2: PyTorch defaults (0.9, 0.999)

---

### 2. Learning Rate Scheduler: **ReduceLROnPlateau**

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # Monitor validation AP (higher is better)
    factor=0.5,        # Reduce LR by 50% when plateau detected
    patience=2         # Wait 2 epochs without improvement
)
```

**Behavior:**
- Monitors: Validation AP (Average Precision)
- Reduces LR by 50% when val AP doesn't improve for 2 consecutive epochs
- Continues training with reduced LR

**Example:**
```
Epoch 1-3: lr = 1e-4 (val AP improving)
Epoch 4-5: lr = 1e-4 (val AP plateaus for 2 epochs)
Epoch 6+:  lr = 5e-5 (reduced by 50%)
```

---

### 3. Early Stopping: **5 Epochs Patience**

```python
early_stopper = EarlyStopMonitor(
    max_round=5,         # Stop if no improvement for 5 epochs
    higher_better=True   # Monitor validation AP (higher = better)
)
```

**Behavior:**
- Monitors: Validation AP
- Stops training if val AP doesn't improve for **5 consecutive epochs**
- Saves best model based on highest val AP

**Example:**
```
Epoch 15: val AP = 0.85 (best so far)
Epoch 16: val AP = 0.84
Epoch 17: val AP = 0.83
Epoch 18: val AP = 0.84
Epoch 19: val AP = 0.83
Epoch 20: val AP = 0.82
→ Early stop! (No improvement for 5 epochs)
```

---

### 4. Gradient Clipping: **Max Norm = 1.0**

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Purpose:**
- Prevents gradient explosion
- Clips gradient norm to maximum 1.0
- Applied after `loss.backward()`, before `optimizer.step()`

---

### 5. Dropout: **0.1 (10%)**

**Applied in:**
- Graph attention layers
- MLP layers (message function, projectors)
- Multimodal fusion layers

**Value:** `dropout=0.1` (default)

---

### 6. Loss Function: **BPR (Bayesian Personalized Ranking)**

```python
loss = -log(sigmoid(pos_score - neg_score))
```

**Purpose:**
- Directly optimizes ranking metrics
- Maximizes margin between positive and negative scores
- Standard for recommender systems

---

## Training Loop Structure

```
For each epoch:
  1. Reset memory state
  2. Train on all batches:
     - Forward pass
     - Compute BPR loss
     - Backward pass
     - Gradient clipping (max_norm=1.0)
     - Optimizer step
     - Detach memory (for TBPTT)
  3. Validate:
     - Backup memory
     - Evaluate on validation set
     - Restore memory
  4. Learning rate scheduling:
     - Scheduler.step(val_ap)
     - Reduces LR if plateau detected
  5. Early stopping check:
     - Stops if no improvement for 5 epochs
  6. Save best model (based on val_ap)
```

---

## Hyperparameter Values

| Parameter | Value | CLI Argument |
|-----------|-------|--------------|
| Learning Rate | `1e-4` | `--lr` |
| Weight Decay | `1e-5` | `--weight-decay` |
| Dropout | `0.1` | `--dropout` |
| Batch Size | `200` | `--batch-size` |
| Max Epochs | `50` | `--epochs` |
| Early Stopping Patience | `5` | `--patience` |
| LR Scheduler Factor | `0.5` | Hardcoded |
| LR Scheduler Patience | `2` | Hardcoded |
| Gradient Clip Norm | `1.0` | Hardcoded |

---

## Summary

**YES**, we use all standard training strategies:
- ✅ Early stopping (patience=5)
- ✅ Dropout (0.1)
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Gradient clipping (max_norm=1.0)
- ✅ Weight decay (L2 regularization, 1e-5)
- ✅ BPR loss (ranking-optimized)

All strategies are **enabled by default** and follow standard TGN/recommender system practices!

