# TGN Module Configuration for MM-TGN

## Summary

Here are the **5 TGN module types** and what we're using in MM-TGN:

| Module Type | Available Options | Our Choice | Notes |
|-------------|------------------|------------|-------|
| **1. Embedding Module** | `graph_attention`, `graph_sum`, `identity`, `time` | `graph_attention` | Standard from TGN paper |
| **2. Memory Updater** | `gru`, `rnn`, `lstm`, `transformer` | `gru` | Default, stable choice |
| **3. Message Function** | `mlp`, `identity` | `mlp` | Processes raw messages |
| **4. Message Aggregator** | `last`, `mean` | `last` | Keep most recent message |
| **5. Memory Module** | Enabled/Disabled | `enabled` | Required for TGN |

---

## Detailed Configuration

### 1. Embedding Module: `graph_attention`

**Options:**
- `graph_attention`: Multi-head temporal graph attention (default) ✅ **We use this**
- `graph_sum`: Sum-based aggregation (simpler)
- `identity`: Just returns memory state
- `time`: Time-only embeddings

**Our Settings:**
- Type: `graph_attention`
- Layers: 2
- Heads: 2
- Neighbors: 15

---

### 2. Memory Updater: `gru`

**Options:**
- `gru`: Gated Recurrent Unit (default) ✅ **We use this**
- `rnn`: Vanilla RNN
- `lstm`: Long Short-Term Memory
- `transformer`: Transformer-based

**Our Settings:**
- Type: `gru`
- Dimension: 172 (matches embedding_dim)

---

### 3. Message Function: `mlp`

**Options:**
- `mlp`: 2-layer MLP that processes raw messages ✅ **We use this**
- `identity`: Pass through raw messages unchanged

**Our Settings:**
- Type: `mlp`
- Input dim: Raw message dimension (varies)
- Output dim: 100 (message_dimension)

---

### 4. Message Aggregator: `last`

**Options:**
- `last`: Keep only the most recent message per node ✅ **We use this**
- `mean`: Average all messages per node

**Our Settings:**
- Type: `last`

---

### 5. Memory Module: `enabled`

**Options:**
- `enabled`: Use TGN memory mechanism ✅ **We use this**
- `disabled`: No memory (static embeddings only)

**Our Settings:**
- Enabled: `True`
- Memory dimension: 172
- Update at start: `True` (update memory before computing embeddings)

---

## Complete Configuration Summary

```python
# From our training script defaults:

TGN_MODULES = {
    "embedding_module": "graph_attention",  # --embedding-module
    "memory_updater": "gru",                # Hardcoded default
    "message_function": "mlp",              # Hardcoded default
    "message_aggregator": "last",           # Hardcoded default
    "use_memory": True,                     # --use-memory (default)
}

HYPERPARAMETERS = {
    "embedding_dim": 172,
    "memory_dim": 172,
    "message_dim": 100,
    "n_layers": 2,
    "n_heads": 2,
    "n_neighbors": 15,
    "dropout": 0.1,
}
```

---

## Code Locations

- **Embedding Module**: `modules/embedding.py` → `get_embedding_module()`
- **Memory Updater**: `modules/memory_updater.py` → `get_memory_updater()`
- **Message Function**: `modules/message_function.py` → `get_message_function()`
- **Message Aggregator**: `modules/message_aggregator.py` → `get_message_aggregator()`
- **Memory Module**: `modules/memory.py` → `Memory` class

---

## For Baseline Comparison

These are the **standard TGN settings** from the original paper. For fair comparison:
- Use the same embedding module (`graph_attention`)
- Use the same memory updater (`gru`)
- Use the same message aggregator (`last`)

The key difference in MM-TGN is:
- **Hybrid node features** (learnable users + projected multimodal items)
- **Multimodal fusion** (MLP/FiLM/Gated for text+image)

---

## Questions?

All module types are hardcoded defaults in `mmtgn.py`. They can be changed but we use the standard TGN paper settings for reproducibility.

