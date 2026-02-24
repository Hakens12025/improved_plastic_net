# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

v1.0.3 optimized neuroplasticity network - a neural network inspired by brain plasticity that performs dynamic topology adjustment during training. Primary use case: FashionMNIST classification.

## Commands

```bash
# Continuous mode (FashionMNIST streaming or file queue)
python main.py

# One-shot ingest: put files in ./todo, run, processed files move to ./done
python insert.py

# Run MNIST experiment
python experiments/v1_0_3_mnist_baseline.py

# Run with custom config
python run_with_config.py
```

## Architecture

**Core Components (v1.0.3 optimized versions):**

- `models/v1_0_3_plastic_net.py:21` - `OptimizedPlasticNet`: Main network with dynamic adjacency mask (`adj_mask`) for topology. Forward pass computes `h = relu(h @ (weights * adj_mask))`. Calls `apply_neuroplasticity()` every `plasticity_interval` batches.

- `models/v1_0_3_topology_manager.py` - `OptimizedTopologyManager`: Precomputes valid connections at init. `get_connection_candidates_fast()` returns neuron pairs within `[min_distance, max_distance]`. O(1) lookup vs original O(C×N) BFS.

- `models/v1_0_3_connection_manager.py` - `OptimizedConnectionManager`: Tracks connection ages and coactivation via EMA (exponential moving average) instead of history windows. Vectorized `can_be_pruned_vectorized()` for batch pruning decisions.

- `training/v1_0_3_engine.py:19` - `OptimizedEvolutionEngine`: Training loop with AdamW, label smoothing (0.1), CosineAnnealingWarmRestarts scheduler, and optional AMP (automatic mixed precision). `train_step()` returns dict with loss, outputs, pruned, added, batch_time.

**Key Data Structures:**
- `adj_mask`: Upper triangular binary mask (N×N) indicating active connections
- `credit_score`: Float32/half tensor tracking connection utility scores
- `connection_age`: int16 tensor tracking connection age for protection period
- `pre_trace`/`post_trace`: LTP/LTD plasticity traces per neuron
- `firing_rate_ema`: EMA of neuron firing rates for homeostasis

**Plasticity Cycle (`apply_neuroplasticity`):**
1. Update connection ages (vectorized)
2. Check activity bounds (skip if avg rate outside [activity_min, activity_max])
3. Grow connections via `_grow_connections_vectorized()`
4. Prune connections via `_prune_connections_balanced()`
5. Update topology manager if changes > threshold

## Configuration

Edit `config.py` for all parameters:
- `NUM_NEURONS`: Network size (default 1500)
- `EPOCHS`: Training epochs (default 15)
- `BATCH_SIZE`: Adjusted dynamically by neuron count
- `PLASTICITY_INTERVAL`: Batches between topology updates
- `INITIAL_SPARSITY`: Starting sparse rate (0.75 = 75% connections pruned)
- `PROTECTION_PERIOD`: Steps before a new connection can be pruned

Environment variables:
- `MODEL_LOAD_PATH`: Checkpoint to resume from
- `CONTINUOUS_DATA_MODE`: "fashion_mnist" (default) or "file_queue"

## Input/Output Formats

File queue mode supports:
- JSON: `{"x": [...784 values...], "y": 3}`
- PT/PTH: `dict` with keys `x,y` or `features,label` (or tuple `(features, label)`)
- NPZ: arrays with `x` and `y`/`label`

Continuous mode outputs JSON to stdout:
```json
{"pred": 3, "confidence": 0.92, "loss": 0.15, "pruned": 5, "added": 12, "batch_time": 0.023}
```
