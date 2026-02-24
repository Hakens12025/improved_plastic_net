# Improved Plastic Net

A neuroplasticity-inspired neural network with dynamic topology adjustment, minimum action principle, and continuous evolution mechanism.

## Versions

| Version | Description |
|---------|-------------|
| [v1.0.2](./v1.0.2/) | Baseline version |
| [v1.0.3](./v1.0.3/) | **Optimized** - 6-12x faster training |

## Quick Start

### v1.0.3 (Recommended)

```bash
cd v1.0.3

# Train model
python experiments/v1_0_3_mnist_baseline.py

# Or continuous inference mode
python main.py

# Load trained model for inference
MODEL_LOAD_PATH=model.pth python main.py
```

### v1.0.2

```bash
cd v1.0.2
python run_with_config.py
```

## Project Structure

```
improved_plastic_net/
├── README.md
├── v1/              # Documentation and shared configs
├── v1.0.2/          # Baseline version
└── v1.0.3/          # Optimized version (6-12x speed improvement)
```

## Key Features

- **Dynamic Topology**: Pruning and growing connections during training
- **Continuous Mode**: Real-time inference after training
- **Neuroplasticity**: Credit score-based connection management
- **Performance Optimized**: Vectorized operations, pre-computed topology

## Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision

## License

For learning and research purposes only.
