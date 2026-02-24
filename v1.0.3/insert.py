"""
One-shot file ingest for v1.0.3.

Usage:
1) Put sample files into ./todo
2) Run: python insert.py
3) Each file triggers one inference + one training step.
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import (
    CONTINUOUS_INPUT_DIM,
    CONTINUOUS_OUTPUT_DIM,
    DEVICE,
    INITIAL_SPARSITY,
    ITERATIONS,
    LEARNING_RATE,
    NUM_NEURONS,
    PLASTICITY_INTERVAL,
    PROTECTION_PERIOD,
    TODO_DIR,
    DONE_DIR
)
from models.v1_0_3_init import OptimizedPlasticNet
from training.v1_0_3_init import OptimizedEvolutionEngine


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device from config."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _coerce_features(features: torch.Tensor, input_dim: int) -> torch.Tensor:
    """Normalize features to a flat tensor of input_dim."""
    flat = features.view(-1).to(torch.float32)
    if flat.numel() != input_dim:
        raise ValueError(f"Expected {input_dim} features, got {flat.numel()}")
    return flat


def _load_sample_from_file(path: Path, input_dim: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Load one sample from a file (json/pt/npz)."""
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, (tuple, list)) and len(payload) == 2:
            features, label = payload
        elif isinstance(payload, dict):
            label = payload.get("y", payload.get("label"))
            features = payload.get("x", payload.get("features"))
        else:
            raise ValueError("Unsupported .pt payload format")
        features = torch.as_tensor(features)
        label = int(torch.as_tensor(label).item())
    elif suffix == ".npz":
        import numpy as np
        payload = np.load(path)
        if "x" not in payload or ("y" not in payload and "label" not in payload):
            raise ValueError("NPZ must include x and y (or label)")
        features = torch.as_tensor(payload["x"])
        label = int(torch.as_tensor(payload.get("y", payload.get("label"))).item())
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "x" not in payload or ("y" not in payload and "label" not in payload):
            raise ValueError("JSON must include x and y (or label)")
        features = torch.as_tensor(payload["x"])
        label = int(payload.get("y", payload.get("label")))
    else:
        return None

    x = _coerce_features(features, input_dim)
    y = torch.tensor(label, dtype=torch.long)
    return x, y


def main() -> None:
    """Process all files in TODO_DIR once."""
    base_dir = Path(__file__).resolve().parent.parent
    todo_dir = Path(TODO_DIR)
    done_dir = Path(DONE_DIR)
    if not todo_dir.is_absolute():
        todo_dir = base_dir / todo_dir
    if not done_dir.is_absolute():
        done_dir = base_dir / done_dir
    todo_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(DEVICE)
    model = OptimizedPlasticNet(
        num_neurons=NUM_NEURONS,
        input_dim=CONTINUOUS_INPUT_DIM,
        output_dim=CONTINUOUS_OUTPUT_DIM,
        iterations=ITERATIONS,
        initial_sparsity=INITIAL_SPARSITY,
        protection_period=PROTECTION_PERIOD
    )
    load_path = os.getenv("MODEL_LOAD_PATH", "").strip()
    if load_path:
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location="cpu")
            state = checkpoint.get("model_state_dict", checkpoint)
            missing, unexpected = model.load_state_dict(state, strict=False)
            with torch.no_grad():
                model.weights.data = model.weights.data.abs()
                if hasattr(model, "inhibitory_mask"):
                    model.weights.data[model.inhibitory_mask, :] *= -1.0
            if missing or unexpected:
                print(
                    json.dumps(
                        {
                            "event": "checkpoint_loaded",
                            "missing_keys": missing,
                            "unexpected_keys": unexpected
                        },
                        ensure_ascii=True
                    )
                )
        else:
            print(json.dumps({"event": "checkpoint_missing", "path": load_path}, ensure_ascii=True))
    engine = OptimizedEvolutionEngine(
        model=model,
        device=device,
        lr=LEARNING_RATE,
        plasticity_interval=PLASTICITY_INTERVAL
    )

    files = sorted([p for p in todo_dir.iterdir() if p.is_file()])
    if not files:
        print(json.dumps({"info": "no files in todo"}, ensure_ascii=True))
        return

    processed = 0
    for path in files:
        sample = None
        try:
            sample = _load_sample_from_file(path, CONTINUOUS_INPUT_DIM)
        except Exception as exc:
            print(json.dumps({"file": path.name, "error": str(exc)}, ensure_ascii=True))
        finally:
            try:
                path.replace(done_dir / path.name)
            except OSError:
                pass

        if sample is None:
            continue

        features, label = sample
        stats = engine.train_step(features, label, apply_plasticity=True)

        outputs = stats["outputs"]
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        print(
            json.dumps(
                {
                    "file": path.name,
                    "pred": int(pred.item()),
                    "confidence": float(conf.item()),
                    "loss": stats["loss"],
                    "pruned": stats["pruned"],
                    "added": stats["added"],
                    "batch_time": stats["batch_time"]
                },
                ensure_ascii=True
            )
        )
        processed += 1

    print(json.dumps({"processed": processed}, ensure_ascii=True))


if __name__ == "__main__":
    main()
