"""
Continuous run entrypoint for v1.0.3.

Behavior:
- Keep the network running after launch.
- When data arrives, run inference once, print output once, then train once.
- When idle, perform occasional pruning/regrowth.
- Data source: FashionMNIST files or a file-queue directory.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from config import (
    CONTINUOUS_BATCH_SIZE,
    CONTINUOUS_DATA_MODE,
    CONTINUOUS_INPUT_DIM,
    CONTINUOUS_OUTPUT_DIM,
    DEVICE,
    ENABLE_IDLE_PLASTICITY,
    FILE_INPUT_DIR,
    FILE_POLL_INTERVAL_SEC,
    FILE_PROCESSED_DIR,
    FASHION_MNIST_AUGMENT,
    FASHION_MNIST_DOWNLOAD,
    FASHION_MNIST_ROOT,
    FASHION_MNIST_TRAIN,
    IDLE_PLASTICITY_GRACE_SEC,
    IDLE_PLASTICITY_INTERVAL_SEC,
    INITIAL_SPARSITY,
    ITERATIONS,
    LEARNING_RATE,
    NUM_NEURONS,
    PLASTICITY_INTERVAL,
    PROTECTION_PERIOD
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


def _iter_file_queue(
    input_dir: Path,
    processed_dir: Path,
    poll_interval: float,
    input_dim: int,
    stop_event: threading.Event
) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield samples from a file-queue directory."""
    input_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    while not stop_event.is_set():
        files = sorted([p for p in input_dir.iterdir() if p.is_file()])
        if not files:
            time.sleep(poll_interval)
            continue
        for path in files:
            sample = None
            try:
                sample = _load_sample_from_file(path, input_dim)
            except Exception as exc:
                print(json.dumps({"file": str(path.name), "error": str(exc)}, ensure_ascii=True))
            finally:
                target = processed_dir / path.name
                try:
                    path.replace(target)
                except OSError:
                    pass
            if sample is not None:
                yield sample


def _build_fashion_mnist_loader() -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    """Create a FashionMNIST DataLoader."""
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    if FASHION_MNIST_AUGMENT:
        train_transform = transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset = torchvision.datasets.FashionMNIST(
        root=FASHION_MNIST_ROOT,
        train=FASHION_MNIST_TRAIN,
        download=FASHION_MNIST_DOWNLOAD,
        transform=train_transform
    )

    return DataLoader(
        dataset,
        batch_size=CONTINUOUS_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


class ContinuousRunner:
    """Run continuous inference + online training."""

    def __init__(self) -> None:
        self.device = resolve_device(DEVICE)
        self.model = OptimizedPlasticNet(
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
                missing, unexpected = self.model.load_state_dict(state, strict=False)
                with torch.no_grad():
                    self.model.weights.data = self.model.weights.data.abs()
                    if hasattr(self.model, "inhibitory_mask"):
                        self.model.weights.data[self.model.inhibitory_mask, :] *= -1.0
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
        self.engine = OptimizedEvolutionEngine(
            model=self.model,
            device=self.device,
            lr=LEARNING_RATE,
            plasticity_interval=PLASTICITY_INTERVAL
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_input_time = time.time()

    def _idle_worker(self) -> None:
        """Run idle plasticity at a fixed interval if no recent data."""
        while not self._stop_event.is_set():
            time.sleep(IDLE_PLASTICITY_INTERVAL_SEC)
            if (time.time() - self._last_input_time) < IDLE_PLASTICITY_GRACE_SEC:
                continue
            with self._lock:
                stats = self.engine.apply_plasticity()
            print(
                json.dumps(
                    {
                        "event": "idle_plasticity",
                        "pruned": stats["pruned"],
                        "added": stats["added"],
                        "plasticity_time": stats["plasticity_time"]
                    },
                    ensure_ascii=True
                )
            )

    def _handle_batch(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """Run one inference + one training step and print output."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        self._last_input_time = time.time()
        with self._lock:
            stats = self.engine.train_step(features, labels, apply_plasticity=True)

        outputs = stats["outputs"]
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        if pred.numel() == 1:
            pred_out = int(pred.item())
            conf_out = float(conf.item())
        else:
            pred_out = [int(x) for x in pred.tolist()]
            conf_out = [float(x) for x in conf.tolist()]

        print(
            json.dumps(
                {
                    "pred": pred_out,
                    "confidence": conf_out,
                    "loss": stats["loss"],
                    "pruned": stats["pruned"],
                    "added": stats["added"],
                    "batch_time": stats["batch_time"]
                },
                ensure_ascii=True
            )
        )

    def _run_fashion_mnist(self) -> None:
        """Stream FashionMNIST samples from disk."""
        loader = _build_fashion_mnist_loader()
        while not self._stop_event.is_set():
            for images, labels in loader:
                if self._stop_event.is_set():
                    break
                features = images.view(images.size(0), -1)
                self._handle_batch(features, labels)

    def _run_file_queue(self) -> None:
        """Process samples from a file-queue directory."""
        input_dir = Path(FILE_INPUT_DIR)
        processed_dir = Path(FILE_PROCESSED_DIR)
        for features, label in _iter_file_queue(
            input_dir=input_dir,
            processed_dir=processed_dir,
            poll_interval=FILE_POLL_INTERVAL_SEC,
            input_dim=CONTINUOUS_INPUT_DIM,
            stop_event=self._stop_event
        ):
            if self._stop_event.is_set():
                break
            self._handle_batch(features, label)

    def run(self) -> None:
        """Main loop."""
        print("Continuous mode started.")
        print(f"Data mode: {CONTINUOUS_DATA_MODE}")
        print(f"Device: {self.device}")

        idle_thread = None
        if ENABLE_IDLE_PLASTICITY:
            idle_thread = threading.Thread(target=self._idle_worker, daemon=True)
            idle_thread.start()

        try:
            if CONTINUOUS_DATA_MODE == "fashion_mnist":
                self._run_fashion_mnist()
            elif CONTINUOUS_DATA_MODE == "file_queue":
                self._run_file_queue()
            else:
                raise ValueError("Unsupported CONTINUOUS_DATA_MODE")
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_event.set()
            if idle_thread is not None:
                idle_thread.join(timeout=1.0)
            print("Continuous mode stopped.")


def main() -> None:
    """Entry point."""
    runner = ContinuousRunner()
    runner.run()


if __name__ == "__main__":
    main()
