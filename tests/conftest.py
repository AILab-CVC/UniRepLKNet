"""Shared pytest fixtures and configuration for UniRepLKNet tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Provide a mock configuration dictionary for testing."""
    return {
        "model": {
            "name": "unireplknet",
            "num_classes": 10,
            "input_size": 224,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 100,
            "warmup_epochs": 5,
            "weight_decay": 0.05,
        },
        "data": {
            "dataset": "test_dataset",
            "data_path": "/tmp/test_data",
            "num_workers": 4,
            "pin_memory": True,
        },
        "augmentation": {
            "mixup": 0.8,
            "cutmix": 1.0,
            "cutmix_minmax": None,
            "mixup_prob": 1.0,
            "mixup_switch_prob": 0.5,
            "mixup_mode": "batch",
        },
    }


@pytest.fixture
def sample_image_tensor() -> torch.Tensor:
    """Create a sample image tensor for testing."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Create a sample batch of data for testing."""
    batch_size = 4
    return {
        "images": torch.randn(batch_size, 3, 224, 224),
        "labels": torch.randint(0, 10, (batch_size,)),
        "masks": torch.ones(batch_size, 224, 224),
    }


@pytest.fixture
def sample_audio_tensor() -> torch.Tensor:
    """Create a sample audio tensor for testing."""
    # 1 second of audio at 16kHz sample rate
    return torch.randn(1, 16000)


@pytest.fixture
def sample_video_tensor() -> torch.Tensor:
    """Create a sample video tensor for testing."""
    # batch_size=1, num_frames=8, channels=3, height=224, width=224
    return torch.randn(1, 8, 3, 224, 224)


@pytest.fixture
def sample_point_cloud() -> torch.Tensor:
    """Create a sample point cloud tensor for testing."""
    # num_points=1024, dimensions=3 (x, y, z)
    return torch.randn(1024, 3)


@pytest.fixture
def sample_time_series() -> np.ndarray:
    """Create a sample time series data for testing."""
    # 100 time steps with 5 features
    return np.random.randn(100, 5)


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model for testing."""
    model = MagicMock()
    model.forward.return_value = torch.randn(1, 10)  # Mock output
    model.parameters.return_value = [torch.randn(10, 10)]
    model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
    return model


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Create a mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = (torch.randn(3, 224, 224), 0)
    return dataset


@pytest.fixture
def mock_dataloader(mock_dataset) -> MagicMock:
    """Create a mock dataloader for testing."""
    dataloader = MagicMock()
    dataloader.__iter__.return_value = iter([
        (torch.randn(32, 3, 224, 224), torch.randint(0, 10, (32,)))
        for _ in range(3)
    ])
    dataloader.dataset = mock_dataset
    return dataloader


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_seed() -> int:
    """Set and return a fixed random seed for reproducible tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically clean up GPU memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def checkpoint_path(temp_dir: Path) -> Path:
    """Create a path for saving checkpoints during tests."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir / "test_checkpoint.pth"


@pytest.fixture
def sample_checkpoint(checkpoint_path: Path, mock_model) -> Path:
    """Create a sample checkpoint file for testing."""
    checkpoint = {
        "epoch": 10,
        "model_state_dict": mock_model.state_dict(),
        "optimizer_state_dict": {"param_groups": [{"lr": 0.001}]},
        "loss": 0.5,
        "accuracy": 0.85,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def env_setup(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("TORCH_HOME", "/tmp/torch_cache")


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and return logs during tests."""
    with caplog.at_level("DEBUG"):
        yield caplog