"""Validation tests to ensure the testing infrastructure is properly set up."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np


class TestSetupValidation:
    """Validation tests for the testing infrastructure."""

    def test_pytest_is_installed(self):
        """Verify pytest is installed and accessible."""
        assert "pytest" in sys.modules

    def test_coverage_is_installed(self):
        """Verify pytest-cov is installed."""
        try:
            import pytest_cov
            assert pytest_cov is not None
        except ImportError:
            pytest.fail("pytest-cov is not installed")

    def test_mock_is_installed(self):
        """Verify pytest-mock is installed."""
        try:
            import pytest_mock
            assert pytest_mock is not None
        except ImportError:
            pytest.fail("pytest-mock is not installed")

    def test_project_structure(self):
        """Verify the project structure is correct."""
        project_root = Path(__file__).parent.parent
        assert project_root.exists()
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "__init__.py").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "unit" / "__init__.py").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "tests" / "integration" / "__init__.py").exists()
        assert (project_root / "tests" / "conftest.py").exists()

    def test_temp_dir_fixture(self, temp_dir):
        """Test the temp_dir fixture works correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test we can write to the temp directory
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

    def test_mock_config_fixture(self, mock_config):
        """Test the mock_config fixture provides expected structure."""
        assert isinstance(mock_config, dict)
        assert "model" in mock_config
        assert "training" in mock_config
        assert "data" in mock_config
        assert mock_config["model"]["name"] == "unireplknet"

    def test_sample_tensors(self, sample_image_tensor, sample_batch, 
                           sample_audio_tensor, sample_video_tensor, 
                           sample_point_cloud):
        """Test all sample tensor fixtures have correct shapes."""
        # Image tensor
        assert sample_image_tensor.shape == (1, 3, 224, 224)
        assert isinstance(sample_image_tensor, torch.Tensor)
        
        # Batch
        assert sample_batch["images"].shape == (4, 3, 224, 224)
        assert sample_batch["labels"].shape == (4,)
        assert sample_batch["masks"].shape == (4, 224, 224)
        
        # Audio tensor
        assert sample_audio_tensor.shape == (1, 16000)
        
        # Video tensor
        assert sample_video_tensor.shape == (1, 8, 3, 224, 224)
        
        # Point cloud
        assert sample_point_cloud.shape == (1024, 3)

    def test_mock_model_fixture(self, mock_model):
        """Test the mock_model fixture behaves correctly."""
        # Test forward pass
        output = mock_model.forward(torch.randn(1, 3, 224, 224))
        assert output.shape == (1, 10)
        
        # Test parameters
        params = list(mock_model.parameters())
        assert len(params) > 0
        
        # Test state dict
        state_dict = mock_model.state_dict()
        assert "layer1.weight" in state_dict

    def test_device_fixture(self, device):
        """Test the device fixture returns a valid device."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]

    def test_random_seed_fixture(self, random_seed):
        """Test the random seed fixture ensures reproducibility."""
        assert random_seed == 42
        
        # Test reproducibility
        tensor1 = torch.randn(5)
        np_array1 = np.random.randn(5)
        
        # Reset seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        tensor2 = torch.randn(5)
        np_array2 = np.random.randn(5)
        
        assert torch.allclose(tensor1, tensor2)
        assert np.allclose(np_array1, np_array2)

    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit test marker works."""
        assert True

    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration test marker works."""
        assert True

    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow test marker works."""
        assert True

    def test_checkpoint_fixtures(self, checkpoint_path, sample_checkpoint):
        """Test checkpoint-related fixtures."""
        assert checkpoint_path.parent.exists()
        assert sample_checkpoint.exists()
        
        # Load and verify checkpoint
        checkpoint = torch.load(sample_checkpoint, map_location="cpu")
        assert checkpoint["epoch"] == 10
        assert checkpoint["loss"] == 0.5
        assert checkpoint["accuracy"] == 0.85

    def test_env_setup_fixture(self, env_setup):
        """Test environment setup fixture."""
        import os
        assert os.environ.get("OMP_NUM_THREADS") == "1"

    def test_capture_logs_fixture(self, capture_logs):
        """Test log capturing fixture."""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug("Test debug message")
        logger.info("Test info message")
        
        assert "Test debug message" in capture_logs.text
        assert "Test info message" in capture_logs.text


@pytest.mark.parametrize("module_name", [
    "Audio", "Image", "Point", "Video", "detection", "segmentation"
])
def test_module_directories_exist(module_name):
    """Test that all module directories exist."""
    project_root = Path(__file__).parent.parent
    module_path = project_root / module_name
    assert module_path.exists(), f"Module directory {module_name} does not exist"


def test_pyproject_toml_exists():
    """Test that pyproject.toml exists and has correct structure."""
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    assert pyproject_path.exists()
    
    # Basic validation of content
    content = pyproject_path.read_text()
    assert "[tool.poetry]" in content
    assert "[tool.pytest.ini_options]" in content
    assert "[tool.coverage.run]" in content
    assert "pytest" in content
    assert "pytest-cov" in content
    assert "pytest-mock" in content