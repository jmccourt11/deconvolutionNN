"""Basic import tests for deconvolutionNN."""

import numpy as np
import pytest
import torch


def test_import_deconvolution_engine():
    """Test that DeconvolutionEngine can be imported."""
    from deconvolutionNN import DeconvolutionEngine

    assert DeconvolutionEngine is not None


def test_import_conv_autoencoder():
    """Test that ConvAutoencoderSkip can be imported."""
    from deconvolutionNN.models import ConvAutoencoderSkip

    assert ConvAutoencoderSkip is not None


def test_import_losses():
    """Test that loss functions can be imported."""
    from deconvolutionNN.core.losses import (
        custom_loss,
        custom_loss2,
        custom_loss3,
        pearson_loss,
    )

    assert all([pearson_loss, custom_loss, custom_loss2, custom_loss3])


def test_import_data_loader():
    """Test that data loader functions can be imported."""
    from deconvolutionNN.core.data_loader import (
        create_data_loaders,
        load_probe_kernel,
        resize_probe,
    )

    assert all([load_probe_kernel, resize_probe, create_data_loaders])


def test_import_trainer():
    """Test that trainer can be imported."""
    from deconvolutionNN.core.train import DeconvolutionTrainer

    assert DeconvolutionTrainer is not None


def test_create_engine():
    """Test that DeconvolutionEngine can be instantiated."""
    from deconvolutionNN import DeconvolutionEngine

    engine = DeconvolutionEngine()
    assert engine is not None
    assert hasattr(engine, "device")
    assert hasattr(engine, "model")
    assert hasattr(engine, "trainer")


def test_create_model():
    """Test that ConvAutoencoderSkip can be instantiated."""
    from deconvolutionNN.models import ConvAutoencoderSkip

    # Create dummy probe kernel
    probe = np.random.rand(64, 64) + 1j * np.random.rand(64, 64)

    model = ConvAutoencoderSkip(probe)
    assert model is not None
    assert hasattr(model, "forward")


def test_model_forward_pass():
    """Test that model can perform forward pass."""
    from deconvolutionNN.models import ConvAutoencoderSkip

    # Create dummy probe kernel
    probe = np.random.rand(64, 64) + 1j * np.random.rand(64, 64)

    model = ConvAutoencoderSkip(probe)

    # Create dummy input
    x = torch.randn(1, 1, 64, 64)

    # Forward pass
    decoded, probe_convolved = model(x)

    assert decoded.shape == (1, 1, 64, 64)
    assert probe_convolved.shape == (1, 1, 64, 64)


def test_loss_functions():
    """Test that loss functions can be called."""
    from deconvolutionNN.core.losses import custom_loss, pearson_loss

    # Create dummy tensors
    output = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    decoded = torch.randn(2, 1, 64, 64)

    # Test pearson loss
    loss1 = pearson_loss(output, target)
    assert isinstance(loss1, torch.Tensor)
    assert loss1.dim() == 0  # Scalar

    # Test custom loss
    loss2 = custom_loss(output, target, decoded)
    assert isinstance(loss2, torch.Tensor)
    assert loss2.dim() == 0  # Scalar


def test_data_loader_functions():
    """Test that data loader functions can be called."""
    from deconvolutionNN.core.data_loader import create_center_mask

    # Test center mask creation
    mask = create_center_mask((64, 64), center_radius=10)
    assert mask.shape == (64, 64)
    assert mask.dtype == bool


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
