"""deconvolutionNN - Neural network-based deconvolution package with web GUI."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core imports
from .core import DeconvolutionEngine
from .core.data_loader import (
    create_data_loaders,
    load_diffraction_patterns,
    load_probe_kernel,
    preprocess_diffraction_patterns,
    resize_probe,
)
from .core.losses import custom_loss, custom_loss2, custom_loss3, pearson_loss
from .core.train import DeconvolutionTrainer, azimuthal_average, evaluate_model
from .models import ConvAutoencoderSkip

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "DeconvolutionEngine",
    "ConvAutoencoderSkip",
    "pearson_loss",
    "custom_loss",
    "custom_loss2",
    "custom_loss3",
    "load_probe_kernel",
    "resize_probe",
    "load_diffraction_patterns",
    "preprocess_diffraction_patterns",
    "create_data_loaders",
    "DeconvolutionTrainer",
    "evaluate_model",
    "azimuthal_average",
]
