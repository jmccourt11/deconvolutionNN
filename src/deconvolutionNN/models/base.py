"""Base model class for neural network models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all neural network models."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the base model.

        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def predict(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Make predictions on input data.

        Args:
            data: Input data

        Returns:
            Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            data = data.to(self.device)
            output = self.forward(data)
            return output.cpu().numpy()

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the model to file.

        Args:
            file_path: Path where to save the model
        """
        file_path = Path(file_path)
        torch.save(self.state_dict(), file_path)

    def load(self, file_path: Union[str, Path]) -> None:
        """Load the model from file.

        Args:
            file_path: Path to the model file

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        self.load_state_dict(torch.load(file_path, map_location=self.device))
