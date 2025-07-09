# deconvolutionNN

A modern Python package for neural network-based deconvolution with an intuitive web GUI.

## Features

- 🧠 Neural network-based deconvolution algorithms
- 🌐 Modern web-based GUI using Streamlit
- 📊 Interactive visualizations with Plotly
- 🔧 Modular and extensible architecture
- 🧪 Comprehensive testing suite
- 📚 Type hints and documentation

## Installation

### Using conda (recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/deconvolutionNN.git
cd deconvolutionNN

# Create and activate conda environment
conda env create -f environment.yml
conda activate deconvolutionNN

# Install the package in development mode
pip install -e .
```

### Using pip

```bash
pip install deconvolutionNN
```

## Quick Start

### Command Line Interface

```python
from deconvolutionNN import DeconvolutionEngine

# Initialize the engine
engine = DeconvolutionEngine()

# Load your data
data = engine.load_data("path/to/your/data.h5")

# Run deconvolution
result = engine.deconvolve(data)

# Save results
engine.save_results(result, "output.h5")
```

### Web GUI

Launch the web interface:

```bash
deconvolutionnn-gui
```

Or run directly with Streamlit:

```bash
streamlit run src/deconvolutionNN/web/gui.py
```

## Development

### Setting up the development environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=deconvolutionNN

# Run specific test categories
pytest -m "not slow"
pytest -m integration
```

### Code formatting and linting

```bash
# Format code
black src/
isort src/

# Lint code
ruff check src/
flake8 src/

# Type checking
mypy src/
```

### Pre-commit hooks

The project uses pre-commit hooks to ensure code quality. These run automatically on commit:

- `black` - Code formatting
- `isort` - Import sorting
- `ruff` - Fast Python linter
- `flake8` - Style guide enforcement
- `pytest` - Run tests

## Project Structure

```
deconvolutionNN/
├── src/
│   └── deconvolutionNN/
│       ├── __init__.py
│       ├── core/           # Core deconvolution functionality
│       ├── models/         # Neural network models
│       ├── utils/          # Utility functions
│       ├── web/            # Web GUI components
│       └── tests/          # Test modules
├── pyproject.toml          # Project configuration
├── environment.yml         # Conda environment
├── README.md              # This file
└── LICENSE                # MIT License
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{deconvolutionNN2024,
  title={deconvolutionNN: Neural network-based deconvolution package},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/deconvolutionNN}
}
```

## Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/deconvolutionNN/issues)
- 📖 Documentation: [Read the Docs](https://deconvolutionNN.readthedocs.io)
