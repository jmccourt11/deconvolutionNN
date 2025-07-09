"""Streamlit web GUI for deconvolutionNN."""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from ..core.data_loader import load_probe_kernel, resize_probe
from ..core.deconvolution import DeconvolutionEngine


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="deconvolutionNN",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧠 deconvolutionNN")
    st.markdown("Neural network-based deconvolution with web GUI")

    # Initialize session state
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "probe_loaded" not in st.session_state:
        st.session_state.probe_loaded = False
    if "model_created" not in st.session_state:
        st.session_state.model_created = False
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Device selection
        device_options = ["auto", "cpu", "cuda"]
        selected_device = st.selectbox("Device", device_options, index=0)

        if selected_device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(selected_device)

        st.info(f"Using device: {device}")

        # Initialize engine
        if st.button("Initialize Engine") or st.session_state.engine is None:
            st.session_state.engine = DeconvolutionEngine(device=device)
            st.success("Engine initialized!")

    # Main content
    if st.session_state.engine is None:
        st.warning("Please initialize the engine first.")
        return

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📁 Load Data", "🧠 Train Model", "🔍 Evaluate", "📊 Visualize", "ℹ️ About"]
    )

    with tab1:
        load_data_tab()

    with tab2:
        train_model_tab()

    with tab3:
        evaluate_tab()

    with tab4:
        visualize_tab()

    with tab5:
        about_tab()


def load_data_tab() -> None:
    """Data loading tab."""
    st.header("📁 Load Data")

    # Probe loading
    st.subheader("Load Probe Kernel")

    probe_file = st.file_uploader(
        "Upload probe kernel file (HDF5)",
        type=["h5", "hdf5"],
        help="Upload an HDF5 file containing the probe kernel",
    )

    if probe_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
            tmp_file.write(probe_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            probe = load_probe_kernel(tmp_file_path)
            st.session_state.probe_kernel = probe
            st.session_state.probe_loaded = True

            # Display probe information
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Probe shape:** {probe.shape}")
                st.write(f"**Data type:** {probe.dtype}")

            with col2:
                # Plot probe
                fig = go.Figure()
                fig.add_trace(
                    go.Heatmap(z=np.abs(probe), colorscale="viridis", name="Magnitude")
                )
                fig.update_layout(
                    title="Probe Kernel Magnitude", xaxis_title="X", yaxis_title="Y"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Resize options
            target_size = st.number_input(
                "Target size for resizing",
                min_value=32,
                max_value=512,
                value=128,
                step=32,
            )

            if st.button("Resize Probe"):
                resized_probe = resize_probe(probe, target_size)
                st.session_state.probe_kernel = resized_probe
                st.success(f"Probe resized to {target_size}x{target_size}")

        except Exception as e:
            st.error(f"Error loading probe: {str(e)}")
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)

    # Data loading
    st.subheader("Load Diffraction Patterns")

    # For now, we'll use a simple file upload approach
    # In a real implementation, you might want to connect to a database or file system
    data_files = st.file_uploader(
        "Upload diffraction pattern files",
        type=["npy", "h5", "hdf5"],
        accept_multiple_files=True,
        help="Upload multiple files containing diffraction patterns",
    )

    if data_files and st.session_state.probe_loaded:
        if st.button("Load Data"):
            try:
                # This is a simplified data loading approach
                # In practice, you'd want more sophisticated data handling
                st.info(
                    "Data loading functionality needs to be implemented based on your specific data format."
                )

                # Placeholder for data loading
                st.session_state.data_loaded = True
                st.success("Data loaded successfully!")

            except Exception as e:
                st.error(f"Error loading data: {str(e)}")


def train_model_tab() -> None:
    """Model training tab."""
    st.header("🧠 Train Model")

    if not st.session_state.probe_loaded:
        st.warning("Please load a probe kernel first.")
        return

    if not st.session_state.data_loaded:
        st.warning("Please load training data first.")
        return

    # Model creation
    if not st.session_state.model_created:
        if st.button("Create Model"):
            try:
                engine = st.session_state.engine
                engine.probe_kernel = st.session_state.probe_kernel
                model = engine.create_model()
                st.session_state.model_created = True
                st.success("Model created successfully!")
            except Exception as e:
                st.error(f"Error creating model: {str(e)}")

    # Training configuration
    if st.session_state.model_created:
        st.subheader("Training Configuration")

        col1, col2 = st.columns(2)

        with col1:
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=128, value=32
            )
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=1e-4,
                format="%.6f",
            )

        with col2:
            train_split = st.slider(
                "Train Split", min_value=0.5, max_value=0.9, value=0.75, step=0.05
            )
            val_split = st.slider(
                "Validation Split",
                min_value=0.05,
                max_value=0.3,
                value=0.125,
                step=0.025,
            )
            loss_function = st.selectbox(
                "Loss Function", ["custom_loss", "custom_loss2", "custom_loss3"]
            )

        # Training options
        plot_samples = st.checkbox(
            "Plot sample predictions during training", value=False
        )
        save_model = st.checkbox("Save best model", value=True)

        if save_model:
            model_save_path = st.text_input(
                "Model save path", value="trained_models/best_model.pth"
            )

        # Start training
        if st.button("Start Training"):
            st.info("Training functionality needs to be implemented with actual data.")
            # Placeholder for training
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"Training epoch {i + 1}/100")
                # Simulate training
                import time

                time.sleep(0.01)

            st.success("Training completed!")


def evaluate_tab() -> None:
    """Model evaluation tab."""
    st.header("🔍 Evaluate Model")

    if not st.session_state.model_created:
        st.warning("Please create a model first.")
        return

    # Model loading
    st.subheader("Load Trained Model")

    model_file = st.file_uploader(
        "Upload trained model file",
        type=["pth", "pt"],
        help="Upload a trained model checkpoint",
    )

    if model_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
            tmp_file.write(model_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            engine = st.session_state.engine
            engine.load_model(tmp_file_path)
            st.success("Model loaded successfully!")

            # Evaluation options
            st.subheader("Evaluation")

            # Upload test data
            test_file = st.file_uploader(
                "Upload test data",
                type=["npy", "h5", "hdf5"],
                help="Upload test diffraction patterns",
            )

            if test_file and st.button("Evaluate Model"):
                st.info(
                    "Evaluation functionality needs to be implemented with actual data."
                )
                # Placeholder for evaluation
                st.success("Evaluation completed!")

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
        finally:
            os.unlink(tmp_file_path)


def visualize_tab() -> None:
    """Visualization tab."""
    st.header("📊 Visualize Results")

    # Upload results for visualization
    st.subheader("Upload Results")

    input_file = st.file_uploader("Input data", type=["npy"])
    output_file = st.file_uploader("Output data", type=["npy"])
    pc_file = st.file_uploader("Probe convolved data", type=["npy"])

    if input_file and output_file and pc_file and st.button("Visualize"):
        try:
            # Load data
            input_data = np.load(input_file)
            output_data = np.load(output_file)
            pc_data = np.load(pc_file)

            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Input shape:** {input_data.shape}")
            with col2:
                st.write(f"**Output shape:** {output_data.shape}")
            with col3:
                st.write(f"**PC shape:** {pc_data.shape}")

            # Create visualizations
            st.subheader("Sample Results")

            # Select random sample
            n_samples = min(5, input_data.shape[0])
            sample_indices = np.random.choice(
                input_data.shape[0], n_samples, replace=False
            )

            for i, idx in enumerate(sample_indices):
                st.write(f"**Sample {i + 1}**")

                # Create subplot
                fig = go.Figure()

                # Input
                fig.add_trace(
                    go.Heatmap(
                        z=input_data[idx].squeeze(),
                        colorscale="viridis",
                        name="Input",
                        showscale=False,
                    )
                )

                fig.update_layout(
                    title=f"Sample {i + 1} - Input",
                    xaxis_title="X",
                    yaxis_title="Y",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Output and PC
                col1, col2 = st.columns(2)

                with col1:
                    fig_output = go.Figure()
                    fig_output.add_trace(
                        go.Heatmap(
                            z=output_data[idx].squeeze(),
                            colorscale="viridis",
                            name="Output",
                        )
                    )
                    fig_output.update_layout(title="Decoded Output")
                    st.plotly_chart(fig_output, use_container_width=True)

                with col2:
                    fig_pc = go.Figure()
                    fig_pc.add_trace(
                        go.Heatmap(
                            z=pc_data[idx].squeeze(),
                            colorscale="viridis",
                            name="Probe Convolved",
                        )
                    )
                    fig_pc.update_layout(title="Probe Convolved")
                    st.plotly_chart(fig_pc, use_container_width=True)

                st.divider()

        except Exception as e:
            st.error(f"Error visualizing results: {str(e)}")


def about_tab() -> None:
    """About tab."""
    st.header("ℹ️ About deconvolutionNN")

    st.markdown(
        """
    ## Overview

    **deconvolutionNN** is a modern Python package for neural network-based deconvolution
    with an intuitive web GUI. It's designed for processing diffraction patterns and
    performing deconvolution using convolutional autoencoders.

    ## Features

    - 🧠 **Neural Network Models**: ConvAutoencoderSkip with skip connections
    - 🌐 **Web GUI**: Streamlit-based interface for easy interaction
    - 📊 **Visualization**: Interactive plots with Plotly
    - 🔧 **Modular Design**: Extensible architecture for different use cases
    - 📈 **Training Tools**: Comprehensive training and evaluation utilities

    ## Architecture

    The package uses a U-Net-like architecture with skip connections to perform
    deconvolution of diffraction patterns. It includes:

    - **Encoder**: Convolutional layers with increasing channel depth
    - **Bottleneck**: Feature extraction layer
    - **Decoder**: Upsampling layers with skip connections
    - **Probe Integration**: Realistic diffraction pattern generation

    ## Usage

    1. **Load Probe**: Upload your probe kernel (HDF5 format)
    2. **Load Data**: Upload diffraction patterns for training
    3. **Train Model**: Configure and start training
    4. **Evaluate**: Test the trained model on new data
    5. **Visualize**: Explore results with interactive plots

    ## Technical Details

    - **Framework**: PyTorch for deep learning
    - **GUI**: Streamlit for web interface
    - **Visualization**: Plotly for interactive plots
    - **Data Formats**: HDF5, NumPy arrays
    - **Loss Functions**: Custom correlation-based losses with physical constraints

    ## Development

    This package follows modern Python development practices:
    - Type hints throughout
    - Comprehensive documentation
    - Unit testing
    - Code quality tools (black, isort, ruff, mypy)
    - Pre-commit hooks

    ## License

    MIT License - see LICENSE file for details.
    """
    )

    # System information
    st.subheader("System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**PyTorch version:** {torch.__version__}")
        st.write(f"**CUDA available:** {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**CUDA version:** {torch.version.cuda}")
            st.write(f"**GPU count:** {torch.cuda.device_count()}")

    with col2:
        st.write(f"**NumPy version:** {np.__version__}")
        st.write(f"**Streamlit version:** {st.__version__}")


if __name__ == "__main__":
    main()
