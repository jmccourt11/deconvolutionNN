"""Loss functions for deconvolution neural networks."""


import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute 1 - Pearson correlation coefficient as a loss function.

    This loss function measures the correlation between predicted and target
    diffraction patterns, encouraging the model to learn the correct
    intensity distribution.

    Args:
        output: Predicted values of shape (B, C, H, W)
        target: Target values of shape (B, C, H, W)

    Returns:
        Loss value: 1 - correlation (to minimize)
    """
    # Flatten the spatial dimensions
    output_flat = output.view(output.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Mean of each image
    output_mean = output_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)

    # Centered variables
    output_centered = output_flat - output_mean
    target_centered = target_flat - target_mean

    # Correlation
    numerator = (output_centered * target_centered).sum(dim=1)
    denominator = torch.sqrt(
        (output_centered**2).sum(dim=1) * (target_centered**2).sum(dim=1)
    )

    # Avoid division by zero
    correlation = numerator / (denominator + 1e-8)

    # Average over batch and convert to loss (1 - correlation)
    loss = 1 - correlation.mean()

    return loss


def custom_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    decoded: torch.Tensor,
    central_beam_radius: int = 64,
) -> torch.Tensor:
    """
    Custom loss function with central beam masking.

    This loss function applies a mask to de-emphasize the central beam region
    in diffraction patterns, focusing the model on the higher-q scattering
    features.

    Args:
        output: Predicted convolved output (B, C, H, W)
        target: Target diffraction pattern (B, C, H, W)
        decoded: Decoded object (B, C, H, W)
        central_beam_radius: Radius of central beam to mask out

    Returns:
        Loss value
    """
    # Create central beam mask
    h, w = output.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=output.device), torch.arange(w, device=output.device)
    )
    center_y, center_x = h // 2, w // 2
    r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create mask that de-emphasizes central beam
    beam_mask = (r > central_beam_radius).float()
    beam_mask = beam_mask.to(output.device)[None, None, :, :]

    # Apply mask to correlation loss
    conv_loss = pearson_loss(output * beam_mask, target * beam_mask)

    return conv_loss


def custom_loss2(
    output: torch.Tensor, target: torch.Tensor, decoded: torch.Tensor
) -> torch.Tensor:
    """
    Custom loss function with sparsity and sharpness constraints.

    This loss function encourages sparsity (sharp peaks) in the decoded image
    while maintaining correlation with the target diffraction pattern.

    Args:
        output: Predicted convolved output (B, C, H, W)
        target: Target diffraction pattern (B, C, H, W)
        decoded: Decoded object (B, C, H, W)

    Returns:
        Combined loss value
    """
    # Main correlation loss between convolved output and target
    conv_loss = pearson_loss(output, target)

    # Encourage sparsity (sharp peaks) in decoded image
    # Using a modified L1 loss that's less harsh on peaks
    peak_loss = torch.mean(
        torch.where(
            decoded > 0.1,  # For values above threshold
            0.1 * torch.abs(decoded),  # Small penalty for peaks
            torch.abs(decoded),  # Larger penalty for non-peak areas
        )
    )

    # Encourage local maxima to be sharp
    # Calculate local max in 3x3 neighborhoods
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    is_local_max = decoded == max_pool(decoded)
    sharpness_loss = torch.mean(
        torch.where(
            is_local_max,
            -decoded,  # Encourage higher values at peaks
            torch.zeros_like(decoded),
        )
    )

    # Combine losses
    plc = 1
    slc = plc / 2
    total_loss = conv_loss + plc * peak_loss + slc * sharpness_loss

    return total_loss


def custom_loss3(
    output: torch.Tensor, target: torch.Tensor, decoded: torch.Tensor
) -> torch.Tensor:
    """
    Advanced custom loss function with multiple physical constraints.

    This loss function incorporates multiple physical constraints:
    - Correlation with target diffraction pattern
    - Centro-symmetry in decoded image
    - Radial weighting (higher-q peaks should be weaker)
    - Sharpness at peaks
    - Background suppression

    Args:
        output: Predicted convolved output (B, C, H, W)
        target: Target diffraction pattern (B, C, H, W)
        decoded: Decoded object (B, C, H, W)

    Returns:
        Combined loss value
    """
    # Add small epsilon to prevent numerical instability
    eps = 1e-6

    # Main correlation loss between convolved output and target
    conv_loss = pearson_loss(output, target)

    # Center distance map for radial weighting
    h, w = decoded.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(h, device=decoded.device), torch.arange(w, device=decoded.device)
    )
    center_y, center_x = h // 2, w // 2
    r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + eps)

    # 1. Encourage centro-symmetry in decoded image
    flipped = torch.flip(decoded, [-2, -1])
    symmetry_loss = F.mse_loss(decoded, flipped)

    # 2. Higher-q peaks should be weaker (with numerical stability)
    radial_weight = torch.exp(-r / (h / 4))
    radial_loss = torch.mean(decoded * (1 - radial_weight)[None, None, :, :])

    # 3. Peaks should be sharp
    # Use Sobel filters for gradient calculation
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=decoded.device, dtype=decoded.dtype
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=decoded.device, dtype=decoded.dtype
    ).view(1, 1, 3, 3)

    # Calculate gradients using convolution
    pad = nn.ReplicationPad2d(1)
    decoded_pad = pad(decoded)
    dx = F.conv2d(decoded_pad, sobel_x)
    dy = F.conv2d(decoded_pad, sobel_y)

    # Detect peaks (with numerical stability)
    max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    is_peak = torch.abs(decoded - max_pool(decoded)) < eps

    # Encourage high gradients at peaks (with numerical stability)
    gradient_magnitude = torch.sqrt(dx**2 + dy**2 + eps)
    sharpness_loss = -torch.mean(gradient_magnitude * is_peak.float())

    # 4. Background should be close to zero
    background_mask = ~is_peak
    background_loss = torch.mean(torch.abs(decoded * background_mask.float() + eps))

    # Clip losses to prevent extreme values
    conv_loss = torch.clamp(conv_loss, -100, 100)
    symmetry_loss = torch.clamp(symmetry_loss, -100, 100)
    radial_loss = torch.clamp(radial_loss, -100, 100)
    sharpness_loss = torch.clamp(sharpness_loss, -100, 100)
    background_loss = torch.clamp(background_loss, -100, 100)

    # Combine losses with smaller weights to start
    c = 3
    total_loss = (
        conv_loss
        + 0.05 * c * symmetry_loss
        + 0.02 * c * radial_loss
        + 0.05 * c * sharpness_loss
        + 0.1 * c * background_loss
    )

    return total_loss
