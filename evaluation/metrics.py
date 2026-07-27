import numpy as np
import torch

from skimage.metrics import structural_similarity


def calculate_mse(
    original,
    reconstructed
):
    """
    Calculate mean squared error.

    Args:
        original:
            Original image tensor.

        reconstructed:
            Reconstructed image tensor.

    Returns:
        MSE value.
    """

    return torch.mean(
        (original - reconstructed) ** 2
    ).item()


def calculate_psnr(
    mse,
    max_value=1.0
):
    """
    Calculate PSNR from MSE.

    Args:
        mse:
            Mean squared error.

        max_value:
            Maximum possible pixel value.

    Returns:
        PSNR value in decibels.
    """

    if mse == 0:
        return float("inf")

    return 10 * np.log10(
        (max_value ** 2) / mse
    )


def calculate_ssim(
    original,
    reconstructed
):
    """
    Calculate average SSIM for a batch.

    Args:
        original:
            Original image batch.

        reconstructed:
            Reconstructed image batch.

    Returns:
        Average SSIM value.
    """

    original = (
        original
        .detach()
        .cpu()
        .numpy()
    )

    reconstructed = (
        reconstructed
        .detach()
        .cpu()
        .numpy()
    )

    ssim_values = []

    for i in range(
        original.shape[0]
    ):

        original_image = (
            original[i]
            .squeeze()
        )

        reconstructed_image = (
            reconstructed[i]
            .squeeze()
        )

        score = structural_similarity(
            original_image,
            reconstructed_image,
            data_range=1.0
        )

        ssim_values.append(score)

    return float(
        np.mean(ssim_values)
    )