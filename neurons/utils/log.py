import sys
import torch
from loguru import logger

import bittensor as bt
from typing import Any

from PIL.Image import Image as ImageType


def image_to_str(image: Any) -> str:
    if isinstance(image, str):
        return f"base64(**bytes:<{len(image)}>**)"

    if isinstance(image, bt.Tensor):
        return f"bt.Tensor({image.shape})"

    if hasattr(image, "shape"):
        return f"shaped({image.shape})"

    if isinstance(image, ImageType):
        return f"PIL.Image({image.width}, {image.height})"

    return f"UNKNOWN IMAGE TYPE {type(image)}"


def sh(message: str):
    return f"{message: <12}"


def summarize_rewards(reward_tensor: torch.Tensor) -> str:
    non_zero = reward_tensor[reward_tensor != 0]
    if len(non_zero) == 0:
        return "All zeros"
    return (
        f"Non-zero: {len(non_zero)}/{len(reward_tensor)}, "
        f"Mean: {reward_tensor.mean():.4f}, "
        f"Max: {reward_tensor.max():.4f}, "
        f"Min non-zero: {non_zero.min():.4f}"
    )
