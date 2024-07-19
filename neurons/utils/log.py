import sys
import torch
from loguru import logger

import bittensor as bt
from typing import Any

from PIL.Image import Image as ImageType


def setup_logger() -> None:
    # Remove the default handler
    logger.remove()

    # Add a custom handler with simplified formatting
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=False,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # This can help with thread-safety
        catch=True,  # Catch exceptions raised during logging
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # Optionally, add a file handler for more detailed logging
    logger.add(
        "detailed_log_{time}.log",
        enqueue=True,
        level="DEBUG",
        rotation="500 MB",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def image_to_log(image: Any) -> str:
    if isinstance(image, str):
        return "base64(**IMAGEDATA**)"

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


# Utility function for coloring logs
def colored_log(
    message: str,
    color: str = "white",
    level: str = "INFO",
) -> None:
    logger.opt(colors=True).log(
        level, f"<bold><{color}>{message}</{color}></bold>"
    )
