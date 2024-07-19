import sys
import torch
from loguru import logger


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
