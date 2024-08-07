"""
Configuration parsing and management utilities for the Alchemy project.
"""

import os
import argparse
from typing import Dict
import bittensor as bt
from loguru import logger
from .device import get_default_device
from .constants import AlchemyHost


def add_args(parser: argparse.ArgumentParser) -> None:
    """
    Add Alchemy-specific arguments to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser to add arguments to.
    """
    parser.add_argument("--netuid", type=int, help="Network netuid", default=26)
    parser.add_argument(
        "--alchemy.name",
        type=str,
        help="Validator name",
        default="image_alchemy_validator",
    )
    parser.add_argument(
        "--alchemy.debug", type=bool, default=False, help="Enable debug logging"
    )
    parser.add_argument(
        "--alchemy.device",
        type=str,
        default=get_default_device(),
        help="Device to run the validator on",
    )
    parser.add_argument(
        "--alchemy.host",
        type=AlchemyHost,
        choices=list(AlchemyHost),
        help="Choose the Alchemy host",
    )
    parser.add_argument(
        "--alchemy.streamlit_port",
        type=int,
        help="Port number for streamlit app",
        default=None,
    )
    parser.add_argument(
        "--alchemy.request_frequency",
        type=int,
        default=35,
        help="Request frequency for the validator",
    )
    parser.add_argument(
        "--alchemy.query_timeout",
        type=float,
        default=20,
        help="Query timeout for the validator",
    )
    parser.add_argument(
        "--alchemy.async_timeout",
        type=float,
        default=1.2,
        help="Async timeout for the validator",
    )
    parser.add_argument(
        "--alchemy.epoch_length",
        type=int,
        default=100,
        help="Epoch length for the validator",
    )


def check_config(to_check: bt.config) -> None:
    """
    Check and validate the configuration object.

    Args:
        to_check (bt.config): The configuration object to check.
    """
    bt.logging.check_config(to_check)

    if to_check.mock:
        to_check.neuron.mock_reward_models = True
        to_check.neuron.mock_gating_model = True
        to_check.neuron.mock_dataset = True
        to_check.wallet._mock = True

    full_path = os.path.expanduser(
        f"{to_check.logging.logging_dir}/{to_check.wallet.name}/{to_check.wallet.hotkey}/netuid{to_check.netuid}/{to_check.alchemy.name}"
    )
    to_check.alchemy.full_path = os.path.expanduser(full_path)
    os.makedirs(to_check.alchemy.full_path, exist_ok=True)


def get_config() -> bt.config:
    """
    Get the global configuration object.

    Returns:
        bt.config: The global configuration object.
    """
    global config
    if config is None:
        parser = argparse.ArgumentParser()
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.axon.add_args(parser)
        add_args(parser)
        config = bt.config(parser)
        check_config(config)
    return config


def update_validator_settings(validator_settings: Dict) -> bt.config:
    """
    Update the validator settings in the global configuration.

    Args:
        validator_settings (Dict): New validator settings to apply.

    Returns:
        bt.config: The updated global configuration object.
    """
    global config
    if not validator_settings:
        logger.error("Failed to update validator settings")
        return config

    config.alchemy.request_frequency = int(
        validator_settings.get(
            "request_frequency",
            config.request_frequency,
        )
    )
    config.alchemy.query_timeout = float(
        validator_settings.get(
            "query_timeout",
            config.query_timeout,
        )
    )
    config.alchemy.async_timeout = int(
        validator_settings.get(
            "async_timeout",
            config.async_timeout,
        )
    )
    config.alchemy.epoch_length = int(
        validator_settings.get(
            "epoch_length",
            config.epoch_length,
        )
    )

    logger.info(
        f"Retrieved the latest validator settings: {validator_settings}"
    )
    return config


config: bt.config = None
