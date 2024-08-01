"""
Main configuration module for the Alchemy project.
This module initializes and manages global configuration objects and utilities.
"""

from .constants import AlchemyHost, validator_run_id
from .device import get_default_device, get_device
from .parser import (
    add_args,
    check_config,
    get_config,
    update_validator_settings,
)
from .clients import (
    get_corcel_api_key,
    get_openai_client,
    get_wallet,
    get_dendrite,
    get_subtensor,
    get_metagraph,
    get_backend_client,
)
from .lists import get_blacklist, get_whitelist, get_warninglist

__all__ = [
    "AlchemyHost",
    "get_default_device",
    "get_device",
    "add_args",
    "check_config",
    "get_config",
    "update_validator_settings",
    "get_corcel_api_key",
    "get_openai_client",
    "get_wallet",
    "get_dendrite",
    "get_subtensor",
    "get_metagraph",
    "get_backend_client",
    "get_blacklist",
    "get_whitelist",
    "get_warninglist",
    "validator_run_id",
]
