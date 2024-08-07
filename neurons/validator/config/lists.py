"""
List management utilities for the Alchemy project.
"""

import traceback
from typing import Any, Dict, List, Set, Tuple

from google.cloud import storage
from loguru import logger

from neurons.utils.common import is_validator
from neurons.utils.gcloud import retrieve_public_file
from neurons.constants import (
    IA_MINER_BLACKLIST,
    IA_MINER_WARNINGLIST,
    IA_MINER_WHITELIST,
    IA_VALIDATOR_BLACKLIST,
    IA_VALIDATOR_WHITELIST,
)


def get_file_name(list_type: str) -> str:
    """Determine the file name based on list type and neuron type."""
    if list_type == "whitelist":
        return IA_VALIDATOR_WHITELIST if is_validator() else IA_MINER_WHITELIST
    elif list_type == "blacklist":
        return IA_VALIDATOR_BLACKLIST if is_validator() else IA_MINER_BLACKLIST
    elif list_type == "warninglist":
        return IA_MINER_WARNINGLIST
    else:
        raise ValueError(f"Invalid list_type: {list_type}")


async def get_list(list_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get a list of a specific type based on the neuron type.

    Args:
        list_type (str): The type of list to retrieve
            ("blacklist", "whitelist", or "warninglist").

    Returns:
        Dict[str, Dict[str, Any]]: The retrieved list.
    """
    file_name = get_file_name(list_type)
    try:
        result = await retrieve_public_file(file_name)
        logger.info(f"Retrieved {list_type}")
        return result
    except Exception as e:
        logger.error(f"Error retrieving {list_type}: {traceback.format_exc()}")
        return {}


async def get_blacklist() -> Tuple[Set[str], Set[str]]:
    """
    Get the current blacklist.

    Returns:
        Tuple[Set[str], Set[str]]:
            A tuple containing the hotkey blacklist and coldkey blacklist.
    """
    blacklist = await get_list("blacklist")
    return (
        {k for k, v in blacklist.items() if v["type"] == "hotkey"},
        {k for k, v in blacklist.items() if v["type"] == "coldkey"},
    )


async def get_whitelist() -> Tuple[Set[str], Set[str]]:
    """
    Get the current whitelist.

    Returns:
        Tuple[Set[str], Set[str]]:
            A tuple containing the hotkey whitelist and coldkey whitelist.
    """
    whitelist = await get_list("whitelist")
    return (
        {k for k, v in whitelist.items() if v["type"] == "hotkey"},
        {k for k, v in whitelist.items() if v["type"] == "coldkey"},
    )


async def get_warninglist() -> Tuple[
    Dict[str, List[str]], Dict[str, List[str]]
]:
    """
    Get the current warninglist.

    Returns:
        Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
            A tuple containing the hotkey warninglist and coldkey warninglist.
    """
    warninglist = await get_list("warninglist")
    return (
        {
            k: [v["reason"], v["resolve_by"]]
            for k, v in warninglist.items()
            if v["type"] == "hotkey"
        },
        {
            k: [v["reason"], v["resolve_by"]]
            for k, v in warninglist.items()
            if v["type"] == "coldkey"
        },
    )
