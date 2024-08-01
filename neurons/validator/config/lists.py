"""
List management utilities for the Alchemy project.
"""

from typing import Any, Dict, List, Set, Tuple
from google.cloud import storage
from loguru import logger

from neurons.utils.gcloud import retrieve_public_file


async def get_storage_client() -> storage.Client:
    """
    Create and return an anonymous storage client.

    Returns:
        storage.Client: An anonymous storage client.
    """
    return storage.Client.create_anonymous_client()


async def get_list(list_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get a list of a specific type.

    Args:
        list_type (str):
            The type of list to retrieve
            (e.g., "blacklist", "whitelist", "warninglist").

    Returns:
        Dict[str, Dict[str, Any]]: The retrieved list.
    """
    try:
        return await retrieve_public_file(list_type)
    except Exception as e:
        logger.error(f"Error retrieving {list_type}: {e}")
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
