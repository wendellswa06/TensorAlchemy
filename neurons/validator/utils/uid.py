import asyncio
import random
import time
import traceback
from typing import List, Tuple, Dict, Any
from functools import wraps
from multiprocessing import Manager

import numpy as np
import bittensor as bt
import torch
from loguru import logger

from neurons.protocol import IsAlive
from neurons.constants import N_NEURONS_TO_QUERY, VPERMIT_TAO
from neurons.validator.config import (
    get_config,
    get_dendrite,
    get_metagraph,
    get_subtensor,
    get_wallet,
    get_blacklist,
)

isalive_threshold = 8

# Global variables to hold our Manager instance and managed dictionary
_manager = None
_isalive_dict = None


def get_manager():
    global _manager
    if _manager is None:
        _manager = Manager()
    return _manager


def get_isalive_dict():
    global _isalive_dict
    if _isalive_dict is None:
        _isalive_dict = get_manager().dict()
    return _isalive_dict


async def check_uid(uid) -> Tuple[bool, float]:
    try:
        t1 = time.perf_counter()
        metagraph: bt.metagraph = get_metagraph()
        response = await get_dendrite().forward(
            synapse=IsAlive(),
            axons=metagraph.axons[uid],
            timeout=get_config().alchemy.async_timeout,
        )

        isalive_dict = get_isalive_dict()

        if response.is_success:
            isalive_dict[uid] = 0
            return True, time.perf_counter() - t1

        if uid in isalive_dict:
            isalive_dict[uid] += 1
        else:
            isalive_dict[uid] = 1

        return False, -1
    except Exception:
        logger.error(f"Error checking UID {uid}: {traceback.format_exc()}")
        return False, -1


def memoize_with_expiration(expiration_time: int):
    """
    Decorator to memoize a function with a time-based expiration.

    Args:
        expiration_time (int): Time in seconds before the cached result expires.
    """
    cache: Dict[Any, Tuple[Any, float]] = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < expiration_time:
                    return result

            result = await func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        return wrapper

    return decorator


def is_uid_available(uid: int, vpermit_tao_limit: int) -> bool:
    """
    Check if a UID is available based on serving status and stake.

    Args:
        uid (int): The UID to check.
        vpermit_tao_limit (int): The validator permit TAO limit.

    Returns:
        bool: True if the UID is available, False otherwise.
    """
    metagraph = get_metagraph()

    if not metagraph.axons[uid].is_serving:
        return False

    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False

    return True


async def filter_available_uids(exclude: List[int] = None) -> List[int]:
    """
    Filter available UIDs based on availability and exclusion list.

    Args:
        exclude (List[int], optional): List of UIDs to exclude. Defaults to None.

    Returns:
        List[int]: List of available UIDs.
    """
    exclude = exclude or []
    metagraph = get_metagraph()

    hotkey_blacklist, coldkey_blacklist = await get_blacklist()

    return [
        uid
        for uid in range(metagraph.n.item())
        if is_uid_available(uid, VPERMIT_TAO)
        and metagraph.axons[uid].hotkey not in hotkey_blacklist
        and metagraph.axons[uid].coldkey not in coldkey_blacklist
        and uid not in exclude
    ]


async def check_uids_alive(uids: List[int]) -> Tuple[List[int], List[float]]:
    """
    Check which UIDs are alive.

    Args:
        uids (List[int]): List of UIDs to check.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing the list of alive UIDs and their response times.
    """
    responses = await asyncio.gather(*[check_uid(uid) for uid in uids])

    alive_uids = []
    response_times = []

    for uid, (is_alive, response_time) in zip(uids, responses):
        if is_alive:
            alive_uids.append(uid)
            response_times.append(response_time)

    return alive_uids, response_times


def update_isalive_dict() -> None:
    isalive_dict = get_isalive_dict()
    # Start following this UID
    for uid in range(get_metagraph().n.item()):
        if uid not in isalive_dict:
            isalive_dict[uid] = 0


@memoize_with_expiration(20)
async def get_all_active_uids() -> List[int]:
    """
    Fetch all active (alive) UIDs. Results are memoized for 20 seconds.

    Returns:
        List[int]: List of all active UIDs.
    """
    logger.info("Fetching all active UIDs")
    update_isalive_dict()

    available_uids = await filter_available_uids()

    # Shuffle to avoid always checking the same UIDs first
    random.shuffle(available_uids)

    all_active_uids = []
    for i in range(0, len(available_uids), N_NEURONS_TO_QUERY):
        batch = available_uids[i : i + N_NEURONS_TO_QUERY]
        active_uids, _ = await check_uids_alive(batch)
        all_active_uids.extend(active_uids)

    logger.info(f"Found {len(all_active_uids)} active UIDs")
    return all_active_uids


async def get_random_uids(
    k: int, exclude: List[int] = None
) -> torch.LongTensor:
    """
    Get random active UIDs.

    Args:
        k (int): Number of random UIDs to select.
        exclude (List[int], optional): List of UIDs to exclude. Defaults to None.

    Returns:
        torch.LongTensor: Tensor of randomly selected active UIDs.
    """
    start_time = time.perf_counter()

    all_active_uids = await get_all_active_uids()

    if exclude:
        all_active_uids = [uid for uid in all_active_uids if uid not in exclude]

    selected_uids = (
        all_active_uids
        if len(all_active_uids) <= k
        else random.sample(all_active_uids, k)
    )

    end_time = time.perf_counter()
    logger.info(
        f"Time to find {len(selected_uids)} random UIDs: {end_time - start_time:.2f}s"
    )

    return torch.tensor(selected_uids, dtype=torch.long)


# Example usage
async def main():
    # Get all active UIDs
    all_active = await get_all_active_uids()
    logger.info(f"Total active UIDs: {len(all_active)}")

    # Get 5 random active UIDs
    random_uids = await get_random_uids(k=5)
    logger.info(f"Random 5 active UIDs: {random_uids.tolist()}")

    # Call get_all_active_uids again (should use cached result)
    all_active_cached = await get_all_active_uids()
    logger.info(f"Total active UIDs (cached): {len(all_active_cached)}")

    # Print the contents of isalive_dict
    logger.info(f"isalive_dict contents: {dict(get_isalive_dict())}")


if __name__ == "__main__":
    asyncio.run(main())
