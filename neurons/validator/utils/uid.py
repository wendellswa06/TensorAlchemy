import asyncio
import random
import time
from typing import List, Tuple, Dict, Any
from functools import wraps

import bittensor as bt
import torch
from loguru import logger

from neurons.constants import N_NEURONS_TO_QUERY, VPERMIT_TAO
from neurons.validator.config import get_metagraph, get_subtensor


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


def is_uid_available(
    uid: int, metagraph: bt.metagraph, vpermit_tao_limit: int
) -> bool:
    """
    Check if a UID is available based on serving status and stake.

    Args:
        uid (int): The UID to check.
        metagraph (bt.metagraph): The metagraph object.
        vpermit_tao_limit (int): The validator permit TAO limit.

    Returns:
        bool: True if the UID is available, False otherwise.
    """
    if not metagraph.axons[uid].is_serving:
        return False

    if metagraph.validator_permit[uid] and metagraph.S[uid] > vpermit_tao_limit:
        return False

    return True


def filter_available_uids(
    metagraph: bt.metagraph, exclude: List[int] = None
) -> List[int]:
    """
    Filter available UIDs based on availability and exclusion list.

    Args:
        metagraph (bt.metagraph): The metagraph object.
        exclude (List[int], optional): List of UIDs to exclude. Defaults to None.

    Returns:
        List[int]: List of available UIDs.
    """
    exclude = exclude or []
    return [
        uid
        for uid in range(metagraph.n.item())
        if is_uid_available(uid, metagraph, VPERMIT_TAO)
        and metagraph.axons[uid].hotkey not in metagraph.hotkey_blacklist
        and metagraph.axons[uid].coldkey not in metagraph.coldkey_blacklist
        and uid not in exclude
    ]


async def check_uids_alive(
    self, uids: List[int]
) -> Tuple[List[int], List[float]]:
    """
    Check which UIDs are alive.

    Args:
        self: The caller object with check_uid method.
        uids (List[int]): List of UIDs to check.

    Returns:
        Tuple[List[int], List[float]]: A tuple containing the list of alive UIDs and their response times.
    """
    tasks = [self.check_uid(uid) for uid in uids]
    responses = await asyncio.gather(*tasks)

    alive_uids = []
    response_times = []

    for uid, (is_alive, response_time) in zip(uids, responses):
        if is_alive:
            alive_uids.append(uid)
            response_times.append(response_time)

    return alive_uids, response_times


@memoize_with_expiration(120)  # Memoize for 60 seconds
async def get_all_active_uids(self, metagraph: bt.metagraph) -> List[int]:
    """
    Fetch all active (alive) UIDs. Results are memoized for 60 seconds.

    Args:
        self: The caller object with check_uid method.
        metagraph (bt.metagraph): The metagraph object.

    Returns:
        List[int]: List of all active UIDs.
    """
    logger.info("Fetching all active UIDs")
    available_uids = filter_available_uids(metagraph)

    # Shuffle to avoid always checking the same UIDs first
    random.shuffle(available_uids)

    all_active_uids = []
    for i in range(0, len(available_uids), N_NEURONS_TO_QUERY):
        batch = available_uids[i : i + N_NEURONS_TO_QUERY]
        active_uids, _ = await check_uids_alive(self, batch)
        all_active_uids.extend(active_uids)

    logger.info(f"Found {len(all_active_uids)} active UIDs")
    return all_active_uids


async def get_random_uids(
    self, metagraph: bt.metagraph, k: int, exclude: List[int] = None
) -> torch.LongTensor:
    """
    Get random active UIDs.

    Args:
        self: The caller object with check_uid method.
        metagraph (bt.metagraph): The metagraph object.
        k (int): Number of random UIDs to select.
        exclude (List[int], optional): List of UIDs to exclude. Defaults to None.

    Returns:
        torch.LongTensor: Tensor of randomly selected active UIDs.
    """
    start_time = time.perf_counter()

    all_active_uids = await get_all_active_uids(self, metagraph)

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
    metagraph = get_metagraph()
    subtensor = get_subtensor()

    # Get all active UIDs
    all_active = await get_all_active_uids(subtensor, metagraph)
    logger.info(f"Total active UIDs: {len(all_active)}")

    # Get 5 random active UIDs
    random_uids = await get_random_uids(subtensor, metagraph, k=5)
    logger.info(f"Random 5 active UIDs: {random_uids.tolist()}")

    # Call get_all_active_uids again (should use cached result)
    all_active_cached = await get_all_active_uids(subtensor, metagraph)
    logger.info(f"Total active UIDs (cached): {len(all_active_cached)}")


if __name__ == "__main__":
    asyncio.run(main())
