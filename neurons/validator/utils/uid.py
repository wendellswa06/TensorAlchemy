import asyncio
import random
import time
from typing import List

import bittensor as bt
import torch
from loguru import logger

from neurons.constants import N_NEURONS_TO_QUERY, VPERMIT_TAO
from neurons.validator.config import get_metagraph, get_subtensor


def check_uid_availability(uid: int, vpermit_tao_limit: int) -> bool:
    metagraph: bt.metagraph = get_metagraph()

    if not metagraph.axons[uid].is_serving:
        return False

    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False

    return True


async def get_random_uids(
    self, k: int, exclude: List[int] = None
) -> torch.LongTensor:
    candidate_uids = []
    avail_uids = []

    for uid in range(self.metagraph.n.item()):
        uid_is_available = check_uid_availability(uid, VPERMIT_TAO)
        uid_is_not_excluded = exclude is None or uid not in exclude
        if (
            uid_is_available
            and (self.metagraph.axons[uid].hotkey not in self.hotkey_blacklist)
            and (
                self.metagraph.axons[uid].coldkey not in self.coldkey_blacklist
            )
        ):
            avail_uids.append(uid)
            if uid_is_not_excluded:
                candidate_uids.append(uid)

    random.seed(time.time())
    random.shuffle(candidate_uids)

    final_uids = []
    t0 = time.perf_counter()
    attempt_counter = 0
    avg_num_list = []

    for uid in range(0, len(candidate_uids), N_NEURONS_TO_QUERY):
        tasks = []

        logger.info(f"UIDs in pool: {final_uids}")
        logger.info(
            f"Querying uids: {candidate_uids[uid:uid+N_NEURONS_TO_QUERY]}"
        )

        t1 = time.perf_counter()

        times_list = []

        for u in candidate_uids[uid : uid + N_NEURONS_TO_QUERY]:
            tasks.append(self.check_uid(u, times_list))

        responses = await asyncio.gather(*tasks)
        attempt_counter += 1

        logger.info(f"Time to get responses: {time.perf_counter() - t1:.2f}s")

        list_slice = times_list[-25:]
        time_sum = sum(list_slice)

        logger.info(
            f"Number of times stored: {len(times_list)}"
            + f"| Average successful response across {len(list_slice)}"
            + f" samples: {time_sum / len(list_slice) if len(list_slice) > 0 else 0:.2f}"
        )

        if True in responses:
            t2 = time.perf_counter()

            temp_list = []

            for i, response in enumerate(responses):
                if response and (len(final_uids) < k):
                    final_uids.append(candidate_uids[uid + i])
                    temp_list.append(candidate_uids[uid + i])
                elif len(final_uids) >= k:
                    break

            logger.info(
                f"Added uids: {temp_list} in {time.perf_counter() - t2:.2f}s"
            )

            avg_num_list.append(len(temp_list))

            if len(final_uids) >= k:
                break

    sum_avg = sum(avg_num_list) / attempt_counter if attempt_counter > 0 else 0

    logger.info(
        f"Time to find all {len(final_uids)} uids: {time.perf_counter() - t0:.2f}s"
        f" in {attempt_counter} attempts"
        f" | Avg active UIDs per attempt: {sum_avg:.2f}"
    )

    uids = (
        torch.tensor(final_uids)
        if len(final_uids) < k
        else torch.tensor(random.sample(final_uids, k))
    )

    return uids
