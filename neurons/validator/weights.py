import queue
import bittensor as bt
from typing import List
from multiprocessing import Queue
import torch
from loguru import logger
from pydantic import BaseModel, ConfigDict

from neurons.validator.config import (
    get_config,
    get_wallet,
    get_metagraph,
    get_subtensor,
    get_backend_client,
)
from neurons.validator.utils import ttl_get_block
from neurons.validator.backend.exceptions import PostWeightsError
from neurons.validator.utils.version import get_validator_spec_version


class SetWeightsTask(BaseModel):
    epoch: int
    hotkeys: List[str]
    weights: List[float]  # Changed from torch.Tensor to List[float]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def tensor_to_list(tensor: torch.Tensor) -> List[float]:
    return tensor.detach().cpu().tolist()


async def set_weights_loop(set_weights_queue: Queue) -> None:
    try:
        weights_event: SetWeightsTask = set_weights_queue.get(block=False)
    except queue.Empty:
        return

    block: int = ttl_get_block()
    epoch: int = weights_event.epoch

    logger.info(f"Gathered a weights setting task for {block=} {epoch=}")

    epoch_length: int = get_config().alchemy.epoch_length

    if block > epoch + epoch_length:
        logger.error("Failed to set weights before next epoch!")
        return

    await set_weights(
        weights_event.hotkeys, torch.tensor(weights_event.weights)
    )


async def set_weights(
    hotkeys: List[str],
    moving_average_scores: torch.Tensor,
) -> None:
    logger.info("Going to set weights...")

    # Ensure tensor is on CPU
    moving_average_scores = moving_average_scores.cpu()

    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(
        moving_average_scores,
        p=1,
        dim=0,
    )

    try:
        await get_backend_client().post_weights(hotkeys, raw_weights)
        logger.info("Posted weights to API")
    except PostWeightsError as e:
        logger.error(f"Error logging weights to the weights API: {e}")
        return  # Added return to prevent further execution on error

    try:
        config: bt.config = get_config()
        metagraph: bt.metagraph = get_metagraph()

        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
            uids=metagraph.uids.cpu(),
            weights=raw_weights,
            netuid=config.netuid,
            metagraph=metagraph,
            subtensor=get_subtensor(),
        )
    except Exception as e:
        logger.error(f"Could not process weights for netuid: {e}")
        return

    logger.info(f"Processed weights: {processed_weights}")
    logger.info(f"Processed weight UIDs: {processed_weight_uids}")

    try:
        get_subtensor().set_weights(
            wallet=get_wallet(),
            netuid=get_config().netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=True,
            version_key=get_validator_spec_version(),
        )
        logger.info("Weights set successfully!")
    except Exception as e:
        logger.error(f"Failed to set weights: {e}")
