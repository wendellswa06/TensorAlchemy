import bittensor as bt
import concurrent.futures

import torch
from loguru import logger

from neurons.validator.backend.exceptions import PostWeightsError
from neurons.validator.utils.version import get_validator_spec_version


async def set_weights(validator: "StableValidator"):
    # Calculate the average reward for each uid across non-zero values.
    # Replace any NaN values with 0.
    raw_weights = torch.nn.functional.normalize(
        validator.moving_average_scores, p=1, dim=0
    )

    try:
        await validator.backend_client.post_weights(
            validator.hotkeys,
            raw_weights,
        )

    except PostWeightsError as e:
        logger.error(f"error logging weights to the weights api: {e}")

    (
        processed_weight_uids,
        processed_weights,
    ) = bt.utils.weight_utils.process_weights_for_netuid(
        uids=validator.metagraph.uids.to("cpu"),
        weights=raw_weights.to("cpu"),
        netuid=validator.config.netuid,
        subtensor=validator.subtensor,
        metagraph=validator.metagraph,
    )
    logger.info("processed_weights", processed_weights)
    logger.info("processed_weight_uids", processed_weight_uids)

    from neurons.validator import validator

    # Set the weights on chain via our subtensor connection.

    # Define a function to set weights that will be executed by the executor
    def set_weights_task():
        try:
            val.subtensor.set_weights(
                wallet=val.wallet,
                netuid=val.config.netuid,
                uids=processed_weight_uids,
                weights=processed_weights,
                wait_for_finalization=True,
                version_key=get_validator_spec_version(),
            )
            logger.info("Weights set successfully!")
        except Exception as e:
            logger.error(f"Failed to set weights {e}")

    # Use an executor to run the weight-setting task
    with concurrent.futures.ThreadPoolExecutor() as executor:
        _future = executor.submit(set_weights_task)
