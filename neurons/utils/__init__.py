import os
import sys
import inspect
import asyncio
import traceback
import multiprocessing

from threading import Timer

import torch

import _thread

from loguru import logger
from google.cloud import storage

from neurons.common_schema import NeuronAttributes
from neurons.constants import (
    IA_MINER_BLACKLIST,
    IA_MINER_WARNINGLIST,
    IA_MINER_WHITELIST,
    IA_VALIDATOR_BLACKLIST,
    IA_VALIDATOR_WEIGHT_FILES,
    IA_VALIDATOR_WHITELIST,
    N_NEURONS,
)
from neurons.utils.exceptions import BittensorBrokenPipe
from neurons.utils.common import is_validator

from neurons.validator.scoring.models.types import (
    RewardModelType,
)
from neurons.utils.log import configure_logging


# Background Loop
class BackgroundTimer(Timer):
    def __str__(self) -> str:
        return self.function.__name__

    def run(self):
        configure_logging()
        self.function(*self.args, **self.kwargs)
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


class MultiprocessBackgroundTimer(multiprocessing.Process):
    def __str__(self) -> str:
        return self.function.__name__

    def __init__(self, interval, function, args=None, kwargs=None):
        super().__init__()
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = multiprocessing.Event()

    def run(self):
        configure_logging()

        logger.info(f"{self.function.__name__} started")

        while not self.finished.is_set():
            try:
                if inspect.iscoroutinefunction(self.function):
                    asyncio.run(self.function(*self.args, **self.kwargs))
                else:
                    self.function(*self.args, **self.kwargs)

                self.finished.wait(self.interval)

            except Exception:
                logger.error(traceback.format_exc())

    def cancel(self):
        self.finished.set()


def get_coldkey_for_hotkey(self, hotkey):
    """
    Look up the coldkey of the caller.
    """
    if hotkey in self.metagraph.hotkeys:
        index = self.metagraph.hotkeys.index(hotkey)
        return self.metagraph.coldkeys[index]
    return None


def send_run_command(command_queue, command, data):
    """
    Send a command to the main process with the associated data.
    """
    command_queue.put((command, data))
    logger.info(f"Sent command: {command} with data: {data}")


def kill_main_process_if_deregistered(
    command_queue,
    neurom_attributes: NeuronAttributes,
):
    # Terminate the miner / validator after deregistration
    if (
        neurom_attributes.background_steps % 5 == 0
        and neurom_attributes.background_steps > 1
    ):
        try:
            if (
                neurom_attributes.wallet_hotkey_ss58_address
                not in neurom_attributes.hotkeys
            ):
                logger.info(f">>> Neuron has deregistered... terminating.")
                try:
                    send_run_command(command_queue, "die", None)
                except Exception as e:
                    logger.info(
                        f"An error occurred trying to terminate the main thread: {e}."
                    )
                try:
                    os.exit(0)
                except Exception as e:
                    logger.error(
                        f"An error occurred trying to use os._exit(): {e}."
                    )
                sys.exit(0)
        except Exception as e:
            logger.error(
                f">>> An unexpected error occurred syncing the metagraph: {e}"
            )


def calculate_human_voting_weight(validator_weights, neuron_attributes):
    if "human_reward_model" not in validator_weights:
        return None
    # NOTE: Scaling factor for the human reward model
    #
    # The human reward model updates the rewards for all
    # neurons (256 on mainnet) in each step, while the
    # other reward models only update rewards for a subset
    # of neurons (e.g., 12) per step.
    #
    # To avoid rewards being updated out of sync,
    # we scale down the human rewards in each step.
    #
    # The scaling factor is calculated as the total number
    # of neurons divided by the number of neurons updated
    # per step,
    #
    # Then multiplied by an adjustment factor (1.5) to account
    # for potential duplicate neuron selections during a full
    # epoch.
    #
    # The adjustment factor of 1.5 was determined empirically
    # based on the observed number of times UIDs received
    # duplicate calls in a full epoch on the mainnet.
    adjustment_factor: float = 1.5
    total_number_of_neurons: int = neuron_attributes.total_number_of_neurons

    return validator_weights["human_reward_model"] / (
        (total_number_of_neurons / N_NEURONS) * adjustment_factor
    )


def process_and_normalize_reward_weights(validator_weights, neuron_attributes):
    if not validator_weights:
        return None
    weights_to_add = []
    reward_names = [
        RewardModelType.IMAGE,
    ]

    for rw_name in reward_names:
        if rw_name in validator_weights:
            weights_to_add.append(validator_weights[rw_name])

    logger.info(f"Raw model weights: {weights_to_add}")
    reward_weights = None
    if weights_to_add:
        # Normalize weights
        if sum(weights_to_add) != 1:
            weights_to_add = normalize_weights(weights_to_add)
            logger.info(f"Normalized model weights: {weights_to_add}")

        reward_weights = torch.tensor(weights_to_add, dtype=torch.float32).to(
            neuron_attributes.device
        )

        logger.info(
            f"Retrieved the latest validator weights: {neuron_attributes.reward_weights}"
        )

    # self.reward_weights = torch.tensor(
    # [v for k, v in validator_weights.items() if "manual" not in k],
    # dtype=torch.float32,
    # ).to(self.device)
    return reward_weights


def update_and_normalize_validator_weights(
    command_queue, neuron_attributes: NeuronAttributes
):
    from neurons.utils.gcloud import retrieve_public_file

    validator_weights = asyncio.run(
        retrieve_public_file(IA_VALIDATOR_WEIGHT_FILES)
    )

    human_voting_weight = calculate_human_voting_weight(
        validator_weights, neuron_attributes
    )

    reward_weights = process_and_normalize_reward_weights(
        validator_weights, neuron_attributes
    )
    send_run_command(
        command_queue,
        "update_validator_weights",
        {
            "human_voting_weight": human_voting_weight,
            "reward_weights": reward_weights,
        },
    )


def validator_background_loop(shared_data):
    """
    Handles terminating the miner after de-registration and
    updating the blacklist and whitelist.
    """
    neuron_attributes: NeuronAttributes = shared_data["neuron_attributes"]
    command_queue = shared_data.get("command_queue")
    kill_main_process_if_deregistered(command_queue, neuron_attributes)
    if neuron_attributes.background_steps % 5 == 0:
        try:
            update_and_normalize_validator_weights(
                command_queue, neuron_attributes
            )
        except Exception as e:
            logger.error(
                f"An error occurred trying to update settings from the cloud: {e}."
            )

    neuron_attributes.background_steps += 1


def miner_background_loop(shared_data):
    """
    Background loop specific for Miner.
    """
    neuron_attributes: NeuronAttributes = shared_data["neuron_attributes"]
    command_queue = shared_data.get("command_queue")
    kill_main_process_if_deregistered(
        command_queue,
        neuron_attributes,
    )


def normalize_weights(weights):
    sum_weights = float(sum(weights))
    normalizer = 1 / sum_weights
    weights = [weight * normalizer for weight in weights]
    if sum(weights) < 1:
        diff = 1 - sum(weights)
        weights[0] += diff

    return weights
