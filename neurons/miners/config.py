import os
import random
import argparse
from typing import Optional

import torch
import bittensor as bt
from loguru import logger

config: bt.config = None
device: torch.device = None
metagraph: bt.metagraph = None


def get_config() -> None:
    global config
    if config is not None:
        return config

    argp = argparse.ArgumentParser(description="Miner Config")

    # Add any args from the parent class
    argp.add_argument(
        "--netuid",
        type=int,
        default=1,
    )
    argp.add_argument(
        "--wandb.project",
        type=str,
        default="",
    )
    argp.add_argument(
        "--wandb.entity",
        type=str,
        default="",
    )
    argp.add_argument(
        "--wandb.api_key",
        type=str,
        default="",
    )
    argp.add_argument(
        "--miner.device",
        type=str,
        default="cuda:0",
    )
    argp.add_argument(
        "--miner.optimize",
        action="store_true",
    )
    argp.add_argument(
        "--miner.seed",
        type=int,
        default=random.randint(0, 100_000_000_000),
    )
    argp.add_argument(
        "--miner.custom_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    argp.add_argument(
        "--miner.custom_refiner",
        type=str,
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
    )
    argp.add_argument(
        "--miner.alchemy_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    argp.add_argument(
        "--miner.alchemy_refiner",
        type=str,
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
    )

    bt.axon.add_args(argp)
    bt.wallet.add_args(argp)
    bt.logging.add_args(argp)
    bt.subtensor.add_args(argp)

    config = bt.config(argp)
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )

    # Ensure the directory for logging exists
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    return config


def get_metagraph(
    netuid: int = 25,
    network: str = "test",
    **kwargs,
) -> bt.metagraph:
    global metagraph
    if not metagraph:
        metagraph = bt.metagraph(
            netuid=netuid,
            network=network,
            **kwargs,
        )

    return metagraph


def get_default_device() -> torch.device:
    return torch.device("cuda:0")


def get_device(new_device: Optional[torch.device] = None) -> torch.device:
    global device
    if not device:
        if new_device is None:
            device = get_default_device()

        else:
            device = new_device

    return device
