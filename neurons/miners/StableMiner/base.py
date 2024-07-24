import asyncio
import sys
import time
import traceback
from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from diffusers.callbacks import SDXLCFGCutoffCallback
from loguru import logger
from neurons.constants import VPERMIT_TAO
from neurons.miners.StableMiner.schema import ModelConfig, TaskType
from neurons.miners.StableMiner.utils.helpers import (
    without_keys,
    setup_model_args,
    log_generation_time,
    filter_nsfw_images,
    setup_refiner_args,
)
from neurons.miners.config import get_bt_miner_config
from neurons.protocol import ImageGeneration, IsAlive, ModelType
from neurons.utils import BackgroundTimer, background_loop
from neurons.utils.defaults import Stats, get_defaults
from neurons.utils.image import image_to_base64
from neurons.utils.log import sh
from neurons.miners.StableMiner.utils import (
    get_caller_stake,
    get_coldkey_for_hotkey,
)
import bittensor as bt


class BaseMiner(ABC):
    def __init__(self) -> None:
        self.storage_client: Any = None
        # TODO: Fix safety checker and processor to allow different values for each task config
        self.safety_checker: Optional[torch.nn.Module] = None
        self.processor: Optional[torch.nn.Module] = None
        self.bt_config = get_bt_miner_config()
        self.t2i_args = {"guidance_scale": 7.5, "num_inference_steps": 20}
        self.i2i_args = {"guidance_scale": 5, "strength": 0.6}
        self.initialize_components()
        self.request_dict: Dict[str, Dict[str, Union[List[float], int]]] = {}

    def initialize_components(self) -> None:
        self.initialize_logging()
        self.initialize_blacklists_and_whitelists()
        self.initialize_event_dict()
        self.initialize_subtensor_connection_and_metagraph()
        self.initialize_wallet_and_defaults()
        self.loop_until_registered()
        self.initialize_transform_function()
        self.start_background_loop()

    def initialize_logging(self) -> None:
        if self.bt_config.logging.debug:
            bt.debug()
            logger.info("Enabling debug mode...")

    def initialize_blacklists_and_whitelists(self) -> None:
        self.hotkey_blacklist: set = set()
        self.coldkey_blacklist: set = set()
        self.coldkey_whitelist: set = set(
            ["5F1FFTkJYyceVGE4DCVN5SxfEQQGJNJQ9CVFVZ3KpihXLxYo"]
        )
        self.hotkey_whitelist: set = set(
            ["5C5PXHeYLV5fAx31HkosfCkv8ark3QjbABbjEusiD3HXH2Ta"]
        )

    def initialize_subtensor_connection_and_metagraph(self) -> None:
        logger.info("Establishing subtensor connection")
        self.subtensor: bt.subtensor = bt.subtensor(config=self.bt_config)
        self.metagraph: bt.metagraph = self.subtensor.metagraph(
            netuid=self.bt_config.netuid
        )

    def initialize_wallet_and_defaults(self) -> None:
        self.wallet: bt.wallet = bt.wallet(config=self.bt_config)
        self.stats: Stats = get_defaults(self)

    def is_whitelisted(
        self, caller_hotkey: str = None, caller_coldkey: str = None
    ) -> bool:
        if caller_hotkey and self._is_in_whitelist(caller_hotkey):
            return True
        if caller_hotkey:
            caller_coldkey = get_coldkey_for_hotkey(caller_hotkey)
            if self._is_in_whitelist(caller_coldkey):
                return True
        if caller_coldkey and self._is_in_whitelist(caller_coldkey):
            return True
        return False

    def _is_in_whitelist(self, key: str) -> bool:
        return key in self.hotkey_whitelist or key in self.coldkey_whitelist

    def initialize_event_dict(self) -> None:
        self.event: Dict[str, Any] = {}
        self.mapping: Dict[str, Dict] = {}

    def initialize_transform_function(self) -> None:
        self.transform: transforms.Compose = transforms.Compose(
            [transforms.PILToTensor()]
        )

    def start_background_loop(self) -> None:
        self.background_steps: int = 1
        self.background_timer: BackgroundTimer = BackgroundTimer(
            300, background_loop, [self, False]
        )
        self.background_timer.daemon = True
        self.background_timer.start()

    def start_axon(self) -> None:
        logger.info(f"Serving axon on port {self.bt_config.axon.port}.")
        self.create_axon()
        self.register_axon()

    def create_axon(self) -> None:
        try:
            self.axon: bt.axon = (
                bt.axon(
                    wallet=self.wallet,
                    ip=bt.utils.networking.get_external_ip(),
                    external_ip=self.bt_config.axon.get("external_ip")
                    or bt.utils.networking.get_external_ip(),
                    config=self.bt_config,
                )
                .attach(
                    forward_fn=self.is_alive,
                    blacklist_fn=self.blacklist_is_alive,
                    priority_fn=self.priority_is_alive,
                )
                .attach(
                    forward_fn=self.generate_image,
                    blacklist_fn=self.blacklist_image_generation,
                    priority_fn=self.priority_image_generation,
                )
                .start()
            )
            logger.info(f"Axon created: {self.axon}", color="green")
        except Exception as e:
            logger.error(f"Failed to create axon: {e}")
            raise

    def register_axon(self) -> None:
        try:
            self.subtensor.serve_axon(
                axon=self.axon, netuid=self.bt_config.netuid
            )
        except Exception as e:
            logger.error(f"Failed to register axon: {e}")
            raise

    def loop_until_registered(self) -> None:
        while True:
            try:
                if self.is_miner_registered():
                    break
                self.handle_unregistered_miner()
            except Exception as e:
                logger.error(f"Error in loop_until_registered: {e}")
                time.sleep(120)

    def is_miner_registered(self) -> bool:
        self.miner_index = self.get_miner_index()
        if self.miner_index is not None:
            logger.info(
                f"Miner {self.bt_config.wallet.hotkey} is registered with uid {self.metagraph.uids[self.miner_index]}"
            )
            return True
        return False

    def handle_unregistered_miner(self) -> None:
        logger.warning(
            f"Miner {self.bt_config.wallet.hotkey} is not registered. Sleeping for 120 seconds..."
        )
        time.sleep(120)
        self.metagraph.sync(subtensor=self.subtensor)

    def nsfw_image_filter(self, images: List[bt.Tensor]) -> List[bool]:
        clip_input = self.processor(
            [self.transform(image) for image in images], return_tensors="pt"
        ).to(self.bt_config.miner.device)
        images, nsfw = self.safety_checker.forward(
            images=images,
            clip_input=clip_input.pixel_values.to(self.bt_config.miner.device),
        )
        return nsfw

    def get_miner_info(self) -> Dict[str, Union[int, float]]:
        try:
            return {
                "block": self.metagraph.block.item(),
                "stake": self.metagraph.stake[self.miner_index].item(),
                "trust": self.metagraph.trust[self.miner_index].item(),
                "consensus": self.metagraph.consensus[self.miner_index].item(),
                "incentive": self.metagraph.incentive[self.miner_index].item(),
                "emissions": self.metagraph.emission[self.miner_index].item(),
            }
        except Exception as e:
            logger.error(f"Error in get_miner_info: {e}")
            return {}

    def get_miner_index(self) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        except ValueError:
            return None

    def check_still_registered(self) -> bool:
        return self.get_miner_index() is not None

    def get_incentive(self) -> float:
        if self.miner_index is not None:
            return self.metagraph.I[self.miner_index].item() * 100_000
        return 0.0

    def get_trust(self) -> float:
        if self.miner_index is not None:
            return self.metagraph.T[self.miner_index].item() * 100
        return 0.0

    def get_consensus(self) -> float:
        if self.miner_index is not None:
            return self.metagraph.C[self.miner_index].item() * 100_000
        return 0.0

    async def is_alive(self, synapse: IsAlive) -> IsAlive:
        logger.info("IsAlive")
        synapse.completion = "True"
        return synapse

    def get_model_config(
        self, model_type: ModelType, task_type: TaskType
    ) -> ModelConfig:
        raise NotImplementedError("Please extend self.get_model_config")

    async def generate_image(self, synapse: ImageGeneration) -> ImageGeneration:
        timeout: float = synapse.timeout
        self.stats.total_requests += 1
        start_time: float = time.perf_counter()

        model_config = self._get_model_config(synapse)
        if model_config is None:
            return synapse

        model_args = setup_model_args(synapse, model_config)
        images = await self._generate_images_with_retries(
            model_args, synapse, model_config
        )

        if len(images) == 0:
            logger.info(f"Failed to generate any images after {3} attempts.")

        if time.perf_counter() - start_time > timeout:
            self.stats.timeouts += 1

        images = filter_nsfw_images(images)
        log_generation_time(start_time)

        synapse.images = [image_to_base64(image) for image in images]

        return synapse

    def _get_model_config(
        self, synapse: ImageGeneration
    ) -> Optional[ModelConfig]:
        try:
            model_type: str = synapse.model_type or ModelType.CUSTOM
            return self.get_model_config(
                model_type, synapse.generation_type.upper()
            )
        except ValueError as e:
            logger.error(f"Error getting model config: {e}")
            return None

    async def _generate_images_with_retries(
        self,
        model_args: Dict[str, Any],
        synapse: ImageGeneration,
        model_config: ModelConfig,
    ) -> List:
        images = []
        for attempt in range(3):
            try:
                seed: int = synapse.seed
                model_args["generator"] = [
                    torch.Generator(
                        device=self.bt_config.miner.device
                    ).manual_seed(seed)
                ]
                model_args["callback_on_step_end"] = SDXLCFGCutoffCallback(
                    cutoff_step_ratio=0.4
                )
                images = self.generate_with_refiner(model_args, model_config)
                logger.info(
                    f"Successful image generation after {attempt + 1} attempt(s)."
                )
                break
            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed to generate image: {e}"
                )
                await asyncio.sleep(5)
        return images

    def generate_with_refiner(
        self, model_args: Dict[str, Any], model_config: ModelConfig
    ) -> List:
        model = model_config.model.to(self.bt_config.miner.device)
        refiner = (
            model_config.refiner.to(self.bt_config.miner.device)
            if model_config.refiner
            else None
        )

        if refiner and self.bt_config.refiner.enable:
            refiner_args = setup_refiner_args(model_args)
            images = model(**model_args).images
            refiner_args["image"] = images
            images = refiner(**refiner_args).images
        else:
            images = model(
                **without_keys(model_args, ["denoising_end", "output_type"])
            ).images
        return images

    def _base_priority(self, synapse: Union[IsAlive, ImageGeneration]) -> float:
        caller_hotkey: str = synapse.axon.hotkey
        try:
            priority: float = 0.0
            if self.is_whitelisted(caller_hotkey=caller_hotkey):
                priority = 25000.0
                logger.info(
                    "Setting the priority of whitelisted key"
                    + f" {caller_hotkey} to {priority}"
                )

            try:
                caller_uid: int = self.metagraph.hotkeys.index(
                    synapse.axon.hotkey
                )
                priority = max(priority, float(self.metagraph.S[caller_uid]))
                logger.info(
                    f"Prioritizing key {synapse.axon.hotkey}"
                    + f" with value: {priority}."
                )
            except ValueError:
                logger.warning(
                    f"Hotkey {synapse.axon.hotkey}" + f" not found in metagraph"
                )

            return priority
        except Exception as e:
            logger.error(f"Error in _base_priority: {e}")
            return 0.0

    def _base_blacklist(
        self,
        synapse: Union[IsAlive, ImageGeneration],
        vpermit_tao_limit: float = VPERMIT_TAO,
        rate_limit: float = 1.0,
    ) -> Tuple[bool, str]:
        try:
            synapse_type: str = type(synapse).__name__
            caller_hotkey: str = synapse.dendrite.hotkey
            caller_coldkey: str = get_coldkey_for_hotkey(caller_hotkey)
            caller_stake: Optional[float] = get_caller_stake(synapse)

            exceeded_rate_limit: bool = self._check_rate_limit_exceeded(
                synapse_type, caller_hotkey, rate_limit
            )
            if self._is_whitelisted_or_registered(
                caller_hotkey,
                caller_coldkey,
                caller_stake,
                synapse_type,
                exceeded_rate_limit,
            ):
                return False, "Whitelisted or registered"

            if exceeded_rate_limit:
                return (
                    True,
                    f"Blacklisted {synapse_type} request from {caller_hotkey}. Rate limit exceeded.",
                )

            if caller_stake is None:
                return (
                    True,
                    f"Blacklisted non-registered hotkey {caller_hotkey}'s {synapse_type} request.",
                )

            if caller_stake < vpermit_tao_limit:
                return (
                    True,
                    f"Blacklisted {synapse_type} request from {caller_hotkey} due to low stake: {caller_stake:.2f} < {vpermit_tao_limit}",
                )

            return False, "Hotkey recognized"
        except Exception as e:
            logger.error(f"Error in blacklist: {traceback.format_exc()}")
            return True, f"Error in blacklist: {str(e)}"

    def _check_rate_limit_exceeded(
        self, synapse_type: str, caller_hotkey: str, rate_limit: float
    ) -> bool:
        if synapse_type != "ImageGeneration":
            return False

        now: float = time.perf_counter()
        if caller_hotkey in self.request_dict:
            delta: float = now - self.request_dict[caller_hotkey]["history"][-1]
            if delta < rate_limit:
                self.request_dict[caller_hotkey]["rate_limited_count"] += 1
                return True

            self.request_dict[caller_hotkey]["history"].append(now)
            self.request_dict[caller_hotkey]["delta"].append(delta)
            self.request_dict[caller_hotkey]["count"] += 1
        else:
            self.request_dict[caller_hotkey] = {
                "history": [time.perf_counter()],
                "delta": [0.0],
                "count": 0,
                "rate_limited_count": 0,
            }
        return False

    def _is_whitelisted_or_registered(
        self,
        caller_hotkey: str,
        caller_coldkey: str,
        caller_stake: Optional[float],
        synapse_type: str,
        exceeded_rate_limit: bool,
    ) -> bool:
        if self.is_whitelisted(
            caller_coldkey=caller_coldkey
        ) or self.is_whitelisted(caller_hotkey=caller_hotkey):
            logger.info(
                f"Whitelisting {synapse_type} request from {caller_hotkey}."
            )
            return True

        if exceeded_rate_limit:
            logger.info(
                f"Blacklisted a {synapse_type} request from {caller_hotkey}. Rate limit exceeded."
            )
            return False

        if caller_stake is None:
            logger.info(
                f"Blacklisted a non-registered hotkey's {synapse_type} request from {caller_hotkey}."
            )
            return False

        if caller_stake < VPERMIT_TAO:
            logger.info(
                f"Blacklisted a {synapse_type} request from {caller_hotkey} due to low stake: {caller_stake:.2f} < {VPERMIT_TAO}"
            )
            return False

        return True

    def blacklist_is_alive(self, synapse: IsAlive) -> Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def blacklist_image_generation(
        self, synapse: ImageGeneration
    ) -> Tuple[bool, str]:
        return self._base_blacklist(synapse)

    def priority_is_alive(self, synapse: IsAlive) -> float:
        return self._base_priority(synapse)

    def priority_image_generation(self, synapse: ImageGeneration) -> float:
        return self._base_priority(synapse)

    def loop(self) -> None:
        logger.info("Starting miner loop.", color="green")
        step: int = 0
        while True:
            try:
                if not self._check_and_handle_registration():
                    continue

                if step % 5 == 0:
                    self._output_statistics(step)

                step += 1
                time.sleep(60)

            except KeyboardInterrupt:
                self._handle_keyboard_interrupt()

            except Exception:
                self._handle_unforeseen_error()

    def _check_and_handle_registration(self) -> bool:
        is_registered: bool = self.check_still_registered()

        if not is_registered:
            logger.info("The miner is not currently registered.", color="red")
            time.sleep(120)
            self.metagraph.sync(subtensor=self.subtensor)
            return False

        return True

    def _output_statistics(self, step: int) -> None:
        log: str = (
            f"Step: {step} | "
            f"Block: {self.metagraph.block.item()} | "
            f"Stake: {self.metagraph.S[self.miner_index]:.2f} | "
            f"Rank: {self.metagraph.R[self.miner_index]:.2f} | "
            f"Trust: {self.metagraph.T[self.miner_index]:.2f} | "
            f"Consensus: {self.metagraph.C[self.miner_index]:.2f} | "
            f"Incentive: {self.metagraph.I[self.miner_index]:.2f} | "
            f"Emission: {self.metagraph.E[self.miner_index]:.2f}"
        )
        logger.info(log, color="green")

        top_requestors = self._get_top_requestors()
        total_requests_counted: int = sum([x[1] for x in top_requestors])

        try:
            top_requestors = sorted(
                top_requestors, key=lambda x: x[1], reverse=True
            )[:10]

            if len(top_requestors) > 0:
                formatted_str: str = "\n".join(
                    [
                        f"Hotkey: {x[0]}, "
                        f"Count: {x[1]} ({((x[1] / total_requests_counted) * 100) if total_requests_counted > 0 else 0:.2f}%), "
                        f"Average delta: {sum(x[2]) / len(x[2]) if len(x[2]) > 0 else 0:.2f}, "
                        f"Rate limited count: {x[3]}"
                        for x in top_requestors
                    ]
                )
                formatted_str = f"{formatted_str}"

                logger.info(f"{sh('Top Callers')} -> Metrics\n{formatted_str}")
        except Exception as e:
            logger.error(f"Error processing top requestors: {e}")

    def _get_top_requestors(self) -> List[Tuple[str, int, List[float], int]]:
        return [
            (k, v["count"], v["delta"], v["rate_limited_count"])
            for k, v in self.request_dict.items()
        ]

    def _handle_keyboard_interrupt(self) -> None:
        self.axon.stop()
        logger.success("Miner killed by keyboard interrupt.")
        sys.exit(0)

    def _handle_unforeseen_error(self) -> None:
        logger.error(f"Unexpected error: {traceback.format_exc()}")
