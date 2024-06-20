import bittensor as bt
import torch
from loguru import logger
from transformers import (
    CLIPImageProcessor,
)

from neurons.safety import StableDiffusionSafetyChecker
from neurons.validator import config as validator_config
from neurons.validator.rewards.models.base import BaseRewardModel
from neurons.validator.rewards.types import RewardModelType


class NSFWRewardModel(BaseRewardModel):
    @property
    def name(self) -> str:
        return RewardModelType.nsfw.value

    def __init__(self):
        super().__init__()
        self.device = validator_config.get_default_device()
        self.safetychecker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.device)
        self.processor = CLIPImageProcessor()

    def reward(self, response) -> float:
        # delete all none images
        for j, image in enumerate(response.images):
            if image is None:
                return 0.0

        if len(response.images) == 0:
            return 0.0
        try:
            clip_input = self.processor(
                [bt.Tensor.deserialize(image) for image in response.images],
                return_tensors="pt",
            ).to(self.device)
            images, has_nsfw_concept = self.safetychecker.forward(
                images=response.images,
                clip_input=clip_input.pixel_values.to(self.device),
            )

            any_nsfw = any(has_nsfw_concept)
            if any_nsfw:
                return 0.0

        except Exception as e:
            logger.error(f"Error in NSFW detection: {e}")
            logger.error(f"images={response.images}")
            return 1.0

        return 1.0

    async def get_rewards(self, responses, rewards, synapse=None) -> torch.FloatTensor:
        return torch.tensor(
            [
                self.reward(response) if reward != 0.0 else 0.0
                for response, reward in zip(responses, rewards)
            ],
            dtype=torch.float32,
        )

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards
