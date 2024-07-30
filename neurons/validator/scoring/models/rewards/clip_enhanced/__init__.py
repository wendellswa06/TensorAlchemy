import bittensor as bt
import torch
from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict

from neurons.utils.image import synapse_to_tensors
from neurons.validator.config import get_device
from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType
from neurons.validator.scoring.models.rewards.clip_enhanced.utils import (
    break_down_prompt,
)


class EnhancedClipRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.ENHANCED_CLIP

    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.prompt_elements = None

    async def prepare_prompt_elements(self, prompt: str):
        self.prompt_elements = await break_down_prompt(prompt)

    async def get_rewards(
        self, synapse: bt.Synapse, responses: List[bt.Synapse]
    ) -> torch.Tensor:
        if not self.prompt_elements:
            await self.prepare_prompt_elements(synapse.prompt)

        def get_reward(response: bt.Synapse) -> float:
            return self.compute_clip_score(response)

        return await super().build_rewards_tensor(
            get_reward, synapse, responses
        )

    def compute_clip_score(self, response: bt.Synapse) -> float:
        if not response.images or any(
            image is None for image in response.images
        ):
            return 0.0

        try:
            image = synapse_to_tensors(response)[0]
            image_input = self.processor(images=image, return_tensors="pt").to(
                self.device
            )

            total_score = 0
            total_importance = 0

            with torch.no_grad():
                image_features = self.model.get_image_features(**image_input)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                for element in self.prompt_elements["elements"]:
                    logger.info(
                        f"EnhancedCLIP testing against: {element['description']}"
                    )
                    text_input = self.processor(
                        text=[element["description"]],
                        return_tensors="pt",
                        padding=True,
                    ).to(self.device)

                    text_features = self.model.get_text_features(**text_input)
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )

                    similarity = (
                        (100.0 * image_features @ text_features.T)
                        .squeeze()
                        .item()
                    )
                    weighted_score = similarity * element["importance"]
                    total_score += weighted_score
                    total_importance += element["importance"]

            final_score = (
                total_score / total_importance if total_importance > 0 else 0
            )
            logger.info(f"Enhanced CLIP similarity score: {final_score:.4f}")
            return final_score

        except Exception as e:
            logger.error(f"Error in Enhanced CLIP scoring: {e}")
            return 0.0
