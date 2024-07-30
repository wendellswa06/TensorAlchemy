import bittensor as bt
import torch
from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from typing import List

from neurons.utils.image import synapse_to_tensors
from neurons.validator.config import get_device
from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType
from neurons.validator.scoring.models.rewards.enhanced_clip.utils import (
    break_down_prompt,
    PromptBreakdown,
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
        self.penalty_factor = 0.7  # Penalty factor for imperfect matches

    async def get_rewards(
        self,
        synapse: bt.Synapse,
        responses: List[bt.Synapse],
    ) -> torch.Tensor:
        prompt_elements: PromptBreakdown = await break_down_prompt(
            synapse.prompt
        )

        def get_reward(response: bt.Synapse) -> float:
            return self.compute_clip_score(prompt_elements, response)

        return await super().build_rewards_tensor(
            get_reward,
            synapse,
            responses,
        )

    def compute_clip_score(
        self,
        prompt_elements: PromptBreakdown,
        response: bt.Synapse,
    ) -> float:
        if not response.images:
            return 0.0

        if any(image is None for image in response.images):
            return 0.0

        try:
            image = synapse_to_tensors(response)[0]
            image_input = self.processor(
                images=image,
                return_tensors="pt",
            ).to(self.device)

            final_score = 1.0  # Start with a perfect score

            with torch.no_grad():
                image_features = self.model.get_image_features(**image_input)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                for element in prompt_elements["elements"]:
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

                    # Normalize similarity to [0, 1] range
                    normalized_similarity = similarity / 100.0

                    # Apply penalty for imperfect match
                    if normalized_similarity < 1.0:
                        element_score = (
                            normalized_similarity * self.penalty_factor
                        )
                    else:
                        element_score = 1.0

                    # Apply importance weighting
                    weighted_score = element_score ** element["importance"]

                    # Multiply the final score
                    final_score *= weighted_score

            logger.info(f"Enhanced CLIP similarity score: {final_score:.4f}")
            return final_score

        except Exception as e:
            logger.error(f"Error in Enhanced CLIP scoring: {e}")
            return 0.0
