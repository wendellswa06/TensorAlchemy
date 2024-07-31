import traceback

import torch
import torch.nn.functional as F
import bittensor as bt

from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from typing import List

from neurons.utils.image import synapse_to_tensor
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

    def compute_clip_score(
        self,
        prompt_elements: PromptBreakdown,
        response: bt.Synapse,
    ) -> float:
        if not response.images or any(
            image is None for image in response.images
        ):
            return 0.0

        try:
            image = synapse_to_tensor(response)

            descriptions = [
                element["description"]
                for element in prompt_elements["elements"]
            ]

            # Process image once
            image_input = self.processor(images=image, return_tensors="pt").to(
                self.device
            )

            with torch.no_grad():
                image_features = self.model.get_image_features(**image_input)
                image_features = F.normalize(image_features, dim=-1)

                # Process all text descriptions at once
                text_inputs = self.processor(
                    text=descriptions,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)

                text_features = self.model.get_text_features(**text_inputs)
                text_features = F.normalize(text_features, dim=-1)

                # Compute cosine similarities for all descriptions
                similarities = F.cosine_similarity(
                    image_features.unsqueeze(1), text_features, dim=-1
                )

                # Apply softmax to get a distribution of scores
                # Adjust this value to control the "peakiness" of the distribution
                temperature = 0.1
                softmax_scores = F.softmax(similarities / temperature, dim=-1)

                # Compute weighted similarity score
                weighted_similarity = (
                    (softmax_scores * similarities).sum().item()
                )

                # Apply a non-linear transformation to amplify differences
                final_score = torch.tanh(
                    torch.tensor(weighted_similarity) * 5
                ).item()

                # Scale to [0, 1] range
                normalized_score = (final_score + 1) / 2

                logger.info(
                    f"Enhanced CLIP similarity score: {normalized_score:.4f}"
                )
                logger.info(
                    f"Raw weighted similarity: {weighted_similarity:.4f}"
                )

                for i, desc in enumerate(descriptions):
                    logger.info(
                        f"Element: {desc}, "
                        + f"Similarity: {similarities[0][i]}, "
                        + f"Softmax Score: {softmax_scores[0][i]}"
                    )

                return normalized_score

        except Exception:
            logger.error(traceback.format_exc())
            return 0.0

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

        rewards = await super().build_rewards_tensor(
            get_reward,
            synapse,
            responses,
        )

        # Log the rewards for each response
        for i, reward in enumerate(rewards):
            logger.info(f"Reward for response {i}: {reward.item():.4f}")

        return rewards
