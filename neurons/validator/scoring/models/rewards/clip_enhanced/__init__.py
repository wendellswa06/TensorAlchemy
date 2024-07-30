import bittensor as bt
import torch
from loguru import logger
from transformers import CLIPProcessor, CLIPModel

from neurons.utils.image import synapse_to_tensors
from neurons.validator.config import get_device
from neurons.validator.scoring.models.base import BaseRewardModel
from neurons.validator.scoring.models.types import RewardModelType
from neurons.validator.scoring.models.rewards.clip_enhanced.utils import (
    break_down_prompt,
)


# Enhanced CLIP Reward Model
# This model extends the base CLIP model
# to provide more nuanced scoring based on breaking
# down prompts into key elements and evaluating them individually.
class EnhancedClipRewardModel(BaseRewardModel):
    @property
    def name(self) -> RewardModelType:
        return RewardModelType.ENHANCED_CLIP

    def __init__(self):
        super().__init__()
        self.device = get_device()

        # Initialize CLIP model and processor
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    async def get_reward(self, response: bt.Synapse) -> float:
        # Validate input
        if not response.images or any(
            image is None for image in response.images
        ):
            return 0.0

        try:
            # Break down the prompt into key elements
            prompt_elements = await break_down_prompt(response.prompt)

            # Process the image
            image = synapse_to_tensors(response)[0]
            image_input = self.processor(
                images=image,
                return_tensors="pt",
            ).to(self.device)

            total_score = 0
            total_importance = 0

            # Evaluate each prompt element
            for element in prompt_elements["elements"]:
                # Process the text description of the element
                text_input = self.processor(
                    text=[element["description"]],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                # Extract features from both image and text
                with torch.no_grad():
                    image_features = self.model.get_image_features(
                        **image_input
                    )
                    text_features = self.model.get_text_features(**text_input)

                # Normalize features
                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )
                text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )

                # Calculate similarity score
                similarity = (
                    (100.0 * image_features @ text_features.T).squeeze().item()
                )

                # Weight the score by the element's importance
                weighted_score = similarity * element["importance"]
                total_score += weighted_score
                total_importance += element["importance"]

            # Calculate final score
            final_score = (
                total_score / total_importance if total_importance > 0 else 0
            )

            logger.info(f"Enhanced CLIP similarity score: {final_score:.4f}")
            return final_score

        except Exception as e:
            logger.error(f"Error in Enhanced CLIP scoring: {e}")
            return 0.0
