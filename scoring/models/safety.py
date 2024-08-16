"""
Simplified Stable Diffusion Safety Checker (Functional Version)

This module implements a streamlined safety checker for Stable Diffusion generated images.
It uses a CLIP-based vision model to detect potential NSFW content.

The implementation follows a functional programming paradigm for improved clarity and testability.
"""

from typing import List
import torch
from torch import nn
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from loguru import logger


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)
        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size,
            config.projection_dim,
            bias=False,
        )
        self.concept_embeds = nn.Parameter(
            torch.ones(17, config.projection_dim),
            requires_grad=False,
        )
        self.concept_thresholds = nn.Parameter(
            torch.ones(17),
            requires_grad=False,
        )

    def forward(self, clip_input: torch.Tensor) -> List[bool]:
        image_embeds = self._get_image_embeddings(clip_input)
        concept_scores = self._calculate_concept_scores(image_embeds)
        has_nsfw_concepts = self._check_nsfw_concepts(concept_scores)

        if any(has_nsfw_concepts):
            logger.warning(
                "Potential NSFW content was detected in one or more images. "
                "A black image will be returned instead. "
                "Try again with a different prompt and/or seed."
            )

        return has_nsfw_concepts

    def _get_image_embeddings(self, clip_input: torch.Tensor) -> torch.Tensor:
        pooled_output = self.vision_model(clip_input)[1]
        return self.visual_projection(pooled_output)

    def _calculate_concept_scores(
        self, image_embeds: torch.Tensor
    ) -> torch.Tensor:
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)
        return cos_dist - self.concept_thresholds

    @staticmethod
    def _check_nsfw_concepts(concept_scores: torch.Tensor) -> List[bool]:
        return (concept_scores > 0).any(dim=1).tolist()


def cosine_distance(
    image_embeds: torch.Tensor, concept_embeds: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the cosine distance between
    image embeddings and concept embeddings.

    Args:
    image_embeds (torch.Tensor): Image embeddings
    concept_embeds (torch.Tensor): Concept embeddings

    Returns:
    torch.Tensor: Cosine distance between the embeddings
    """
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_concept_embeds = nn.functional.normalize(concept_embeds)
    return torch.mm(normalized_image_embeds, normalized_concept_embeds.t())
