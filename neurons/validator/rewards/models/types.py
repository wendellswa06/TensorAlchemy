from pydantic import BaseModel
from enum import Enum
from typing import Dict, Tuple

import torch

from neurons.validator.rewards.models.base import BaseRewardModel


class RewardModelType(str, Enum):
    # Masking models
    # TODO: Maybe move these out
    NSFW = "NSFW"
    BLACKLIST = "BLACKLIST"

    # Reward models
    EMPTY = "EMPTY"
    HUMAN = "HUMAN"
    IMAGE = "IMAGE"
    SIMILARITY = "SIMILARITY"


class PackedRewardModel(BaseModel):
    weight: float

    model: BaseRewardModel

    class Config:
        arbitrary_types_allowed = True

    @property
    def name(self) -> RewardModelType:
        return self.model.name

    def apply(
        self,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        return self.model.apply(*args, **kwargs)


ModelStorage = Dict[RewardModelType, PackedRewardModel]
