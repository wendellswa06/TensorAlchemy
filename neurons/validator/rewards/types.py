from typing import List, Optional

import torch
from pydantic import BaseModel, Field

from neurons.validator.rewards.models.types import RewardModelType


class ScoringResult(BaseModel):
    type: RewardModelType
    scores: torch.Tensor
    normalized: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class ScoringResults(BaseModel):
    combined_scores: torch.Tensor
    scores: List[ScoringResult] = Field(default=[])

    class Config:
        arbitrary_types_allowed = True

    def get_score(self, to_find: RewardModelType) -> Optional[ScoringResult]:
        for item in self.scores:
            if item.type == to_find:
                return item

        return None

    def add_score(self, other: ScoringResult) -> None:
        self.scores.append(other)

    def add_scores(self, others: List[ScoringResult]) -> None:
        self.scores += others

    def update(self, other: "ScoringResults") -> None:
        self.scores += other.scores
