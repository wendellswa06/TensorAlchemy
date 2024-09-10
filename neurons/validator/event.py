from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from scoring.models.types import RewardModelType


class RewardScore(BaseModel):
    uids: List[int]
    scores: List[float]
    normalized: List[float]
    raw: Optional[List[float]] = None


class EventSchema(BaseModel):
    images: List[str]  # Assuming images are stored as base64 strings
    task_type: str
    block: float
    uids: List[int]
    hotkeys: List[str]
    prompt: str
    step_length: float
    model_type: str
    rewards: Dict[str, RewardScore]
    stake: List[float]
    rank: List[float]
    vtrust: List[float]
    dividends: List[float]
    emissions: List[float]
    set_weights: Optional[List[List[float]]] = None

    @classmethod
    def from_dict(cls, event_dict: dict) -> "EventSchema":
        rewards = {}
        for reward_type in RewardModelType:
            if reward_type.value in event_dict:
                reward_data = event_dict[reward_type.value]
                rewards[reward_type.value] = RewardScore(
                    uids=reward_data.uids.tolist(),
                    scores=reward_data.scores.tolist(),
                    normalized=reward_data.normalized.tolist(),
                    raw=reward_data.raw.tolist()
                    if reward_data.raw is not None
                    else None,
                )

        return cls(
            task_type=event_dict["task_type"],
            model_type=event_dict["model_type"],
            block=event_dict["block"],
            uids=event_dict["uids"].tolist(),
            hotkeys=event_dict["hotkeys"],
            prompt=event_dict["prompt"],
            step_length=event_dict["step_length"],
            images=[
                image.tolist() if hasattr(image, "tolist") else image
                for image in event_dict["images"]
            ],
            rewards=rewards,
            stake=event_dict["stake"],
            rank=event_dict["rank"],
            vtrust=event_dict["vtrust"],
            dividends=event_dict["dividends"],
            emissions=event_dict["emissions"],
            set_weights=event_dict.get("set_weights"),
        )

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


def convert_enum_keys_to_strings(data):
    if isinstance(data, dict):
        return {
            k.value
            if isinstance(k, Enum)
            else k: convert_enum_keys_to_strings(v)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [convert_enum_keys_to_strings(item) for item in data]
    elif isinstance(data, Enum):
        return data.value
    return data
