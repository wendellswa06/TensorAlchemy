from enum import Enum
from typing import Type


import torch
from typing import Any, Dict, Optional

from diffusers import DiffusionPipeline
from pydantic import BaseModel
from pydantic.config import ConfigDict

from neurons.protocol import ModelType


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    args: Dict[str, Any]
    model: DiffusionPipeline
    model_config = ConfigDict(arbitrary_types_allowed=True)
    refiner: Optional[Any] = None


class TaskConfig(BaseModel):
    model_type: ModelType
    task_type: TaskType
    pipeline: Type
    torch_dtype: torch.dtype
    use_safetensors: bool
    variant: str
    scheduler: Optional[Type] = None
    safety_checker: Optional[Type] = None
    safety_checker_model_name: Optional[str] = None
    processor: Optional[Type] = None
    refiner_class: Optional[Type] = None
    refiner_model_name: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
