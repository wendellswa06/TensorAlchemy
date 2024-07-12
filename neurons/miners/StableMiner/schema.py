from enum import Enum
from typing import Type


import torch
from typing import Any, Dict, Optional

from diffusers import DiffusionPipeline
from pydantic import BaseModel


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "TEXT_TO_IMAGE"
    IMAGE_TO_IMAGE = "IMAGE_TO_IMAGE"


class ModelConfig(BaseModel):
    args: Dict[str, Any]
    model: DiffusionPipeline
    refiner: Optional[Type]


class TaskConfig(BaseModel):
    task_type: TaskType
    pipeline: Type
    torch_dtype: torch.dtype
    use_safetensors: bool
    variant: str
    scheduler: Optional[Type] = None
    safety_checker: Optional[Type] = None
    safety_checker_model_name: Optional[str] = None
    processor: Optional[Type] = None
    refiner_class: Optional[Type]
    refiner_model_name: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
