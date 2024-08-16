import torch
from typing import List, Optional

from loguru import logger

from neurons.protocol import ModelType
from neurons.config import get_config

from neurons.miners.StableMiner.model_loader import ModelLoader
from neurons.miners.StableMiner.schema import (
    TaskType,
    TaskConfig,
    MinerConfig,
    TaskModelConfig,
)

from neurons.miners.StableMiner.base import BaseMiner
from neurons.miners.StableMiner.utils import warm_up


class StableMiner(BaseMiner):
    def __init__(self, task_configs: List[TaskConfig]) -> None:
        logger.info("Starting StableMiner initialization")
        super().__init__()

        self.task_configs = task_configs
        self.miner_config = MinerConfig()
        # TODO: Fix safety checker and processor to allow different values for each task config
        self.safety_checker: Optional[torch.nn.Module] = None
        self.processor: Optional[torch.nn.Module] = None

        logger.info("Initializing StableMiner...")
        self.initialize_all_models()
        self.optimize_models()
        self.start_axon()
        self.loop()

    def initialize_all_models(self) -> None:
        for task_config in self.task_configs:
            logger.info(
                f"Initializing models for task: {task_config.task_type}..."
            )
            self.initialize_model_for_task(task_config)
        self.setup_model_configs()
        self.log_gpu_memory_usage("after initializing models")

    def initialize_model_for_task(self, task_config: TaskConfig) -> None:
        self.log_gpu_memory_usage("before freeing cache")
        torch.cuda.empty_cache()
        self.log_gpu_memory_usage("after freeing cache")
        logger.info(f"Loading model for task: {task_config.task_type}")
        model = self.load_model(
            get_config().miner.custom_model, task_config.task_type
        )

        if task_config.model_type not in self.miner_config.model_configs:
            self.miner_config.model_configs[task_config.model_type] = {}

        self.miner_config.model_configs[task_config.model_type][
            task_config.task_type
        ] = TaskModelConfig(model=model)

        if (
            task_config.safety_checker
            and task_config.safety_checker_model_name
            and not self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].safety_checker
        ):
            self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].safety_checker = ModelLoader(
                get_config().miner
            ).load_safety_checker(
                task_config.safety_checker,
                task_config.safety_checker_model_name,
            )
            # TODO: temporary hack so nsfw_image_filter works; refactor later to allow different safety_checkers
            self.safety_checker = self.miner_config.model_configs[
                task_config.model_type
            ][task_config.task_type].safety_checker

        if (
            task_config.processor
            and not self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].processor
        ):
            self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].processor = ModelLoader(get_config().miner).load_processor(
                task_config.processor
            )
            # TODO: temporary hack so nsfw_image_filter works; refactor later to allow different safety_checkers
            self.processor = self.miner_config.model_configs[
                task_config.model_type
            ][task_config.task_type].processor

        if (
            get_config().refiner.enable
            and task_config.refiner_class
            and task_config.refiner_model_name
        ):
            logger.info(f"Loading refiner for task: {task_config.task_type}")
            self.miner_config.model_configs[task_config.model_type][
                task_config.task_type
            ].refiner = ModelLoader(get_config()).load_refiner(
                model, task_config
            )
            logger.info(f"Refiner loaded for task: {task_config.task_type}")
            self.log_gpu_memory_usage(
                f"after loading model for task {task_config.task_type}"
            )

    def get_model_config(
        self, model_type: ModelType, task_type: TaskType
    ) -> TaskModelConfig:
        if model_type not in self.miner_config.model_configs:
            raise ValueError(f"{model_type} was not found in model_configs!")
        if task_type not in self.miner_config.model_configs[model_type]:
            raise ValueError(
                f"{task_type} was not found in model_configs {model_type}!"
            )

        return self.miner_config.model_configs[model_type][task_type]

    def get_config_for_task_type(self, task_type: TaskType):
        for config in self.task_configs:
            if config.task_type == task_type:
                return config
        logger.info(f" No config found for task type {task_type}..")
        return None

    def load_model(
        self, model_name: str, task_type: TaskType
    ) -> torch.nn.Module:
        try:
            logger.info(f"Loading model {model_name} for task {task_type}...")
            task_config = self.get_config_for_task_type(task_type)
            model_loader = ModelLoader(get_config())
            model = model_loader.load(model_name, task_config)
            logger.info(f"Model {model_name} loaded successfully.")
            return model
        except Exception as e:
            logger.error(
                f"Error loading {task_type.value} model: {e}, skipping..."
            )

    def setup_model_configs(self) -> None:
        logger.info("Setting up model configurations...")

        for task_config in self.task_configs:
            model_type = task_config.model_type
            task_type = task_config.task_type

            if model_type not in self.miner_config.model_configs:
                self.miner_config.model_configs[model_type] = {}

            self.miner_config.model_configs[model_type][
                task_type
            ].args = self.get_args_for_task(task_type)

        logger.info("Model configurations set up successfully.")

    def get_args_for_task(self, task_type: TaskType) -> dict:
        if task_type == TaskType.TEXT_TO_IMAGE:
            return self.t2i_args
        elif task_type == TaskType.IMAGE_TO_IMAGE:
            return self.i2i_args
        else:
            return {}

    def log_gpu_memory_usage(self, stage: str) -> None:
        try:
            allocated = torch.cuda.memory_allocated() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            free = total - allocated

            logger.info(f"GPU memory allocated {stage}: {allocated:.2f} MB")
            logger.info(
                f"Max GPU memory allocated {stage}: {max_allocated:.2f} MB"
            )
            logger.info(f"Total GPU memory: {total:.2f} MB")
            logger.info(f"Free GPU memory: {free:.2f} MB")
        except Exception as e:
            logger.error(f"Failed to log GPU memory usage {stage}: {str(e)}")

    def optimize_models(self) -> None:
        logger.info("Optimizing models...")
        if not get_config().miner.optimize:
            logger.info("Model optimization is disabled.")
            return

        try:
            for model_type, tasks in self.miner_config.model_configs.items():
                for task_type, config in tasks.items():
                    if config.model:
                        logger.info(f"Compiling model for task: {task_type}")
                        config.model.unet = torch.compile(
                            config.model.unet,
                            mode="reduce-overhead",
                            fullgraph=True,
                        )
                        logger.info(f"Warming up model for task: {task_type}")
                        warm_up(config.model, config.args)
            logger.info("Models optimized successfully.")
        except Exception as e:
            logger.error(f"Error optimizing models: {e}")
            raise
