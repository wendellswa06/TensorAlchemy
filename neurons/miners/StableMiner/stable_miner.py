from typing import Dict, List
import torch
from base import BaseMiner
from loguru import logger

from neurons.miners.StableMiner.model_loader import ModelLoader
from neurons.miners.StableMiner.schema import ModelConfig, TaskType, TaskConfig
from neurons.protocol import ModelType
from utils import warm_up


class StableMiner(BaseMiner):
    def __init__(self, task_configs: List[TaskConfig]) -> None:
        self.task_configs = task_configs
        self.model_configs = self.initialize_nested_dict()
        self.safety_checkers = self.initialize_nested_dict()
        self.processors = self.initialize_nested_dict()
        self.refiners = self.initialize_nested_dict()
        self.models = self.initialize_nested_dict()

        super().__init__()

        try:
            logger.info("Initializing StableMiner...")
            self.initialize_all_models()
            self.optimize_models()
            self.start_axon()
            self.loop()
        except Exception as e:
            logger.error(f"Error in StableMiner initialization: {e}")
            raise

    def initialize_nested_dict(
        self,
    ) -> Dict[ModelType, Dict[TaskType, torch.nn.Module]]:
        nested_dict = {}
        for task_config in self.task_configs:
            if task_config.model_type not in nested_dict:
                nested_dict[task_config.model_type] = {}
        return nested_dict

    def initialize_all_models(self) -> None:
        try:
            for task_config in self.task_configs:
                logger.info(f"Initializing models for task: {task_config.task_type}...")
                self.initialize_model_for_task(task_config)
                self.setup_model_configs()
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def initialize_model_for_task(self, task_config: TaskConfig) -> None:
        self.ensure_nested_dict_entries(task_config)

        self.models[task_config.model_type][task_config.task_type] = self.load_model(
            self.config.miner.custom_model, task_config.task_type
        )

        if task_config.safety_checker and task_config.safety_checker_model_name:
            self.safety_checkers[task_config.model_type][
                task_config.task_type
            ] = ModelLoader(self.config.miner).load_safety_checker(
                task_config.safety_checker, task_config.safety_checker_model_name
            )
            logger.info(f"Safety checker loaded for task: {task_config.task_type}")

        if task_config.processor:
            self.processors[task_config.model_type][
                task_config.task_type
            ] = ModelLoader(self.config.miner).load_processor(task_config.processor)
            logger.info(f"Processor loaded for task: {task_config.task_type}")

        if task_config.refiner and task_config.refiner_model_name:
            self.refiners[task_config.model_type][task_config.task_type] = ModelLoader(
                self.config.miner
            ).load_refiner(
                self.models[task_config.model_type][task_config.task_type], task_config
            )
            logger.info(f"Refiner loaded for task: {task_config.task_type}")

    def ensure_nested_dict_entries(self, task_config: TaskConfig) -> None:
        model_type = task_config.model_type
        task_type = task_config.task_type

        if task_type not in self.models[model_type]:
            self.models[model_type][task_type] = None
        if (
            task_config.safety_checker
            and task_type not in self.safety_checkers[model_type]
        ):
            self.safety_checkers[model_type][task_type] = None
        if task_config.processor and task_type not in self.processors[model_type]:
            self.processors[model_type][task_type] = None
        if task_config.refiner and task_type not in self.refiners[model_type]:
            self.refiners[model_type][task_type] = None

    def get_model_config(
        self, model_type: ModelType, task_type: TaskType
    ) -> ModelConfig:
        if model_type not in self.model_configs:
            raise ValueError(f"{model_type} was not found in model_configs!")
        if task_type not in self.model_configs[model_type]:
            raise ValueError(
                f"{task_type} was not found in model_configs {model_type}!"
            )

        return self.model_configs[model_type][task_type]

    def load_model(self, model_name: str, task_type: TaskType) -> torch.nn.Module:
        try:
            logger.info(f"Loading model {model_name} for task {task_type}...")
            config = next(tc for tc in self.task_configs if tc.task_type == task_type)
            model_loader = ModelLoader(self.config.miner)
            model = model_loader.load(model_name, config)
            logger.info(f"Model {model_name} loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Error loading {task_type.value} model: {e}")
            raise

    def setup_model_configs(self) -> None:
        logger.info("Setting up model configurations...")

        for task_config in self.task_configs:
            model_type = task_config.model_type
            task_type = task_config.task_type

            if model_type not in self.model_configs:
                self.model_configs[model_type] = {}

            self.model_configs[model_type][task_type] = ModelConfig(
                args=self.get_args_for_task(task_type),
                model=self.models[model_type][task_type],
                refiner=self.refiners[model_type].get(task_type, None),
            )

        logger.info("Model configurations set up successfully.")

    def get_args_for_task(self, task_type: TaskType) -> dict:
        if task_type == TaskType.TEXT_TO_IMAGE:
            return self.t2i_args
        elif task_type == TaskType.IMAGE_TO_IMAGE:
            return self.i2i_args
        else:
            return {}

    def optimize_models(self) -> None:
        logger.info("Optimizing models...")
        if not self.config.miner.optimize:
            logger.info("Model optimization is disabled.")
            return

        try:
            for model_type, tasks in self.model_configs.items():
                for task_type, config in tasks.items():
                    if config.model:
                        config.model.unet = torch.compile(
                            config.model.unet,
                            mode="reduce-overhead",
                            fullgraph=True,
                        )
                        logger.info(
                            f">>> Warming up {model_type} {task_type} model with compile... "
                            "this takes roughly two minutes..."
                        )
                        warm_up(config.model, config.args)
            logger.info("Models optimized successfully.")
        except Exception as e:
            logger.error(f"Error optimizing models: {e}")
            raise
