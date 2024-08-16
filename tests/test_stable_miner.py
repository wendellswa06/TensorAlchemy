import unittest
from unittest.mock import patch, MagicMock
from diffusers import (
    AutoPipelineForText2Image,
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
)
import torch
from neurons.miners.StableMiner.model_loader import ModelLoader
from neurons.miners.StableMiner.schema import TaskType, TaskConfig, MinerConfig
from neurons.miners.StableMiner.stable_miner import StableMiner
from neurons.protocol import ModelType
from scoring.models.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from loguru import logger


class MockConfig:
    class Logging:
        debug = True
        logging_dir = "/tmp"

    class Wallet:
        name = "test_wallet"
        hotkey = "test_hotkey"

    class Miner:
        device = "cuda:0"
        optimize = True
        seed = 42
        custom_model = "stabilityai/stable-diffusion-xl-base-1.0"
        custom_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"
        alchemy_model = "stabilityai/stable-diffusion-xl-base-1.0"
        alchemy_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

    class Axon:
        port = 8080
        external_ip = "127.0.0.1"

    class Refiner:
        enable = "true"

    netuid = 1
    logging = Logging()
    wallet = Wallet()
    miner = Miner()
    axon = Axon()
    refiner = Refiner()
    model_configs = MinerConfig(model_configs={})
    full_path = "/tmp/test_wallet/test_hotkey/netuid1/miner"


class TestStableMiner(unittest.TestCase):
    def setUp(self):
        self.patches = {
            "mock_load_model": patch(
                "neurons.miners.StableMiner.stable_miner.ModelLoader.load",
                return_value=self.create_mock_model(),
            ),
            "mock_load_safety_checker": patch(
                "neurons.miners.StableMiner.stable_miner.ModelLoader.load_safety_checker",
                return_value=MagicMock(),
            ),
            "mock_load_processor": patch(
                "neurons.miners.StableMiner.stable_miner.ModelLoader.load_processor",
                return_value=MagicMock(),
            ),
            "mock_subtensor": patch("bittensor.subtensor", autospec=True),
            "mock_wallet": patch("bittensor.wallet", autospec=True),
            "mock_compile": patch("torch.compile", autospec=True),
            "mock_start_axon": patch(
                "neurons.miners.StableMiner.stable_miner.StableMiner.start_axon",
                return_value=None,
            ),
            "mock_loop_until_registered": patch(
                "neurons.miners.StableMiner.stable_miner.StableMiner.loop_until_registered",
                return_value=None,
            ),
            "mock_base_loop": patch(
                "neurons.miners.StableMiner.base.BaseMiner.loop",
                return_value=None,
            ),
            "mock_get_bt_miner_config": patch(
                "neurons.miners.StableMiner.base.get_bt_miner_config",
                return_value=MockConfig(),
            ),
        }
        self.mocks = {
            name: patcher.start() for name, patcher in self.patches.items()
        }
        self.addCleanup(
            lambda: [patcher.stop() for patcher in self.patches.values()]
        )

    def create_mock_model(self):
        mock_model = MagicMock(spec=DiffusionPipeline)
        mock_model.unet = MagicMock()
        return mock_model

    def test_initialization(self):
        mock_load_model = self.mocks["mock_load_model"]
        mock_load_safety_checker = self.mocks["mock_load_safety_checker"]
        mock_load_processor = self.mocks["mock_load_processor"]

        task_configs = [
            TaskConfig(
                model_type=ModelType.CUSTOM,
                task_type=TaskType.TEXT_TO_IMAGE,
                pipeline=AutoPipelineForText2Image,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                scheduler=DPMSolverMultistepScheduler,
                safety_checker=StableDiffusionSafetyChecker,
                safety_checker_model_name="dummy_safety_checker_model_name",
                processor=CLIPImageProcessor,
            ),
            TaskConfig(
                model_type=ModelType.CUSTOM,
                task_type=TaskType.IMAGE_TO_IMAGE,
                pipeline=AutoPipelineForImage2Image,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                scheduler=DPMSolverMultistepScheduler,
                safety_checker=StableDiffusionSafetyChecker,
                safety_checker_model_name="dummy_safety_checker_model_name",
                processor=CLIPImageProcessor,
            ),
        ]

        logger.info("Creating StableMiner instance")
        miner = StableMiner(task_configs)

        self.assertEqual(mock_load_model.call_count, 2)
        self.assertEqual(mock_load_safety_checker.call_count, 2)
        self.assertEqual(mock_load_processor.call_count, 2)

        self.assertIsNotNone(
            miner.miner_config.model_configs[ModelType.CUSTOM][
                TaskType.TEXT_TO_IMAGE
            ].safety_checker
        )
        self.assertIsNotNone(
            miner.miner_config.model_configs[ModelType.CUSTOM][
                TaskType.TEXT_TO_IMAGE
            ].processor
        )
        self.assertIsNotNone(
            miner.miner_config.model_configs[ModelType.CUSTOM][
                TaskType.IMAGE_TO_IMAGE
            ].safety_checker
        )
        self.assertIsNotNone(
            miner.miner_config.model_configs[ModelType.CUSTOM][
                TaskType.IMAGE_TO_IMAGE
            ].processor
        )
        self.assertIsNotNone(miner.miner_config.model_configs)

    def test_load_model(self):
        mock_load_model = self.mocks["mock_load_model"]

        task_config = TaskConfig(
            model_type=ModelType.CUSTOM,
            task_type=TaskType.TEXT_TO_IMAGE,
            pipeline=AutoPipelineForText2Image,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            scheduler=DPMSolverMultistepScheduler,
            safety_checker=None,
            processor=None,
        )

        loader = ModelLoader(config=MagicMock())

        logger.info("Loading model in test_load_model")
        model = loader.load("dummy_model_name", task_config)

        self.assertEqual(model, mock_load_model.return_value)
        mock_load_model.assert_called_once_with("dummy_model_name", task_config)
