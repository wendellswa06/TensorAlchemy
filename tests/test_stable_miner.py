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
from neurons.miners.StableMiner.schema import TaskType, TaskConfig
from neurons.miners.StableMiner.stable_miner import StableMiner
from neurons.protocol import ModelType
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from loguru import logger


class MockConfig:
    class Logging:
        debug = True
        logging_dir = "/tmp"

    class Wallet:
        name = "test_wallet"
        hotkey = "test_hotkey"

    class Wandb:
        project = ""
        entity = ""
        api_key = ""

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

    netuid = 1
    logging = Logging()
    wallet = Wallet()
    wandb = Wandb()
    miner = Miner()
    axon = Axon()


class TestStableMiner(unittest.TestCase):
    def setUp(self):
        patchers = [
            patch(
                "neurons.miners.StableMiner.stable_miner.ModelLoader.load",
                return_value=self.create_mock_model(),
            ),
            patch(
                "neurons.miners.StableMiner.stable_miner.ModelLoader.load_safety_checker",
                return_value=MagicMock(),
            ),
            patch(
                "neurons.miners.StableMiner.stable_miner.ModelLoader.load_processor",
                return_value=MagicMock(),
            ),
            patch(
                "neurons.miners.StableMiner.base.get_config", return_value=MockConfig()
            ),
            patch("bittensor.subtensor", autospec=True),
            patch("bittensor.wallet", autospec=True),
            patch("torch.compile", autospec=True),
            patch(
                "neurons.miners.StableMiner.stable_miner.StableMiner.start_axon",
                return_value=None,
            ),
            patch(
                "neurons.miners.StableMiner.stable_miner.StableMiner.loop_until_registered",
                return_value=None,
            ),
            patch("neurons.miners.StableMiner.base.BaseMiner.loop", return_value=None),
        ]
        self.mocks = [p.start() for p in patchers]
        self.addCleanup(lambda: [p.stop() for p in patchers])

    def create_mock_model(self):
        mock_model = MagicMock(spec=DiffusionPipeline)
        mock_model.unet = MagicMock()
        return mock_model

    def test_initialization(self):
        (
            mock_load_model,
            mock_load_safety_checker,
            mock_load_processor,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self.mocks

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

        self.assertIsNotNone(miner.safety_checkers)
        self.assertIsNotNone(miner.processors)
        self.assertIsNotNone(miner.model_configs)

    def test_load_model(self):
        mock_load_model, _, _, _, _, _, _, _, _, _ = self.mocks

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


if __name__ == "__main__":
    unittest.main()
