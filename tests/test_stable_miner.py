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
            "mock_get_miner_config": patch(
                "neurons.miners.StableMiner.base.get_miner_config",
                return_value=MockConfig(),
            ),
            "mock_get_metagraph": patch(
                "neurons.config.get_metagraph",
                return_value=self.create_mock_metagraph(),
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

    def create_mock_metagraph(self):
        mock_metagraph = MagicMock()
        mock_metagraph.hotkeys = ["test_hotkey_1", "test_hotkey_2"]
        mock_metagraph.S = torch.tensor([1.0, 2.0])
        mock_metagraph.T = torch.tensor([0.5, 0.7])
        mock_metagraph.C = torch.tensor([0.8, 0.9])
        mock_metagraph.I = torch.tensor([0.3, 0.4])
        mock_metagraph.E = torch.tensor([0.2, 0.3])
        return mock_metagraph

    def test_initialization(self):
        # ... (keep the existing test_initialization method as is)

    def test_load_model(self):
        # ... (keep the existing test_load_model method as is)

    def test_get_miner_index(self):
        task_configs = [
            TaskConfig(
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
        ]

        with patch("neurons.config.get_wallet") as mock_get_wallet:
            mock_wallet = MagicMock()
            mock_wallet.hotkey.ss58_address = "test_hotkey_1"
            mock_get_wallet.return_value = mock_wallet

            miner = StableMiner(task_configs)
            miner_index = miner.get_miner_index()

            self.assertEqual(miner_index, 0)

    def test_get_miner_info(self):
        task_configs = [
            TaskConfig(
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
        ]

        miner = StableMiner(task_configs)
        miner.miner_index = 0

        miner_info = miner.get_miner_info()

        expected_info = {
            "block": self.mocks["mock_get_metagraph"].return_value.block.item(),
            "stake": 1.0,
            "trust": 0.5,
            "consensus": 0.8,
            "incentive": 0.3,
            "emissions": 0.2,
        }

        self.assertEqual(miner_info, expected_info)
