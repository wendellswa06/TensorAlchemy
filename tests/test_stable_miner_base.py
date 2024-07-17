import time

import pytest
from unittest.mock import MagicMock, patch
from bittensor import subtensor, wallet, metagraph

from neurons.miners.StableMiner.stable_miner import StableMiner
from neurons.protocol import IsAlive, ImageGeneration
from neurons.miners.StableMiner.schema import TaskConfig, ModelType, TaskType
import torch
from diffusers import DiffusionPipeline


class TestStableMinerAsBase:
    @pytest.fixture
    @patch("neurons.miners.StableMiner.base.get_bt_miner_config")
    @patch("bittensor.subtensor")
    @patch("bittensor.wallet")
    @patch("bittensor.metagraph")
    @patch("bittensor.axon")
    @patch("neurons.miners.StableMiner.wandb_utils.WandbUtils")
    def stable_miner(
        self,
        mock_wandb,
        mock_axon,
        mock_metagraph,
        mock_wallet,
        mock_subtensor,
        mock_get_bt_miner_config,
    ):
        task_configs = [
            TaskConfig(
                task_type=TaskType.TEXT_TO_IMAGE,
                model_type=ModelType.CUSTOM,
                pipeline=DiffusionPipeline,
                torch_dtype=torch.float32,
                use_safetensors=True,
                variant="default",
            )
        ]
        mock_config = MagicMock()
        mock_config.axon.port = 1234
        mock_config.axon.get.return_value = None
        mock_get_bt_miner_config.return_value = mock_config
        mock_subtensor.return_value = MagicMock()
        mock_wallet.return_value = MagicMock()
        mock_metagraph.return_value = MagicMock()
        mock_axon.return_value.attach.return_value.start.return_value = (
            mock_axon.return_value
        )

        with patch.object(StableMiner, "loop", return_value=None):
            miner = StableMiner(task_configs)
            miner.subtensor = mock_subtensor.return_value
            miner.wallet = mock_wallet.return_value
            miner.metagraph = mock_metagraph.return_value
            return miner

    def test_init(self, stable_miner):
        assert stable_miner.bt_config is not None
        assert stable_miner.wallet is not None
        assert stable_miner.subtensor is not None
        assert stable_miner.metagraph is not None

    @patch("bittensor.axon")
    @patch("bittensor.utils.networking.get_external_ip", return_value="127.0.0.1")
    @patch("bittensor.subtensor.serve_axon")
    def test_start_axon(
        self, mock_serve_axon, mock_get_external_ip, mock_axon, stable_miner
    ):
        stable_miner.wallet = MagicMock()
        stable_miner.bt_config.axon.port = 1234
        stable_miner.bt_config.axon.external_ip = "127.0.0.1"
        stable_miner.start_axon()
        assert mock_axon.call_count == 1
        assert stable_miner.axon is not None

    def test_loop_until_registered(self, stable_miner):
        call_count = 0

        def get_miner_index_side_effect():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None
            return 0

        stable_miner.get_miner_index = MagicMock(
            side_effect=get_miner_index_side_effect
        )
        stable_miner.metagraph.uids = [1]
        stable_miner.bt_config.wallet.hotkey = "test_hotkey"
        stable_miner.wallet.hotkey.ss58_address = "test_hotkey"

        with patch("time.sleep", return_value=None):
            stable_miner.loop_until_registered()
            assert stable_miner.miner_index == 0

    def test_get_miner_index(self, stable_miner):
        stable_miner.wallet.hotkey.ss58_address = "test_hotkey"
        stable_miner.metagraph.hotkeys = ["test_hotkey"]
        assert stable_miner.get_miner_index() == 0

    def test_check_still_registered(self, stable_miner):
        stable_miner.get_miner_index = MagicMock(return_value=1)
        assert stable_miner.check_still_registered() is True

    def test_get_miner_info(self, stable_miner):
        stable_miner.metagraph.block.item.return_value = 1
        stable_miner.metagraph.stake.__getitem__.return_value.item.return_value = 1.0
        stable_miner.metagraph.trust.__getitem__.return_value.item.return_value = 1.0
        stable_miner.metagraph.consensus.__getitem__.return_value.item.return_value = (
            1.0
        )
        stable_miner.metagraph.incentive.__getitem__.return_value.item.return_value = (
            1.0
        )
        stable_miner.metagraph.emission.__getitem__.return_value.item.return_value = 1.0
        miner_info = stable_miner.get_miner_info()
        assert miner_info == {
            "block": 1,
            "stake": 1.0,
            "trust": 1.0,
            "consensus": 1.0,
            "incentive": 1.0,
            "emissions": 1.0,
        }

    @patch("torchvision.transforms.Compose")
    def test_setup_model_args(self, mock_compose, stable_miner):
        synapse = MagicMock(spec=ImageGeneration)
        synapse.prompt = "test_prompt"
        synapse.width = 512
        synapse.height = 512
        synapse.num_images_per_prompt = 1
        synapse.guidance_scale = 7.5
        model_config = MagicMock()
        model_config.args = {}
        model_args = stable_miner.setup_model_args(synapse, model_config)
        assert model_args["prompt"] == ["test_prompt"]
        assert model_args["width"] == 512
        assert model_args["height"] == 512
        assert model_args["num_images_per_prompt"] == 1
        assert model_args["guidance_scale"] == 7.5

    @patch("neurons.miners.StableMiner.base.get_coldkey_for_hotkey")
    def test_base_priority(self, mock_get_coldkey_for_hotkey, stable_miner):
        synapse = MagicMock(spec=IsAlive)
        synapse.axon = MagicMock()
        synapse.axon.hotkey = "test_hotkey"
        stable_miner.hotkey_whitelist = {"test_hotkey"}
        stable_miner.coldkey_whitelist = {"test_coldkey"}

        mock_get_coldkey_for_hotkey.return_value = "test_coldkey"
        stable_miner.metagraph.hotkeys.index.return_value = 0
        stable_miner.metagraph.S.__getitem__.return_value = 25000.0

        priority = stable_miner._base_priority(synapse)
        assert priority == 25000.0

    @patch("neurons.miners.StableMiner.base.get_coldkey_for_hotkey")
    @patch("neurons.miners.StableMiner.base.get_caller_stake")
    def test_base_blacklist(
        self, mock_get_caller_stake, mock_get_coldkey_for_hotkey, stable_miner
    ):
        synapse = MagicMock(spec=IsAlive)
        synapse.dendrite = MagicMock()
        synapse.dendrite.hotkey = "test_hotkey"
        stable_miner.coldkey_whitelist = {"test_coldkey"}
        stable_miner.hotkey_whitelist = {"test_hotkey"}
        mock_get_coldkey_for_hotkey.return_value = "test_coldkey"
        mock_get_caller_stake = MagicMock(return_value=1000)
        stable_miner.request_dict = {
            "test_hotkey": {
                "history": [time.perf_counter() - 2],
                "delta": [2.0],
                "count": 1,
                "rate_limited_count": 0,
            }
        }
        is_blacklisted, reason = stable_miner._base_blacklist(synapse)
        assert reason == "Whitelisted coldkey recognized."
        assert is_blacklisted is False

    @patch.object(StableMiner, "_base_priority", return_value=100)
    def test_priority_is_alive(self, mock_base_priority, stable_miner):
        synapse = MagicMock(spec=IsAlive)
        priority = stable_miner.priority_is_alive(synapse)
        assert priority == 100

    @patch.object(StableMiner, "_base_blacklist", return_value=(False, "Allowed"))
    def test_blacklist_is_alive(self, mock_base_blacklist, stable_miner):
        synapse = MagicMock(spec=IsAlive)
        is_blacklisted, reason = stable_miner.blacklist_is_alive(synapse)
        assert is_blacklisted is False
        assert reason == "Allowed"
