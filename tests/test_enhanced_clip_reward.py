import pytest
from unittest.mock import patch, MagicMock, AsyncMock

import torch
from loguru import logger

from neurons.validator.scoring.models.rewards.enhanced_clip import (
    EnhancedClipRewardModel,
)

# Import the mock functions and fixtures
from tests.fixtures import mock_get_metagraph, TEST_IMAGES, generate_synapse

# Define the mock configurations
mock_configs = {
    "neurons.validator.config": {
        "get_metagraph": mock_get_metagraph,
        "get_openai_client": MagicMock(),
        "get_corcel_api_key": MagicMock(return_value="mock_api_key"),
    },
    "neurons.validator.scoring.models.base": {
        "get_metagraph": mock_get_metagraph
    },
    "neurons.validator.scoring.models.rewards.enhanced_clip.utils": {
        "openai_breakdown": AsyncMock(),
    },
}


# Define the patch decorator
def apply_patches(func):
    for module, mocks in mock_configs.items():
        func = patch.multiple(module, **mocks)(func)
    return func


@pytest.fixture
def mock_openai_response():
    return {
        "elements": [
            {"description": "Majestic eagle"},
            {"description": "Purple umbrella"},
            {"description": "Eagle carrying umbrella in talons"},
            {"description": "Soaring eagle"},
            {"description": "Bustling stadium"},
            {"description": "Awestruck spectators"},
            {"description": "Eagle flying over stadium"},
        ]
    }


@pytest.fixture
@apply_patches
def patched_model(mock_openai_response):
    mock_configs[
        "neurons.validator.scoring.models.rewards.enhanced_clip.utils"
    ]["openai_breakdown"].return_value = mock_openai_response
    model = EnhancedClipRewardModel()
    model.device = "cpu"
    return model


@pytest.fixture
def mock_synapse():
    return generate_synapse("hotkey_0", TEST_IMAGES["COMPLEX_A"])


@apply_patches
class TestEnhancedClipRewardModel:
    @pytest.mark.asyncio
    async def test_valid_image(self, patched_model, mock_synapse):
        with patch.object(
            patched_model, "compute_clip_score", return_value=0.7
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [mock_synapse]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)

    @pytest.mark.asyncio
    async def test_no_images(self, patched_model, mock_synapse):
        mock_synapse.images = []
        reward = await patched_model.get_rewards(mock_synapse, [mock_synapse])
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(r == 0.0 for r in reward)

    @pytest.mark.asyncio
    async def test_multiple_images(self, patched_model, mock_synapse):
        synapse1 = generate_synapse("hotkey_1", TEST_IMAGES["COMPLEX_B"])
        synapse2 = generate_synapse("hotkey_2", TEST_IMAGES["COMPLEX_C"])
        with patch.object(
            patched_model, "compute_clip_score", side_effect=[0.7, 0.8]
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [synapse1, synapse2]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)

    @pytest.mark.asyncio
    async def test_empty_prompt(self, patched_model, mock_synapse):
        mock_synapse.prompt = ""
        with patch.object(
            patched_model, "compute_clip_score", return_value=0.5
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [mock_synapse]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)

    @pytest.mark.asyncio
    async def test_different_image_sizes(self, patched_model, mock_synapse):
        synapse1 = generate_synapse("hotkey_1", TEST_IMAGES["COMPLEX_D"])
        synapse2 = generate_synapse("hotkey_2", TEST_IMAGES["COMPLEX_E"])
        with patch.object(
            patched_model, "compute_clip_score", side_effect=[0.7, 0.9]
        ):
            reward = await patched_model.get_rewards(
                mock_synapse, [synapse1, synapse2]
            )
        assert isinstance(reward, torch.Tensor)
        assert reward.shape == (10,)
        assert all(0 <= r <= 1 for r in reward)

    @pytest.mark.asyncio
    async def test_eagle_fish_umbrella(self, patched_model):
        right_synapse = generate_synapse(
            "hotkey_1", TEST_IMAGES["EAGLE_UMBRELLA"]
        )
        wrong_synapse = generate_synapse(
            "hotkey_0", TEST_IMAGES["ELEPHANT_BASKET"]
        )

        prompt = (
            "A majestic eagle, "
            + "carrying an purple umbrella in its talons, "
            + "soars over a bustling stadium filled with awestruck spectators."
        )
        right_synapse.prompt = prompt
        wrong_synapse.prompt = prompt

        rewards = await patched_model.get_rewards(
            right_synapse,
            [wrong_synapse, right_synapse],
        )

        logger.info(f"Basket reward: {rewards[0].item()}")
        logger.info(f"Umbrella reward: {rewards[1].item()}")

        assert isinstance(rewards, torch.Tensor)
        assert rewards.shape == (10,)
        assert rewards[0] < rewards[1], (
            f"Wrong image reward ({rewards[0].item()}) "
            + f"should be less than right image reward ({rewards[1].item()})"
        )
