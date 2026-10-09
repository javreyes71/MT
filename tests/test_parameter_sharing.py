"""Tests F4: ParameterSharingVecEnv + MaskablePPO + action masks."""
import pytest
import numpy as np
from unittest.mock import MagicMock

class FakeMultiEnv:
    def __init__(self, n_agents=3, max_phases=4, obs_dim=20):
        from gymnasium import spaces
        self.agent_ids = [f"tls_{i}" for i in range(n_agents)]
        self.num_agents = n_agents
        self.max_phases = max_phases
        self._obs_dim = obs_dim
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(max_phases)
        self._action_dims = [2, 3, 4][:n_agents]

    def get_action_mask(self, agent_id):
        idx = self.agent_ids.index(agent_id)
        n_phases = self._action_dims[idx]
        mask = np.zeros(self.max_phases, dtype=np.float32)
        mask[:n_phases] = 1.0
        return mask

    def reset(self):
        return {aid: np.random.randn(self._obs_dim).astype(np.float32) for aid in self.agent_ids}

    def step(self, actions):
        obs = {aid: np.random.randn(self._obs_dim).astype(np.float32) for aid in self.agent_ids}
        rewards = {aid: -0.5 for aid in self.agent_ids}
        dones = {aid: False for aid in self.agent_ids}
        truncs = {aid: False for aid in self.agent_ids}
        infos = {aid: {} for aid in self.agent_ids}
        return obs, rewards, dones, truncs, infos

    def close(self): pass


class TestParameterSharingVecEnv:
    def setup_method(self):
        from src.agents.parameter_sharing import ParameterSharingVecEnv
        self.multi_env = FakeMultiEnv(n_agents=3, max_phases=4, obs_dim=20)
        self.wrapper = ParameterSharingVecEnv(self.multi_env)

    def test_spaces(self):
        assert self.wrapper.observation_space.shape == (20,)
        assert self.wrapper.action_space.n == 4
        assert self.wrapper.num_envs == 3

    def test_reset_returns_batch(self):
        obs = self.wrapper.reset()
        assert obs.shape == (3, 20)

    def test_action_masks_shape(self):
        self.wrapper.reset()
        mask = self.wrapper.action_masks()
        assert mask.shape == (3, 4)
        assert mask.dtype == bool
        
        # Agent 0: 2 phases
        assert mask[0, 0] == True
        assert mask[0, 2] == False
        
        # Agent 2: 4 phases
        assert mask[2, 3] == True

    def test_step_batch(self):
        self.wrapper.reset()
        actions = np.array([0, 1, 2])
        obs, rewards, dones, infos = self.wrapper.step(actions)
        assert obs.shape == (3, 20)
        assert rewards.shape == (3,)
        assert dones.shape == (3,)
        assert len(infos) == 3
        assert infos[0]["agent_id"] == "tls_0"

class TestMaskablePPOImport:
    def test_sb3_contrib_available(self):
        from sb3_contrib import MaskablePPO
        assert MaskablePPO is not None
