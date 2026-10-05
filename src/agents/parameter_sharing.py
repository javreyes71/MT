"""SharedPolicyWrapper v2 + ParameterSharingAgent con MaskablePPO.

Corrige:
  B2 — SharedPolicyWrapper roto: recompensa intermedia era 0.0 y ahora
       se usa un diseño sincrónico que no parte los episodios.
  B3 — PPO vanilla ignora acciones inválidas → MaskablePPO con máscara.
  B8 — Recompensa no normalizada → VecNormalize(reward=True).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from typing import Dict, Any, Tuple, List

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.environment.multi_agent_env import MultiAgentTrafficEnv


class SharedPolicyWrapper(gym.Env):
    """Wrapper sincrónico que expone un step() Gymnasium estándar.

    En cada step() recolecta la acción de UN agente y la almacena.
    Cuando todos los agentes han decidido, ejecuta el paso real en SUMO,
    calcula la recompensa global y la devuelve igualmente a cada agente
    en las siguientes N llamadas.

    Compatible con MaskablePPO: implementa action_masks().
    """

    def __init__(self, multi_env: MultiAgentTrafficEnv):
        super().__init__()
        self.multi_env = multi_env
        self.observation_space = multi_env.observation_space
        self.action_space = multi_env.action_space

        self.agent_ids = multi_env.agent_ids
        self.num_agents = len(self.agent_ids)

        # Estado interno
        self._agent_idx = 0
        self._obs_dict: Dict[str, np.ndarray] = {}
        self._actions: Dict[str, int] = {}
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        self._info: Dict[str, Any] = {}

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        self._obs_dict = self.multi_env.reset()
        self._agent_idx = 0
        self._actions = {}
        self._reward = 0.0
        self._terminated = False
        self._truncated = False
        self._info = {}

        first_agent = self.agent_ids[0]
        return self._obs_dict[first_agent], {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_agent = self.agent_ids[self._agent_idx]
        self._actions[current_agent] = int(action)
        self._agent_idx += 1

        if self._agent_idx >= self.num_agents:
            # Todos decidieron → ejecutar paso real en SUMO
            obs, rewards, dones, truncs, infos = self.multi_env.step(self._actions)

            self._obs_dict = obs
            reward_vals = list(rewards.values())
            self._reward = sum(reward_vals) / len(reward_vals)
            self._terminated = all(dones.values())
            self._truncated = all(truncs.values())
            self._info = infos.get(self.agent_ids[0], {})

            # Reiniciar ronda
            self._agent_idx = 0
            self._actions = {}

        next_agent = self.agent_ids[self._agent_idx]
        return (
            self._obs_dict[next_agent],
            self._reward,
            self._terminated,
            self._truncated,
            self._info,
        )

    def action_masks(self) -> np.ndarray:
        """Retorna la máscara de acciones válidas para el agente actual.

        Requerido por MaskablePPO de sb3-contrib.
        """
        current_agent = self.agent_ids[self._agent_idx]
        mask = self.multi_env.get_action_mask(current_agent)
        return mask.astype(bool)

    def close(self):
        self.multi_env.close()


class ParameterSharingAgent:
    """MAPPO con MaskablePPO y VecNormalize.

    Usa MaskablePPO de sb3-contrib para enmascarar acciones inválidas
    (fases que no existen en un TLS particular) y VecNormalize para
    normalizar la recompensa en runtime.
    """

    def __init__(self, multi_env: MultiAgentTrafficEnv, config: dict):
        from sb3_contrib import MaskablePPO

        training = config.get("training", {})
        paths = config.get("paths", {})

        self.shared_env = SharedPolicyWrapper(multi_env)

        # LR schedule
        lr = training.get("learning_rate", 3e-4)
        lr_decay = training.get("lr_decay", None)
        if lr_decay == "linear":
            lr_schedule = lambda progress: lr * (0.1 + 0.9 * progress)
        else:
            lr_schedule = lr

        # VecNormalize: normaliza reward, NO obs (ya normalizada por OPW)
        tb_dir = paths.get("tensorboard_dir", "tensorboard")
        vec_env = DummyVecEnv([lambda: self.shared_env])
        self.vec_env = VecNormalize(
            vec_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=training.get("gamma", 0.99),
        )

        self.model = MaskablePPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=lr_schedule,
            n_steps=training.get("n_steps", 256),
            batch_size=training.get("batch_size", 128),
            gamma=training.get("gamma", 0.99),
            ent_coef=training.get("ent_coef", 0.01),
            n_epochs=training.get("n_epochs", 5),
            gae_lambda=training.get("gae_lambda", 0.95),
            max_grad_norm=training.get("max_grad_norm", 0.5),
            vf_coef=training.get("vf_coef", 0.5),
            clip_range=training.get("clip_range", 0.2),
            verbose=1,
            tensorboard_log=tb_dir,
            seed=training.get("seed", 42),
        )

    def train(self, total_timesteps: int, callbacks=None, reset_num_timesteps=True):
        """Entrena MaskablePPO con Parameter Sharing."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
        )

    def predict(
        self, observations: Dict[str, np.ndarray], deterministic=True
    ) -> Dict[str, int]:
        actions = {}
        for aid, obs in observations.items():
            act, _ = self.model.predict(obs, deterministic=deterministic)
            actions[aid] = int(act)
        return actions

    def save(self, path: str):
        self.model.save(path)
        # Guardar también las estadísticas de VecNormalize
        stats_path = path + "_vecnorm.pkl"
        self.vec_env.save(stats_path)

    @classmethod
    def load(cls, path: str, multi_env: MultiAgentTrafficEnv, config: dict):
        from sb3_contrib import MaskablePPO

        agent = cls.__new__(cls)
        agent.shared_env = SharedPolicyWrapper(multi_env)
        vec_env = DummyVecEnv([lambda: agent.shared_env])

        stats_path = path + "_vecnorm.pkl"
        if os.path.exists(stats_path):
            agent.vec_env = VecNormalize.load(stats_path, vec_env)
        else:
            agent.vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

        agent.model = MaskablePPO.load(path, env=agent.vec_env)
        return agent
