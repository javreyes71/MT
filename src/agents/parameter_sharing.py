"""ParameterSharingAgent con MaskablePPO y VecEnv Sincrónico.

Fase 1 + Fase 2:
Usa un VecEnv donde cada "ambiente" en el batch es un agente.
Esto permite a PPO asignar créditos individuales y preservar las 
transiciones de Markov (obs_{i,t} -> obs_{i,t+1}).
"""

import gymnasium as gym
import numpy as np
import os
from typing import Dict, Any, Tuple, List

from stable_baselines3.common.vec_env import VecEnv, VecNormalize, VecMonitor
from src.environment.multi_agent_env import MultiAgentTrafficEnv


class ParameterSharingVecEnv(VecEnv):
    """Wrapper que convierte un MultiAgentEnv en un VecEnv para SB3.
    
    Cada agente opera como un sub-entorno independiente en el batch.
    """
    def __init__(self, multi_env: MultiAgentTrafficEnv):
        self.multi_env = multi_env
        self.agent_ids = multi_env.agent_ids
        num_envs = len(self.agent_ids)
        super().__init__(num_envs, multi_env.observation_space, multi_env.action_space)
        self._actions = {}
        self._dones = np.zeros(self.num_envs, dtype=bool)

    def reset(self):
        obs_dict = self.multi_env.reset()
        self._dones = np.zeros(self.num_envs, dtype=bool)
        return np.array([obs_dict[aid] for aid in self.agent_ids], dtype=np.float32)

    def step_async(self, actions: np.ndarray):
        self._actions = {aid: act for aid, act in zip(self.agent_ids, actions)}

    def step_wait(self):
        obs_d, rew_d, term_d, trunc_d, info_d = self.multi_env.step(self._actions)
        
        obs = np.array([obs_d[aid] for aid in self.agent_ids], dtype=np.float32)
        rewards = np.array([rew_d[aid] for aid in self.agent_ids], dtype=np.float32)
        dones = np.array([term_d[aid] or trunc_d[aid] for aid in self.agent_ids], dtype=bool)
        
        infos = []
        for i, aid in enumerate(self.agent_ids):
            info = info_d[aid].copy()
            info["agent_id"] = aid
            if dones[i]:
                info["terminal_observation"] = obs[i]
            infos.append(info)

        self._dones = dones
        
        # Auto-reset for VecEnv if terminated
        if dones.all():
            obs_d = self.multi_env.reset()
            obs = np.array([obs_d[aid] for aid in self.agent_ids], dtype=np.float32)

        return obs, rewards, dones, infos

    def close(self):
        self.multi_env.close()

    def get_attr(self, attr_name, indices=None):
        if attr_name == "action_masks":
            # Sb3_contrib comprueba si los sub-entornos tienen la función
            return [True for _ in range(self.num_envs)]
        return [getattr(self.multi_env, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            # sb3_contrib espera una lista de máscaras (una por cada sub-entorno)
            return [self.multi_env.get_action_mask(aid) for aid in self.agent_ids]
        return [None for _ in range(self.num_envs)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def action_masks(self) -> np.ndarray:
        """Fallback en caso de que MaskablePPO lo llame directamente en el VecEnv."""
        masks = [self.multi_env.get_action_mask(aid) for aid in self.agent_ids]
        return np.array(masks, dtype=bool)


class ParameterSharingAgent:
    """MAPPO con MaskablePPO, VecEnv propio y VecNormalize."""

    def __init__(self, multi_env: MultiAgentTrafficEnv, config: dict):
        from sb3_contrib import MaskablePPO

        training = config.get("training", {})
        paths = config.get("paths", {})

        # Env Wrapping
        vec_env = ParameterSharingVecEnv(multi_env)
        monitored_env = VecMonitor(vec_env)

        self.vec_env = VecNormalize(
            monitored_env,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=training.get("gamma", 0.99),
        )

        # LR schedule
        lr = training.get("learning_rate", 3e-4)
        lr_decay = training.get("lr_decay", None)
        if lr_decay == "linear":
            lr_schedule = lambda progress: lr * (0.1 + 0.9 * progress)
        else:
            lr_schedule = lr

        tb_dir = paths.get("tensorboard_dir", "tensorboard")

        self.model = MaskablePPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=lr_schedule,
            n_steps=training.get("n_steps", 1024),
            batch_size=training.get("batch_size", 1024),
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

    def train(self, total_timesteps: int, callbacks=None, reset_num_timesteps=True, progress_bar=True):
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def predict(self, observations: Dict[str, np.ndarray], deterministic=True) -> Dict[str, int]:
        actions = {}
        # Para evaluación directa iteramos sobre el diccionario. 
        # (evaluate.py ya maneja la vectorización de forma independiente)
        for aid, obs in observations.items():
            act, _ = self.model.predict(obs, deterministic=deterministic)
            actions[aid] = int(act)
        return actions

    def save(self, path: str):
        self.model.save(path)
        stats_path = path + "_vecnorm.pkl"
        self.vec_env.save(stats_path)

    @classmethod
    def load(cls, path: str, multi_env: MultiAgentTrafficEnv, config: dict):
        from sb3_contrib import MaskablePPO
        
        agent = cls.__new__(cls)
        vec_env = ParameterSharingVecEnv(multi_env)
        monitored_env = VecMonitor(vec_env)

        stats_path = path + "_vecnorm.pkl"
        if os.path.exists(stats_path):
            agent.vec_env = VecNormalize.load(stats_path, monitored_env)
        else:
            agent.vec_env = VecNormalize(monitored_env, norm_obs=False, norm_reward=True)

        agent.model = MaskablePPO.load(path, env=agent.vec_env)
        return agent
