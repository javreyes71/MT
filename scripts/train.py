"""Script de entrenamiento limpio para MaskablePPO con Parameter Sharing.

Uso:
    python scripts/train.py
    python scripts/train.py --config config/default.yaml --timesteps 500000
    python scripts/train.py --gui   # con visualizaciÃ³n SUMO
"""

import argparse
import os
import sys
import time

# AÃ±adir raÃ­z del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.environment.multi_agent_env import MultiAgentTrafficEnv
from src.agents.parameter_sharing import ParameterSharingAgent

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    CallbackList,
    BaseCallback
)

class SaveVecNormalizeCallback(BaseCallback):
    """Guarda el estado de VecNormalize en cada checkpoint."""
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, agent, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.agent = agent

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            stats_path = os.path.join(
                self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_vecnorm.pkl"
            )
            self.agent.vec_env.save(stats_path)
            if self.verbose > 0:
                print(f"Guardando VecNormalize en {stats_path}")
        return True

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento MaskablePPO")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path a modelo para continuar")
    args = parser.parse_args()

    config = load_config(args.config)
    training = config.get("training", {})
    paths = config.get("paths", {})

    total_timesteps = args.timesteps or training.get("total_timesteps", 1_000_000)
    models_dir = paths.get("models_dir", "models")
    tb_dir = paths.get("tensorboard_dir", "tensorboard")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(paths.get("results_dir", "results"), exist_ok=True)

    # â”€â”€ Entorno â”€â”€
    print(f"Creando entorno multi-agente...")
    multi_env = MultiAgentTrafficEnv(config=config, gui=args.gui)
    print(f"  Agentes: {multi_env.num_agents}")
    print(f"  Obs dim: {multi_env.observation_space.shape}")
    print(f"  Act dim: {multi_env.action_space.n}")
    print(f"  Max phases: {multi_env.max_phases}")

    # â”€â”€ Agente â”€â”€
    if args.resume:
        print(f"Cargando modelo desde {args.resume}...")
        agent = ParameterSharingAgent.load(args.resume, multi_env, config)
    else:
        agent = ParameterSharingAgent(multi_env, config)

    # â”€â”€ Callbacks â”€â”€
    checkpoint_freq = training.get("checkpoint_freq", 10_000)
    checkpoint_cb = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=models_dir,
        name_prefix="maskable_ppo",
    )
    vecnorm_cb = SaveVecNormalizeCallback(
        save_freq=checkpoint_freq,
        save_path=models_dir,
        name_prefix="maskable_ppo",
        agent=agent,
        verbose=1
    )

    callbacks = CallbackList([checkpoint_cb, vecnorm_cb])

    # â”€â”€ Entrenamiento â”€â”€
    print(f"\n{'='*60}")
    print(f"  MaskablePPO + Parameter Sharing + VecNormalize")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Agentes:   {multi_env.num_agents}")
    print(f"  TensorBoard: {tb_dir}")
    print(f"{'='*60}\n")

    t0 = time.time()
    agent.train(
        total_timesteps=total_timesteps,
        callbacks=callbacks,
        reset_num_timesteps=(args.resume is None),
    )
    elapsed = time.time() - t0

    # â”€â”€ Guardar â”€â”€
    final_path = os.path.join(models_dir, "maskable_ppo_final")
    agent.save(final_path)
    print(f"\nModelo guardado en {final_path}")
    print(f"Tiempo total: {elapsed/60:.1f} minutos")

    multi_env.close()


if __name__ == "__main__":
    main()
