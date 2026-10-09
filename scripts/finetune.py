"""Script de Fine-Tuning: carga un modelo preentrenado y lo perfecciona.

Usa un learning rate reducido y exploración mínima para refinar
la política sin arriesgar colapsos (gridlock).

Uso:
    python scripts/finetune.py --model-path models/parameter_sharing/model_phase_10 --total-steps 250000 --phases 5
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.environment.multi_agent_env import MultiAgentTrafficEnv
from src.agents.parameter_sharing import ParameterSharingAgent
from src.callbacks.metrics_callback import MetricsCallback


def main():
    parser = argparse.ArgumentParser(description="🔧 Fine-Tuning de modelo preentrenado")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Ruta al modelo .zip a cargar (ej: models/parameter_sharing/model_phase_10)")
    parser.add_argument("--total-steps", type=int, default=250_000,
                        help="Pasos SUMO reales totales del fine-tuning")
    parser.add_argument("--phases", type=int, default=5,
                        help="Número de fases de fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate inicial para fine-tuning (más bajo que entrenamiento)")
    parser.add_argument("--ent-coef", type=float, default=0.005,
                        help="Coeficiente de entropía (exploración mínima para fine-tuning)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_global_seed(args.seed)
    
    # Crear entorno
    env = MultiAgentTrafficEnv(config=config, gui=False)
    num_agents = len(env.agent_ids)
    
    # Cargar modelo preentrenado con ParameterSharingAgent.load (MaskablePPO)
    print(f"📂 Cargando modelo desde {args.model_path}...")
    agent = ParameterSharingAgent.load(args.model_path, env, config)
    print(f"✅ Modelo cargado exitosamente")
    
    # Ajustar hiperparámetros para fine-tuning
    lr_final = args.lr / 10
    lr_schedule = lambda progress: lr_final + (args.lr - lr_final) * progress
    
    agent.model.learning_rate = lr_schedule
    agent.model.ent_coef = args.ent_coef
    agent.model.clip_range = lambda _: 0.1  # Clip más estrecho = actualizaciones conservadoras
    agent.model.n_epochs = 3  # Menos épocas = menos riesgo de overfitting
    
    # Calcular pasos SB3 (multiplicar por num_agents como en train_phased.py)
    real_steps_per_phase = args.total_steps // args.phases
    sb3_steps_per_phase = real_steps_per_phase * num_agents
    
    # Directorios
    model_dir = os.path.join(config['paths']['models_dir'], "finetune")
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [MetricsCallback(config['paths']['results_dir'], mode="finetune")]
    
    print(f"\n🔧 Fine-Tuning de modelo preentrenado:")
    print(f"   Modelo base: {args.model_path}")
    print(f"   Agentes: {num_agents}")
    print(f"   Pasos SUMO reales: {args.total_steps:,}")
    print(f"   Fases: {args.phases} ({real_steps_per_phase:,} reales por fase)")
    print(f"   Learning rate: {args.lr} → {lr_final}")
    print(f"   Entropía: {args.ent_coef}")
    
    print("\n=======================================================")
    for phase in range(1, args.phases + 1):
        print(f"🔄 FINE-TUNE FASE {phase}/{args.phases} | {real_steps_per_phase:,} pasos reales ({sb3_steps_per_phase:,} SB3)...")
        
        agent.train(
            sb3_steps_per_phase,
            callbacks=callbacks,
            reset_num_timesteps=False
        )
        
        phase_path = os.path.join(model_dir, f"model_finetune_phase_{phase}")
        agent.save(phase_path)
        print(f"✅ Fase {phase} completada. Modelo guardado en: {phase_path}.zip")
        print("=======================================================\n")
    
    env.close()
    print("🎉 Fine-Tuning concluido exitosamente.")


if __name__ == "__main__":
    main()
