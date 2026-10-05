"""Script de Fine-Tuning: carga un modelo preentrenado y lo perfecciona.

Usa un learning rate reducido y exploración mínima para refinar
la política sin arriesgar colapsos (gridlock).
"""
import argparse
import sys
import os

# INYECCIÓN DE LIBSUMO PARA MÁXIMO RENDIMIENTO
try:
    import libsumo
    sys.modules['traci'] = libsumo
    print("⚡ [OPTIMIZACIÓN] Usando libsumo (100x más rápido)")
except ImportError:
    print("⚠️ libsumo no encontrado. Usando TraCI estándar.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.environment.multi_agent_env import MultiAgentTrafficEnv
from src.agents.parameter_sharing import SharedPolicyWrapper
from src.callbacks.metrics_callback import MetricsCallback


def main():
    parser = argparse.ArgumentParser(description="🔧 Fine-Tuning de modelo preentrenado")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Ruta al modelo .zip a cargar (ej: models/parameter_sharing/model_phase_4)")
    parser.add_argument("--total-steps", type=int, default=500_000,
                        help="Pasos totales del fine-tuning")
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
    
    steps_per_phase = args.total_steps // args.phases
    
    print(f"🔧 Fine-Tuning de modelo preentrenado:")
    print(f"   Modelo base: {args.model_path}")
    print(f"   Pasos totales: {args.total_steps}")
    print(f"   Fases: {args.phases} ({steps_per_phase} pasos por fase)")
    print(f"   Learning rate: {args.lr}")
    print(f"   Entropía: {args.ent_coef}")
    
    # Crear entorno
    env = MultiAgentTrafficEnv(config=config, gui=False)
    wrapped_env = SharedPolicyWrapper(env)
    
    # Cargar modelo preentrenado con nuevos hiperparámetros
    lr_final = args.lr / 10  # Decay hasta lr/10
    lr_schedule = lambda progress: lr_final + (args.lr - lr_final) * progress
    
    model = PPO.load(
        args.model_path,
        env=wrapped_env,
        learning_rate=lr_schedule,
        ent_coef=args.ent_coef,
        n_epochs=3,  # Menos épocas = menos riesgo de overfitting
        batch_size=256,
        clip_range=0.1,  # Clip más estrecho = actualizaciones más conservadoras
    )
    print(f"✅ Modelo cargado exitosamente")
    
    # Directorios
    model_dir = f"{config['paths']['models_dir']}/finetune"
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [MetricsCallback(config['paths']['results_dir'], mode="finetune")]
    
    print("\n=======================================================")
    for phase in range(1, args.phases + 1):
        print(f"🔄 FINE-TUNE FASE {phase}/{args.phases} | {steps_per_phase} timesteps...")
        
        model.learn(total_timesteps=steps_per_phase, callback=callbacks, reset_num_timesteps=False)
        
        phase_path = f"{model_dir}/model_finetune_phase_{phase}"
        model.save(phase_path)
        print(f"✅ Fase {phase} completada. Modelo guardado en: {phase_path}.zip")
        print("=======================================================\n")
    
    env.close()
    print("🎉 Fine-Tuning concluido exitosamente.")


if __name__ == "__main__":
    main()
