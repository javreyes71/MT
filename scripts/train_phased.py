"""Script para entrenar al agente RL en fases consecutivas.

Ejecuta el entrenamiento en N fases de M pasos cada una. 
Al final de cada fase guarda el modelo para que pueda ser ejecutado en SUMO,
y guarda un registro de estadísticas para verificar el aprendizaje.

Uso:
    python scripts/train_phased.py --total-steps 500000 --phases 10
    python scripts/train_phased.py --total-steps 1500000 --phases 15  # Entrenamiento largo
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
    parser = argparse.ArgumentParser(description="🧠 Entrenamiento por fases (Curriculum)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--total-steps", type=int, default=500_000, help="Pasos REALES de SUMO deseados")
    parser.add_argument("--phases", type=int, default=10, help="Número de fases en las que dividir el entrenamiento")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Path a modelo para continuar entrenamiento")
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_global_seed(args.seed)
    
    # Crear directorios
    model_dir = os.path.join(config['paths']['models_dir'], "parameter_sharing")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    
    callbacks = [MetricsCallback(config['paths']['results_dir'], mode="parameter_sharing")]
    
    # Crear entorno multi-agente
    env = MultiAgentTrafficEnv(config=config, gui=False)
    num_agents = len(env.agent_ids)
    
    # Crear o cargar agente
    if args.resume:
        print(f"📂 Cargando modelo desde {args.resume}...")
        agent = ParameterSharingAgent.load(args.resume, env, config)
    else:
        agent = ParameterSharingAgent(env, config)
    
    # ParameterSharingVecEnv: un step() de SUMO genera num_agents transiciones
    # para PPO. Multiplicamos para que PPO ejecute suficientes pasos reales.
    real_steps_per_phase = args.total_steps // args.phases
    sb3_steps_per_phase = real_steps_per_phase * num_agents
    total_sb3_steps = sb3_steps_per_phase * args.phases

    print(f"\n🚀 Iniciando entrenamiento por fases:")
    print(f"   Algoritmo: MaskablePPO + Parameter Sharing")
    print(f"   Pasos SUMO reales deseados: {args.total_steps:,}")
    print(f"   Agentes (multiplicador VecEnv): {num_agents}")
    print(f"   Pasos SB3 por fase: {sb3_steps_per_phase:,} ({real_steps_per_phase:,} reales)")
    print(f"   Fases: {args.phases}")
    print(f"   Total pasos SB3: {total_sb3_steps:,}")
    
    print("\n=======================================================")
    for phase in range(1, args.phases + 1):
        print(f"🔄 FASE {phase}/{args.phases} | {real_steps_per_phase:,} pasos SUMO reales ({sb3_steps_per_phase:,} SB3)...")
        
        # reset_num_timesteps=False en fases 2+ para que el LR decay
        # y el contador de PPO continúen desde donde quedaron
        is_first_phase = (phase == 1) and (args.resume is None)
        agent.train(
            sb3_steps_per_phase, 
            callbacks=callbacks,
            reset_num_timesteps=is_first_phase
        )
        
        # Guardar modelo de la fase
        phase_model_path = os.path.join(model_dir, f"model_phase_{phase}")
        agent.save(phase_model_path)
        print(f"✅ Fase {phase} completada. Modelo guardado en: {phase_model_path}.zip")
        print("=======================================================\n")
        
    env.close()
    print("🎉 Entrenamiento en fases concluido exitosamente.")

if __name__ == "__main__":
    main()
