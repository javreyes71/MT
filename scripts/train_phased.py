"""Script para entrenar al agente RL en fases consecutivas.

Ejecuta el entrenamiento en N fases de M pasos cada una. 
Al final de cada fase guarda el modelo para que pueda ser ejecutado en SUMO,
y guarda un registro de estadísticas para verificar el aprendizaje.

CORRECCIÓN v3: Compensa el multiplicador 88x del SharedPolicyWrapper
multiplicando los pasos reales deseados por num_agents.
"""
import argparse
import sys
import os

# INYECCIÓN DE LIBSUMO (OVERRIDE DE TRACI) PARA MÁXIMO RENDIMIENTO
# Evita los sockets TCP de TraCI, ejecutando C++ directamente en la RAM.
try:
    import libsumo
    sys.modules['traci'] = libsumo
    print("⚡ [ÓPTIMIZACIÓN] Usando libsumo en lugar de TraCI (100x más rápido)")
except ImportError:
    print("⚠️ [ADVERTENCIA] libsumo no encontrado. Usando TraCI (será muy lento).")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.seed import set_global_seed
from src.environment.traffic_env import TrafficSumoEnv
from src.environment.multi_agent_env import MultiAgentTrafficEnv
from src.agents.centralized import CentralizedAgent
from src.agents.independent import IndependentAgent
from src.agents.parameter_sharing import ParameterSharingAgent
from src.callbacks.metrics_callback import MetricsCallback

def main():
    parser = argparse.ArgumentParser(description="🧠 Entrenamiento por fases (Curriculum)")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--mode", type=str, choices=["centralized", "independent", "parameter_sharing"], default="parameter_sharing")
    parser.add_argument("--total-steps", type=int, default=20_000, help="Pasos REALES de SUMO deseados")
    parser.add_argument("--phases", type=int, default=10, help="Número de fases en las que dividir el entrenamiento")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    config = load_config(args.config)
    config['marl']['mode'] = args.mode
    set_global_seed(args.seed)
    
    # Crear directorios
    model_dir = f"{config['paths']['models_dir']}/{args.mode}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(f"{config['paths']['results_dir']}", exist_ok=True)
    
    callbacks = [MetricsCallback(config['paths']['results_dir'], mode=args.mode)]
    
    # Iniciar Entorno
    if args.mode == "centralized":
        env = TrafficSumoEnv(config=config, gui=False)
        agent = CentralizedAgent(env, config)
        num_agents = 1  # Centralizado: 1 paso wrapper = 1 paso SUMO
    elif args.mode == "independent":
        env = MultiAgentTrafficEnv(config=config, gui=False)
        agent = IndependentAgent(env, config)
        num_agents = 1  # Independent: gestiona internamente
    else:  # parameter_sharing
        env = MultiAgentTrafficEnv(config=config, gui=False)
        agent = ParameterSharingAgent(env, config)
        # CORRECCIÓN v3: SharedPolicyWrapper llama step() una vez POR AGENTE.
        # PPO cuenta cada llamada como 1 timestep, pero solo se ejecuta un paso
        # real de SUMO cada N llamadas (donde N = num_agents = 88).
        # Debemos multiplicar para que PPO ejecute suficientes pasos reales.
        num_agents = len(env.agent_ids)
    
    # Calcular pasos SB3 necesarios para lograr los pasos SUMO reales deseados
    real_steps_per_phase = args.total_steps // args.phases
    sb3_steps_per_phase = real_steps_per_phase * num_agents
    total_sb3_steps = sb3_steps_per_phase * args.phases

    print(f"\n🚀 Iniciando entrenamiento por fases:")
    print(f"   Modo: {args.mode}")
    print(f"   Pasos SUMO reales deseados: {args.total_steps:,}")
    print(f"   Agentes (multiplicador wrapper): {num_agents}")
    print(f"   Pasos SB3 por fase: {sb3_steps_per_phase:,} ({real_steps_per_phase:,} reales)")
    print(f"   Fases: {args.phases}")
    print(f"   Total pasos SB3: {total_sb3_steps:,}")
    
    print("\n=======================================================")
    for phase in range(1, args.phases + 1):
        print(f"🔄 FASE {phase}/{args.phases} | {real_steps_per_phase:,} pasos SUMO reales ({sb3_steps_per_phase:,} SB3)...")
        
        # reset_num_timesteps=False en fases 2+ para que el LR decay
        # y el contador de PPO continúen desde donde quedaron
        is_first_phase = (phase == 1)
        agent.train(
            sb3_steps_per_phase, 
            callbacks=callbacks,
            reset_num_timesteps=is_first_phase
        )
        
        # Guardar modelo de la fase
        phase_model_path = f"{model_dir}/model_phase_{phase}"
        agent.save(phase_model_path)
        print(f"✅ Fase {phase} completada. Modelo guardado en: {phase_model_path}.zip")
        print("=======================================================\n")
        
    env.close()
    print("🎉 Entrenamiento en fases concluido exitosamente.")

if __name__ == "__main__":
    main()
