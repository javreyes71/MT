import argparse
import sys
import os
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.environment.multi_agent_env import MultiAgentTrafficEnv
from src.agents.parameter_sharing import ParameterSharingAgent, ParameterSharingVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    parser = argparse.ArgumentParser(description="Entrenamiento Curriculum Learning")
    parser.add_argument("--config", type=str, default="config/caso_estudio.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Ruta del modelo a cargar")
    parser.add_argument("--start_phase", type=int, default=1, help="Fase a iniciar (1, 2, 3)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Curriculum phases: (timesteps, route_file, phase_name)
    curriculum = [
        (3_000_000, "sumo/caso_estudio_bajo_42.rou.xml", "bajo"),
        (3_000_000, "sumo/caso_estudio_medio_42.rou.xml", "medio"),
        (3_000_000, "sumo/caso_estudio_alto_42.rou.xml", "alto"),
    ]
    
    models_dir = "models_curriculum"
    tb_dir = "tensorboard_curriculum"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    
    agent = None
    
    for i, (steps, route, phase_name) in enumerate(curriculum):
        phase_num = i + 1
        if phase_num < args.start_phase:
            print(f"Saltando fase {phase_num}: {phase_name}...")
            continue
        print(f"\n{'='*50}")
        print(f" INICIANDO FASE {i+1}: {phase_name.upper()}")
        print(f" Rutas: {route}")
        print(f" Pasos: {steps}")
        print(f"{'='*50}\n")
        
        phase_config = deepcopy(config)
        phase_config['simulation']['route_files'] = route
        
        # Sobreescribir las carpetas de tensorboard para que no se mezcle
        if 'paths' not in phase_config:
            phase_config['paths'] = {}
        phase_config['paths']['tensorboard_large_dir'] = tb_dir
        phase_config['paths']['tensorboard_dir'] = tb_dir
        
        env = MultiAgentTrafficEnv(config=phase_config, gui=False)
        
        if agent is None:
            if args.resume and phase_num == args.start_phase:
                print(f"Reanudando desde {args.resume}...")
                agent = ParameterSharingAgent.load(args.resume, env, phase_config)
                agent.model.tensorboard_log = tb_dir
            else:
                agent = ParameterSharingAgent(env, phase_config)
                agent.model.tensorboard_log = tb_dir
        else:
            from stable_baselines3.common.vec_env import VecMonitor
            new_vec_env = ParameterSharingVecEnv(env)
            monitored_env = VecMonitor(new_vec_env)
            agent.model.set_env(monitored_env)
            agent.vec_env = monitored_env
            
        checkpoint_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=models_dir,
            name_prefix=f"acurriculum_phase_{i+1}_{phase_name}",
        )
        
        is_first_loaded = (phase_num == args.start_phase and args.resume is None)
        agent.model.learn(total_timesteps=steps, callback=checkpoint_cb, reset_num_timesteps=is_first_loaded)
        agent.save(os.path.join(models_dir, f"acurriculum_final_phase_{i+1}_{phase_name}"))
        
        env.close()

if __name__ == "__main__":
    main()
