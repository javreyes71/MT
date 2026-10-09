import os
import sys
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.environment.traffic_env import TrafficSumoEnv
from src.environment.multi_agent_env import MultiAgentTrafficEnv
from src.agents.parameter_sharing import ParameterSharingAgent
from src.utils.config import load_config

def evaluate_model(model_path, config_path, route_file):
    config = load_config(config_path)
    config["simulation"]["route_files"] = route_file
    config["simulation"]["gui"] = False
    
    env = MultiAgentTrafficEnv(config=config, gui=False)
    agent = ParameterSharingAgent(env, config)
    
    from sb3_contrib import MaskablePPO
    agent.model = MaskablePPO.load(model_path, env=agent.vec_env)
    
    obs_dict = env.reset()
    total_rewards = 0
    steps = 0
    
    print(f"\nEvaluando {model_path} en {route_file}...")
    t0 = time.time()
    
    done = False
    while not done:
        actions = agent.predict(obs_dict, deterministic=True)
        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        total_rewards += sum(rewards.values())
        steps += 1
        
        if any(terminated.values()) or any(truncated.values()):
            done = True
            
    t1 = time.time()
    print(f"Hecho en {t1-t0:.2f}s. Pasos: {steps}. Recompensa total: {total_rewards:.2f}")
    
    return total_rewards

if __name__ == "__main__":
    print("=== Evaluando Fase 2 en ALTO ===")
    model = "models_curriculum/acurriculum_final_phase_2_medio.zip"
    evaluate_model(model, "config/caso_estudio.yaml", "sumo/caso_estudio_alto_42.rou.xml")

