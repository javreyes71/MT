"""Script para visualizar un modelo entrenado en SUMO (con GUI)."""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.environment.traffic_env import TrafficSumoEnv
from src.environment.multi_agent_env import MultiAgentTrafficEnv
from src.agents.centralized import CentralizedAgent
from src.agents.independent import IndependentAgent
from src.agents.parameter_sharing import ParameterSharingAgent

def visualize(agent, env, config, mode):
    print("👁️ Visualizando comportamiento del agente en SUMO GUI...")
    if mode == "centralized":
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < config['simulation']['duration']:
            action, _ = agent.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
    else:
        obs_dict = env.reset()
        done = False
        step = 0
        while not done and step < config['simulation']['duration']:
            actions = agent.predict(obs_dict)
            obs_dict, rewards, terminateds, truncateds, infos = env.step(actions)
            done = all(terminateds.values()) or all(truncateds.values())
            step += 1
    print("✅ Visualización completada!")

def main():
    parser = argparse.ArgumentParser(description="👁️ Visualizar agente de control de tráfico")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--model", type=str, required=True, help="Ruta al modelo entrenado")
    parser.add_argument("--mode", type=str, choices=["centralized", "independent", "parameter_sharing"], required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.mode == "centralized":
        env = TrafficSumoEnv(config=config, gui=True)
        agent = CentralizedAgent.load(args.model, env)
        visualize(agent, env, config, args.mode)
        env.close()
    elif args.mode == "independent":
        multi_env = MultiAgentTrafficEnv(config=config, gui=True)
        agent = IndependentAgent.load(args.model, multi_env, config)
        visualize(agent, multi_env, config, args.mode)
        multi_env.close()
    elif args.mode == "parameter_sharing":
        multi_env = MultiAgentTrafficEnv(config=config, gui=True)
        agent = ParameterSharingAgent.load(args.model, multi_env, config)
        visualize(agent, multi_env, config, args.mode)
        multi_env.close()

if __name__ == "__main__":
    main()
