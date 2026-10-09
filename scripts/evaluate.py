"""Evaluador unificado de políticas (baseline y RL).

Uso:
    python scripts/evaluate.py --policy fixed_time
    python scripts/evaluate.py --policy max_pressure
    python scripts/evaluate.py --policy random
    python scripts/evaluate.py --policy rl --model_path models/parameter_sharing/model_phase_10.zip
    python scripts/evaluate.py --all   # corre todas + comparación

Ejecuta N semillas × 3 niveles de demanda y recoge métricas de tripinfo y teleports.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import libsumo as traci

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.environment.traffic_env import TrafficSumoEnv
from src.baselines import FixedTimePolicy, MaxPressurePolicy, AdaptiveActuatedPolicy, AdaptiveDelayBasedPolicy, RandomPolicy


class RLPolicy:
    """Envoltorio para cargar y ejecutar un modelo RL (MaskablePPO) entrenado con Parameter Sharing."""
    def __init__(self, model_path: str, config_path: str = "config/default.yaml"):
        self.config_path = config_path
        if not os.path.exists(model_path):
            sys.exit(f"Error: Modelo no encontrado en {model_path}")
        
        self.model_path = model_path
        self._name = "RL_MaskablePPO"
        self.agent = None
        # Para engañar al chequeo de hasattr(policy, 'model')
        self.model = True

    @property
    def name(self) -> str:
        return self._name

    def get_actions(self, agent_ids, signals, obs, env):
        if self.agent is None:
            from src.agents.parameter_sharing import ParameterSharingAgent
            from src.environment.multi_agent_env import MultiAgentTrafficEnv
            from src.utils.config import load_config
            
            config = load_config(self.config_path)
            self.multi_env = MultiAgentTrafficEnv(config=config, gui=False)
            self.multi_env.base_env = env
            self.agent = ParameterSharingAgent.load(self.model_path, self.multi_env, config)
            
        self.multi_env.base_env = env
        
        actions_dict = {}
        # Recopilar todas las observaciones locales en un solo batch
        local_obs_list = []
        for tid in agent_ids:
            local_obs = self.multi_env._get_local_obs(tid)
            local_obs_list.append(local_obs)
            
        batch_obs = np.array(local_obs_list, dtype=np.float32)
        
        # Normalizar en bloque si es necesario
        if hasattr(self.agent, 'vec_env') and self.agent.vec_env is not None:
            batch_obs = self.agent.vec_env.normalize_obs(batch_obs)
        
        action_masks = np.array([self.multi_env.get_action_mask(tid) for tid in agent_ids], dtype=bool)
        actions, _ = self.agent.model.predict(batch_obs, deterministic=True, action_masks=action_masks)
        
        # Mapear de vuelta al diccionario
        for i, tid in enumerate(agent_ids):
            actions_dict[tid] = int(actions[i])
            
        return actions_dict


def parse_tripinfo(tripinfo_path: str) -> dict:
    """Extrae métricas agregadas de un archivo tripinfo.xml de SUMO."""
    if not os.path.exists(tripinfo_path):
        return {}

    tree = ET.parse(tripinfo_path)
    root = tree.getroot()

    durations = []
    wait_times = []
    time_losses = []
    speeds = []
    halts = []  # Añadido para métricas de seguridad Stop-and-Go

    for trip in root.findall("tripinfo"):
        durations.append(float(trip.get("duration", 0)))
        wait_times.append(float(trip.get("waitingTime", 0)))
        time_losses.append(float(trip.get("timeLoss", 0)))
        halts.append(float(trip.get("waitingCount", 0)))  # Número de paradas (Stop-and-Go)

        dur = float(trip.get("duration", 1))
        route_len = float(trip.get("routeLength", 0))
        if dur > 0:
            speeds.append(route_len / dur)

    n = len(durations)
    if n == 0:
        return {"n_trips": 0}

    return {
        "n_trips": n,
        "avg_duration": np.mean(durations),
        "avg_wait": np.mean(wait_times),
        "avg_time_loss": np.mean(time_losses),
        "avg_speed": np.mean(speeds) if speeds else 0.0,
        "avg_halts": np.mean(halts),  # Métrica añadida
        "p95_wait": np.percentile(wait_times, 95),
        "max_wait": np.max(wait_times),
    }


def run_episode(env: TrafficSumoEnv, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    
    total_teleports = 0
    total_inference_time = 0.0

    is_native = getattr(policy, 'type_id', None) is not None
    if is_native:
        for tid in env.tls_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tid)
                if logics:
                    logic = logics[0]
                    logic.type = policy.type_id
                    traci.trafficlight.setProgramLogic(tid, logic)
            except Exception:
                pass

    while True:
        t0 = time.perf_counter()
        if hasattr(policy, 'model'):
            actions_dict = policy.get_actions(env.tls_ids, env.signals, obs, env)
        else:
            actions_dict = policy.get_actions(env.tls_ids, env.signals)
        
        inference_time = time.perf_counter() - t0
        total_inference_time += inference_time

        if actions_dict is None:
            action_list = None
        else:
            action_list = [actions_dict.get(tid, 0) for tid in env.tls_ids]
            
        obs, reward, terminated, truncated, info = env.step(action_list)
        
        total_teleports += traci.simulation.getStartingTeleportNumber()
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return {
        'total_reward': total_reward, 
        'steps': steps, 
        'seed': seed,
        'teleports': total_teleports,
        'avg_inference_ms': (total_inference_time / steps) * 1000 if steps > 0 else 0.0
    }


def evaluate_policy(policy, config: dict, levels: list, seeds: list, route_prefix: str) -> pd.DataFrame:
    """Evalúa una política en múltiples niveles × semillas."""
    results = []

    for level in levels:
        for seed in seeds:
            route_file = f"sumo/{route_prefix}_{level}_{seed}.rou.xml"
            if not os.path.exists(route_file):
                print(f"  Saltando {route_file} (no existe)")
                continue

            eval_config = config.copy()
            eval_config["simulation"] = config["simulation"].copy()
            
            # Inyectar para que TrafficSumoEnv use estos valores
            eval_config["simulation"]["route_files"] = route_file
            tripinfo_path = os.path.join("results", f"tripinfo_eval_{policy.name}_{level}_{seed}.xml")
            eval_config["simulation"]["tripinfo_output"] = tripinfo_path

            env = TrafficSumoEnv(config=eval_config, gui=False)

            print(f"  {policy.name} | nivel={level} seed={seed}...", end=" ", flush=True)
            t0 = time.time()

            try:
                base_metrics = run_episode(env, policy, seed)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            finally:
                # IMPORTANTE: Cerrar el entorno hace que SUMO termine de escribir el XML
                env.close()
                time.sleep(1.0) # Asegurar flush de disco

            # Parsear tripinfo AHORA que SUMO lo cerró correctamente
            metrics = parse_tripinfo(tripinfo_path)
            
            # Combinar métricas
            metrics.update(base_metrics)
            metrics["level"] = level
            metrics["policy"] = policy.name
            elapsed = time.time() - t0
            metrics["wall_time"] = elapsed
            results.append(metrics)

            avg_w = metrics.get("avg_wait", -1)
            halts = metrics.get("avg_halts", -1)
            teleports = metrics.get("teleports", 0)
            ms = metrics.get("avg_inference_ms", 0)
            n = metrics.get("n_trips", 0)
            print(f"trips={n} avg_wait={avg_w:.1f}s halts={halts:.1f} teleports={teleports} inf={ms:.2f}ms")

    return pd.DataFrame(results)


def print_comparison(df: pd.DataFrame):
    """Imprime tabla comparativa de políticas."""
    if df.empty:
        print("No hay resultados para comparar.")
        return

    print(f"\n{'='*90}")
    print(f"  RESULTADOS DE EVALUACIÓN (Tesis)")
    print(f"{'='*90}\n")

    summary = df.groupby(["policy", "level"]).agg(
        n_trips=("n_trips", "mean"),
        avg_wait=("avg_wait", "mean"),
        avg_halts=("avg_halts", "mean"),
        teleports=("teleports", "sum"),
        avg_inf_ms=("avg_inference_ms", "mean"),
        total_reward=("total_reward", "mean"),
    ).round(2)

    print(summary.to_string())
    print()

    global_summary = df.groupby("policy").agg(
        avg_wait=("avg_wait", "mean"),
        avg_halts=("avg_halts", "mean"),
        teleports=("teleports", "sum"),
        avg_inf_ms=("avg_inference_ms", "mean"),
    ).round(2)

    print("Resumen Global por Política:")
    print(global_summary.to_string())


def main():
    parser = argparse.ArgumentParser(description="Evaluación de políticas")
    parser.add_argument("--policy", type=str,
                        choices=["fixed_time", "max_pressure", "scats_proxy", "scoot_proxy", "rl", "all"],
                        default="all")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--route_prefix", type=str, default="routes")
    # Semillas por defecto fijadas para la robustez (N=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--levels", type=str, nargs="+", default=["bajo", "medio", "alto"])
    parser.add_argument("--output", type=str, default="results/evaluation.csv")
    parser.add_argument("--model_path", type=str, default="models/parameter_sharing/model_phase_10.zip",
                        help="Ruta al modelo RL si policy es 'rl' o 'all'")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs("results", exist_ok=True)

    policies = {
        "fixed_time": FixedTimePolicy(),
        "max_pressure": MaxPressurePolicy(),
        "scats_proxy": AdaptiveActuatedPolicy(),
        "scoot_proxy": AdaptiveDelayBasedPolicy(),
    }
    
    if args.policy in ["rl", "all"]:
        policies["rl"] = RLPolicy(args.model_path, args.config)

    if args.policy == "all":
        selected = list(policies.values())
    else:
        selected = [policies[args.policy]]

    all_results = []

    for policy in selected:
        print(f"\n{'─'*50}")
        print(f"  Evaluando: {policy.name}")
        print(f"{'─'*50}")
        df = evaluate_policy(policy, config, args.levels, args.seeds, args.route_prefix)
        all_results.append(df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nResultados guardados en {args.output}")
        print_comparison(combined)
        
        # Automatización de generación de imágenes PNG para el informe
        try:
            import plot_metrics
            out_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else "."
            plot_dir = os.path.join(out_dir, "plots")
            plot_metrics.generate_all_plots(args.output, plot_dir)
        except Exception as e:
            print(f"Error generando las tablas PNG: {e}")

if __name__ == "__main__":
    main()
