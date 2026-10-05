"""Evaluador unificado de políticas (baseline y RL).

Uso:
    python scripts/evaluate.py --policy fixed_time
    python scripts/evaluate.py --policy max_pressure
    python scripts/evaluate.py --policy random
    python scripts/evaluate.py --all   # corre todas + comparación

Ejecuta N semillas × 3 niveles de demanda y recoge métricas de tripinfo.
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.environment.traffic_env import TrafficSumoEnv
from src.baselines import FixedTimePolicy, MaxPressurePolicy, RandomPolicy


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

    for trip in root.findall("tripinfo"):
        durations.append(float(trip.get("duration", 0)))
        wait_times.append(float(trip.get("waitingTime", 0)))
        time_losses.append(float(trip.get("timeLoss", 0)))

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
        "p95_wait": np.percentile(wait_times, 95),
        "max_wait": np.max(wait_times),
    }


def run_episode(env: TrafficSumoEnv, policy, seed: int) -> dict:
    """Corre un episodio completo con una política y retorna métricas brutas."""
    env.reset()
    total_reward = 0.0
    steps = 0

    while True:
        actions_dict = policy.get_actions(env.tls_ids, env.signals)
        action_list = [actions_dict.get(tid, 0) for tid in env.tls_ids]

        obs, reward, terminated, truncated, info = env.step(action_list)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return {"total_reward": total_reward, "steps": steps, "seed": seed}


def evaluate_policy(policy, config: dict, levels: list, seeds: list) -> pd.DataFrame:
    """Evalúa una política en múltiples niveles × semillas."""
    results = []

    for level in levels:
        for seed in seeds:
            route_file = f"sumo/routes_{level}_{seed}.rou.xml"
            if not os.path.exists(route_file):
                print(f"  Saltando {route_file} (no existe)")
                continue

            eval_config = config.copy()
            eval_config["simulation"] = config["simulation"].copy()

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

            # Parsear tripinfo AHORA que SUMO lo cerró correctamente
            tripinfo_path = os.path.join("results", "tripinfo.xml")
            metrics = parse_tripinfo(tripinfo_path)
            
            # Combinar métricas
            metrics.update(base_metrics)
            metrics["level"] = level
            metrics["policy"] = policy.name
            elapsed = time.time() - t0
            metrics["wall_time"] = elapsed
            results.append(metrics)

            avg_w = metrics.get("avg_wait", -1)
            n = metrics.get("n_trips", 0)
            print(f"trips={n} avg_wait={avg_w:.1f}s ({elapsed:.1f}s)")

    return pd.DataFrame(results)


def print_comparison(df: pd.DataFrame):
    """Imprime tabla comparativa de políticas."""
    if df.empty:
        print("No hay resultados para comparar.")
        return

    print(f"\n{'='*70}")
    print(f"  RESULTADOS DE EVALUACIÓN")
    print(f"{'='*70}\n")

    summary = df.groupby(["policy", "level"]).agg(
        n_trips=("n_trips", "mean"),
        avg_wait=("avg_wait", "mean"),
        avg_wait_std=("avg_wait", "std"),
        avg_duration=("avg_duration", "mean"),
        avg_speed=("avg_speed", "mean"),
        p95_wait=("p95_wait", "mean"),
        total_reward=("total_reward", "mean"),
    ).round(2)

    print(summary.to_string())
    print()

    global_summary = df.groupby("policy").agg(
        avg_wait=("avg_wait", "mean"),
        avg_wait_std=("avg_wait", "std"),
        avg_duration=("avg_duration", "mean"),
        avg_speed=("avg_speed", "mean"),
        total_reward=("total_reward", "mean"),
    ).round(2)

    print("Resumen global:")
    print(global_summary.to_string())


def main():
    parser = argparse.ArgumentParser(description="Evaluación de políticas")
    parser.add_argument("--policy", type=str,
                        choices=["fixed_time", "max_pressure", "random", "all"],
                        default="all")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--levels", type=str, nargs="+", default=["bajo", "medio", "alto"])
    parser.add_argument("--output", type=str, default="results/evaluation.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    os.makedirs("results", exist_ok=True)

    policies = {
        "fixed_time": FixedTimePolicy(),
        "max_pressure": MaxPressurePolicy(),
        "random": RandomPolicy(seed=42),
    }

    if args.policy == "all":
        selected = list(policies.values())
    else:
        selected = [policies[args.policy]]

    all_results = []

    for policy in selected:
        print(f"\n{'─'*50}")
        print(f"  Evaluando: {policy.name}")
        print(f"{'─'*50}")
        df = evaluate_policy(policy, config, args.levels, args.seeds)
        all_results.append(df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nResultados guardados en {args.output}")
        print_comparison(combined)


if __name__ == "__main__":
    main()
