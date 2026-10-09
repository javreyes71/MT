"""Script de línea base sin RL para evaluación comparativa.

Ejecuta la simulación SUMO con semáforos en modo automático (tiempos fijos)
y recolecta métricas para comparar con los agentes RL.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.metrics import MetricsCollector
import libsumo as traci
import sumolib


def run_baseline(config: dict, episodes: int, gui: bool) -> None:
    """Ejecuta la simulación de línea base sin RL."""

    if 'SUMO_HOME' not in os.environ:
        sys.exit("❌ Error: Variable de entorno SUMO_HOME no definida.")

    sumo_binary = sumolib.checkBinary('sumo-gui') if gui else sumolib.checkBinary('sumo')
    config_file = config['simulation']['config_file']
    duration = config['simulation'].get('duration', 3600)
    delta_time = config['simulation'].get('delta_time', 5.0)

    results_dir = os.path.join(config['paths']['results_dir'], 'baseline')
    os.makedirs(results_dir, exist_ok=True)

    print(f"🚦 Ejecutando línea base (tiempos fijos) por {episodes} episodio(s)...")

    all_summaries = []

    for ep in range(episodes):
        print(f"\n📊 Episodio {ep + 1}/{episodes}")
        collector = MetricsCollector()

        sumo_cmd = [sumo_binary, "-c", config_file, "--start", "--no-warnings"]
        traci.start(sumo_cmd)

        step = 0
        sim_time = 0.0
        while sim_time < duration:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()

            # Recolectar métricas cada delta_time pasos (mismo ritmo que el RL)
            if step % int(delta_time) == 0:
                collector.collect_step(step)

            step += 1

            # Terminar si ya no quedan vehículos después de un mínimo
            if traci.vehicle.getIDCount() == 0 and step > 100:
                print(f"   🏁 Simulación terminó en paso {step} (sin vehículos)")
                break

        traci.close()

        # Exportar métricas de este episodio
        ep_file = os.path.join(results_dir, f"baseline_ep{ep + 1}_steps.csv")
        collector.export_csv(ep_file)
        print(f"   💾 Métricas guardadas en {ep_file}")

        summary = collector.get_summary()
        summary['episodio'] = ep + 1
        all_summaries.append(summary)

        print(f"   📈 Resumen: espera_prom={summary.get('mean_avg_waiting_time', 0):.2f}s, "
              f"vel_prom={summary.get('mean_speed', 0):.2f}m/s, "
              f"detenidos={summary.get('mean_halted_vehicles', 0)}")

    # Exportar resumen de todos los episodios
    if all_summaries:
        import pandas as pd
        df = pd.DataFrame(all_summaries)
        summary_file = os.path.join(results_dir, "baseline_summary.csv")
        df.to_csv(summary_file, index=False)
        print(f"\n✅ Resumen de todos los episodios guardado en {summary_file}")

        # Promedios finales
        print("\n📊 Promedios finales del baseline:")
        for col in df.select_dtypes(include='number').columns:
            if col != 'episodio':
                print(f"   {col}: {df[col].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(description="🚦 Ejecutar línea base (tiempos fijos) en SUMO")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Archivo de configuración YAML")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Número de episodios (default: 3)")
    parser.add_argument("--gui", action="store_true",
                        help="Mostrar GUI de SUMO")
    args = parser.parse_args()

    config = load_config(args.config)
    run_baseline(config, args.episodes, args.gui)


if __name__ == "__main__":
    main()
