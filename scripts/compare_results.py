"""Script de comparación y visualización de resultados experimentales.

Genera gráficas comparativas entre baseline (tiempos fijos) y los agentes RL
para la tesis: Sistema Multi-Agente para Control Adaptativo de Tráfico.
"""
import argparse
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para generar imágenes

# Estilo profesional para la tesis
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

# Colores consistentes por modo
COLORS = {
    'baseline': '#888888',
    'centralized': '#2196F3',
    'independent': '#FF9800',
    'parameter_sharing': '#4CAF50',
}

LABELS = {
    'baseline': 'Tiempos Fijos (Baseline)',
    'centralized': 'PPO Centralizado',
    'independent': 'Independent Learners',
    'parameter_sharing': 'MAPPO (Parameter Sharing)',
}


def load_step_data(results_dir: str) -> dict:
    """Carga datos de pasos de todos los experimentos disponibles."""
    data = {}

    # Baseline
    baseline_files = glob.glob(os.path.join(results_dir, 'baseline', '*_steps.csv'))
    if baseline_files:
        frames = [pd.read_csv(f) for f in baseline_files]
        data['baseline'] = pd.concat(frames, ignore_index=True)
        print(f"   📂 Baseline: {len(baseline_files)} archivo(s), {len(data['baseline'])} filas")

    # Modos RL
    for mode in ['centralized', 'independent', 'parameter_sharing']:
        mode_files = glob.glob(os.path.join(results_dir, mode, '*_steps.csv'))
        if not mode_files:
            mode_files = glob.glob(os.path.join(results_dir, f'{mode}*.csv'))
        if mode_files:
            frames = [pd.read_csv(f) for f in mode_files]
            data[mode] = pd.concat(frames, ignore_index=True)
            print(f"   📂 {LABELS.get(mode, mode)}: {len(mode_files)} archivo(s), {len(data[mode])} filas")

    return data


def load_summary_data(results_dir: str) -> dict:
    """Carga los resúmenes de cada experimento."""
    summaries = {}

    baseline_summary = os.path.join(results_dir, 'baseline', 'baseline_summary.csv')
    if os.path.exists(baseline_summary):
        summaries['baseline'] = pd.read_csv(baseline_summary)

    for mode in ['centralized', 'independent', 'parameter_sharing']:
        summary_file = os.path.join(results_dir, mode, f'{mode}_summary.csv')
        if os.path.exists(summary_file):
            summaries[mode] = pd.read_csv(summary_file)

    return summaries


def plot_metric_comparison(data: dict, metric: str, title: str, ylabel: str,
                           output_path: str, smooth_window: int = 20):
    """Genera gráfico de líneas comparando una métrica entre experimentos."""
    fig, ax = plt.subplots()

    for mode, df in data.items():
        if metric in df.columns:
            values = df[metric].values
            # Suavizado con media móvil
            if len(values) > smooth_window:
                smoothed = pd.Series(values).rolling(window=smooth_window, min_periods=1).mean()
            else:
                smoothed = values

            ax.plot(smoothed, label=LABELS.get(mode, mode),
                    color=COLORS.get(mode, '#000000'), linewidth=1.5, alpha=0.85)

    ax.set_title(title)
    ax.set_xlabel('Paso de Simulación')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path)
    plt.close()
    print(f"   📊 Gráfico guardado: {output_path}")


def plot_bar_comparison(summaries: dict, metrics: list, titles: list,
                        ylabels: list, output_path: str):
    """Genera gráfico de barras comparando métricas promedio."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    modes = list(summaries.keys())

    for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[i]
        values = []
        labels = []
        colors = []

        for mode in modes:
            df = summaries[mode]
            if metric in df.columns:
                values.append(df[metric].mean())
                labels.append(LABELS.get(mode, mode))
                colors.append(COLORS.get(mode, '#000000'))

        bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)

        # Añadir valores sobre las barras
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(values),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"   📊 Gráfico de barras guardado: {output_path}")


def generate_comparison_table(summaries: dict, output_path: str):
    """Genera tabla LaTeX de comparación para la tesis."""
    rows = []
    for mode, df in summaries.items():
        row = {'Método': LABELS.get(mode, mode)}
        for col in df.select_dtypes(include='number').columns:
            if col != 'episodio':
                row[col] = f"{df[col].mean():.2f} ± {df[col].std():.2f}"
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)

    # También generar versión LaTeX
    latex_path = output_path.replace('.csv', '.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("% Tabla generada automáticamente por compare_results.py\n")
        f.write(result_df.to_latex(index=False, escape=False))

    print(f"   📋 Tabla comparativa: {output_path}")
    print(f"   📋 Tabla LaTeX: {latex_path}")


def main():
    parser = argparse.ArgumentParser(
        description="📊 Comparar resultados de experimentos de control de tráfico")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directorio con los resultados")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="Directorio para las gráficas")
    parser.add_argument("--smooth", type=int, default=20,
                        help="Ventana de suavizado para gráficos de línea")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("📊 Cargando datos de experimentos...")
    step_data = load_step_data(args.results_dir)
    summaries = load_summary_data(args.results_dir)

    if not step_data and not summaries:
        print("❌ No se encontraron datos de experimentos en", args.results_dir)
        print("   Ejecuta primero: python scripts/baseline.py")
        print("   y luego:         python scripts/train.py --mode centralized")
        return

    # --- Gráficos de series de tiempo ---
    print("\n📈 Generando gráficos de series de tiempo...")

    metric_configs = [
        ('avg_waiting_time', 'Tiempo de Espera Promedio por Paso',
         'Tiempo de Espera (s)', 'waiting_time_comparison.png'),
        ('total_halted_vehicles', 'Vehículos Detenidos por Paso',
         'Cantidad de Vehículos', 'halted_vehicles_comparison.png'),
        ('avg_speed', 'Velocidad Promedio por Paso',
         'Velocidad (m/s)', 'speed_comparison.png'),
        ('throughput', 'Throughput por Paso',
         'Vehículos Procesados', 'throughput_comparison.png'),
        ('total_co2_emissions', 'Emisiones CO₂ por Paso',
         'CO₂ (mg)', 'co2_comparison.png'),
        ('avg_queue_length', 'Longitud de Cola Promedio',
         'Vehículos en Cola', 'queue_length_comparison.png'),
    ]

    for metric, title, ylabel, filename in metric_configs:
        available = {m: d for m, d in step_data.items() if metric in d.columns}
        if available:
            plot_metric_comparison(
                available, metric, title, ylabel,
                os.path.join(args.output_dir, filename),
                smooth_window=args.smooth
            )

    # --- Gráfico de barras resumen ---
    if summaries:
        print("\n📊 Generando gráficos de barras comparativas...")
        bar_metrics = []
        bar_titles = []
        bar_ylabels = []

        candidate_metrics = [
            ('mean_avg_waiting_time', 'Tiempo de Espera Promedio', 'Segundos (s)'),
            ('mean_speed', 'Velocidad Promedio', 'm/s'),
            ('mean_halted_vehicles', 'Vehículos Detenidos', 'Cantidad'),
        ]

        for metric, title, ylabel in candidate_metrics:
            if any(metric in df.columns for df in summaries.values()):
                bar_metrics.append(metric)
                bar_titles.append(title)
                bar_ylabels.append(ylabel)

        if bar_metrics:
            plot_bar_comparison(
                summaries, bar_metrics, bar_titles, bar_ylabels,
                os.path.join(args.output_dir, 'summary_comparison.png')
            )

        # --- Tabla comparativa ---
        print("\n📋 Generando tabla comparativa...")
        generate_comparison_table(
            summaries,
            os.path.join(args.output_dir, 'comparison_table.csv')
        )

    print(f"\n✅ Todas las gráficas guardadas en: {args.output_dir}/")
    print("   Usa estas imágenes directamente en tu tesis LaTeX.")


if __name__ == "__main__":
    main()
