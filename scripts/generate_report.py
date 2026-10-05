"""Genera un reporte PDF con resultados comparativos: RL vs Tiempos Fijos (Baseline).

Autor: Javier Reyes Gunther
Tesis: Sistema Multi-Agente para Control Adaptativo de Tráfico Urbano Mediante RL
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Estilo profesional
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'baseline': '#D32F2F',
    'rl': '#1976D2',
}


def create_report(results_dir: str, output_path: str):
    """Genera el reporte PDF completo."""

    # --- Cargar datos ---
    baseline_files = []
    baseline_dir = os.path.join(results_dir, 'baseline')
    if os.path.exists(baseline_dir):
        baseline_files = sorted([
            os.path.join(baseline_dir, f)
            for f in os.listdir(baseline_dir)
            if f.endswith('_steps.csv')
        ])

    if not baseline_files:
        print("❌ No se encontraron datos del baseline. Ejecuta primero: python scripts/baseline.py")
        return

    # Cargar todos los episodios del baseline
    baseline_dfs = [pd.read_csv(f) for f in baseline_files]
    baseline_all = pd.concat(baseline_dfs, ignore_index=True)

    # Promediar por step number (si hay múltiples episodios)
    baseline_avg = baseline_all.groupby('step').mean().reset_index()

    # Intentar cargar datos de RL si existen
    rl_data = None
    training_file = os.path.join(results_dir, 'training_metrics.csv')
    if os.path.exists(training_file):
        rl_data = pd.read_csv(training_file)

    # Buscar datos detallados del RL
    rl_step_data = None
    for mode in ['centralized', 'independent', 'parameter_sharing']:
        mode_dir = os.path.join(results_dir, mode)
        if os.path.exists(mode_dir):
            mode_files = sorted([
                os.path.join(mode_dir, f)
                for f in os.listdir(mode_dir)
                if f.endswith('_steps.csv')
            ])
            if mode_files:
                rl_step_data = pd.concat([pd.read_csv(f) for f in mode_files], ignore_index=True)
                break

    print(f"📊 Datos cargados:")
    print(f"   Baseline: {len(baseline_files)} episodio(s), {len(baseline_all)} filas")
    if rl_step_data is not None:
        print(f"   RL: {len(rl_step_data)} filas")
    elif rl_data is not None:
        print(f"   RL: resumen disponible ({len(rl_data)} filas)")

    # --- Generar PDF ---
    with PdfPages(output_path) as pdf:

        # === PÁGINA 1: Portada ===
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        ax.text(0.5, 0.75,
                'Sistema Multi-Agente para Control\nAdaptativo de Tráfico Urbano',
                ha='center', va='center', fontsize=24, fontweight='bold',
                color='#1565C0')
        ax.text(0.5, 0.60,
                'Reporte de Resultados Experimentales',
                ha='center', va='center', fontsize=16, color='#424242')
        ax.text(0.5, 0.50,
                'Comparación: Control RL vs Tiempos Fijos (Cíclico)',
                ha='center', va='center', fontsize=14, color='#616161')

        ax.text(0.5, 0.35,
                f'Javier Reyes Gunther\n'
                f'Universidad de Los Lagos — Ingeniería Civil en Informática\n'
                f'Profesor Guía: Felipe Olivares Acuña',
                ha='center', va='center', fontsize=11, color='#757575')
        ax.text(0.5, 0.20,
                f'Generado: {datetime.now().strftime("%d/%m/%Y %H:%M")}',
                ha='center', va='center', fontsize=10, color='#9E9E9E')

        # Línea decorativa
        ax.axhline(y=0.42, xmin=0.2, xmax=0.8, color='#1565C0', linewidth=2)

        pdf.savefig(fig)
        plt.close()

        # === PÁGINA 2: Resumen de métricas del baseline ===
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Análisis del Baseline (Tiempos Fijos)', fontsize=16, fontweight='bold', y=0.98)

        smooth = 15  # ventana de suavizado

        # Tiempo de espera
        ax = axes[0, 0]
        vals = baseline_avg['avg_waiting_time']
        ax.plot(baseline_avg['step'], vals.rolling(smooth, min_periods=1).mean(),
                color=COLORS['baseline'], linewidth=1.5, label='Tiempos Fijos')
        ax.fill_between(baseline_avg['step'],
                        vals.rolling(smooth, min_periods=1).min(),
                        vals.rolling(smooth, min_periods=1).max(),
                        alpha=0.15, color=COLORS['baseline'])
        ax.set_title('Tiempo de Espera Promedio')
        ax.set_ylabel('Segundos (s)')
        ax.set_xlabel('Paso de Simulación')
        ax.legend()

        # Vehículos detenidos
        ax = axes[0, 1]
        vals = baseline_avg['total_halted_vehicles']
        ax.plot(baseline_avg['step'], vals.rolling(smooth, min_periods=1).mean(),
                color=COLORS['baseline'], linewidth=1.5, label='Tiempos Fijos')
        ax.fill_between(baseline_avg['step'],
                        vals.rolling(smooth, min_periods=1).min(),
                        vals.rolling(smooth, min_periods=1).max(),
                        alpha=0.15, color=COLORS['baseline'])
        ax.set_title('Vehículos Detenidos')
        ax.set_ylabel('Cantidad')
        ax.set_xlabel('Paso de Simulación')
        ax.legend()

        # Velocidad promedio
        ax = axes[1, 0]
        vals = baseline_avg['avg_speed']
        ax.plot(baseline_avg['step'], vals.rolling(smooth, min_periods=1).mean(),
                color=COLORS['baseline'], linewidth=1.5, label='Tiempos Fijos')
        ax.fill_between(baseline_avg['step'],
                        vals.rolling(smooth, min_periods=1).min(),
                        vals.rolling(smooth, min_periods=1).max(),
                        alpha=0.15, color=COLORS['baseline'])
        ax.set_title('Velocidad Promedio')
        ax.set_ylabel('m/s')
        ax.set_xlabel('Paso de Simulación')
        ax.legend()

        # Emisiones CO2
        ax = axes[1, 1]
        vals = baseline_avg['total_co2_emissions']
        ax.plot(baseline_avg['step'], vals.rolling(smooth, min_periods=1).mean(),
                color=COLORS['baseline'], linewidth=1.5, label='Tiempos Fijos')
        ax.fill_between(baseline_avg['step'],
                        vals.rolling(smooth, min_periods=1).min(),
                        vals.rolling(smooth, min_periods=1).max(),
                        alpha=0.15, color=COLORS['baseline'])
        ax.set_title('Emisiones CO₂')
        ax.set_ylabel('mg')
        ax.set_xlabel('Paso de Simulación')
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # === PÁGINA 3: Comparación RL vs Baseline (si hay datos RL) ===
        if rl_data is not None:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle('Comparación: RL (Centralizado) vs Tiempos Fijos',
                         fontsize=16, fontweight='bold', y=0.98)

            # Calcular promedios del baseline para comparar
            bl_means = {
                'avg_waiting_time': baseline_all['avg_waiting_time'].mean(),
                'total_halted_vehicles': baseline_all['total_halted_vehicles'].mean(),
                'avg_speed': baseline_all['avg_speed'].mean(),
                'total_co2_emissions': baseline_all['total_co2_emissions'].mean(),
            }

            rl_means = {}
            if rl_data is not None and len(rl_data) > 0:
                col_map = {
                    'avg_waiting_time': 'mean_avg_waiting_time',
                    'total_halted_vehicles': 'mean_halted_vehicles',
                    'avg_speed': 'mean_speed',
                    'total_co2_emissions': 'total_co2_emissions',
                }
                for key, col in col_map.items():
                    if col in rl_data.columns:
                        rl_means[key] = rl_data[col].mean()

            comparisons = [
                ('avg_waiting_time', 'Tiempo de Espera Promedio', 's', True),   # lower is better
                ('total_halted_vehicles', 'Vehículos Detenidos', 'Cantidad', True),
                ('avg_speed', 'Velocidad Promedio', 'm/s', False),  # higher is better
                ('total_co2_emissions', 'Emisiones CO₂', 'mg', True),
            ]

            for idx, (metric, title, unit, lower_better) in enumerate(comparisons):
                ax = axes[idx // 2, idx % 2]
                bl_val = bl_means.get(metric, 0)
                rl_val = rl_means.get(metric, 0)

                bars = ax.bar(['Tiempos Fijos\n(Baseline)', 'RL\n(Centralizado)'],
                              [bl_val, rl_val],
                              color=[COLORS['baseline'], COLORS['rl']],
                              alpha=0.85, edgecolor='white', linewidth=2)

                # Etiquetas de valor
                for bar, val in zip(bars, [bl_val, rl_val]):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.02 * max(bl_val, rl_val),
                            f'{val:.1f}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

                # Porcentaje de mejora
                if bl_val > 0:
                    if lower_better:
                        improvement = ((bl_val - rl_val) / bl_val) * 100
                    else:
                        improvement = ((rl_val - bl_val) / bl_val) * 100
                    color = '#4CAF50' if improvement > 0 else '#F44336'
                    sign = '+' if improvement > 0 else ''
                    ax.text(0.95, 0.95, f'{sign}{improvement:.1f}%',
                            transform=ax.transAxes, ha='right', va='top',
                            fontsize=14, fontweight='bold', color=color,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                      edgecolor=color, alpha=0.8))

                ax.set_title(title)
                ax.set_ylabel(unit)

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

        # === PÁGINA 4: Evolución temporal detallada ===
        fig, axes = plt.subplots(3, 1, figsize=(11, 10))
        fig.suptitle('Evolución Temporal — Comportamiento del Tráfico',
                     fontsize=16, fontweight='bold', y=0.98)

        # Tiempo de espera a lo largo de la simulación
        ax = axes[0]
        ax.plot(baseline_avg['step'], baseline_avg['avg_waiting_time'],
                color=COLORS['baseline'], linewidth=0.5, alpha=0.3)
        ax.plot(baseline_avg['step'],
                baseline_avg['avg_waiting_time'].rolling(smooth, min_periods=1).mean(),
                color=COLORS['baseline'], linewidth=2, label='Tiempos Fijos (suavizado)')
        ax.set_title('Tiempo de Espera Promedio — Evolución Completa')
        ax.set_ylabel('Segundos (s)')
        ax.legend()

        # Throughput acumulado
        ax = axes[1]
        ax.plot(baseline_avg['step'], baseline_avg['throughput'].cumsum(),
                color=COLORS['baseline'], linewidth=2, label='Tiempos Fijos')
        ax.set_title('Throughput Acumulado (Vehículos que Completaron su Ruta)')
        ax.set_ylabel('Vehículos')
        ax.legend()

        # Cola promedio
        ax = axes[2]
        ax.plot(baseline_avg['step'],
                baseline_avg['avg_queue_length'].rolling(smooth, min_periods=1).mean(),
                color=COLORS['baseline'], linewidth=2, label='Tiempos Fijos')
        ax.set_title('Longitud de Cola Promedio')
        ax.set_ylabel('Vehículos en Cola')
        ax.set_xlabel('Paso de Simulación')
        ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

        # === PÁGINA 5: Tabla resumen ===
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.6])
        ax.axis('off')

        ax.text(0.5, 0.95, 'Tabla Resumen de Métricas',
                ha='center', va='top', fontsize=18, fontweight='bold',
                transform=ax.transAxes)

        metrics_table = [
            ['Métrica', 'Tiempos Fijos\n(Baseline)', 'RL Centralizado\n(Preliminar)', 'Mejora'],
        ]

        metric_names = {
            'avg_waiting_time': 'Espera Prom. (s)',
            'total_halted_vehicles': 'Vehículos Detenidos',
            'avg_speed': 'Velocidad Prom. (m/s)',
            'total_co2_emissions': 'CO₂ Total (mg)',
            'throughput': 'Throughput',
            'avg_queue_length': 'Cola Promedio',
        }

        bl_summary = baseline_all.describe().loc['mean']
        rl_summary = rl_data.iloc[0] if rl_data is not None and len(rl_data) > 0 else None

        col_map_rl = {
            'avg_waiting_time': 'mean_avg_waiting_time',
            'total_halted_vehicles': 'mean_halted_vehicles',
            'avg_speed': 'mean_speed',
            'total_co2_emissions': 'total_co2_emissions',
            'throughput': 'total_throughput',
            'avg_queue_length': 'mean_queue_length',
        }

        for metric, name in metric_names.items():
            bl_val = bl_summary.get(metric, 0)
            rl_col = col_map_rl.get(metric, '')
            rl_val = rl_summary.get(rl_col, None) if rl_summary is not None and rl_col in (rl_summary.index if rl_summary is not None else []) else None

            bl_str = f'{bl_val:.2f}'
            if rl_val is not None:
                rl_str = f'{rl_val:.2f}'
                if bl_val > 0:
                    lower_better = metric != 'avg_speed' and metric != 'throughput'
                    if lower_better:
                        imp = ((bl_val - rl_val) / bl_val) * 100
                    else:
                        imp = ((rl_val - bl_val) / bl_val) * 100
                    imp_str = f'{imp:+.1f}%'
                else:
                    imp_str = 'N/A'
            else:
                rl_str = '—'
                imp_str = '—'

            metrics_table.append([name, bl_str, rl_str, imp_str])

        table = ax.table(cellText=metrics_table, loc='center', cellLoc='center',
                         colWidths=[0.30, 0.25, 0.25, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0)

        # Estilizar encabezado
        for j in range(4):
            table[0, j].set_facecolor('#1565C0')
            table[0, j].set_text_props(color='white', fontweight='bold')

        # Alternar colores de filas
        for i in range(1, len(metrics_table)):
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(4):
                table[i, j].set_facecolor(color)

        ax.text(0.5, 0.05,
                'Nota: Los valores de RL corresponden a un entrenamiento preliminar (1,000 timesteps).\n'
                'Los resultados finales se obtendrán con entrenamientos de 200,000+ timesteps.',
                ha='center', va='bottom', fontsize=9, style='italic', color='#757575',
                transform=ax.transAxes)

        pdf.savefig(fig)
        plt.close()

    print(f"\n✅ Reporte PDF generado: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="📄 Generar reporte PDF de resultados")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/reporte_resultados.pdf")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
