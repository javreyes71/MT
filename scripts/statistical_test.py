"""Análisis Estadístico para Tesis (Fase 4 del Roadmap).

Lee los resultados de la evaluación (results/evaluation.csv) generados por evaluate.py
y ejecuta pruebas de significancia estadística (Test de Wilcoxon) para demostrar
la superioridad del modelo RL sobre los baselines (especialmente MaxPressure).
También genera Boxplots de las distribuciones.

Uso:
    python scripts/statistical_test.py
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configuración para gráficos calidad Tesis
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def main():
    csv_path = "results/evaluation.csv"
    output_dir = "results/figures/statistics"
    
    if not os.path.exists(csv_path):
        print(f"❌ No se encontró {csv_path}. Debes correr evaluate.py primero.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Políticas a comparar (por defecto RL vs MaxPressure)
    target = "RL_MaskablePPO"
    baseline = "MaxPressure"
    
    if target not in df['policy'].values or baseline not in df['policy'].values:
        print(f"⚠️ Atención: Asegúrate de que '{target}' y '{baseline}' estén en el CSV.")
        print(f"Políticas detectadas: {df['policy'].unique()}")
        
    # --- 1. Generación de Boxplots ---
    print("\n📈 Generando Boxplots de Distribución...")
    metrics_to_plot = {
        'avg_wait': ('Tiempo de Espera Promedio (s)', 'Espera Promedio'),
        'avg_halts': ('Número de Paradas Promedio', 'Efecto Stop-and-Go'),
        'avg_duration': ('Tiempo de Viaje (s)', 'Duración del Viaje')
    }
    
    for metric, (ylabel, title) in metrics_to_plot.items():
        if metric in df.columns:
            plt.figure()
            sns.boxplot(data=df, x='policy', y=metric, hue='level', palette='Set2')
            plt.title(f'Distribución de {title} por Demanda')
            plt.ylabel(ylabel)
            plt.xlabel('Algoritmo')
            plt.grid(axis='y', alpha=0.3)
            out_path = os.path.join(output_dir, f'boxplot_{metric}.png')
            plt.savefig(out_path)
            plt.close()
            print(f"   📊 Guardado: {out_path}")
            
    # --- 2. Test Estadístico de Wilcoxon (Pareado) ---
    print("\n🧮 Ejecutando Test de Wilcoxon de los Rangos con Signo...")
    print(f"   Comparando: {target} vs {baseline}")
    
    results_text = []
    results_text.append(f"INFORME ESTADÍSTICO - {target} vs {baseline}\n")
    results_text.append("="*50 + "\n")
    
    for level in df['level'].unique():
        df_level = df[df['level'] == level]
        
        target_data = df_level[df_level['policy'] == target].sort_values('seed')
        baseline_data = df_level[df_level['policy'] == baseline].sort_values('seed')
        
        if len(target_data) == 0 or len(baseline_data) == 0:
            continue
            
        # Asegurarnos de que las semillas coincidan (Pareado)
        merged = pd.merge(target_data, baseline_data, on='seed', suffixes=('_rl', '_base'))
        
        results_text.append(f"\n>> DEMANDA: {level.upper()} (N={len(merged)} episodios)\n")
        
        for metric, (ylabel, _) in metrics_to_plot.items():
            col_rl = f'{metric}_rl'
            col_base = f'{metric}_base'
            
            if col_rl in merged.columns and col_base in merged.columns:
                diff = merged[col_rl] - merged[col_base]
                
                # Si todos los valores son iguales, Wilcoxon tira error
                if (diff == 0).all():
                    results_text.append(f"  - {ylabel}: Empate exacto en todas las semillas.")
                    continue
                    
                stat, p_value = stats.wilcoxon(merged[col_rl], merged[col_base])
                
                mean_rl = merged[col_rl].mean()
                mean_base = merged[col_base].mean()
                improvement = ((mean_base - mean_rl) / mean_base) * 100 if mean_base > 0 else 0
                
                sig = "SIGNIFICATIVO (✅)" if p_value < 0.05 else "NO SIGNIFICATIVO (❌)"
                
                results_text.append(f"  - {ylabel}:")
                results_text.append(f"      {target}: {mean_rl:.2f} | {baseline}: {mean_base:.2f} ({improvement:+.1f}%)")
                results_text.append(f"      P-Value: {p_value:.4f} -> {sig}\n")
                
    # Guardar reporte de texto
    report_path = os.path.join(output_dir, "reporte_wilcoxon.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(results_text)
        
    for line in results_text:
        print(line, end="")
        
    print(f"\n✅ Análisis guardado en {output_dir}/")

if __name__ == "__main__":
    main()
