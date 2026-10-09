import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Constantes de proyección ambiental (Calibradas para parque chileno - MMA)
CO2_IDLING_G_S = 0.64     # gramos de CO2 por segundo de espera (ralentí)
CO2_HALT_G = 11.55        # gramos de CO2 extra por frenar y volver a acelerar
FUEL_IDLING_ML_S = 0.28   # mililitros de combustible por segundo
FUEL_HALT_ML = 5.0        # mililitros extra por frenar y acelerar

POLICY_COLORS = {
    'ADRL': '#00BCD4',         # Cyan
    'ACurriculum': '#E91E63',  # Magenta
    'SCATS': '#FF9800',        # Naranja
    'SCOOT': '#FFC107',        # Amarillo
    'FixedTime': '#9E9E9E',    # Gris
    'MaxPressure': '#F44336'   # Rojo
}

def clean_policy_name(name):
    if 'SCATS' in name: return 'SCATS'
    if 'SCOOT' in name: return 'SCOOT'
    if 'ACurriculum' in name: return 'ACurriculum'
    if 'ADRL' in name or 'RL_MaskablePPO' in name: return 'ADRL'
    if 'FixedTime' in name: return 'FixedTime'
    if 'MaxPressure' in name: return 'MaxPressure'
    return name

def calculate_projections(df):
    """Añade columnas de proyección de CO2 y Combustible."""
    # Promedio por vehículo
    df['CO2_Promedio_g'] = (df['avg_wait'] * CO2_IDLING_G_S) + (df['avg_halts'] * CO2_HALT_G)
    df['Fuel_Promedio_ml'] = (df['avg_wait'] * FUEL_IDLING_ML_S) + (df['avg_halts'] * FUEL_HALT_ML)
    return df

def create_dashboard(df, title, out_path):
    sns.set_theme(style="whitegrid")
    
    # ORDENAR de menor a mayor (orden creciente) por Tiempo de Espera
    df = df.sort_values(by='Tiempo_Espera_s', ascending=True)
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
    
    # 1. Tiempo de Espera
    ax1 = fig.add_subplot(231)
    sns.barplot(data=df.reset_index(), x='policy', y='Tiempo_Espera_s', ax=ax1, palette=POLICY_COLORS, hue='policy', legend=False)
    ax1.set_title('Tiempo de Espera\n(Segundos en promedio que un vehículo pasa a 0 km/h)', fontweight='bold', fontsize=11, pad=10)
    ax1.set_xlabel('')
    ax1.set_ylabel('Segundos (s)')
    ax1.tick_params(axis='x', rotation=25)
    for i in ax1.containers:
        ax1.bar_label(i, fmt='%.1f', padding=3)
        
    # 2. Combustible
    ax2 = fig.add_subplot(232)
    sns.barplot(data=df.reset_index(), x='policy', y='Fuel_Promedio_ml', ax=ax2, palette=POLICY_COLORS, hue='policy', legend=False)
    ax2.set_title('Combustible Consumido\n(Mide el gasto en ralentí y los picos por frenazos)', fontweight='bold', fontsize=11, pad=10)
    ax2.set_xlabel('')
    ax2.set_ylabel('Mililitros (ml)')
    ax2.tick_params(axis='x', rotation=25)
    for i in ax2.containers:
        ax2.bar_label(i, fmt='%.1f', padding=3)

    # 3. Emisiones CO2
    ax3 = fig.add_subplot(233)
    sns.barplot(data=df.reset_index(), x='policy', y='CO2_Promedio_g', ax=ax3, palette=POLICY_COLORS, hue='policy', legend=False)
    ax3.set_title('Emisiones de CO2\n(Impacto de gases por aceleraciones desde cero)', fontweight='bold', fontsize=11, pad=10)
    ax3.set_xlabel('')
    ax3.set_ylabel('Gramos (g)')
    ax3.tick_params(axis='x', rotation=25)
    for i in ax3.containers:
        ax3.bar_label(i, fmt='%.1f', padding=3)
        
    # 4. Tabla de Datos ampliada
    ax4 = fig.add_subplot(212)
    ax4.axis('off')
    
    table_data = df.round(2).reset_index()
    table_data.rename(columns={'policy': 'Algoritmo (Control)', 'Tiempo_Espera_s': 'Espera (s)', 'Paradas_Promedio': 'Paradas (Halts)', 'Teletransportaciones': 'Teletransportaciones', 'CO2_Promedio_g': 'CO2/Veh (g)', 'Fuel_Promedio_ml': 'Combustible/Veh (ml)'}, inplace=True)
    
    table = ax4.table(cellText=table_data.values,
                      colLabels=table_data.columns,
                      cellLoc='center',
                      loc='center')
                      
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2C3E50')
        elif col == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#ECF0F1')
            
    # Fuente
    fig.text(0.5, 0.02, "Fuente: Elaboración propia basada en proyecciones termodinámicas del simulador de tráfico SUMO.", ha='center', fontsize=12, style='italic', color='gray')
            
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.3)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_plots(csv_path="results/evaluation.csv", out_dir="results/plots"):
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Limpiar nombres
    df['policy'] = df['policy'].apply(clean_policy_name)
    
    df = calculate_projections(df)
    os.makedirs(out_dir, exist_ok=True)
    
    global_df = df.groupby("policy").agg(
        Tiempo_Espera_s=("avg_wait", "mean"),
        Paradas_Promedio=("avg_halts", "mean"),
        Teletransportaciones=("teleports", "sum"),
        CO2_Promedio_g=("CO2_Promedio_g", "mean"),
        Fuel_Promedio_ml=("Fuel_Promedio_ml", "mean")
    )
    create_dashboard(global_df, "Desempeño Global de Algoritmos (Promedio diario)", os.path.join(out_dir, "tabla_global_v2.png"))
    
    horarios_map = {
        "bajo": "11:00 AM (Horario Valle)",
        "medio": "19:30 PM (Horario de Transición)",
        "alto": "18:00 PM (Horario Punta)"
    }
    
    for level, level_name in horarios_map.items():
        level_df = df[df["level"] == level].groupby("policy").agg(
            Tiempo_Espera_s=("avg_wait", "mean"),
            Paradas_Promedio=("avg_halts", "mean"),
            Teletransportaciones=("teleports", "sum"),
            CO2_Promedio_g=("CO2_Promedio_g", "mean"),
            Fuel_Promedio_ml=("Fuel_Promedio_ml", "mean")
        )
        if not level_df.empty:
            create_dashboard(level_df, f"Evaluación de Tráfico - Escenario: {level_name}", os.path.join(out_dir, f"tabla_{level}_v2.png"))
            
    print(f"Dashboards PNG generados exitosamente en '{out_dir}'")

if __name__ == "__main__":
    generate_all_plots()
