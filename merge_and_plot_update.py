import pandas as pd
import os

# Archivos de entrada
f_base = "results_large/evaluation.csv" # AquÃ­ estÃ¡n los baselines (FixedTime, SCATS, SCOOT)
f_adrl_15m = "results_large/eval_adrl_15M.csv"
f_acur = "results_large/eval_acurriculum.csv" # El de 10M que ya tenÃ­amos
f_out = "results_large/evaluation_final.csv"

dfs = []

# Cargar baselines (excluyendo cualquier RL viejo)
if os.path.exists(f_base):
    df_base = pd.read_csv(f_base)
    df_base = df_base[df_base["policy"] != "RL_MaskablePPO"]
    df_base = df_base[df_base["policy"] != "ADRL"]
    df_base = df_base[df_base["policy"] != "ADRL_Large"]
    df_base = df_base[df_base["policy"] != "ACurriculum"]
    dfs.append(df_base)

# Cargar nuevo ADRL
if os.path.exists(f_adrl_15m):
    df_adrl = pd.read_csv(f_adrl_15m)
    df_adrl["policy"] = "ADRL_Large (15.6M)"
    dfs.append(df_adrl)

# Cargar ACurriculum viejo (ya que no hay uno nuevo aÃºn)
if os.path.exists(f_acur):
    df_acur = pd.read_csv(f_acur)
    df_acur["policy"] = "ACurriculum (Fase 1: 10M)"
    dfs.append(df_acur)

if dfs:
    df_final = pd.concat(dfs, ignore_index=True)
    df_final.to_csv(f_out, index=False)
    print("MÃ©tricas unificadas guardadas en", f_out)
    
    # Graficar
    import sys
    sys.path.append(os.getcwd())
    from scripts.plot_metrics import generate_all_plots
    generate_all_plots(f_out, "results_large/plots_final")
    print("GrÃ¡ficos generados en results_large/plots_final")
