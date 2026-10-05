# visualize.py
import time
import os
import argparse
import glob
from stable_baselines3 import PPO
from traffic_env_sumo import TrafficSumoEnv

def get_latest_model(models_dir="models/ppo_multi_agent"):
    list_of_files = glob.glob(f"{models_dir}/*.zip")
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file.replace(".zip", "")

def main():
    parser = argparse.ArgumentParser(description="Visualizar el agente de tráfico en SUMO.")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo (sin .zip). Por defecto, el último guardado.")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = get_latest_model()
        if model_path is None:
            print("❌ No se encontraron modelos en la carpeta models/ppo_multi_agent/")
            return
        print(f"🔵 Usando el último modelo encontrado: {model_path}")

    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ No encuentro {model_path}.zip. Verifica la ruta del modelo."); return

    print("🔵 Abriendo visualizador (el entorno generará el estado)...")
    env = TrafficSumoEnv(gui=True)
    
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Error cargando modelo. Asegúrate de que coincida con el código actual. Detalles: {e}")
        return

    obs, _ = env.reset()
    done = False
    
    print("🟢 Simulación iniciada.")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time.sleep(0.05) # Control de velocidad visual

    env.close()

if __name__ == "__main__":
    main()