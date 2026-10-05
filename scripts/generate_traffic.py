"""Script para generar tráfico aleatorio para SUMO en múltiples niveles."""
import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Generador de demanda multiescenario para SUMO")
    parser.add_argument("--net", type=str, default="sumo/network.net.xml")
    parser.add_argument("--outdir", type=str, default="sumo")
    parser.add_argument("--seeds", type=int, default=3, help="Cantidad de semillas por nivel")
    args = parser.parse_args()
    
    random_trips_path = os.path.join(os.environ.get("SUMO_HOME", ""), "tools", "randomTrips.py")
    if not os.path.exists(random_trips_path):
        print("❌ Error: No se encontró randomTrips.py. Verifique SUMO_HOME.")
        sys.exit(1)
        
    levels = {
        'bajo': 2.0,   # inserción cada 2.0s
        'medio': 1.0,  # inserción cada 1.0s
        'alto': 0.7    # inserción cada 0.7s (calibrado)
    }
    
    end_time = 3000
    
    for level, p_val in levels.items():
        for i in range(args.seeds):
            seed = 42 + i
            routes_file = os.path.join(args.outdir, f"routes_{level}_{seed}.rou.xml")
            
            cmd = [
                sys.executable, random_trips_path,
                "-n", args.net,
                "-r", routes_file,
                "-e", str(end_time),
                "-p", str(p_val),
                "--seed", str(seed),
                "--fringe-factor", "10",
                "--validate",
                "--vclass", "passenger"
            ]
            
            print(f"🛣️ Generando {routes_file} (nivel {level}, seed {seed}, p={p_val})...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            
    print("✅ Generación de demanda completada.")

if __name__ == "__main__":
    main()
