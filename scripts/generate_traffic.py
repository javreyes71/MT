import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Generador de demanda multiescenario para SUMO")
    parser.add_argument("--net", type=str, default="sumo/network.net.xml")
    parser.add_argument("--outdir", type=str, default="sumo")
    parser.add_argument("--prefix", type=str, default="routes", help="Prefijo de archivo")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    args = parser.parse_args()
    
    random_trips_path = os.path.join(os.environ.get("SUMO_HOME", ""), "tools", "randomTrips.py")
    if not os.path.exists(random_trips_path):
        print("Error: No se encontró randomTrips.py.")
        sys.exit(1)
        
    levels = {
        'bajo': 2.0,
        'medio': 1.0,
        'alto': 0.7
    }
    
    for level, p_val in levels.items():
        for seed in args.seeds:
            routes_file = os.path.join(args.outdir, f"{args.prefix}_{level}_{seed}.rou.xml")
            
            cmd = [
                sys.executable, random_trips_path,
                "-n", args.net,
                "-r", routes_file,
                "-e", "3600",
                "-p", str(p_val),
                "--seed", str(seed),
                "--fringe-factor", "10",
                "--validate",
                "--vclass", "passenger"
            ]
            
            print(f"Generando {routes_file}...")
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

if __name__ == "__main__":
    main()
