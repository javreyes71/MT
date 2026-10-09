import os

# Factores empíricos de Osorno:
# Parque Automotriz: ~55,000 vehículos
# Hora Punta (10%): ~5,500 veh/hr en toda la ciudad.
# Fracción en el cuadrante céntrico (Mackenna/Los Carrera): ~40%
# Vehículos Hora Punta Centro: ~2,200 veh/hr

# Conversión a parámetro -p (período en segundos = 3600 / veh_hr)
niveles = {
    "bajo": 3600 / 800,     # 800 veh/hr (Valle) -> p=4.5
    "medio": 3600 / 1500,   # 1500 veh/hr (Transición) -> p=2.4
    "alto": 3600 / 2200,    # 2200 veh/hr (Punta Centro) -> p=1.6
    "extremo": 3600 / 3000  # 3000 veh/hr (Lluvia/Caos) -> p=1.2
}

os.system("mkdir -p sumo")

for nivel, p in niveles.items():
    print(f"Generando tráfico {nivel.upper()} (p={p:.2f}s, aprox {3600/p:.0f} veh/h)...")
    
    # 1. Generar viajes aleatorios pero con franjas coherentes de entrada/salida al centro
    os.system(f"python \"C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py\" "
              f"-n sumo/caso_estudio.net.xml "
              f"-p {p} "
              f"--seed 42 "
              f"-e 3600 "
              f"-o sumo/caso_estudio_{nivel}_42.trips.xml "
              f"--fringe-factor 5")
              
    # 2. Enrutar usando DUAROUTER
    os.system(f"duarouter -n sumo/caso_estudio.net.xml "
              f"-t sumo/caso_estudio_{nivel}_42.trips.xml "
              f"-o sumo/caso_estudio_{nivel}_42.rou.xml "
              f"--ignore-errors true "
              f"--no-warnings true")

print("Rutas realistas de Osorno Centro generadas correctamente.")
