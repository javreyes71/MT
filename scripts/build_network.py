"""
Script para preprocesar y limpiar mapas de OpenStreetMap (.osm) 
transformándolos en redes de SUMO (.net.xml) optimizadas para Inteligencia Artificial.

Este script aplica heurísticas agresivas para eliminar el 'ruido morfológico'
(nodos fragmentados, bordes internos minúsculos) que causan gridlocks artificiales.
"""

import argparse
import subprocess
import os
import sys

def build_network(osm_file: str, output_file: str, join_dist: float):
    if not os.path.exists(osm_file):
        print(f"❌ Error: El archivo {osm_file} no existe.")
        sys.exit(1)

    print(f"🏗️  Compilando y limpiando red desde: {osm_file}")
    print(f"🧹 Usando distancia agresiva de fusión de intersecciones: {join_dist}m")

    # Comando de netconvert con parámetros de limpieza de topología urbana
    cmd = [
        "netconvert",
        "--osm-files", osm_file,
        "-o", output_file,
        
        # --- LIMPIEZA DE INTERSECCIONES (El cast morfológico) ---
        "--junctions.join", "true",           # Une intersecciones complejas
        "--junctions.join-dist", str(join_dist), # Distancia agresiva para fusionar (ej. rotondas o cruces deformes)
        "--edges.join", "true",               # Une calles que fueron fragmentadas en OSM
        "--geometry.remove", "true",          # Simplifica la geometría
        "--remove-edges.isolated", "true",    # Elimina calles que no conectan con nada
        "--roundabouts.guess", "true",        # Infiere rotondas reales
        "--ramps.guess", "true",              # Infiere rampas de aceleración
        "--output.street-names", "true",      # Mantiene los nombres de las calles para filtrar corredores
        "--keep-edges.by-vclass", "passenger",# Solo mantiene calles de vehículos ligeros (excluye trenes/peatones)
        
        # --- CONFIGURACIÓN DE SEMÁFOROS (Optimizado para RL) ---
        "--tls.guess-signals", "true",        # Adivina dónde deben ir los semáforos
        "--tls.default-type", "static",       # Genera fases base que la IA luego pisará
        "--tls.join", "true",                 # Fusiona múltiples semáforos de un cruce en UN solo controlador
        "--tls.green.time", "30"              # Tiempo verde de respaldo por defecto
    ]

    try:
        # Añadir SUMO_HOME/bin al path temporal si existe
        env = os.environ.copy()
        if 'SUMO_HOME' in env:
            env['PATH'] += os.pathsep + os.path.join(env['SUMO_HOME'], 'bin')
            
        # Ejecutamos netconvert
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        print(f"✅ Red compilada exitosamente en: {output_file}")
        
        # Mostrar algunas advertencias relevantes si las hay
        warnings = [line for line in result.stderr.split('\n') if 'Warning' in line]
        if warnings:
            print(f"⚠️  Se manejaron {len(warnings)} advertencias morfológicas durante la conversión.")
            
    except subprocess.CalledProcessError as e:
        print("❌ Error crítico al compilar la red.")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: No se encontró el comando 'netconvert'. Asegúrate de que SUMO_HOME/bin esté en tu PATH o ejecuta desde el SumoShel.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte un archivo OSM a una red limpia de SUMO.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Ruta al archivo .osm de entrada (ej: sumo/micro-map.osm)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Ruta al archivo .net.xml de salida (ej: sumo/network.net.xml)")
    parser.add_argument("--join-dist", "-d", type=float, default=20.0, help="Distancia en metros para fusionar nodos conflictivos (default: 20.0)")
    
    args = parser.parse_args()
    build_network(args.input, args.output, args.join_dist)
