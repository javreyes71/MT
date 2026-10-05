# Control de Tráfico con Reinforcement Learning en SUMO 🚦🧠

Este proyecto forma parte de una tesis universitaria orientada a la optimización del control de tráfico utilizando métodos de Aprendizaje por Refuerzo (RL) y Multi-Agent Reinforcement Learning (MARL) en el simulador SUMO.

## Arquitectura del Sistema

```text
[ SUMO (Simulador de Tráfico) ]
           ^   |
 (Acciones)|   |(Observaciones & Recompensas)
           |   v
[ TrafficEnv / MultiAgentTrafficEnv ]
           ^   |
           |   | (Estados, Recompensas)
           |   v
[ Agentes de RL (SB3 / PyTorch) ]
   - Centralizado
   - Independiente (MARL)
   - Parameter Sharing (MARL)
```

## Prerrequisitos
- Python 3.10+
- SUMO (Simulation of Urban MObility) 1.18+
- Docker (opcional)

## Instalación

### Opción 1: Instalación Local
1. Clona el repositorio
2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Asegúrate de tener la variable de entorno `SUMO_HOME` configurada.

### Opción 2: Docker
El proyecto cuenta con un `Dockerfile` y un `docker-compose.yml` para facilitar la ejecución de pruebas y entrenamientos de manera aislada.

## Uso

### Entrenamiento
El script de entrenamiento soporta tres modos de agentes. Puedes correrlos mediante Docker Compose o de manera local:

```bash
# Modo Centralizado
python scripts/train.py --mode centralized --timesteps 500000

# Modo Independiente
python scripts/train.py --mode independent --timesteps 500000

# Parameter Sharing
python scripts/train.py --mode parameter_sharing --timesteps 500000
```
*(Añade la flag `--gui` si quieres ver el entrenamiento en SUMO-GUI).*

### Evaluación y Línea Base
Puedes probar una línea base sin aprendizaje por refuerzo:
```bash
python scripts/baseline.py --episodes 10 --gui
```

Para evaluar un modelo entrenado:
```bash
python scripts/evaluate.py --mode centralized --model models/centralized/final.zip --episodes 5
```

### Visualización
```bash
python scripts/visualize.py --mode centralized --model models/centralized/final.zip
```

### Generación de Tráfico
```bash
python scripts/generate_traffic.py --emergency
```

## Configuración
La configuración se maneja principalmente mediante archivos YAML en `config/`.
Ejemplo:
- `simulation`: Parámetros de SUMO, archivos de red y tiempos.
- `training`: Algoritmo a usar (PPO por defecto), timesteps, semillas.
- `marl`: Modo multagente (`centralized`, `independent`, `parameter_sharing`).
- `paths`: Carpetas de salida (`models/`, `results/`).

## Estructura del Proyecto

```text
Control-de-Trafico-con-RL-en-SUMO/
├── config/             # Archivos YAML de configuración
├── models/             # Modelos entrenados guardados (Ignorado en git)
├── results/            # Métricas y CSVs (Ignorado en git)
├── scripts/            # Scripts de ejecución (train, eval, baseline, etc.)
├── src/                # Código fuente principal
│   ├── agents/         # Implementaciones de agentes (centralizado, MARL)
│   ├── callbacks/      # Callbacks de SB3
│   ├── environment/    # Entornos y wrappers (TrafficSumoEnv)
│   ├── rewards/        # Lógica modular de recompensas
│   └── utils/          # Utilidades (config, seed, metrics)
├── sumo/               # Archivos nativos de SUMO (.net.xml, .rou.xml, etc.)
├── tensorboard/        # Logs de Tensorboard
├── tests/              # Pruebas automatizadas (pytest)
├── Dockerfile          # Configuración para la imagen Docker
├── docker-compose.yml  # Servicios para correr diferentes experimentos
├── requirements.txt    # Dependencias de Python
└── README.md           # Este archivo
```

## Ejecución de Pruebas Automáticas (Tests)
Utilizamos `pytest` para las pruebas unitarias:
```bash
pytest tests/
```
Esto correrá las pruebas para observaciones, sistema modular de recompensas y vehículos de emergencia.
