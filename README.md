# Control de Tráfico con Reinforcement Learning en SUMO 🚦🧠

Este proyecto forma parte de una tesis orientada a la optimización del control de tráfico en la ciudad de Osorno utilizando métodos de Multi-Agent Reinforcement Learning (MARL) en el simulador SUMO.

El repositorio implementa una arquitectura robusta basada en **MaskablePPO con Parameter Sharing**, incorporando técnicas avanzadas de observación (OPW v2) y recompensas modulares con desagregación por agente.

## 🚀 Características Principales

*   **Arquitectura MARL Parameter Sharing**: Todos los agentes (semáforos) comparten la misma red neuronal mediante `MaskablePPO` (evitando elegir fases inválidas) y `VecNormalize` (estabilizando las recompensas).
*   **MDP v2 (Markov Decision Process)**:
    *   **Observación (OPW v2)**: *Observation Padding Wrapper* que estandariza intersecciones heterogéneas (diferente número de carriles y fases) en un vector de tamaño fijo con máscaras de validez, codificación *one-hot* topológica, y resumen de vecinos.
    *   **Recompensa Modular**: Sistema con 6 componentes independientes, todos con desagregación por agente:
        - `EcoDelayReward` — equilibra delay y emisiones CO₂ (activo por defecto)
        - `CongestionPenalty` — penaliza colas y tiempos de espera
        - `PressureReward` — minimiza desequilibrio entrante/saliente (MaxPressure)
        - `NeighborhoodPressureReward` — blending con presión de vecinos (CityLight)
        - `StabilityPenalty` — penaliza cambios de fase excesivos
        - `CO2Penalty` — penaliza emisiones totales
*   **Control Determinista de Fases**: Los agentes controlan completamente los semáforos a través de máquinas de estado dedicadas (`TrafficSignal`), desactivando el programa estático de SUMO para evitar interferencias. Respetan restricciones físicas duras como tiempos mínimos/máximos de verde y transiciones amarillas.
*   **Baselines Integrados**: Evaluación comparativa rigurosa contra políticas clásicas (Tiempo Fijo, MaxPressure, Random) a través de múltiples semillas y niveles de demanda.

## 📋 Prerrequisitos

*   Python 3.10+
*   [SUMO](https://eclipse.dev/sumo/) (Simulation of Urban MObility) instalado y configurado.
*   Variable de entorno `SUMO_HOME` apuntando al directorio de instalación de SUMO.

## 🛠️ Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/javreyes71/MT.git
   cd MT
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   # Activar en Windows:
   venv\Scripts\activate
   # Activar en Linux/Mac:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

## 🎮 Uso del Proyecto

### 1. Generación de Tráfico
Genera rutas para distintos niveles de demanda (bajo, medio, alto) y distintas semillas:
```bash
python scripts/generate_traffic.py
```

### 2. Entrenamiento (RL)

**Opción A — Entrenamiento simple:**
```bash
python scripts/train.py --timesteps 1000000
```
*(Opcional: añade `--gui` para visualizar la simulación en tiempo real).*

**Opción B — Entrenamiento por fases (recomendado):**
```bash
python scripts/train_phased.py --total-steps 500000 --phases 10
```
Guarda un checkpoint al final de cada fase en `models/parameter_sharing/`.

Puedes monitorear el entrenamiento en vivo:
```bash
tensorboard --logdir tensorboard
```

### 3. Fine-Tuning (Perfeccionamiento)
Carga un modelo preentrenado y lo refina con learning rate reducido:
```bash
python scripts/finetune.py --model-path models/parameter_sharing/model_phase_10 --total-steps 250000 --phases 5
```

### 4. Evaluación (RL vs Baselines)
Evalúa las políticas a lo largo de todas las demandas y semillas, generando una tabla comparativa y exportando a CSV:

```bash
# Evaluar todas las políticas (Tiempo Fijo, MaxPressure, Random + RL)
python scripts/evaluate.py --policy all --model_path models/parameter_sharing/model_phase_10.zip

# Evaluar solo MaxPressure
python scripts/evaluate.py --policy max_pressure

# Evaluar solo el modelo RL
python scripts/evaluate.py --policy rl --model_path models/parameter_sharing/model_phase_10.zip
```

## ⚙️ Configuración (`config/default.yaml`)

Toda la lógica de entrenamiento y recompensas se maneja a través de un YAML.
*   `simulation`: Archivos de red, pasos de simulación, $g_{min}$, $g_{max}$, tiempos amarillos.
*   `training`: Hiperparámetros de MaskablePPO (batch size, learning rate, gamma=0.99, ent_coef).
*   `reward`: Componentes de recompensa activables/desactivables independientemente.
*   `marl`: Radio de comunicación para observación de vecinos.

## 🏗️ Estructura del Proyecto

```text
├── config/             # Configuración centralizada YAML
├── scripts/            # Scripts ejecutables
│   ├── train.py        # Entrenamiento MaskablePPO directo
│   ├── train_phased.py # Entrenamiento por fases con checkpoints
│   ├── finetune.py     # Fine-tuning de modelo preentrenado
│   ├── evaluate.py     # Evaluación multi-semilla (baselines + RL)
│   ├── generate_traffic.py  # Generación de rutas por demanda/semilla
│   └── ...             # build_network, compare_results, generate_report, etc.
├── src/                
│   ├── agents/         # ParameterSharingAgent + ParameterSharingVecEnv
│   ├── baselines/      # Políticas base: FixedTime, MaxPressure, Random
│   ├── callbacks/      # MetricsCallback (TensorBoard + CSV)
│   ├── environment/    # MultiAgentTrafficEnv, TrafficSignal, OPW v2
│   ├── rewards/        # Sistema modular de recompensa (6 componentes)
│   └── utils/          # NetworkGraph (BFS), DMSGL grouper, MetricsCollector
├── sumo/               # Archivos nativos de la simulación (.net.xml, .rou.xml, etc.)
├── tests/              # Batería de pruebas automatizadas (pytest)
└── tensorboard/        # Logs de entrenamiento
```

## 🧪 Pruebas Automatizadas (Testing)

El repositorio cuenta con una extensa suite de tests unitarios que verifican desde las matemáticas del MDP v2 y observación heterogénea (OPW), hasta las restricciones estrictas de cambio de semáforos.

Para ejecutar la batería de pruebas:
```bash
pytest tests/ -v
```
