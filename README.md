# Control de Tráfico con Reinforcement Learning en SUMO 🚦🧠

Este proyecto forma parte de una tesis orientada a la optimización del control de tráfico en la ciudad de Osorno utilizando métodos de Multi-Agent Reinforcement Learning (MARL) en el simulador SUMO.

El repositorio implementa una arquitectura robusta basada en **MaskablePPO con Parameter Sharing**, incorporando técnicas avanzadas de observación (OPW) y recompensas basadas en presión y alineamiento de vecindario (inspirado en CityLight / HAPS-PPO).

## 🚀 Características Principales

*   **Arquitectura MARL Parameter Sharing**: Todos los agentes (semáforos) comparten la misma red neuronal mediante `MaskablePPO` (evitando elegir fases inválidas) y `VecNormalize` (estabilizando las recompensas).
*   **MDP v2 (Markov Decision Process)**:
    *   **Observación (OPW v2)**: *Observation Padding Wrapper* que estandariza intersecciones heterogéneas (diferente número de carriles y fases) en un vector de tamaño fijo con máscaras de validez y codificación *one-hot* topológica.
    *   **Recompensa Modular (CityLight)**: Combina penalización por colas, presión local (MaxPressure) y *neighborhood blending* (presión media de los vecinos) para fomentar la cooperación regional y evitar el *gridlock*.
*   **Control Determinista de Fases**: Los agentes controlan completamente los semáforos a través de máquinas de estado dedicadas (`TrafficSignal`), desactivando el programa estático de SUMO para evitar interferencias. Respetan restricciones físicas duras como tiempos mínimos/máximos de verde y transiciones amarillas.
*   **Baselines Integrados**: Evaluación comparativa rigurosa contra políticas clásicas (Tiempo Fijo estático, MaxPressure, Random) a través de múltiples semillas y niveles de demanda.

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
Entrena el modelo usando Parameter Sharing (MaskablePPO):
```bash
python scripts/train.py --timesteps 1000000
```
*(Opcional: añade `--gui` para visualizar la simulación en tiempo real).*

Puedes monitorear el entrenamiento en vivo:
```bash
tensorboard --logdir tensorboard
```

### 3. Evaluación (RL vs Baselines)
Evalúa las políticas base o los modelos entrenados a lo largo de todas las demandas y semillas, generando una tabla comparativa y exportando a CSV:

```bash
# Evaluar todas las políticas base (Tiempo Fijo, MaxPressure, Random)
python scripts/evaluate.py --all

# Evaluar solo MaxPressure
python scripts/evaluate.py --policy max_pressure

# (Próximamente) Evaluar modelo RL:
# python scripts/evaluate.py --policy rl --model models/maskable_ppo_final
```

## ⚙️ Configuración (`config/default.yaml`)

Toda la lógica de entrenamiento y recompensas se maneja a través de un YAML.
*   `simulation`: Archivos de red, pasos de simulación, $g_{min}$, $g_{max}$, tiempos amarillos.
*   `training`: Hiperparámetros de MaskablePPO (batch size, learning rate, gamma=0.99, ent_coef).
*   `reward`: Componentes de recompensa ajustables independientemente (congestión, presión, vecindario).
*   `marl`: Radio de comunicación para observación de vecinos.

## 🏗️ Estructura del Proyecto

```text
├── config/             # Configuración centralizada YAML
├── scripts/            # Scripts ejecutables (train.py, evaluate.py, etc.)
├── src/                
│   ├── agents/         # Implementación MARL (Parameter Sharing)
│   ├── baselines/      # Políticas base: FixedTime, MaxPressure, Random
│   ├── environment/    # MultiAgentTrafficEnv, TrafficSignal, OPW
│   ├── rewards/        # Sistema modular de recompensa (Congestion, Pressure, etc.)
│   └── utils/          # Grafo de red (NetworkGraph), parseadores, configuración
├── sumo/               # Archivos nativos de la simulación (.net.xml, .rou.xml, etc.)
└── tests/              # Batería de pruebas automatizadas (pytest)
```

## 🧪 Pruebas Automatizadas (Testing)

El repositorio cuenta con una extensa suite de tests unitarios (56 tests) que verifican desde las matemáticas del MDP v2 y observación heterogénea (OPW), hasta las restricciones estrictas de cambio de semáforos.

Para ejecutar la batería de pruebas:
```bash
pytest tests/ -v
```
