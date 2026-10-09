# 🧠 CEREBRO DE CONTEXTO — PROYECTO DE TRÁFICO MARL
> **Instrucción para LLMs:** Este documento es la "fuente de la verdad" del proyecto. Úsalo para mantener el contexto gastando la menor cantidad de tokens posible. No asumas características fuera de este stack.

## 🎯 Objetivo Principal
Optimizar el control de semáforos en la ciudad de Osorno superando a los métodos tradicionales (Tiempo Fijo, MaxPressure).
**Métricas clave a minimizar (Prioridades):**
1. Tiempos de espera (Delay / Queue).
2. Emisiones de CO2.
3. Consumo de combustible.

## 🛠️ Stack Tecnológico
* **Simulador:** SUMO (Simulation of Urban MObility) vía `libsumo` (C++ directo, no TCP).
* **Lenguaje & Librerías:** Python 3.10+, Gymnasium, Stable-Baselines3 + SB3-Contrib (`MaskablePPO`).

## 📐 Arquitectura Multi-Agente (MARL)
* **Algoritmo:** `MaskablePPO` con **Parameter Sharing** (CTDE). No es MAPPO — es un único PPO con action masking compartido entre todos los agentes.
* **Topología:** Cada semáforo (intersección) con ≥2 fases verdes es un agente.
* **Red Neuronal:** Todos los agentes comparten **una única red `MlpPolicy`** central.
* **VecEnv:** `ParameterSharingVecEnv` convierte el entorno multi-agente en un `VecEnv` de SB3 donde cada sub-entorno es un agente. `VecNormalize` estabiliza recompensas (`norm_reward=True`, `norm_obs=False`).

## 🧩 Diseño del MDP (Markov Decision Process)
### 1. Espacio de Observación (OPW v2)
* **Problema resuelto:** Intersecciones heterogéneas (diferente número de carriles y fases).
* **Solución:** `ObservationPaddingWrapper` (OPW). Estandariza la observación en un vector de tamaño fijo:
  - `halts[max_lanes]` — cola normalizada por capacidad [0,1]
  - `in_occ[max_lanes]` — ocupación entrante [0,1]
  - `avg_out_occ` — ocupación saliente media [0,1]
  - `phase_norm` — fase actual normalizada [0,1]
  - `can_act_flag` — indicador de si puede cambiar {0,1}
  - `time_norm` — tiempo en fase / g_max [0,1]
  - `local_pressure` — presión local [-1,1]
  - `neighbor_pressure_avg` — presión media vecinos [0,1]
  - `neighbor_queue_avg` — cola media vecinos [0,1]
  - `type_onehot[n_groups]` — one-hot del grupo de intersección (DMSGL)
  - `lane_mask[max_lanes]` — máscara de carriles válidos {0,1}
  - `phase_mask[max_phases]` — máscara de fases válidas {0,1}

### 2. Espacio de Acción
* **Control:** Determinista por fases. El agente elige qué fase verde activar.
* **Restricciones Duras:** Manejadas por `TrafficSignal` (no por la IA). Transiciones a amarillo automáticas, $g_{min}=15$s, $g_{max}=35$s, amarillo fijo $3$s.
* **Action Masking:** Si `can_act() == False` (amarillo o $\tau < g_{min}$), la única acción válida es mantener la fase actual. Todas las fases paddeadas (> `num_green_phases`) están enmascaradas.
* **Anti-starvation:** `must_switch()` fuerza rotación si se excede $g_{max}$.

### 3. Recompensa (Estado Actual)
* **Componente activo:** `EcoDelayReward` — combina penalización por colas (delay) y emisiones de CO₂, normalizado por capacidad del semáforo. Es el **único componente con desagregación real por agente**.
  ```
  r_i = -(w_delay · halts_i/cap_i + w_eco · co2_i/(cap_i · 2000))
  ```
* **Componentes disponibles pero deshabilitados:** `CongestionPenalty`, `PressureReward`, `NeighborhoodPressureReward`, `StabilityPenalty`, `CO2Penalty`.
* **⚠️ Limitación:** Los componentes deshabilitados solo tienen implementación global (`calculate()`), no por agente. Activarlos daría la misma recompensa a todos los agentes.

## 📂 Estructura del Repositorio
```text
├── config/              # Configuración centralizada YAML (default.yaml)
├── scripts/             # Scripts ejecutables
│   ├── train.py         # ✅ Entrenamiento MaskablePPO (funcional)
│   ├── evaluate.py      # ✅ Evaluación multi-semilla (funcional)
│   ├── train_phased.py  # ⚠️ Imports rotos (CentralizedAgent, IndependentAgent)
│   ├── finetune.py      # ⚠️ Import roto (SharedPolicyWrapper) + usa PPO en vez de MaskablePPO
│   ├── baseline.py      # Evaluación de baselines (legacy)
│   ├── generate_traffic.py  # Generación de rutas por demanda/semilla
│   └── ...              # build_network, compare_results, generate_report, etc.
├── src/
│   ├── agents/          # ParameterSharingAgent + ParameterSharingVecEnv
│   ├── baselines/       # FixedTimePolicy, MaxPressurePolicy, RandomPolicy
│   ├── callbacks/       # MetricsCallback (logging a TensorBoard + CSV)
│   ├── environment/     # TrafficSumoEnv, MultiAgentTrafficEnv, TrafficSignal, OPW
│   ├── rewards/         # Sistema modular: eco_delay, congestion, pressure, etc.
│   └── utils/           # NetworkGraph (BFS), DMSGL grouper, MetricsCollector, config
├── sumo/                # Archivos nativos (.net.xml, .rou.xml, .sumocfg)
├── tests/               # Tests unitarios (pytest)
└── tensorboard/         # Logs de entrenamiento
```

## ⚠️ Reglas de Desarrollo
1. Toda nueva función de recompensa debe ir en `src/rewards/` e implementar **tanto** `calculate()` como `calculate_agent()`.
2. Los valores de la red se estabilizan con `VecNormalize(norm_reward=True)`.
3. Todo control físico del semáforo debe respetar la máquina de estados en `TrafficSignal`.
4. No usar `print()` — usar `logging.getLogger(__name__)`.
5. Verificar que los scripts importan módulos existentes antes de documentarlos como funcionales.

## 🐛 Issues Conocidos
- `train_phased.py` y `finetune.py` no funcionan (imports rotos).
- Las recompensas `congestion`, `pressure`, `neighborhood`, `stability` no tienen `calculate_agent()` individual.
- La red SUMO se lee dos veces (en `TrafficSumoEnv` y en `NetworkGraph`).
- Las dataclasses en `config.py` son código muerto — nunca se instancian.
