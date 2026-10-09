# 🧠 Guía de Entrenamiento RL — Control de Tráfico Osorno

## Requisitos Previos

Antes de ejecutar cualquier comando, abre una **terminal PowerShell en VSCode** (`Ctrl+ñ`) y navega a la carpeta del proyecto:

```powershell
cd "c:\Users\Ress\Desktop\MT"
```

---

## Variables de Entorno (copiar y pegar antes de cada sesión)

```powershell
$env:SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
$env:PYTHONIOENCODING="utf-8"
```

---

## Paso 0: Verificar que todo funcione

```powershell
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Se esperan **52 tests** pasando.

---

## Paso 1: Generar Tráfico (solo si cambiaste el mapa)

```powershell
.\venv\Scripts\python.exe scripts/generate_traffic.py
```

Genera archivos de rutas para 3 niveles de demanda (bajo, medio, alto) × múltiples semillas en `sumo/`.

---

## Paso 2: Ejecutar Baselines (Evaluación de referencia)

Evalúa todas las políticas base (Tiempo Fijo, MaxPressure, Random) en todas las demandas y semillas:

```powershell
.\venv\Scripts\python.exe scripts/evaluate.py --policy all
```

Resultados en: `results/evaluation.csv`

---

## Paso 3: Entrenamiento MaskablePPO por Fases (Recomendado)

Para monitorear el aprendizaje y guardar checkpoints frecuentes, usa el entrenamiento por fases. Cada fase guarda un modelo independiente.

**Entrenamiento corto (prueba):**
```powershell
.\venv\Scripts\python.exe scripts/train_phased.py --total-steps 500000 --phases 10
```

**Entrenamiento largo (noche):**
```powershell
.\venv\Scripts\python.exe scripts/train_phased.py --total-steps 1500000 --phases 15
```

| Fase | Pasos acumulados | Modelo guardado |
|------|-----------------|-----------------|
| 1    | 50,000          | `models/parameter_sharing/model_phase_1.zip` |
| ...  | ...             | ... |
| 10   | 500,000         | `models/parameter_sharing/model_phase_10.zip` |

**Alternativa — Entrenamiento directo (sin fases):**
```powershell
.\venv\Scripts\python.exe scripts/train.py --timesteps 1000000
```

---

## Paso 4: Fine-Tuning (Perfeccionamiento)

Si quieres perfeccionar un modelo existente con exploración reducida:

```powershell
.\venv\Scripts\python.exe scripts/finetune.py --model-path models/parameter_sharing/model_phase_10 --total-steps 250000 --phases 5
```

Modelos en: `models/finetune/model_finetune_phase_X.zip`

---

## Paso 5: Evaluación RL vs Baselines

Evaluar el modelo entrenado contra todas las políticas base:

```powershell
.\venv\Scripts\python.exe scripts/evaluate.py --policy all --model_path models/parameter_sharing/model_phase_10.zip
```

Para evaluar solo el modelo RL:
```powershell
.\venv\Scripts\python.exe scripts/evaluate.py --policy rl --model_path models/parameter_sharing/model_phase_10.zip
```

---

## Paso 6: Visualización en SUMO (Ver el agente en acción)

Para ver a los semáforos actuando en tiempo real con la interfaz gráfica de SUMO:

```powershell
.\venv\Scripts\python.exe scripts/evaluate.py --policy rl --model_path models/parameter_sharing/model_phase_10.zip --gui
```

⚠️ **Importante:**
1. Al abrir la interfaz de SUMO, ajusta el "Delay (ms)" en la parte superior (ej. 50-100ms) para que los autos no se muevan tan rápido que no puedas verlos.
2. Si prefieres evaluar *sin* la interfaz visual (para obtener las métricas rápidamente), quita el flag `--gui`.

---

## Paso 7: Generar Reporte PDF y Gráficas

```powershell
.\venv\Scripts\python.exe scripts/compare_results.py
.\venv\Scripts\python.exe scripts/generate_report.py
```

PDF generado en: `results/reporte_resultados.pdf`

---

## Paso 8: Ver Curvas de Entrenamiento en TensorBoard

```powershell
.\venv\Scripts\python.exe -m tensorboard.main --logdir tensorboard
```

Luego abre en tu navegador: `http://localhost:6006`

---

## 📌 Notas Técnicas

- **Algoritmo:** MaskablePPO con Parameter Sharing (no es MAPPO). Un único PPO con action masking compartido entre todos los agentes.
- **Recompensa activa por defecto:** `EcoDelayReward` — equilibra delay y emisiones CO₂. Se configura en `config/default.yaml`.
- **Multiplicador de pasos:** El `ParameterSharingVecEnv` multiplica los pasos de SUMO × número de agentes para PPO. `train_phased.py` compensa automáticamente.
