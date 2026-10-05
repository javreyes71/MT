# 🧠 Guía de Entrenamiento RL — Control de Tráfico Osorno

## Requisitos Previos

Antes de ejecutar cualquier comando, abre una **terminal PowerShell en VSCode** (`Ctrl+ñ`) y navega a la carpeta del proyecto:

```powershell
cd "c:\Users\javie\Desktop\MT\CODE\Control-de-Trafico-con-RL-en-SUMO"
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

---

## Paso 1: Generar Tráfico (solo si cambiaste el mapa)

```powershell
.\venv\Scripts\python.exe scripts/generate_traffic.py
```

---

## Paso 2: Ejecutar Baseline (Control Cíclico Tradicional)

Genera los datos de referencia (3 episodios de semáforos con tiempos fijos):

```powershell
.\venv\Scripts\python.exe scripts/baseline.py --episodes 3
```

Resultados en: `results/baseline/`

---

## Paso 3: Entrenamiento MAPPO por Fases (Entrenamiento Corto/Recomendado)

Para monitorear mejor el aprendizaje y guardar puntos de control (`checkpoints`) con mayor frecuencia, configuramos 500,000 pasos divididos en 10 fases (cada fase guarda un modelo cada ~15-30 minutos).

```powershell
.\venv\Scripts\python.exe scripts/train_phased.py --mode parameter_sharing --total-steps 500000 --phases 10
```

| Fase | Pasos acumulados | Modelo guardado |
|------|-----------------|-----------------|
| 1    | 50,000          | `models/parameter_sharing/model_phase_1.zip` |
| ...  | ...             | ... |
| 10   | 500,000         | `models/parameter_sharing/model_phase_10.zip` |

**Nota:** Si deseas un entrenamiento de noche (largo), puedes usar `--total-steps 1500000 --phases 15`.

---

## Paso 4: Fine-Tuning (Perfeccionamiento)

Si quieres perfeccionar un modelo existente con exploración reducida:

```powershell
.\venv\Scripts\python.exe scripts/finetune.py --model-path models/parameter_sharing/model_phase_10 --total-steps 250000 --phases 5
```

Modelos en: `models/finetune/model_finetune_phase_X.zip`

---

## Paso 5: Visualización Gráfica (Ver el aprendizaje en vivo)

Para ver a los semáforos actuando en tiempo real con la interfaz gráfica de SUMO:

```powershell
.\venv\Scripts\python.exe scripts/evaluate.py --mode parameter_sharing --model-path models/parameter_sharing/model_phase_10 --gui
```

⚠️ **Importante:**
1. Al abrir la interfaz de SUMO, ajusta el "Delay (ms)" en la parte superior (ej. 50-100ms) para que los autos no se muevan tan rápido que no puedas verlos.
2. Si prefieres evaluar *sin* la interfaz visual (para obtener las métricas rápidamente), quita el flag `--gui`.

---

## Paso 6: Generar Reporte PDF y Gráficas

```powershell
.\venv\Scripts\python.exe scripts/compare_results.py
.\venv\Scripts\python.exe scripts/generate_report.py
```

PDF generado en: `results/reporte_resultados.pdf`

---

## Paso 7: Ver Curvas de Entrenamiento en TensorBoard

```powershell
.\venv\Scripts\python.exe -m tensorboard.main --logdir tensorboard
```

Luego abre en tu navegador: `http://localhost:6006`

---

