param([int]$pidToWait)
Wait-Process -Id $pidToWait
$env:PYTHONIOENCODING="utf-8"
venv\Scripts\python.exe scripts/train.py --config config/caso_estudio_sc.yaml --timesteps 10000000
