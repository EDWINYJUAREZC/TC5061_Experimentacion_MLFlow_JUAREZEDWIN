# Baseline reproducible y MLflow

## Archivos

- `train.py`: carga, validación, limpieza, partición, entrenamiento, evaluación y registro de una corrida.
- `run_five_runs.py`: lanza cinco configuraciones con la misma semilla y registra todas en `agricultura_iot_fan_baseline_6_sensores`.
- `TC5061_Notebook_Baseline_JuarezEdwin.ipynb`: carga, descripción y exploración visual; no entrena ni evalúa el modelo.
- `requirements.txt`: versiones fijadas para reproducir el entorno o con versiones msa actualizadas.

## Instalación y ejecución en Windows

Desde el aprograma de PowerShell:

1.- Accedemos a la carpeta donde se guarden los archivos, colocamos el siguiente codigo :
           cd "ruta de la carpeta donde se encuentran los archivos"  

2.- Se revisa la version de Python que se tiene:
           Python --version 

3.- Se crea el entorno y se instala las dependencias:
           py -(version de Python) -m --version
           .\.venv\Scripts\python.exe -m pip install --upgrade pip
           .\.venv\Scripts\python.exe -m pip install -r requirements.txt

4.- Se ejecutan las 5 corridas:
           .\.venv\Scripts\python.exe .\run_five_runs.py --data-path "..\work\iot_agri\IoTProcessed_Data.csv" --seed 42

5.- Se abre MLflow Tracking y compara las corridas:
           .\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --workers 1
    
    Deja esa ventana abierta. En tu navegador entra a:
http://127.0.0.1:5000

En la interfaz:
1. Abre el experimento agricultura_iot_fan_baseline_6_sensores.
2. Si aparecen corridas con estado fallido, filtra o selecciona únicamente las que indiquen FINISHED.
3. Marca las cinco corridas terminadas y selecciona Compare.
4. Compara las métricas accuracy y f1_macro, junto con los parámetros n_estimators, max_depth y min_samples_leaf.
5. Al terminar, vuelve a PowerShell y presiona Ctrl+C para detener la interfaz.
MLflow guarda los parámetros, métricas, el modelo y los artefactos de evaluación —matriz de confusión y archivo JSON—. También puedes revisar el resumen tabular en comparison_results.csv.

