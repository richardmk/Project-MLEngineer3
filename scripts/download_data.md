# Descarga del dataset

El CSV crudo (~470 MB) supera el límite de 100 MB de GitHub, por eso no se versiona.

1. Descargar desde Kaggle: [Synthetic Financial Datasets For Fraud Detection (PaySim)](https://www.kaggle.com/datasets/ealaxi/paysim1)
   - Con la CLI de Kaggle (requiere `~/.kaggle/kaggle.json`):
     ```bash
     kaggle datasets download -d ealaxi/paysim1 -p data/raw --unzip
     ```
2. Verificar que el archivo quede en:
   ```
   data/raw/PS_20174392719_1491204439457_log.csv
   ```
3. Ejecutar los notebooks en orden desde `notebooks/01_preprocesamiento.ipynb`.

Las muestras livianas versionadas en `data/*.csv` permiten correr la app de monitoreo sin el CSV crudo.
