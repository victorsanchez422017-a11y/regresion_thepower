# Predicción del rendimiento de estudiantes: regresión y clasificación

Proyecto de Machine Learning (The Power, módulo 11). A partir de datos de 1000 estudiantes,
se construyen dos modelos: uno de **regresión** que predice la `nota_final` y otro de
**clasificación** que predice si el alumno **aprueba** (`nota_final >= 60`).

## Estructura

```
├── data/dataset_estudiantes.csv
├── notebooks/proyecto.ipynb
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/proyecto.ipynb
```

## 1. Análisis exploratorio

- 1000 alumnos, 9 variables predictoras, sin duplicados.
- **Nulos:** `horas_sueno` (15 %), `horario_estudio_preferido` (10 %) y `estilo_aprendizaje` (5 %).
- **Fuga de datos:** `aprobado` es exactamente `nota_final >= 60`. Por eso ninguno de los dos
  targets se usa como predictor del otro.
- **Desbalance:** 89,8 % aprobados y 10,2 % suspensos.
- Los valores en los topes (asistencia 100 %, nota 100...) son censura natural, no outliers, y se mantienen.
- Predictores más correlacionados con la nota: horas de estudio (0,51), nota anterior (0,47)
  y asistencia (0,32). `edad` y `horas_sueno` casi no influyen.

## 2. Preprocesamiento

- Split 80/20 estratificado por `aprobado` (800 train / 200 test), único para ambos modelos.
- Numéricas: imputación con la mediana y estandarización.
- `nivel_dificultad`: codificación ordinal (Fácil < Medio < Difícil).
- `tiene_tutor`: binaria.
- `horario_estudio_preferido` y `estilo_aprendizaje`: nulos como categoría "Desconocido" y one-hot.
- Todo dentro de un `Pipeline` para que la imputación y el escalado se ajusten solo con train
  (sin fuga de datos) y se repitan en cada fold de la validación cruzada.

## 3. Regresión (`nota_final`)

| Modelo | MAE (CV) | R² (CV) |
|---|---|---|
| Línea base (media) | 7,61 | 0 |
| **Regresión lineal** | **6,04** | **0,365** |
| Random Forest | 6,33 | 0,314 |

**Modelo elegido: regresión lineal.** Es más simple y rinde mejor: la relación es básicamente lineal.

**Test:** MAE 6,12, RMSE 7,45, R² 0,395. Generaliza bien, y los residuos no muestran patrón.

**Hallazgos:**
- Las variables con más peso son las horas de estudio, la nota anterior y la asistencia.
- Cada nivel de dificultad resta ~1,4 puntos.
- Los alumnos con tutor sacan 6,5 puntos más de media, pero casi todo se explica por otras variables:
  estudian 13,6 h frente a 7,7 h y tenían mejor nota anterior. A igualdad de lo demás, el efecto
  del tutor es de ~0,8 puntos.

## 4. Clasificación (`aprobado`)

La accuracy no sirve como métrica: un modelo que siempre dice "aprueba" acierta el 90 %. Se evalúa
la clase **suspenso** con recall, F1 y balanced accuracy.

| Modelo (CV) | Balanced acc. | Recall suspensos | F1 suspensos |
|---|---|---|---|
| Línea base | 0,50 | 0 | 0 |
| **Regresión logística** | **0,725** | **0,693** | 0,362 |
| Random Forest | 0,636 | 0,315 | 0,371 |

**Modelo elegido: regresión logística** (`class_weight="balanced"`). Se prioriza el recall: es preferible
avisar a un alumno de más que dejar sin avisar a uno que va a suspender.

**Test (20 suspensos y 180 aprobados):** recall de suspensos 0,65, precisión 0,20, ROC-AUC 0,815.
Detecta 13 de los 20 suspensos, con 51 falsas alarmas.

## Conclusiones y limitaciones

- Los dos modelos mejoran claramente a sus líneas base, y en ambos gana el modelo simple.
- El modelo de regresión sirve para estimar el nivel general del alumno (error medio de ~6 puntos),
  no su nota exacta. El R² de ~0,4 indica que gran parte de la nota depende de factores que el dataset no recoge.
- El clasificador sirve como **filtro para priorizar a quién revisar**, no como veredicto: la precisión es baja.
- El test solo tiene 20 suspensos, así que sus métricas son poco estables.
- **Mejoras posibles:** ajustar el umbral de decisión con validación cruzada, probar modelos de boosting
  y recoger más variables.
