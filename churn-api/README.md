# Customer Churn Prediction — End-to-End Machine Learning Project
## Descripción general

Este proyecto desarrolla un modelo de Machine Learning para predecir el abandono de clientes (churn) en una empresa de telecomunicaciones, utilizando información histórica sobre contratos, servicios, pagos y comportamiento del cliente.

El objetivo es anticiparse a la pérdida de clientes y permitir que la empresa tome acciones preventivas (retención, descuentos, soporte personalizado).

## Objetivo del proyecto

Predecir si un cliente abandonará la empresa (Churn = 1) o permanecerá (Churn = 0), a partir de variables históricas.

En términos de negocio:

Retener clientes es entre 5 y 7 veces más barato que adquirir nuevos.
Este modelo ayuda a reducir pérdidas económicas identificando clientes de alto riesgo con anticipación.

## Enfoque de Ciencia de Datos

El proyecto sigue un flujo end-to-end típico de un entorno empresarial:

Exploración y comprensión del dataset

Limpieza y validación de datos

Análisis exploratorio (EDA)

Preparación de datos (scaling + encoding)

Entrenamiento de modelos de clasificación

Evaluación con métricas adecuadas para churn

Selección del modelo final

Simulación de predicción en un cliente real

Exportación del modelo para uso en producción

## 📂 Dataset

Fuente: Telco Customer Churn Dataset

Observaciones: clientes individuales

Variable objetivo: Churn (0 = No abandona, 1 = Abandona)

Tipos de variables:

Demográficas

Tipo de contrato

Servicios contratados

Información de pagos

Antigüedad (tenure)

## Limpieza y preparación de datos

Conversión de TotalCharges a tipo numérico

Eliminación de valores faltantes generados por la conversión

Eliminación de columnas no predictivas (customerID)

Separación de variables:

Numéricas → StandardScaler

Categóricas → OneHotEncoder

Uso de Pipeline + ColumnTransformer para evitar data leakage

## Análisis Exploratorio (EDA)

Se analizaron relaciones clave entre churn y variables relevantes:

Distribución de churn

Churn vs tipo de contrato

Churn vs antigüedad (tenure)

Churn vs cargos mensuales

Churn vs servicios adicionales

Churn vs método de pago

Hallazgos principales:

Clientes con contratos mes a mes presentan mayor churn

Menor antigüedad se asocia con mayor abandono

Cargos mensuales más altos tienden a aumentar el riesgo

La ausencia de servicios como TechSupport y OnlineSecurity incrementa el churn

## Modelos de Machine Learning

Se entrenaron y compararon dos modelos:

## Logistic Regression (Baseline)

Modelo interpretable

Sirve como referencia inicial

Útil para entender relaciones lineales

## Random Forest Classifier

Captura relaciones no lineales

Mejor desempeño en detección de churn

Mayor capacidad predictiva

Ambos modelos se entrenaron usando:

Train/Test split (80/20)

Stratify en la variable objetivo para manejar desbalance

## Métricas de evaluación

Dado que el churn es un problema sensible al costo del error, se priorizaron métricas más allá del accuracy:

Recall (Churn = 1) → métrica clave de negocio

Precision

F1-score

ROC-AUC

Matriz de confusión

Curva ROC comparativa

📌 En churn, un falso negativo (cliente que se va y no se detecta) es más costoso que un falso positivo.

## Ajuste de umbral (Threshold)

Se implementó un threshold personalizado (0.35) para:

Aumentar el recall

Detectar más clientes en riesgo

Alinear el modelo con objetivos de negocio

## Ejemplo de uso (predicción real)

El proyecto incluye un ejemplo de inferencia con un cliente nuevo:

Se calcula la probabilidad de churn

Se genera una predicción binaria

Se aplica un threshold ajustable

Esto simula un escenario real de uso empresarial.

## Exportación del modelo

El modelo final se guarda usando joblib:

churn_model.pkl

Esto permite:

Integrarlo en una API

Usarlo en dashboards

Ejecutarlo periódicamente en producción

## API con FastAPI

El modelo final se exportó y se consumió mediante una API REST.

Endpoint principal

POST /predict

Ejemplo de request:
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "Yes",
  "tenure": 60,
  "PhoneService": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "Yes",
  "DeviceProtection": "Yes",
  "TechSupport": "Yes",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Two year",
  "PaperlessBilling": "No",
  "PaymentMethod": "Bank transfer (automatic)",
  "MonthlyCharges": 55.0,
  "TotalCharges": 3300.0
}

Ejemplo de response:
{
  "churn_probability": 0.006,
  "churn_prediction": 0,
  "threshold": 0.5
}

Tecnologías usadas

Python

pandas, numpy

scikit-learn

matplotlib, seaborn

FastAPI

Uvicorn

joblib

## Aplicación en un entorno real

Flujo típico de negocio:

Entrenamiento con datos históricos

Evaluación mensual de clientes activos

Identificación de clientes de alto riesgo

Acciones de retención:

Descuentos

Contacto telefónico

Mejora de servicios

Ofertas personalizadas

## Limitaciones y trabajo futuro

Implementar validación temporal

Ajustar class_weight o técnicas de balanceo

Análisis de interpretabilidad (SHAP / feature importance)

Monitoreo de performance y data drift


## Autor

Proyecto desarrollado por Fernando Gómez

⭐ Conclusión

Este proyecto demuestra un enfoque end-to-end de ciencia de datos aplicado a un problema empresarial real, integrando análisis exploratorio, modelado, métricas correctas y una visión clara de negocio.