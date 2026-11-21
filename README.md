---
## title: Air Quality ML Predictor
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🌬️ Air Quality ML Predictor · MLOps & Forecasting Dashboard

Sistema *end-to-end* de predicción y pronóstico de calidad del aire (**PM2.5**) para 5 ciudades españolas (**Alicante, Madrid, Barcelona, Valencia y Sevilla**). Combina datos meteorológicos y ambientales en tiempo real, modelos de Machine Learning en Scikit-Learn, persistencia en PostgreSQL (Neon), ingesta automatizada con GitHub Actions y un dashboard interactivo desplegado en Hugging Face Spaces.

## ✨ Características Principales

* **Inferencia Instantánea (**$t_0$**):** Predicción en vivo de concentración de partículas finas `PM2.5` ($\mu g/m^3$) a partir de variables meteorológicas (temperatura, humedad relativa, velocidad/dirección del viento, precipitación) y codificación cíclica temporal ($\sin/\cos$ de hora y mes).

* **Pronóstico Autoregresivo (**$84\text{h} \rightarrow 12\text{h}$**):** Modelo *Ridge Regression* entrenado sobre ventanas deslizantes de las últimas 84 horas para proyectar las próximas 12 horas, con *fallback* híbrido meteorológico automático.

* **Dashboard Interactivo & Analítica Geoespacial:**

  * Medidor radial (*Gauge*) dinámico según los umbrales de la **OMS** ($\le 12$, $12\text{–}35.4$ y $> 35.4\ \mu g/m^3$).

  * Gráfica de serie temporal con **Chart.js** y bandas horizontales de niveles de contaminación.

  * Mapa interactivo con **Leaflet.js** sincronizado bidireccionalmente con un ranking comparativo multi-ciudad.

* **Ciclo MLOps & Reentrenamiento Bajo Demanda:**

  * Ingesta horaria automatizada mediante **GitHub Actions** (`worker.py`) con validación de calidad de datos y registro de *fallbacks*.

  * Monitorización automática de *drift* por antigüedad del modelo o volumen de nuevos datos acumulados (`MAE` y `RMSE` almacenados en Neon).

  * Reentrenamiento bajo demanda desde la interfaz web mediante un endpoint protegido (`POST /model/retrain`) con recarga en caliente en memoria.

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────┐       ┌─────────────────────────────┐
│   Open-Meteo & OpenAQ   │       │      GitHub Actions         │
│   (Datos Meteo + PM2.5) │◄──────┤   (Cron Horario: worker.py) │
└────────────┬────────────┘       └──────────────┬──────────────┘
             │                                   │
             ▼                                   ▼
┌───────────────────────────────────────────────────────────────┐
│                 PostgreSQL Serverless (Neon)                  │
│         • air_measurements        • model_metrics             │
└───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────┐
│           FastAPI Backend + Scikit-Learn (Docker HF)          │
│  • /predict/{city}   • /forecast/{city}   • /cities/summary   │
│  • /metrics/latest   • /model/retrain (X-Admin-Token)         │
└───────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────┐
│       Frontend Dashboard (Tailwind CSS · Chart.js · Leaflet)  │
└───────────────────────────────────────────────────────────────┘

```

## 🛠️ Stack Tecnológico

* **Backend & API:** Python 3.11, FastAPI, Uvicorn, Pydantic, HTTPX.

* **Machine Learning:** Scikit-Learn (`LinearRegression`, `Ridge`, `Pipeline`, `StandardScaler`), Pandas, NumPy, Joblib.

* **Base de Datos & ORM:** PostgreSQL (Neon), SQLAlchemy.

* **Frontend:** HTML5, Tailwind CSS, Chart.js, Leaflet.js (OpenStreetMap Dark Mode).

* **DevOps / MLOps:** Docker, GitHub Actions, Hugging Face Spaces.

## 📂 Estructura del Proyecto

```
air-quality-ml-api/
├── .github/
│   └── workflows/
│       └── hourly_ingestion.yml   # Pipeline horario de ingesta hacia Neon
├── app/
│   ├── config.py                  # Configuración y variables de entorno
│   ├── database.py                # Conexión y sesión SQLAlchemy
│   ├── main.py                    # Endpoints FastAPI y gestión de modelos
│   ├── models.py                  # Tablas ORM (air_measurements, model_metrics)
│   └── schemas.py                 # Esquemas de validación Pydantic
├── model/
│   ├── linear_model.joblib        # Modelo de inferencia instantánea
│   └── forecast_model.joblib      # Modelo autoregresivo 84h -> 12h
├── static/
│   └── index.html                 # Dashboard web interactivo
├── Dockerfile                     # Configuración del contenedor para HF Spaces
├── requirements.txt               # Dependencias Python
├── seed_history.py                # Poblado inicial de histórico de 7 días
├── train_model.py                 # Entrenamiento del modelo instantáneo
├── train_forecast_model.py        # Entrenamiento del modelo 84h -> 12h + métricas
└── worker.py                      # Worker asíncrono de ingesta horaria

```

## 🚀 Instalación y Ejecución en Local

1. **Clonar el repositorio y crear el entorno virtual:**

   ```
   git clone https://github.com/tu-usuario/air-quality-ml-api.git
   cd air-quality-ml-api
   python -m venv .venv
   # En Windows:
   .venv\Scripts\activate
   # En Linux/Mac:
   source .venv/bin/activate
   
   ```

2. **Instalar dependencias:**

   ```
   pip install -r requirements.txt
   
   ```

3. **Configurar variables de entorno:**
   Crea un archivo `.env` en la raíz del proyecto con tus credenciales:

   ```
   DATABASE_URL=postgresql://usuario:password@host.neon.tech/neondb?sslmode=require
   OPENAQ_API_KEY=tu_api_key_de_openaq
   ADMIN_TOKEN=tu_clave_secreta_para_reentrenar
   
   ```

4. **Poblar datos históricos y entrenar modelos (opcional si ya existen en Neon):**

   ```
   python seed_history.py
   python train_model.py
   python train_forecast_model.py
   
   ```

5. **Iniciar el servidor:**

   ```
   uvicorn app.main:app --reload
   
   ```

   Accede al dashboard en **`http://127.0.0.1:8000/`** y a la documentación interactiva Swagger en **`http://127.0.0.1:8000/docs`**.

## 📡 Endpoints Principales de la API

| Método | Endpoint | Descripción | 
| ----- | ----- | ----- | 
| `GET` | `/` | Sirve el dashboard web principal. | 
| `GET` | `/health` | Comprueba el estado de la API, la carga de modelos y la conexión a Neon. | 
| `GET` | `/predict/{city}` | Predicción instantánea de `PM2.5` y telemetría meteorológica de la ciudad. | 
| `GET` | `/forecast/{city}` | Serie temporal con las últimas 84h reales y el pronóstico de las próximas 12h. | 
| `GET` | `/cities/summary` | Resumen en vivo de las 5 ciudades ordenadas de aire más limpio a más contaminado. | 
| `GET` | `/metrics/latest` | Últimas métricas (`MAE`, `RMSE`) y evaluación automática de necesidad de reentrenamiento. | 
| `POST` | `/model/retrain` | Reentrena y recarga el modelo en memoria (requiere cabecera `X-Admin-Token`). | 

## 🔐 Despliegue en Hugging Face Spaces & GitHub Actions

* **Hugging Face Spaces:** Configurar como *Space* con SDK **Docker** (puerto `7860`) y añadir `DATABASE_URL`, `OPENAQ_API_KEY` y `ADMIN_TOKEN` en *Settings → Variables and secrets*.

* **GitHub Actions:** Añadir `DATABASE_URL` y `OPENAQ_API_KEY` en *Settings → Secrets and variables → Actions* para habilitar la ingesta horaria automática.