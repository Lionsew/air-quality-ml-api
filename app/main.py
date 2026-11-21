from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import desc, text, func
from sqlalchemy.orm import Session

from app.config import settings
from app.database import Base, engine, get_db
from app.models import AirMeasurement, ModelMetric
from app.schemas import (
    CitiesSummaryResponse,
    CitySummaryItem,
    ForecastResponse,
    HealthResponse,
    PredictionResponse,
    TimeSeriesPoint,
    WeatherMetrics,
)

# Diccionario en memoria para mantener los modelos cargados durante todo el ciclo de vida
ml_assets = {}

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "linear_model.joblib"
FORECAST_MODEL_PATH = BASE_DIR / "model" / "forecast_model.joblib"
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Asegurar que las tablas existan en la base de datos
    Base.metadata.create_all(bind=engine)
    
    # 2. Cargar los pipelines entrenados de Scikit-Learn en memoria
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}. Ejecuta primero train_model.py.")
    
    ml_assets["model"] = joblib.load(MODEL_PATH)
    ml_assets["forecast_model"] = joblib.load(FORECAST_MODEL_PATH) if FORECAST_MODEL_PATH.exists() else None
    yield
    # Limpieza al apagar el servidor
    ml_assets.clear()


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "API de predicción y pronóstico de calidad del aire (PM2.5) mediante Regresión Lineal y Ridge Autoregresivo. "
        "Combina datos meteorológicos de Open-Meteo y mediciones de OpenAQ almacenadas en PostgreSQL."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Sirve la interfaz gráfica principal del dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check(db: Session = Depends(get_db)):
    """Verifica el estado del servicio, la carga del modelo ML y la conexión a base de datos."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    model_ok = "model" in ml_assets and ml_assets["model"] is not None

    return HealthResponse(
        status="ok" if (db_ok and model_ok) else "degraded",
        model_loaded=model_ok,
        database_connected=db_ok,
    )

@app.get("/metrics/latest", tags=["Monitoring"])
def get_latest_metrics(db: Session = Depends(get_db)):
    """
    Devuelve las últimas métricas de entrenamiento y evalúa si el modelo necesita reentrenarse
    según su antigüedad (> 7 días) o el volumen de nuevos datos acumulados (> 150 registros).
    """
    metric = (
        db.query(ModelMetric)
        .filter(ModelMetric.model_name == "autoregressive_84h_ridge")
        .order_by(desc(ModelMetric.trained_at))
        .first()
    )

    # Caso 1: Nunca se ha entrenado el modelo autoregresivo
    if not metric:
        total_records = db.query(func.count(AirMeasurement.id)).scalar() or 0
        return {
            "available": False,
            "needs_retraining": True,
            "reason": f"Modelo 84h aún sin entrenar ({total_records} registros disponibles en Neon).",
        }

    now_utc = datetime.now(timezone.utc)
    trained_at = metric.trained_at
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=timezone.utc)

    hours_since_train = (now_utc - trained_at).total_seconds() / 3600.0

    # Contar nuevos registros reales ingresados desde el último entrenamiento
    new_records_count = (
        db.query(func.count(AirMeasurement.id))
        .filter(
            AirMeasurement.recorded_at > trained_at,
            AirMeasurement.is_fallback == False,  # noqa: E712
        )
        .scalar()
        or 0
    )

    # Reglas de decisión para sugerir reentrenamiento
    reasons = []
    if hours_since_train >= 168:  # 7 días
        reasons.append(f"Han pasado {int(hours_since_train // 24)} días desde el último entrenamiento")
    if new_records_count >= 150:  # ~30h nuevas en las 5 ciudades
        reasons.append(f"Hay {new_records_count} nuevos registros reales listos para incorporar")

    needs_retraining = len(reasons) > 0

    return {
        "available": True,
        "model_name": metric.model_name,
        "mae": metric.mae,
        "rmse": metric.rmse,
        "samples_count": metric.samples_count,
        "trained_at": trained_at,
        "hours_since_train": round(hours_since_train, 1),
        "new_records_since_train": new_records_count,
        "needs_retraining": needs_retraining,
        "reason": " · ".join(reasons) if needs_retraining else "Modelo actualizado y dentro de parámetros.",
    }


@app.get("/predict/{city}", response_model=PredictionResponse, tags=["Predictions"])
def predict_air_quality(city: str, db: Session = Depends(get_db)):
    """
    Obtiene el último registro meteorológico y de calidad del aire de la ciudad indicada
    y devuelve la predicción instantánea de concentración de PM2.5.
    """
    record = (
        db.query(AirMeasurement)
        .filter(AirMeasurement.city.ilike(city.strip()))
        .order_by(desc(AirMeasurement.recorded_at))
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron datos recientes para '{city}'. Ciudades disponibles: Alicante, Madrid, Barcelona, Valencia, Sevilla."
        )

    hour = record.recorded_at.hour
    month = record.recorded_at.month

    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)

    feature_df = pd.DataFrame([{
        "temperature_2m": record.temperature_2m,
        "relative_humidity_2m": record.relative_humidity_2m,
        "wind_speed_10m": record.wind_speed_10m,
        "wind_direction_10m": record.wind_direction_10m,
        "precipitation": record.precipitation,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }])

    model = ml_assets["model"]
    prediction = float(model.predict(feature_df)[0])
    prediction = max(0.0, round(prediction, 2))

    return PredictionResponse(
        city=record.city,
        coordinates={"latitude": record.latitude, "longitude": record.longitude},
        recorded_at=record.recorded_at,
        input_metrics=WeatherMetrics(
            temperature_2m=record.temperature_2m,
            relative_humidity_2m=record.relative_humidity_2m,
            wind_speed_10m=record.wind_speed_10m,
            wind_direction_10m=record.wind_direction_10m,
            precipitation=record.precipitation,
            pm25_recorded=record.pm25,
        ),
        predicted_pm25=prediction,
    )


@app.get("/forecast/{city}", response_model=ForecastResponse, tags=["Predictions"])
async def forecast_air_quality(city: str, db: Session = Depends(get_db)):
    """
    Devuelve el histórico de las últimas 84 horas y la predicción para las próximas 12 horas.
    """
    records = (
        db.query(AirMeasurement)
        .filter(AirMeasurement.city.ilike(city.strip()))
        .order_by(desc(AirMeasurement.recorded_at))
        .limit(84)
        .all()
    )

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron datos históricos para '{city}'."
        )

    records = list(reversed(records))
    latest = records[-1]

    series: list[TimeSeriesPoint] = [
        TimeSeriesPoint(
            timestamp=r.recorded_at,
            pm25=round(float(r.pm25), 2),
            is_forecast=False,
        )
        for r in records
    ]

    forecast_model = ml_assets.get("forecast_model")

    # 1. Usar modelo autoregresivo Ridge si está entrenado y tenemos las 84h completas
    if forecast_model is not None and len(records) == 84:
        past_84 = np.array([float(r.pm25) for r in records])
        ts = latest.recorded_at
        stats = [np.mean(past_84), np.std(past_84), np.min(past_84), np.max(past_84)]
        cyclical = [
            np.sin(2 * np.pi * ts.hour / 24.0),
            np.cos(2 * np.pi * ts.hour / 24.0),
            np.sin(2 * np.pi * ts.month / 12.0),
            np.cos(2 * np.pi * ts.month / 12.0),
        ]
        X_in = np.concatenate([past_84, stats, cyclical]).reshape(1, -1)
        preds_12 = forecast_model.predict(X_in)[0]

        for step, val in enumerate(preds_12, start=1):
            series.append(
                TimeSeriesPoint(
                    timestamp=latest.recorded_at + timedelta(hours=step),
                    pm25=max(0.0, round(float(val), 2)),
                    is_forecast=True,
                )
            )
        model_used = "autoregressive_84h_ridge"

    # 2. Fallback híbrido con pronóstico horario Open-Meteo + linear_model.joblib
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latest.latitude,
            "longitude": latest.longitude,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
            ],
            "forecast_hours": 13,
            "timezone": "UTC",
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            hourly = resp.json().get("hourly", {})

        times = hourly.get("time", [])[1:13]
        rows = []
        parsed_times = []

        for idx, t_str in enumerate(times, start=1):
            dt = pd.to_datetime(t_str, utc=True)
            parsed_times.append(dt.to_pydatetime())
            rows.append({
                "temperature_2m": hourly["temperature_2m"][idx],
                "relative_humidity_2m": hourly["relative_humidity_2m"][idx],
                "wind_speed_10m": hourly["wind_speed_10m"][idx],
                "wind_direction_10m": hourly["wind_direction_10m"][idx],
                "precipitation": hourly["precipitation"][idx],
                "hour_sin": np.sin(2 * np.pi * dt.hour / 24.0),
                "hour_cos": np.cos(2 * np.pi * dt.hour / 24.0),
                "month_sin": np.sin(2 * np.pi * dt.month / 12.0),
                "month_cos": np.cos(2 * np.pi * dt.month / 12.0),
            })

        feature_df = pd.DataFrame(rows)
        preds_12 = ml_assets["model"].predict(feature_df)

        for dt, val in zip(parsed_times, preds_12):
            series.append(
                TimeSeriesPoint(
                    timestamp=dt,
                    pm25=max(0.0, round(float(val), 2)),
                    is_forecast=True,
                )
            )
        model_used = "meteo_hybrid_12h"
    
    return ForecastResponse(
        city=latest.city,
        model_used=model_used,
        history_hours=len(records),
        forecast_hours=12,
        series=series,
    )

from fastapi import Depends, FastAPI, Header, HTTPException
from train_forecast_model import train

# ... dentro de app/main.py:

@app.post("/model/retrain", tags=["Monitoring"])
def retrain_forecast_model(x_admin_token: str = Header(default="")):
    """
    Ejecuta el reentrenamiento del modelo 84h -> 12h bajo demanda, guarda las nuevas
    métricas en Neon y recarga el modelo en memoria sin reiniciar la API.
    """
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token de administrador inválido. Acceso denegado."
        )

    try:
        train()
        if FORECAST_MODEL_PATH.exists():
            ml_assets["forecast_model"] = joblib.load(FORECAST_MODEL_PATH)
        return {
            "status": "ok",
            "message": "Modelo reentrenado y recargado en memoria correctamente."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante el reentrenamiento: {str(e)}"
        )

@app.get("/cities/summary", response_model=CitiesSummaryResponse, tags=["Predictions"])
def get_cities_summary(db: Session = Depends(get_db)):
    """
    Devuelve el último estado meteorológico, coordenadas y predicción de PM2.5
    para todas las ciudades monitorizadas, ordenadas de aire más limpio a más contaminado.
    """
    target_cities = ["Alicante", "Madrid", "Barcelona", "Valencia", "Sevilla"]
    latest_records = []

    for city_name in target_cities:
        rec = (
            db.query(AirMeasurement)
            .filter(AirMeasurement.city.ilike(city_name))
            .order_by(desc(AirMeasurement.recorded_at))
            .first()
        )
        if rec:
            latest_records.append(rec)

    if not latest_records:
        return CitiesSummaryResponse(cities=[])

    # Preparar matriz de características en bloque para inferencia rápida
    rows = []
    for r in latest_records:
        hour = r.recorded_at.hour
        month = r.recorded_at.month
        rows.append({
            "temperature_2m": r.temperature_2m,
            "relative_humidity_2m": r.relative_humidity_2m,
            "wind_speed_10m": r.wind_speed_10m,
            "wind_direction_10m": r.wind_direction_10m,
            "precipitation": r.precipitation,
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "month_sin": np.sin(2 * np.pi * month / 12.0),
            "month_cos": np.cos(2 * np.pi * month / 12.0),
        })

    feature_df = pd.DataFrame(rows)
    preds = ml_assets["model"].predict(feature_df)

    items = []
    for r, pred_val in zip(latest_records, preds):
        items.append(
            CitySummaryItem(
                city=r.city,
                latitude=r.latitude,
                longitude=r.longitude,
                predicted_pm25=max(0.0, round(float(pred_val), 2)),
                recorded_pm25=round(float(r.pm25), 2),
                temperature_2m=round(float(r.temperature_2m), 1),
                wind_speed_10m=round(float(r.wind_speed_10m), 1),
                recorded_at=r.recorded_at,
            )
        )

    # Ordenar de menor a mayor concentración predicha (ranking de aire más limpio)
    items.sort(key=lambda x: x.predicted_pm25)

    return CitiesSummaryResponse(cities=items)   