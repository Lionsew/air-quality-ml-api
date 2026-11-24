from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from neuralforecast import NeuralForecast
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
from train_forecast_model import (
    ALL_EXOG_COLS,
    NHITS_MODEL_DIR,
    WEATHER_EXOG_COLS,
    add_temporal_features,
    build_nhits_dataframe,
    train,
)

# Diccionario en memoria para mantener los modelos cargados durante todo el ciclo de vida
ml_assets = {}

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "linear_model.joblib"
STATIC_DIR = BASE_DIR / "static"


def _build_weather_features_df(records: list[AirMeasurement]) -> pd.DataFrame:
    """Construye el DataFrame de features meteorológicas y cíclicas para linear_model.joblib."""
    rows = []
    for r in records:
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
    return pd.DataFrame(rows)


async def _fetch_open_meteo_hourly(latitude: float, longitude: float, past_days: int = 3, forecast_hours: int = 24) -> pd.DataFrame:
    """Obtiene el histórico reciente y pronóstico horario de Open-Meteo como DataFrame indexado por `ds`."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": WEATHER_EXOG_COLS,
        "past_days": past_days,
        "forecast_hours": forecast_hours,
        "timezone": "UTC",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10.0)
        resp.raise_for_status()
        hourly = resp.json().get("hourly", {})

    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=["ds"] + WEATHER_EXOG_COLS)

    df_meteo = pd.DataFrame({
        "ds": pd.to_datetime(times, utc=True),
        "temperature_2m": hourly.get("temperature_2m", []),
        "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
        "wind_speed_10m": hourly.get("wind_speed_10m", []),
        "wind_direction_10m": hourly.get("wind_direction_10m", []),
        "precipitation": hourly.get("precipitation", []),
    })
    return df_meteo.drop_duplicates(subset=["ds"], keep="last").sort_values("ds")


def _predict_nhits_for_city(nhits_model: NeuralForecast, records: list[AirMeasurement], meteo_df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Calcula con NHITS:
    1) Las estimaciones sobre la ventana histórica de 84h (walk-forward 12h antes de t0 -> t0)
       para comparar Real vs Modelo con precisión real de NHITS.
    2) El pronóstico futuro de +12h a partir del último timestamp registrado.
    """
    full_df = build_nhits_dataframe(records)
    city_id = full_df["unique_id"].iloc[-1]

    # Contexto de las últimas 84h hasta t0
    context_df = full_df.tail(84).copy()
    last_ts = context_df["ds"].max()

    # Generar exactamente las 12 marcas horarias posteriores a last_ts para futr_df
    future_ds = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=12, freq="h", tz="UTC")
    futr_base = pd.DataFrame({"unique_id": city_id, "ds": future_ds})

    if not meteo_df.empty:
        futr_base = pd.merge(futr_base, meteo_df, on="ds", how="left")
    else:
        for col in WEATHER_EXOG_COLS:
            futr_base[col] = np.nan

    # Si falta algún dato futuro en Open-Meteo, rellenar con los últimos valores conocidos
    for col in WEATHER_EXOG_COLS:
        last_known = float(context_df[col].iloc[-1])
        futr_base[col] = futr_base[col].ffill().bfill().fillna(last_known)

    futr_df = add_temporal_features(futr_base)[["unique_id", "ds"] + ALL_EXOG_COLS]
    future_preds_df = nhits_model.predict(df=context_df, futr_df=futr_df).sort_values("ds")

    # Estimación histórica con NHITS mediante ventanas deslizantes sobre full_df
    hist_est_map: dict[pd.Timestamp, float] = {}
    if len(full_df) >= 96:
        # Evaluar bloques de 12h sobre el histórico usando NHITS
        for end_idx in range(96, len(full_df) + 1, 12):
            sub_hist = full_df.iloc[max(0, end_idx - 96):end_idx - 12].copy()
            sub_futr = full_df.iloc[end_idx - 12:end_idx][["unique_id", "ds"] + ALL_EXOG_COLS].copy()
            if len(sub_hist) >= 24 and len(sub_futr) == 12:
                block_pred = nhits_model.predict(df=sub_hist, futr_df=sub_futr)
                for _, prow in block_pred.iterrows():
                    hist_est_map[prow["ds"]] = max(0.0, round(float(prow["NHITS"]), 2))

    # También evaluar las últimas 12h exactas ending at last_ts para que t0 tenga su estimación NHITS
    if len(full_df) >= 36:
        sub_hist_last = full_df.iloc[max(0, len(full_df) - 96):-12].copy()
        sub_futr_last = full_df.iloc[-12:][["unique_id", "ds"] + ALL_EXOG_COLS].copy()
        block_last = nhits_model.predict(df=sub_hist_last, futr_df=sub_futr_last)
        for _, prow in block_last.iterrows():
            hist_est_map[prow["ds"]] = max(0.0, round(float(prow["NHITS"]), 2))

    hist_series = pd.Series(hist_est_map)
    return hist_series, future_preds_df


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo base no encontrado en {MODEL_PATH}. Ejecuta primero train_model.py.")
    
    ml_assets["model"] = joblib.load(MODEL_PATH)
    ml_assets["nhits_model"] = NeuralForecast.load(path=str(NHITS_MODEL_DIR)) if NHITS_MODEL_DIR.exists() else None
    yield
    ml_assets.clear()


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "API de predicción y pronóstico de calidad del aire (PM2.5) mediante Regresión Lineal y Deep Learning NHITS "
        "con variables meteorológicas y temporales. Combina datos de Open-Meteo y OpenAQ en PostgreSQL."
    ),
    version="1.2.0",
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
    Devuelve las últimas métricas de entrenamiento del modelo NHITS y evalúa si necesita reentrenarse
    según su antigüedad (> 7 días) o el volumen de nuevos datos acumulados (> 150 registros).
    """
    metric = (
        db.query(ModelMetric)
        .filter(ModelMetric.model_name == "nhits_84h_exog")
        .order_by(desc(ModelMetric.trained_at))
        .first()
    )

    if not metric:
        total_records = db.query(func.count(AirMeasurement.id)).scalar() or 0
        return {
            "available": False,
            "needs_retraining": True,
            "reason": f"Modelo NHITS aún sin entrenar ({total_records} registros disponibles en Neon).",
        }

    now_utc = datetime.now(timezone.utc)
    trained_at = metric.trained_at
    if trained_at.tzinfo is None:
        trained_at = trained_at.replace(tzinfo=timezone.utc)

    hours_since_train = (now_utc - trained_at).total_seconds() / 3600.0

    new_records_count = (
        db.query(func.count(AirMeasurement.id))
        .filter(
            AirMeasurement.recorded_at > trained_at,
            AirMeasurement.is_fallback == False,  # noqa: E712
        )
        .scalar()
        or 0
    )

    reasons = []
    if hours_since_train >= 168:
        reasons.append(f"Han pasado {int(hours_since_train // 24)} días desde el último entrenamiento")
    if new_records_count >= 150:
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
        "reason": " · ".join(reasons) if needs_retraining else "Modelo NHITS actualizado y dentro de parámetros.",
    }


@app.get("/predict/{city}", response_model=PredictionResponse, tags=["Predictions"])
def predict_air_quality(city: str, db: Session = Depends(get_db)):
    """
    Obtiene el último registro meteorológico y de calidad del aire de la ciudad indicada
    y devuelve la predicción instantánea de concentración de PM2.5.
    """
    records = (
        db.query(AirMeasurement)
        .filter(AirMeasurement.city.ilike(city.strip()))
        .order_by(desc(AirMeasurement.recorded_at))
        .limit(96)
        .all()
    )

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron datos recientes para '{city}'. Ciudades disponibles: Alicante, Madrid, Barcelona, Valencia, Sevilla."
        )

    records = list(reversed(records))
    record = records[-1]

    nhits_model = ml_assets.get("nhits_model")
    prediction = None

    if nhits_model is not None and len(records) >= 36:
        try:
            full_df = build_nhits_dataframe(records)
            if len(full_df) >= 36:
                sub_hist = full_df.iloc[max(0, len(full_df) - 96):-12].copy()
                sub_futr = full_df.iloc[-12:][["unique_id", "ds"] + ALL_EXOG_COLS].copy()
                block_last = nhits_model.predict(df=sub_hist, futr_df=sub_futr)
                prediction = max(0.0, round(float(block_last["NHITS"].iloc[-1]), 2))
        except Exception:
            prediction = None

    if prediction is None:
        feature_df = _build_weather_features_df([record])
        prediction = max(0.0, round(float(ml_assets["model"].predict(feature_df)[0]), 2))

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
    Devuelve el histórico de las últimas 84 horas (con valor real y estimación del modelo
    NHITS para comparar la desviación) y la proyección a 12 horas con NHITS + variables exógenas.
    """
    # Obtenemos hasta 168 registros para dar contexto suficiente al backtest de NHITS sobre las últimas 84h
    all_records = (
        db.query(AirMeasurement)
        .filter(AirMeasurement.city.ilike(city.strip()))
        .order_by(desc(AirMeasurement.recorded_at))
        .limit(168)
        .all()
    )

    if not all_records:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron datos históricos para '{city}'."
        )

    all_records = list(reversed(all_records))
    display_records = all_records[-84:]
    latest = display_records[-1]

    meteo_df = await _fetch_open_meteo_hourly(latest.latitude, latest.longitude, past_days=4, forecast_hours=24)
    nhits_model = ml_assets.get("nhits_model")

    # 1. Usar modelo profundo NHITS con variables meteorológicas + todas las variables temporales
    if nhits_model is not None and len(all_records) >= 24:
        try:
            hist_est_series, future_preds_df = _predict_nhits_for_city(nhits_model, all_records, meteo_df)

            # Fallback puntual solo para los primeros puntos donde no haya ventana previa de 84h
            fallback_preds = ml_assets["model"].predict(_build_weather_features_df(display_records))

            series: list[TimeSeriesPoint] = []
            for r, fb_val in zip(display_records, fallback_preds):
                ts_floor = pd.to_datetime(r.recorded_at, utc=True).floor("h")
                est_val = hist_est_series.get(ts_floor, None)
                series.append(
                    TimeSeriesPoint(
                        timestamp=r.recorded_at,
                        pm25=round(float(r.pm25), 2),
                        predicted_pm25=float(est_val) if est_val is not None else None,
                        is_forecast=False,
                    )
                )

            for _, row in future_preds_df.iterrows():
                pred_clean = max(0.0, round(float(row["NHITS"]), 2))
                series.append(
                    TimeSeriesPoint(
                        timestamp=row["ds"].to_pydatetime(),
                        pm25=pred_clean,
                        predicted_pm25=pred_clean,
                        is_forecast=True,
                    )
                )

            return ForecastResponse(
                city=latest.city,
                model_used="nhits_84h_exog",
                history_hours=len(display_records),
                forecast_hours=12,
                series=series,
            )
        except Exception as e:
            print(f"[WARN] Fallback en /forecast/{city} por error en NHITS: {e}")

    # 2. Fallback híbrido con pronóstico horario Open-Meteo + linear_model.joblib
    hist_preds = ml_assets["model"].predict(_build_weather_features_df(display_records))
    series = [
        TimeSeriesPoint(
            timestamp=r.recorded_at,
            pm25=round(float(r.pm25), 2),
            predicted_pm25=max(0.0, round(float(pred_val), 2)),
            is_forecast=False,
        )
        for r, pred_val in zip(display_records, hist_preds)
    ]

    last_ts = pd.to_datetime(latest.recorded_at, utc=True).floor("h")
    future_ds = pd.date_range(start=last_ts + pd.Timedelta(hours=1), periods=12, freq="h", tz="UTC")
    futr_base = pd.DataFrame({"ds": future_ds})
    if not meteo_df.empty:
        futr_base = pd.merge(futr_base, meteo_df, on="ds", how="left")
    else:
        for col in WEATHER_EXOG_COLS:
            futr_base[col] = np.nan

    for col in WEATHER_EXOG_COLS:
        futr_base[col] = futr_base[col].ffill().bfill().fillna(float(getattr(latest, col)))

    rows = []
    for _, row in futr_base.iterrows():
        dt = row["ds"]
        rows.append({
            "temperature_2m": row["temperature_2m"],
            "relative_humidity_2m": row["relative_humidity_2m"],
            "wind_speed_10m": row["wind_speed_10m"],
            "wind_direction_10m": row["wind_direction_10m"],
            "precipitation": row["precipitation"],
            "hour_sin": np.sin(2 * np.pi * dt.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * dt.hour / 24.0),
            "month_sin": np.sin(2 * np.pi * dt.month / 12.0),
            "month_cos": np.cos(2 * np.pi * dt.month / 12.0),
        })

    preds_12 = ml_assets["model"].predict(pd.DataFrame(rows))
    for dt, val in zip(future_ds, preds_12):
        pred_clean = max(0.0, round(float(val), 2))
        series.append(
            TimeSeriesPoint(
                timestamp=dt.to_pydatetime(),
                pm25=pred_clean,
                predicted_pm25=pred_clean,
                is_forecast=True,
            )
        )

    return ForecastResponse(
        city=latest.city,
        model_used="meteo_hybrid_12h",
        history_hours=len(display_records),
        forecast_hours=12,
        series=series,
    )


@app.post("/model/retrain", tags=["Monitoring"])
def retrain_forecast_model(x_admin_token: str = Header(default="")):
    """
    Ejecuta el reentrenamiento del modelo NHITS (84h -> 12h) bajo demanda, guarda las nuevas
    métricas en Neon y recarga el modelo en memoria sin reiniciar la API.
    """
    if x_admin_token != settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token de administrador inválido. Acceso denegado."
        )

    try:
        train()
        if NHITS_MODEL_DIR.exists():
            ml_assets["nhits_model"] = NeuralForecast.load(path=str(NHITS_MODEL_DIR))
        return {
            "status": "ok",
            "message": "Modelo NHITS reentrenado y recargado en memoria correctamente."
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

    feature_df = _build_weather_features_df(latest_records)
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

    items.sort(key=lambda x: x.predicted_pm25)
    return CitiesSummaryResponse(cities=items)
