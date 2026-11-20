import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.database import Base, SessionLocal, engine
from app.models import AirMeasurement, ModelMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_OUT = BASE_DIR / "model" / "forecast_model.joblib"

HISTORY_HOURS = 84
FORECAST_HOURS = 12


def build_sliding_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye ventanas deslizantes por ciudad:
    - X: 84 valores horarios previos de PM2.5 + 4 estadísticos + 4 features cíclicas (92 features)
    - y: 12 valores horarios futuros de PM2.5 (t+1 ... t+12)
    """
    X_list, y_list = [], []
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], utc=True)

    for city, group in df.groupby("city"):
        group = group.sort_values("recorded_at").drop_duplicates(subset=["recorded_at"])

        # Re-muestrear a frecuencia horaria regular e interpolar huecos cortos (máx 3h)
        hourly = (
            group.set_index("recorded_at")[["pm25"]]
            .resample("1h")
            .mean()
            .interpolate(method="linear", limit=3)
            .dropna()
            .reset_index()
        )

        total_len = len(hourly)
        min_required = HISTORY_HOURS + FORECAST_HOURS

        if total_len < min_required:
            logger.warning(
                f"[{city}] Solo hay {total_len}h continuas (se necesitan al menos {min_required}h). Saltando ciudad."
            )
            continue

        values = hourly["pm25"].values
        timestamps = hourly["recorded_at"]

        for i in range(HISTORY_HOURS, total_len - FORECAST_HOURS + 1):
            past_84 = values[i - HISTORY_HOURS : i]
            future_12 = values[i : i + FORECAST_HOURS]

            ts = timestamps.iloc[i]
            hour_sin = np.sin(2 * np.pi * ts.hour / 24.0)
            hour_cos = np.cos(2 * np.pi * ts.hour / 24.0)
            month_sin = np.sin(2 * np.pi * ts.month / 12.0)
            month_cos = np.cos(2 * np.pi * ts.month / 12.0)

            stats = [
                float(np.mean(past_84)),
                float(np.std(past_84)),
                float(np.min(past_84)),
                float(np.max(past_84)),
            ]

            features = np.concatenate([past_84, stats, [hour_sin, hour_cos, month_sin, month_cos]])
            X_list.append(features)
            y_list.append(future_12)

    if not X_list:
        return np.empty((0, 92)), np.empty((0, FORECAST_HOURS))

    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def train():
    # 1. Asegurar que la tabla model_metrics exista en Neon
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        query = db.query(AirMeasurement).statement
        df = pd.read_sql(query, db.bind)

        if df.empty:
            logger.warning("La tabla air_measurements está vacía. Ejecuta seed_history.py primero.")
            return

        # 2. Filtrar datos sintetizados por fallback
        if "is_fallback" in df.columns:
            initial_rows = len(df)
            df = df[df["is_fallback"] == False].copy()  # noqa: E712
            logger.info(f"Filtro Data Quality: {initial_rows - len(df)} filas fallback excluidas.")

        logger.info(f"Registros válidos cargados desde Neon: {len(df)} filas en {df['city'].nunique()} ciudades.")
        X, y = build_sliding_windows(df)

        if len(X) < 15:
            logger.warning(
                f"Muestras insuficientes para entrenar ({len(X)} ventanas). "
                "Acumula más horas (mínimo ~110h por ciudad) y vuelve a ejecutar."
            )
            return

        # 3. División Train / Test y entrenamiento del pipeline
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=15.0)),
        ])
        pipeline.fit(X_train, y_train)

        # 4. Evaluación en conjunto de test
        y_pred = np.maximum(0.0, pipeline.predict(X_test))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(root_mean_squared_error(y_test, y_pred))

        # 5. Guardar artefacto serializado del modelo
        MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_OUT)

        # 6. Registrar métricas en PostgreSQL (Neon)
        metric_record = ModelMetric(
            model_name="autoregressive_84h_ridge",
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            samples_count=int(len(X)),
        )
        db.add(metric_record)
        db.commit()

        logger.info(
            f"Entrenamiento completado ({len(X)} ventanas) -> MAE: {mae:.2f} µg/m³ | RMSE: {rmse:.2f} µg/m³"
        )
        logger.info(f"Artefacto guardado en {MODEL_OUT} y métricas registradas en Neon.")

    except Exception as e:
        db.rollback()
        logger.error(f"Error durante el entrenamiento: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    train()