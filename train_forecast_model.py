import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from app.database import SessionLocal
from app.models import AirMeasurement, ModelMetric

BASE_DIR = Path(__file__).resolve().parent
NHITS_MODEL_DIR = BASE_DIR / "model" / "nhits_forecast_model"

# Festivos españoles estándar (anuales fijos) para features de calendario
FIXED_HOLIDAYS_MD = {
    (1, 1),   # Año Nuevo
    (1, 6),   # Epifanía
    (5, 1),   # Día del Trabajador
    (8, 15),  # Asunción
    (10, 12), # Fiesta Nacional
    (11, 1),  # Todos los Santos
    (12, 6),  # Día de la Constitución
    (12, 8),  # Inmaculada Concepción
    (12, 25), # Navidad
    (12, 31), # Nochevieja
}

WEATHER_EXOG_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
]

TEMPORAL_EXOG_COLS = [
    "hour",
    "day_of_week",
    "month",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "is_holiday",
    "is_workday",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
]

ALL_EXOG_COLS = WEATHER_EXOG_COLS + TEMPORAL_EXOG_COLS


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade todas las variables temporales y de calendario a partir de la columna `ds`."""
    out = df.copy()
    ds = pd.to_datetime(out["ds"], utc=True)
    out["ds"] = ds

    out["hour"] = ds.dt.hour.astype(int)
    out["day_of_week"] = ds.dt.dayofweek.astype(int)
    out["month"] = ds.dt.month.astype(int)
    out["day_of_year"] = ds.dt.dayofyear.astype(int)
    out["week_of_year"] = ds.dt.isocalendar().week.astype(int)
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

    is_hol = [1 if (m, d) in FIXED_HOLIDAYS_MD else 0 for m, d in zip(out["month"], ds.dt.day)]
    out["is_holiday"] = pd.Series(is_hol, index=out.index, dtype=int)
    out["is_workday"] = ((out["is_weekend"] == 0) & (out["is_holiday"] == 0)).astype(int)

    out["is_month_start"] = ds.dt.is_month_start.astype(int)
    out["is_month_end"] = ds.dt.is_month_end.astype(int)
    out["is_quarter_start"] = ds.dt.is_quarter_start.astype(int)
    out["is_quarter_end"] = ds.dt.is_quarter_end.astype(int)
    out["is_year_start"] = ds.dt.is_year_start.astype(int)
    out["is_year_end"] = ds.dt.is_year_end.astype(int)
    return out


def build_nhits_dataframe(records: list[AirMeasurement]) -> pd.DataFrame:
    """Convierte registros ORM de AirMeasurement al formato Nixtla (unique_id, ds, y + exógenas)."""
    rows = []
    for r in records:
        rows.append({
            "unique_id": r.city,
            "ds": pd.to_datetime(r.recorded_at, utc=True).floor("h"),
            "y": float(r.pm25),
            "temperature_2m": float(r.temperature_2m),
            "relative_humidity_2m": float(r.relative_humidity_2m),
            "wind_speed_10m": float(r.wind_speed_10m),
            "wind_direction_10m": float(r.wind_direction_10m),
            "precipitation": float(r.precipitation),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["unique_id", "ds"]).drop_duplicates(subset=["unique_id", "ds"], keep="last")

    # Reindexar a frecuencia horaria continua por ciudad para evitar saltos en NeuralForecast
    full_dfs = []
    for city_id, group in df.groupby("unique_id"):
        full_idx = pd.date_range(start=group["ds"].min(), end=group["ds"].max(), freq="h", tz="UTC")
        g = group.set_index("ds").reindex(full_idx)
        g.index.name = "ds"
        g["unique_id"] = city_id
        for col in ["y"] + WEATHER_EXOG_COLS:
            g[col] = g[col].ffill().bfill().fillna(0.0)
        full_dfs.append(g.reset_index())

    df_clean = pd.concat(full_dfs, ignore_index=True)
    df_clean = add_temporal_features(df_clean)
    return df_clean[["unique_id", "ds", "y"] + ALL_EXOG_COLS]


def train():
    """
    Entrena el modelo NHITS (84h -> 12h) optimizado para reducir el overfit.
    """
    db = SessionLocal()
    try:
        records = (
            db.query(AirMeasurement)
            .order_by(AirMeasurement.city, AirMeasurement.recorded_at)
            .all()
        )
        if len(records) < 500:  # Aumentamos ligeramente el umbral de seguridad si es posible
            raise RuntimeError(f"Datos insuficientes en BD para entrenar NHITS ({len(records)} registros).")

        df = build_nhits_dataframe(records)
        horizon = 12
        input_size = 84  # 7 * 12h ventana de contexto
        
        # Validacion más amplia (ej. 7 días = 168 horas, o al menos 4 veces el horizonte)
        val_size = horizon * 14 

        min_len = df.groupby("unique_id").size().min()
        if min_len < input_size + val_size + horizon:
            raise RuntimeError(
                f"Se requieren más registros por ciudad para la validación ampliada."
            )

        # 1. Evaluación retrospectiva (cross_validation con un val_size realista)
        eval_model = NHITS(
                h=horizon,
                input_size=input_size,
                futr_exog_list=ALL_EXOG_COLS,
                hist_exog_list=WEATHER_EXOG_COLS,
                stack_types=["identity", "identity", "identity"],
                n_blocks=[1, 1, 1],
                # Arquitectura reducida pero uniforme para evitar conflictos de dimensiones
                mlp_units=[[128, 128], [128, 128], [128, 128]],
                n_pool_kernel_size=[2, 2, 1],
                n_freq_downsample=[4, 2, 1],
                max_steps=400,
                val_check_steps=50,
                early_stop_patience_steps=6,
                learning_rate=5e-4,
                batch_size=32,
                scaler_type="robust",
                accelerator="auto",
                random_seed=42,
            )
        nf_eval = NeuralForecast(models=[eval_model], freq="h")
        cv_df = nf_eval.cross_validation(df=df, h=horizon, n_windows=1, step_size=horizon, val_size=val_size)

        y_true = np.clip(cv_df["y"].values, 0.0, None)
        y_pred = np.clip(cv_df["NHITS"].values, 0.0, None)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(root_mean_squared_error(y_true, y_pred))

        # 2. Modelo final con la misma arquitectura regularizada
        prod_model = NHITS(
                h=horizon,
                input_size=input_size,
                futr_exog_list=ALL_EXOG_COLS,
                hist_exog_list=WEATHER_EXOG_COLS,
                stack_types=["identity", "identity", "identity"],
                n_blocks=[1, 1, 1],
                # Arquitectura reducida pero uniforme para evitar conflictos de dimensiones
                mlp_units=[[128, 128], [128, 128], [128, 128]],
                n_pool_kernel_size=[2, 2, 1],
                n_freq_downsample=[4, 2, 1],
                max_steps=400,
                val_check_steps=50,
                early_stop_patience_steps=6,
                learning_rate=5e-4,
                batch_size=32,
                scaler_type="robust",
                accelerator="auto",
                random_seed=42,
            )
        nf_prod = NeuralForecast(models=[prod_model], freq="h")
        nf_prod.fit(df=df, val_size=val_size)

        NHITS_MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        nf_prod.save(path=str(NHITS_MODEL_DIR), model_index=None, overwrite=True, save_dataset=False)

        metric = ModelMetric(
            model_name="nhits_84h_exog",
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            samples_count=len(df),
            trained_at=datetime.now(timezone.utc),
        )
        db.add(metric)
        db.commit()
        print(f"Modelo NHITS entrenado correctamente. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    finally:
        db.close()


if __name__ == "__main__":
    train()
