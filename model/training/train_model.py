import os
import joblib
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==============================================================================
# 1. CARGA Y PIVOTADO DE DATOS DE OPENAQ
# ==============================================================================
csv_path = "model\\training\\qualitydata.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

df_aq_raw = pd.read_csv(csv_path)

# Normalizar fecha en UTC
df_aq_raw["datetimeUtc"] = pd.to_datetime(df_aq_raw["datetimeUtc"], utc=True)

# Pivotar para obtener una fila por hora y una columna por cada contaminante
df_aq_pivoted = df_aq_raw.pivot_table(
    index="datetimeUtc",
    columns="parameter",
    values="value",
    aggfunc="mean"
).reset_index()

df_aq_pivoted.columns.name = None
df_aq_pivoted.rename(columns={"datetimeUtc": "date"}, inplace=True)
print(f"[OpenAQ] {df_aq_pivoted.shape[0]} registros horarios procesados.")


# ==============================================================================
# 2. CONSULTA Y PROCESAMIENTO DE OPEN-METEO
# ==============================================================================
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Coordenadas de la estación Rabassa (Alicante): 38.351111, -0.513889
url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
params = {
    "latitude": 38.351111,
    "longitude": -0.513889,
    "start_date": "2022-01-01",
    "end_date": "2023-06-01",  # Ajustado al rango presente en el dataset de OpenAQ
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
        "precipitation_probability", "precipitation", "rain", "showers", "snowfall",
        "snow_depth", "wind_speed_10m", "wind_speed_80m", "wind_speed_100m", "wind_speed_120m",
        "wind_speed_180m", "wind_speed_200m", "wind_direction_10m", "wind_direction_80m",
        "wind_direction_100m", "wind_direction_120m", "wind_direction_180m", "wind_direction_200m",
        "temperature_80m", "temperature_120m", "temperature_180m", "wind_gusts_10m"
    ],
}

responses = openmeteo.weather_api(url, params=params)
response = responses[0]

hourly = response.Hourly()
variables_list = params["hourly"]

hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
}

for i, var_name in enumerate(variables_list):
    hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

hourly_dataframe = pd.DataFrame(data=hourly_data)
print(f"[Open-Meteo] {hourly_dataframe.shape[0]} registros horarios obtenidos.")


# ==============================================================================
# 3. UNIÓN DE DATOS Y LIMPIEZA
# ==============================================================================
df_merged = pd.merge(hourly_dataframe, df_aq_pivoted, on="date", how="inner")

# Recomendación 1: Descartar nulos únicamente en la variable objetivo (PM2.5)
target_col = "pm25"
df_clean = df_merged.dropna(subset=[target_col]).copy()
df_clean.sort_values(by="date", inplace=True)
df_clean.reset_index(drop=True, inplace=True)

print(f"[Dataset Limpio] {df_clean.shape[0]} registros válidos con PM2.5.")


# ==============================================================================
# 4. INGENIERÍA DE CARACTERÍSTICAS (TEMPORALES Y CÍCLICAS)
# ==============================================================================
# Extraer componentes de fecha
df_clean["hour"] = df_clean["date"].dt.hour
df_clean["month"] = df_clean["date"].dt.month
df_clean["dayofweek"] = df_clean["date"].dt.dayofweek

# Recomendación 2: Codificación cíclica (seno y coseno) para hora y mes
df_clean["hour_sin"] = np.sin(2 * np.pi * df_clean["hour"] / 24.0)
df_clean["hour_cos"] = np.cos(2 * np.pi * df_clean["hour"] / 24.0)
df_clean["month_sin"] = np.sin(2 * np.pi * df_clean["month"] / 12.0)
df_clean["month_cos"] = np.cos(2 * np.pi * df_clean["month"] / 12.0)

# Definir conjunto de variables predictoras
feature_cols = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

X = df_clean[feature_cols]
y = df_clean[target_col]


# ==============================================================================
# 5. DIVISIÓN TEMPORAL Y ENTRENAMIENTO CON PIPELINE
# ==============================================================================
# División cronológica (80% entrenamiento, 20% prueba)
split_index = int(len(df_clean) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Recomendación 3: Pipeline con StandardScaler + LinearRegression
ml_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

ml_pipeline.fit(X_train, y_train)

# Evaluación
y_pred = ml_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados del Modelo ---")
print(f"RMSE en test: {rmse:.2f} µg/m³")
print(f"R² en test:   {r2:.4f}")


# ==============================================================================
# 6. EXPORTACIÓN DEL MODELO PARA LA API
# ==============================================================================
os.makedirs("model", exist_ok=True)
model_output_path = os.path.join("model", "linear_model.joblib")

# Se guarda el pipeline completo (incluye el escalador entrenado)
joblib.dump(ml_pipeline, model_output_path)
print(f"\nPipeline exportado exitosamente a: {model_output_path}")

# Guardar también el dataset consolidado por si se requiere análisis exploratorio
df_clean.to_csv("dataset_consolidado_alicante.csv", index=False)
print("Dataset consolidado guardado en: dataset_consolidado_alicante.csv")
