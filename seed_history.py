import asyncio
import logging
from datetime import datetime, timedelta, timezone
import httpx
import pandas as pd
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import AirMeasurement

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_CITIES = [
    {"name": "Londres", "lat": 51.507351, "lon": -0.127758},
    {"name": "Barcelona", "lat": 41.385064, "lon": 2.173404},
    {"name": "Roma", "lat": 41.902783, "lon": 12.496366},
    {"name": "Mumbai", "lat": 19.075984, "lon": 72.877656},
    {"name": "Tokyo", "lat": 35.676192, "lon": 139.650311},
    {"name": "Nueva York", "lat": 40.712776, "lon": -74.005974},
]

DAYS_HISTORY = 90  # Últimos 3 meses


def _get_date_range() -> tuple[str, str]:
    """Calcula start_date (hace 90 días) y end_date (hoy) en formato YYYY-MM-DD UTC."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=DAYS_HISTORY)
    return start.isoformat(), today.isoformat()


async def fetch_historical_weather(client: httpx.AsyncClient, lat: float, lon: float) -> pd.DataFrame:
    start_date, end_date = _get_date_range()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    resp = await client.get(url, params=params, timeout=30.0)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


async def fetch_historical_pm25(client: httpx.AsyncClient, lat: float, lon: float) -> pd.DataFrame:
    start_date, end_date = _get_date_range()
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm2_5"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }
    resp = await client.get(url, params=params, timeout=30.0)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


async def seed_city(client: httpx.AsyncClient, db: Session, city_info: dict):
    name = city_info["name"]
    lat = city_info["lat"]
    lon = city_info["lon"]

    try:
        weather_df, aq_df = await asyncio.gather(
            fetch_historical_weather(client, lat, lon),
            fetch_historical_pm25(client, lat, lon),
        )

        merged = pd.merge(weather_df, aq_df, on="time", how="inner").dropna()
        now_utc = datetime.now(timezone.utc)
        merged = merged[merged["time"] <= now_utc].copy()

        existing_times = {
            row[0].replace(tzinfo=timezone.utc) if row[0].tzinfo is None else row[0]
            for row in db.query(AirMeasurement.recorded_at)
            .filter(AirMeasurement.city == name)
            .all()
        }

        records_to_insert = []
        for _, row in merged.iterrows():
            ts = row["time"].to_pydatetime()
            if ts in existing_times:
                continue

            records_to_insert.append(
                AirMeasurement(
                    city=name,
                    latitude=lat,
                    longitude=lon,
                    temperature_2m=float(row["temperature_2m"]),
                    relative_humidity_2m=float(row["relative_humidity_2m"]),
                    wind_speed_10m=float(row["wind_speed_10m"]),
                    wind_direction_10m=float(row["wind_direction_10m"]),
                    precipitation=float(row["precipitation"]),
                    pm25=float(row["pm2_5"]),
                    is_fallback=False,
                    recorded_at=ts,
                )
            )

        if records_to_insert:
            # Inserción en bloques de 500 para evitar timeouts en Neon PostgreSQL
            batch_size = 500
            for i in range(0, len(records_to_insert), batch_size):
                db.bulk_save_objects(records_to_insert[i:i + batch_size])
                db.commit()
            logger.info(f"[{name}] Insertados {len(records_to_insert)} registros históricos (últimos {DAYS_HISTORY} días).")
        else:
            logger.info(f"[{name}] No hay nuevos registros históricos que insertar.")

    except Exception as e:
        db.rollback()
        logger.error(f"[{name}] Error poblando historial: {e}")


async def run_seed():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        async with httpx.AsyncClient() as client:
            for city in TARGET_CITIES:
                await seed_city(client, db, city)
                await asyncio.sleep(1.0)  # Pausa breve para respetar rate limits de Open-Meteo
    finally:
        db.close()


if __name__ == "__main__":
    logger.info(f"Iniciando carga histórica (últimos {DAYS_HISTORY} días) hacia la base de datos...")
    asyncio.run(run_seed())
    logger.info("Carga histórica completada.")