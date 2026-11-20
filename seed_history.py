import asyncio
import logging
from datetime import datetime, timezone
import httpx
import pandas as pd
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import AirMeasurement

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_CITIES = [
    {"name": "Alicante", "lat": 38.351111, "lon": -0.513889},
    {"name": "Madrid", "lat": 40.416775, "lon": -3.703790},
    {"name": "Barcelona", "lat": 41.385064, "lon": 2.173404},
    {"name": "Valencia", "lat": 39.469907, "lon": -0.376288},
    {"name": "Sevilla", "lat": 37.389092, "lon": -5.984459},
]

PAST_DAYS = 7


async def fetch_historical_weather(client: httpx.AsyncClient, lat: float, lon: float) -> pd.DataFrame:
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
        "past_days": PAST_DAYS,
        "forecast_days": 0,
        "timezone": "UTC",
    }
    resp = await client.get(url, params=params, timeout=15.0)
    resp.raise_for_status()
    hourly = resp.json().get("hourly", {})
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


async def fetch_historical_pm25(client: httpx.AsyncClient, lat: float, lon: float) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm2_5"],
        "past_days": PAST_DAYS,
        "forecast_days": 0,
        "timezone": "UTC",
    }
    resp = await client.get(url, params=params, timeout=15.0)
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
            db.bulk_save_objects(records_to_insert)
            db.commit()
            logger.info(f"[{name}] Insertados {len(records_to_insert)} registros históricos en Neon.")
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
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Iniciando carga histórica (últimos 7 días) hacia Neon...")
    asyncio.run(run_seed())
    logger.info("Carga histórica completada.")