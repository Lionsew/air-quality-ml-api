import asyncio
import logging
from datetime import datetime, timezone
import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.database import SessionLocal, Base, engine
from app.models import AirMeasurement

# Configuración básica de logs para monitorizar la ingesta
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Ciudades de referencia para tu porfolio
TARGET_CITIES = [
    {"name": "Alicante", "lat": 38.351111, "lon": -0.513889},
    {"name": "Madrid", "lat": 40.416775, "lon": -3.703790},
    {"name": "Barcelona", "lat": 41.385064, "lon": 2.173404},
    {"name": "Valencia", "lat": 39.469907, "lon": -0.376288},
    {"name": "Sevilla", "lat": 37.389092, "lon": -5.984459},
]


async def fetch_open_meteo(client: httpx.AsyncClient, lat: float, lon: float) -> dict:
    """Consulta las variables meteorológicas actuales en Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
        ],
    }
    response = await client.get(url, params=params, timeout=10.0)
    response.raise_for_status()
    current = response.json().get("current", {})

    return {
        "temperature_2m": float(current.get("temperature_2m", 20.0)),
        "relative_humidity_2m": float(current.get("relative_humidity_2m", 50.0)),
        "wind_speed_10m": float(current.get("wind_speed_10m", 5.0)),
        "wind_direction_10m": float(current.get("wind_direction_10m", 180.0)),
        "precipitation": float(current.get("precipitation", 0.0)),
    }


async def fetch_openaq(client: httpx.AsyncClient, lat: float, lon: float) -> tuple[float, bool]:
    """
    Consulta la estación más cercana en OpenAQ v3 para obtener PM2.5.
    Devuelve (valor_pm25, is_fallback) filtrando lecturas físicamente anómalas.
    """
    url = "https://api.openaq.org/v3/locations"
    headers = {"X-API-Key": settings.OPENAQ_API_KEY}
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": 25000,  # Radio de 25 km
        "limit": 5,
    }

    try:
        response = await client.get(url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        results = response.json().get("results", [])

        for station in results:
            for sensor in station.get("sensors", []):
                if sensor.get("parameter", {}).get("name") == "pm25":
                    sensor_id = sensor.get("id")
                    meas_url = f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements"
                    meas_resp = await client.get(meas_url, params={"limit": 1}, headers=headers, timeout=10.0)
                    if meas_resp.status_code == 200:
                        meas_data = meas_resp.json().get("results", [])
                        if meas_data:
                            val = float(meas_data[0]["value"])
                            if 0.0 <= val <= 500.0:
                                return val, False
                            logger.warning(f"Lectura anómala descartada ({val} µg/m³) en ({lat}, {lon}).")

        logger.warning(f"No se encontró sensor PM2.5 activo válido cerca de ({lat}, {lon}). Usando fallback.")
        return 12.5, True
    except Exception as e:
        logger.error(f"Error consultando OpenAQ en ({lat}, {lon}): {e}")
        return 12.5, True


async def process_city(client: httpx.AsyncClient, db: Session, city_info: dict):
    """Ejecuta ambas consultas en paralelo para una ciudad y guarda el registro."""
    name = city_info["name"]
    lat = city_info["lat"]
    lon = city_info["lon"]

    try:
        weather_data, (pm25_val, is_fallback) = await asyncio.gather(
            fetch_open_meteo(client, lat, lon),
            fetch_openaq(client, lat, lon),
        )

        record = AirMeasurement(
            city=name,
            latitude=lat,
            longitude=lon,
            temperature_2m=weather_data["temperature_2m"],
            relative_humidity_2m=weather_data["relative_humidity_2m"],
            wind_speed_10m=weather_data["wind_speed_10m"],
            wind_direction_10m=weather_data["wind_direction_10m"],
            precipitation=weather_data["precipitation"],
            pm25=pm25_val,
            is_fallback=is_fallback,
            recorded_at=datetime.now(timezone.utc),
        )
        db.add(record)
        db.commit()
        flag_str = " [FALLBACK]" if is_fallback else ""
        logger.info(f"[{name}] Guardado -> Temp: {record.temperature_2m}°C | PM2.5: {record.pm25} µg/m³{flag_str}")

    except Exception as e:
        db.rollback()
        logger.error(f"[{name}] Fallo en la ingesta: {e}")


async def run_ingestion():
    """Función principal que inicializa tablas y sincroniza todas las ciudades."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        async with httpx.AsyncClient() as client:
            tasks = [process_city(client, db, city) for city in TARGET_CITIES]
            await asyncio.gather(*tasks)
    finally:
        db.close()


if __name__ == "__main__":
    logger.info("Iniciando worker de ingesta de datos...")
    asyncio.run(run_ingestion())
    logger.info("Ingesta finalizada.")