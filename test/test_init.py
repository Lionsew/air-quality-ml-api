from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models import AirMeasurement

# 1. Configurar base de datos SQLite temporal en memoria para tests aislados
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Sobrescribir la dependencia de base de datos de FastAPI
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_database():
    """Crea las tablas antes de cada test y las destruye al terminar."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_health_endpoint():
    """Verifica que el endpoint /health responda 200 y confirme modelo y DB."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["database_connected"] is True


def test_predict_city_success():
    """Inserta un registro de prueba y verifica la predicción del modelo ML."""
    db = TestingSessionLocal()
    sample = AirMeasurement(
        city="Alicante",
        latitude=38.351111,
        longitude=-0.513889,
        temperature_2m=22.5,
        relative_humidity_2m=60.0,
        wind_speed_10m=12.0,
        wind_direction_10m=140.0,
        precipitation=0.0,
        pm25=18.0,
        recorded_at=datetime.now(timezone.utc),
    )
    db.add(sample)
    db.commit()
    db.close()

    with TestClient(app) as client:
        # Probamos insensibilidad a mayúsculas/minúsculas ("alicante")
        response = client.get("/predict/alicante")
        assert response.status_code == 200
        data = response.json()
        assert data["city"] == "Alicante"
        assert "predicted_pm25" in data
        assert isinstance(data["predicted_pm25"], float)
        assert data["predicted_pm25"] >= 0.0


def test_predict_city_not_found():
    """Verifica el manejo controlado 404 para ciudades sin datos."""
    with TestClient(app) as client:
        response = client.get("/predict/CiudadInexistente")
        assert response.status_code == 404
        assert "No se encontraron datos" in response.json()["detail"]