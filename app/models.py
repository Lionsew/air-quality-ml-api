from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from app.database import Base


class AirMeasurement(Base):
    __tablename__ = "air_measurements"

    id = Column(Integer, primary_key=True, index=True)
    city = Column(String, index=True, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    temperature_2m = Column(Float, nullable=False)
    relative_humidity_2m = Column(Float, nullable=False)
    wind_speed_10m = Column(Float, nullable=False)
    wind_direction_10m = Column(Float, nullable=False)
    precipitation = Column(Float, nullable=False)
    pm25 = Column(Float, nullable=False)
    is_fallback = Column(Boolean, default=False, nullable=False)
    recorded_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
    )


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True, nullable=False)
    mae = Column(Float, nullable=False)
    rmse = Column(Float, nullable=False)
    samples_count = Column(Integer, nullable=False)
    trained_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )