from datetime import datetime
from typing import Dict, List
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool


class WeatherMetrics(BaseModel):
    temperature_2m: float
    relative_humidity_2m: float
    wind_speed_10m: float
    wind_direction_10m: float
    precipitation: float
    pm25_recorded: float


class PredictionResponse(BaseModel):
    city: str
    coordinates: Dict[str, float]
    recorded_at: datetime
    input_metrics: WeatherMetrics
    predicted_pm25: float


class TimeSeriesPoint(BaseModel):
    timestamp: datetime
    pm25: float
    predicted_pm25: float | None = None
    is_forecast: bool

class ForecastResponse(BaseModel):
    city: str
    model_used: str
    history_hours: int
    forecast_hours: int
    series: List[TimeSeriesPoint]

class CitySummaryItem(BaseModel):
    city: str
    latitude: float
    longitude: float
    predicted_pm25: float
    recorded_pm25: float
    temperature_2m: float
    wind_speed_10m: float
    recorded_at: datetime


class CitiesSummaryResponse(BaseModel):
    cities: List[CitySummaryItem]