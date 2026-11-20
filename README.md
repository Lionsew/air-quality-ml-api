---
title: Air Quality ML Predictor
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🌍 Air Quality & Weather ML Prediction API

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat&logo=postgresql)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker)](https://www.docker.com/)

API REST de predicción de calidad del aire (**PM2.5**) basada en Machine Learning. Diseñada con arquitectura desacoplada entre la ingesta asíncrona de datos externos (**Open-Meteo** & **OpenAQ**) y el servicio de inferencia de baja latencia (<10 ms).

---

## 🛠️ Arquitectura del Sistema

```mermaid
graph LR
    A[Open-Meteo API] -->|Clima| C(Async Worker)
    B[OpenAQ v3 API] -->|PM2.5| C
    C -->|Persistencia| D[(PostgreSQL / SQLite)]
    D -->|Último Snapshot| E[FastAPI Server]
    F[Scikit-Learn Pipeline] --> E
    E -->|JSON + Predicción| G[Cliente / Swagger UI]