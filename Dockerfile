FROM python:3.11-slim

# Crear usuario no-root (requerido por seguridad en Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copiar e instalar dependencias
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente, interfaz gráfica y modelo entrenado
COPY --chown=user app/ ./app/
COPY --chown=user model/ ./model/
COPY --chown=user static/ ./static/

# Hugging Face Spaces expone el puerto 7860 por defecto
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]