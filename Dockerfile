FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app
ENV MINIMA_LOCAL_POLICY=chase

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY models/model.gguf ./model.gguf
COPY src/ ./src/

ENTRYPOINT ["python", "-m", "minima.main"]
