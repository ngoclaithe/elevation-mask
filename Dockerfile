FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY samples ./samples

ENV PYTHONUNBUFFERED=1 \
    ENABLE_FLORENCE=true \
    ENABLE_SAM=true \
    ENABLE_VL_CRITIC=false \
    JOB_DIR=/app/data/jobs

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
