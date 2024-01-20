FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL -o /usr/bin/hadolint \
    https://github.com/hadolint/hadolint/releases/download/v2.7.0/hadolint-Linux-x86_64 && \
    chmod +x /usr/bin/hadolint

COPY app.py .
COPY knn-model.joblib .

RUN pip install scikit-learn
RUN pip install joblib
RUN pip install pydantic
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install prometheus-fastapi-instrumentator




CMD ["python3.8", "app.py"]

