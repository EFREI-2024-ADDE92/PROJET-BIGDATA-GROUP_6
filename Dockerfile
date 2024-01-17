FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app

COPY app.py .
COPY knn-model.joblib .

RUN pip install scikit-learn
RUN pip install joblib
RUN pip install fastapi
RUN pip install uvicorn

CMD ["python3", "app.py"]