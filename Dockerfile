# docker build -t finwise-api .   &&   docker run -p 8000:8000 -e WEB_CONCURRENCY=2 finwise-api
FROM python:3.10-slim
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY data_pipeline.py .
COPY serving ./serving
COPY artifacts/preprocessor.pkl artifacts/champion_model.pkl artifacts/champion_model_metadata.json ./artifacts/
ENV WEB_CONCURRENCY=2
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
# WEB_CONCURRENCY = number of worker processes. Rule of thumb: one per CPU core.
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port 8000"]
