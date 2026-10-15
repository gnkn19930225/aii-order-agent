FROM python:3.12-slim

WORKDIR /app
# config.py 不打包進映像檔，部署時由 Secret Manager 掛載到 /config/config.py
ENV PYTHONUNBUFFERED=1 TZ=Asia/Taipei PYTHONPATH=/config

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# vectordb/ 要先在本機用 build_index.py 建好，會一起打包進映像檔
COPY . .

# Cloud Run 用 PORT 環境變數告訴程式要聽哪個 port
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
