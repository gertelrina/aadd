FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY aadd ./aadd
COPY weights ./weights
COPY examples ./examples
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -e ".[app]"

EXPOSE 8501
# `docker run -p 8501:8501 aadd` -> UI; `docker run aadd aadd scan /data -o /data/report.csv` -> CLI
ENTRYPOINT ["aadd"]
CMD ["app", "--headless", "--port", "8501"]
