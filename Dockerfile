FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY aadd ./aadd
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir . \
 && python -c "from aadd.model import resolve_model_path as r; r('v3')"   # bake v3 into the image

# docker run -v /path/to/audio:/data aadd scan /data --model v3 --report /data/report.html
ENTRYPOINT ["aadd"]
CMD ["--help"]
