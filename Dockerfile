FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

COPY app /workspace/app
COPY entrypoint.sh /workspace/entrypoint.sh

RUN chmod +x /workspace/entrypoint.sh

CMD ["/workspace/entrypoint.sh"]