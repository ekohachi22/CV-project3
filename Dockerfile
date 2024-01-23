FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim

ENV POETRY_VERSION=1.4.2
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update              && \
    apt-get upgrade -y          && \
    apt-get autoclean -y        && \
    apt-get autoremove -y       && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
ENV PORT=8000

COPY app .