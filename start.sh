#!/bin/bash
set -euo pipefail

if command -v wget >/dev/null 2>&1; then
  echo "wget установлен: $(wget --version | head -1)"
else
  echo "wget не установлен"
  sudo apt update && sudo apt install wget -y
fi

if command -v docker >/dev/null 2>&1; then
  echo "Docker установлен: $(docker --version | head -1)"
else
  echo "Docker не установлен"
  sudo apt update
  sudo apt install -y docker.io
  sudo systemctl enable docker --now
fi

echo "Идет скачивание файла данных"
mkdir -p data/raw
wget https://github.com/learn2pr0ge/credit-card-clients/releases/download/v1.0-data/UCI_Credit_Card.csv
cp UCI_Credit_Card.csv data/raw
rm UCI_Credit_Card.csv

echo "Идет скачивание файла модели"
wget https://github.com/learn2pr0ge/credit-card-clients/releases/download/v-1.0-model/model.pkl
cp model.pkl models/
rm model.pkl

docker build -t credit-scoring .
docker run --rm -p 80:8000 credit-scoring
