PROJECT_NAME = credit-scoring-model
PYTHON_INTERPRETER = python
DOCKER_IMAGE = credit-scoring
DATA_URL = https://github.com/learn2pr0ge/credit-card-clients/releases/download/v1.0-data/UCI_Credit_Card.csv

## Установка зависимостей
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Скачиваем датасет в data/raw
.PHONY: data
data:
	mkdir -p data/raw
	wget -O data/raw/UCI_Credit_Card.csv $(DATA_URL)

## Подготовка данных
.PHONY: prepare
prepare:
	$(PYTHON_INTERPRETER) src/run_pipeline.py

## Обучение модели
.PHONY: train
train:
	$(PYTHON_INTERPRETER) src/models/train.py

## Прогоняем весь DVC pipeline
.PHONY: repro
repro:
	dvc repro

## Запуск тестов
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Запуск тестов с подсчетом покрытия
.PHONY: test-cov
test-cov:
	$(PYTHON_INTERPRETER) -m pytest --cov=src --cov-report=term --cov-report=html tests

## Проверка кода линтерами flake8, isort, black
.PHONY: lint
lint:
	flake8 src tests
	isort --check --diff src tests
	black --check src tests

## Форматирование кода через isort и black
.PHONY: format
format:
	isort src tests
	black src tests

## Локальный запуск FastAPI через uvicorn
.PHONY: serve
serve:
	uvicorn api_model:app --app-dir src/api --host 0.0.0.0 --port 8000

## Сборка Docker образа
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

## Запуск Docker контейнера
.PHONY: docker-run
docker-run:
	docker run --rm -p 8000:8000 $(DOCKER_IMAGE)

## Чистим скомпилированные файлы и артефакты тестов
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage coverage.xml


