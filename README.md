# gRPC ML Service

Минимальный gRPC‑сервис для инференса обученной модели c эндпоинтами `/health` и `/predict`. Репозиторий содержит готовую модель (`models/model.pkl`), Python‑реализацию сервера и клиента, а также Dockerfile для сборки образа.

## Стек и окружение
- Python 3.11 (для локального запуска достаточно 3.9+)
- gRPC (`grpcio`, `grpcio-tools`)
- Scikit-learn (dummy‑модель logistic regression)

## Переменные окружения
- `PORT` — порт сервера (по умолчанию `50051`)
- `MODEL_PATH` — путь к модели (по умолчанию `models/model.pkl`)
- `MODEL_VERSION` — версия модели (по умолчанию `v0.0.0`, в Docker — `v1.0.0`)
- `GRPC_SERVER` — адрес сервера для клиента (по умолчанию `localhost:50051`)
- `PREDICT_FEATURES` — список фич через запятую для клиента (по умолчанию `0.5,-1.2,0.3,1.1`)

## Локальный запуск
```bash
pip install -r requirements.txt
# (опционально) переобучить dummy-модель
py -3 scripts/train_dummy_model.py
# запустить сервер
py -3 -m server.server
# в новом терминале дернуть клиент
py -3 -m client.client
```

Ожидаемый вывод клиента:
```
Calling /health...
{"status": "ok", "modelVersion": "v0.0.0"}
Calling /predict...
{"prediction": "1", "confidence": 0.9, "modelVersion": "v0.0.0"}
```

## Проверка через grpcurl
```bash
grpcurl -plaintext localhost:50051 mlservice.v1.PredictionService/Health
grpcurl -plaintext -d "{\"features\":[0.5,-1.2,0.3,1.1]}" \
  localhost:50051 mlservice.v1.PredictionService/Predict
```

## Сборка и запуск Docker
```bash
docker build -t grpc-ml-service .
docker run -p 50051:50051 grpc-ml-service
# после старта проверяем
grpcurl -plaintext localhost:50051 mlservice.v1.PredictionService/Health
py -3 -m client.client  # или свой grpcurl
```

## Генерация gRPC stubs из proto
Файлы `protos/model_pb2.py` и `protos/model_pb2_grpc.py` уже сгенерированы. При изменении контракта перегенерируйте их:
```bash
py -3 -m grpc_tools.protoc -I=protos \
  --python_out=protos --grpc_python_out=protos protos/model.proto
```

## Структура
```
protos/          # model.proto + сгенерированные stubs
server/          # gRPC сервер (python -m server.server)
client/          # простой клиент для health/predict
models/          # готовая модель model.pkl
scripts/         # train_dummy_model.py для пересоздания модели
Dockerfile       # сборка slim-образа
```
