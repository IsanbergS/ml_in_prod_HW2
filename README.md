# gRPC ML Service

Minimal gRPC service with two endpoints: `/health` (status + model version) and `/predict` (class label + confidence). Uses a tiny scikit-learn logistic regression model stored at `models/model.pkl`. Dockerfile included for containerized run.

## Requirements
- Python 3.11+ (3.9+ should work)
- grpcio, grpcio-tools
- scikit-learn, numpy, joblib

## Environment variables
- `PORT` — gRPC port (default `50051`)
- `MODEL_PATH` — path to model file (default `models/model.pkl`)
- `MODEL_VERSION` — model version string (default `v0.0.0`, in Docker `v1.0.0`)
- `GRPC_SERVER` — client target, host:port (default `localhost:50051`)
- `PREDICT_FEATURES` — comma-separated features for client (default `0.5,-1.2,0.3,1.1`)

## Local run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |  Linux/mac: source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.train_dummy_model       # (optional) re-train model.pkl
python -m server.server                   # start gRPC server on PORT
# in another shell
python -m client.client                   # calls /health and /predict
```

Example output:
```
Calling /health...
{"status": "ok", "modelVersion": "v0.0.0"}
Calling /predict...
{"prediction": "1", "confidence": 0.9, "modelVersion": "v0.0.0"}
```

### grpcurl
```bash
grpcurl -plaintext localhost:50051 mlservice.v1.PredictionService/Health
grpcurl -plaintext -d "{\"features\":[0.5,-1.2,0.3,1.1]}" \
  localhost:50051 mlservice.v1.PredictionService/Predict
```

## Docker
```bash
docker build -t grpc-ml-service .
docker run -p 50051:50051 grpc-ml-service
# then call grpcurl or python -m client.client
```

## Regenerating stubs from proto
Generated files `protos/model_pb2.py` and `protos/model_pb2_grpc.py` are committed. To regenerate:
```bash
python -m grpc_tools.protoc -I=protos \
  --python_out=protos --grpc_python_out=protos protos/model.proto
```

## Project layout
```
protos/          # model.proto + generated stubs
server/          # gRPC server (python -m server.server)
client/          # simple client for health/predict
models/          # serialized model.pkl
scripts/         # train_dummy_model.py to rebuild model
Dockerfile       # slim image with server entrypoint
```
