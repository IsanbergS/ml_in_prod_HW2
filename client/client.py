import json
import os
from typing import List

import grpc

from protos import model_pb2, model_pb2_grpc


def call_health(stub: model_pb2_grpc.PredictionServiceStub):
    response = stub.Health(model_pb2.HealthRequest(), timeout=5)
    print(json.dumps({"status": response.status, "modelVersion": response.modelVersion}))


def call_predict(stub: model_pb2_grpc.PredictionServiceStub, features: List[float]):
    request = model_pb2.PredictRequest(features=features)
    response = stub.Predict(request, timeout=5)
    print(
        json.dumps(
            {
                "prediction": response.prediction,
                "confidence": round(response.confidence, 4),
                "modelVersion": response.modelVersion,
            }
        )
    )


def parse_features(env_value: str) -> List[float]:
    if not env_value:
        return [0.5, -1.2, 0.3, 1.1]
    return [float(item.strip()) for item in env_value.split(",") if item.strip()]


if __name__ == "__main__":
    target = os.getenv("GRPC_SERVER", "localhost:50051")
    features_env = os.getenv("PREDICT_FEATURES", "")
    features = parse_features(features_env)

    channel = grpc.insecure_channel(target)
    stub = model_pb2_grpc.PredictionServiceStub(channel)

    print("Calling /health...")
    call_health(stub)
    print("Calling /predict...")
    call_predict(stub, features)
