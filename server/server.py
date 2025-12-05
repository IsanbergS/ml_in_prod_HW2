import logging
import os
from concurrent import futures
from pathlib import Path
from typing import Sequence

import grpc
import joblib
import numpy as np

from protos import model_pb2, model_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("grpc-server")


def _load_model(model_path: Path):
    """Load a persisted sklearn model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    LOGGER.info("Loading model from %s", model_path)
    return joblib.load(model_path)


class PredictionService(model_pb2_grpc.PredictionServiceServicer):
    """Implements health and predict RPCs."""

    def __init__(self, model, model_version: str) -> None:
        self.model = model
        self.model_version = model_version

    def Health(self, request, context):
        return model_pb2.HealthResponse(status="ok", modelVersion=self.model_version)

    def Predict(self, request, context):
        features: Sequence[float] = request.features
        if not features:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("features must not be empty")
            return model_pb2.PredictResponse()

        try:
            expected_features = getattr(self.model, "n_features_in_", None)
            if expected_features is not None and len(features) != expected_features:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    f"expected {expected_features} features, got {len(features)}"
                )
                return model_pb2.PredictResponse()

            vector = np.asarray(features, dtype=float).reshape(1, -1)
            predictions = self.model.predict(vector)
            label = str(predictions[0])

            confidence = 1.0
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(vector)
                confidence = float(np.max(proba))
            elif hasattr(self.model, "decision_function"):
                score = float(self.model.decision_function(vector))
                confidence = float(1 / (1 + np.exp(-score)))

            return model_pb2.PredictResponse(
                prediction=label,
                confidence=confidence,
                modelVersion=self.model_version,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Prediction failed: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Failed to run prediction")
            return model_pb2.PredictResponse()


def serve() -> None:
    model_path = Path(os.getenv("MODEL_PATH", "models/model.pkl")).resolve()
    model_version = os.getenv("MODEL_VERSION", "v0.0.0")
    port = os.getenv("PORT", "50051")

    try:
        model = _load_model(model_path)
    except Exception:  # pylint: disable=broad-except
        LOGGER.exception("Unable to load model from %s", model_path)
        raise

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_PredictionServiceServicer_to_server(
        PredictionService(model, model_version), server
    )
    server.add_insecure_port(f"[::]:{port}")
    LOGGER.info("Starting gRPC server on port %s", port)
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down gRPC server...")
        server.stop(grace=None)


if __name__ == "__main__":
    serve()
