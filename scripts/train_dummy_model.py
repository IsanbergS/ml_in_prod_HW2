import joblib
import numpy as np
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


def train_and_save(path: str) -> None:
    """Train a tiny logistic regression model and persist it to disk."""
    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X, y)
    joblib.dump(model, path)


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "models" / "model.pkl"
    np.random.seed(42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_and_save(str(output_path))
    print(f"Saved model to {output_path}")
