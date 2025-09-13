import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("fraud_detection_experiment")

DATA_PATH = "data/creditcard.csv"  # update to your dataset path


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def train():
    X, y = load_data()

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle imbalance with scale_pos_weight
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",  # precision-recall AUC is good for imbalance
        "scale_pos_weight": scale_pos_weight,
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 300,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    with mlflow.start_run():
        model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])

        # Predictions
        preds = model.predict(dval)
        pred_labels = (preds > 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_val, pred_labels)
        prec = precision_score(y_val, pred_labels, zero_division=0)
        rec = recall_score(y_val, pred_labels, zero_division=0)
        f1 = f1_score(y_val, pred_labels, zero_division=0)
        roc_auc = roc_auc_score(y_val, preds)
        pr_auc = average_precision_score(y_val, preds)

        # Log params + metrics
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "val_accuracy": float(acc),
                "val_precision": float(prec),
                "val_recall": float(rec),
                "val_f1": float(f1),
                "val_roc_auc": float(roc_auc),
                "val_pr_auc": float(pr_auc),
            }
        )

        # Log confusion matrix + classification report
        print("Confusion matrix:")
        print(confusion_matrix(y_val, pred_labels))
        print("\nClassification report:")
        print(classification_report(y_val, pred_labels, digits=4))

        # Save to MLflow + local artifact
        mlflow.xgboost.log_model(model, artifact_path="fraud-xgb-model")

        os.makedirs("src/model", exist_ok=True)
        model.save_model("src/model/fraud_xgb.json")

        print(
            f"\nMetrics:\n"
            f"Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, "
            f"F1={f1:.4f}, ROC AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}"
        )


if __name__ == "__main__":
    train()
