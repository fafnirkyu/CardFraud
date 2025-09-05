# Credit Card Fraud Detection — End-to-End MLOps Pipeline

This project is a **production-ready MLOps pipeline** built around a credit card fraud detection use case.  
It demonstrates how to take a real-world imbalanced dataset, train a model, and deploy it using modern MLOps tools.

---

## 📌 Features
- **Data Preprocessing & Engineering**
  - Handles class imbalance with **SMOTE** oversampling.
  - Scales features with `StandardScaler`.
  - Splits data into train/test sets with stratification.

- **Model Training**
  - Fraud detection model using **XGBoost**.
  - Experiment tracking via **MLflow**.
  - Full evaluation with **precision, recall, F1-score, ROC-AUC, PR-AUC**.

- **Deployment**
  - REST API with **FastAPI**.
  - Containerized using **Docker**.
  - Ready for cloud deployment (AWS/GCP/Render/Heroku).

- **MLOps Tools**
  - **MLflow** for experiment tracking and model registry.
  - Configurable hyperparameters via CLI or YAML.
  - CI/CD ready (GitHub Actions / Docker Hub integration).

---

## 📝 Project Structure
```
mlops-churn/
│── data/               # (not included in repo) dataset storage        # Jupyter notebooks for EDA
│── src/
│   ├── app/
        └── main.py # main script       
│   └── train.py   # Training script with MLflow tracking
│── Dockerfile          # Container setup
│── docker-compose.yml  # Local testing with MLflow + API
│── requirements.txt    # Python dependencies
│── .gitignore          # Ignored files (datasets, venv, artifacts)
│── README.md           # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/fafnirkyu/CardFraud.git
cd CardFraud
```

### 2. Create virtual environment
```bash
python -m venv .env
source .env/bin/activate  # Linux/Mac/SSH
.env\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model

From the **SSH terminal** (or VM console):

```bash
cd ~/CardFraud
source .env/bin/activate
python src/train.py
```

- Automatically tracks experiments with **MLflow**  
- Logs metrics and model artifacts to `mlruns/`

### Example Metrics (Credit Card Fraud Model)

| Metric       | Value    |
|-------------|---------|
| Accuracy    | 0.9995  |
| Precision   | 0.8737  |
| Recall      | 0.8469  |
| F1-Score    | 0.8601  |
| ROC AUC     | 0.9794  |
| PR AUC      | 0.8746  |

Run MLflow UI locally:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

---

## 🌐 Running the API

### Local (without Docker):
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### With Docker (locally or on AWS EC2):
```bash
docker-compose up --build -d
```

- **FastAPI endpoint:** `http://<VM_or_EC2_IP>:8000/docs`  
- **MLflow tracking UI:** `http://<VM_or_EC2_IP>:5000`

> ✅ AWS Deployment Example:  
> - EC2 instance (free tier) with Docker installed  
> - `docker-compose` runs API + MLflow  
> - Accessible from your public EC2 IP for demos or resume showcase

---

## 📊 Example API Request
```json
POST /predict
{
  "features": [0.1, -1.2, 0.34, ...]
}
```

Response:
```json
{
  "fraud_probability": 0.87,
  "prediction": 1
}
```

---

## 🧑‍💻 Tech Stack
- **Python 3.10**
- **FastAPI** — API deployment
- **XGBoost** — fraud detection model
- **MLflow** — experiment tracking
- **Docker** — containerization
- **AWS EC2 (optional)** — cloud deployment
- **GitHub Actions** — CI/CD ready

---

## 📦 Deployment Options
- **Local** with Docker + FastAPI + MLflow  
- **Cloud / Resume-ready showcase**:
  - AWS EC2 (Docker + docker-compose)
  - GCP Cloud Run / Vertex AI
  - Render / Railway / Heroku (quick deployment)

---

## 📌 Notes
- Dataset (`creditcard.csv`) is **not included** in the repo (too large).  
  Download from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
  Place it inside `data/creditcard.csv` before training.
- `.gitignore` prevents committing datasets or model artifacts.

---

## ✨ Deliverable
> **End-to-End MLOps Pipeline:**  
> Trained **fraud detection model** with:
> - Accuracy: 0.9995  
> - Precision: 0.8737  
> - Recall: 0.8469  
> - F1-Score: 0.8601  
> - ROC-AUC: 0.9794  
> - PR-AUC: 0.8746  
>
> Deployed via **FastAPI + Docker**, tracked with **MLflow**, and production-ready for cloud deployment (AWS, GCP, or Render).  

---

## 📜 License
MIT License — free to use and modify.

