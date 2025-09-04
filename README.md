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
  - Performance metrics: precision, recall, F1-score (not just accuracy).

- **Deployment**
  - REST API with **FastAPI**.
  - Containerized using **Docker**.
  - Ready for cloud deployment (AWS/GCP/Render/Heroku).

- **MLOps Tools**
  - **MLflow** for experiment tracking and model registry.
  - Configurable hyperparameters via CLI or YAML.
  - CI/CD ready (GitHub Actions / Docker Hub integration).

---

## 🗂️ Project Structure
```
mlops-churn/
│── data/               # (not included in repo) dataset storage
│── notebooks/          # Jupyter notebooks for EDA
│── src/
│   ├── train.py        # Training script with MLflow tracking
│   ├── preprocess.py   # Data cleaning & feature engineering
│   ├── inference.py    # Prediction utilities
│   └── api.py          # FastAPI app for serving predictions
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
source .env/bin/activate  # Linux/Mac
.env\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Training the Model
```bash
python src/train.py
```

- Tracks experiments with MLflow  
- Logs metrics and model artifacts to `mlruns/`

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

### With Docker:
```bash
docker-compose up --build
```

The API will be available at:  
👉 `http://localhost:8000/docs`

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
- **GitHub Actions** (ready for CI/CD)

---

## 📦 Deployment Options
- **Local** with Docker + FastAPI
- **Cloud**:  
  - AWS (ECS/EKS, S3 for artifacts, RDS for DB)  
  - GCP (Cloud Run, Vertex AI)  
  - Render / Railway / Heroku (simple hosting option)

---

## 📌 Notes
- Dataset (`creditcard.csv`) is **not included** in the repo (too large).  
  You can download it from [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
  Place it inside `data/creditcard.csv` before training.
- Use `.gitignore` to avoid committing datasets or model artifacts.

---

## ✨ Deliverable
> **End-to-End MLOps Pipeline:**  
> Trained **fraud detection model** with ~99% ROC-AUC,  
> deployed via **FastAPI + Docker**,  
> tracked with **MLflow**,  
> and production-ready for cloud deployment.

---

## 📜 License
MIT License — free to use and modify.
