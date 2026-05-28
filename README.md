# Churn MLOps Pipeline

An end-to-end machine learning and MLOps-style project for telecom customer churn prediction.

This project turns a customer churn prediction model into a production-like system with:

- Training pipeline
- Data preprocessing
- Model evaluation
- Saved model artifact
- FastAPI prediction service
- Swagger API documentation
- Streamlit frontend
- Docker support
- Prediction test with Pytest

---

## Project Overview

Customer churn prediction is a common business problem in telecom and subscription-based companies. The goal is to identify customers who are likely to leave so that the business can take retention actions such as discounts, service improvements, contract upgrades, or support outreach.

Unlike a basic notebook-based machine learning project, this project is structured as an MLOps-style system. It includes model training, model persistence, API inference, frontend prediction, testing, and Docker support.

---

## MLOps Architecture

```text
Raw Data
   ↓
Data Loading
   ↓
Preprocessing
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Saved Model Artifact
   ↓
FastAPI Prediction Service
   ↓
Streamlit Frontend
```

---

## Project Structure

```text
churn-mlops-pipeline/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── churn_model.joblib
│
├── results/
│   └── metrics.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   └── utils.py
│
├── tests/
│   └── test_prediction.py
│
├── .gitignore
├── Dockerfile
├── main.py
├── README.md
└── requirements.txt
```

---

## Dataset

The project uses the Maven Analytics Telecom Customer Churn dataset.

Raw dataset file:

```text
data/raw/telecom_customer_churn.csv
```

The raw dataset is excluded from GitHub using `.gitignore`.

The dataset includes customer demographics, service subscriptions, contract information, billing behavior, revenue variables, and customer status.

---

## Target Definition

The original target column is:

```text
Customer Status
```

It is converted into a binary target:

| Customer Status | Churn |
|---|---:|
| Churned | 1 |
| Stayed | 0 |
| Joined | 0 |

---

## Data Leakage Prevention

The following fields are removed before model training:

```text
Customer ID
Customer Status
Churn Category
Churn Reason
Latitude
Longitude
```

`Churn Category` and `Churn Reason` are removed because they are only available after a customer has already churned. Keeping them would create data leakage.

---

## Preprocessing

The preprocessing pipeline performs:

- Binary target creation
- Missing value handling
- Removal of ID and leakage columns
- Train/test split with stratification
- Categorical encoding through OneHotEncoder
- Numeric imputation and scaling
- Pipeline-based preprocessing for training and inference consistency

---

## Model

The model used in this project is:

```text
Gradient Boosting Classifier
```

The model is wrapped inside a scikit-learn Pipeline that includes:

- Numeric preprocessing
- Categorical preprocessing
- Gradient Boosting classifier

The trained model is saved as:

```text
models/churn_model.joblib
```

---

## Model Performance

The final model achieved:

| Metric | Value |
|---|---:|
| Accuracy | 0.8460 |
| Precision | 0.7357 |
| Recall | 0.6551 |
| F1-score | 0.6931 |
| ROC-AUC | 0.9115 |

Confusion Matrix:

```text
[[947, 88], [129, 245]]
```

Metrics are saved to:

```text
results/metrics.csv
```

---

## Training Pipeline

Run the training pipeline:

```bash
python main.py
```

This command:

1. Loads the raw churn dataset
2. Applies preprocessing
3. Splits the data into train and test sets
4. Trains the Gradient Boosting model
5. Evaluates model performance
6. Saves metrics
7. Saves the trained model artifact

Expected outputs:

```text
data/processed/processed_churn_data.csv
models/churn_model.joblib
results/metrics.csv
```

---

## FastAPI Prediction Service

The project includes a FastAPI service for real-time churn prediction.

Run the API:

```bash
uvicorn api.main:app --reload
```

Open the API root:

```text
http://127.0.0.1:8000
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Root endpoint

```http
GET /
```

Returns API status message.

### Health check

```http
GET /health
```

Returns API health status.

### Prediction endpoint

```http
POST /predict
```

Example response:

```json
{
  "churn_probability": 0.5153,
  "churn_probability_percent": "51.53%",
  "prediction": 1,
  "risk_label": "High Risk"
}
```

---

## Streamlit Frontend

The project also includes a Streamlit frontend for interactive churn risk prediction.

Run the app:

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app allows users to:

- Enter customer demographic information
- Select contract and service options
- Adjust monthly charges and tenure
- Predict churn risk
- View churn probability
- View business interpretation
- Inspect raw prediction output

Risk interpretation:

| Probability Range | Risk Level |
|---|---|
| Below 40% | Low Risk |
| 40% to 70% | Moderate Risk |
| 70% and above | High Risk |

---

## Testing

The project includes a simple prediction test using Pytest.

Run tests:

```bash
pytest
```

The test checks that the prediction output contains:

- `churn_probability`
- `churn_probability_percent`
- `prediction`
- `risk_label`

It also verifies that the churn probability is between 0 and 1 and that the prediction label is valid.

---

## Docker Support

This project includes a Dockerfile for containerizing the FastAPI service.

Build the Docker image:

```bash
docker build -t churn-mlops-pipeline .
```

Run the container:

```bash
docker run -p 8000:8000 churn-mlops-pipeline
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Zahra-ziaee/churn-mlops-pipeline.git
cd churn-mlops-pipeline
```

### 2. Create and activate virtual environment

Windows PowerShell:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add dataset

Place the dataset file here:

```text
data/raw/telecom_customer_churn.csv
```

### 5. Train the model

```bash
python main.py
```

### 6. Run the API

```bash
uvicorn api.main:app --reload
```

### 7. Run the Streamlit app

In another terminal:

```bash
streamlit run app/streamlit_app.py
```

### 8. Run tests

```bash
pytest
```

---

## Business Interpretation

The churn model can help telecom companies identify customers who are likely to churn and prioritize retention strategies.

Possible retention actions include:

- Offering discounts to high-risk customers
- Encouraging month-to-month customers to switch to longer contracts
- Providing premium tech support
- Offering online security services
- Targeting customers with low referrals or short tenure
- Creating customer engagement campaigns

---

## Current Status

Completed:

- Data loading
- Data preprocessing
- Leakage prevention
- Train/test split
- Gradient Boosting model training
- Model evaluation
- Metrics saving
- Model artifact saving with Joblib
- FastAPI prediction service
- Swagger API documentation
- Streamlit frontend
- Dockerfile
- Pytest prediction test
- Git/GitHub-ready project structure

Planned next steps:

- Add prediction logging
- Add batch prediction endpoint
- Add CI workflow with GitHub Actions
- Add model versioning
- Add monitoring dashboard
- Add deployment instructions for cloud hosting

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Pydantic
- Streamlit
- Pytest
- Docker
- Git
- GitHub

---

## Author

Zahra Ziaee

MSc Computer Science - Data Mining  
Focus: Machine Learning, Customer Analytics, MLOps, API-based ML Systems, and Business-Oriented Data Science