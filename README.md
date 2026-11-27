# Churn Prediction ML

Machine Learning project for predicting customer churn using Python, Scikit-learn, FastAPI, and Docker.

## Overview

This project implements a Machine Learning solution to estimate the probability of customer churn in a telecommunications company.

The solution covers exploratory data analysis, preprocessing, model training, evaluation, model persistence, and predictions through a REST API.

## Tech Stack

- Python 3.14
- Pandas and NumPy
- Scikit-learn
- FastAPI and Pydantic
- Pytest
- Ruff
- Poetry
- Joblib

Docker integration is planned for a future development stage.

## Dataset

The project uses the Telco Customer Churn dataset, containing 7,043 customer records.

The target variable is `Churn`, indicating whether a customer canceled their service.

The dataset includes information about customer demographics, subscription services, contract types, payment methods, and charges.

The original dataset is not included in the repository.

## Machine Learning

The project implements a supervised binary classification pipeline using Logistic Regression.

The final model uses a classification threshold of 0.40.

### Final Model Evaluation

| Metric | Result |
|---|---:|
| Accuracy | 77.71% |
| Precision | 56.82% |
| Recall | 66.84% |
| F1-score | 61.43% |
| ROC-AUC | 84.21% |

These metrics were obtained from the held-out test dataset.

## Architecture

The prediction API follows a layered architecture:

`HTTP Request → FastAPI → Controller → Service → Workflow → ML Pipeline`

Responsibilities are separated into:

- **DTO:** Request and response validation.
- **Controller:** Coordinates prediction requests.
- **Service:** Handles the prediction operation.
- **Workflow:** Orchestrates model inference.
- **Inference:** Loads the persisted model and generates predictions.
- **Configuration:** Resolves the model artifact path.

Dependency injection is used to improve modularity and testability.

## Installation

Requirements:

- Python 3.14
- Poetry
- Trained model artifact

Install dependencies:

```bash
poetry install
```

## Model Configuration

By default, the API loads the model from:

```text
models/churn_logistic_regression.joblib
```

The model location can be customized using the `CHURN_MODEL_PATH` environment variable:

```bash
export CHURN_MODEL_PATH=/path/to/model.joblib
```

The trained model artifact is excluded from Git version control.

## Running the API

Start the development server:

```bash
PYTHONPATH=src poetry run uvicorn churn_predict_ml.api.main:app --reload
```

Access the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

`GET /health`

Returns:

```json
{
  "status": "ok"
}
```

### Customer Churn Prediction

`POST /api/v1/predict`

Example request:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.50,
  "TotalCharges": 1026.00
}
```

Example response:

```json
{
  "churn_probability": 0.7273789506004618,
  "churn_prediction": 1
}
```

The probability represents the estimated likelihood of churn. A prediction of `1` indicates that the probability meets or exceeds the model's classification threshold.

### HTTP Responses

| Status | Description |
|---|---|
| 200 | Prediction completed successfully |
| 422 | Invalid request data |
| 503 | Prediction model unavailable |
| 500 | Internal prediction error |

## Testing

Run the complete test suite:

```bash
PYTHONPATH=src poetry run pytest -q
```

Run code quality checks:

```bash
poetry run ruff check .
poetry run ruff format --check .
```

The current implementation has **116 passing tests**, including unit tests, API tests, error handling tests, and integration tests using the trained model.

## Project Status

Completed:

- Exploratory data analysis
- Data preprocessing
- Model training and comparison
- Hyperparameter tuning
- Model evaluation and threshold selection
- Model persistence
- FastAPI prediction endpoint
- Input validation and error handling
- Automated testing

Planned improvements:

- Docker containerization
- Deployment
- Additional operational improvements

## License

No license has been specified yet.