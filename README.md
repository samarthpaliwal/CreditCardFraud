# CreditCardFraud

It detects the chance of a credit card fraud in real time using ML. The project trains a logistic regression classifier on the public Kaggle credit card transactions dataset (284,807 anonymized transactions, 492 of them fraudulent), handles the heavy class imbalance by undersampling legitimate transactions, and exposes the trained model through a small Flask HTTP API that scores a single transaction as fraud or not.

## Requirements

- Python 3.8+
- `creditcard.csv` from the Kaggle "Credit Card Fraud Detection" dataset (not included; about 150 MB). Place it in the repository root.
- Python packages: `pandas`, `numpy`, `scikit-learn`, `joblib`, `flask`, and for the exploration script also `matplotlib` and `seaborn`

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install pandas numpy scikit-learn joblib flask matplotlib seaborn
# download creditcard.csv from Kaggle and put it next to the scripts
```

## Usage

### 1. Explore and evaluate (optional)

```bash
python fraud_detector.py
```

Prints the first rows, column info, missing-value count and class distribution, then shows a class-count bar chart. It scales `Amount` and `Time`, does a stratified 80/20 split, undersamples the non-fraud class in the training set to match the fraud count, trains a logistic regression, and prints a confusion matrix heatmap, a classification report (precision, recall, F1) and the ROC AUC score. Each chart opens in a matplotlib window; close it to continue.

### 2. Train and save the production model

```bash
python train_and_save.py
```

Fits a `StandardScaler` on `Amount` and `Time`, trains the balanced logistic regression on the full dataset, and writes two files: `data_scaler.joblib` and `fraud_model.joblib`. Console output shows the enforced column order and `Model training complete and saved.`

### 3. Serve predictions

```bash
python app.py
```

Starts Flask on `http://0.0.0.0:5000`. Send a JSON object with the raw transaction fields `Time`, `Amount` and `V1` through `V28`:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406, "Amount": 0.0, "V1": -2.31, "V2": 1.95, "V3": -1.61, "V4": 3.99, "V5": -0.52, "V6": -1.43, "V7": -2.54, "V8": 1.39, "V9": -2.77, "V10": -2.77, "V11": 3.20, "V12": -2.90, "V13": -0.60, "V14": -4.29, "V15": 0.39, "V16": -1.14, "V17": -2.83, "V18": -0.02, "V19": 0.42, "V20": 0.13, "V21": 0.52, "V22": -0.04, "V23": -0.47, "V24": 0.32, "V25": 0.04, "V26": 0.18, "V27": 0.26, "V28": -0.14}'
```

Response:

```json
{"is_fraud": true, "fraud_probability": "0.9xxx", "confidence": "9x.xx%"}
```

If `Time` or `Amount` is missing the API returns HTTP 400 with `{"error": "Missing Time or Amount in input data"}`; any other preprocessing failure (for example a missing `V` column) also returns 400 with the exception text.

## How it works

- Preprocessing: `Amount` and `Time` are standardized with a `StandardScaler`; the 28 PCA components `V1`..`V28` are used as-is. The final feature order is fixed to `['scaled_amount', 'scaled_time', 'V1', ..., 'V28']` in both training and serving so the model never sees shuffled columns.
- Imbalance: random undersampling keeps all 492 fraud rows and samples an equal number of legitimate rows (`random_state=42`), producing a balanced training set.
- Model: `sklearn.linear_model.LogisticRegression`. In `fraud_detector.py` it is evaluated on the untouched, imbalanced 20% test split; `train_and_save.py` trains the deployable copy on all data.
- Serving: `app.py` loads the saved scaler and model once at startup, rebuilds the exact feature frame from the request body, and returns the class prediction and probabilities.

Note that `train_and_save.py` fits the scaler on both columns jointly while `fraud_detector.py` fits two separate scalers; the saved scaler and `app.py` are consistent with each other, which is what matters for serving.

## Project layout

- `fraud_detector.py`: end-to-end exploration script with charts and evaluation metrics.
- `train_and_save.py`: trains the model and writes `data_scaler.joblib` and `fraud_model.joblib`.
- `app.py`: Flask API exposing `POST /predict`.
