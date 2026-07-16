# Algerian Forest Fire Predictor

End-to-end ML lifecycle project predicting forest fire risk (FWI — Fire Weather Index) from weather and regional data, built on the Algerian Forest Fires dataset. Served as a Flask web app.

---

## What it does

A user enters weather/fire-index readings through a web form, and a trained Ridge Regression model predicts the Fire Weather Index (FWI) — a measure of forest fire risk.

**Input features:**
- Temperature
- RH (Relative Humidity)
- Ws (Wind speed)
- Rain
- FFMC (Fine Fuel Moisture Code)
- DMC (Duff Moisture Code)
- ISI (Initial Spread Index)
- Classes (fire / not fire, from the source dataset)
- Region

Inputs are scaled using a pre-fit `StandardScaler` before being passed to the Ridge Regression model.

---

## Tech stack

**Backend:** Flask
**ML:** scikit-learn (Ridge Regression, StandardScaler)
**Data handling:** pandas, numpy
**EDA/visualization:** matplotlib, seaborn
**Deployment:** Gunicorn, configured for platforms like Render (reads `PORT` from environment)

---

## Project structure

```
.
├── application.py                          # Flask app — home page + prediction route
├── models/
│   ├── ridge.pkl                             # Trained Ridge Regression model
│   └── scaler.pkl                             # Fitted StandardScaler
├── noteboooks/
│   ├── 2.0-EDA And FE Algerian Forest Fires.ipynb   # Exploratory data analysis + feature engineering
│   ├── 3.0-Model Training.ipynb                      # Model training and evaluation
│   └── Algerian_forest_fires_dataset_UPDATE.csv       # Source dataset
├── templates/
│   ├── index.html
│   └── home.html                              # Prediction form + results
└── requirements.txt
```

---

## How it works

1. `noteboooks/2.0-EDA And FE Algerian Forest Fires.ipynb` — exploratory analysis and feature engineering on the Algerian Forest Fires dataset.
2. `noteboooks/3.0-Model Training.ipynb` — trains the Ridge Regression model and fits the `StandardScaler`; both are pickled to `models/`.
3. `application.py` loads the pickled model and scaler at startup, exposes a form at `/`, and serves predictions at `/predictdata` (GET shows the form, POST returns a prediction).

---

## Setup

```bash
pip install -r requirements.txt
python application.py
```

App runs on `http://localhost:5000` by default (or the port set via the `PORT` environment variable, for deployment platforms like Render).

---

## Model performance

Ridge Regression (the model actually deployed in `application.py`), evaluated on the held-out test set:

- **R² Score:** 0.9843
- **MAE:** 0.564

Other models were compared in `3.0-Model Training.ipynb` (Linear Regression, Lasso, ElasticNet, and their cross-validated variants) — Ridge was selected as the final deployed model.
