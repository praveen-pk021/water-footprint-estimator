# Water Footprint Estimator

A capstone-ready ML + Streamlit project to estimate household water footprint and provide actionable reduction suggestions.

## Project Structure

```text
water-footprint-estimator/
+-- data/
¦   +-- water_data.csv
+-- models/
¦   +-- model.pkl
+-- app.py
+-- data_generator.py
+-- train_model.py
+-- utils.py
+-- requirements.txt
+-- README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Pipeline

1. Generate dataset:

```bash
python data_generator.py
```

2. Train and save best model:

```bash
python train_model.py
```

3. Start Streamlit app:

```bash
streamlit run app.py
```

## What Makes This Capstone-Ready

- Synthetic dataset grounded on real-world water coefficients.
- Two-model training pipeline (Linear Regression + Random Forest), auto-selecting best by R2.
- Dashboard with:
  - Prediction and risk classification (Low / Moderate / High)
  - Water component breakdown chart
  - 30-day trend simulation
  - Practical recommendations
- Deployment-ready structure for Streamlit Cloud.

## Deploy on Streamlit Cloud

1. Push repository to GitHub.
2. Open https://share.streamlit.io.
3. Select repo and `app.py`.
4. Deploy.

## Viva Talking Points

- Why synthetic data was used (control + explainability).
- Why model comparison improves reliability.
- Meaning of MAE and R2.
- Impact of feature engineering on model quality.