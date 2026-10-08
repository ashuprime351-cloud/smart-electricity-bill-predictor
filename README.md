# ⚡ Smart Electricity Bill Predictor

A clean, academic-style Streamlit web application that uses a **Gradient Boosting** machine-learning model to estimate your monthly electricity bill in Indian Rupees.

---

## 🚀 Quick Start

```bash
# 1 – Install dependencies
pip install -r requirements.txt

# 2 – Launch the app
streamlit run app.py
```

The app auto-generates the dataset and trains the model on the first run — no manual steps needed.

---

## 📁 Project Structure

```
├── app.py                  # Main Streamlit application
├── generate_data.py        # Synthetic dataset generator
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── data/
│   └── electricity_data.csv   # Auto-generated training data (2 000 rows)
└── model/
    ├── gb_model.pkl           # Trained GradientBoostingRegressor
    └── scaler.pkl             # StandardScaler for feature normalisation
```

---

## 🎛️ Input Features

| Feature | Description |
|---|---|
| Monthly Consumption (kWh) | Total units consumed this month |
| Number of People | Household size (1–10) |
| Daily Usage Hours | Average hours of electricity use per day |
| Previous Month (kWh) | Last month's consumption for comparison |

---

## 🤖 ML Model

| Property | Value |
|---|---|
| Algorithm | `GradientBoostingRegressor` (scikit-learn) |
| Training samples | 2 000 synthetic rows |
| Tariff basis | Approximate Indian domestic slab rates |
| Metrics | MAE & R² reported after training |

---

## 📊 App Features

- 🔮 **Instant bill prediction** in Indian Rupees
- 🟢🟡🔴 **Normal / Moderate / High** usage badge
- 📈 **Bar chart** comparing current vs previous month consumption
- 💚 **3 personalised energy-saving tips** based on usage level
- 📋 Input summary table
- ⚠️ Disclaimer about estimate accuracy

---

## ⚠️ Disclaimer

This tool provides an **AI-generated estimate** for educational purposes only. Actual electricity bills depend on your state's tariff structure, sanctioned load, fixed charges, taxes, and other levies. Consult your official electricity bill or DISCOM website for accurate charges.

---

## 🛠️ Requirements

- Python 3.9+
- streamlit, pandas, numpy, scikit-learn, matplotlib, joblib
