# XAI-Based Intrusion Detection System (XAI-IDS)

A Django web application that detects network intrusions using a hybrid machine learning ensemble and explains its predictions using **SHAP** and **LIME** — making the model's decisions transparent and interpretable.

---

## What it does

- Trains a **Hybrid Voting Classifier** (Random Forest + MLP + XGBoost) on network traffic data
- Classifies traffic into 4 categories: **Normal, DDoS, Brute Force, Bot**
- Explains every prediction using **SHAP** (global feature importance) and **LIME** (local per-prediction explanation)
- Provides a **live network simulation dashboard** to test traffic in real time
- Shows confidence scores with attack-specific indicators for each detection

---

## Tech Stack

| Category | Tools |
|---|---|
| Backend | Django, Python |
| Machine Learning | scikit-learn (RandomForest, MLP, VotingClassifier), XGBoost, SMOTE |
| Explainability | SHAP, LIME |
| Data | CIC-IDS2017 dataset |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML, Bootstrap |

---

## Features

- **Model Training** — Upload a dataset, train the ensemble, and view accuracy metrics with confusion matrix
- **Single Prediction** — Input network traffic features manually and get an instant prediction with SHAP/LIME explanation
- **Auto Prediction** — Batch predict on an uploaded CSV file
- **Network Simulation** — Simulate live network traffic and watch the model classify it in real time
- **Admin Panel** — Manage registered users

---

## Project Structure

```
explainable_ai/
├── manage.py
├── README.md
├── .gitignore
│
├── explainable_ai/                         # Django project config
│   ├── settings.py
│   ├── urls.py
│   ├── views.py
│   ├── asgi.py
│   └── wsgi.py
│
├── users/                                  # Main app — ML logic & predictions
│   ├── views.py                            # Training, SHAP, LIME, prediction
│   ├── models.py                           # User registration model
│   ├── admin.py
│   ├── apps.py
│   ├── migrations/
│   │   └── 0001_initial.py
│   └── templatetags/
│       └── custom_filters.py
│
├── admins/                                 # Admin panel app
│   ├── views.py
│   ├── models.py
│   ├── admin.py
│   └── apps.py
│
├── templates/                              # All HTML templates
│   ├── base.html
│   ├── index.html
│   ├── AdminLogin.html
│   ├── UserLogin.html
│   ├── UserRegister.html
│   ├── admins/
│   │   ├── AdminHome.html
│   │   └── viewregisterusers.html
│   └── users/
│       ├── UserHome.html
│       ├── Training.html
│       ├── Prediction.html
│       ├── AutoPrediction.html
│       ├── NetworkSimulation.html
│       ├── UploadData.html
│       └── ViewDataset.html
│
└── media/                                  # Generated at runtime (not in repo)
    ├── rf_model.pkl
    ├── scaler.pkl
    ├── feature_cols.pkl
    ├── label_encoder.pkl
    ├── accuracy_results.pkl
    └── Balanced_IDS_Data.csv
```

---

## Setup & Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/XAI-IDS.git
cd XAI-IDS
```

**2. Install dependencies**                 # Needs some time to install all the dependencies 
```bash
pip install -r requirements.txt
```

**3. Run migrations**
```bash
python manage.py migrate
```

**4. Start the server**
```bash
python manage.py runserver
```

Then open `http://127.0.0.1:8000` in your browser.

> **Note:** The trained model files (`rf_model.pkl`, `scaler.pkl`, etc.) are not included in the repo due to file size. Use the Training page to train a fresh model after uploading the dataset.

---

## Reference

> Gaspar, P., Silva, J., & Silva, F. (2024). *Explainable AI for Intrusion Detection Systems: LIME and SHAP Applicability on Multi-Layer Perceptron.* IEEE Access.
