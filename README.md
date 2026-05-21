# ⚽ FIFA World Cup Intelligence Hub

<div align="center">

<img src="https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExdHZ0YXcycjJob2twdGZyNWZkdjZxZ3pxMGQzbnMxbnc2ZTVoMHlrdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/pXyZ8LcM7PrNgN3UtO/giphy.gif" width="50%" />

<h3>
AI • Football Analytics • Player Intelligence • Match Prediction
</h3>

<p>
A next-generation football analytics platform that combines machine learning, player clustering, interactive scouting dashboards, and real-time football intelligence.
</p>

<br>

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge\&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-111111?style=for-the-badge\&logo=plotly)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge\&logo=pandas)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/MIT-License-green?style=for-the-badge)]()

<br>

[🚀 Live Demo](https://fifa-world-cup-intelligence-app.streamlit.app/) •
[📊 Features](#-features) •
[🧠 ML Pipeline](#-machine-learning-pipeline) •
[⚙️ Installation](#️-installation) •
[📸 Screenshots](#-screenshots)

</div>

---

# 🌍 Overview

The **FIFA World Cup Intelligence Hub** is an AI-powered football analytics platform designed to simulate how modern football scouting and prediction systems work.

The project combines:

* ⚽ Match outcome prediction
* 🧠 Machine learning clustering
* 📊 Interactive visual analytics
* 🔍 Player intelligence dashboards
* 📈 Radar profile comparisons
* 🌐 Real-time football insights

Built entirely using public datasets, machine learning pipelines, and interactive data applications.

---

# ✨ Features

---

## 🏆 1. Match Outcome Predictor

Predicts:

* Home Win
* Draw
* Away Win

using a trained **Random Forest classifier** built on historical FIFA World Cup data.

### ✅ Model Inputs

* Historical win rate
* Goal difference
* Head-to-head records
* Team strength
* Home advantage

### 📈 Output

Interactive probability visualizations with Plotly.

```text
Brazil vs Germany

🏠 Home Win: 45.2%
🤝 Draw: 27.1%
✈️ Away Win: 27.7%
```

---

## 🧬 2. Player Intelligence Dashboard

A football scouting dashboard inspired by:

* EA FC
* FBRef
* Sofascore
* Transfermarkt

### 🚀 Current Capabilities

✅ Player search
✅ Side-by-side comparison
✅ Radar analytics
✅ Player portraits
✅ Archetype clustering
✅ PCA visualization
✅ Market value intelligence

### ⚡ Example Comparisons

* Messi vs Ronaldo
* Mbappé vs Haaland
* Pedri vs Bellingham

---

## 🧠 3. Machine Learning Player Clustering

Players are grouped into football archetypes using:

### 🔹 K-Means Clustering

The system automatically identifies:

* Attackers
* Defensive Anchors
* Elite Stars
* Utility Midfielders
* International Veterans
* Aggressive Defenders

### 🔹 PCA Dimensionality Reduction

Projects footballers into a 2D intelligence map for interactive exploration.

---

## 📡 4. Real-Time Sentiment Engine *(Coming Soon)*

Future module:

* Live Twitter/X sentiment analysis
* Match emotion tracking
* Goal reaction spikes
* NLP-powered fan mood detection

Using:

* Hugging Face Transformers
* Plotly Dash
* Twitter/X APIs

---

# 🧠 Machine Learning Pipeline

## ⚽ Match Predictor

| Component     | Details                    |
| ------------- | -------------------------- |
| Model         | Random Forest              |
| Training Data | 964 FIFA World Cup matches |
| Period        | 1930–2022                  |
| Accuracy      | ~70%                       |
| Framework     | Scikit-learn               |

---

## 🧬 Player Clustering Pipeline

### Feature Engineering

Players are transformed into advanced metrics:

```python
goals_per_90
assists_per_90
cards_per_90
minutes_per_match
international_caps
market_value_in_eur
```

### ML Workflow

```text
Raw Match Data
      ↓
Player Aggregation
      ↓
Feature Engineering
      ↓
Standard Scaling
      ↓
K-Means Clustering
      ↓
PCA Visualization
      ↓
Interactive Dashboard
```

---

# 📊 Tech Stack

## 💻 Core Technologies

| Category      | Stack                     |
| ------------- | ------------------------- |
| Language      | Python                    |
| Dashboard     | Streamlit                 |
| ML            | Scikit-learn              |
| Data          | Pandas, NumPy             |
| Visualization | Plotly, Matplotlib        |
| NLP           | Hugging Face Transformers |
| Deployment    | Streamlit Cloud           |

---

# 📂 Project Structure

```bash
fifa-world-cup-intelligence-hub/
│
├── app/
│   ├── main.py
│   └── pages/
│       └── 1_Player_Dashboard.py
│
├── data/
│   ├── results.csv
│   ├── players.csv
│   ├── appearances.csv
│   ├── games.csv
│   └── player_clusters.csv
│
├── models/
│   ├── worldcup_predictor.pkl
│   └── encoders.pkl
│
├── notebooks/
│   ├── 01_match_predictor_prototype.ipynb
│   └── 02_player_clustering.ipynb
│
├── src/
│   └── predictor.py
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/manasscodes/fifa-world-cup-intelligence-hub.git

cd fifa-world-cup-intelligence-hub
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
```

### macOS/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Streamlit App

```bash
streamlit run app/main.py
```

---

# 📸 Screenshots

## ⚽ Player Comparison Dashboard

```text
✅ Player portraits
✅ Radar analytics
✅ Archetype intelligence
✅ Interactive comparisons
```

---

## 📊 PCA Football Intelligence Map

```text
Clusters footballers by playstyle and performance profile.
```

---

# 🛣 Roadmap

## ✅ Completed

* Match prediction engine
* Streamlit application
* ML clustering pipeline
* Player comparison radar
* PCA visualization
* Archetype intelligence system

---

## 🚀 Upcoming

* FIFA-style stat cards
* Similar player recommendation engine
* Live sentiment tracker
* Elo rating integration
* Team chemistry analysis
* xG / xA advanced metrics
* Tactical formation intelligence

---

# 🧪 Example Predictions

| Fixture              | Home Win | Draw  | Away Win |
| -------------------- | -------- | ----- | -------- |
| Brazil vs Germany    | 45.2%    | 27.1% | 27.7%    |
| Argentina vs Germany | 13.1%    | 33.5% | 53.4%    |
| Germany vs Curaçao   | 98.0%    | 0.0%  | 2.0%     |

---

# 🌟 Why This Project Matters

Modern football clubs increasingly rely on:

* predictive analytics
* player intelligence systems
* machine learning scouting
* performance visualization

This project recreates core concepts used in professional football analytics environments using open data and accessible tooling.

---

# 👨‍💻 Author

## Manas Kolaskar

Aspiring AI/ML Engineer passionate about:

* football analytics
* machine learning
* data storytelling
* intelligent sports systems

### 🌐 Connect

* GitHub: [https://github.com/manasscodes](https://github.com/manasscodes)

---

# ⭐ Support

If you liked this project:

* ⭐ Star the repository
* 🍴 Fork the project
* 🧠 Share feedback
* ⚽ Connect with fellow football + AI enthusiasts

---

<div align="center">

# ⚽ Built With Data, Football, and Machine Learning

<img src="https://media.giphy.com/media/l0HlQ7LRalQqdWfao/giphy.gif" width="100%" />

</div>
