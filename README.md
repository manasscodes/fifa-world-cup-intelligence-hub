# ⚽ FIFA World Cup Intelligence Hub

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fifa-world-cup-intelligence-app.streamlit.app/)  
*A multi‑module World Cup analytics platform that predicts match outcomes, clusters players by style, and tracks live fan sentiment.*

---

## 📌 Features

### 1. Match Outcome Predictor (Live)
- Predicts **win / draw / loss** probabilities for any World Cup fixture.
- Uses a **Random Forest** model trained on **964 historical World Cup matches** (1930–2022).
- Features: team strength, goal difference, head‑to‑head records, home advantage.
- Interactive bar chart built with **Plotly**.

### 2. Player Performance Dashboard (Coming Soon)
- Explore Transfermarkt player stats.
- Cluster players by playing style using **K‑Means**.
- Radar charts and scatter plots for player comparison.

### 3. Real‑Time Sentiment Tracker (Coming Soon)
- Live Twitter sentiment analysis during matches.
- **Hugging Face Transformers** for emotion detection.
- **Plotly Dash** dashboard showing sentiment spikes at key match events.

---

## 🧠 Tech Stack

- **Languages:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit‑learn (Random Forest, K‑Means)
- **NLP:** Hugging Face Transformers
- **Visualisation:** Plotly, Seaborn, Matplotlib
- **Dashboard:** Streamlit, Plotly Dash
- **Deployment:** Streamlit Community Cloud

---

## 📂 Project Structure

fifa-world-cup-intelligence-hub/ ├── data/ # Raw datasets (not committed) ├── notebooks/ # Exploratory & prototyping notebooks │ └── 01_match_predictor_prototype.ipynb ├── src/ # Core ML module │ └── predictor.py ├── models/ # Serialised model & mappings ├── app/ # Streamlit app │ ├── main.py │ └── pages/ └── README.md

---

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/manas-kolaskar/fifa-world-cup-intelligence-hub.git
   cd fifa-world-cup-intelligence-hub

   ```
   
## Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Install dependencies
```
```bash
pip install -r requirements.txt
```

## Download the datasets

International Football Results (1872-2026)

Place results.csv in the data/ folder.

Run the app

```bash
streamlit run app/main.py
```

## 🧪 Model Details

Training data: 964 World Cup matches (1930–2022)

Algorithm: Random Forest with balanced class weights

Accuracy: ~70% (on a 20% hold‑out test set)

Features:

home_advantage

home_strength / away_strength (historical win rates)

home_gd / away_gd (average goal difference)

h2h_smart (head‑to‑head win ratio or strength‑based prior)

The model and all mappings are serialised with joblib in the models/ directory.

## 📊 Example Predictions

| Home Team | Away Team | Home Win | Draw | Away Win |
|---|---|---|---|---|
| Brazil | Germany | 45.2% | 27.1% | 27.7% |
| Argentina | Germany | 13.1% | 33.5% | 53.4% |
| Germany | Curaçao | 98.0% | 0.0% | 2.0% |

## 🛣 Roadmap

Match predictor model and Streamlit UI

Player performance clustering dashboard

Real‑time Twitter sentiment tracker

Add 2026 World Cup fixture predictions

Improve model with Elo ratings or recent form

## 🙋‍♂️ About This Project
Built as a capstone personal project by Manas Kolaskar — a data science enthusiast passionate about football and machine learning. Every component was designed from scratch using public datasets and free tools.
