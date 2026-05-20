# app/main.py
import sys
import os

# Add the project root to sys.path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt

from src.predictor import predict_match, explain_prediction, get_explainer, FEATURE_NAMES

# -------------- Page config --------------
st.set_page_config(
    page_title="FIFA World Cup Intelligence Hub",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ FIFA World Cup Intelligence Hub")
st.markdown("### Match Outcome Predictor")

# -------------- Build absolute path to models/ --------------
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_mappings_path = os.path.join(_BASE_DIR, 'models', 'team_mappings.pkl')

# -------------- Load team mappings --------------
mappings = joblib.load(_mappings_path)
home_strength_map = mappings['home_strength_map']
away_strength_map = mappings['away_strength_map']
all_teams = sorted(set(home_strength_map.keys()) | set(away_strength_map.keys()))

# -------------- Sidebar: Team Selection --------------
st.sidebar.header("Select Teams")
home_team = st.sidebar.selectbox(
    "Home Team",
    all_teams,
    index=all_teams.index('Brazil') if 'Brazil' in all_teams else 0
)
away_team = st.sidebar.selectbox(
    "Away Team",
    all_teams,
    index=all_teams.index('Germany') if 'Germany' in all_teams else 1
)
neutral = st.sidebar.checkbox("Neutral Venue", value=True)
show_shap = st.sidebar.checkbox("Show SHAP explanation")

# -------------- Main area: Predict or show default --------------
if st.sidebar.button("Predict Outcome"):
    probs = predict_match(home_team, away_team, neutral=neutral)

    st.subheader(f"{home_team} vs {away_team}")
    st.write(f"Venue: {'Neutral' if neutral else 'Home advantage for ' + home_team}")

    prob_df = pd.DataFrame({
        'Outcome': ['Home Win', 'Draw', 'Away Win'],
        'Probability': [probs['home_win'], probs['draw'], probs['away_win']]
    })

    fig = px.bar(
        prob_df,
        x='Outcome',
        y='Probability',
        color='Outcome',
        color_discrete_map={'Home Win': '#1f77b4', 'Draw': '#7f7f7f', 'Away Win': '#d62728'},
        title="Predicted Outcome Probabilities",
        text_auto='.1%'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    max_outcome = max(probs, key=probs.get)
    emoji = {'home_win': '🏠', 'draw': '🤝', 'away_win': '✈️'}
    st.info(
        f"Most likely outcome: **{max_outcome.replace('_', ' ').title()}** "
        f"{emoji.get(max_outcome, '')} ({probs[max_outcome]:.1%})"
    )

    # -------------- SHAP Waterfall --------------
    if show_shap:
        st.subheader("🔍 Why this prediction?")
        st.markdown("How each feature pushed the prediction toward or away from this outcome:")

        shap_vals = explain_prediction(home_team, away_team, neutral=neutral)
        predicted_class = max_outcome  # 'home_win', 'draw', or 'away_win'

        explainer = get_explainer()
        class_index = {'home_win': 0, 'draw': 1, 'away_win': 2}

        exp = shap.Explanation(
            values=np.array(shap_vals[predicted_class]),
            base_values=explainer.expected_value[class_index[predicted_class]],
            feature_names=FEATURE_NAMES
        )

        fig_shap, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(exp, show=False)
        st.pyplot(fig_shap)
        plt.close(fig_shap)

else:
    # Default display before first prediction
    st.info("👈 Use the sidebar to select two teams and click **Predict Outcome** to see the probabilities.")
    st.markdown("""
    #### How it works:
    - A Random Forest model trained on **964 historical World Cup matches** (1930–2022) predicts the outcome.
    - Features include team strength, goal difference, head‑to‑head history, and venue.
    - Check **Show SHAP explanation** to see how each feature influenced the prediction.
    """)

    # Show an example chart for visual appeal
    placeholder_df = pd.DataFrame({
        'Outcome': ['Home Win', 'Draw', 'Away Win'],
        'Probability': [0.45, 0.25, 0.30]
    })
    fig = px.bar(
        placeholder_df,
        x='Outcome',
        y='Probability',
        color='Outcome',
        color_discrete_map={'Home Win': '#1f77b4', 'Draw': '#7f7f7f', 'Away Win': '#d62728'},
        title="Example: Typical World Cup outcome distribution",
        text_auto='.1%'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)