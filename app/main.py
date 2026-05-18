# app/main.py
import sys
import os

# Add the project root to sys.path so we can import from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from src.predictor import predict_match

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
else:
    # Default display before first prediction
    st.info("👈 Use the sidebar to select two teams and click **Predict Outcome** to see the probabilities.")
    st.markdown("""
    #### How it works:
    - A Random Forest model trained on **964 historical World Cup matches** (1930–2022) predicts the outcome.
    - Features include team strength, goal difference, head‑to‑head history, and venue.
    - Probabilities are shown as a bar chart.
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