# src/predictor.py
import os
import joblib
import pandas as pd
import numpy as np

# Build absolute paths based on this file's location
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_CURRENT_DIR, '..')
_MODELS_DIR = os.path.join(_PROJECT_ROOT, 'models')

_model_path = os.path.join(_MODELS_DIR, 'wc_predictor_v1.pkl')
_features_path = os.path.join(_MODELS_DIR, 'feature_columns.pkl')
_mappings_path = os.path.join(_MODELS_DIR, 'team_mappings.pkl')

# Load model and mappings
_model = joblib.load(_model_path)
_feature_cols = joblib.load(_features_path)
_mappings = joblib.load(_mappings_path)

_home_strength_map = _mappings['home_strength_map']
_away_strength_map = _mappings['away_strength_map']
_team_gd_dict = _mappings['team_gd_dict']
_h2h_records = _mappings['h2h_records']

DEBUT_HOME_WIN = 0.34
DEBUT_AWAY_WIN = 0.28
UNKNOWN_GD = -2.0

def expected_win_prob(home_gd, away_gd, scale=1.2):
    diff = home_gd - away_gd
    return 1 / (1 + np.exp(-scale * diff))

def get_h2h_smart(home, away, home_gd_val, away_gd_val):
    direct = _h2h_records.get((home, away))
    reverse = _h2h_records.get((away, home))
    if direct is not None:
        return direct
    elif reverse is not None:
        return 1 - reverse
    else:
        return expected_win_prob(home_gd_val, away_gd_val)

def predict_match(home_team, away_team, neutral=True):
    home_adv = 0 if neutral else 1
    home_str = _home_strength_map.get(home_team, DEBUT_HOME_WIN)
    away_str = _away_strength_map.get(away_team, DEBUT_AWAY_WIN)
    home_gd = _team_gd_dict.get(home_team, UNKNOWN_GD)
    away_gd = _team_gd_dict.get(away_team, UNKNOWN_GD)
    h2h = get_h2h_smart(home_team, away_team, home_gd, away_gd)

    X = pd.DataFrame([{
        'home_advantage': home_adv,
        'home_strength': home_str,
        'away_strength': away_str,
        'home_gd': home_gd,
        'away_gd': away_gd,
        'h2h_smart': h2h
    }], columns=_feature_cols)

    probs = _model.predict_proba(X)[0]
    return {cls: round(prob, 4) for cls, prob in zip(_model.classes_, probs)}
