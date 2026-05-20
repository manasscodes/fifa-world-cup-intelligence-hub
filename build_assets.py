# build_assets.py
import pandas as pd
import joblib

# Load the cleaned historical data you saved earlier
df = pd.read_pickle('data/historical_wc_clean.pkl')

# Recreate the feature columns in the exact order used by the model
feature_cols = [
    'home_advantage',
    'home_strength',
    'away_strength',
    'home_gd',
    'away_gd',
    'h2h_smart'
]

# Build X_train (the feature matrix used during training)
X_train = df[feature_cols]

# Save it
joblib.dump(X_train, 'models/X_train.pkl')
print("X_train saved to models/X_train.pkl")