import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

print("Loading Master Dataset...")
df = pd.read_csv('master_features_dataset.csv')

# --- NEW: FILTER OUT NEUTRAL TRIALS ---
# We keep only "Clear" emotions. 
# Ratings 1-2 = Low, 6-7 = High. We drop 3, 4, 5.
df = df[(df['Valence'] <= 2) | (df['Valence'] >= 6)]
df = df[(df['Arousal'] <= 2) | (df['Arousal'] >= 6)]

print(f"Trials remaining after removing neutral ratings: {len(df)}")

# 1. PREPARE DATA
feature_cols = [c for c in df.columns if c not in ['Participant', 'Block', 'Trigger', 'Valence', 'Arousal', 'Quadrant']]

# Fix the warning: Cast columns to float64 explicitly
df[feature_cols] = df[feature_cols].astype('float64')

X = df[feature_cols].values
y = df['Quadrant'].values
groups = df['Participant'].values

# 2. STANDARDIZE FEATURES PER PARTICIPANT
print("Standardizing features per participant...")
scaler = StandardScaler()
for p in np.unique(groups):
    mask = groups == p
    df.loc[mask, feature_cols] = scaler.fit_transform(df.loc[mask, feature_cols])

X = df[feature_cols].values

# 3. FEATURE SELECTION (Using the top 15 features)
clf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
clf_temp.fit(X, y)
importances = clf_temp.feature_importances_
top_15_indices = np.argsort(importances)[::-1][:15]
X_selected = X[:, top_15_indices]

# 4. LEAVE-ONE-SUBJECT-OUT (LOSO) CROSS-VALIDATION
loso = LeaveOneGroupOut()
y_true = []
y_pred = []

print("\nRunning LOSO Cross-Validation on CLEAR emotional states...")
clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)

for train_idx, test_idx in loso.split(X_selected, y, groups):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train
    clf.fit(X_train, y_train)
    # Predict
    y_true.extend(y_test)
    y_pred.extend(clf.predict(X_test))

print("\n--- LOSO Cross-Validation Report (Cleaned Data) ---")
print(classification_report(y_true, y_pred))