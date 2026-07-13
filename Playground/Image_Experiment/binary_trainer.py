import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

print("Loading Master Dataset...")
df = pd.read_csv('master_features_dataset.csv')

# --- CLEANING: Remove neutral trials (3, 4, 5) ---
df = df[(df['Valence'] <= 2) | (df['Valence'] >= 6)]
df = df[(df['Arousal'] <= 2) | (df['Arousal'] >= 6)]

# Convert labels to Binary (0 = Low, 1 = High)
df['Valence_Binary'] = (df['Valence'] >= 6).astype(int)
df['Arousal_Binary'] = (df['Arousal'] >= 6).astype(int)

# Prepare Features
feature_cols = [c for c in df.columns if c not in ['Participant', 'Block', 'Trigger', 'Valence', 'Arousal', 'Quadrant', 'Valence_Binary', 'Arousal_Binary']]
df[feature_cols] = df[feature_cols].astype('float64')

groups = df['Participant'].values

# Standardize features per participant
print("Standardizing features per participant...")
scaler = StandardScaler()
for p in np.unique(groups):
    mask = groups == p
    df.loc[mask, feature_cols] = scaler.fit_transform(df.loc[mask, feature_cols])

X = df[feature_cols].values

# --- BINARY TRAINING FUNCTION ---
def run_loso(X, y_binary, group_ids, target_name):
    print(f"\n{'='*20} TRAINING: {target_name} {'='*20}")
    
    # Feature Selection for this specific task
    clf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_temp.fit(X, y_binary)
    top_indices = np.argsort(clf_temp.feature_importances_)[::-1][:15]
    X_sel = X[:, top_indices]
    
    loso = LeaveOneGroupOut()
    y_true, y_pred = [], []
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
    
    for train_idx, test_idx in loso.split(X_sel, y_binary, group_ids):
        clf.fit(X_sel[train_idx], y_binary[train_idx])
        y_true.extend(y_binary[test_idx])
        y_pred.extend(clf.predict(X_sel[test_idx]))
        
    print(classification_report(y_true, y_pred, target_names=['Low', 'High']))

# Run both models
run_loso(X, df['Valence_Binary'].values, groups, "VALENCE (Low vs High)")
run_loso(X, df['Arousal_Binary'].values, groups, "AROUSAL (Low vs High)")