import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

print("Loading Master Dataset...")
df = pd.read_csv('master_features_dataset.csv')

# --- CLEANING: Keep only Clear emotions ---
df = df[(df['Valence'] <= 2) | (df['Valence'] >= 6)]
df = df[(df['Arousal'] <= 2) | (df['Arousal'] >= 6)]

# Convert labels to Binary
df['Valence_Binary'] = (df['Valence'] >= 6).astype(int)
df['Arousal_Binary'] = (df['Arousal'] >= 6).astype(int)

feature_cols = [c for c in df.columns if c not in ['Participant', 'Block', 'Trigger', 'Valence', 'Arousal', 'Quadrant', 'Valence_Binary', 'Arousal_Binary']]
df[feature_cols] = df[feature_cols].astype('float64')

groups = df['Participant'].values

# Standardize features
scaler = StandardScaler()
for p in np.unique(groups):
    mask = groups == p
    df.loc[mask, feature_cols] = scaler.fit_transform(df.loc[mask, feature_cols])

X = df[feature_cols].values

# --- NEW: Grid Search Helper Function ---
def get_best_params(X, y):
    print("  -> Running Grid Search (this may take a minute)...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='f1_macro')
    grid.fit(X, y)
    print(f"  -> Best Params Found: {grid.best_params_}")
    return grid.best_params_

# --- BINARY TRAINING FUNCTION WITH TUNING ---
def train_and_validate(X, y_binary, group_ids, target_name):
    print(f"\n{'='*20} {target_name} {'='*20}")
    
    # Feature Selection (Top 15)
    clf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_temp.fit(X, y_binary)
    top_indices = np.argsort(clf_temp.feature_importances_)[::-1][:15]
    X_sel = X[:, top_indices]
    
    # Tune hyperparameters
    best_params = get_best_params(X_sel, y_binary)
    
    # LOSO Cross-Validation
    loso = LeaveOneGroupOut()
    y_true, y_pred = [], []
    
    # Apply best params
    clf = RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)
    
    for train_idx, test_idx in loso.split(X_sel, y_binary, group_ids):
        clf.fit(X_sel[train_idx], y_binary[train_idx])
        y_true.extend(y_binary[test_idx])
        y_pred.extend(clf.predict(X_sel[test_idx]))
        
    print(classification_report(y_true, y_pred, target_names=['Low', 'High']))

# Execute
train_and_validate(X, df['Valence_Binary'].values, groups, "VALENCE")
train_and_validate(X, df['Arousal_Binary'].values, groups, "AROUSAL")