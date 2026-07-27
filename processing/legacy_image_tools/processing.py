import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from autoreject import AutoReject
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pywt
import scipy.stats as stats
from scipy.signal import welch

print("Starting Fully Automated EEG Processing Pipeline...")

# ==========================================
# 1. LOAD DATA & UNICORN MONTAGE
# ==========================================
bdf_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/P12_2026-06-04/Image_Experiment/EEG/UnicornRecorder_04_06_2026_15_14_58.bdf'
raw = mne.io.read_raw_bdf(bdf_path, preload=True)

# NEW: Map the generic Unicorn channel names to the standard 10-20 physical locations
unicorn_mapping = {
    'EEG 1': 'Fz',
    'EEG 2': 'C3',
    'EEG 3': 'Cz',
    'EEG 4': 'C4',
    'EEG 5': 'Pz',
    'EEG 6': 'PO7',
    'EEG 7': 'Oz',
    'EEG 8': 'PO8'
}
raw.rename_channels(unicorn_mapping)

# Now MNE knows exactly where these electrodes sit on the 3D head model
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='ignore')

events = mne.find_events(raw, stim_channel='Status')
custom_scale = dict(eeg=77.45)

# RESTORED: Plot Raw
raw.plot(events=events, block=True, duration=10.0, scalings=custom_scale, title="1. Raw EEG (Close window to continue)")

# ==========================================
# 2. RE-REFERENCING (Average)
# ==========================================
print("Applying Average Reference...")
raw.set_eeg_reference(ref_channels='average', projection=False)

# ==========================================
# 3. FILTERING (4th Order Butterworth)
# ==========================================
print("Applying 4th Order Butterworth Bandpass Filter (0.1 - 40 Hz)...")
iir_params = dict(order=4, ftype='butter', output='sos')
raw_filtered = raw.copy().filter(
    l_freq=0.1, 
    h_freq=40.0, 
    method='iir', 
    iir_params=iir_params
)

# ==========================================
# 4. EPOCHING
# ==========================================
print("Creating Epochs...")
csv_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/P12_2026-06-04/Image_Experiment/emotion_ratings_15-14-53.csv'
behavioral_data = pd.read_csv(csv_path)

event_dict = {}

# Loop through every trial in your CSV
for index, row in behavioral_data.iterrows():
    trigger = row['trigger_sent']
    val = row['valence_rating']
    arou = row['arousal_rating']
    default_cat = str(row['category']).strip() # e.g., 'HAHV'
    
    # Skip if there's no trigger code at all
    if pd.isna(trigger):
        continue
        
    trigger = int(trigger)
    
    # --- NEW: Fallback Logic ---
    if pd.isna(val) or pd.isna(arou):
        # If ratings are missing, fallback to the pre-assigned category string
        dynamic_category = default_cat
    else:
        # Determine High vs Low based on a 1-7 scale (Midpoint is 4)
        v_label = "HV" if val > 4 else "LV"
        a_label = "HA" if arou > 4 else "LA"
        dynamic_category = f"{a_label}{v_label}"
    
    # Add it to MNE's event dictionary using the hierarchical "/" tag
    event_dict[f"{dynamic_category}/{trigger}"] = trigger

# Create the epochs using your new personalized classifications!
# Note: 'on_missing="ignore"' is still highly recommended here just in case the EEG hardware dropped a trigger!
epochs = mne.Epochs(raw_filtered, events, event_id=event_dict, tmin=-0.5, tmax=1.0, baseline=(-0.5, 0), preload=True, on_missing='ignore')

# ==========================================
# 5. MOTION ARTIFACT REMOVAL (ICA + ACCELEROMETER)
# ==========================================
print("\nPerforming ICA and Gyro/Accel Correlation...")
eeg_chans = list(unicorn_mapping.values()) 
acc_chans = [ch for ch in epochs.ch_names if 'acc' in ch.lower() or 'gyr' in ch.lower() or 'ax' in ch.lower()]

if not acc_chans:
    print("WARNING: Could not find Accelerometer channels. Skipping ICA correction.")
else:
    epochs_eeg = epochs.copy().pick(eeg_chans)
    acc_data = epochs.copy().pick(acc_chans).get_data() 
    
    ica = mne.preprocessing.ICA(n_components=len(eeg_chans), random_state=42, method='fastica')
    ica.fit(epochs_eeg)
    
    src_epochs = ica.get_sources(epochs_eeg)
    sources = src_epochs.get_data()
    
    n_epochs, n_comps, n_times = sources.shape
    n_acc = acc_data.shape[1]
    
    # Calculate average correlation across all epochs
    pearson_coeffs = np.zeros((n_epochs, n_comps, n_acc))
    for ep in range(n_epochs):
        for c in range(n_comps):
            for a in range(n_acc):
                r, _ = pearsonr(sources[ep, c, :], acc_data[ep, a, :])
                pearson_coeffs[ep, c, a] = np.abs(r)
                
    mean_r_per_comp = np.mean(pearson_coeffs, axis=0) 
    
    # NEW: Define a hard threshold for correlation
    CORR_THRESHOLD = 0.30 
    bad_comps = set()
    
    print("\n--- ICA Component Correlations with Accelerometer ---")
    for c in range(n_comps):
        max_r = np.max(mean_r_per_comp[c, :]) # Get highest correlation with any acc channel
        print(f"Component {c}: Max Correlation = {max_r:.3f}")
        if max_r > CORR_THRESHOLD:
            bad_comps.add(c)
                
    print(f"\nIdentified artifactual ICs (Threshold > {CORR_THRESHOLD}): {list(bad_comps)}")
    
    if bad_comps:
        ica.exclude = list(bad_comps)
        epochs_eeg = ica.apply(epochs_eeg)
        print("Successfully removed artifactual sources and reconstructed EEG.")
    
    epochs = epochs_eeg

# ==========================================
# 6. AUTOREJECT (Automated Micro-Artifact Cleaning)
# ==========================================
print("\nRunning AutoReject (This may take a minute or two)...")
ar = AutoReject(n_jobs=-1, random_state=42, verbose=False)
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

# RESTORED: Show exactly what AutoReject fixed
reject_log.plot('horizontal')
plt.show(block=True)

epochs = epochs_clean

# ==========================================
# 7. TIME-FREQUENCY FOR ALL CATEGORIES
# ==========================================
print("\nCalculating Time-Frequency power for all categories...")
freqs = np.logspace(*np.log10([4, 30]), num=20)
n_cycles = freqs / 2.

categories = ['HAHV', 'LALV', 'HALV', 'LAHV']

for cat in categories:
    try:
        power = epochs[cat].compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False).average()
        # RESTORED: Plotting Time-Frequency. Note we use 'Fz' instead of 'EEG 1' now!
        power.plot(['Fz'], baseline=(-0.5, 0), mode='logratio', title=f'{cat} Frequency Power (Fz)')
    except KeyError:
        print(f"--> Skipping {cat}: No triggers found for this category in this session.")

# RESTORED: Pause script so you can actually see the Time-Frequency graphs
plt.show(block=True)

# ==========================================
# 8. FEATURE EXTRACTION (Relative Band Power)
# ==========================================
print("\nExtracting features (Temporal, Frequency, and Wavelet Domains)...")

# --- Robust Map Generation with Proxy Values ---
valence_map = {}
arousal_map = {}

for index, row in behavioral_data.iterrows():
    trigger = row['trigger_sent']
    if pd.isna(trigger): continue
    trigger = int(trigger)
    
    val = row['valence_rating']
    arou = row['arousal_rating']
    cat = str(row['category']).strip()
    
    # If a rating is missing, parse the "HAHV" string to assign a proxy score (6 for High, 2 for Low)
    if pd.isna(val):
        val = 6.0 if 'HV' in cat else 2.0
    if pd.isna(arou):
        arou = 6.0 if 'HA' in cat else 2.0
        
    valence_map[trigger] = val
    arousal_map[trigger] = arou

sfreq = epochs.info['sfreq'] 

X = []      
y_val = []  
y_arou = [] 
y_quadrant = [] 
dropped_trials = 0

# Extract all data at once into memory for speed 
# Shape: (n_epochs, n_channels, n_times)
all_epoch_data = epochs.get_data()

for i, event in enumerate(epochs.events):
    trigger_code = event[2]
    val_rating = valence_map.get(trigger_code, np.nan)
    arou_rating = arousal_map.get(trigger_code, np.nan)
    
    if pd.isna(val_rating) or pd.isna(arou_rating):
        dropped_trials += 1
        continue 
        
    y_val.append(val_rating)
    y_arou.append(arou_rating)
    
    v_label = "HV" if val_rating > 4 else "LV"
    a_label = "HA" if arou_rating > 4 else "LA"
    y_quadrant.append(f"{a_label}{v_label}")
    
    # Get the raw voltage data for this specific epoch
    epoch_data = all_epoch_data[i]
    
    epoch_features = []
    
    for ch_idx in range(epoch_data.shape[0]): 
        ch_signal = epoch_data[ch_idx, :]
        
        # --- 1. TEMPORAL & STATISTICAL FEATURES ---
        mean_val = np.mean(ch_signal)
        var_val = np.var(ch_signal)
        skew_val = stats.skew(ch_signal)
        kurt_val = stats.kurtosis(ch_signal)
        ptp_val = np.ptp(ch_signal) # Peak-to-peak amplitude
        zcr = ((ch_signal[:-1] * ch_signal[1:]) < 0).sum() # Zero-crossing rate
        
        # Hjorth Parameters (Mobility & Complexity)
        dy = np.diff(ch_signal)
        ddy = np.diff(dy)
        var_y = np.var(ch_signal)
        var_dy = np.var(dy)
        var_ddy = np.var(ddy)
        mobility = np.sqrt(var_dy / var_y) if var_y > 0 else 0
        complexity = (np.sqrt(var_ddy / var_dy) / mobility) if mobility > 0 else 0
        
        # --- 2. FREQUENCY FEATURES ---
        # nperseg handles window size; bounded by signal length
        nperseg = int(sfreq) if len(ch_signal) >= sfreq else len(ch_signal)
        freqs_w, psd = welch(ch_signal, fs=sfreq, nperseg=nperseg)
        
        # Safely extract bands (fallback to 0 if band is missing)
        delta = np.mean(psd[(freqs_w >= 1) & (freqs_w < 4)]) if any((freqs_w >= 1) & (freqs_w < 4)) else 0
        theta = np.mean(psd[(freqs_w >= 4) & (freqs_w < 8)]) if any((freqs_w >= 4) & (freqs_w < 8)) else 0
        alpha = np.mean(psd[(freqs_w >= 8) & (freqs_w < 12)]) if any((freqs_w >= 8) & (freqs_w < 12)) else 0
        beta  = np.mean(psd[(freqs_w >= 12) & (freqs_w <= 30)]) if any((freqs_w >= 12) & (freqs_w <= 30)) else 0
        
        psd_norm = psd / (np.sum(psd) + 1e-12)
        spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        # --- 3. WAVELET FEATURES ---
        # Using Daubechies 4 wavelet. Level 5 effectively isolates 2-4 Hz and 4-8 Hz at a 250Hz sample rate.
        coeffs = pywt.wavedec(ch_signal, 'db4', level=5)
        cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
        
        # cA5 approximates 0-4 Hz (Delta). cD5 approximates 4-8 Hz (Theta).
        theta_wav_energy = np.sum(cD5**2)
        delta_wav_energy = np.sum(cA5**2)
        theta_wav_var = np.var(cD5)
        delta_wav_var = np.var(cA5)

        # Concatenate all features for this single channel (17 features total)
        ch_features = [
            mean_val, var_val, skew_val, kurt_val, ptp_val, zcr, 
            mobility, complexity, 
            delta, theta, alpha, beta, spec_entropy,
            theta_wav_energy, delta_wav_energy, theta_wav_var, delta_wav_var
        ]
        
        epoch_features.extend(ch_features)
            
    X.append(epoch_features)

X = np.array(X)
y_val = np.array(y_val)
y_arou = np.array(y_arou)
y_quadrant = np.array(y_quadrant)

print(f"Clean Feature matrix shape: {X.shape}")

# ==========================================
# 9. FEATURE SELECTION & CLASSIFICATION
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\nAssigning Feature Names...")
feature_types = [
    "Mean", "Var", "Skew", "Kurt", "PtP", "ZCR", 
    "Mobility", "Complexity", 
    "Delta_PSD", "Theta_PSD", "Alpha_PSD", "Beta_PSD", "Spec_Entropy",
    "Theta_Wav_Energy", "Delta_Wav_Energy", "Theta_Wav_Var", "Delta_Wav_Var"
]

# Generate the 136 column names based on the 8 channels and 17 features
feature_names = np.array([f"{ch}_{feat}" for ch in epochs.ch_names for feat in feature_types])

print("Training Initial AI Model to Evaluate Features...")

X_train, X_test, y_val_train, y_val_test, y_arou_train, y_arou_test, y_quad_train, y_quad_test = train_test_split(
    X, y_val, y_arou, y_quadrant, test_size=0.2, random_state=42, stratify=y_quadrant
)

# Train the initial classifier to map the data
quad_model = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
quad_model.fit(X_train, y_quad_train)

# --- NEW: Extract and Print Top 15 Features ---
importances = quad_model.feature_importances_
top_15_indices = np.argsort(importances)[::-1][:15]

print("\n" + "="*45)
print(" TOP 15 MOST IMPORTANT FEATURES")
print("="*45)
for i, idx in enumerate(top_15_indices):
    print(f"{i+1:2d}. {feature_names[idx]:<25} (Score: {importances[idx]:.4f})")
print("="*45)

# --- NEW: Re-train with ONLY the Top 15 Features ---
print("\nRe-training Model using ONLY the Top 15 Features...")
X_train_selected = X_train[:, top_15_indices]
X_test_selected = X_test[:, top_15_indices]

optimized_model = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
optimized_model.fit(X_train_selected, y_quad_train)

optimized_predictions = optimized_model.predict(X_test_selected)
optimized_accuracy = accuracy_score(y_quad_test, optimized_predictions)

print("\n--- Optimized Classification Results (Predicting Quadrant) ---")
print(f"Overall Accuracy: {optimized_accuracy * 100:.2f}%")
print("Detailed Report:")
print(classification_report(y_quad_test, optimized_predictions, zero_division=0))

# --- REGRESSION ---
val_model = RandomForestRegressor(n_estimators=100, random_state=42)
arou_model = RandomForestRegressor(n_estimators=100, random_state=42)

val_model.fit(X_train_selected, y_val_train)
arou_model.fit(X_train_selected, y_arou_train)

print("\n--- Optimized Regression Results (Predicting 1-7 Scale) ---")
print(f"Valence Mean Absolute Error: {mean_absolute_error(y_val_test, val_model.predict(X_test_selected)):.2f} points off")
print(f"Arousal Mean Absolute Error: {mean_absolute_error(y_arou_test, arou_model.predict(X_test_selected)):.2f} points off")