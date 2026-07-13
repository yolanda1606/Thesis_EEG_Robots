import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from autoreject import AutoReject
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

print("Starting Fully Automated EEG Processing Pipeline...")

# ==========================================
# 1. LOAD DATA & UNICORN MONTAGE
# ==========================================
bdf_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/M01_2026-06-05/Image Experiment/EEG/UnicornRecorder_05_06_2026_09_46_29.bdf'
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
csv_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/M01_2026-06-05/Image Experiment/emotion_ratings_09-46-32.csv'
behavioral_data = pd.read_csv(csv_path)

event_dict = {}

# Loop through every trial in your CSV
for index, row in behavioral_data.iterrows():
    trigger = row['trigger_sent']
    val = row['valence_rating']
    arou = row['arousal_rating']
    
    # 1. Skip if the rating is missing (NaN)
    if pd.isna(val) or pd.isna(arou) or pd.isna(trigger):
        continue
        
    trigger = int(trigger)
    
    # 2. Determine High vs Low based on a 1-7 scale (Midpoint is 4)
    # We classify > 4 as High, and <= 4 as Low.
    if val > 4:
        v_label = "HV"
    else:
        v_label = "LV"
        
    if arou > 4:
        a_label = "HA"
    else:
        a_label = "LA"
        
    # 3. Smash them together (e.g., "HA" + "HV" = "HAHV")
    dynamic_category = f"{a_label}{v_label}"
    
    # 4. Add it to MNE's event dictionary using the hierarchical "/" tag
    event_dict[f"{dynamic_category}/{trigger}"] = trigger

# Create the epochs using your new personalized classifications!
epochs = mne.Epochs(raw_filtered, events, event_id=event_dict, tmin=-0.5, tmax=2.0, baseline=(-0.5, 0), preload=True)

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
print("\nExtracting features (Relative Band Power)...")

valence_map = dict(zip(behavioral_data['trigger_sent'], behavioral_data['valence_rating']))
arousal_map = dict(zip(behavioral_data['trigger_sent'], behavioral_data['arousal_rating']))

psd = epochs.compute_psd(method='welch', fmin=4, fmax=30, tmin=0.0, tmax=2.0)
data = psd.get_data()  
freqs = psd.freqs
bands = {'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30)}

X = []      
y_val = []  
y_arou = [] 
y_quadrant = [] 
dropped_trials = 0

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
    
    epoch_features = []
    for ch in range(data.shape[1]): 
        # Calculate TOTAL power for this channel to normalize
        total_power = np.mean(data[i, ch, :])
        
        for band_name, (fmin, fmax) in bands.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.mean(data[i, ch, idx])
            
            # NEW: Calculate RELATIVE power (percentage)
            relative_power = band_power / total_power
            epoch_features.append(relative_power)
            
    X.append(epoch_features)

X = np.array(X)
y_val = np.array(y_val)
y_arou = np.array(y_arou)
y_quadrant = np.array(y_quadrant)

print(f"Clean Feature matrix shape: {X.shape}")

# ==========================================
# 9. REGRESSION & CLASSIFICATION TRAINING
# ==========================================
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("\nTraining Valence and Arousal AI Models...")

X_train, X_test, y_val_train, y_val_test, y_arou_train, y_arou_test, y_quad_train, y_quad_test = train_test_split(
    X, y_val, y_arou, y_quadrant, test_size=0.2, random_state=42, stratify=y_quadrant # STRATIFY keeps class ratios equal
)

# --- 1. REGRESSION ---
val_model = RandomForestRegressor(n_estimators=100, random_state=42)
arou_model = RandomForestRegressor(n_estimators=100, random_state=42)

val_model.fit(X_train, y_val_train)
arou_model.fit(X_train, y_arou_train)

print("\n--- Regression Results (Predicting 1-7 Scale) ---")
print(f"Valence Mean Absolute Error: {mean_absolute_error(y_val_test, val_model.predict(X_test)):.2f} points off")
print(f"Arousal Mean Absolute Error: {mean_absolute_error(y_arou_test, arou_model.predict(X_test)):.2f} points off")

# --- 2. CLASSIFICATION (With Class Balancing) ---
# NEW: class_weight='balanced' forces the AI to respect small categories
quad_model = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42)
quad_model.fit(X_train, y_quad_train)

quad_predictions = quad_model.predict(X_test)
quad_accuracy = accuracy_score(y_quad_test, quad_predictions)

print("\n--- Classification Results (Predicting Quadrant) ---")
print(f"Overall Accuracy: {quad_accuracy * 100:.2f}%")
print("Detailed Report:")
print(classification_report(y_quad_test, quad_predictions, zero_division=0))