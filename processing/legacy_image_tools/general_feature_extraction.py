import os
import glob
import mne
import pywt
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.signal import welch
from scipy.stats import pearsonr
from autoreject import AutoReject
import warnings

# Suppress warnings for a clean console
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

print("Starting Master Data Preprocessing & Feature Extraction...")
print("-" * 60)

base_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/'

unicorn_mapping = {
    'EEG 1': 'Fz', 'EEG 2': 'C3', 'EEG 3': 'Cz', 'EEG 4': 'C4',
    'EEG 5': 'Pz', 'EEG 6': 'PO7', 'EEG 7': 'Oz', 'EEG 8': 'PO8'
}
montage = mne.channels.make_standard_montage('standard_1020')
ch_names_ordered = list(unicorn_mapping.values())

# Setup column names for the final CSV
feature_types = [
    "Mean", "Var", "Skew", "Kurt", "PtP", "ZCR", 
    "Mobility", "Complexity", 
    "Delta_PSD", "Theta_PSD", "Alpha_PSD", "Beta_PSD", "Spec_Entropy",
    "Theta_Wav_Energy", "Delta_Wav_Energy", "Theta_Wav_Var", "Delta_Wav_Var"
]
feature_names = [f"{ch}_{feat}" for ch in ch_names_ordered for feat in feature_types]
all_columns = ['Participant', 'Block', 'Trigger', 'Valence', 'Arousal', 'Quadrant'] + feature_names

# Master list to hold all rows of data before saving
master_dataset = []

participant_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('P') and os.path.isdir(os.path.join(base_path, d))])

for p_dir in participant_dirs:
    full_p_dir = os.path.join(base_path, p_dir)
    exp_path = os.path.join(full_p_dir, 'Image Experiment')
    if not os.path.exists(exp_path):
        exp_path = os.path.join(full_p_dir, 'Image_Experiment')
        
    if not os.path.exists(exp_path): continue

    bdf_files = sorted(glob.glob(os.path.join(exp_path, 'EEG', 'UnicornRecorder_*.bdf')))
    csv_files = sorted(glob.glob(os.path.join(exp_path, 'emotion_ratings_*.csv')))
    
    if not bdf_files or not csv_files: continue

    print(f"\n>>> Extracting features for: {p_dir}")

    for session_idx, (bdf_path, csv_path) in enumerate(zip(bdf_files, csv_files)):
        print(f"  -> Processing Block {session_idx + 1}...")
        
        try:
            # 1. LOAD & PREPROCESS
            raw = mne.io.read_raw_bdf(bdf_path, preload=True)
            raw.rename_channels(unicorn_mapping)
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            events = mne.find_events(raw, stim_channel='Status', verbose=False)
            
            raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
            iir_params = dict(order=4, ftype='butter', output='sos')
            raw.filter(l_freq=0.1, h_freq=40.0, method='iir', iir_params=iir_params, verbose=False)
            
            # 2. BEHAVIORAL DATA FALLBACKS
            behavioral_data = pd.read_csv(csv_path)
            event_dict = {}
            valence_map, arousal_map, quad_map = {}, {}, {}
            
            for index, row in behavioral_data.iterrows():
                trigger = row['trigger_sent']
                if pd.isna(trigger): continue
                trigger = int(trigger)
                
                val = row['valence_rating']
                arou = row['arousal_rating']
                cat = str(row['category']).strip()
                
                # Robust Fallback
                if pd.isna(val): val = 6.0 if 'HV' in cat else 2.0
                if pd.isna(arou): arou = 6.0 if 'HA' in cat else 2.0
                    
                v_label = "HV" if val > 4 else "LV"
                a_label = "HA" if arou > 4 else "LA"
                dynamic_quad = f"{a_label}{v_label}"
                
                event_dict[f"{dynamic_quad}/{trigger}"] = trigger
                valence_map[trigger] = val
                arousal_map[trigger] = arou
                quad_map[trigger] = dynamic_quad

            # 3. EPOCHING (tmax set to 2.0 for Welch/Wavelet window)
            epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=2.0, 
                                baseline=(-0.5, 0), preload=True, on_missing='ignore', verbose=False)
            
            if len(epochs) == 0: continue

            # 4. ICA ARTIFACT REMOVAL
            eeg_chans = ch_names_ordered
            acc_chans = [ch for ch in epochs.ch_names if 'acc' in ch.lower() or 'gyr' in ch.lower() or 'ax' in ch.lower()]
            
            if acc_chans:
                epochs_eeg = epochs.copy().pick(eeg_chans)
                acc_data = epochs.copy().pick(acc_chans).get_data()
                
                ica = mne.preprocessing.ICA(n_components=len(eeg_chans), random_state=42, method='fastica')
                ica.fit(epochs_eeg, verbose=False)
                sources = ica.get_sources(epochs_eeg).get_data()
                
                n_epochs, n_comps, _ = sources.shape
                n_acc = acc_data.shape[1]
                pearson_coeffs = np.zeros((n_epochs, n_comps, n_acc))
                
                for ep in range(n_epochs):
                    for c in range(n_comps):
                        for a in range(n_acc):
                            r, _ = pearsonr(sources[ep, c, :], acc_data[ep, a, :])
                            pearson_coeffs[ep, c, a] = np.abs(r)
                            
                mean_r_per_comp = np.mean(pearson_coeffs, axis=0)
                bad_comps = [c for c in range(n_comps) if np.max(mean_r_per_comp[c, :]) > 0.30]
                
                if bad_comps:
                    ica.exclude = bad_comps
                    epochs = ica.apply(epochs_eeg, verbose=False)
                else:
                    epochs = epochs_eeg
            else:
                epochs = epochs.copy().pick(eeg_chans)
                
            # 5. AUTOREJECT
            print("  -> Running AutoReject...")
            ar = AutoReject(n_jobs=-1, random_state=42, verbose=False)
            epochs_clean, _ = ar.fit_transform(epochs, return_log=True)
            
            # 6. FEATURE EXTRACTION
            print("  -> Extracting 136 Multi-Domain Features...")
            sfreq = epochs_clean.info['sfreq'] 
            all_epoch_data = epochs_clean.get_data()
            
            for i, event in enumerate(epochs_clean.events):
                trigger_code = event[2]
                val_rating = valence_map.get(trigger_code, np.nan)
                arou_rating = arousal_map.get(trigger_code, np.nan)
                quad_rating = quad_map.get(trigger_code, "UNKNOWN")
                
                epoch_data = all_epoch_data[i]
                epoch_features = []
                
                for ch_idx in range(epoch_data.shape[0]): 
                    ch_signal = epoch_data[ch_idx, :]
                    
                    # 1. TEMPORAL
                    mean_val = np.mean(ch_signal)
                    var_val = np.var(ch_signal)
                    skew_val = stats.skew(ch_signal)
                    kurt_val = stats.kurtosis(ch_signal)
                    ptp_val = np.ptp(ch_signal) 
                    zcr = ((ch_signal[:-1] * ch_signal[1:]) < 0).sum() 
                    
                    dy = np.diff(ch_signal)
                    ddy = np.diff(dy)
                    var_y = np.var(ch_signal)
                    var_dy = np.var(dy)
                    var_ddy = np.var(ddy)
                    mobility = np.sqrt(var_dy / var_y) if var_y > 0 else 0
                    complexity = (np.sqrt(var_ddy / var_dy) / mobility) if mobility > 0 else 0
                    
                    # 2. FREQUENCY
                    nperseg = int(sfreq) if len(ch_signal) >= sfreq else len(ch_signal)
                    freqs_w, psd = welch(ch_signal, fs=sfreq, nperseg=nperseg)
                    
                    delta = np.mean(psd[(freqs_w >= 1) & (freqs_w < 4)]) if any((freqs_w >= 1) & (freqs_w < 4)) else 0
                    theta = np.mean(psd[(freqs_w >= 4) & (freqs_w < 8)]) if any((freqs_w >= 4) & (freqs_w < 8)) else 0
                    alpha = np.mean(psd[(freqs_w >= 8) & (freqs_w < 12)]) if any((freqs_w >= 8) & (freqs_w < 12)) else 0
                    beta  = np.mean(psd[(freqs_w >= 12) & (freqs_w <= 30)]) if any((freqs_w >= 12) & (freqs_w <= 30)) else 0
                    
                    psd_norm = psd / (np.sum(psd) + 1e-12)
                    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
                    
                    # 3. WAVELET
                    coeffs = pywt.wavedec(ch_signal, 'db4', level=5)
                    cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
                    
                    theta_wav_energy = np.sum(cD5**2)
                    delta_wav_energy = np.sum(cA5**2)
                    theta_wav_var = np.var(cD5)
                    delta_wav_var = np.var(cA5)

                    ch_features = [
                        mean_val, var_val, skew_val, kurt_val, ptp_val, zcr, 
                        mobility, complexity, 
                        delta, theta, alpha, beta, spec_entropy,
                        theta_wav_energy, delta_wav_energy, theta_wav_var, delta_wav_var
                    ]
                    epoch_features.extend(ch_features)
                
                # Create the full row: Meta-data + 136 Features
                row_data = [p_dir, session_idx + 1, trigger_code, val_rating, arou_rating, quad_rating] + epoch_features
                master_dataset.append(row_data)
                
            print(f"  -> Success. Appended {len(epochs_clean)} trials to master dataset.")

        except Exception as e:
            print(f"  -> ERROR processing Block {session_idx + 1}: {e}")

print("\n" + "="*60)
print("Processing Complete! Saving to CSV...")

# Convert the massive list of lists into a Pandas DataFrame and save it
final_df = pd.DataFrame(master_dataset, columns=all_columns)
final_df.to_csv('master_features_dataset.csv', index=False)

print(f"Successfully saved 'master_features_dataset.csv'.")
print(f"Final Dataset Shape: {final_df.shape[0]} rows (trials) x {final_df.shape[1]} columns (features/labels).")