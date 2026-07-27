import os
import glob
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from autoreject import AutoReject
import warnings

# Suppress MNE warnings for cleaner console output
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

base_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/'

# Standard Unicorn 10-20 mapping
unicorn_mapping = {
    'EEG 1': 'Fz', 'EEG 2': 'C3', 'EEG 3': 'Cz', 'EEG 4': 'C4',
    'EEG 5': 'Pz', 'EEG 6': 'PO7', 'EEG 7': 'Oz', 'EEG 8': 'PO8'
}
montage = mne.channels.make_standard_montage('standard_1020')

print("Starting Master EEG Preprocessing Loop...")
print("-" * 50)

# Find all valid participant directories
participant_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('P') and os.path.isdir(os.path.join(base_path, d))])

for p_dir in participant_dirs:
    print(f"\n>>> Processing Participant: {p_dir}")
    full_p_dir = os.path.join(base_path, p_dir)
    
    # Handle folder naming variations
    exp_path = os.path.join(full_p_dir, 'Image Experiment')
    if not os.path.exists(exp_path):
        exp_path = os.path.join(full_p_dir, 'Image_Experiment')
        
    if not os.path.exists(exp_path):
        print(f"Skipping {p_dir}: No Image Experiment folder found.")
        continue

    # Create the safe graphs output directory
    graphs_dir = os.path.join(exp_path, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    
    # Get all BDF and CSV files
    bdf_files = sorted(glob.glob(os.path.join(exp_path, 'EEG', 'UnicornRecorder_*.bdf')))
    # Note: We sort to ensure CSVs align with their respective BDFs in split sessions
    csv_files = sorted(glob.glob(os.path.join(exp_path, 'emotion_ratings_*.csv')))
    
    if not bdf_files or not csv_files:
        print(f"Skipping {p_dir}: Missing BDF or CSV files.")
        continue

    # Handle multiple sessions/blocks per participant
    for session_idx, (bdf_path, csv_path) in enumerate(zip(bdf_files, csv_files)):
        print(f"  -> Loading Block {session_idx + 1}")
        
        try:
            # 1. LOAD DATA
            raw = mne.io.read_raw_bdf(bdf_path, preload=True)
            raw.rename_channels(unicorn_mapping)
            raw.set_montage(montage, match_case=False, on_missing='ignore')
            events = mne.find_events(raw, stim_channel='Status', verbose=False)
            
            # 2. RE-REFERENCING & FILTERING
            raw.set_eeg_reference(ref_channels='average', projection=False, verbose=False)
            iir_params = dict(order=4, ftype='butter', output='sos')
            raw.filter(l_freq=0.1, h_freq=40.0, method='iir', iir_params=iir_params, verbose=False)
            
            # 3. EPOCHING
            behavioral_data = pd.read_csv(csv_path)
            event_dict = {}
            for index, row in behavioral_data.iterrows():
                trigger = row['trigger_sent']
                if pd.isna(trigger): continue
                trigger = int(trigger)
                
                val = row['valence_rating']
                arou = row['arousal_rating']
                default_cat = str(row['category']).strip()
                
                if pd.isna(val) or pd.isna(arou):
                    dynamic_category = default_cat
                else:
                    v_label = "HV" if val > 4 else "LV"
                    a_label = "HA" if arou > 4 else "LA"
                    dynamic_category = f"{a_label}{v_label}"
                
                event_dict[f"{dynamic_category}/{trigger}"] = trigger

            epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=1.0, 
                                baseline=(-0.5, 0), preload=True, on_missing='ignore', verbose=False)
            
            if len(epochs) == 0:
                print(f"  -> WARNING: No matching epochs found for Block {session_idx + 1}. Skipping.")
                continue

            # 4. ICA ARTIFACT REMOVAL
            eeg_chans = list(unicorn_mapping.values())
            acc_chans = [ch for ch in epochs.ch_names if 'acc' in ch.lower() or 'gyr' in ch.lower() or 'ax' in ch.lower()]
            
            if acc_chans:
                epochs_eeg = epochs.copy().pick(eeg_chans)
                acc_data = epochs.copy().pick(acc_chans).get_data()
                
                ica = mne.preprocessing.ICA(n_components=len(eeg_chans), random_state=42, method='fastica')
                ica.fit(epochs_eeg, verbose=False)
                
                src_epochs = ica.get_sources(epochs_eeg)
                sources = src_epochs.get_data()
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
            ar = AutoReject(n_jobs=-1, random_state=42, verbose=False)
            epochs_clean, _ = ar.fit_transform(epochs, return_log=True)
            
            # 6. TIME-FREQUENCY ANALYSIS & GRAPH SAVING
            print("  -> Generating and saving Time-Frequency graphs...")
            freqs = np.logspace(*np.log10([4, 30]), num=20)
            n_cycles = freqs / 2.
            categories = ['HAHV', 'LALV', 'HALV', 'LAHV']
            
            for cat in categories:
                try:
                    power = epochs_clean[cat].compute_tfr(method="morlet", freqs=freqs, n_cycles=n_cycles, return_itc=False).average()
                    # Plot without showing the UI window
                    fig = power.plot(['Fz'], baseline=(-0.5, 0), mode='logratio', title=f'{cat} Power (Fz)', show=False)[0]
                    
                    # Save safely to the graphs folder
                    safe_filename = f"{p_dir}_Block{session_idx+1}_{cat}_TFR.png"
                    fig.savefig(os.path.join(graphs_dir, safe_filename))
                    plt.close(fig) # Free up memory!
                except KeyError:
                    pass # Ignore if category is missing in this block
                    
            print(f"  -> Block {session_idx + 1} complete. {len(epochs_clean)} clean epochs retained.")

        except Exception as e:
            print(f"  -> ERROR processing Block {session_idx + 1}: {e}")

print("\n" + "="*50)
print("Master Preprocessing Complete.")
print("Check the 'graphs' folders in each participant directory.")