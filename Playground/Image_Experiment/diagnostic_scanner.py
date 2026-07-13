import os
import glob
import pandas as pd
import mne
import warnings

# Suppress MNE warnings for a cleaner console output
warnings.filterwarnings('ignore')
mne.set_log_level('WARNING')

base_path = '/home/sysgen/Projects/Yolanda/Thesis_EEG_Robots/data/'

print(f"{'Participant':<15} | {'CSVs':<5} | {'BDFs':<5} | {'CSV Triggers':<15} | {'EEG Triggers':<15} | {'Usable Epochs':<15}")
print("-" * 85)

total_usable_epochs = 0
participants_scanned = 0
participants_with_errors = []

# Find all participant directories starting with 'P'
participant_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('P') and os.path.isdir(os.path.join(base_path, d))])

for p_dir in participant_dirs:
    full_p_dir = os.path.join(base_path, p_dir)
    
    # 1. Handle folder naming inconsistencies
    exp_path = os.path.join(full_p_dir, 'Image Experiment')
    if not os.path.exists(exp_path):
        exp_path = os.path.join(full_p_dir, 'Image_Experiment')
        
    if not os.path.exists(exp_path):
        print(f"{p_dir:<15} | {'--':<5} | {'--':<5} | {'Missing Experiment Folder':<45}")
        participants_with_errors.append(p_dir)
        continue
        
    # 2. Find CSV and BDF files
    # Exclude .psydat and look specifically for emotion_ratings
    csv_files = glob.glob(os.path.join(exp_path, 'emotion_ratings_*.csv'))
    
    # Look for UnicornRecorder (ignoring UnicornRawDataRecorder if present)
    bdf_files = glob.glob(os.path.join(exp_path, 'EEG', 'UnicornRecorder_*.bdf'))
    
    num_csvs = len(csv_files)
    num_bdfs = len(bdf_files)
    
    if num_csvs == 0 or num_bdfs == 0:
        print(f"{p_dir:<15} | {num_csvs:<5} | {num_bdfs:<5} | {'Missing required CSV or BDF files':<45}")
        participants_with_errors.append(p_dir)
        continue

    # 3. Calculate Epoch Overlap for the main files (or combined if multiple)
    csv_triggers = set()
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'trigger_sent' in df.columns:
                # Extract valid triggers
                triggers = df['trigger_sent'].dropna().astype(int).tolist()
                csv_triggers.update(triggers)
        except Exception as e:
            pass
            
    eeg_triggers = set()
    for bdf_file in bdf_files:
        try:
            # preload=False keeps it fast by only reading the header and event channel
            raw = mne.io.read_raw_bdf(bdf_file, preload=False)
            events = mne.find_events(raw, stim_channel='Status', verbose=False)
            eeg_triggers.update(events[:, 2])
        except Exception as e:
            pass
            
    # The usable epochs are the exact intersection of what the CSV expected and what the EEG recorded
    usable_epochs = csv_triggers.intersection(eeg_triggers)
    count_usable = len(usable_epochs)
    
    total_usable_epochs += count_usable
    participants_scanned += 1
    
    print(f"{p_dir:<15} | {num_csvs:<5} | {num_bdfs:<5} | {len(csv_triggers):<15} | {len(eeg_triggers):<15} | {count_usable:<15}")

print("-" * 85)
print(f"Total Participants Scanned: {participants_scanned}")
print(f"Total Usable Epochs Across Dataset: {total_usable_epochs}")

if participants_with_errors:
    print(f"\nWARNING: The following participants have missing data/folders: {', '.join(participants_with_errors)}")