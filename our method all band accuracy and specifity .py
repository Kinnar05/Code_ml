import os
import glob
import numpy as np
import pandas as pd
import mne
import pywt
from scipy import signal
from scipy.stats import ttest_ind
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

# ==========================================
# 1. CONFIGURATION & PARAMETERS
# ==========================================
DATA_PATH = '/kaggle/input/eeg-dataset/'
# 19 Key Channels from the paper
CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
            'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
SFREQ = 256        # Original Sampling frequency
TARGET_SFREQ = 100 # Downsample to 100Hz 
EPOCH_LEN = 30.0    # User Request: 5 seconds
OVERLAP = 0.5      # User Request: 50% overlap
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def get_subject_info(filename):
    """
    Parses filename to extract Group (MDD/H) and ID.
    """
    base = os.path.basename(filename)
    parts = base.split(' ')
    # parts[0] is 'MDD' or 'H', parts[1] is 'Sxx'
    group_str = parts[0]
    subject_id = f"{parts[0]}_{parts[1]}"
    label = 1 if 'MDD' in group_str else 0
    return subject_id, label

def preprocess_subject(raw, target_channels, target_fs):
    """
    Applies filtering and normalization per subject.
    """
    # 1. Select Channels
available_chs = raw.ch_names
    picks = []
    for target in target_channels:
        for ch in available_chs:
            clean_ch = ch.replace('EEG ', '').replace('-LE', '').strip()
            if clean_ch.lower() == target.lower():
                picks.append(ch)
                break
    
    if len(picks) != len(target_channels):
        # Channel mismatch, skip this file
        return None
        
    raw.pick_channels(picks)
    
    # 2. Bandpass Filter (0.1 - 70 Hz)
    raw.filter(0.1, 70.0, fir_design='firwin')
    
    # 3. Notch Filter (50 Hz)
    raw.notch_filter(50.0, fir_design='firwin')
    
    # 4. Resample (100 Hz)
    raw.resample(target_fs)
    
    # 5. Z-score Normalization (Per recording)
    data = raw.get_data() # (n_ch, n_times)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data = (data - mean) / (std + 1e-8)
    
    return data

def load_dataset(path):
    print("Searching for EC (Eyes Closed) files...")
    # Filter for EC files only as requested
    files = glob.glob(os.path.join(path, '*EC.edf'))
    
    processed_records = []
    durations = []
    
    print(f"Found {len(files)} EC files. Loading and preprocessing...")
    
    for f in files:
        try:
            raw = mne.io.read_raw_edf(f, preload=True)
            sid, label = get_subject_info(f)
            
            data = preprocess_subject(raw, CHANNELS, TARGET_SFREQ)
            
            if data is not None:
                duration_sec = data.shape[1] / TARGET_SFREQ
                processed_records.append({
                    'data': data,
                    'id': sid,
                    'label': label,
                    'samples': data.shape[1]
                })
                durations.append(duration_sec)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    # Trim to minimum duration
    if not durations:
        raise ValueError("No valid data found.")
min_samples = int(min(durations) * TARGET_SFREQ)
    print(f"Trimming all files to minimum duration: {min(durations):.2f}s ({min_samples} samples)")
    
    # Create Epochs
    # Window: 5s, Stride: 2.5s (50% overlap)
    window_samples = int(EPOCH_LEN * TARGET_SFREQ)
    stride_samples = int(window_samples * (1 - OVERLAP))
    
    all_epochs = []
    all_labels = []
    all_groups = []
    
    for record in processed_records:
        # Crop data to min_samples
        data = record['data'][:, :min_samples]
        
        # Sliding window
        n_cols = data.shape[1]
        for start in range(0, n_cols - window_samples + 1, stride_samples):
            end = start + window_samples
            epoch = data[:, start:end]
            
            all_epochs.append(epoch)
            all_labels.append(record['label'])
            all_groups.append(record['id'])
            
    return np.array(all_epochs), np.array(all_labels), np.array(all_groups)
# ==========================================
# 3. WAVELET COHERENCE (CUSTOM)
# ==========================================
def compute_wavelet_coherence(epochs, fs):
    """
    Computes Wavelet Coherence Connectivity Matrix using PyWavelets.
    Returns: (n_epochs, n_bands, n_pairs)
    """
    n_epochs, n_ch, n_times = epochs.shape
    n_pairs = (n_ch * (n_ch - 1)) // 2
    n_bands = len(BANDS)
    
    # 1. Define Scales for Morlet Wavelet
    # We want frequencies 0.5 to 30 Hz
    freqs = np.arange(0.5, 30.5, 0.5)
    wavelet = 'cmor1.5-1.0' 
    scales = pywt.frequency2scale(wavelet, freqs / fs)
    
    features = np.zeros((n_epochs, n_bands, n_pairs))
    
    print(f"Computing Wavelet Coherence for {n_epochs} epochs...")
    
    # Define smoothing window (approx 1 sec)
    win_len = int(1.0 * fs)
    window = np.ones(win_len) / win_len
    
    for i in range(n_epochs):
        data = epochs[i]
        
        # Calculate CWT for all channels
        cwt_list = []
        for ch in range(n_ch):
            # CWT returns (coefficients, frequencies)
            coefs, _ = pywt.cwt(data[ch], scales, wavelet, sampling_period=1/fs)
            cwt_list.append(coefs)
        cwt_all = np.array(cwt_list) # Shape: (n_ch, n_freqs, n_times)
        
        pair_idx = 0
        for ch1 in range(n_ch):
            for ch2 in range(ch1 + 1, n_ch):
                
                # Cross Spectrum and Power Spectra
                # Using np.conjugate() to fix the previous AttributeError
                Sxy = cwt_all[ch1] * np.conjugate(cwt_all[ch2])
                Sxx = np.abs(cwt_all[ch1]) ** 2
                Syy = np.abs(cwt_all[ch2]) ** 2
                
                # Smooth in Time using 1D convolution
                Sxy_smooth = signal.convolve(Sxy, window[None, :], mode='same')
                Sxx_smooth = signal.convolve(Sxx, window[None, :], mode='same')
                Syy_smooth = signal.convolve(Syy, window[None, :], mode='same')
                
                # Coherence Squared
                R2 = (np.abs(Sxy_smooth) ** 2) / (Sxx_smooth * Syy_smooth + 1e-10)
                
                # Average within Bands
                for b, (b_name, (f_min, f_max)) in enumerate(BANDS.items()):
                    f_mask = (freqs >= f_min) & (freqs <= f_max)
                    if np.any(f_mask):
                        # Mean over frequency band AND time
                        val = np.mean(R2[f_mask, :])
                        features[i, b, pair_idx] = val
                
                pair_idx += 1
                
        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{n_epochs} epochs")
            
    return features

# ==========================================
# 4. CROSS-VALIDATION & EVALUATION
# ==========================================
def run_classification_pipeline():
    # A. Load & Preprocess
    X_epochs, y, groups = load_dataset(DATA_PATH)
    print(f"Total Epochs: {X_epochs.shape[0]}, Total Subjects: {len(np.unique(groups))}")
    
    # B. Extract Features
    X_features = compute_wavelet_coherence(X_epochs, TARGET_SFREQ)
    
    results = {}
    
    # C. Loop over Frequency Bands
    for b_idx, band in enumerate(BANDS.keys()):
        print(f"\nEvaluating Band: {band}")
        
        X_band = X_features[:, b_idx, :]
        
        acc_scores = []
        spec_scores = []
        
        # Stratified Group K-Fold (5 Splits)
        sgkf = StratifiedGroupKFold(n_splits=5)
        
        for train_idx, test_idx in sgkf.split(X_band, y, groups):
            X_train, X_test = X_band[train_idx], X_band[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # --- Feature Selection (T-Test) on TRAINING set only ---
            p_values = []
            for col in range(X_train.shape[1]):
                group0 = X_train[y_train == 0, col] # HC
                group1 = X_train[y_train == 1, col] # MDD
                _, p = ttest_ind(group0, group1, equal_var=False)
                p_values.append(p)
            
            p_values = np.array(p_values)
            
            # Select features with p < 0.05
            mask = p_values < 0.05
            
            if np.sum(mask) == 0:
                # Fallback to ensure classification runs
                mask = np.ones(X_train.shape[1], dtype=bool) 
                
            X_train_sel = X_train[:, mask]
            X_test_sel = X_test[:, mask]
            
            # --- SVM Classification ---
            scaler = StandardScaler()
            X_train_sel = scaler.fit_transform(X_train_sel)
            X_test_sel = scaler.transform(X_test_sel)
            
            clf = SVC(kernel='linear', class_weight='balanced', random_state=42)
            clf.fit(X_train_sel, y_train)
            y_pred = clf.predict(X_test_sel)
            
            # Metrics (Accuracy and Specificity)
            acc = accuracy_score(y_test, y_pred)
            
            # Confusion matrix: (TN, FP, FN, TP)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            acc_scores.append(acc)
            spec_scores.append(spec)
            
        results[band] = {
            'acc_mean': np.mean(acc_scores),
            'acc_std': np.std(acc_scores),
            'spec_mean': np.mean(spec_scores),
            'spec_std': np.std(spec_scores)
        }
        
    return results
# ==========================================
# 5. EXECUTION & OUTPUT
# ==========================================
if __name__ == "__main__":
    final_results = run_classification_pipeline()
    
    print("\n" + "="*50)
    print("TABLE I: Accuracy Reproduction (Proposed Method)")
    print("="*50)
    print(f"{'Band':<10} | {'Accuracy (Mean ± Std)'}")
    print("-" * 50)
    for band in BANDS.keys():
        r = final_results[band]
        print(f"{band:<10} | {r['acc_mean']:.4f} ± {r['acc_std']:.4f}")
        
    print("\n" + "="*50)
    print("TABLE II: Specificity Reproduction (Proposed Method)")
    print("="*50)
    print(f"{'Band':<10} | {'Specificity (Mean ± Std)'}")
    print("-" * 50)
    for band in BANDS.keys():
        r = final_results[band]
        print(f"{band:<10} | {r['spec_mean']:.4f} ± {r['spec_std']:.4f}")
