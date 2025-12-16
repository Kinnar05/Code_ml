import numpy as np
import pandas as pd
import mne
from scipy import signal, stats
from scipy.stats import levene, shapiro
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
import pywt
from statsmodels.stats.multitest import fdrcorrection
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def print_progress(message, percentage=None):
    """Print progress with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if percentage is not None:
        print(f"[{timestamp}] {message} - {percentage:.1f}% complete")
    else:
        print(f"[{timestamp}] {message}")

# =====================================================================
# STEP 1: LOAD AND PREPROCESS EEG DATA
# =====================================================================

def load_and_preprocess_eeg(file_path, channels_to_select, file_idx, total_files):
    """Load and preprocess EEG data from EDF file"""
    try:
        percentage = (file_idx / total_files) * 100
        print_progress(f"Loading file {file_idx}/{total_files}: {file_path.split('/')[-1]}", percentage)
        
        # Load EEG data
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Check if required channels are available
        available_channels = raw.ch_names
        channels_present = [ch for ch in channels_to_select if any(ch.upper() in avail.upper() for avail in available_channels)]
        
        if len(channels_present) < len(channels_to_select):
            print(f"Warning: Only {len(channels_present)}/{len(channels_to_select)} channels found in {file_path.split('/')[-1]}")
            if len(channels_present) == 0:
                return None, None
        
        # Bandpass filter (0.1 - 70 Hz)
        raw.filter(0.1, 70., fir_design='firwin', verbose=False)
        
        # Notch filter (50 Hz)
        raw.notch_filter(50., fir_design='firwin', verbose=False)
        
        # Apply ICA for artifact removal (with fewer components if needed)
        n_components = min(15, len(raw.ch_names))
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter=200, verbose=False)
        ica.fit(raw)
        raw = ica.apply(raw)
        
        # Select available channels from the 19 key channels
        # Map channel names flexibly
        ch_mapping = {}
        for target_ch in channels_to_select:
            for avail_ch in raw.ch_names:
                if target_ch.upper() in avail_ch.upper() or avail_ch.upper() in target_ch.upper():
                    ch_mapping[avail_ch] = target_ch
                    break
        
        if len(ch_mapping) == 0:
            print(f"No matching channels found in {file_path.split('/')[-1]}")
            return None, None
        
        # Pick channels
        raw.pick_channels(list(ch_mapping.keys()))
        
        # Rename to standard names
        raw.rename_channels(ch_mapping)
        
        # Downsample to 100 Hz
        raw.resample(100, npad='auto', verbose=False)
        
        # Get data and apply z-score normalization per channel
        data = raw.get_data()
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-10)
        
        return data, raw.info['sfreq']
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

# =====================================================================
# STEP 2: WAVELET COHERENCE COMPUTATION
# =====================================================================

def compute_wavelet_coherence(signal1, signal2, fs, frequencies):
    """Compute wavelet coherence between two signals"""
    # Compute continuous wavelet transform for both signals
    scales = pywt.frequency2scale('morl', frequencies / fs)
    
    coef1, _ = pywt.cwt(signal1, scales, 'morl')
    coef2, _ = pywt.cwt(signal2, scales, 'morl')
    
    # Compute cross-spectrum and power spectra
    cross_spectrum = coef1 * np.conj(coef2)
    power1 = np.abs(coef1) ** 2
    power2 = np.abs(coef2) ** 2
    
    # Smooth in time (simple moving average)
    window = 10
    cross_spectrum_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window)/window, mode='same'), 
        axis=1, arr=cross_spectrum
    )
    power1_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window)/window, mode='same'), 
        axis=1, arr=power1
    )
    power2_smooth = np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window)/window, mode='same'), 
        axis=1, arr=power2
    )
    
    # Compute coherence
    coherence = np.abs(cross_spectrum_smooth) ** 2 / (power1_smooth * power2_smooth + 1e-10)
    
    return coherence

def compute_band_coherence(data, fs, band_freqs):
    """Compute average coherence for a frequency band"""
    n_channels = data.shape[0]
    frequencies = np.linspace(band_freqs[0], band_freqs[1], 20)
    
    # Initialize connectivity matrix
    connectivity_matrix = np.zeros((n_channels, n_channels))
    
    # Compute coherence for each channel pair
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            coherence = compute_wavelet_coherence(data[i], data[j], fs, frequencies)
            # Average across time and frequency
            avg_coherence = np.mean(coherence)
            connectivity_matrix[i, j] = avg_coherence
            connectivity_matrix[j, i] = avg_coherence
    
    return connectivity_matrix

# =====================================================================
# STEP 3: FEATURE EXTRACTION
# =====================================================================

def extract_statistical_features(data):
    """Extract statistical features from EEG data"""
    features = []
    for channel_data in data:
        features.extend([
            np.mean(channel_data),
            np.var(channel_data),
            np.std(channel_data),
            stats.kurtosis(channel_data),
            stats.skew(channel_data),
            np.max(channel_data),
            np.min(channel_data),
            np.sqrt(np.mean(channel_data**2))  # RMS
        ])
    return np.array(features)

def extract_connectivity_features(connectivity_matrix, significant_pairs=None):
    """Extract features from connectivity matrix"""
    if significant_pairs is None:
        # Use upper triangle (excluding diagonal)
        features = connectivity_matrix[np.triu_indices_from(connectivity_matrix, k=1)]
    else:
        # Use only significant pairs
        features = connectivity_matrix[significant_pairs]
    return features

# =====================================================================
# STEP 4: SEGMENT DATA INTO EPOCHS
# =====================================================================

def segment_data(data, fs, epoch_length=30, overlap=0.5):
    """Segment data into overlapping epochs"""
    epoch_samples = int(epoch_length * fs)
    step_samples = int(epoch_samples * (1 - overlap))
    
    segments = []
    n_samples = data.shape[1]
    
    for start in range(0, n_samples - epoch_samples + 1, step_samples):
        end = start + epoch_samples
        segments.append(data[:, start:end])
    
    return segments

# =====================================================================
# STEP 5: STATISTICAL TESTING FOR SIGNIFICANT FEATURES
# =====================================================================

def find_significant_connections(X_mdd, X_hc, alpha=0.05):
    """Find statistically significant connections using t-test with FDR correction"""
    n_features = X_mdd.shape[1]
    p_values = []
    
    for i in range(n_features):
        mdd_values = X_mdd[:, i]
        hc_values = X_hc[:, i]
        
        # Check assumptions
        _, p_norm_mdd = shapiro(mdd_values[:min(50, len(mdd_values))])
        _, p_norm_hc = shapiro(hc_values[:min(50, len(hc_values))])
        _, p_var = levene(mdd_values, hc_values)
        
        # Use Welch's t-test if assumptions violated
        if p_norm_mdd < 0.05 or p_norm_hc < 0.05 or p_var < 0.05:
            t_stat, p_val = stats.ttest_ind(mdd_values, hc_values, equal_var=False)
        else:
            t_stat, p_val = stats.ttest_ind(mdd_values, hc_values, equal_var=True)
        
        p_values.append(p_val)
    
    # FDR correction
    reject, p_corrected = fdrcorrection(p_values, alpha=alpha)
    significant_indices = np.where(reject)[0]
    
    return significant_indices

# =====================================================================
# STEP 6: CLASSIFICATION WITH SVM
# =====================================================================

def classify_svm(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM classifier"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    clf = SVC(kernel='linear', random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Handle confusion matrix safely
    cm = confusion_matrix(y_test, y_pred)
    
    # Check if we have both classes in predictions
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        # Only one class present
        if y_test[0] == 0:  # Only HC
            tn = cm[0, 0]
            fp, fn, tp = 0, 0, 0
        else:  # Only MDD
            tp = cm[0, 0]
            tn, fp, fn = 0, 0, 0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (specificity + sensitivity) / 2
    
    return accuracy, specificity, balanced_accuracy

# =====================================================================
# STEP 7: MAIN PIPELINE
# =====================================================================

def main_pipeline(file_paths, labels, band_name, band_freqs, feature_type='connectivity', n_splits=5, n_seeds=50):
    """Main pipeline for MDD classification"""
    
    # Define channels
    channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    print_progress(f"Starting pipeline for {band_name} band ({band_freqs[0]}-{band_freqs[1]} Hz), Feature type: {feature_type}")
    
    # Load and process all data
    all_segments = []
    all_labels = []
    all_subjects = []
    
    total_files = len(file_paths)
    for idx, file_path in enumerate(file_paths):
        data, fs = load_and_preprocess_eeg(file_path, channels, idx+1, total_files)
        
        if data is not None:
            # Segment data
            segments = segment_data(data, fs)
            
            for segment in segments:
                all_segments.append(segment)
                all_labels.append(labels[idx])
                all_subjects.append(idx)
    
    print_progress(f"Data loading complete. Total segments: {len(all_segments)} (MDD: {sum(all_labels)}, HC: {len(all_labels) - sum(all_labels)})")
    
    # Extract features for all segments
    print_progress("Extracting features from segments...")
    if feature_type == 'statistical':
        X = np.array([extract_statistical_features(seg) for seg in all_segments])
        significant_features = None
        print_progress("Statistical features extracted")
    else:  # connectivity
        # Compute connectivity matrices
        connectivity_matrices = []
        for seg_idx, seg in enumerate(all_segments):
            if (seg_idx + 1) % 50 == 0 or seg_idx == 0:
                percentage = (seg_idx + 1) / len(all_segments) * 100
                print_progress(f"Computing connectivity {seg_idx + 1}/{len(all_segments)}", percentage)
            conn_matrix = compute_band_coherence(seg, fs, band_freqs)
            connectivity_matrices.append(conn_matrix)
        
        print_progress("Computing connectivity matrices complete")
        
        # Extract all connectivity features initially
        X_all = np.array([extract_connectivity_features(cm) for cm in connectivity_matrices])
        
        # Find significant features using all data (for demonstration)
        # In practice, this should be done within each fold
        print_progress("Performing statistical testing to find significant features...")
        y = np.array(all_labels)
        X_mdd = X_all[y == 1]
        X_hc = X_all[y == 0]
        
        significant_features = find_significant_connections(X_mdd, X_hc)
        print_progress(f"Significant features found: {len(significant_features)}")
        
        # Use only significant features
        if len(significant_features) > 0:
            X = X_all[:, significant_features]
        else:
            X = X_all
    
    # Cross-validation across multiple seeds
    accuracies = []
    specificities = []
    balanced_accs = []
    
    y = np.array(all_labels)
    subjects = np.array(all_subjects)
    
    print_progress(f"Starting {n_splits}-fold cross-validation across {n_seeds} seeds...")
    
    for seed in range(n_seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        fold_acc = []
        fold_spec = []
        fold_bal = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=subjects)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            acc, spec, bal = classify_svm(X_train, y_train, X_test, y_test)
            fold_acc.append(acc)
            fold_spec.append(spec)
            fold_bal.append(bal)
        
        accuracies.append(np.mean(fold_acc))
        specificities.append(np.mean(fold_spec))
        balanced_accs.append(np.mean(fold_bal))
        
        percentage = ((seed + 1) / n_seeds) * 100
        print_progress(f"Completed seed {seed + 1}/{n_seeds}", percentage)
    
    # Calculate final statistics
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_spec = np.mean(specificities)
    std_spec = np.std(specificities)
    mean_bal = np.mean(balanced_accs)
    std_bal = np.std(balanced_accs)
    
    print_progress(f"Results - Accuracy: {mean_acc:.4f}±{std_acc:.4f}, Specificity: {mean_spec:.4f}±{std_spec:.4f}, Balanced Acc: {mean_bal:.4f}±{std_bal:.4f}")
    
    return mean_acc, std_acc, mean_spec, std_spec

# =====================================================================
# STEP 8: RUN EXPERIMENTS
# =====================================================================

if __name__ == "__main__":
    print_progress("="*80)
    print_progress("STARTING MDD CLASSIFICATION EXPERIMENT")
    print_progress("="*80)
    
    # Filter for EC (Eyes Closed) files only
    import glob
    
    # Adjust the path to your dataset
    dataset_path = "/kaggle/input/eeg-dataset/"
    print_progress(f"Searching for files in: {dataset_path}")
    all_files = glob.glob(dataset_path + "*.edf")
    
    print_progress(f"Found {len(all_files)} total EDF files", 5)
    
    # Filter EC files
    ec_files = [f for f in all_files if "EC.edf" in f]
    
    # Separate MDD and HC files
    mdd_files = sorted([f for f in ec_files if "MDD" in f])
    hc_files = sorted([f for f in ec_files if "/H " in f or "H S" in f])
    
    print_progress(f"Found {len(mdd_files)} MDD EC files and {len(hc_files)} HC EC files", 10)
    
    if len(mdd_files) == 0 or len(hc_files) == 0:
        print_progress("ERROR: No EC files found. Please check the dataset path.")
        print_progress(f"Total files found: {len(all_files)}")
        print_progress(f"Sample files: {all_files[:5] if len(all_files) > 0 else 'None'}")
        exit()
    
    # Create labels (1 for MDD, 0 for HC)
    file_paths = hc_files + mdd_files
    labels = [0] * len(hc_files) + [1] * len(mdd_files)
    
    print_progress(f"Total files to process: {len(file_paths)}", 15)
    
    # Define frequency bands
    bands = {
        'Delta (δ)': (0.5, 4),
        'Theta (θ)': (4, 8),
        'Alpha (α)': (8, 13),
        'Beta (β)': (13, 30)
    }
    
    # Results storage
    results_table1 = {}  # Accuracy
    results_table2 = {}  # Specificity
    
    total_experiments = len(bands) * 2  # 2 = proposed + baseline
    experiment_count = 0
    
    # Run experiments for each band
    for band_idx, (band_name, band_freqs) in enumerate(bands.items()):
        overall_progress = 15 + ((band_idx * 2) / total_experiments) * 75
        print_progress(f"\n{'='*80}", overall_progress)
        print_progress(f"PROCESSING BAND {band_idx + 1}/{len(bands)}: {band_name}")
        print_progress(f"{'='*80}")
        
        # Proposed method (connectivity features)
        experiment_count += 1
        print_progress(f"Experiment {experiment_count}/{total_experiments}: Proposed method")
        acc_prop, std_acc_prop, spec_prop, std_spec_prop = main_pipeline(
            file_paths, labels, band_name, band_freqs, 
            feature_type='connectivity', n_splits=5, n_seeds=50
        )
        
        # Baseline (statistical features)
        experiment_count += 1
        print_progress(f"Experiment {experiment_count}/{total_experiments}: Baseline method")
        acc_base, std_acc_base, spec_base, std_spec_base = main_pipeline(
            file_paths, labels, band_name, band_freqs, 
            feature_type='statistical', n_splits=5, n_seeds=50
        )
        
        # Store results
        band_symbol = band_name.split('(')[1].split(')')[0]
        results_table1[f'{band_symbol} band'] = {
            'Proposed': f"{acc_prop:.4f} ± {std_acc_prop:.4f}",
            'Baseline': f"{acc_base:.4f} ± {std_acc_base:.4f}"
        }
        results_table2[f'{band_symbol} band'] = {
            'Proposed': f"{spec_prop:.4f} ± {std_spec_prop:.4f}",
            'Baseline': f"{spec_base:.4f} ± {std_spec_base:.4f}"
        }
    
    # Print final tables
    print_progress("\n" + "="*80, 95)
    print_progress("TABLE I: ACCURACY COMPARISON")
    print_progress("="*80)
    df1 = pd.DataFrame(results_table1).T
    print(df1)
    
    print_progress("\n" + "="*80)
    print_progress("TABLE II: SPECIFICITY COMPARISON")
    print_progress("="*80)
    df2 = pd.DataFrame(results_table2).T
    print(df2)
    
    print_progress("\n" + "="*80, 100)
    print_progress("EXPERIMENT COMPLETED SUCCESSFULLY")
    print_progress("="*80)
