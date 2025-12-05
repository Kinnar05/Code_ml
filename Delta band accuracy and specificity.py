import os
import glob
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats, signal
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import pywt
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
DATA_DIR = '/kaggle/input/eeg-dataset/'
TARGET_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                   'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
SAMPLING_RATE = 100
EPOCH_DURATION = 30
OVERLAP = 0.5         
SEEDS = 5             
FOLDS = 5             

# DWT configuration for Wavelet Coherence
WAVELET_NAME = 'db4'
# DWT levels for approximating frequency bands at 100 Hz sampling rate:
DWT_BANDS = {
    'Delta': 4, # A4: 0-3.125 Hz 
    'Theta': 3, # D3: 3.125-6.25 Hz (Proxy for Theta/Alpha)
    'Alpha': 2, # D2: 6.25-12.5 Hz (Proxy for Alpha/Beta)
    'Beta': 1   # D1: 12.5-25 Hz (Proxy for Beta/Gamma)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 1. ROBUST DATA LOADING & PREPROCESSING
# ==========================================

def get_file_list(data_dir):
    """
    Finds files and assigns labels based on filename.
    MDD = 1, Healthy (H) = 0. Uses ALL files (EC, EO, TASK).
    """
    all_files = glob.glob(os.path.join(data_dir, "*.edf"))
    
    # --- CRITICAL CHANGE: Use all files, no 'EC' filter ---
    subjects = []
    labels = [] 
    
    print(f"DEBUG: Found {len(all_files)} total files for processing (EC, EO, and TASK).")

    for f in all_files:
        filename = os.path.basename(f)
        if 'MDD' in filename:
            labels.append(1)
            subjects.append(f)
        elif 'H S' in filename or 'H ' in filename:
            labels.append(0)
            subjects.append(f)
    
    return np.array(subjects), np.array(labels)

def standardize_channels(raw):
    current_chs = raw.info['ch_names']
    mapping = {}
    for ch in current_chs:
        clean_name = ch.replace('EEG', '').replace('-LE', '').replace('REF', '').replace('-', '').strip()
        if clean_name in TARGET_CHANNELS:
            mapping[ch] = clean_name
    raw.rename_channels(mapping)
    return raw

def preprocess_eeg(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw = standardize_channels(raw)
        available_chs = raw.info['ch_names']
        picked_chs = [ch for ch in TARGET_CHANNELS if ch in available_chs]
        
        if len(picked_chs) < 15: return None
        raw.pick_channels(picked_chs)
        
        raw.filter(0.1, 70, verbose=False)
        raw.notch_filter(50, verbose=False)
        
        if raw.info['sfreq'] != SAMPLING_RATE:
            raw.resample(SAMPLING_RATE)
        
        events = mne.make_fixed_length_events(raw, duration=EPOCH_DURATION, overlap=EPOCH_DURATION*OVERLAP)
        if len(events) == 0: return None
            
        epochs = mne.Epochs(raw, events, tmin=0, tmax=EPOCH_DURATION, baseline=None, verbose=False)
        data = epochs.get_data(copy=True)
        
        if data.shape[0] == 0: return None

        for i in range(data.shape[0]):
            data[i] = stats.zscore(data[i], axis=1)
            
        return data
        
    except Exception as e:
        # print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

# ==========================================
# 2. FEATURE EXTRACTION (MULTI-BAND WAVELET)
# ==========================================

def extract_baseline_features(epoch_data):
    # Statistical features (8 per channel)
    features = []
    features.extend(np.mean(epoch_data, axis=1))
    features.extend(np.var(epoch_data, axis=1))
    features.extend(np.std(epoch_data, axis=1))
    features.extend(stats.kurtosis(epoch_data, axis=1))
    features.extend(stats.skew(epoch_data, axis=1))
    features.extend(np.max(epoch_data, axis=1))
    features.extend(np.min(epoch_data, axis=1))
    features.extend(np.sqrt(np.mean(epoch_data**2, axis=1)))
    return np.array(features)

def extract_band_dwt(data, level):
    """Extracts a specific frequency band component using DWT."""
    band_data = np.zeros_like(data)
    max_level = 4
    
    for i in range(data.shape[0]):
        coeffs = pywt.wavedec(data[i], WAVELET_NAME, level=max_level)
        
        coeffs_band = [np.zeros_like(c) for c in coeffs]
        
        if level == max_level: # Approximation (e.g., Delta)
            coeffs_band[max_level] = coeffs[max_level]
        elif level >= 1 and level < max_level: # Detail (e.g., Theta, Alpha, Beta)
            coeffs_band[level] = coeffs[level]
        else:
            continue
            
        band_signal = pywt.waverec(coeffs_band, WAVELET_NAME)[:data.shape[1]]
        band_data[i, :] = band_signal
        
    return band_data

def calculate_band_coherence(epoch_data, level):
    """Calculates functional connectivity (Coherence) on a DWT-filtered band."""
    n_channels = epoch_data.shape[0]
    filtered_band = extract_band_dwt(epoch_data, level)
    analytic = signal.hilbert(filtered_band, axis=1)
    
    connectivity_values = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Calculate Magnitude Squared Coherence (MSC) proxy
            cross_spec = np.mean(analytic[i] * np.conj(analytic[j]))
            auto_spec_i = np.mean(np.abs(analytic[i])**2)
            auto_spec_j = np.mean(np.abs(analytic[j])**2)
            coh = np.abs(cross_spec)**2 / (auto_spec_i * auto_spec_j)
            connectivity_values.append(coh)
            
    return np.array(connectivity_values)

# ==========================================
# 3. MODELS (EEGNet & SVM)
# ==========================================

class EEGNet(nn.Module):
    def __init__(self, nb_classes=2, Chans=19, Samples=3000):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.depthwiseConv = nn.Conv2d(8, 16, (Chans, 1), groups=8, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.activation = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.25)
        self.separableConv = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(0.25)
        
        with torch.no_grad():
            dummy = torch.zeros((1, 1, Chans, Samples))
            x = self.conv1(dummy)
            x = self.batchnorm1(x)
            x = self.depthwiseConv(x)
            x = self.batchnorm2(x)
            x = self.activation(x)
            x = self.avgpool1(x)
            x = self.dropout1(x)
            x = self.separableConv(x)
            x = self.batchnorm3(x)
            x = self.activation(x)
            x = self.avgpool2(x)
            x = self.dropout2(x)
            flatten_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flatten_size, nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def train_eegnet(X_train, y_train, X_test, y_test):
    train_dataset = TensorDataset(torch.Tensor(X_train).unsqueeze(1), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test).unsqueeze(1), torch.LongTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = EEGNet(Chans=X_train.shape[1], Samples=X_train.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(5):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            
    return np.array(all_preds)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_experiment():
    print(">>> 1. Identifying Data Files...")
    subjects, labels = get_file_list(DATA_DIR)
    
    if len(subjects) == 0:
        raise ValueError("No EDF files found. Check dataset path.")

    all_epochs_raw = []
    all_epochs_baseline = []
    all_epochs_wavelet = []
    subject_indices = []
    y_ground_truth = []
    
    print(">>> 2. Preprocessing & Multi-Band Feature Extraction...")
    
    for i, (sub, lbl) in enumerate(zip(subjects, labels)):
        data = preprocess_eeg(sub)
        if data is None: 
            continue
            
        # Extract features for all epochs
        for epoch in data:
            all_epochs_raw.append(epoch)
            all_epochs_baseline.append(extract_baseline_features(epoch))
            
            # --- Extract Wavelet Coherence features for all bands ---
            multi_band_features = []
            for band_name, level in DWT_BANDS.items():
                band_coh_features = calculate_band_coherence(epoch, level)
                multi_band_features.extend(band_coh_features)
            
            all_epochs_wavelet.append(np.array(multi_band_features))
            
            subject_indices.append(i)
            y_ground_truth.append(lbl)
    
    
    if len(all_epochs_raw) == 0:
        raise ValueError("Found array with 0 samples! Preprocessing failed for all files. Check channel names or file content.")

    X_raw = np.array(all_epochs_raw)
    X_base = np.array(all_epochs_baseline)
    X_wave = np.array(all_epochs_wavelet)
    y = np.array(y_ground_truth)
    groups = np.array(subject_indices)
    
    print(f"DEBUG: Processed {len(np.unique(groups))} subjects, yielding {len(all_epochs_raw)} epochs from all conditions.")
    print(f"Data Shapes: Raw {X_raw.shape}, Baseline {X_base.shape}, Wavelet {X_wave.shape} (Multi-Band, All Data)")
    
    results = {
        'Proposed (All Data, Multi-Band)': {'acc': [], 'spec': []},
        'Baseline (Stat)': {'acc': [], 'spec': []},
        'EEGNet': {'acc': [], 'spec': []}
    }
    
    print(">>> 3. Running Cross-Validation...")
    for seed in range(SEEDS):
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        unique_subs = np.unique(groups)
        
        sub_labels = []
        for s in unique_subs:
            mask = groups == s
            if np.any(mask):
                sub_labels.append(y[mask][0])

        for train_idx_sub, test_idx_sub in skf.split(unique_subs, sub_labels):
            train_subs = unique_subs[train_idx_sub]
            test_subs = unique_subs[test_idx_sub]
            
            train_mask = np.isin(groups, train_subs)
            test_mask = np.isin(groups, test_subs)
            
            y_train, y_test = y[train_mask], y[test_mask]
            
            # --- Baseline (Stat) ---
            scaler = StandardScaler()
            X_base_train = scaler.fit_transform(X_base[train_mask])
            X_base_test = scaler.transform(X_base[test_mask])
            svm = SVC(kernel='linear', class_weight='balanced')
            svm.fit(X_base_train, y_train)
            y_pred_base = svm.predict(X_base_test)
            
            # --- Proposed (All Data, Multi-Band) ---
            X_wave_train_raw = X_wave[train_mask]
            X_wave_test_raw = X_wave[test_mask]
            
            # Feature Selection: Selecting 10% of the total multi-band features.
            k_features = max(1, int(X_wave.shape[1] * 0.1))
            selector = SelectKBest(f_classif, k=k_features)
            
            try:
                X_wave_train = selector.fit_transform(X_wave_train_raw, y_train)
                X_wave_test = selector.transform(X_wave_test_raw)
            except:
                X_wave_train = X_wave_train_raw
                X_wave_test = X_wave_test_raw
                
            scaler_w = StandardScaler()
            X_wave_train = scaler_w.fit_transform(X_wave_train)
            X_wave_test = scaler_w.transform(X_wave_test)
            svm_w = SVC(kernel='rbf', class_weight='balanced') 
            svm_w.fit(X_wave_train, y_train)
            y_pred_wave = svm_w.predict(X_wave_test)
            
            # --- EEGNet ---
            y_pred_eeg = train_eegnet(X_raw[train_mask], y_train, X_raw[test_mask], y_test)
            
            def calc_metrics(y_true, y_pred):
                labels = [0, 1]
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                tn, fp, fn, tp = cm.ravel()
                
                total = tn + fp + fn + tp
                acc = (tp + tn) / total if total > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                return acc, spec

            acc_b, spec_b = calc_metrics(y_test, y_pred_base)
            acc_w, spec_w = calc_metrics(y_test, y_pred_wave)
            acc_e, spec_e = calc_metrics(y_test, y_pred_eeg)
            
            results['Baseline (Stat)']['acc'].append(acc_b)
            results['Baseline (Stat)']['spec'].append(spec_b)
            results['Proposed (All Data, Multi-Band)']['acc'].append(acc_w)
            results['Proposed (All Data, Multi-Band)']['spec'].append(spec_w)
            results['EEGNet']['acc'].append(acc_e)
            results['EEGNet']['spec'].append(spec_e)

    return results

if __name__ == "__main__":
    try:
        final_results = run_experiment()
        
        print("\n" + "="*50)
        print("PERFORMANCE COMPARISON (All Data, Multi-Band DWT-Enhanced)")
        print("="*50)
        
        print("\nTABLE I: Mean Accuracy (5-Fold CV, 5 Seeds)")
        print("-" * 50)
        print(f"{'Method':<30} | {'Mean Accuracy':<15}")
        print("-" * 50)
        for method, metrics in final_results.items():
            mean_acc = np.mean(metrics['acc'])
            std_acc = np.std(metrics['acc'])
            print(f"{method:<30} | {mean_acc:.4f} ± {std_acc:.4f}")
            
        print("\nTABLE II: Mean Specificity")
        print("-" * 50)
        print(f"{'Method':<30} | {'Mean Specificity':<15}")
        print("-" * 50)
        for method, metrics in final_results.items():
            mean_spec = np.mean(metrics['spec'])
            std_spec = np.std(metrics['spec'])
            print(f"{method:<30} | {mean_spec:.4f} ± {std_spec:.4f}")
            
    except ValueError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"CRITICAL FAILURE during experiment execution: {e}")
