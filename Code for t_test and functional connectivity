import mne
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from statsmodels.stats.multitest import multipletests

# --- 1. CONFIGURATION & CONSTANTS ---
FREQ_BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 45)
}

TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
    'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
]

TARGET_SFREQ = 100 

def get_file_paths(base_path='/kaggle/input/eeg-dataset/'):
    """Separates files into MDD and Healthy based on filename."""
    all_files = glob.glob(os.path.join(base_path, '*.edf'))
    mdd_files = [f for f in all_files if 'MDD' in os.path.basename(f)]
    hc_files = [f for f in all_files if 'H ' in os.path.basename(f) or 'H S' in os.path.basename(f)]
    return mdd_files, hc_files

# --- 2. PREPROCESSING ---
def preprocess_eeg(file_path):
    """Loads, filters, downsamples, and normalizes EEG data."""
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        available_chans = raw.info['ch_names']
        picks = []
        for target in TARGET_CHANNELS:
            found = [ch for ch in available_chans if target in ch]
            if found:
                picks.append(found[0])
        
        if len(picks) < len(TARGET_CHANNELS):
            return None 
            
        raw.pick_channels(picks)
        raw.filter(0.1, 70, verbose=False)
        raw.notch_filter(50, verbose=False)
        raw.resample(TARGET_SFREQ, verbose=False)
        
        data = raw.get_data()
        data = stats.zscore(data, axis=1)
        return data
    except Exception: 
        return None

# --- 3. WAVELET COHERENCE & CONNECTIVITY MATRIX ---
def compute_connectivity_matrix(data, sfreq, freqs_of_interest):
    """
    Computes the Functional Connectivity Matrix for an epoch using Wavelet Coherence.
    """
    n_channels = data.shape[0]
    
    # Calculate CWT using Morlet Wavelets (output='complex' gives Wx(t,f))
    tfr = mne.time_frequency.tfr_array_morlet(
        data[np.newaxis, :, :], sfreq=sfreq, freqs=freqs_of_interest, 
        n_cycles=freqs_of_interest/2., output='complex', verbose=False
    )[0] 

    matrices = {band: np.zeros((n_channels, n_channels)) for band in FREQ_BANDS}
    power = np.abs(tfr) ** 2
    
    for band, (f_min, f_max) in FREQ_BANDS.items():
        freq_mask = (freqs_of_interest >= f_min) & (freqs_of_interest <= f_max)
        if not np.any(freq_mask): continue
            
        W_band = tfr[:, freq_mask, :]
        P_band = power[:, freq_mask, :]
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    matrices[band][i, j] = 1.0
                    continue
                
                # Formula: |<Wx Wy*>|^2 / (<|Wx|^2> <|Wy|^2>)
                cross_spec = W_band[i] * np.conj(W_band[j])
                mean_cross = np.mean(cross_spec)
                mean_power_i = np.mean(P_band[i]) 
                mean_power_j = np.mean(P_band[j]) 
                
                coh = (np.abs(mean_cross) ** 2) / (mean_power_i * mean_power_j + 1e-10)
                
                matrices[band][i, j] = coh
                matrices[band][j, i] = coh
                
    return matrices

# --- 4. MAIN PROCESSING PIPELINE (NEW EPOCH/STRIDE) ---
def process_dataset(mdd_paths, hc_paths):
    all_segments_mdd = {band: [] for band in FREQ_BANDS}
    all_segments_hc = {band: [] for band in FREQ_BANDS}
    freqs = np.logspace(np.log10(0.5), np.log10(45), 20)
    
    print(f"Found {len(mdd_paths)} MDD files and {len(hc_paths)} HC files. Collecting segments...")
    
    # --- UPDATED SEGMENTATION ---
    epoch_len = 500 # 5s * 100Hz = 500 samples
    stride = 250    # 2.5s * 100Hz = 250 samples (50% overlap)
    # -----------------------------

    # Process MDD and HC files
    all_paths = [(p, all_segments_mdd) for p in mdd_paths] + [(p, all_segments_hc) for p in hc_paths]
    
    for fp, storage in all_paths:
        data = preprocess_eeg(fp)
        if data is None: continue
        
        for start in range(0, data.shape[1] - epoch_len, stride):
            segment = data[:, start:start+epoch_len]
            segment_matrices = compute_connectivity_matrix(segment, TARGET_SFREQ, freqs)
            
            for band in FREQ_BANDS:
                storage[band].append(segment_matrices[band])
                
    return all_segments_mdd, all_segments_hc

# --- 5. STATISTICAL ANALYSIS & VISUALIZATION (ASTERSISK ADDED) ---
def plot_fc_and_t_stats(mdd_data, hc_data):
    
    for band in FREQ_BANDS:
        mdd_stack = np.array(mdd_data[band])
        hc_stack = np.array(hc_data[band])
        
        if len(mdd_stack) < 2 or len(hc_stack) < 2:
            print(f"Not enough segments collected for band {band}")
            continue
            
        print(f"\n--- Analyzing {band} Band (MDD segments: {mdd_stack.shape[0]}, HC segments: {hc_stack.shape[0]}) ---")
        
        # 1. Calculate Average FC Matrices
        mean_hc_matrix = np.mean(hc_stack, axis=0)
        mean_mdd_matrix = np.mean(mdd_stack, axis=0)

        # 2. Perform Independent T-Test
        t_vals, p_vals = stats.ttest_ind(mdd_stack, hc_stack, axis=0, equal_var=False)
        
        # 3. FDR Correction
        p_flat = p_vals.flatten()
        mask_valid = ~np.isnan(p_flat)
        
        _, p_adj_flat, _, _ = multipletests(p_flat[mask_valid], alpha=0.05, method='fdr_bh')
        
        # Reconstruct adjusted p-matrix
        p_adj = np.ones_like(p_flat)
        p_adj[mask_valid] = p_adj_flat
        p_adj = p_adj.reshape(p_vals.shape)
        
        # Create T-statistic matrix for plotting (Masking non-significant)
        significant_mask = p_adj < 0.05
        t_masked = t_vals.copy()
        t_masked[~significant_mask] = 0
        
        # --- NEW: Create annotation matrix for significance (asterisks) ---
        annot_matrix = np.full(t_masked.shape, '', dtype=str)
        annot_matrix[significant_mask] = '*'
        max_abs_t = np.max(np.abs(t_masked)) or 1.0

        # --- PLOTTING ALL THREE MATRICES ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        def format_ax(ax, title, xticklabels=TARGET_CHANNELS, yticklabels=TARGET_CHANNELS):
            ax.set_title(title, fontsize=12)
            ax.set_xticks(np.arange(len(xticklabels)) + 0.5)
            ax.set_yticks(np.arange(len(yticklabels)) + 0.5)
            ax.set_xticklabels(xticklabels, rotation=90, fontsize=8)
            ax.set_yticklabels(yticklabels, rotation=0, fontsize=8)
            ax.tick_params(axis='both', which='major', length=0)
        
        # Plot 1: HC Average FC Matrix
        sns.heatmap(mean_hc_matrix, ax=axes[0], cmap='jet', vmin=0, vmax=1.0, 
                    cbar_kws={'label': 'Coherence Value'})
        format_ax(axes[0], f'HC Average Functional Connectivity ({band})')

        # Plot 2: MDD Average FC Matrix
        sns.heatmap(mean_mdd_matrix, ax=axes[1], cmap='jet', vmin=0, vmax=1.0, 
                    cbar_kws={'label': 'Coherence Value'})
        format_ax(axes[1], f'MDD Average Functional Connectivity ({band})')
        
        # Plot 3: Significant T-Statistics Matrix (WITH ANNOTATION)
        sns.heatmap(t_masked, ax=axes[2], cmap='bwr', center=0, 
                    vmin=-max_abs_t, vmax=max_abs_t,
                    annot=annot_matrix,               # <-- Asterisk annotation
                    fmt='s',                          # <-- Format as string
                    annot_kws={"size": 18, "color": "black"}, # Style the asterisk
                    cbar_kws={'label': 'T-statistic'})
        format_ax(axes[2], f'Significant T-Stats ({band}) with FDR*')
        
        plt.suptitle(f'Functional Connectivity Analysis in the {band} Band', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    mdd_files, hc_files = get_file_paths()
    
    if mdd_files and hc_files:
        res_mdd, res_hc = process_dataset(mdd_files, hc_files)
        plot_fc_and_t_stats(res_mdd, res_hc)
    else:
        print("No files found. Please ensure the files are correctly mounted in the expected Kaggle path.")
