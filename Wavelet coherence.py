import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d 
import scipy.stats as stats
import glob
import os

# --- 1. CONFIGURATION ---
TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
    'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
]
TARGET_SFREQ = 100 
EPOCH_LEN_SAMPLES = 500 # 5 seconds * 100 Hz
CHANNEL_PAIR_INDICES = (0, 1) # Fp1 (index 0) and Fp2 (index 1)
FREQ_GRID_POINTS = 100 

# --- 2. FILE HANDLING & PREPROCESSING ---
def get_file_paths(base_path='/kaggle/input/eeg-dataset/'):
    """Separates files into MDD and Healthy based on filename."""
    all_files = glob.glob(os.path.join(base_path, '*.edf'))
    mdd_files = [f for f in all_files if 'MDD' in os.path.basename(f)]
    hc_files = [f for f in all_files if 'H ' in os.path.basename(f) or 'H S' in os.path.basename(f)]
    return mdd_files, hc_files

def preprocess_eeg(file_path):
    """Loads, filters, downsamples, and z-score normalizes EEG data."""
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        available_chans = raw.info['ch_names']
        picks = []
        for target in TARGET_CHANNELS:
            found = [ch for ch in available_chans if target in ch]
            if found:
                picks.append(found[0])
        
        # Check if the required channels for the pair exist
        if len(picks) <= CHANNEL_PAIR_INDICES[1]:
             print(f"Error: Could not find required channels in file: {os.path.basename(file_path)}")
             return None
            
        raw.pick_channels(picks)
        raw.filter(0.1, 70, verbose=False)
        raw.notch_filter(50, verbose=False)
        raw.resample(TARGET_SFREQ, verbose=False)
        data = raw.get_data()
        data = stats.zscore(data, axis=1)
        return data
    except Exception as e: 
        print(f"Error during preprocessing: {e}")
        return None

# --- 3. WAVELET COHERENCE CORE LOGIC ---
def smooth_map(data_map, smoothing_window=7):
    """
    Approximation of the smoothing operator (S) using 2D Boxcar convolution.
    """
    kernel = np.ones((smoothing_window, smoothing_window)) / (smoothing_window**2)
    smoothed = convolve2d(data_map, kernel, mode='same', boundary='symm')
    return smoothed

def calculate_wavelet_data(segment, sfreq, freqs, ch_idx_x, ch_idx_y):
    """Calculates Wavelet Coherence map for a pair of channels."""
    
    x = segment[ch_idx_x, :]
    y = segment[ch_idx_y, :]
    
    # Calculate CWT for both signals
    Wx = mne.time_frequency.tfr_array_morlet(
        x[np.newaxis, np.newaxis, :], sfreq=sfreq, freqs=freqs, 
        n_cycles=freqs/2., output='complex', verbose=False
    ).squeeze()
    
    Wy = mne.time_frequency.tfr_array_morlet(
        y[np.newaxis, np.newaxis, :], sfreq=sfreq, freqs=freqs, 
        n_cycles=freqs/2., output='complex', verbose=False
    ).squeeze()
    
    # Cross and Auto Spectra for Coherence
    Sxy = Wx * np.conj(Wy)
    Sxx = np.abs(Wx)**2
    Syy = np.abs(Wy)**2
    
    # Apply Smoothing (S operator)
    S_Sxy = smooth_map(Sxy.real, 7) + 1j * smooth_map(Sxy.imag, 7)
    S_Sxx = smooth_map(Sxx, 7)
    S_Syy = smooth_map(Syy, 7)
    
    # Calculate Wavelet Coherence Map
    R2 = (np.abs(S_Sxy)**2) / (S_Sxx * S_Syy + 1e-10)
    
    return R2

# --- 4. PLOTTING FUNCTION (Panel C Only) ---
def plot_wavelet_coherence_only(segment, group_label, sfreq, freqs, ch_x_name, ch_y_name):
    """Generates a single-panel plot of Wavelet Coherence with LINEAR Y-axis."""
    
    # Calculate the required data map
    R2 = calculate_wavelet_data(segment, sfreq, freqs, *CHANNEL_PAIR_INDICES)
    
    time_vec = np.linspace(0, segment.shape[1]/sfreq, segment.shape[1])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    # Panel C: Wavelet Coherence (R^2)
    im = ax.pcolormesh(time_vec, freqs, R2, cmap='jet', shading='auto', vmin=0, vmax=1.0)
    ax.set_title(f'Wavelet Coherence: {ch_x_name}-{ch_y_name} ({group_label})')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    
    # Linear Y-axis scaling
    y_ticks_linear = [0, 10, 20, 30, 40] 
    ax.set_yticks(y_ticks_linear)
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='Coherence Magnitude Squared (R²)', ticks=np.linspace(0, 1, 6))

    plt.tight_layout()
    plt.show()

# --- EXECUTION ---
if __name__ == "__main__":
    mdd_files, hc_files = get_file_paths()
    
    if mdd_files and hc_files:
        freqs = np.logspace(np.log10(0.5), np.log10(45), FREQ_GRID_POINTS)
        ch_x, ch_y = TARGET_CHANNELS[CHANNEL_PAIR_INDICES[0]], TARGET_CHANNELS[CHANNEL_PAIR_INDICES[1]]
        
        # 1. Process HC Segment
        hc_data = preprocess_eeg(hc_files[0])
        hc_segment = hc_data[:, :EPOCH_LEN_SAMPLES] if hc_data is not None and hc_data.shape[1] >= EPOCH_LEN_SAMPLES else None
        
        if hc_segment is not None:
            # Only print Fp1-Fp2 for HC
            print(f"HC Subject: Fp1-Fp2 Wavelet Coherence Plot.")
            plot_wavelet_coherence_only(hc_segment, "Healthy Control (HC)", TARGET_SFREQ, freqs, ch_x, ch_y)
        
        # 2. Process MDD Segment
        mdd_data = preprocess_eeg(mdd_files[0])
        mdd_segment = mdd_data[:, :EPOCH_LEN_SAMPLES] if mdd_data is not None and mdd_data.shape[1] >= EPOCH_LEN_SAMPLES else None

        if mdd_segment is not None:
            # Only print Fp1-Fp2 for MDD
            print(f"MDD Subject: Fp1-Fp2 Wavelet Coherence Plot.")
            plot_wavelet_coherence_only(mdd_segment, "Major Depressive Disorder (MDD)", TARGET_SFREQ, freqs, ch_x, ch_y)
        
        if hc_segment is None and mdd_segment is None:
            print("Could not load and segment data for any subject.")
    else:
        print("Required files not found. Cannot generate WTC comparison.")
