from Utils.Helper import (
    compute_correlation_matrices, average_correlation_matrix,
    map_channels_to_regions, get_channel_positions
)
from Utils.Plot_utils import (
    plot_correlation_matrix, plot_topomap, plot_topomap_with_regions, plot_raster
)
import mne
import numpy as np

healthy_files = [f'Data/h{i:02}.edf' for i in range(1, 15)]
schizophrenic_files = [f'Data/s{i:02}.edf' for i in range(1, 15)]



def load_eeg_data(file_paths):
    raw_data = []
    for file in file_paths:
        raw = mne.io.read_raw_edf(file, preload=True)
        raw_data.append(raw)
    return raw_data


healthy_eeg_data = load_eeg_data(healthy_files)
schizophrenic_eeg_data = load_eeg_data(schizophrenic_files)

# Compute correlation matrices
healthy_corr_matrices = compute_correlation_matrices(healthy_eeg_data)
schizophrenic_corr_matrices = compute_correlation_matrices(
    schizophrenic_eeg_data)

# Average the correlation matrices
avg_healthy_corr_matrix = average_correlation_matrix(healthy_corr_matrices)
avg_schizo_corr_matrix = average_correlation_matrix(
    schizophrenic_corr_matrices)

# Plot correlation matrices
channel_names = healthy_eeg_data[0].info['ch_names']
plot_correlation_matrix(avg_healthy_corr_matrix,
                        'Average Correlation Matrix - Healthy Subjects', channel_names)
plot_correlation_matrix(avg_schizo_corr_matrix,
                        'Average Correlation Matrix - Schizophrenic Subjects', channel_names)

# Compute the difference in correlation matrices
correlation_difference = avg_healthy_corr_matrix - avg_schizo_corr_matrix

# Map channel names to anatomical regions and get channel positions
region_names = map_channels_to_regions(channel_names)
positions = get_channel_positions(channel_names)

# Plot topographic maps
avg_corr_diff_per_channel = np.mean(correlation_difference, axis=1)
plot_topomap(avg_corr_diff_per_channel, channel_names, positions,
             'Topomap of Average Correlation Difference')
plot_topomap_with_regions(avg_corr_diff_per_channel, region_names,
                          positions, 'Topomap of Correlation Difference with Anatomical Regions')

# Plot raster plot for the first 10 seconds of healthy data
sfreq = healthy_eeg_data[0].info['sfreq'] 
time = np.arange(0, healthy_eeg_data[0].n_times) / sfreq
plot_raster(healthy_eeg_data[0].get_data()[:, :int(sfreq*10)], time[:int(sfreq*10)],
            'Raster Plot - First 10 Seconds (Healthy Subject)', ch_names=channel_names)

# Plot raster plot for the first 10 seconds of schizophrenic data
plot_raster(schizophrenic_eeg_data[0].get_data()[:, :int(sfreq*10)], time[:int(sfreq*10)],
            'Raster Plot - First 10 Seconds (Schizophrenic Subject)', ch_names=channel_names)