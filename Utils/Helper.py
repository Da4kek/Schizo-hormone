import mne 
import numpy as np 
from mne.preprocessing import ICA
from mne.io import concatenate_raws


def load_eeg_data(file_paths):
    raw_data = []
    for file in file_paths:
        raw = mne.io.read_raw_edf(file, preload=True)
        raw_data.append(raw)
    return raw_data


def compute_correlation_matrices(raw_data):
    correlation_matrices = []
    for raw in raw_data:
        data = raw.get_data()
        correlation_matrix = np.corrcoef(data)
        correlation_matrices.append(correlation_matrix)
    return correlation_matrices


def average_correlation_matrix(correlation_matrices):
    return np.mean(correlation_matrices, axis=0)


def map_channels_to_regions(channel_names):
    channel_to_region = {
        'Fp1': 'Left Prefrontal', 'Fp2': 'Right Prefrontal',
        'F7': 'Left Frontal', 'F8': 'Right Frontal', 'F3': 'Left Frontal', 'F4': 'Right Frontal',
        'Fz': 'Frontal Midline', 'FC5': 'Left Fronto-Central', 'FC6': 'Right Fronto-Central',
        'FC1': 'Left Fronto-Central', 'FC2': 'Right Fronto-Central',
        'T7': 'Left Temporal', 'T8': 'Right Temporal', 'C3': 'Left Central', 'C4': 'Right Central',
        'Cz': 'Central Midline', 'CP5': 'Left Centro-Parietal', 'CP6': 'Right Centro-Parietal',
        'CP1': 'Left Centro-Parietal', 'CP2': 'Right Centro-Parietal',
        'P7': 'Left Parietal', 'P8': 'Right Parietal', 'P3': 'Left Parietal', 'P4': 'Right Parietal',
        'Pz': 'Parietal Midline', 'O1': 'Left Occipital', 'O2': 'Right Occipital'
    }
    return [channel_to_region.get(ch, ch) for ch in channel_names]


def get_channel_positions(channel_names):
    montage = mne.channels.make_standard_montage('standard_1020')
    pos = np.array([montage.get_positions()['ch_pos'][ch][:2]
                   for ch in channel_names if ch in montage.ch_names])
    return pos


def extract_features(eeg_data):
    features = []
    for raw in eeg_data:
        corr_matrix = np.corrcoef(raw.get_data())
        features.append(corr_matrix.flatten())
    return np.array(features)


def extract_region_features(eeg_data, brain_regions):
    features = {}
    for region, channels in brain_regions.items():
        region_features = []
        for raw in eeg_data:
            data = raw.copy().pick_channels(channels).get_data()
            corr_matrix = np.corrcoef(data)
            region_features.append(corr_matrix.flatten())
        features[region] = np.array(region_features)
    return features



def simulate_region_influence(features_by_region, region_name, influence_factor):
    simulated_features = {}
    for region, features in features_by_region.items():
        if region == region_name:
            simulated_features[region] = features * influence_factor
        else:
            simulated_features[region] = features
    return simulated_features


def verify_simulation_with_diff(simulated_features, diff_features):
    verification_results = {}
    for region in simulated_features.keys():
        simulated_diff = np.mean(
            simulated_features[region] - diff_features[region], axis=0)
        verification_results[region] = np.linalg.norm(simulated_diff)
    return verification_results


def preprocess_eeg(raw, eog_ch_names=None):
    raw.set_eeg_reference('average', projection=True)

    raw.filter(0.1, 40., fir_design='firwin')

    ica = ICA(n_components=15, random_state=97)
    ica.fit(raw)

    if eog_ch_names:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_ch_names)
        ica.exclude = eog_indices
    else:
        print("No EOG channels specified, skipping EOG artifact detection.")

    print("Skipping ECG artifact detection.")

    raw = ica.apply(raw)

    return raw

