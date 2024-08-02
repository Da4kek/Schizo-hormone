import mne
import numpy as np
import os
import pyedflib

from Utils.Helper import load_eeg_data

healthy_files = [f'Data/h{i:02}.edf' for i in range(1, 15)]
schizophrenic_files = [f'Data/s{i:02}.edf' for i in range(1, 15)]


healthy_eeg_data = load_eeg_data(healthy_files)
schizophrenic_eeg_data = load_eeg_data(schizophrenic_files)


def compute_eeg_differences(healthy_data, schizophrenic_data):
    diff_data = []
    for h, s in zip(healthy_data, schizophrenic_data):
        h_data = h.get_data()
        s_data = s.get_data()
        min_length = min(h_data.shape[1], s_data.shape[1])
        h_data_trimmed = h_data[:, :min_length]
        s_data_trimmed = s_data[:, :min_length]

        diff = h_data_trimmed - s_data_trimmed
        diff_data.append(diff)
    return diff_data


diff_eeg_data = compute_eeg_differences(
    healthy_eeg_data, schizophrenic_eeg_data)

def save_eeg_differences_as_edf(diff_data, template_raw, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, diff in enumerate(diff_data):
        n_channels, n_samples = diff.shape
        channel_names = template_raw[i].info['ch_names']
        sfreq = int(template_raw[i].info['sfreq'])
        dmin = -32768
        dmax = 32767
        pmin = diff.min()
        pmax = diff.max()

        physical_min = pmin
        physical_max = pmax
        digital_min = dmin
        digital_max = dmax
        scaling = (physical_max - physical_min) / (digital_max - digital_min)
        diff_scaled = (diff - physical_min) / scaling + digital_min

        output_path = os.path.join(output_dir, f'diff_{i+1:02}.edf')
        with pyedflib.EdfWriter(output_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
            signal_headers = [{
                'label': channel_names[j],
                'dimension': 'uV',
                'sample_rate': sfreq,
                'physical_min': physical_min,
                'physical_max': physical_max,
                'digital_min': digital_min,
                'digital_max': digital_max,
                'transducer': '',
                'prefilter': ''
            } for j in range(n_channels)]
            f.setSignalHeaders(signal_headers)
            f.writeSamples(diff_scaled)
        print(f'Saved: {output_path}')


save_eeg_differences_as_edf(diff_eeg_data, healthy_eeg_data, 'Diff_EEG')

print("All difference EEG files have been generated and saved.")
