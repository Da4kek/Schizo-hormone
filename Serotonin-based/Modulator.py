import numpy as np


def modulation(epochs_list, all_channels, serotonin_level=0.8, histamine_level=0.2, receptor_activity=None):
    receptor_activity = receptor_activity or {
        '5-HT1A': {'frontal': 0.5, 'mPFC': 0.7},
        '5-HT2A': {'frontal': 0.8, 'mPFC': 0.6}
    }

    modulation_factor = serotonin_level * (.1 - histamine_level)
    modulated_epochs_list = []

    frontal_lobe_channels = [i for i, ch in enumerate(all_channels) if ch in [
        'F3', 'F4', 'F7', 'F8', 'Fz']]
    mPFC_channels = [i for i, ch in enumerate(
        all_channels) if ch in ['Fp1', 'Fp2']]

    for epochs in epochs_list:
        data = epochs.get_data()
        modulated_data = np.copy(data)
        for channel in frontal_lobe_channels:
            modulated_data[:, channel, :] *= (
                .1 - receptor_activity['5-HT1A']['frontal'] * modulation_factor)

        for channel in mPFC_channels:
            modulated_data[:, channel,
                           :] *= (.1 - receptor_activity['5-HT1A']['mPFC'] * modulation_factor)

        for channel in frontal_lobe_channels:
            modulated_data[:, channel, :] *= (
                .1 + receptor_activity['5-HT2A']['frontal'] * modulation_factor)

        for channel in mPFC_channels:
            modulated_data[:, channel,
                           :] *= (.1 + receptor_activity['5-HT2A']['mPFC'] * modulation_factor)

        modulated_epochs_list.append(modulated_data)

    return modulated_epochs_list
