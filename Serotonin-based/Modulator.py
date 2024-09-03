import numpy as np


class SerotoninEEGModulator:
    def __init__(self, serotonin_level=0.8, histamine_level=0.2, receptor_activity=None):
        self.serotonin_level = serotonin_level
        self.histamine_level = histamine_level
        self.receptor_activity = receptor_activity or {
            '5-HT1A': {'frontal': 0.5, 'mPFC': 0.7},
            '5-HT2A': {'frontal': 0.8, 'mPFC': 0.6}
        }

    def modulate(self, eeg_data,all_channels):
        modulation_factor = self.serotonin_level * (1 - self.histamine_level)
        modulated_eeg = np.copy(eeg_data)


        frontal_lobe_channels = [i for i, ch in enumerate(all_channels) if ch in [
        'F3', 'F4', 'F7', 'F8', 'Fz']]
        mPFC_channels = [i for i, ch in enumerate(all_channels) if ch in ['Fp1', 'Fp2']]

        for channel in frontal_lobe_channels:
            modulated_eeg[:, channel, :] *= (
                1 - self.receptor_activity['5-HT1A']['frontal'] * modulation_factor)

        for channel in mPFC_channels:
            modulated_eeg[:, channel, :] *= (
                1 - self.receptor_activity['5-HT1A']['mPFC'] * modulation_factor)

        for channel in frontal_lobe_channels:
            modulated_eeg[:, channel, :] *= (
                1 + self.receptor_activity['5-HT2A']['frontal'] * modulation_factor)

        for channel in mPFC_channels:
            modulated_eeg[:, channel, :] *= (
                1 + self.receptor_activity['5-HT2A']['mPFC'] * modulation_factor)

        modulated_eeg = (modulated_eeg - np.mean(modulated_eeg)
                         ) / np.std(modulated_eeg)

        return modulated_eeg
