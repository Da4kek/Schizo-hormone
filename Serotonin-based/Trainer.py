from reservoirpy.nodes import ESN
from Modulator import SerotoninEEGModulator
import numpy as np


class SerotoninESNTrainer(SerotoninEEGModulator):
    def __init__(self, n_reservoir=500, spectral_radius=0.95, sparsity=0.1, leak_rate=0.5,
                 input_scaling=0.5, input_shift=0.0, **kwargs):
        """
        Initialize the ESN trainer with serotonin modulation.

        :param n_reservoir: Number of reservoir neurons.
        :param spectral_radius: Spectral radius of the reservoir.
        :param sparsity: Sparsity of the reservoir connections.
        :param leak_rate: Leak rate of the reservoir neurons.
        :param input_scaling: Input scaling factor.
        :param input_shift: Input shift value.
        """
        super().__init__(**kwargs)
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.input_scaling = input_scaling
        self.input_shift = input_shift

        self.esn = ESN(
            units=n_reservoir,
            sr=spectral_radius,
            sparsity=sparsity,
            lr=leak_rate,
            input_scaling=input_scaling,
            input_shift=input_shift
        )

    def train_esn(self, eeg_data):
        """
        Train the ESN on the modulated EEG data.

        :param eeg_data: The original EEG data (n_epochs, n_channels, n_times).
        """
        modulated_eeg_data = self.modulate(eeg_data)
        n_epochs, n_channels, n_times = modulated_eeg_data.shape
        X = modulated_eeg_data.reshape(n_epochs, -1)
        y = np.roll(X, -1, axis=0)  

        X = X[:-1]  
        y = y[:-1]  

        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        self.esn.fit(X, y)

    def predict_future_epochs(self, eeg_data, n_predictions=100):
        """
        Predict the next `n_predictions` epochs using the trained ESN.

        :param eeg_data: The EEG data to start predictions from.
        :param n_predictions: Number of future epochs to predict.
        :return: Predicted future EEG data.
        """
        last_known_epoch = eeg_data[-1].reshape(1, -1)

        predictions = []
        current_input = last_known_epoch

        for _ in range(n_predictions):
            next_prediction = self.esn(current_input)
            predictions.append(next_prediction.flatten())
            current_input = next_prediction

        future_predictions = np.array(
            predictions).reshape(-1, eeg_data.shape[1], eeg_data.shape[2])
        return future_predictions
