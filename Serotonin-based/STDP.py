import numpy as np
import mne
import matplotlib.pyplot as plt


class STDPNetwork:
    def __init__(self, num_neurons, A_plus, A_minus, tau_plus, tau_minus, learning_rate=0.01):
        self.num_neurons = num_neurons
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.learning_rate = learning_rate
        # Initialize synaptic weights randomly
        self.weights = np.random.rand(num_neurons, num_neurons)

    def update_weights(self, delta_t):
        # Ensure that delta_t is compatible with the weights matrix
        if delta_t.shape != self.weights.shape:
            raise ValueError(
                f"Shape mismatch: delta_t shape {delta_t.shape} and weights shape {self.weights.shape} do not match.")

        # Vectorized STDP weight update
        delta_w = np.where(delta_t <= 0,
                           self.A_plus * np.exp(delta_t / self.tau_plus),
                           0)
        self.weights += self.learning_rate * delta_w

    def train(self, spike_trains, num_epochs):
        # Number of features should match the number of neurons
        num_neurons = spike_trains.shape[1]
        if num_neurons != self.num_neurons:
            raise ValueError(
                f"Mismatch in number of neurons: spike_trains has {num_neurons} features, but network was initialized with {self.num_neurons} neurons.")

        for epoch in range(num_epochs):
            for t in range(spike_trains.shape[0]):
                # Target spike train for all neurons at time t
                S_d = spike_trains[t, :]

                # Compute time differences between all neuron pairs (vectorized)
                # delta_t[i, j] = S_j[t] - S_i[t]
                delta_t = np.subtract.outer(S_d, S_d)

                # Update weights based on delta_t
                self.update_weights(delta_t)

    def predict(self, input_spike_train, prediction_horizon=100):
        predicted_spikes = []
        # Start prediction from the last known spike train
        current_input = input_spike_train[:, -1]

        for _ in range(prediction_horizon):
            # Compute next spikes based on current input and weights
            next_spikes = np.dot(
                self.weights, current_input) > np.random.rand(self.num_neurons)
            predicted_spikes.append(next_spikes.astype(int))
            current_input = next_spikes  # Update the input for the next prediction step

        predicted_spikes = np.array(predicted_spikes).T
        return predicted_spikes

    def trend_analysis(self, predicted_spikes):
        # Simple trend analysis: Count spikes and check for increases or decreases
        spike_count = np.sum(predicted_spikes, axis=0)
        trends = np.diff(spike_count)

        trend_changes = []
        for i, change in enumerate(trends):
            if change > 0:
                trend_changes.append((i, "Increase"))
            elif change < 0:
                trend_changes.append((i, "Decrease"))

        return trend_changes


