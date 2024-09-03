import numpy as np
import matplotlib.pyplot as plt


class SpikeEncoder:
    def __init__(self):
        self.m_in = 0
        self.std_in = 1
        self.m_v = 0
        self.std_v = 1

    def encode(self, x_in):
        V_x = np.zeros(len(x_in))
        X_encoded = np.zeros(len(x_in))

        for k in range(1, len(x_in)):
            V_x[k] = (x_in[k] - x_in[k - 1])

            self.m_in = ((k - 1) * self.m_in + V_x[k]) / k

            self.std_in = ((k - 1) * self.std_in +
                           abs(V_x[k] - self.std_in)) / k

            V_Nx = (V_x[k] - self.m_in) / self.std_in

            self.m_v = ((k - 1) * self.m_v + V_Nx) / k

            self.std_v = ((k - 1) * self.std_v + abs(V_Nx - self.std_v)) / k

            if V_Nx >= self.m_v + (self.std_v / 2):
                X_encoded[k] = 1
            else:
                X_encoded[k] = 0

        return X_encoded
