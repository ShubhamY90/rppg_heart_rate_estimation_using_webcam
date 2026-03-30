import os
import numpy as np
from tensorflow.keras.models import load_model

class LSTMPredictor:

    def __init__(self):

        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )

        model_path = os.path.join(base_dir, "rppg_lstm_model.keras")

        self.model = load_model(model_path)

    def predict(self, signal):

        signal = np.array(signal)

        # match typical rPPG LSTM window
        signal = signal[-150:]

        # normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

        signal = signal.reshape(1,150,1)

        bpm = self.model.predict(signal, verbose=0)

        return float(bpm[0][0])