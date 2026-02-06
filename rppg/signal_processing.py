import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, fs, low=0.75, high=3.0, order=3):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)

def estimate_bpm(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    valid = (freqs >= 0.75) & (freqs <= 3.0)
    peak_freq = freqs[valid][np.argmax(fft_mag[valid])]

    return peak_freq * 60