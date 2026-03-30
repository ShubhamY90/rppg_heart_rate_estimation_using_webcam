def dynamic_fusion_3branch(bpm_signal, bpm_lstm,
                            bpm_rf, filtered_signal):
    """
    YOUR CONTRIBUTION:
    3-branch adaptive fusion:
      - Branch 1: Signal Processing (POS + FFT)
      - Branch 2: LSTM model
      - Branch 3: RhythmFormer transformer
    Weights computed dynamically from SNR + agreement
    """
    from scipy.fft import fft, fftfreq
    import numpy as np

    # ── Compute SNR ───────────────────────────────────────
    fps  = 30.0
    N    = len(filtered_signal)
    freqs = fftfreq(N, d=1.0/fps)
    mag   = np.abs(fft(filtered_signal))
    valid = (freqs >= 0.7) & (freqs <= 3.5)
    if valid.any():
        v_mag = mag[valid]
        snr   = float(v_mag.max()**2 /
                      (np.sum(v_mag**2) + 1e-8))
    else:
        snr = 0.1

    snr_norm = np.clip(snr / 0.3, 0, 1.0)

    # ── Base weights ──────────────────────────────────────
    # Higher SNR → trust signal processing more
    w_sp   = 0.25 + 0.25 * snr_norm      # 0.25 – 0.50
    w_lstm = 0.25 + 0.10 * (1-snr_norm)  # 0.25 – 0.35
    w_rf   = 0.50 - 0.35 * snr_norm      # 0.15 – 0.50

    # ── Agreement bonus ───────────────────────────────────
    # If RF and SP agree → boost both, reduce LSTM
    if bpm_rf is not None and bpm_signal is not None:
        rf_sp_diff = abs(bpm_rf - bpm_signal)
        if rf_sp_diff < 5.0:       # strong agreement
            w_sp  += 0.10
            w_rf  += 0.10
            w_lstm = max(0.10, w_lstm - 0.20)

    # ── Handle missing RF ─────────────────────────────────
    if bpm_rf is None:
        # Redistribute RF weight to SP and LSTM
        extra  = w_rf
        w_sp  += extra * 0.6
        w_lstm += extra * 0.4
        w_rf   = 0.0
        bpm_rf = bpm_signal  # fallback value (unused)

    # ── Normalize weights to sum = 1 ──────────────────────
    total  = w_sp + w_lstm + w_rf
    w_sp  /= total
    w_lstm /= total
    w_rf  /= total

    # ── Weighted fusion ───────────────────────────────────
    bpm_final = (w_sp   * bpm_signal +
                 w_lstm * bpm_lstm   +
                 w_rf   * bpm_rf)

    return bpm_final, w_sp, w_lstm, w_rf, snr
