# rppg/fusion5.py
# 5-branch dynamic weighted fusion:
#   Branch 1 : POS + FFT  (signal processing)
#   Branch 2 : LSTM
#   Branch 3 : RhythmFormer  (transformer)
#   Branch 4 : DeepPhys      (two-stream CNN)
#   Branch 5 : PhysNet       (3D CNN)
#
# Weights are computed dynamically from SNR and inter-branch agreement.

import numpy as np
from scipy.fft import fft, fftfreq


def _compute_snr(filtered_signal, fps=30.0, low=0.7, high=3.5):
    N     = len(filtered_signal)
    freqs = fftfreq(N, d=1.0/fps)
    mag   = np.abs(fft(filtered_signal))
    valid = (freqs >= low) & (freqs <= high)
    if not valid.any():
        return 0.1
    v_mag = mag[valid]
    return float(v_mag.max()**2 / (np.sum(v_mag**2) + 1e-8))


def dynamic_fusion_5branch(bpm_signal, bpm_lstm, bpm_rf,
                            bpm_dp, bpm_pn,
                            filtered_signal, fps=30.0):
    """
    5-branch SNR-adaptive fusion.

    Parameters
    ----------
    bpm_signal  : float  — POS + FFT estimate
    bpm_lstm    : float  — LSTM estimate
    bpm_rf      : float | None — RhythmFormer estimate
    bpm_dp      : float | None — DeepPhys estimate
    bpm_pn      : float | None — PhysNet estimate
    filtered_signal : array-like — bandpass-filtered rPPG waveform
    fps         : float

    Returns
    -------
    bpm_final, w_sp, w_lstm, w_rf, w_dp, w_pn, snr
    """
    snr      = _compute_snr(filtered_signal, fps)
    snr_norm = float(np.clip(snr / 0.3, 0, 1.0))

    # ── Base weights (tuned heuristics) ────────────────────────────────────
    # High SNR → favour classical signal processing
    # Low  SNR → favour deep models (more robust to noise)
    w_sp   = 0.20 + 0.20 * snr_norm        # 0.20 – 0.40
    w_lstm = 0.15 + 0.05 * (1-snr_norm)    # 0.15 – 0.20
    w_rf   = 0.30 - 0.15 * snr_norm        # 0.15 – 0.30
    w_dp   = 0.20 - 0.10 * snr_norm        # 0.10 – 0.20
    w_pn   = 0.15 - 0.05 * snr_norm        # 0.10 – 0.15

    # ── Agreement bonus ─────────────────────────────────────────────────────
    # When deep models agree with POS → boost them, reduce LSTM
    deep_estimates = [(bpm_rf, 'rf'), (bpm_dp, 'dp'), (bpm_pn, 'pn')]
    agreeing = [(b, t) for b, t in deep_estimates
                if b is not None and abs(b - bpm_signal) < 5.0]
    if len(agreeing) >= 2:
        w_sp   += 0.05
        for _, tag in agreeing:
            if tag == 'rf': w_rf += 0.05
            if tag == 'dp': w_dp += 0.05
            if tag == 'pn': w_pn += 0.05
        w_lstm = max(0.05, w_lstm - 0.10)

    # ── Handle missing deep models — redistribute weight ────────────────────
    def _redistribute(w_missing, w_sp, w_lstm, w_rf, w_dp, w_pn, tag):
        extra = w_missing
        w_sp   += extra * 0.40
        w_lstm += extra * 0.20
        others = [(w_rf,'rf'),(w_dp,'dp'),(w_pn,'pn')]
        remaining = [(w,t) for w,t in others if t != tag]
        share = extra * 0.40 / max(len(remaining), 1)
        if tag != 'rf': w_rf += share
        if tag != 'dp': w_dp += share
        if tag != 'pn': w_pn += share
        if tag == 'rf': w_rf = 0.0
        if tag == 'dp': w_dp = 0.0
        if tag == 'pn': w_pn = 0.0
        return w_sp, w_lstm, w_rf, w_dp, w_pn

    if bpm_rf is None:
        w_sp,w_lstm,w_rf,w_dp,w_pn = _redistribute(w_rf,w_sp,w_lstm,w_rf,w_dp,w_pn,'rf')
        bpm_rf = bpm_signal
    if bpm_dp is None:
        w_sp,w_lstm,w_rf,w_dp,w_pn = _redistribute(w_dp,w_sp,w_lstm,w_rf,w_dp,w_pn,'dp')
        bpm_dp = bpm_signal
    if bpm_pn is None:
        w_sp,w_lstm,w_rf,w_dp,w_pn = _redistribute(w_pn,w_sp,w_lstm,w_rf,w_dp,w_pn,'pn')
        bpm_pn = bpm_signal

    # ── Normalise ────────────────────────────────────────────────────────────
    total  = w_sp + w_lstm + w_rf + w_dp + w_pn
    w_sp   /= total
    w_lstm /= total
    w_rf   /= total
    w_dp   /= total
    w_pn   /= total

    # ── Weighted fusion ──────────────────────────────────────────────────────
    bpm_final = (w_sp   * bpm_signal +
                 w_lstm * bpm_lstm   +
                 w_rf   * bpm_rf     +
                 w_dp   * bpm_dp     +
                 w_pn   * bpm_pn)

    return bpm_final, w_sp, w_lstm, w_rf, w_dp, w_pn, snr
