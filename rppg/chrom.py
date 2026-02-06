import numpy as np

def chrom_signal(R, G, B):
    # Normalize
    Rn = R / np.mean(R)
    Gn = G / np.mean(G)
    Bn = B / np.mean(B)

    # CHROM components
    X = 3 * Rn - 2 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn

    # Adaptive weighting
    alpha = np.std(X) / (np.std(Y) + 1e-8)

    return X - alpha * Y
