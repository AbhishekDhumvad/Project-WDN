"""
basic_example.py - a SIMPLIFIED illustration of the ideas behind the project
(leak localisation with a reduced number of pressure sensors).

This is NOT the research code.  It only shows the basic building blocks on a small network:
  1. leak "signatures": the pressure change at every junction caused by a leak (EPANET simulation),
  2. sensor ranking: switch all sensors off except one, rank the junctions by how much leak information they carry
     (variance), and the principal components (PCA) of the leak signatures,
  3. a small encoder-decoder (NumPy) that compresses the signatures of the selected sensors, with a classifier on the
     compressed (latent) code that names the leak location,
  4. the hierarchical position decision: first the half of the pipe, then the part (25 %) inside that half.

Leaks are modelled here simply as emitters at junctions (the full method places them anywhere along a pipe).
Detection, robust training, model-error handling, multiple leaks and the hydraulic check of the full method are not
included.

    pip install epyt numpy
    python basic_example.py ../networks/Hanoi.inp

Copyright (c) 2026 Abhishek Dhumvad. All rights reserved.
"""
import sys
import numpy as np
from epyt import epanet


# --------------------------------------------------------------------------------------------- 1. leak signatures
def pressures(G):
    """Steady-state (snapshot) junction pressures."""
    G.setTimeSimulationDuration(0)
    return np.asarray(G.getComputedHydraulicTimeSeries().Pressure[0], float)[: G.getNodeJunctionCount()]


def leak_signatures(inp, coefficients=(5.0, 10.0, 20.0), noise=0.05, copies=5, seed=0):
    """Pressure change of every junction for a leak (emitter) at every junction - several sizes, noisy copies."""
    rng = np.random.default_rng(seed)
    G = epanet(inp)
    nj = G.getNodeJunctionCount()
    base = pressures(G)
    X, y = [], []
    for j in range(1, nj + 1):
        for c in coefficients:
            G.setNodeEmitterCoeff(j, c)
            dp = pressures(G) - base                      # leak -> pressure drop pattern
            G.setNodeEmitterCoeff(j, 0.0)
            for _ in range(copies):                       # measurement noise
                X.append(dp + rng.normal(0, noise, nj)); y.append(j - 1)
    names = list(G.getNodeNameID())[:nj]
    G.unload()
    return np.array(X), np.array(y), names


# --------------------------------------------------------------------------------------------- 2. sensor ranking
def rank_single_sensors(X):
    """'One sensor on at a time': junctions ranked by the variance of their pressure change over all leaks."""
    return np.argsort(-X.var(0))


def pca_summary(X):
    """Explained variance of the principal components of the leak signatures."""
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2 / np.sum(s ** 2)
    return ev, int(np.searchsorted(np.cumsum(ev), 0.95) + 1)


# --------------------------------------------------------------------------------------------- 3. encoder-decoder
class TinyEncoderDecoder:
    """One hidden layer each way + a softmax classifier on the latent code (trained jointly, plain gradient descent)."""

    def __init__(self, n_in, n_cls, hidden=32, latent=8, seed=0):
        r = np.random.default_rng(seed); s = lambda a, b: r.normal(0, 1 / np.sqrt(a), (a, b))
        self.W = dict(E1=s(n_in, hidden), E2=s(hidden, latent), D1=s(latent, hidden), D2=s(hidden, n_in), C=s(latent, n_cls))

    def encode(self, X):
        return np.tanh(X @ self.W["E1"]) @ self.W["E2"]

    def fit(self, X, y, epochs=2000, lr=0.2, alpha=0.1):
        W = self.W; n = len(X); Y = np.eye(W["C"].shape[1])[y]
        for _ in range(epochs):
            H = np.tanh(X @ W["E1"]); Z = H @ W["E2"]                        # encoder
            Hd = np.tanh(Z @ W["D1"]); Xr = Hd @ W["D2"]                     # decoder (reconstruction)
            L = Z @ W["C"]; P = np.exp(L - L.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
            gXr = alpha * 2 * (Xr - X) / n; gL = (P - Y) / n                 # reconstruction + classification loss
            gD2 = Hd.T @ gXr; gHd = gXr @ W["D2"].T * (1 - Hd ** 2); gD1 = Z.T @ gHd
            gC = Z.T @ gL; gZ = gHd @ W["D1"].T + gL @ W["C"].T
            gE2 = H.T @ gZ; gH = gZ @ W["E2"].T * (1 - H ** 2); gE1 = X.T @ gH
            for k, g in (("E1", gE1), ("E2", gE2), ("D1", gD1), ("D2", gD2), ("C", gC)): W[k] -= lr * g
        return self

    def predict_proba(self, X):
        L = self.encode(X) @ self.W["C"]; P = np.exp(L - L.max(1, keepdims=True)); return P / P.sum(1, keepdims=True)


# --------------------------------------------------------------------------------------------- 4. half -> part
def decide_part(conf):
    """conf = confidences of the 4 parts of a pipe (0-25, 25-50, 50-75, 75-100 %): first the half, then the part."""
    conf = np.asarray(conf, float)
    half = 0 if conf[:2].sum() >= conf[2:].sum() else 1
    return 2 * half + int(np.argmax(conf[2 * half:2 * half + 2]))


# --------------------------------------------------------------------------------------------- demo
if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "../networks/Hanoi.inp"
    X, y, names = leak_signatures(inp)
    order = rank_single_sensors(X)
    ev, r95 = pca_summary(X)
    print(f"{len(names)} junctions; {r95} principal components hold 95 % of the leak information")
    print("best single sensors:", [names[i] for i in order[:8]])
    rng = np.random.default_rng(1); test = rng.random(len(X)) < 0.2
    for k in (len(names), len(names) // 2, len(names) // 4, 3):
        s = np.sort(order[:k]); sc = X[~test][:, s].std(0) + 1e-9
        m = TinyEncoderDecoder(k, len(names)).fit(X[~test][:, s] / sc, y[~test])
        acc = (m.predict_proba(X[test][:, s] / sc).argmax(1) == y[test]).mean()
        print(f"{k:3d} sensors: leak junction identified in {100 * acc:.0f} % of the test leaks")
    print("part decided from confidences [0.1, 0.2, 0.4, 0.3]:", ["0-25 %", "25-50 %", "50-75 %", "75-100 %"][decide_part([0.1, 0.2, 0.4, 0.3])])
