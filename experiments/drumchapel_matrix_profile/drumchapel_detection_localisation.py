"""
drumchapel_detection_localisation.py  -  DRUMCHAPEL: leak DETECTION + LOCALISATION + ONE MONTH OF OPERATION
=============================================================================================================
Stand-alone script (only needs Drumchapel.inp).  Three questions:

  1. WHEN  - Matrix Profile leak DETECTION (same method as Leak_detection.py)
             Every 1-hour piece of the sensor changes is compared with leak-free days; pieces never seen in normal
             operation give an alarm -> detected start time and time window.  Trained on leak-free days only.
  2. WHERE - Encoder-decoder EMBEDDING LOCALISATION (same method as leak_localisation_net1.py)
             Leaks are simulated at many positions on EVERY pipe -> fingerprints -> encoder-decoder -> embedding
             database.  The fingerprint of a detected leak (taken over the DETECTED window) is matched with the
             database -> leaking pipe + position on the pipe (first 70 % or last 30 %).
  3. MONTH - 30 days of operation in 15-min steps: daily demand cycle (morning / evening peaks), weekends,
             slow trend, random fluctuation, and leaks that start, run for hours or days and are then repaired.
             The detector + localiser run over the whole month -> water demand vs time, leakage, detections.

Sensors: hydraulic head (= pressure + elevation) at every junction, flow and velocity in every pipe.
TEST data is unseen: new random days, new leak positions / sizes / times, sensor noise and unknown demand changes.

Outputs (folder drumchapel_detection_localisation_outputs, next to this script)
  training_data.xlsx, all_embeddings.csv, test_data.xlsx, localisation_results.xlsx, month_simulation.xlsx,
  confusion_matrix.png, embedding_space.png, training_curve.png, month_demand.png, leak_localisation_report.html
First run: simulates + trains (about 10-20 min, uses several CPU cores). Later runs reuse trained_model_cache.pkl.
Requirements: pip install epyt numpy pandas openpyxl matplotlib   (plotly optional - lets the HTML work offline)
"""

# =====================================================================================================
# PART A - HELPERS, EPANET NETWORK AND SIMULATION
# =====================================================================================================
from pathlib import Path
import os, sys, json, heapq, pickle, time, contextlib, tempfile, webbrowser, warnings, shutil
from multiprocessing import util as mp_util
import numpy as np
import pandas as pd
from epyt import epanet

warnings.filterwarnings("ignore")
STEP_S = 900          # 15-minute hydraulic + reporting step
# Sensor features used for detection / localisation (can be overridden in the SETTINGS section):
#   "H" = hydraulic head at every junction (pressure + elevation)    "Q" = flow in every open pipe
#   "V" = velocity in every open pipe                                 "Qin" = only the inlet flow meters
FEATURES = ("H", "Q", "V")
FRONT_SPLIT = 0.70    # leak is in the "front" zone if position < 70 % of pipe length, else "rear" zone


@contextlib.contextmanager
def quiet():
    """Silence EPyT / EPANET messages ("EPANET version ... loaded", "Close toolkit ...").
    Redirects both the Python streams (Spyder / Jupyter / VS Code / IDLE) and the OS-level output."""
    for st in (sys.stdout, sys.stderr):        # streams can be None (pythonw / IDE worker processes)
        try: st.flush()
        except Exception: pass
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
        try:
            saved, null = [os.dup(1), os.dup(2)], os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, 1); os.dup2(null, 2)
        except Exception:
            saved = None
        try:
            yield
        finally:
            if saved:
                os.dup2(saved[0], 1); os.dup2(saved[1], 2)
                for fd in saved + [null]: os.close(fd)


def hhmm(h):
    m = int(round(h * 60)); return f"{m // 60:02d}:{m % 60:02d}"


def hours(text):
    """'06:30' -> 6.5 h.  (A float like 8.30 would mean 8 h 18 min, not 8:30.)"""
    if isinstance(text, str):
        h, m = text.split(":"); return int(h) + int(m) / 60
    return float(text)


def zone(p):
    return f"0-{FRONT_SPLIT:.0%}" if p < FRONT_SPLIT else f"{FRONT_SPLIT:.0%}-100%"


class Network:
    """EPANET network + three kinds of model:
       * the real network (with or without a leak) -> 'measured' data
       * the SCADA-driven expected model: tanks become fixed-head nodes whose head and the pump status are
         copied from the measurements at every 15-min step -> 'expected' data. A leak therefore shows up only
         while it is active (no tank / pump after-effects in the residuals)."""

    def __init__(self, inp, daily_pattern=None):
        self.inp = Path(inp)
        if not self.inp.exists():
            raise FileNotFoundError(f"INP file not found: {self.inp}")
        self.lines = self.inp.read_text(encoding="utf-8", errors="ignore").splitlines()
        self.daily_pattern = daily_pattern
        # private working copy: EPyT writes '<name>_temp.inp' next to the file it opens -> every process (parallel
        # workers) gets its own folder, so they never fight over one temp file (Windows locks open files)
        self.tmp = Path(tempfile.mkdtemp(prefix="leakinp_"))
        mp_util.Finalize(None, shutil.rmtree, args=(str(self.tmp), True), exitpriority=0)   # removed at exit (also in workers)
        self.work_inp = self.tmp / self.inp.name
        self.work_inp.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        with quiet(): G = epanet(str(self.work_inp))
        try:
            nid, lid = list(G.getNodeNameID()), list(G.getLinkNameID())
            conn, status = G.getNodesConnectingLinksID(), G.getLinkInitialStatus()
            self.junctions = list(G.getNodeJunctionNameID())
            self.tanks = list(G.getNodeTankNameID()) if G.getNodeTankCount() else []
            self.pumps = list(G.getLinkPumpNameID()) if G.getLinkPumpCount() else []
            self.pipes_all = list(G.getLinkPipeNameID())
            self.ends = {l: (str(conn[i][0]), str(conn[i][1])) for i, l in enumerate(lid)}
            self.length = {l: float(x) for l, x in zip(lid, G.getLinkLength())}
            # pipes that are CLOSED in the INP carry no flow -> a leak there is invisible; they are excluded
            self.pipes = [p for p in self.pipes_all if status[lid.index(p)] == 1]
            self.elev = dict(zip(nid, map(float, G.getNodeElevations())))
            c = G.getNodeCoordinates()
            xy = lambda d, i: float(d.get(i, d.get(str(i), 0.0)))
            self.xy = {n: (xy(c["x"], i + 1), xy(c["y"], i + 1)) for i, n in enumerate(nid)}
            try: self.units = (G.units.NodePressureUnits, G.units.LinkFlowUnits, G.units.LinkVelocityUnits)
            except Exception: self.units = ("pressure", "flow", "velocity")
        finally:
            with quiet(): G.unload()
        self.len_unit = "ft" if "psi" in str(self.units[0]).lower() else "m"
        # SCADA flow meters: pipes leaving a tank or a pump (network inlets)
        inlet_nodes = set(self.tanks) | {self.ends[p][1] for p in self.pumps}
        self.flow_sensors = [i for i, p in enumerate(self.pipes_all) if set(self.ends[p]) & inlet_nodes]
        self.open_idx = [self.pipes_all.index(p) for p in self.pipes]
        self.z = np.array([self.elev[j] for j in self.junctions])          # junction elevations
        self.vertices, out, sec = {}, [], None
        for line in self.lines:     # read vertices + write the SCADA-driven expected model
            s = line.split(";")[0].strip(); tok = s.split()
            if s.startswith("["):
                sec = s.upper(); out.append("[RESERVOIRS]" if sec == "[TANKS]" else line); continue
            if sec == "[VERTICES]" and len(tok) >= 3:
                self.vertices.setdefault(tok[0], []).append((float(tok[1]), float(tok[2])))
            if sec == "[TANKS]" and len(tok) >= 3:
                out.append(f"{tok[0]} {float(tok[1]) + float(tok[2]):.4f}"); continue
            if sec in ("[CONTROLS]", "[RULES]") and tok: continue
            out.append(line)
        self.scada_inp = self.tmp / "scada_model.inp"
        self.scada_inp.write_text("\n".join(out), encoding="utf-8")
        self._dist = {}

    # ---------- geometry -------------------------------------------------------------------------
    def polyline(self, pipe):
        a, b = self.ends[pipe]
        return [self.xy[a]] + self.vertices.get(pipe, []) + [self.xy[b]]

    def point_on_pipe(self, pipe, frac):
        pts = np.array(self.polyline(pipe)); seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.cumsum(seg); d = frac * cum[-1]; k = min(int(np.searchsorted(cum, d)), len(seg) - 1)
        r = (d - (cum[k] - seg[k])) / max(seg[k], 1e-12)
        return tuple(pts[k] + r * (pts[k + 1] - pts[k]))

    def _dijkstra(self, src):
        if src not in self._dist:
            adj = {}
            for p in self.pipes:
                a, b = self.ends[p]; adj.setdefault(a, []).append((b, self.length[p])); adj.setdefault(b, []).append((a, self.length[p]))
            d, pq = {src: 0.0}, [(0.0, src)]
            while pq:
                du, u = heapq.heappop(pq)
                if du > d.get(u, np.inf): continue
                for v, w in adj.get(u, []):
                    if du + w < d.get(v, np.inf): d[v] = du + w; heapq.heappush(pq, (du + w, v))
            self._dist[src] = d
        return self._dist[src]

    def leak_distance(self, p1, x1, p2, x2):
        """Distance ALONG THE PIPES between leak point (p1, x1) and (p2, x2) - the localisation error."""
        if p1 == p2: return abs(x1 - x2) * self.length[p1]
        (a1, b1), (a2, b2) = self.ends[p1], self.ends[p2]
        d1 = {a1: x1 * self.length[p1], b1: (1 - x1) * self.length[p1]}
        d2 = {a2: x2 * self.length[p2], b2: (1 - x2) * self.length[p2]}
        return min(d1[u] + self._dijkstra(u).get(v, np.inf) + d2[v] for u in d1 for v in d2)

    def blocks(self, features=None):
        """Sensor groups: (feature, column indices, column names)."""
        out = []
        for f in features or FEATURES:
            if f == "H": out.append(("H", None, [f"H_{j}" for j in self.junctions]))
            if f in ("Q", "V"): out.append((f, self.open_idx, [f"{f}_{self.pipes_all[i]}" for i in self.open_idx]))
            if f == "Qin": out.append(("Q", self.flow_sensors, [f"Qin_{self.pipes_all[i]}" for i in self.flow_sensors]))
        return out

    @property
    def sensor_names(self):
        return [n for _, _, names in self.blocks() for n in names]

    def adjacent(self, p1, p2):
        return p1 != p2 and bool(set(self.ends[p1]) & set(self.ends[p2]))

    # ---------- models ---------------------------------------------------------------------------
    def leak_inp(self, leaks):
        """Copy of the INP with every (pipe, pos) in `leaks` split at pos (0-1) and a junction LEAK_<pipe> inserted."""
        leaks = dict(leaks); lines_j = []
        for pipe, pos in leaks.items():
            if not 0 < pos < 1: raise ValueError("leak position must be between 0 and 1")
            a, b = self.ends[pipe]
            lines_j.append(f"LEAK_{pipe}\t{self.elev[a] + pos * (self.elev[b] - self.elev[a]):.4f}")   # interpolated elevation
        out, sec = [], None
        for line in self.lines:
            s = line.strip(); tok = s.split(";")[0].split()
            if s.startswith("["):
                if sec == "[JUNCTIONS]": out += lines_j
                sec = s.upper(); out.append(line); continue
            if tok and tok[0] in leaks:
                pipe = tok[0]
                if sec == "[PIPES]":
                    a, b = self.ends[pipe]; pos = leaks[pipe]; L, rest = float(tok[3]), " ".join(tok[4:])
                    out += [f"{pipe}_A {a} LEAK_{pipe} {L * pos:.4f} {rest}", f"{pipe}_B LEAK_{pipe} {b} {L * (1 - pos):.4f} {rest}"]
                    continue
                if sec in ("[VERTICES]", "[STATUS]"): continue
            out.append(line)
        path = self.tmp / f"leak_{abs(hash(tuple(sorted(leaks)))) % 10 ** 8}.inp"
        path.write_text("\n".join(out), encoding="utf-8")
        return path

    def open(self, pipe=None, pos=None, scada=False, leaks=None):
        """Load the real network (pipe=None), a leak model (one leak: pipe + pos; several: leaks=[(pipe, pos), ...])
        or the SCADA-driven expected model.  EPyT keeps ONE model open at a time -> close() before the next."""
        leaks = list(leaks or ([(pipe, pos)] if pipe is not None else []))
        path = self.scada_inp if scada else (self.leak_inp(leaks) if leaks else self.work_inp)
        with quiet(): G = epanet(str(path))
        G.setTimeHydraulicStep(STEP_S); G.setTimeReportingStep(STEP_S)
        if self.daily_pattern is not None:
            G.setPattern(1, list(self.daily_pattern))
        lid, lp = list(G.getLinkNameID()), [p for p, _ in leaks]
        return dict(G=G, j=[G.getNodeIndex(n) - 1 for n in self.junctions],
                    la=[lid.index(f"{p}_A" if p in lp else p) for p in self.pipes_all],
                    lb=[lid.index(f"{p}_B" if p in lp else p) for p in self.pipes_all],
                    tk=[G.getNodeIndex(n) for n in self.tanks], pu=[G.getLinkIndex(p) for p in self.pumps],
                    lpipes=lp, lnodes=[G.getNodeIndex(f"LEAK_{p}") for p in lp])

    @staticmethod
    def close(h):
        with quiet(): h["G"].unload()

    def run(self, h, hours_=24.0, coef=0.0, on=0.0, off=0.0, dmult=1.0, dnoise=None, bc=None, schedule=None):
        """Extended-period simulation; results kept only at the 15-min report steps (EPANET's extra
        intermediate steps caused by tank/pump control events are skipped).
        Leak emitter coefficient = coef for on <= t < off (hours).  dnoise: per-step demand factors that the
        model does not know.  bc: measured tank heads H + pump status S imposed on the expected model.
        Returns t (h), P (junction pressure), Q / V (pipe flow / velocity), H (tank head), S (pump status)."""
        G = h["G"]; G.setTimeSimulationDuration(int(round(hours_ * 3600)))
        dm_arr = np.ndim(dmult) > 0                    # dmult may be one value or one value per 15-min step
        G.setOptionsPatternDemandMultiplier(float(dmult[0] if dm_arr else dmult))
        out = {k: [] for k in "tPQVHSDL"}
        sched = schedule if schedule is not None else {p_: [(coef, on, off)] for p_ in h["lpipes"]}
        with quiet():
            G.openHydraulicAnalysis(); G.initializeHydraulicAnalysis()
            while True:
                hr = G.getTimeHTime() / 3600; k = int(round(hr * 4))
                act = []
                for p_, node in zip(h["lpipes"], h["lnodes"]):
                    c_ = next((c for c, a, b in sched.get(p_, []) if c > 0 and a <= hr + 1e-9 < b), 0.0)
                    G.setNodeEmitterCoeff(node, c_); act.append(c_ > 0)
                if dnoise is not None or dm_arr:
                    base = dmult[min(k, len(dmult) - 1)] if dm_arr else dmult
                    G.setOptionsPatternDemandMultiplier(float(base * (dnoise[min(k, len(dnoise) - 1)] if dnoise is not None else 1.0)))
                if bc is not None and k < len(bc["t"]):
                    for i, n in enumerate(h["tk"]): G.setNodeElevations(n, float(bc["H"][k][i]))
                    for i, l in enumerate(h["pu"]): G.setLinkStatus(l, int(bc["S"][k][i]))
                t = G.runHydraulicAnalysis()
                if t % STEP_S == 0 and (not out["t"] or t / 3600 > out["t"][-1]):
                    q, v, hd, st = G.getLinkFlows(), G.getLinkVelocity(), G.getNodeHydraulicHead(), G.getLinkStatus()
                    out["t"].append(t / 3600); out["P"].append(G.getNodePressure()[h["j"]])
                    out["Q"].append(0.5 * (q[h["la"]] + q[h["lb"]])); out["V"].append(0.5 * (v[h["la"]] + v[h["lb"]]))
                    out["H"].append([hd[n - 1] for n in h["tk"]]); out["S"].append([st[l - 1] for l in h["pu"]])
                    ad = G.getNodeActualDemand()            # consumer demand (all junctions) + leak outflow
                    out["D"].append(float(ad[h["j"]].sum())); out["L"].append(float(sum(ad[n_ - 1] for n_, a_ in zip(h["lnodes"], act) if a_)))
                if G.nextHydraulicAnalysisStep() <= 0: break
            G.closeHydraulicAnalysis()
        return {k: np.array(v, float) for k, v in out.items()}

    def expected(self, obs, dmult=1.0, h=None):
        """Model-expected sensor values under the measured tank levels / pump status (h: open SCADA model)."""
        own = h is None
        h = self.open(scada=True) if own else h
        try: return self.run(h, obs["t"][-1], dmult=dmult, bc=obs)
        finally:
            if own: self.close(h)

    def simulate_leak(self, pipe, pos, coef, on, off, hours_=24.0, dmult=1.0, dnoise=None):
        """pipe=None -> leak-free day."""
        h = self.open(pipe, pos)
        try: return self.run(h, hours_, coef, on, off, dmult, dnoise)
        finally: self.close(h)


def demand_noise(rng, hours_, std):
    """Smooth random fluctuation of all demands (per 15-min step) that the model does NOT know."""
    if std <= 0: return None
    n = int(hours_ * 4) + 1
    return 1.0 + np.convolve(rng.normal(0, std, n + 4), np.ones(4) / 2, mode="same")[:n]


def residual(net, obs, exp, rng=None, p_noise=0.0, q_noise_rel=0.01, features=None):
    """Sensor CHANGE = measured - model-expected value (+ sensor noise), for every sensor group:
       H : hydraulic head  = pressure + elevation  at every junction        (noise p_noise)
       Q : flow in every open pipe,  V : velocity in every open pipe          (noise q_noise_rel x mean value)
    Outside a leak the change is ~0 (noise only); during a leak heads drop and flows / velocities change."""
    parts = []
    for f, idx, _ in net.blocks(features):
        if f == "H":
            R = (obs["P"] + net.z) - (exp["P"] + net.z)
            if rng is not None and p_noise > 0: R = R + rng.normal(0, p_noise, R.shape)
        else:
            o, e = obs[f][:, idx], exp[f][:, idx]; R = o - e
            if rng is not None and q_noise_rel > 0:
                sd = q_noise_rel * (np.abs(e).mean(0) + 0.01 * np.abs(e).mean())
                R = R + rng.normal(0, 1, R.shape) * sd
        parts.append(R)
    return np.hstack(parts)


def normal_days(net, rng, n_days, p_noise, d_noise):
    """Leak-free days with unknown demand fluctuation + sensor noise -> residuals of normal operation."""
    obs, h = [], net.open()
    for _ in range(n_days): obs.append(net.run(h, 24.0, dnoise=demand_noise(rng, 24.0, d_noise)))
    net.close(h)
    return [residual(net, o, net.expected(o), rng, p_noise) for o in obs]


def sensor_scale(net, days, features=None):
    """Per-sensor normal variability (std on leak-free days). Each sensor GROUP (head / flow / velocity) gets
    the same total weight, so the 136 heads do not outvote the flows or vice versa."""
    std, k, out = np.concatenate(days).std(0), 0, []
    for _, _, names in net.blocks(features):
        sd = std[k:k + len(names)]; sd = np.maximum(sd, 0.1 * np.median(sd) + 1e-9)   # floor for ~constant sensors
        out.append(sd * np.sqrt(len(names) * len(net.blocks(features)))); k += len(names)
    return np.concatenate(out)

# =====================================================================================================
# PART B - LEAK DETECTION WITH THE MATRIX PROFILE  (answers: WHEN did the leak start, how long did it last)
# =====================================================================================================
def mp_join(A, B, m):
    """Multivariate AB-join MATRIX PROFILE (non-normalised).
    For every length-m subsequence (window) of A: distance to its nearest-neighbour window in B.
    B = leak-free days, so a large value means 'this behaviour was never seen in normal operation'.
    Non-normalised on purpose: a leak is a pressure DROP / flow RISE which z-normalisation would erase."""
    WA = np.lib.stride_tricks.sliding_window_view(A, m, axis=0).reshape(len(A) - m + 1, -1)
    WB = np.lib.stride_tricks.sliding_window_view(B, m, axis=0).reshape(len(B) - m + 1, -1)
    d2 = (WA ** 2).sum(1)[:, None] + (WB ** 2).sum(1)[None, :] - 2 * WA @ WB.T
    return np.sqrt(np.maximum(d2.min(1), 0) / m)


def point_score(mp, n, m):
    """Score of time step t = min of the matrix profile over all windows containing t:
    high only if EVERY window through t is abnormal -> sharp start / end times."""
    pad = np.concatenate([np.full(m - 1, np.inf), mp, np.full(m - 1, np.inf)])
    s = np.lib.stride_tricks.sliding_window_view(pad, m).min(1)[:n]
    return np.where(np.isfinite(s), s, 0.0)


class MatrixProfileDetector:
    def __init__(self, scale, m=4, factor=1.2, min_steps=2):
        self.scale, self.m, self.factor, self.min_steps = scale, m, factor, min_steps

    def fit(self, ref_days, null_days):
        """ref_days : leak-free residual days = what 'normal' looks like (matrix-profile reference).
        null_days: other leak-free days -> threshold = factor x highest score seen there (no false alarms)."""
        self.ref = np.concatenate([r / self.scale for r in ref_days])
        self.threshold = self.factor * max(self.score(r).max() for r in null_days)
        self.threshold1 = self.factor * max(self.score(r, 1).max() for r in null_days)   # single-step version
        return self

    def score(self, R, m=None):
        m = m or self.m; X = R / self.scale
        return point_score(mp_join(X, self.ref, m), len(X), m)

    def detect(self, t, R):
        """Return (score, events).  Event: start_h, end_h, peak, strength, i0, i1 - strongest event first.
        Boundaries are refined with the 1-step matrix profile (fixes windows that only partly cover a leak)."""
        s, s1 = self.score(R), self.score(R, 1); on = s > self.threshold
        ev, i = [], 0
        while i < len(on):
            if not on[i]: i += 1; continue
            j = i
            while j + 1 < len(on) and (on[j + 1] or (j + 2 < len(on) and on[j + 2])): j += 1   # bridge 1-step gaps
            a, b = i, j      # trim at most m-1 edge steps that only look abnormal because their window does
            while a < min(i + self.m - 1, j) and s1[a] <= self.threshold1: a += 1
            if s1[a] <= self.threshold1: a = i                     # weak leak: keep the matrix-profile start
            while b > max(j - self.m + 1, a) and s1[b] <= self.threshold1: b -= 1
            if s1[b] <= self.threshold1: b = j
            if b - a + 1 < self.min_steps: a, b = i, j             # never let the refinement delete an event
            if b - a + 1 >= self.min_steps:
                ev.append(dict(start_h=float(t[a]), end_h=float(min(t[b] + STEP_S / 3600, t[-1])),
                               peak=float(s[a:b + 1].max()), strength=float((s[a:b + 1] - self.threshold).sum()),
                               i0=a, i1=b + 1))
            i = j + 1
        ev.sort(key=lambda e: -e["strength"])
        return s, ev


def build_detector(net, rng, p_noise, d_noise, m=4, factor=1.2, n_ref=4, n_null=12):
    days = normal_days(net, rng, n_ref + n_null, p_noise, d_noise)
    return MatrixProfileDetector(sensor_scale(net, days[:n_ref]), m, factor).fit(days[:n_ref], days[n_ref:])

# =====================================================================================================
# PART C - LEAK LOCALISATION WITH AN ENCODER-DECODER EMBEDDING  (answers: WHICH pipe, WHERE on the pipe)
# =====================================================================================================
def demand_response(net, obs, dmult=1.0, h=None):
    """Expected values + the sensors' response to a +5 % change of ALL demands (same SCADA conditions)."""
    exp = net.expected(obs, dmult, h)
    return exp, residual(net, net.expected(obs, dmult * 1.05, h), exp)


def signature(R, i0, i1, S=None):
    """Leak FINGERPRINT = mean residual over the leak window, scaled to unit length.
    Direction of the vector tells WHERE the leak is; its length only tells how big it is (removed).
    S: response to a uniform demand change -> its direction is projected out of every time step so a demand
    fluctuation can not be mistaken for a leak location."""
    R = R[i0:i1]
    if S is not None:
        S = S[i0:i1]; R = R - (np.sum(R * S, 1) / (np.sum(S * S, 1) + 1e-12))[:, None] * S
    f = R.mean(0)
    return f / (np.linalg.norm(f) + 1e-12)


class EncoderDecoder:
    """Supervised (multi-task) AUTOENCODER = encoder-decoder neural network, written in NumPy.

        fingerprint x --ENCODER--> 64/128 tanh units --> z (BOTTLENECK = EMBEDDING, 16/32 numbers)
        z --DECODER--> tanh units --> x_hat   (must rebuild x  -> z keeps the information of x)
        z --CLASSIFIER--> softmax over (pipe x 70/30 zone) -> leaks at the same place get similar z

    Loss = reconstruction MSE + cross-entropy.  Regularisation against over-fitting: weight decay,
    input noise (denoising), early stopping on validation leaks at positions never used for training."""

    def __init__(self, n_in, n_cls, hidden=64, latent=16, seed=0):
        r = np.random.default_rng(seed); g = lambda a, b: r.normal(0, np.sqrt(2 / (a + b)), (a, b))
        self.W = dict(W1=g(n_in, hidden), b1=np.zeros(hidden), W2=g(hidden, latent), b2=np.zeros(latent),
                      W3=g(latent, hidden), b3=np.zeros(hidden), W4=g(hidden, n_in), b4=np.zeros(n_in),
                      Wc=g(latent, n_cls), bc=np.zeros(n_cls))
        self.history = []

    def encode(self, X):
        W = self.W; return np.tanh(X @ W["W1"] + W["b1"]) @ W["W2"] + W["b2"]

    def _forward(self, x):
        W = self.W
        h1 = np.tanh(x @ W["W1"] + W["b1"]); z = h1 @ W["W2"] + W["b2"]
        h3 = np.tanh(z @ W["W3"] + W["b3"]); xh = h3 @ W["W4"] + W["b4"]
        lg = z @ W["Wc"] + W["bc"]; pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
        return h1, z, h3, xh, pr

    def evaluate(self, X, y):
        _, _, _, xh, pr = self._forward(X)
        loss = ((xh - X) ** 2).mean() - np.log(pr[np.arange(len(y)), y] + 1e-12).mean()
        return float(loss), float((pr.argmax(1) == y).mean())

    def fit(self, X, y, Xv, yv, epochs=400, lr=2e-3, batch=128, weight_decay=1e-3, patience=40,
            input_noise=0.02, seed=0, log=print):
        W, r = self.W, np.random.default_rng(seed)
        m = {k: np.zeros_like(v) for k, v in W.items()}; v2 = {k: np.zeros_like(v) for k, v in W.items()}
        Y = np.eye(W["bc"].size)[y]; step, best, best_W, wait = 0, np.inf, None, 0
        sub = r.choice(len(X), min(len(X), 5000), replace=False)
        for ep in range(epochs):
            idx = r.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]; x0, yy, n = X[b], Y[b], len(b)
                x = x0 + r.normal(0, input_noise, x0.shape)                 # denoising input
                h1, z, h3, xh, pr = self._forward(x)
                dxh = 2 * (xh - x0) / xh.size; dlg = (pr - yy) / n           # back-propagation
                d3 = (dxh @ W["W4"].T) * (1 - h3 ** 2); dz = d3 @ W["W3"].T + dlg @ W["Wc"].T
                d1 = (dz @ W["W2"].T) * (1 - h1 ** 2)
                G = dict(W4=h3.T @ dxh, b4=dxh.sum(0), W3=z.T @ d3, b3=d3.sum(0), Wc=z.T @ dlg, bc=dlg.sum(0),
                         W2=h1.T @ dz, b2=dz.sum(0), W1=x.T @ d1, b1=d1.sum(0))
                step += 1
                for k in W:                                                  # AdamW update
                    m[k] = 0.9 * m[k] + 0.1 * G[k]; v2[k] = 0.999 * v2[k] + 0.001 * G[k] ** 2
                    W[k] -= lr * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v2[k] / (1 - 0.999 ** step)) + 1e-8)
                    if k[0] == "W": W[k] -= lr * weight_decay * W[k]
            tr, va = self.evaluate(X[sub], y[sub]), self.evaluate(Xv, yv)
            self.history.append(dict(epoch=ep + 1, train_loss=tr[0], train_acc=tr[1], val_loss=va[0], val_acc=va[1]))
            if va[0] < best - 1e-4: best, best_W, wait = va[0], {k: v.copy() for k, v in W.items()}, 0
            else: wait += 1
            if log and (ep + 1) % 25 == 0:
                log(f"   epoch {ep + 1:4d} | train loss {tr[0]:.3f} acc {tr[1]:.1%} | validation loss {va[0]:.3f} acc {va[1]:.1%}")
            if wait >= patience:
                if log: log(f"   early stopping at epoch {ep + 1} (best validation loss at epoch {ep + 1 - wait})")
                break
        self.W = best_W
        self.best_epoch = int(np.argmin([h["val_loss"] for h in self.history])) + 1
        return self


class EmbeddingLocaliser:
    """REFERENCE EMBEDDING DATABASE: the embedding of every simulated leak (all pipes, all positions).
    An unknown leak is encoded and compared (cosine similarity) with the database:
      pipe     = weighted vote of the k most similar reference leaks
      position = weighted mean position of those neighbours that lie on the chosen pipe."""

    def __init__(self, model, X, pipes, positions, k=7, tau=20.0):
        self.model, self.k, self.tau = model, k, tau
        self.Z = self.embed(X); self.pipes = np.asarray(pipes); self.pos = np.asarray(positions, float)

    def embed(self, X):
        Z = self.model.encode(np.atleast_2d(X)); return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    def locate(self, x):
        sim = self.Z @ self.embed(x)[0]; nn = np.argsort(-sim)[: self.k]
        w = np.exp(self.tau * (sim[nn] - sim[nn[0]]))
        votes = pd.Series(w).groupby(self.pipes[nn]).sum().sort_values(ascending=False)
        pipe = votes.index[0]; same = self.pipes[nn] == pipe
        pos = float(np.sum(w[same] * self.pos[nn][same]) / np.sum(w[same]))
        return dict(pipe=pipe, pos=pos, confidence=float(votes.iloc[0] / votes.sum()), similarity=float(sim[nn[0]]),
                    candidates=", ".join(f"{p} ({v / votes.sum():.0%})" for p, v in votes.head(3).items()))

    def reference_table(self):
        df = pd.DataFrame(self.Z, columns=[f"z{i + 1}" for i in range(self.Z.shape[1])])
        df.insert(0, "Pipe", self.pipes); df.insert(1, "Zone", [zone(p) for p in self.pos])
        g = df.groupby(["Pipe", "Zone"]); out = g.mean(); out.insert(0, "N_reference_leaks", g.size())
        return out.reset_index()


_NET_CACHE = {}


def _simulate_pipe(args):
    """Worker (runs in parallel): all training leaks of ONE pipe -> fingerprints."""
    inp, pattern, pipe, k, cfg, scale = args
    if inp not in _NET_CACHE: _NET_CACHE[inp] = Network(inp, pattern)
    net, rng = _NET_CACHE[inp], np.random.default_rng([cfg["SEED"], k])
    n_pos = cfg["POSITIONS_PER_PIPE"]; n_front = int(round(n_pos * FRONT_SPLIT)); n_rear = n_pos - n_front
    front = 0.03 + (np.arange(n_front) + rng.uniform(0.1, 0.9, n_front)) / n_front * (FRONT_SPLIT - 0.04)
    rear = FRONT_SPLIT + 0.01 + (np.arange(n_rear) + rng.uniform(0.1, 0.9, n_rear)) / max(n_rear, 1) * 0.26
    runs = []
    for pos in np.concatenate([front, rear]):                    # 1) leak simulations
        h = net.open(pipe, pos)
        try:
            for _ in range(cfg["RUNS_PER_POSITION"]):
                coef = rng.uniform(*cfg["LEAK_COEF_RANGE"]); dur = float(rng.choice(cfg["TRAIN_DURATIONS_H"]))
                on = rng.integers(0, int((24 - dur) * 4) + 1) / 4; dm = float(rng.choice(cfg["DEMAND_MULTIPLIERS"]))
                runs.append((pos, coef, on, on + dur, dm, net.run(h, on + dur, coef, on, on + dur, dm)))
        finally:
            net.close(h)
    X, rows, raw, h = [], [], [], net.open(scada=True)           # 2) expected values + fingerprints
    try:
        for pos, coef, on, off, dm, sim in runs:
            (exp, S), i0, i1 = demand_response(net, sim, dm, h), int(on * 4), int(off * 4)
            if cfg.get("SAVE_RAW_TRAINING_DATA"): raw.append(dict(Pipe=pipe, Position=round(pos, 4), Coefficient=round(coef, 4), Start=hhmm(on), End=hhmm(off),
                            Demand_multiplier=dm, t=sim["t"], P=sim["P"], Q=sim["Q"], V=sim["V"]))
            for a in range(cfg["NOISE_COPIES"] + 1):              # copy 0 noise-free, others with sensor noise
                R = residual(net, sim, exp, rng if a else None, cfg["PRESSURE_NOISE"])
                X.append(signature(R / scale, i0, i1, S / scale))
                rows.append(dict(Pipe=pipe, Position=round(pos, 4), Zone=zone(pos), Coefficient=round(coef, 4),
                                 Start=hhmm(on), End=hhmm(off), Demand_multiplier=dm, Noise_copy=a))
    finally:
        net.close(h)
    return X, rows, raw


def generate_training_set(net, cfg, scale, log=print):
    """Leaks at POSITIONS_PER_PIPE positions on EVERY open pipe, RUNS_PER_POSITION random start / size / demand
    each. Pipes are simulated in parallel on several CPU cores when possible (falls back to one core)."""
    jobs = [(str(net.inp), net.daily_pattern, p, k, cfg, scale) for k, p in enumerate(net.pipes)]
    workers = cfg.get("WORKERS") or max(1, (os.cpu_count() or 2) - 1)
    X, rows, raw = [], [], []

    def collect(results):
        for done, (x, r, w) in enumerate(results, 1):
            X.extend(x); rows.extend(r); raw.extend(w)
            if log and done % max(1, len(jobs) // 6) == 0:
                log(f"   simulated {done}/{len(jobs)} pipes ({len(rows)} fingerprints)")
    if workers > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(workers) as ex: collect(ex.map(_simulate_pipe, jobs))
            log and log(f"   ({workers} CPU cores used)")
            return np.array(X, np.float32), pd.DataFrame(rows), raw
        except Exception as e:                       # e.g. some Windows IDEs block worker processes
            log and log(f"   parallel run not possible here ({type(e).__name__}: {str(e)[:120]}) -> using one CPU core")
            X, rows, raw = [], [], []
    collect(_simulate_pipe(j) for j in jobs)
    return np.array(X, np.float32), pd.DataFrame(rows), raw


def split_by_position(meta, every=5):
    """Validation = every 5th training position of each pipe (whole positions held out, incl. all copies),
    so validation measures how well the model handles leak positions it has NEVER seen."""
    val = np.zeros(len(meta), bool)
    for _, g in meta.groupby("Pipe"):
        pos = np.sort(g.Position.unique()); held = set(pos[2::every])
        val[g.index[g.Position.isin(held)]] = True
    return val


def train_localiser(net, X, meta, cfg, out_dir, log=print):
    """Train the encoder-decoder with a train / validation split, report how well it generalises,
    then build the final reference embedding database from ALL simulated leaks."""
    cls = (meta.Pipe + "|" + meta.Zone).astype("category"); y = cls.cat.codes.to_numpy()
    va = split_by_position(meta); tr = ~va
    model = EncoderDecoder(X.shape[1], len(cls.cat.categories), cfg["HIDDEN_DIM"], cfg["LATENT_DIM"], cfg["SEED"])
    model.fit(X[tr], y[tr], X[va], y[va], cfg["MAX_EPOCHS"], weight_decay=cfg["WEIGHT_DECAY"],
              patience=cfg["PATIENCE"], seed=cfg["SEED"], log=log)
    # ---- generalisation check on the held-out positions (references = training positions only) ----
    loc_tr = EmbeddingLocaliser(model, X[tr], meta.Pipe[tr], meta.Position[tr])
    q = meta[va & (meta.Noise_copy > 0).to_numpy()]
    q = q.iloc[:: max(1, len(q) // 3000)]
    rows = []
    for i, m_ in q.iterrows():
        r = loc_tr.locate(X[i])
        rows.append(dict(Pipe=m_.Pipe, ok=r["pipe"] == m_.Pipe, adj=net.adjacent(r["pipe"], m_.Pipe),
                         zone_ok=r["pipe"] == m_.Pipe and zone(r["pos"]) == m_.Zone,
                         dist=net.leak_distance(m_.Pipe, m_.Position, r["pipe"], r["pos"])))
    v = pd.DataFrame(rows); h = pd.DataFrame(model.history); b = model.best_epoch - 1
    u = net.len_unit
    report = pd.DataFrame([
        ("Training fingerprints / validation fingerprints", f"{tr.sum()} / {va.sum()}"),
        ("Best epoch (early stopping)", model.best_epoch),
        ("Classifier accuracy - training", f"{h.train_acc[b]:.1%}"),
        ("Classifier accuracy - validation (unseen positions)", f"{h.val_acc[b]:.1%}"),
        ("Generalisation gap (train - validation)", f"{h.train_acc[b] - h.val_acc[b]:.1%}"),
        ("Validation: correct pipe", f"{v.ok.mean():.1%}"),
        ("Validation: correct pipe OR directly connected pipe", f"{(v.ok | v.adj).mean():.1%}"),
        ("Validation: correct pipe AND 70/30 zone", f"{v.zone_ok.mean():.1%}"),
        (f"Validation: median localisation error ({u})", f"{v.dist.median():.1f}"),
        (f"Validation: 90 % of errors below ({u})", f"{v.dist.quantile(0.9):.1f}"),
        (f"Median pipe length ({u})", f"{np.median([net.length[p] for p in net.pipes]):.1f}"),
    ], columns=["Metric", "Value"])
    per_pipe = v.groupby("Pipe").agg(Validation_leaks=("ok", "size"), Pipe_correct=("ok", "mean"),
                                     Zone_correct=("zone_ok", "mean"), Median_error=("dist", "median")).reset_index()
    save_training_curve(h, model.best_epoch, out_dir / "training_curve.png")
    log("\n   Generalisation check (validation leaks at positions never used for training):")
    for _, r in report.iterrows(): log(f"     {r.Metric:<55} {r.Value}")
    return EmbeddingLocaliser(model, X, meta.Pipe, meta.Position), report, per_pipe, h


def save_training_curve(h, best, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, k, t in ((ax[0], "loss", "Loss"), (ax[1], "acc", "Classification accuracy")):
        a.plot(h.epoch, h[f"train_{k}"], label="training"); a.plot(h.epoch, h[f"val_{k}"], label="validation (unseen positions)")
        a.axvline(best, ls="--", c="grey", label=f"chosen epoch {best}"); a.set_xlabel("epoch"); a.set_title(t)
        a.grid(alpha=.3); a.legend()
    fig.suptitle("Encoder-decoder training: training vs validation (early stopping prevents over-fitting)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

# =====================================================================================================
# PART D - OUTPUTS: EXCEL + INTERACTIVE HTML
# =====================================================================================================
def timeseries_frame(net, sim):
    """15-min table: time + pressure (every junction) + flow and velocity (every pipe)."""
    t = sim["t"]; pu, qu, vu = net.units
    parts = [pd.DataFrame({"Time": [hhmm(x) for x in t], "Time_h": t})]
    for key, ids, u, name in (("P", net.junctions, pu, "Pressure"), ("Q", net.pipes_all, qu, "Flow"),
                              ("V", net.pipes_all, vu, "Velocity")):
        parts.append(pd.DataFrame(sim[key], columns=[f"{name}_{i} [{u}]" for i in ids]))
    return pd.concat(parts, axis=1)


def save_excel(path, sheets):
    try: import xlsxwriter; engine = "xlsxwriter"          # ~3x faster for big workbooks
    except ImportError: engine = None                      # falls back to openpyxl
    with pd.ExcelWriter(path, engine=engine) as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=str(name)[:31], index=False)
    return path


def open_file(path):
    try:
        os.startfile(str(path)) if sys.platform.startswith("win") else webbrowser.open(Path(path).resolve().as_uri())
    except Exception:
        pass


def random_tests(net, rng, n, coef_range, durations=(1, 2, 3, 4)):
    out = []
    for i in range(n):
        d = float(rng.choice(durations)); on = rng.integers(0, int((24 - d) * 4) + 1) / 4
        out.append(dict(leak_id=f"Random_{i + 1}", pipe=str(rng.choice(net.pipes)), position=round(float(rng.uniform(0.05, 0.95)), 2),
                        coefficient=round(float(rng.uniform(*coef_range)), 3), start=hhmm(on), end=hhmm(on + d)))
    return out


def metrics(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Accuracy=(tp + tn) / max(tp + fp + fn + tn, 1), Precision=p, Recall=r,
                F1=2 * p * r / (p + r) if p + r else 0.0, False_alarm_rate=fp / (fp + tn) if fp + tn else 0.0)


def print_table(df):
    """Plain fixed-width table (left-aligned) that stays readable in any console."""
    cells = [[str(c) for c in df.columns]] + [["" if v is None or (isinstance(v, float) and np.isnan(v)) else
                                                (str(int(v)) if isinstance(v, float) and v == int(v) else str(v)) for v in r] for r in df.values]
    w = [max(len(r[i]) for r in cells) for i in range(len(cells[0]))]
    for k, r in enumerate(cells):
        print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
        if k == 0: print("  " + "  ".join("-" * x for x in w))


SENSOR_TEXT = {"H": "hydraulic head (junctions)", "Q": "flow (pipes)", "V": "velocity (pipes)", "Qin": "inlet flow meters"}


def change_names(net):
    u = {"H": net.units[0], "Q": net.units[1], "V": net.units[2], "Qin": net.units[1]}
    word = {"H": "head", "Q": "flow", "V": "velocity", "Qin": "inlet flow"}
    return [f"{n.split('_', 1)[1]} {word[f]} change [{u[f]}]" for f, _, names in net.blocks() for n in names]


def network_info(net):
    """Junction elevations and pipe data (used by the hydraulic model and the head feature)."""
    pipes, sec = {}, None
    for line in net.lines:
        s_ = line.split(";")[0].strip(); t_ = s_.split()
        if s_.startswith("["): sec = s_.upper(); continue
        if sec == "[PIPES]" and len(t_) >= 6: pipes[t_[0]] = (float(t_[4]), float(t_[5]))
    j = pd.DataFrame({"Junction": net.junctions, f"Elevation [{net.len_unit}]": net.z})
    p_ = pd.DataFrame([{"Pipe": p, "Start node": net.ends[p][0], "End node": net.ends[p][1], f"Length [{net.len_unit}]": net.length[p],
                        "Diameter": pipes.get(p, (None, None))[0], "Roughness": pipes.get(p, (None, None))[1],
                        "Status": "open" if p in net.pipes else "closed"} for p in net.pipes_all])
    return j, p_


def epanet_sheets(net, labels, sims, changes, leak_on=None):
    """Raw EPANET results of many days stacked in one sheet per quantity (rows = day x 15-min step)."""
    key = []
    for k, (lab, sim) in enumerate(zip(labels, sims)):
        d = {"Day": lab, "Time": [hhmm(x) for x in sim["t"]]}
        if leak_on is not None: d["Leak on"] = leak_on[k].astype(int)
        key.append(pd.DataFrame(d))
    key = pd.concat(key, ignore_index=True)
    stack = lambda arrs, cols: pd.concat([key, pd.DataFrame(np.vstack(arrs), columns=cols).round(4)], axis=1)
    pu, qu, vu = net.units
    return {"EPANET_Pressure": stack([s_["P"] for s_ in sims], [f"{j} [{pu}]" for j in net.junctions]),
            "EPANET_Head": stack([s_["P"] + net.z for s_ in sims], [f"{j} [{pu}]" for j in net.junctions]),
            "EPANET_Flow": stack([s_["Q"] for s_ in sims], [f"{p} [{qu}]" for p in net.pipes_all]),
            "EPANET_Velocity": stack([s_["V"] for s_ in sims], [f"{p} [{vu}]" for p in net.pipes_all]),
            "Sensor_change": stack(changes, change_names(net))}


def three_tables(net, sim):
    pu, qu, vu = net.units
    return {"Pressure": pd.DataFrame(sim["P"], columns=[f"{j} [{pu}]" for j in net.junctions]),
            "Head": pd.DataFrame(sim["P"] + net.z, columns=[f"{j} [{pu}]" for j in net.junctions]),
            "Flow": pd.DataFrame(sim["Q"], columns=[f"{p} [{qu}]" for p in net.pipes_all]),
            "Velocity": pd.DataFrame(sim["V"], columns=[f"{p} [{vu}]" for p in net.pipes_all])}


def safe_save(path, sheets):
    """Write the workbook; if it is open in Excel (Windows locks it) save a copy with a new name."""
    try:
        return save_excel(path, sheets)
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{time.strftime('%H%M%S')}{path.suffix}")
        print(f"  ! {path.name} is open in Excel -> saved as {alt.name}")
        return save_excel(alt, sheets)

def write_html(path, net, title, cases, table, notes="", extra_html=""):
    """Interactive report (Plotly). Click a table row to show that case on the map / timeline.
    case keys: true_pipe, true_pos, true_on, true_off, det_pipe, det_pos, t, score, threshold, det_on, det_off"""
    geo = {p: net.polyline(p) for p in net.pipes_all}
    for c in cases:
        if c.get("true_pipe"):
            c["true_xy"] = net.point_on_pipe(c["true_pipe"], c["true_pos"]); c["true_split"] = net.point_on_pipe(c["true_pipe"], FRONT_SPLIT)
        if c.get("det_pipe"): c["det_xy"] = net.point_on_pipe(c["det_pipe"], c["det_pos"])
        for k in ("true_on", "true_off", "det_on", "det_off"):
            if c.get(k) is not None: c[k + "_s"] = hhmm(c[k])
    data = json.dumps(dict(geo=geo, nodes={n: net.xy[n] for n in net.junctions}, cases=cases),
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    try:      # embed plotly.js from the installed plotly package -> works offline; else use the CDN
        import plotly
        plotly_tag = "<script>" + (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8") + "</script>"
    except Exception:
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    tbl = table.to_html(index=False, classes="tbl", border=0, na_rep="-")
    html = """<!doctype html><html><head><meta charset="utf-8"><title>@@TITLE@@</title>@@PLOTLY@@
<style>body{font-family:Segoe UI,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}h1{font-size:20px;margin:0 0 4px}
.note{color:#444;font-size:13px;margin-bottom:10px}.tbl{border-collapse:collapse;font-size:12px;background:#fff;width:100%}
.tbl th{background:#263238;color:#fff;padding:5px 7px;position:sticky;top:0}.tbl td{padding:4px 7px;border-bottom:1px solid #ddd;text-align:center}
.tbl tr:hover td{background:#e3f2fd;cursor:pointer}.tbl tr.sel td{background:#bbdefb;font-weight:600}.wrap{max-height:320px;overflow:auto;border:1px solid #ccc}
#info{font-size:14px;margin:10px 0;padding:8px 12px;background:#fff;border-left:5px solid #d62728}.grid{display:flex;flex-wrap:wrap;gap:10px}
#map{flex:3;min-width:520px;height:640px}#ts{flex:2;min-width:420px;height:640px}</style></head><body>
<h1>@@TITLE@@</h1><div class="note">@@NOTES@@ Click a row to show that case.</div>@@EXTRA@@<h2 style="font-size:16px">Results table</h2><div class="wrap">@@TABLE@@</div><div id="info"></div>
<div class="grid"><div id="map"></div><div id="ts"></div></div><script>
const D=@@DATA@@;const rows=[...document.querySelectorAll('.wrap .tbl tbody tr')];
function line(p,color,w,name,dash){const x=[],y=[];for(const q of [].concat(p)){for(const [a,b] of D.geo[q]){x.push(a);y.push(b)}x.push(null);y.push(null)}
 return {x,y,mode:'lines',line:{color,width:w,dash:dash||'solid'},name,hoverinfo:'skip'}}
function pt(xy,color,sym,name,txt){return {x:[xy[0]],y:[xy[1]],mode:'markers+text',marker:{size:15,color,symbol:sym,line:{color:'#fff',width:2}},
 text:[txt],textposition:'top center',textfont:{color,size:13},name,hovertext:[name+': '+txt],hoverinfo:'text'}}
function show(i){const c=D.cases[i];rows.forEach((r,k)=>r.classList.toggle('sel',k==i));const pipes=Object.keys(D.geo),tr=[line(pipes,'#b0b7bf',1.5,'Network pipes')];
 const nx=[],ny=[],nt=[];for(const [n,[a,b]] of Object.entries(D.nodes)){nx.push(a);ny.push(b);nt.push(n)}
 tr.push({x:nx,y:ny,text:nt,mode:'markers',marker:{size:4,color:'#555'},name:'Junctions',hoverinfo:'text'});
 const hv=pipes.map(p=>{const g=D.geo[p],k=Math.floor(g.length/2),a=g[k],b=g[Math.max(k-1,0)];return [(a[0]+b[0])/2,(a[1]+b[1])/2,p]});
 tr.push({x:hv.map(h=>h[0]),y:hv.map(h=>h[1]),text:hv.map(h=>'Pipe '+h[2]),mode:'markers',marker:{size:9,opacity:0},hoverinfo:'text',showlegend:false});
 let txt='';
 if(c.true_pipe){tr.push(line(c.true_pipe,'#d62728',9,'Induced leak pipe: '+c.true_pipe));
  tr.push(pt(c.true_split,'#ff9800','line-ns-open','70 % mark of induced pipe','70%'));
  tr.push(pt(c.true_xy,'#d62728','circle','Induced leak',c.true_pipe+' @ '+(100*c.true_pos).toFixed(0)+'%'));
  txt='<b>Induced leak:</b> pipe <b>'+c.true_pipe+'</b> at '+(100*c.true_pos).toFixed(0)+'% of its length, active '+c.true_on_s+' - '+c.true_off_s;}
 else txt='<b>Leak-free day</b> (false-alarm test)';
 if(c.det_pipe){tr.push(line(c.det_pipe,'#1f77b4',4,'Detected pipe: '+c.det_pipe,c.det_pipe==c.true_pipe?'dot':'solid'));
  tr.push(pt(c.det_xy,'#1f77b4','x','Detected leak',c.det_pipe+' @ '+(100*c.det_pos).toFixed(0)+'%'));
  txt+=' &nbsp;|&nbsp; <b>Localised:</b> pipe <b style="color:#1f77b4">'+c.det_pipe+'</b> at '+(100*c.det_pos).toFixed(0)+'%';}
 if(c.det_on!=null) txt+=' &nbsp;|&nbsp; <b>Detected leak time:</b> <b style="color:#2e7d32">'+c.det_on_s+' - '+c.det_off_s+'</b>';
 else if(c.score) txt+=' &nbsp;|&nbsp; <b>No leak detected</b>';
 document.getElementById('info').innerHTML=txt;
 Plotly.react('map',tr,{title:'Network map - case '+(i+1),xaxis:{scaleanchor:'y',showgrid:false},yaxis:{showgrid:false},
  legend:{orientation:'h',y:-0.05},margin:{l:40,r:10,t:40,b:10},hovermode:'closest',plot_bgcolor:'#fff'},{scrollZoom:true,displaylogo:false});
 const ts=document.getElementById('ts');
 if(c.score){ts.style.display='';const sh=[];
  if(c.true_pipe) sh.push({type:'rect',xref:'x',yref:'paper',x0:c.true_on,x1:c.true_off,y0:0,y1:1,fillcolor:'#d62728',opacity:0.15,line:{width:0}});
  if(c.det_on!=null) sh.push({type:'rect',xref:'x',yref:'paper',x0:c.det_on,x1:c.det_off,y0:0.04,y1:0.96,fillcolor:'rgba(0,0,0,0)',line:{color:'#2e7d32',width:3}});
  Plotly.react('ts',[{x:c.t,y:c.score,mode:'lines',name:'Matrix-profile anomaly score',line:{color:'#263238'}},
   {x:[0,24],y:[c.threshold,c.threshold],mode:'lines',name:'Threshold',line:{dash:'dash',color:'#ff9800'}},
   {x:[null],y:[null],mode:'lines',line:{color:'#d62728',width:10},opacity:0.3,name:'Induced leak interval'},
   {x:[null],y:[null],mode:'lines',line:{color:'#2e7d32',width:3},name:'Detected leak range'}],
   {title:'Leak detection (Matrix Profile)',shapes:sh,xaxis:{title:'Time of day (h)',dtick:2,range:[0,24]},yaxis:{title:'Anomaly score'},
   legend:{orientation:'h',y:-0.15},margin:{l:50,r:10,t:40,b:10}},{displaylogo:false})}
 else ts.style.display='none';}
rows.forEach((r,i)=>r.onclick=()=>show(i));show(0);</script></body></html>"""
    for k, v in (("TITLE", title), ("NOTES", notes), ("EXTRA", extra_html), ("TABLE", tbl), ("DATA", data), ("PLOTLY", plotly_tag)):
        html = html.replace(f"@@{k}@@", v)
    Path(path).write_text(html, encoding="utf-8")
    return path

# =====================================================================================================
# PART E - COMPLETE STUDY: train (or load) the embedding localiser, then test leaks end-to-end
# =====================================================================================================
def img_tag(path, width=900):
    import base64
    return f'<img style="max-width:100%;width:{width}px;background:#fff;border:1px solid #ddd" src="data:image/png;base64,' + \
        base64.b64encode(Path(path).read_bytes()).decode() + '">'


def save_pipe_confusion_png(path, cm, zcm, title):
    """Left: actual pipe vs localised pipe.  Right: 0-70 % / 70-100 % zone (leaks with the correct pipe)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = len(cm); fig, ax = plt.subplots(1, 2, figsize=(min(6 + 0.45 * n, 26) + 5, min(5 + 0.4 * n, 24)),
                                         gridspec_kw={"width_ratios": [max(n, 4), 4]})
    ax[0].imshow(cm.values, cmap="Blues")
    ax[0].set_xticks(range(cm.shape[1]), cm.columns, rotation=90, fontsize=8 if n > 20 else 10)
    ax[0].set_yticks(range(n), cm.index, fontsize=8 if n > 20 else 10)
    ax[0].set_xlabel("Localised (predicted) pipe"); ax[0].set_ylabel("Actual leak pipe"); ax[0].set_title(title, fontsize=11)
    if n <= 30:
        for (a, b), v in np.ndenumerate(cm.values):
            if v: ax[0].text(b, a, int(v), ha="center", va="center", fontsize=9, color="white" if v > cm.values.max() / 2 else "black")
    ax[1].imshow(zcm.values, cmap="Greens")
    for (a, b), v in np.ndenumerate(zcm.values):
        ax[1].text(b, a, int(v), ha="center", va="center", fontsize=14, color="white" if v > zcm.values.max() / 2 else "black")
    ax[1].set_xticks([0, 1], zcm.columns); ax[1].set_yticks([0, 1], zcm.index)
    acc = np.trace(zcm.values) / max(zcm.values.sum(), 1)
    ax[1].set_title(f"Position on the pipe (correct-pipe leaks)\nzone accuracy {acc:.1%}", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_embedding_png(path, loc, test_Z=None, test_pipes=None, max_labels=25):
    """2-D picture of the bottleneck embeddings (PCA): every dot = one simulated leak, colour = its pipe.
    Leaks in the same pipe form a cluster -> the embedding 'knows' where a leak is."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Z = loc.Z; mu = Z.mean(0); _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False); P = (Z - mu) @ Vt[:2].T
    pipes = list(dict.fromkeys(loc.pipes)); cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, p in enumerate(pipes):
        m = loc.pipes == p
        ax.scatter(P[m, 0], P[m, 1], s=6, color=cmap(i % 20), alpha=0.5, label=p if len(pipes) <= max_labels else None)
        if len(pipes) <= 60: ax.text(*P[m].mean(0), p, fontsize=8, fontweight="bold", ha="center")
    if test_Z is not None and len(test_Z):
        T = (test_Z - mu) @ Vt[:2].T
        ax.scatter(T[:, 0], T[:, 1], marker="x", c="k", s=40, label="test leaks (unknown)")
    ax.set_xlabel("embedding direction 1 (PCA)"); ax.set_ylabel("embedding direction 2 (PCA)")
    ax.set_title(f"Bottleneck embeddings of {len(Z)} simulated leaks ({Z.shape[1]} numbers each, shown in 2-D)\n"
                 "each colour = one pipe; leaks in the same pipe cluster together")
    if len(pipes) <= max_labels: ax.legend(fontsize=7, ncol=2, markerscale=3, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def build_test_leaks(net, cfg, user_tests):
    """Your test leaks + LEAKS_PER_TEST_PIPE extra leaks (random position, size, time) on every test pipe."""
    rng = np.random.default_rng(cfg["SEED"] + 1)
    pipes = list(net.pipes) if cfg["TEST_PIPES"] == "ALL" else list(cfg["TEST_PIPES"])
    others = [p for p in net.pipes if p not in pipes]                  # + randomly chosen extra test pipes
    pipes += [str(p) for p in rng.choice(others, min(cfg.get("N_RANDOM_TEST_PIPES", 0), len(others)), replace=False)]
    tests, count = [], {}
    for t in user_tests:
        count[t["pipe"]] = count.get(t["pipe"], 0) + 1
        tests.append(dict(t, name=f"{t['pipe']} leak {count[t['pipe']]}"))
    for p in pipes:
        if p not in net.pipes: raise ValueError(f"Test pipe {p} is not an open pipe of {net.inp.name}")
        for _ in range(max(cfg["LEAKS_PER_TEST_PIPE"] - sum(t["pipe"] == p for t in user_tests), 0)):
            d = float(rng.choice((1.0, 2.0, 3.0, 4.0))); on = rng.integers(0, int((24 - d) * 4) + 1) / 4
            count[p] = count.get(p, 0) + 1
            tests.append(dict(name=f"{p} leak {count[p]}", pipe=p, position=round(float(rng.uniform(0.05, 0.95)), 2),
                              coefficient=round(float(rng.uniform(*cfg["LEAK_COEF_RANGE"])), 3), start=hhmm(on), end=hhmm(on + d)))
    order = {p: i for i, p in enumerate(dict.fromkeys([t["pipe"] for t in tests]))}
    return sorted(tests, key=lambda t: order[t["pipe"]])


def run_study(inp, cfg, user_tests, out_dir, title, detect, daily_pattern=None, readme=(), log=print):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Results will be saved in:\n  {out_dir.resolve()}\n")
    net = Network(inp, daily_pattern); rng = np.random.default_rng(cfg["SEED"]); u = net.len_unit
    log(f"Network      : {net.inp.name}: {len(net.junctions)} junctions, {len(net.pipes)} open pipes")
    log("Sensors      : " + ", ".join(f"{len(n)} x {SENSOR_TEXT[f]}" for f, _, n in net.blocks()) +
        "  (head = pressure + elevation)")

    log("\n[1] Simulating 16 leak-free days -> normal sensor variability" + (" + Matrix-Profile threshold" if detect else ""))
    days = normal_days(net, rng, 16, cfg["PRESSURE_NOISE"], cfg["DEMAND_NOISE"]); scale = sensor_scale(net, days[:4])
    det = MatrixProfileDetector(scale, cfg["MP_WINDOW"], cfg["THRESHOLD_FACTOR"]).fit(days[:4], days[4:]) if detect else None
    if det: log(f"    Matrix Profile window {cfg['MP_WINDOW'] * 15} min, threshold {det.threshold:.2f}")

    # ------------------------------ TRAINING (embedding database) ------------------------------
    cache = out_dir / "trained_model_cache.pkl"
    skip = ("TEST_PIPES", "LEAKS_PER_TEST_PIPE", "N_RANDOM_TEST_PIPES", "N_LEAK_FREE_TESTS", "WORKERS", "SAVE_RAW_TRAINING_DATA",
            "OPEN_OUTPUT_FOLDER", "MONTH_DAYS", "MONTH_LEAKS", "WEEKEND_FACTOR", "MONTH_TREND")
    key = repr({k: cfg[k] for k in sorted(cfg) if k not in skip}) + repr(FEATURES) + net.inp.read_text(errors="ignore")
    loc = None
    if cache.exists():
        try:
            with open(cache, "rb") as f: c = pickle.load(f)
            if c.get("key") == key:
                X, meta, loc, report, per_pipe, hist = (c[k] for k in ("X", "meta", "loc", "report", "per_pipe", "hist"))
                log("\n[2] Trained model loaded from trained_model_cache.pkl (settings unchanged; delete it to retrain)")
        except Exception:
            loc = None
    if loc is None:
        t0 = time.time()
        log(f"\n[2] TRAINING DATA: simulating leaks on ALL {len(net.pipes)} pipes x {cfg['POSITIONS_PER_PIPE']} positions x "
            f"{cfg['RUNS_PER_POSITION']} runs (random size, start time, demand)")
        X, meta, raw = generate_training_set(net, cfg, scale, log)
        log(f"    {len(X)} fingerprints ({len(raw) or len(X) // (cfg['NOISE_COPIES'] + 1)} EPANET leak simulations) in {time.time() - t0:.0f} s")
        if raw:
            log("    writing raw EPANET training data (training_raw_epanet.csv) ...")
            parts = []
            for i, r in enumerate(raw, 1):
                d = pd.DataFrame({"Simulation": i, "Leak pipe": r["Pipe"], "Position": r["Position"], "Emitter coef": r["Coefficient"],
                                  "Leak start": r["Start"], "Leak end": r["End"], "Demand multiplier": r["Demand_multiplier"],
                                  "Time": [hhmm(x) for x in r["t"]], "Leak on": (r["t"] >= hours(r["Start"])).astype(int)})
                parts.append(pd.concat([d, pd.DataFrame(np.round(r["P"], 4), columns=[f"P_{j}" for j in net.junctions]),
                                        pd.DataFrame(np.round(r["Q"], 4), columns=[f"Q_{p}" for p in net.pipes_all]),
                                        pd.DataFrame(np.round(r["V"], 4), columns=[f"V_{p}" for p in net.pipes_all])], axis=1))
            pd.concat(parts, ignore_index=True).to_csv(out_dir / "training_raw_epanet.csv", index=False)
        log("\n[3] Training the encoder-decoder (validation = leak positions never used for training) ...")
        loc, report, per_pipe, hist = train_localiser(net, X, meta, cfg, out_dir, log)
        with open(cache, "wb") as f:
            pickle.dump(dict(key=key, X=X, meta=meta, loc=loc, report=report, per_pipe=per_pipe, hist=hist), f)
    # ---- the reference embedding database (also stored in trained_model_cache.pkl) -> readable files ----
    all_emb = pd.concat([meta.reset_index(drop=True),
                         pd.DataFrame(np.round(loc.Z, 5), columns=[f"z{i + 1}" for i in range(loc.Z.shape[1])])], axis=1)
    all_emb.to_csv(out_dir / "all_embeddings.csv", index=False)
    log(f"    embedding database: {loc.Z.shape[0]} stored embeddings x {loc.Z.shape[1]} numbers "
        f"({loc.Z.shape[0] // len(net.pipes)} per pipe) -> all_embeddings.csv + training_data.xlsx")
    tsum = meta[meta.Noise_copy == 0].groupby("Pipe", sort=False).agg(
        Positions=("Position", "nunique"), Simulated_leaks=("Position", "size"), Min_coef=("Coefficient", "min"),
        Max_coef=("Coefficient", "max")).reset_index()

    # ------------------------------ TESTING ------------------------------
    tests = build_test_leaks(net, cfg, user_tests)
    tests += [dict(name=f"No-leak day {i + 1}", pipe=None) for i in range(cfg.get("N_LEAK_FREE_TESTS", 0) if detect else 0)]
    n_leaks = sum(t["pipe"] is not None for t in tests)
    log(f"\n[4] TESTING: {n_leaks} new leaks on {len(set(t['pipe'] for t in tests if t['pipe']))} test pipes"
        + (f" + {len(tests) - n_leaks} leak-free days" if detect else "") + " (new random days, never seen in training)")
    rows, cases, sims, changes, ons, steps, test_Z = [], [], [], [], [], [], []
    for n, ev in enumerate(tests, 1):
        leak = ev["pipe"] is not None
        on, off = (hours(ev["start"]), hours(ev["end"])) if leak else (0.0, 0.0)
        obs = net.simulate_leak(ev["pipe"], ev.get("position"), ev.get("coefficient", 0), on, off, 24.0, 1.0,
                                demand_noise(rng, 24.0, cfg["DEMAND_NOISE"]))
        exp, S = demand_response(net, obs)
        R = residual(net, obs, exp, rng, cfg["PRESSURE_NOISE"])
        actual = (obs["t"] >= on) & (obs["t"] < off)
        row = {"No.": n, "Test": ev["name"], "Leak pipe": ev["pipe"] or "none", "Emitter coef": ev.get("coefficient"),
               "Leak time (induced)": f"{ev['start']} - {ev['end']}" if leak else "-"}
        case = dict(true_pipe=ev["pipe"], true_pos=ev.get("position"), true_on=on if leak else None, true_off=off if leak else None)
        if detect:
            score, events = det.detect(obs["t"], R)
            alarm = np.zeros(len(actual), bool)
            for e_ in events: alarm[e_["i0"]:e_["i1"]] = True
            hit = [e_ for e_ in events if actual[e_["i0"]:e_["i1"]].any()]
            e = hit[0] if hit else (events[0] if events else None)
            steps.append((actual, alarm))
            row |= {"Leak time (detected)": f"{hhmm(e['start_h'])} - {hhmm(e['end_h'])}" if e else "not detected",
                    "Start error (min)": round((e["start_h"] - on) * 60) if hit else None,
                    "Detection": ("DETECTED" if hit else "MISSED") if leak else ("FALSE ALARM" if events else "NO ALARM (correct)")}
            case |= dict(t=obs["t"], score=score, threshold=det.threshold, det_on=e["start_h"] if e else None, det_off=e["end_h"] if e else None)
            i0, i1 = (e["i0"], e["i1"]) if e else (None, None)
        else:
            i0, i1 = int(on * 4), int(off * 4)               # localisation only: the leak window is known
        f = signature(R / scale, i0, i1, S / scale) if i0 is not None else None
        r = loc.locate(f) if f is not None else None
        if f is not None and leak: test_Z.append(loc.embed(f)[0])
        ok = bool(r and leak and r["pipe"] == ev["pipe"])
        row |= {"Induced position (%)": round(100 * ev["position"]) if leak else None,
                "Induced zone": zone(ev["position"]) if leak else "-",
                "Localised pipe": r["pipe"] if r else "-",
                "Localised position (%)": round(100 * r["pos"]) if r else None,
                "Localised zone": zone(r["pos"]) if r else "-",
                "Pipe correct": ("YES" if ok else "NO") if leak else "-",
                "Zone correct": ("YES" if ok and zone(r["pos"]) == zone(ev["position"]) else "NO") if leak else "-",
                "Wrong pipe is a neighbour": "YES" if r and leak and net.adjacent(r["pipe"], ev["pipe"]) else "",
                f"Error ({u})": round(net.leak_distance(ev["pipe"], ev["position"], r["pipe"], r["pos"]), 1) if r and leak else None,
                "Confidence": f"{r['confidence']:.0%}" if r else "-", "Top-3 candidate pipes": r["candidates"] if r else "-"}
        rows.append(row); cases.append(case | dict(det_pipe=r["pipe"] if r else None, det_pos=r["pos"] if r else None))
        sims.append(obs); changes.append(R); ons.append(actual)

    # ------------------------------ EVALUATION ------------------------------
    table = pd.DataFrame(rows); L = table[table["Leak pipe"] != "none"]; n = len(L)
    Lc = L[L["Localised pipe"] != "-"]
    pipes_t = list(dict.fromkeys(L["Leak pipe"]))
    cols = pipes_t + sorted(set(Lc["Localised pipe"]) - set(pipes_t))
    cm = pd.crosstab(pd.Categorical(Lc["Leak pipe"], pipes_t), pd.Categorical(Lc["Localised pipe"], cols), dropna=False)
    cm.index.name, cm.columns.name = "Actual pipe", "Localised pipe"
    okp = Lc[Lc["Pipe correct"] == "YES"]
    zcm = pd.crosstab(pd.Categorical(okp["Induced zone"], [zone(0.1), zone(0.9)]),
                      pd.Categorical(okp["Localised zone"], [zone(0.1), zone(0.9)]), dropna=False)
    zcm.index = [f"Actual {z}" for z in zcm.index]; zcm.columns = [f"Localised {z}" for z in zcm.columns]
    per = []
    for p in pipes_t:
        tp = int(cm.loc[p, p]) if p in cm.columns else 0; fn = int(cm.loc[p].sum()) - tp
        fp = int(cm[p].sum()) - tp if p in cm.columns else 0; g = L[L["Leak pipe"] == p]
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        per.append({"Test pipe": p, "Test leaks": len(g), "Correct pipe": f"{tp}/{len(g)}", "Recall": round(rec, 3),
                    "Precision": round(prec, 3), "F1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
                    "Zone correct": f"{int((g['Zone correct'] == 'YES').sum())}/{len(g)}",
                    f"Median error ({u})": round(g[f"Error ({u})"].median(), 1) if g[f"Error ({u})"].notna().any() else None})
    per = pd.DataFrame(per)
    acc = (L["Pipe correct"] == "YES").mean()
    met = [("Test leaks", n), ("Pipe accuracy (correct pipe)", f"{acc:.1%}"),
           ("Pipe precision (macro average over test pipes)", f"{per.Precision.mean():.1%}"),
           ("Pipe recall (macro average over test pipes)", f"{per.Recall.mean():.1%}"),
           ("Pipe F1 (macro average)", f"{per.F1.mean():.3f}"),
           ("Correct pipe OR directly connected pipe", f"{((L['Pipe correct'] == 'YES') | (L['Wrong pipe is a neighbour'] == 'YES')).mean():.1%}"),
           ("Correct pipe AND correct 70/30 zone", f"{(L['Zone correct'] == 'YES').mean():.1%}"),
           (f"Median localisation error along the pipes ({u})", f"{L[f'Error ({u})'].median():.1f}"),
           (f"Median pipe length ({u})", f"{np.median([net.length[p] for p in net.pipes]):.0f}")]
    if detect:
        F = table[table["Leak pipe"] == "none"]; d_ok = L.Detection == "DETECTED"
        cme = metrics(int(d_ok.sum()), int((F.Detection == "FALSE ALARM").sum()), int((~d_ok).sum()), int((F.Detection != "FALSE ALARM").sum()))
        a = np.concatenate([s_[0] for s_ in steps]); p_ = np.concatenate([s_[1] for s_ in steps])
        cms = metrics(int((a & p_).sum()), int((~a & p_).sum()), int((a & ~p_).sum()), int((~a & ~p_).sum()))
        met = [("DETECTION per leak event: TP / FN / FP / TN", f"{cme['TP']} / {cme['FN']} / {cme['FP']} / {cme['TN']}"),
               ("Detection accuracy / precision / recall", f"{cme['Accuracy']:.1%} / {cme['Precision']:.1%} / {cme['Recall']:.1%}"),
               ("Detection per 15-min step: accuracy / precision / recall", f"{cms['Accuracy']:.1%} / {cms['Precision']:.1%} / {cms['Recall']:.1%}"),
               ("Start time within 15 min", f"{int((L['Start error (min)'].abs() <= 15).sum())}/{n}"),
               ("LOCALISATION (on the detected time window)", "")] + met
    metrics_df = pd.DataFrame(met, columns=["Metric", "Value"])

    # ------------------------------ SAVE ------------------------------
    log("\n[5] Writing results ...")
    save_pipe_confusion_png(out_dir / "confusion_matrix.png", cm, zcm, f"Which pipe? accuracy {acc:.1%} ({n} test leaks)")
    save_embedding_png(out_dir / "embedding_space.png", loc, np.array(test_Z) if test_Z else None)
    rd = pd.DataFrame(list(readme), columns=["File / topic", "Sheet / term", "Meaning"])
    files = [safe_save(out_dir / "training_data.xlsx", {
                 "README": rd, "Training_summary": tsum, "Model_validation": report, "Validation_per_pipe": per_pipe,
                 "Training_history": hist, "Reference_embeddings": loc.reference_table(), "All_embeddings": all_emb,
                 "Fingerprints": pd.concat([meta, pd.DataFrame(X.round(5), columns=net.sensor_names)], axis=1)[meta.Noise_copy == 0]}),
             safe_save(out_dir / "test_data.xlsx", {"README": rd, "Junction_elevations": network_info(net)[0],
                                                     "Pipe_data": network_info(net)[1],
                                                     **epanet_sheets(net, [f"{r_['No.']}. {r_['Test']}" for r_ in rows], sims, changes, ons)}),
             safe_save(out_dir / "localisation_results.xlsx", {
                 "README": rd, "Results": table, "Metrics": metrics_df, "Pipe_confusion_matrix": cm.reset_index(),
                 "Zone_confusion_matrix": zcm.reset_index().rename(columns={"index": ""}), "Per_pipe_summary": per})]
    for f_ in ("all_embeddings.csv", "confusion_matrix.png", "embedding_space.png", "training_curve.png", "training_raw_epanet.csv"):
        if (out_dir / f_).exists(): files.append(out_dir / f_)
    extra = (f'<h2 style="font-size:16px">Metrics</h2>{metrics_df.to_html(index=False, classes="tbl", border=0)}'
             f'<h2 style="font-size:16px">Confusion matrices</h2>{img_tag(out_dir / "confusion_matrix.png", 1100)}'
             f'<h2 style="font-size:16px">Embedding space (bottleneck of the encoder-decoder)</h2>{img_tag(out_dir / "embedding_space.png", 750)}'
             f'<h2 style="font-size:16px">Training vs validation</h2>{img_tag(out_dir / "training_curve.png", 900)}')
    month_ev = None
    if detect and cfg.get("MONTH_LEAKS") is not None:                    # one month of operation
        mf, mh, month_ev = run_month(net, cfg, det, loc, scale, rng, out_dir, log)
        files += mf; extra = mh + extra
    files.append(write_html(out_dir / "leak_localisation_report.html", net, title, cases, table.drop(columns=["Top-3 candidate pipes"]),
                            notes="Red = induced leak, blue = localised leak, orange tick = 70 % mark of the induced pipe.", extra_html=extra))

    # ------------------------------ CONSOLE REPORT ------------------------------
    show = ["No.", "Test", "Emitter coef", "Induced position (%)", "Localised pipe", "Localised position (%)", "Pipe correct",
            "Zone correct", f"Error ({u})"]
    if detect: show[3:3] = ["Leak time (induced)", "Leak time (detected)"]
    log("\n" + "=" * 100 + "\nTEST RESULTS\n" + "=" * 100)
    for p in pipes_t:
        log(f"\nTEST PIPE {p}"); print_table(table.loc[table["Leak pipe"] == p, show])
    if detect and (table["Leak pipe"] == "none").any():
        log("\nLEAK-FREE DAYS"); print_table(table.loc[table["Leak pipe"] == "none", ["No.", "Test", "Leak time (detected)", "Detection"]])
    log("\nPER-PIPE SUMMARY"); print_table(per)
    log("\nPIPE CONFUSION MATRIX (rows = actual pipe, columns = localised pipe)")
    if cm.shape[1] <= 30: print_table(cm.reset_index())
    else: log("  (large - see confusion_matrix.png / localisation_results.xlsx)")
    log("\nZONE CONFUSION MATRIX (leaks with the correct pipe)"); print_table(zcm.reset_index().rename(columns={"index": ""}))
    log("\nMETRICS"); print_table(metrics_df)
    if month_ev is not None:
        log("\nONE MONTH OF OPERATION - LEAK EVENTS")
        print_table(month_ev[[c for c in ["Leak", "Leak pipe", "Emitter coef", "Leak started", "Leak repaired", "Detected from",
                                          "Detected to", "Localised pipe", "Induced position (%)", "Localised position (%)"] if c in month_ev]])
    what = {"training_data.xlsx": "TRAINING data: fingerprints of all simulated leaks, embeddings, validation",
            "all_embeddings.csv": f"EMBEDDING DATABASE: {loc.Z.shape[1]}-number embedding of every simulated leak (same as in the .pkl)",
            "training_raw_epanet.csv": "TRAINING data: raw EPANET pressure / flow / velocity of every training leak",
            "test_data.xlsx": "TEST data: raw EPANET pressure / head / flow / velocity + sensor changes of every test day",
            "localisation_results.xlsx": "results, metrics, confusion matrices, per-pipe summary",
            "month_simulation.xlsx": "ONE MONTH: 15-min demand / leak flow / supply, leak events, daily + hourly summaries",
            "month_demand.png": "ONE MONTH: demand vs time, detections, daily pattern, day x hour map, daily volumes"}
    log("\nFILES WRITTEN:")
    for f_ in files: log(f"  {Path(f_).resolve()}" + (f"\n      -> {what[Path(f_).name]}" if Path(f_).name in what else ""))
    open_file(files[-1])
    if cfg.get("OPEN_OUTPUT_FOLDER", True): open_file(out_dir)
    return table

# =====================================================================================================
# PART F - ONE MONTH OF OPERATION: water demand over time, leaks that start and get fixed
# =====================================================================================================
def month_demand(rng, days, weekend_factor, trend):
    """KNOWN demand level per 15-min step (calendar effects the utility knows):
    weekends use a bit more water, plus a slow trend over the month (e.g. weather)."""
    n = days * 96 + 1; d = np.minimum(np.arange(n) // 96, days - 1)
    weekend = np.isin(d % 7, (5, 6))                         # day 1 = Monday
    return (1.0 + trend * np.sin(2 * np.pi * np.arange(n) / n)) * np.where(weekend, weekend_factor, 1.0)


def month_abs_hours(day, hhmm_):
    return (int(day) - 1) * 24 + hours(hhmm_)


def locate_long(loc, R, S, i0, i1, piece=12):
    """Localise a detected event that may last many hours / days.  The localiser was trained on 1-3 h leaks, so a
    long event is cut into ~3 h pieces (12 x 15 min); every piece is localised and the pieces VOTE:
    pipe = most frequent pipe, position = median position of the pieces that chose that pipe."""
    n = max(1, round((i1 - i0) / piece)); b = np.linspace(i0, i1, n + 1).astype(int)
    rs = [loc.locate(signature(R, a, c, S)) for a, c in zip(b[:-1], b[1:])]
    pipes_ = [r["pipe"] for r in rs]; best = max(pipes_, key=pipes_.count)
    return dict(pipe=best, pos=float(np.median([r["pos"] for r in rs if r["pipe"] == best])),
                pieces=f"{pipes_.count(best)}/{len(rs)}")


def run_month(net, cfg, det, loc, scale, rng, out_dir, log=print):
    """Simulate MONTH_DAYS days in 15-min steps with the leaks of MONTH_LEAKS, detect them with the Matrix Profile
    and localise every detected event with the embedding database."""
    days, leaks = cfg["MONTH_DAYS"], cfg["MONTH_LEAKS"]
    log(f"\n[6] ONE MONTH OF OPERATION: {days} days, {len(leaks)} leak events (start -> repaired)")
    known = month_demand(rng, days, cfg["WEEKEND_FACTOR"], cfg["MONTH_TREND"])
    noise = demand_noise(rng, days * 24.0, cfg["DEMAND_NOISE"])            # unknown random part
    sched, pos_of = {}, {}
    for L in leaks:
        pos_of.setdefault(L["pipe"], L["position"])
        sched.setdefault(L["pipe"], []).append((L["coefficient"], month_abs_hours(L["start_day"], L["start"]),
                                                month_abs_hours(L["end_day"], L["end"])))
    h = net.open(leaks=list(pos_of.items())) if leaks else net.open()
    obs = net.run(h, days * 24.0, dmult=known, dnoise=noise, schedule=sched); net.close(h)
    exp, S = demand_response(net, obs, known)
    R = residual(net, obs, exp, rng, cfg["PRESSURE_NOISE"])
    score, events = det.detect(obs["t"], R)
    events.sort(key=lambda e: e["start_h"])
    t = obs["t"]; actual = np.zeros(len(t), bool)
    for L in leaks:
        actual |= (t >= month_abs_hours(L["start_day"], L["start"])) & (t < month_abs_hours(L["end_day"], L["end"]))
    alarm = np.zeros(len(t), bool)
    for e in events: alarm[e["i0"]:e["i1"]] = True
    dtxt = lambda x: f"day {int(x // 24) + 1} {hhmm(x % 24)}"
    # ---- match every induced leak with the detected event that overlaps it, localise the event ----
    ev_rows, used = [], set()
    for k, L in enumerate(leaks, 1):
        on, off = month_abs_hours(L["start_day"], L["start"]), month_abs_hours(L["end_day"], L["end"])
        hit = [e for e in events if e["start_h"] < off and e["end_h"] > on]
        e = hit[0] if hit else None
        r = None
        if e:
            used.add(id(e)); r = locate_long(loc, R / scale, S / scale, e["i0"], e["i1"])
        ev_rows.append({"Leak": f"Month leak {k}", "Leak pipe": L["pipe"], "Emitter coef": L["coefficient"],
                        "Induced position (%)": round(100 * L["position"]),
                        "Leak started": dtxt(on), "Leak repaired": dtxt(off), "Duration (h)": round(off - on, 2),
                        "Leaked volume (m3)": round(float(obs["L"][(t >= on) & (t < off)].sum()) * STEP_S / 1000, 1),
                        "Detected from": dtxt(e["start_h"]) if e else "not detected", "Detected to": dtxt(e["end_h"]) if e else "-",
                        "Start error (min)": round((e["start_h"] - on) * 60) if e else None,
                        "Localised pipe": r["pipe"] if r else "-", "Localised position (%)": round(100 * r["pos"]) if r else None,
                        "3-h pieces voting for this pipe": r["pieces"] if r else "-",
                        "Pipe correct": ("YES" if r["pipe"] == L["pipe"] else "NO") if r else "-",
                        "Zone correct": ("YES" if r["pipe"] == L["pipe"] and zone(r["pos"]) == zone(L["position"]) else "NO") if r else "-",
                        f"Error ({net.len_unit})": round(net.leak_distance(L["pipe"], L["position"], r["pipe"], r["pos"]), 1) if r else None})
    for e in events:
        if id(e) not in used:
            ev_rows.append({"Leak": "FALSE ALARM", "Detected from": dtxt(e["start_h"]), "Detected to": dtxt(e["end_h"])})
    ev = pd.DataFrame(ev_rows) if ev_rows else pd.DataFrame(columns=["Leak", "Detected from", "Detected to"])
    # ---- 15-min table, daily and hourly summaries ----
    qu = net.units[1]; vol = STEP_S / 1000
    dayno = np.minimum((t // 24).astype(int) + 1, days)
    m15 = pd.DataFrame({"Day": dayno, "Weekday": [("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[(d - 1) % 7] for d in dayno],
                        "Time": [hhmm(x % 24) for x in t], "Hours from start": t,
                        f"Consumer demand [{qu}]": obs["D"].round(3), f"Leak flow [{qu}]": obs["L"].round(3),
                        f"Total water supplied [{qu}]": (obs["D"] + obs["L"]).round(3),
                        "Known demand level": known.round(3), "Leak on (induced)": actual.astype(int),
                        "Anomaly score": score.round(3), "Alarm (detected)": alarm.astype(int)})
    daily = m15.groupby(["Day", "Weekday"], sort=False).agg(**{
        "Consumer demand (m3)": (f"Consumer demand [{qu}]", lambda x: round(x.sum() * vol, 1)),
        "Leakage (m3)": (f"Leak flow [{qu}]", lambda x: round(x.sum() * vol, 2)),
        f"Peak demand [{qu}]": (f"Consumer demand [{qu}]", "max"),
        "Leak hours": ("Leak on (induced)", lambda x: x.sum() / 4), "Alarm hours": ("Alarm (detected)", lambda x: x.sum() / 4)}).reset_index()
    m15["Hour"] = (t % 24).astype(int)
    hourly = m15.groupby("Hour").agg(**{f"Mean demand [{qu}]": (f"Consumer demand [{qu}]", "mean"),
                                        f"Min demand [{qu}]": (f"Consumer demand [{qu}]", "min"),
                                        f"Max demand [{qu}]": (f"Consumer demand [{qu}]", "max")}).round(3).reset_index()
    m15 = m15.drop(columns="Hour")
    a, p_ = actual, alarm
    cmm = metrics(int((a & p_).sum()), int((~a & p_).sum()), int((a & ~p_).sum()), int((~a & ~p_).sum()))
    ok_det = sum(1 for r_ in ev_rows if r_["Leak"] != "FALSE ALARM" and r_["Detected from"] != "not detected")
    log(f"    month: {ok_det}/{len(leaks)} leaks detected, {int((ev['Leak'] == 'FALSE ALARM').sum())} false alarms, "
        f"15-min accuracy {cmm['Accuracy']:.2%}, precision {cmm['Precision']:.1%}, recall {cmm['Recall']:.1%}")
    # ---- figures ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 11)); gs = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1])
    ax0 = fig.add_subplot(gs[0, :]); ax1 = fig.add_subplot(gs[1, :], sharex=ax0)
    ax0.plot(t / 24 + 1, obs["D"] + obs["L"], lw=0.7, color="#1f77b4", label="total water supplied (demand + leak)")
    ax0.plot(t / 24 + 1, obs["D"], lw=0.7, color="#7f7f7f", label="consumer demand")
    for L in leaks:
        on, off = month_abs_hours(L["start_day"], L["start"]), month_abs_hours(L["end_day"], L["end"])
        ax0.axvspan(on / 24 + 1, off / 24 + 1, color="#d62728", alpha=0.25)
        ax0.text((on + off) / 48 + 1, ax0.get_ylim()[1] if False else (obs["D"] + obs["L"]).max() * 1.02, f"{L['pipe']}", color="#d62728",
                 ha="center", fontsize=9, fontweight="bold")
    for d in range(1, days + 1):
        if (d - 1) % 7 in (5, 6): ax0.axvspan(d, d + 1, color="#000", alpha=0.04, lw=0)
    ax0.set_ylabel(f"flow [{qu}]"); ax0.legend(loc="upper left", fontsize=9)
    ax0.set_title("One month of operation: water demand every 15 min (red = leak active, grey bands = weekend)")
    ax1.plot(t / 24 + 1, score, lw=0.7, color="#263238", label="Matrix-Profile anomaly score")
    ax1.axhline(det.threshold, ls="--", color="#ff9800", label="threshold")
    for e in events: ax1.axvspan(e["start_h"] / 24 + 1, e["end_h"] / 24 + 1, color="#2e7d32", alpha=0.35)
    ax1.set_ylabel("anomaly score"); ax1.set_xlabel("day of month"); ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Leak detection over the month (green = detected window)"); ax1.set_xlim(1, days + 1)
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.fill_between(hourly.Hour, hourly.iloc[:, 2], hourly.iloc[:, 3], color="#1f77b4", alpha=0.2, label="min - max")
    ax2.plot(hourly.Hour, hourly.iloc[:, 1], color="#1f77b4", lw=2, label="mean")
    ax2.set_xticks(range(0, 24, 3)); ax2.set_xlabel("hour of day"); ax2.set_ylabel(f"demand [{qu}]")
    ax2.set_title("Daily demand pattern (all days)"); ax2.legend(fontsize=8)
    ax3 = fig.add_subplot(gs[2, 1])
    hm = (obs["D"] + obs["L"])[:days * 96].reshape(days, 96)
    im = ax3.imshow(hm, aspect="auto", cmap="viridis", extent=[0, 24, days + 0.5, 0.5])
    ax3.set_xlabel("hour of day"); ax3.set_ylabel("day"); ax3.set_title("Water supplied: day x hour"); fig.colorbar(im, ax=ax3)
    ax4 = fig.add_subplot(gs[2, 2])
    ax4.bar(daily.Day, daily["Consumer demand (m3)"], color="#7f7f7f", label="consumer demand")
    ax4.bar(daily.Day, daily["Leakage (m3)"], bottom=daily["Consumer demand (m3)"], color="#d62728", label="leakage")
    ax4.set_xlabel("day"); ax4.set_ylabel("volume per day (m3)"); ax4.set_title("Daily volume: demand + leakage"); ax4.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "month_demand.png", dpi=140); plt.close(fig)
    # ---- workbook ----
    readme = pd.DataFrame([
        ("Month_15min", "Every 15 min of the month: consumer demand, leak flow, total water supplied, known demand level "
                        "(weekend / trend), leak on (induced), anomaly score and alarm (detected)."),
        ("Leak_events", "Every induced leak: pipe, size, position, when it started and was repaired, leaked volume, detected window, "
                        "localised pipe / position. Long events are cut into ~3 h pieces that are localised separately and "
                        "vote (column '3-h pieces voting for this pipe'). Detected events that match no leak are listed as "
                        "FALSE ALARM (a short alarm caused by an unusual random demand peak, not by a leak)."),
        ("Daily_summary", "Per day: consumed volume, leaked volume, peak demand, hours with a leak / with an alarm."),
        ("Hourly_profile", "Average, minimum and maximum demand for every hour of the day over the whole month."),
        ("Metrics_15min", "Confusion matrix of the detector over all 15-min steps of the month.")], columns=["Sheet", "Meaning"])
    mfile = safe_save(out_dir / "month_simulation.xlsx", {
        "README": readme, "Month_15min": m15, "Leak_events": ev, "Daily_summary": daily, "Hourly_profile": hourly,
        "Metrics_15min": pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in cmm.items()}])})
    # ---- interactive chart for the HTML report ----
    mdata = json.dumps(dict(x=(t / 24 + 1).round(4).tolist(), s=(obs["D"] + obs["L"]).round(3).tolist(), d=obs["D"].round(3).tolist(),
                            sc=score.round(3).tolist(), thr=det.threshold,
                            leaks=[[month_abs_hours(L["start_day"], L["start"]) / 24 + 1, month_abs_hours(L["end_day"], L["end"]) / 24 + 1, L["pipe"]] for L in leaks],
                            dets=[[e["start_h"] / 24 + 1, e["end_h"] / 24 + 1] for e in events]))
    html = f"""<h2 style="font-size:16px">One month of operation - water demand vs time (zoom with the mouse)</h2>
<div class="note">Blue = total water supplied, grey = consumer demand, red band = leak active (induced), green band = leak detected by the
Matrix Profile. Morning and evening peaks repeat every day; a leak adds extra flow until it is repaired.</div>
<div id="month" style="height:520px;background:#fff"></div>{ev.to_html(index=False, classes="tbl", border=0, na_rep="")}
<script>(function(){{const M={mdata};const sh=[];
M.leaks.forEach(l=>sh.push({{type:'rect',xref:'x',yref:'paper',x0:l[0],x1:l[1],y0:0.5,y1:1,fillcolor:'#d62728',opacity:0.25,line:{{width:0}}}}));
M.dets.forEach(d=>sh.push({{type:'rect',xref:'x',yref:'paper',x0:d[0],x1:d[1],y0:0,y1:0.45,fillcolor:'#2e7d32',opacity:0.3,line:{{width:0}}}}));
Plotly.newPlot('month',[{{x:M.x,y:M.s,name:'total water supplied',line:{{color:'#1f77b4',width:1}}}},
 {{x:M.x,y:M.d,name:'consumer demand',line:{{color:'#7f7f7f',width:1}}}},
 {{x:M.x,y:M.sc,name:'anomaly score',yaxis:'y2',line:{{color:'#263238',width:1}}}},
 {{x:[M.x[0],M.x[M.x.length-1]],y:[M.thr,M.thr],name:'threshold',yaxis:'y2',line:{{dash:'dash',color:'#ff9800'}}}}],
 {{shapes:sh,annotations:M.leaks.map(l=>({{x:(l[0]+l[1])/2,y:1,yref:'paper',text:l[2],showarrow:false,font:{{color:'#d62728'}}}})),
  xaxis:{{title:'day of month',dtick:1}},yaxis:{{title:'flow',domain:[0.5,1]}},yaxis2:{{title:'anomaly score',domain:[0,0.45]}},
  legend:{{orientation:'h',y:-0.15}},margin:{{l:60,r:20,t:20,b:40}}}},{{displaylogo:false,scrollZoom:true}});}})();</script>
{img_tag(out_dir / "month_demand.png", 1100)}"""
    return [mfile, out_dir / "month_demand.png"], html, ev


# =====================================================================================================
# SETTINGS  (edit here)
# =====================================================================================================
try:
    HERE = Path(__file__).resolve().parent
except NameError:                      # running inside Jupyter
    HERE = Path.cwd()
INPUT_FILE = Path(r"C:\Users\abhis\Desktop\Thesis Work\EPANET Python toolkit\INP file\Drumchapel.inp")
if not INPUT_FILE.exists():
    INPUT_FILE = HERE / "Drumchapel.inp"
OUTPUT_DIR = HERE / "drumchapel_detection_localisation_outputs"     # all results are written here

CFG = dict(
    SEED=42,
    # ---- TRAINING of the localiser: leaks simulated on EVERY open pipe -> reference embedding database ----
    POSITIONS_PER_PIPE=20,          # leak positions per pipe (14 in 0-70 %, 6 in 70-100 % of the length)
    RUNS_PER_POSITION=3,            # each with random size, start time, duration, demand level
    NOISE_COPIES=3,                 # extra copies of each fingerprint with sensor noise (no extra EPANET runs)
    LEAK_COEF_RANGE=(0.2, 1.0),     # emitter coefficient range (LPS/m^0.5) for training AND test leaks
    TRAIN_DURATIONS_H=(1.0, 2.0, 3.0),
    DEMAND_MULTIPLIERS=(0.9, 1.0, 1.1),
    SAVE_RAW_TRAINING_DATA=False,   # True = raw EPANET data of all ~9000 training leaks to CSV (several GB!)
    # ---- sensors / noise ----
    PRESSURE_NOISE=0.01,            # m
    DEMAND_NOISE=0.02,              # 2 % unknown demand fluctuation
    # ---- DETECTION (Matrix Profile, trained on 16 leak-free days) ----
    MP_WINDOW=4,                    # window = 4 x 15 min = 1 h
    THRESHOLD_FACTOR=1.2,           # threshold = 1.2 x highest score on 12 leak-free days
    # ---- encoder-decoder ----
    HIDDEN_DIM=128, LATENT_DIM=32,  # encoder width and bottleneck (embedding) size
    WEIGHT_DECAY=1e-3, MAX_EPOCHS=400, PATIENCE=30,
    # ---- TESTING (unseen leaks) ----
    TEST_PIPES=["P93", "P16", "P122", "P133", "P35", "P87"],   # your test pipes
    N_RANDOM_TEST_PIPES=6,          # + this many randomly chosen other pipes
    LEAKS_PER_TEST_PIPE=4,          # leaks per test pipe (yours + random position / size / time)
    N_LEAK_FREE_TESTS=10,           # leak-free test days (false-alarm check)
    # ---- ONE MONTH OF OPERATION ----
    MONTH_DAYS=30,
    WEEKEND_FACTOR=1.08,            # weekends use 8 % more water
    MONTH_TREND=0.04,               # slow +-4 % change of demand over the month
    MONTH_LEAKS=[                   # leaks during the month: day 1-30, time "HH:MM"; a leak may last several days
        dict(pipe="P93", position=0.46, coefficient=0.657, start_day=4, start="07:00", end_day=4, end="10:30"),
        dict(pipe="P16", position=0.66, coefficient=0.40, start_day=9, start="22:00", end_day=10, end="06:00"),
        dict(pipe="P87", position=0.50, coefficient=0.30, start_day=15, start="14:30", end_day=17, end="09:00"),
        dict(pipe="P122", position=0.85, coefficient=0.80, start_day=22, start="18:00", end_day=22, end="20:00"),
        dict(pipe="P35", position=0.30, coefficient=0.25, start_day=27, start="02:00", end_day=27, end="05:00"),
    ],
    WORKERS=None,                   # CPU processes for the training simulations (None = automatic, 1 = no parallel)
    OPEN_OUTPUT_FOLDER=True,
)

# Your own test leaks (always included): pipe, position 0-1, emitter coefficient, start/end "HH:MM"
TEST_LEAKS = [
    dict(pipe="P93", position=0.46, coefficient=0.657, start="06:00", end="08:30"),
    dict(pipe="P16", position=0.66, coefficient=0.970, start="10:00", end="11:30"),
    dict(pipe="P122", position=0.85, coefficient=0.300, start="22:00", end="23:30"),
    dict(pipe="P87", position=0.50, coefficient=0.500, start="14:30", end="16:00"),
]

README = [
    ("HOW IT WORKS", "1. DETECTION", "Matrix Profile: every hour of sensor changes is compared with 16 leak-free days; "
                                     "behaviour never seen before = alarm -> detected start time and window."),
    ("", "2. LOCALISATION", "Leaks simulated at 20 positions on EVERY pipe -> fingerprints -> encoder-decoder -> 32-number "
                            "embedding (bottleneck). A detected leak's embedding is matched with all stored embeddings -> pipe + position."),
    ("", "3. ONE MONTH", "30 days in 15-min steps with daily demand cycle, weekends, trend and leaks that start and get repaired."),
    ("", "Sensors", "Hydraulic head (= pressure + elevation) at every junction, flow and velocity in every pipe."),
    ("", "Unseen test data", "Test leaks use new random days, positions, sizes and times, sensor noise and unknown demand changes."),
    ("training_data.xlsx", "Training_summary", "Simulated training leaks per pipe."),
    ("", "Fingerprints", "Input of the encoder-decoder: one row per simulated leak + fingerprint value of every sensor."),
    ("", "All_embeddings", "THE EMBEDDING DATABASE: every simulated leak (pipe, position, size, time) + its embedding z1..z32."),
    ("", "Reference_embeddings", "Average embedding per pipe and zone (summary, for reporting)."),
    ("", "Model_validation / Training_history", "Accuracy on leak positions held out of training; loss / accuracy per epoch."),
    ("all_embeddings.csv", "", "Same as the All_embeddings sheet, as CSV."),
    ("test_data.xlsx", "EPANET_Pressure / _Head / _Flow / _Velocity", "RAW EPANET output of every test day (rows = test x 15-min step)."),
    ("", "Sensor_change", "Measured minus model-expected value of every sensor (input of detection and localisation)."),
    ("localisation_results.xlsx", "Results", "One row per test day: induced pipe / position / time vs detected time, localised pipe / position."),
    ("", "Metrics", "Detection TP / FN / FP / TN, accuracy, precision, recall; localisation accuracy, precision, recall, F1, error."),
    ("", "Pipe / Zone confusion matrix", "Actual vs localised pipe; actual vs localised 70/30 zone."),
    ("month_simulation.xlsx", "", "One month: 15-min demand, leak flow, supply, leak events, daily and hourly summaries."),
    ("trained_model_cache.pkl", "", "Trained encoder-decoder + embedding database, reused on the next run. Delete it to retrain."),
    ("TERMS", "Emitter coef", "Leak size: leak flow = coef x sqrt(pressure at the leak), e.g. 0.3 x sqrt(30 m) = 1.6 L/s."),
    ("", "Position (%)", "Where on the pipe: 0 % = start node, 100 % = end node (as in the INP file)."),
    ("", "Precision / Recall", "Precision: of all alarms (or leaks put in pipe X), how many were right. "
                               "Recall: of all real leaks (in pipe X), how many were found."),
]


if __name__ == "__main__":
    print("=" * 100 + "\nDRUMCHAPEL - LEAK DETECTION (MATRIX PROFILE) + LOCALISATION (ENCODER-DECODER EMBEDDINGS) + ONE MONTH\n" + "=" * 100)
    run_study(INPUT_FILE, CFG, TEST_LEAKS, OUTPUT_DIR,
              "Drumchapel - leak detection (Matrix Profile) + localisation (encoder-decoder embeddings)",
              detect=True, readme=README)
