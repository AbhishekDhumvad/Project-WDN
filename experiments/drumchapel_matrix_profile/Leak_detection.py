"""
Leak_detection.py  -  DRUMCHAPEL: REAL-TIME leak DETECTION with the Matrix Profile  (stand-alone script)
=======================================================================================================
Questions answered:  Is there a leak?  HOW FAST is it detected after it starts (detection delay)?
                     Are several leaks at the same time each detected?

How it works
  1. EPANET simulates one day (00:00 - 24:00, a sensor reading every 15 min) with the leak switched on between
     start and end.  A leak is either SUDDEN (burst) or GROWING (reaches full size within 1 or 2 h).
     Unknown random demand changes (2 %) and sensor noise make the test realistic.
  2. "Sensor change" = measured value - value the model expects without a leak, for every sensor
     (hydraulic head = pressure + elevation at each junction, flow and velocity in each pipe).  Outside a leak ~0.
  3. REAL-TIME MATRIX PROFILE: at every new reading, only the data up to NOW is used.  The last 15, 30 and
     60 min of sensor changes are compared with the most similar piece of the leak-free reference days.
     Something never seen in normal operation gives a high score.  ALARM = score above the threshold.
     Detection delay = alarm time - real leak start (leaks start at a random minute, between two readings).
  4. LOOKING BACK: after the alarm, the whole day is analysed again to estimate when the leak started and ended.
  The detector learns ONLY from leak-free days; the true leak times are used only to judge the result.

Data sets  (80 / 20 split of the pipes)
  The 151 open pipes of Drumchapel.inp (154 pipes, 3 closed) are split at random: 80 % TRAINING pipes and
  20 % TEST pipes (the user-defined leak pipes are always training pipes; SPLIT_SEED fixes the split).
  NORMAL days   : leak-free days -> what normal operation looks like + the thresholds.
  TRAINING leaks: leaks on the TRAINING pipes only (user-defined leaks and multi-leak days + random leaks on every other
                  training pipe + random 2-5-leak days) -> used ONLY to tune the detector settings automatically
                  (threshold and new-leak rules). Never part of the test.
  TEST (unseen) : leaks ONLY on the TEST pipes: days with one leak (random order) and days with 2, 3, 4 or 5 overlapping
                  leaks, random positions, times, sizes, sudden / growing, + a few leak-free days.
  For every leak the detector reports WHEN it was detected (and the detection delay). A new leak during an active
  alarm is recognised by a sudden NEW sensor pattern. No repair time is reported: when a leak will be repaired is
  not known in real operation (internally the alarm is switched off once the sensors are normal again, so that
  the next leak can be detected).

Outputs (folder Drumchapel_matrix_profile_leak_outputs)
  training_data.xlsx, simulated_data.xlsx, detection_results.xlsx, confusion_matrix.png, detection_delay.png,
  test_leaks_map.png, leak_detection_report.html

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
STEP_S = 900          # sensor reading interval = hydraulic + reporting step (s); a script may set its own
# Sensor features used for detection / localisation (can be overridden in the SETTINGS section):
#   "H" = hydraulic head at every junction (pressure + elevation)    "Q" = flow in every open pipe
#   "V" = velocity in every open pipe                                 "Qin" = only the inlet flow meters
FEATURES = ("H", "Q", "V")
ZONE_EDGES = (0.0, 0.70, 1.0)   # zones along a pipe (fraction of its length from the start node); a script may set its own


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


def zone_index(p):
    """Number of the zone (0, 1, ...) that position p (0-1 along the pipe) falls in."""
    return int(np.clip(np.searchsorted(ZONE_EDGES, p, side="right") - 1, 0, len(ZONE_EDGES) - 2))


def zone(p):
    """Zone label of position p, e.g. '20-40%'."""
    i = zone_index(p); return f"{100 * ZONE_EDGES[i]:.0f}-{100 * ZONE_EDGES[i + 1]:.0f}%"


def zone_labels():
    return [zone(0.5 * (a + b)) for a, b in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:])]


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
                hr = G.getTimeHTime() / 3600; k = int(round(hr * 3600 / STEP_S))   # reading number
                act = []
                for p_, node in zip(h["lpipes"], h["lnodes"]):
                    c_ = next((leak_coef(e_, hr) for e_ in sched.get(p_, []) if e_[0] > 0 and e_[1] <= hr + 1e-9 < e_[2]), 0.0)
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

    def simulate_leak(self, pipe, pos, coef, on, off, hours_=24.0, dmult=1.0, dnoise=None, growth_h=0.0):
        """pipe=None -> leak-free day.  growth_h > 0: the leak grows linearly to full size within growth_h hours."""
        h = self.open(pipe, pos)
        try: return self.run(h, hours_, dmult=dmult, dnoise=dnoise,
                             schedule={pipe: [(coef, on, off, growth_h)]} if pipe is not None else {})
        finally: self.close(h)

    def simulate_leaks(self, leaks, hours_=24.0, dmult=1.0, dnoise=None):
        """Several leaks in one simulation. leaks: list of dict(pipe, position, coefficient, on, off[, growth_h])."""
        h = self.open(leaks=list({L_["pipe"]: L_["position"] for L_ in leaks}.items()))
        sched = {}
        for L_ in leaks: sched.setdefault(L_["pipe"], []).append((L_["coefficient"], L_["on"], L_["off"], L_.get("growth_h", 0.0)))
        try: return self.run(h, hours_, dmult=dmult, dnoise=dnoise, schedule=sched)
        finally: self.close(h)


def leak_coef(ev, hr):
    """Emitter coefficient of a scheduled leak (coef, on, off[, growth_h]) at time hr (h).
    growth_h = 0: sudden leak (burst).  growth_h > 0: the leak grows step by step to full size in growth_h hours."""
    c, on = ev[0], ev[1]; g = ev[3] if len(ev) > 3 else 0.0
    return c if g <= 0 else c * min(1.0, (hr - on + STEP_S / 3600) / g)


def demand_noise(rng, hours_, std):
    """Smooth random fluctuation of all demands that the model does NOT know: one factor per sensor reading
    (made on a 15-min grid and interpolated when readings are more frequent)."""
    if std <= 0: return None
    n = int(hours_ * 4) + 1
    f = 1.0 + np.convolve(rng.normal(0, std, n + 4), np.ones(4) / 2, mode="same")[:n]
    if STEP_S == 900: return f
    return np.interp(np.arange(int(round(hours_ * 3600 / STEP_S)) + 1) * STEP_S / 900, np.arange(n), f)


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

    # ---------------- REAL-TIME (online) detection: only data up to "now" is used ----------------
    def fit_online(self, null_days, windows=(1, 2, 4), top_k=0, ref_days=None):
        """Thresholds for the real-time detector: for every window length L (in readings) the highest
        trailing-window score on the leak-free threshold days.
        top_k > 0 adds a second view: the TOP-K SENSORS matrix profile (idea of the k-dimensional matrix profile,
        mSTAMP): every sensor is compared on its own with the leak-free reference days and only the k most abnormal
        sensors count -> a leak that strongly changes only a few sensors (e.g. on a main supply pipe) is not
        diluted by the hundreds of sensors that do not change."""
        self.windows = tuple(windows); self.top_k = int(top_k)
        if self.top_k:
            ref = np.concatenate(ref_days); sd = ref.std(0); self.top_sd = np.maximum(sd, 0.1 * np.median(sd) + 1e-9)
            self.top_ref = np.sort(ref / self.top_sd, axis=0)
            self.thr_top = {L: max(self.top_score(r, L).max() for r in null_days) for L in self.windows}
        self.thr_on = {L: max(self.trailing(r, L).max() for r in null_days) for L in self.windows}
        return self

    def top_score(self, R, L):
        """Top-k sensors: per sensor the distance to its closest leak-free reference value (1-D matrix profile),
        averaged over the last L readings; score = mean of the k largest sensor distances."""
        Z = R / self.top_sd; D = np.empty_like(Z)
        for j in range(Z.shape[1]):
            r = self.top_ref[:, j]; k = np.clip(np.searchsorted(r, Z[:, j]), 1, len(r) - 1)
            D[:, j] = np.minimum(np.abs(Z[:, j] - r[k - 1]), np.abs(Z[:, j] - r[k]))
        if L > 1:
            D = np.vstack([np.zeros((L - 1, D.shape[1])), np.lib.stride_tricks.sliding_window_view(D, L, axis=0).mean(-1)])
        return np.sort(D, axis=1)[:, -self.top_k:].mean(1)

    def trailing(self, R, L):
        """Real-time Matrix Profile score: at every reading t, the distance between the LAST L readings
        (t-L+1 ... t) and the most similar L readings of the leak-free reference days.  Uses no future data.
        The first L-1 readings of a day have no full window yet -> 0."""
        X = R / self.scale; s = np.zeros(len(X))
        if len(X) >= L: s[L - 1:] = mp_join(X, self.ref, L)
        return s

    def online_scores(self, R):
        """Score of every window length divided by its threshold (> 1 = abnormal for that window length).
        With the top-k sensor view, each entry is the larger of the two views."""
        out = {L: self.trailing(R, L) / self.thr_on[L] for L in self.windows}
        if getattr(self, "top_k", 0):
            for L in self.windows: out[L] = np.maximum(out[L], self.top_score(R, L) / self.thr_top[L])
        return out

    def fit_new_leak(self, null_days):
        """Normal size of the change between two readings on leak-free days (noise + demand fluctuation)."""
        self.jump_noise = max(self.jumps(r)[0].max() for r in null_days)
        return self

    def jumps(self, R):
        """Change between consecutive readings that is NOT just the current anomaly getting bigger / smaller:
        d = X(t) - X(t-1) with its part along X(t-1) removed.  A new leak somewhere else adds a NEW pattern -> large."""
        X = R / self.scale; nx = np.linalg.norm(X, axis=1); j = np.zeros(len(X)); cos = np.ones(len(X))
        for i in range(1, len(X)):
            u = X[i - 1] / (nx[i - 1] + 1e-12); d = X[i] - X[i - 1]; j[i] = np.linalg.norm(d - (d @ u) * u)
            cos[i] = X[i] @ X[i - 1] / (nx[i] * nx[i - 1] + 1e-12)
        return j, nx, cos

    def monitor(self, t, R, scores=None, factor=None, confirm=1, clear=2, jump_factor=1.5, jump_rel=0.3, swap_cos=0.3, grace=2,
                track_repairs=True):
        """REAL-TIME MONITORING of one day, reading by reading (only data up to 'now' is used).
          * LEAK DETECTED (new alarm): combined score > 1 for `confirm` readings in a row.
          * NEW LEAK DURING AN ACTIVE ALARM: the sensor pattern suddenly changes in a NEW direction
            (change > jump_factor x normal change between readings, and > jump_rel x the current anomaly) and the
            anomaly grows - or the reading stays abnormal with a clearly different pattern (repair + new leak).
          * ONE OF SEVERAL LEAKS REPAIRED: the anomaly suddenly drops in a new direction; the drop is compared with
            the pattern each running leak added when it was detected -> the most similar leak has ended.
          * ALARM ENDS (all leaks repaired): the latest reading is normal again for `clear` readings in a row; the
            windows then start fresh, so a later leak gets its own detection.
          No new-leak / repair check during the first `grace` readings after an event (the change is still building up).
        Returns (combined score, alarms).  Every alarm has 'leaks': one entry per leak the detector found, with
        i_det (reading where it was detected), how, i_end (first reading where it was found repaired)."""
        f = self.factor if factor is None else factor
        raw = scores if scores is not None else self.online_scores(R)
        X = R / self.scale; j, nx, cos = self.jumps(R); thr_j = jump_factor * self.jump_noise
        ucos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        comb_all = np.zeros(len(t)); alarms = []; active = False; seg0 = 0; n_hi = 0; n_lo = 0; seen1 = False
        cur = None; last = -99; run = []
        for i in range(len(t)):
            avail = [L for L in self.windows if L <= i - seg0 + 1]           # windows that start after the last repair
            comb = max([raw[L][i] / f for L in avail] or [0.0]); s1 = raw[min(self.windows)][i] / f
            comb_all[i] = max(comb, 1e-3)
            if not active:
                n_hi = n_hi + 1 if comb > 1 else 0
                if n_hi >= confirm:
                    active, n_lo, seen1 = True, 0, s1 > 1
                    cur = dict(i0=i, raise_h=float(t[i]), leaks=[]); last = i
                    run = [dict(i_det=i, how="new alarm", sig=X[i].copy(), i_end=None)]; cur["leaks"] += run
            else:
                big = i >= 1 and i - last > grace and j[i] > thr_j and (j[i] > jump_rel * nx[i - 1] or cos[i] < swap_cos)
                d = X[i] - X[i - 1] if i >= 1 else X[i]
                if big and nx[i] > nx[i - 1] and j[i] > jump_rel * nx[i - 1]:              # something was ADDED
                    L_ = dict(i_det=i, how="new leak while an alarm was active", sig=d.copy(), i_end=None)
                    run.append(L_); cur["leaks"].append(L_); last = i
                elif big and nx[i] < nx[i - 1]:                                            # something was REMOVED
                    if track_repairs and len(run) >= 2:
                        k = int(np.argmax([ucos(-d, L_["sig"]) for L_ in run]))
                        run[k]["i_end"] = i; run.pop(k); last = i
                        rest = np.sum([L_["sig"] for L_ in run], axis=0)
                        if s1 > 1 and ucos(X[i], rest) < swap_cos:                         # ... and a new one started
                            L_ = dict(i_det=i, how="new leak right after a repair", sig=X[i] - rest, i_end=None)
                            run.append(L_); cur["leaks"].append(L_)
                    elif s1 > 1 and cos[i] < swap_cos:                                     # only leak repaired + new leak
                        for L_ in run: L_["i_end"] = i
                        L_ = dict(i_det=i, how="new leak right after a repair", sig=X[i].copy(), i_end=None)
                        run = [L_]; cur["leaks"].append(L_); last = i
                normal = (s1 <= 1) if seen1 else (comb <= 1); seen1 |= s1 > 1
                n_lo = n_lo + 1 if normal else 0
                if n_lo >= clear:
                    e = i - clear + 1
                    for L_ in run: L_["i_end"] = e
                    cur.update(i1=e, clear_h=float(t[e])); alarms.append(cur)
                    active, seg0, n_hi, cur, run = False, i + 1, 0, None, []
        if active: cur.update(i1=len(t), clear_h=None); alarms.append(cur)
        for a in alarms:
            a["detections"] = [(L_["i_det"], L_["how"]) for L_ in a["leaks"]]
        return comb_all, alarms

    def alarms(self, t, scores, factor=None, confirm=2, clear=2):
        """Real-time alarm logic on the per-window scores.
        combined score = max over window lengths of score / (factor x threshold)  (> 1 = abnormal)
        ALARM RAISED when the combined score is > 1 for `confirm` readings in a row (time = that reading);
        ALARM CLEARED when it is <= 1 for `clear` readings in a row.  Returns (combined score, alarm list)."""
        f = self.factor if factor is None else factor
        comb = np.max([s / f for s in scores.values()], axis=0); above = comb > 1
        ev, n_hi, n_lo, active, i0 = [], 0, 0, False, 0
        for i, a in enumerate(above):
            if not active:
                n_hi = n_hi + 1 if a else 0
                if n_hi >= confirm: active, i0, n_lo = True, i, 0
            else:
                n_lo = n_lo + 1 if not a else 0
                if n_lo >= clear:
                    ev.append(dict(i0=i0, i1=i - clear + 1, raise_h=float(t[i0]), clear_h=float(t[i - clear + 1]))); active = False; n_hi = 0
        if active: ev.append(dict(i0=i0, i1=len(above), raise_h=float(t[i0]), clear_h=None))
        return comb, ev


def build_detector(net, rng, p_noise, d_noise, m=4, factor=1.2, n_ref=4, n_null=12):
    days = normal_days(net, rng, n_ref + n_null, p_noise, d_noise)
    return MatrixProfileDetector(sensor_scale(net, days[:n_ref]), m, factor).fit(days[:n_ref], days[n_ref:])

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


# =====================================================================================================
# PART D2 - FIGURES
# =====================================================================================================
def save_confusion_png(path, cms):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cms), figsize=(6.2 * len(cms), 5))
    for ax, (title, m) in zip(np.atleast_1d(axes), cms.items()):
        M = np.array([[m["TP"], m["FN"]], [m["FP"], m["TN"]]])
        ax.imshow(M, cmap="Blues")
        for (a, b), v in np.ndenumerate(M):
            ax.text(b, a, f"{['TP', 'FN', 'FP', 'TN'][2 * a + b]}\n{v}", ha="center", va="center", fontsize=15,
                    color="white" if v > M.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Detected: LEAK", "Detected: NO LEAK"]); ax.set_yticks([0, 1], ["Actual: LEAK", "Actual: NO LEAK"])
        ax.set_title(f"{title}\nAccuracy {m['Accuracy']:.1%} | Precision {m['Precision']:.1%} | Recall {m['Recall']:.1%} | F1 {m['F1']:.2f}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def save_delay_png(path, single, sweep):
    """HOW FAST is a leak detected?  (a) share of leaks detected within x minutes, (b) delay vs leak size,
    (c, d) leak-size sweep: median delay and detection rate vs emitter coefficient."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = 2 if len(sweep) else 1
    fig, axes = plt.subplots(rows, 2, figsize=(13, 4.6 * rows)); axes = np.atleast_2d(axes)
    ax = axes[0, 0]; x = np.arange(0, 241, 5)
    for name, g, c in (("all leaks", single, "#263238"), ("sudden leaks", single[single.Growth == 0], "#d62728"),
                       ("growing leaks", single[single.Growth > 0], "#1f77b4")):
        if len(g): ax.step(x, [100 * (g.Delay <= v).sum() / len(g) for v in x], where="post", color=c, lw=2, label=f"{name} ({len(g)})")
    ax.set_xlabel("minutes after the leak started"); ax.set_ylabel("% of leaks detected"); ax.set_ylim(0, 102)
    ax.set_title("(a) Time to detection: % of leaks detected within x minutes"); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    ax = axes[0, 1]; top = max(60.0, float(np.nanmax(single.Delay)) if single.Delay.notna().any() else 60.0) + 30
    for grow, c, lab in ((False, "#d62728", "sudden"), (True, "#1f77b4", "growing")):
        g = single[(single.Growth > 0) == grow]
        ax.scatter(g["Emitter coef"], g.Delay, color=c, s=36, label=f"{lab} - detected", alpha=0.8)
        m = g[g.Delay.isna()]
        ax.scatter(m["Emitter coef"], np.full(len(m), top), color=c, marker="x", s=60, label=f"{lab} - missed")
    ax.axhline(top - 15, color="#999", lw=0.8, ls=":"); ax.set_xscale("log")
    ax.set_xlabel("emitter coefficient (leak size)"); ax.set_ylabel("detection delay (min)  -  x = missed")
    ax.set_title("(b) Detection delay of every test leak"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    if len(sweep):
        ax = axes[1, 0]
        for (typ, start), g in sweep.groupby(["Leak type", "Start period"]):
            agg = g.groupby("Emitter coef")["Delay"].median()
            ax.plot(agg.index, agg.values, marker="o", ls="-" if typ == "sudden" else "--", label=f"{typ}, start {start}")
        ax.set_title("(c) Leak-size sweep: median detection delay (detected leaks)"); ax.set_ylabel("median delay (min)")
        ax = axes[1, 1]
        for pipe, g in sweep.groupby("Pipe", sort=False):
            agg = 100 * g.groupby("Emitter coef")["Detected"].mean()
            ax.plot(agg.index, agg.values, marker="o", label=f"pipe {pipe}")
        ax.set_title("(d) Leak-size sweep: % of leaks detected, per pipe"); ax.set_ylabel("% detected (all start times, sudden + growing)")
        for ax in axes[1]:
            ax.set_xscale("log"); ax.set_xlabel("emitter coefficient (leak size)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def img_tag(path, width=1000):
    import base64
    return (f'<img style="max-width:100%;width:{width}px;background:#fff;border:1px solid #ddd" src="data:image/png;base64,'
            + base64.b64encode(Path(path).read_bytes()).decode() + '">')


def save_result_map_png(path, net, ok_pipes, miss_pipes, multi_pipes, train_pipes):
    """Network map: which pipes had a test leak and whether it was detected."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 13))
    for p in net.pipes_all:
        xy = np.array(net.polyline(p)); ax.plot(xy[:, 0], xy[:, 1], color="#c9ced3", lw=1, zorder=1)
    n_tr, n_te = len(train_pipes), len(set(ok_pipes) | set(miss_pipes) | set(multi_pipes))
    for pipes, color, lw, ls, lab in ((train_pipes, "#b0b4b8", 2.5, "-", f"TRAINING pipe ({n_tr} pipes = 80 %, leaks used for tuning)"),
                                      (multi_pipes, "#1f77b4", 3, "--", "TEST pipe - leak only on a multi-leak test day"),
                                      (ok_pipes, "#2e7d32", 4.5, "-", f"TEST pipe ({n_te} pipes = 20 %) - single test leaks detected"),
                                      (miss_pipes, "#d62728", 5, "-", "TEST pipe - a single test leak missed")):
        for k, p in enumerate(pipes):
            xy = np.array(net.polyline(p)); ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, ls=ls, zorder=2, label=lab if k == 0 else None)
    colour = {p: "#4a4f55" for p in train_pipes}           # pipe name just ABOVE every pipe: training grey, test in colour
    for pipes, c in ((multi_pipes, "#1f77b4"), (ok_pipes, "#2e7d32"), (miss_pipes, "#d62728")):
        for p in pipes: colour[p] = c
    for p, c in sorted(colour.items(), key=lambda kv: kv[0] not in train_pipes):     # test names drawn last (on top)
        g = net.polyline(p); k = len(g) // 2; a, b = g[k], g[max(k - 1, 0)]; test = p not in train_pipes
        ax.annotate(p, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), xytext=(0, 6 if test else 4.5), textcoords="offset points",
                    fontsize=7.5 if test else 5, color=c, ha="center", va="bottom", zorder=5 if test else 4,
                    fontweight="bold" if test else "normal",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=c if test else "none", lw=0.6, alpha=0.9 if test else 0.7))
    ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10, frameon=False)
    ax.set_title("Drumchapel network - 80 / 20 split: training pipes (grey) and TEST pipes (colour); name above each pipe", fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# =====================================================================================================
# PART D3 - DETECTION REPORT (HTML)
# =====================================================================================================
def write_detection_html(path, net, title, cards, cms, days, single_table, multi_table, delay_html, mapd, tuning_html):
    geo = {p: net.polyline(p) for p in net.pipes_all}
    data = json.dumps(dict(geo=geo, days=days, cms=cms, map=mapd),
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if o is None else float(o)))
    try:
        import plotly
        plotly_tag = "<script>" + (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8") + "</script>"
    except Exception:
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    html = """<!doctype html><html><head><meta charset="utf-8"><title>@@TITLE@@</title>@@PLOTLY@@
<style>body{font-family:Segoe UI,Arial,sans-serif;margin:18px;background:#f6f7f9;color:#222}h1{font-size:21px;margin:0 0 8px}
h2{font-size:16px;margin:24px 0 6px}h3{font-size:14px;margin:14px 0 4px}.note{font-size:13px;color:#555;margin:2px 0 8px;max-width:1150px}
.cards{display:flex;gap:10px;flex-wrap:wrap}.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 14px;font-size:13px}
.card b{font-size:19px;display:block}.tbl{border-collapse:collapse;font-size:13px;background:#fff}
.tbl th{background:#263238;color:#fff;padding:6px 8px;text-align:left}.tbl td{padding:5px 8px;border-bottom:1px solid #e3e3e3}
.click tbody tr:hover td{background:#e3f2fd;cursor:pointer}.click tr.sel td{background:#fff3c4}
.scroll{max-height:460px;overflow:auto;border:1px solid #ddd}.ok{color:#1b5e20;font-weight:600}.bad{color:#b71c1c;font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:12px}.box{background:#fff}</style></head><body><h1>@@TITLE@@</h1>
<div class="note">The open pipes were split at random: 80 % TRAINING pipes - their leaks (user-defined and random leaks) were used only to
tune the detector settings (section 7) - and 20 % TEST pipes. All results below are on UNSEEN leaks on the TEST pipes only,
at random positions, times and sizes.</div><div class="cards">@@CARDS@@</div>
<h2>1. Confusion matrices</h2><div class="note"><b>Per leak event</b>: days with one leak and leak-free days - a leak day is TP if the
leak is detected in real time while it is on, a leak-free day is TN if no leak is reported. <b>Per sensor reading</b>: every reading
of every test day (also the multi-leak days) - "leak" = at least one leak active, "detected" = the leak alarm is on.</div>
<div class="row"><div id="cm0" class="box" style="width:440px;height:400px"></div><div id="cm1" class="box" style="width:440px;height:400px"></div></div>
<h2>2. How fast are leaks detected?</h2><div class="note">The detector works in REAL TIME:
at every new sensor reading (every @@RI@@ min) it only uses data up to that moment. Detection delay = time the leak was detected -
time the leak really started. Only the START of a leak is reported: when a leak will be repaired is not known in real
operation, so no repair time is given.</div>@@DELAY@@
<h2>3. Training and test pipes on the network</h2><div class="note">Thin grey = TRAINING pipes (80 %, not tested). Coloured and
thick = TEST pipes (20 %): green = all single test leaks on the pipe detected, red = a single test leak missed. Every pipe has its
name (as in the INP file) just above it: test pipes in bold colour, training pipes in small grey. Buttons above the map: all names /
test pipe names only / no names. Move the mouse over a pipe to see its test leaks (also those on the multi-leak days) and whether they
were detected. Scroll to zoom (the names get easier to read), double-click to reset.</div>
<div id="map" class="box" style="height:680px"></div>
<h2>4. Days with ONE leak</h2><div class="note">Leak active (induced) = when the simulated leak was switched on and off in EPANET
(known only in the simulation). Leak detected at = time of the reading at which the detector reported the leak. Click a row to see
the day (section 6).</div><div class="click scroll">@@SINGLE@@</div>
<h2>5. Days with several leaks at the same time (@@MR@@ leaks per day)</h2><div class="note">Every leak gets its own detection: a new leak during an active alarm is recognised by a
sudden NEW sensor pattern. Leaks that start between the same two readings are detected together. Click a row to see the day.</div>
<div class="click scroll">@@MULTI@@</div>
<h2>6. What one day looks like to the real-time detector</h2><div class="note">Black = real-time anomaly score (score / threshold;
above 1 = abnormal). Red bands = leaks on (simulated); green box = leak alarm on; dashed black lines = leak detected.
Choose a day: <select id="pick"></select></div><div id="score" class="box" style="height:460px"></div>
<h2>7. Training: how the detector settings were tuned on the training pipes (80 %)</h2>@@TUNING@@
<script>
const D=@@DATA@@, T=D.days;
document.querySelectorAll('.click tbody tr').forEach(r=>{const c=r.cells[r.cells.length-1];
 c.className=/MISSED|FALSE/.test(c.textContent)?'bad':'ok';
 r.onclick=()=>{show(+r.cells[0].textContent-1);document.getElementById('score').scrollIntoView({behavior:'smooth'})}});
['Per leak event (days)','Per sensor reading'].forEach((ttl,k)=>{const m=D.cms[ttl];
 Plotly.newPlot('cm'+k,[{type:'heatmap',z:[[m.TP,m.FN],[m.FP,m.TN]],x:['Detected: LEAK','Detected: NO LEAK'],y:['Actual: LEAK','Actual: NO LEAK'],
  colorscale:'Blues',showscale:false,text:[['TP '+m.TP,'FN '+m.FN],['FP '+m.FP,'TN '+m.TN]],texttemplate:'%{text}',textfont:{size:18},hoverinfo:'text'}],
 {title:{text:'<b>'+ttl+'</b><br>Accuracy '+(100*m.Accuracy).toFixed(1)+'% | Precision '+(100*m.Precision).toFixed(1)+'% | Recall '+(100*m.Recall).toFixed(1)+'%',font:{size:13}},
  yaxis:{autorange:'reversed'},margin:{l:110,r:10,t:70,b:50}},{displaylogo:false})});
function line(ps,color,w,name,dash){const x=[],y=[];for(const p of ps){for(const [a,b] of D.geo[p]){x.push(a);y.push(b)}x.push(null);y.push(null)}
 return {x,y,mode:'lines',line:{color,width:w,dash:dash||'solid'},name,hoverinfo:'skip'}}
function mid(p){const g=D.geo[p],k=Math.floor(g.length/2),a=g[k],b=g[Math.max(k-1,0)];return [(a[0]+b[0])/2,(a[1]+b[1])/2]}
const all=Object.keys(D.geo), tr=[line(all,'#c9ced3',1.3,'Network pipes')];
const info=D.map.info||{}, hv=all.map(mid);
tr.push({x:hv.map(h=>h[0]),y:hv.map(h=>h[1]),text:all.map(p=>'<b>Pipe '+p+'</b>'+(info[p]?'<br>'+info[p].join('<br>'):'')),mode:'markers',
 marker:{size:10,opacity:0},hoverinfo:'text',showlegend:false});
if(D.map.train.length) tr.push(line(D.map.train,'#b0b4b8',3,'Training pipes (80 %)'));
if(D.map.multi.length) tr.push(line(D.map.multi,'#1f77b4',3,'Test pipe - leak only on a multi-leak day','dash'));
if(D.map.ok.length) tr.push(line(D.map.ok,'#2e7d32',5,'Test pipe (20 %) - single leaks detected'));
if(D.map.miss.length) tr.push(line(D.map.miss,'#d62728',6,'Test pipe - a single leak missed'));
const col={};D.map.train.forEach(p=>col[p]='#4a4f55');
[[D.map.multi,'#1f77b4'],[D.map.ok,'#2e7d32'],[D.map.miss,'#d62728']].forEach(([ps,c])=>ps.forEach(p=>col[p]=c));
const isTest=p=>col[p]!=='#4a4f55';
// pipe names: a little ABOVE the pipe (yshift in pixels, stays above when zooming), on a white background
const ann=Object.keys(col).sort((a,b)=>isTest(a)-isTest(b)).map(p=>{const m=mid(p),t=isTest(p);
 return {x:m[0],y:m[1],text:t?'<b>'+p+'</b>':p,showarrow:false,yshift:t?11:9,font:{size:t?12:9,color:col[p]},
  bgcolor:t?'rgba(255,255,255,0.92)':'rgba(255,255,255,0.75)',bordercolor:t?col[p]:'rgba(0,0,0,0)',borderwidth:t?1:0,borderpad:1}});
const annTest=ann.filter(a=>a.text.startsWith('<b>'));
Plotly.newPlot('map',tr,{title:{text:'<b>Drumchapel network - training pipes (grey) and TEST pipes (colour)</b>',font:{size:14},x:0.01},
 xaxis:{scaleanchor:'y',visible:false},yaxis:{visible:false},
 legend:{x:0.01,y:0.97,xanchor:'left',yanchor:'top',bgcolor:'rgba(255,255,255,0.9)',bordercolor:'#ccc',borderwidth:1,font:{size:12}},
 annotations:ann,
 updatemenus:[{type:'buttons',direction:'right',showactive:true,x:1,xanchor:'right',y:1.0,yanchor:'bottom',pad:{t:2,b:2},font:{size:11},
  buttons:[{label:'All pipe names',method:'relayout',args:[{annotations:ann}]},
           {label:'Test pipe names only',method:'relayout',args:[{annotations:annTest}]},
           {label:'No names',method:'relayout',args:[{annotations:[]}]}]}],
 margin:{l:10,r:10,t:60,b:10},hovermode:'closest',plot_bgcolor:'#fff'},{scrollZoom:true,displaylogo:false});
const pick=document.getElementById('pick');T.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=t.label;pick.appendChild(o)});
pick.onchange=()=>show(+pick.value);
function show(i){const t=T[i];document.querySelectorAll('.click tbody tr').forEach(r=>r.classList.toggle('sel',+r.cells[0].textContent-1==i));pick.value=i;const sh=[];
 t.leaks.forEach(l=>sh.push({type:'rect',xref:'x',yref:'paper',x0:l[0],x1:l[1],y0:0,y1:1,fillcolor:'#d62728',opacity:0.12,line:{width:0}}));
 t.alarms.forEach(a=>sh.push({type:'rect',xref:'x',yref:'paper',x0:a[0],x1:a[1]==null?24:a[1],y0:0.02,y1:0.98,fillcolor:'rgba(0,0,0,0)',line:{color:'#2e7d32',width:3}}));
 t.dets.forEach(m=>sh.push({type:'line',xref:'x',yref:'paper',x0:m[0],x1:m[0],y0:0,y1:1,line:{color:'#000',width:2,dash:'dash'}}));
 Plotly.react('score',[{x:t.t,y:t.score,mode:'lines',name:'Real-time anomaly score',line:{color:'#263238'}},
  {x:[0,24],y:[1,1],mode:'lines',name:'Threshold',line:{dash:'dash',color:'#ff9800'}},
  {x:t.dets.map(m=>m[0]),y:t.dets.map(()=>1),mode:'markers',marker:{symbol:'diamond',size:11,color:'#000'},name:'Leak detected',text:t.dets.map(m=>m[1]),hoverinfo:'text'},
  {x:[null],y:[null],mode:'lines',line:{color:'rgba(214,39,40,0.3)',width:10},name:'Leak on (induced)'},
  {x:[null],y:[null],mode:'lines',line:{color:'#2e7d32',width:3},name:'Leak alarm on'}],
  {title:{text:t.label+(t.info?' - '+t.info:''),font:{size:14}},shapes:sh,xaxis:{title:'Time of day (h)',dtick:2,range:[0,24]},
   yaxis:{title:'score / threshold (log scale)',type:'log'},legend:{orientation:'h',y:-0.2},margin:{l:60,r:10,t:40,b:10}},{displaylogo:false})}
show(0);
</script></body></html>"""
    cards_html = "".join(f'<div class="card">{k}<b>{v}</b></div>' for k, v in cards.items())
    for k, v in (("TITLE", title), ("CARDS", cards_html), ("SINGLE", single_table.to_html(index=False, classes="tbl", border=0, na_rep="")),
                 ("MULTI", multi_table.to_html(index=False, classes="tbl", border=0, na_rep="")), ("DELAY", delay_html),
                 ("TUNING", tuning_html), ("RI", str(READING_INTERVAL_MIN)), ("MR", MULTI_TXT), ("DATA", data), ("PLOTLY", plotly_tag)):
        html = html.replace(f"@@{k}@@", v)
    Path(path).write_text(html, encoding="utf-8")
    return path


# =====================================================================================================
# SETTINGS  (edit here)
# =====================================================================================================
try:
    HERE = Path(__file__).resolve().parent
except NameError:                     # running inside Jupyter
    HERE = Path.cwd()
INPUT_FILE = Path(r"C:\Users\abhis\Desktop\Thesis Work\EPANET Python toolkit\INP file\Drumchapel.inp")
if not INPUT_FILE.exists():
    INPUT_FILE = HERE / "Drumchapel.inp"
OUTPUT_DIR = HERE / "Drumchapel_matrix_profile_leak_outputs"   # all results are written here
OPEN_OUTPUT_FOLDER = True             # open the output folder automatically at the end

# ---- 80 / 20 SPLIT OF THE PIPES (Drumchapel.inp: 154 pipes, 151 open - P132, P151, P152 are closed) ----
TRAIN_FRACTION = 0.80              # 80 % of the open pipes = TRAINING pipes, 20 % = TEST pipes (chosen at random)
SPLIT_SEED = 42                    # a number = always the same random split; None = a new random split on every run
SAVE_RAW_TRAINING_DATA = True      # training_raw_epanet.xlsx: 3 sheets (Pressure, Flow, Velocity), one row per reading of every
                                   # normal and training day: leak pipe, size, type, start / end, demand factor, time, leak on, values

# ---- TRAINING (80 % of the pipes): leaks used ONLY to tune the detector settings, never shown as test results ----
# User-defined leak pipes and leaks (always in the training part):
USER_PIPES = {"P93": {"coefficient": 0.657, "position": 0.46},
                 "P16": {"coefficient": 0.970, "position": 0.66},
                 "P122": {"coefficient": 0.300, "position": 0.85},
                 "P133": {"coefficient": 0.500, "position": 0.50},
                 "P35": {"coefficient": 0.890, "position": 0.10},
                 "P72": {"coefficient": 0.020, "position": 0.72}}
USER_LEAKS = [{"pipe": "P93", "start": "06:00", "end": "08:30"},          # times "HH:MM"; optional "growth_h"
              {"pipe": "P16", "start": "10:00", "end": "11:30"},
              {"pipe": "P122", "start": "22:00", "end": "23:30", "growth_h": 1.0}]
LEAKS_PER_USER_PIPE = 6            # user-defined leaks + extra random leaks on each user-defined pipe
LEAKS_PER_TRAINING_PIPE = 1        # random leaks (random position, time, size) on every OTHER training pipe
USER_MULTI_LEAK_DAYS = [           # user-defined days with several leaks (training only)
    [{"pipe": "P93", "position": 0.46, "coefficient": 0.657, "start": "08:07", "end": "11:07"},
     {"pipe": "P34", "position": 0.50, "coefficient": 0.500, "start": "08:07", "end": "11:07"}],
    [{"pipe": "P93", "position": 0.46, "coefficient": 0.657, "start": "08:07", "end": "11:07"},
     {"pipe": "P34", "position": 0.50, "coefficient": 0.500, "start": "09:12", "end": "12:12"}],
    [{"pipe": "P93", "position": 0.46, "coefficient": 0.657, "start": "08:07", "end": "08:52"},
     {"pipe": "P34", "position": 0.50, "coefficient": 0.500, "start": "09:04", "end": "12:04"}],
]
N_TRAINING_MULTI_LEAK_DAYS = 20    # + random days with 2-5 leaks on random training pipes
N_TRAINING_NO_LEAK_DAYS = 10       # leak-free days used to check that the tuned settings give no false detections

# ---- TEST (20 % of the pipes, never used in training): positions, times, sizes, sudden / growing all random ----
LEAKS_PER_TEST_PIPE = 2            # days with ONE leak: 2 per test pipe (shown in random order)
N_TEST_MULTI_LEAK_DAYS = 30        # days with SEVERAL leaks at the same time
LEAKS_PER_MULTI_DAY = (2, 3, 4, 5) #   number of leaks per day: drawn AT RANDOM for every day from these numbers
                                   #   (so some days 2, others 3, 4 or 5), each leak on a different random TEST pipe
MULTI_TXT = f"{min(LEAKS_PER_MULTI_DAY)}-{max(LEAKS_PER_MULTI_DAY)}"   # text "2-5" used in tables and the report
MULTI_START_SPREAD_H = 4.0         #   all leaks of a day start within 4 h -> they overlap
N_NO_LEAK_DAYS = 3                 # test days WITHOUT any leak -> no false detection
DURATIONS_H = (0.5, 1.0, 2.0, 3.0, 4.0)
COEF_RANGE = (0.05, 1.0)           # emitter coefficient of single test leaks (log-uniform: small AND large leaks)
MULTI_COEF_RANGE = (0.1, 1.0)      # emitter coefficient of the leaks on the multi-leak days
GROWTH_CHOICES_H = (0, 0, 1, 2)    # 0 = SUDDEN leak (burst), 1 / 2 = leak GROWS to full size in 1 / 2 h

# ---- leak-size sweep on random TEST pipes: how fast is a leak of a given size detected? ----
SIZE_SWEEP = True
N_SWEEP_PIPES = 6
SWEEP_COEFS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SWEEP_START_HOURS = (3, 10, 18)    # night / day / evening peak; the minute is random
SWEEP_GROWTH_H = (0, 2)            # sudden and growing (2 h)
SWEEP_DURATION_H = 4.0

# ---- normal behaviour (leak-free days) ----
N_TRAIN_DAYS = 48                  # leak-free days: 8 = "normal" reference of the Matrix Profile, 40 = threshold setting
N_REFERENCE_DAYS = 8

# ---- sensor readings ----
READING_INTERVAL_MIN = 15          # a new sensor reading every 15 min (a leak can only be detected when a reading arrives)
STEP_S = 60 * READING_INTERVAL_MIN # (used by the simulation: EPANET hydraulic + report step in seconds)
RAW_EXCEL_EVERY_MIN = 15           # raw EPANET sheets in Excel: one row every 15 min

# ---- detector (real time) ----
MP_WINDOWS = (1, 2, 4)             # real-time windows in readings: last 15, 30 and 60 min
MP_WINDOW = 4                      # window of the looking-back analysis (1 h)
TOP_K_SENSORS = 4                  # second view: the 4 most abnormal sensors on their own (catches leaks on the main supply
                                   # pipes that change only a few sensors strongly); 0 = all-sensor view only
CONFIRM_READINGS = 1               # leak detected at the first abnormal reading
CLEAR_READINGS = 2                 # alarm switched off after 2 normal readings in a row (so the next leak can be detected)
# settings TUNED AUTOMATICALLY on the training leaks (80 % of the pipes) (every combination below is tried; the best one is used)
AUTO_TUNE = True
TUNE_GRID = dict(factor=(1.3, 1.5, 1.7),      # threshold factor (threshold = factor x highest score on leak-free days)
                 jump=(1.0, 1.5, 2.0),        # new leak: change > jump x largest change between readings on a normal day
                 rel=(0.1, 0.2, 0.3),         #   ... and > rel x the current anomaly
                 swap=(0.3, 0.5),             # one leak stops + a new one starts: pattern similarity below this
                 grace=(0, 1, 2))             # readings after a detection before the next new-leak check
DEFAULT_SETTINGS = dict(factor=1.5, jump=1.5, rel=0.3, swap=0.3, grace=2)   # used when AUTO_TUNE = False
# plain names of the 5 settings (used in the HTML report, Excel files and console)
SETTING_NAMES = {
    "factor": ("Alarm threshold factor",
               "How far above normal the anomaly score must go before a leak is reported: threshold = this number x the highest "
               "score seen on the leak-free days (1.7 = 70 % above the most unusual normal moment). Higher = fewer false alarms, "
               "but small leaks are found later."),
    "jump": ("New-leak jump size",
             "While an alarm is on: how big a sudden change between two readings must be to count as a NEW leak, "
             "compared with the biggest change between two readings on a normal day (1.0 = at least as big)."),
    "rel": ("New-leak jump share",
            "The sudden change must also be at least this share of the current leak signal (0.3 = 30 %), so that the normal "
            "wobble of a big leak is not taken for a new leak."),
    "swap": ("Pattern-change similarity",
             "If the sensor pattern after a sudden drop looks less similar than this (0 = different, 1 = identical) to the "
             "pattern before, one leak stopped and a NEW one started at the same time (so the new leak is not missed)."),
    "grace": ("Wait after an event (readings)",
              "Number of readings to wait after a leak was detected before looking for the next new leak "
              "(0 = check again at the very next reading)."),
}
PRESSURE_NOISE = 0.01              # pressure sensor noise (m)
DEMAND_NOISE = 0.02                # unknown random demand fluctuation (2 %)
SEED = 42

README = [
    ("HOW THE DATA IS MADE", "", "EPANET (through EPyT) loads Drumchapel.inp and runs a 24-h hydraulic simulation. At every reading "
                                 f"(every {READING_INTERVAL_MIN} min) the script reads pressure at each junction and flow + velocity in each pipe = the raw EPANET data."),
    ("", "", "Each day has a random demand fluctuation (2 %). Leak days have one or more leaks (emitter at the leak point) between start "
             "and end; a SUDDEN leak has full size at once, a GROWING leak reaches full size within 1 or 2 h."),
    ("THREE DATA SETS", "Normal days", f"{N_TRAIN_DAYS} leak-free days: what normal operation looks like (Matrix-Profile reference) and the thresholds."),
    ("80 / 20 SPLIT", "Pipe_split", f"The 151 open pipes are split at random: {TRAIN_FRACTION:.0%} TRAINING pipes and {1 - TRAIN_FRACTION:.0%} "
                                    "TEST pipes (the user-defined leak pipes are always training pipes). No test leak is on a training pipe."),
    ("", "Training leaks", f"Leaks on the TRAINING pipes: the user-defined leaks and multi-leak days, {LEAKS_PER_USER_PIPE} leaks on each user-defined pipe, "
                           f"{LEAKS_PER_TRAINING_PIPE} random leak on every other training pipe and {N_TRAINING_MULTI_LEAK_DAYS} random days with "
                           f"{MULTI_TXT} leaks. Used ONLY to tune the detector settings (the leak patterns "
                           "the Matrix-Profile detector has to react to); never part of the test."),
    ("", "Test leaks", f"UNSEEN, only on the TEST pipes: {LEAKS_PER_TEST_PIPE} days with one leak per test pipe and {N_TEST_MULTI_LEAK_DAYS} "
                       f"days with {MULTI_TXT} overlapping leaks, random positions, times and sizes, "
                       f"+ {N_NO_LEAK_DAYS} leak-free days."),
    ("HOW THE DETECTOR WORKS", "Real time", "At every new reading only the data up to that moment is used. The last "
                                            f"{', '.join(str(READING_INTERVAL_MIN * w) for w in MP_WINDOWS)} min of sensor changes are "
                                            "compared with the most similar piece of the leak-free days (Matrix Profile). Score above "
                                            "the threshold = LEAK DETECTED."),
    ("", "Two views", "(1) all sensors together (multivariate Matrix Profile) and (2) the {TOP_K_SENSORS} most abnormal sensors on "
                      "their own (k-dimensional Matrix Profile idea) - a leak on a main supply pipe changes only a few sensors strongly "
                      "and would be diluted in view (1). Leak detected when either view is above its threshold."),
    ("", "New leak", "While an alarm is on, a sudden change of the sensor pattern in a NEW direction = another leak started."),
    ("", "Only the start is reported", "The detector reports WHEN a leak is detected. When a leak will be repaired is not known in "
                                       "real operation, so no repair time is reported. (Internally the alarm is switched off when the "
                                       "sensors are normal again, so that the next leak can be detected.)"),
    ("", "Tuning", "Threshold factor and new-leak settings are chosen automatically: every combination is tried on the training "
                   "leaks; the best (most leaks found, fewest false detections, no false detection on leak-free days) is used."),
    ("training_data.xlsx", "Tuning / Chosen_settings", "All tried setting combinations with their score on the training leaks; the chosen one. "
                                                       "The 5 settings: " + "; ".join(f"{v[0]} = {v[1]}" for v in SETTING_NAMES.values())),
    ("HOW THE TRAINING DATA IS GENERATED", "1. Pipe split", "The open pipes are split at random into 80 % training pipes and 20 % test pipes."),
    ("", "2. Training days", "Every training day is one 24-h EPANET simulation (EPyT) with a sensor reading every 15 min and a random "
                             "2 % demand fluctuation (Demand factor). The 48 normal days differ from each other by this random demand "
                             "fluctuation and the sensor noise: 8 show what normal operation looks like, 40 set the alarm threshold."),
    ("", "3. Leaks", "A leak = an emitter (leak flow = coefficient x sqrt(pressure)) at the leak position, switched on at the leak start "
                     "and off at the leak end; a growing leak reaches its full size within 1-2 h. Sources: 'User-defined' = pipe, position, "
                     "size and time set in USER_PIPES / USER_LEAKS / USER_MULTI_LEAK_DAYS; 'Random' = random training pipe, position, "
                     "time, size, sudden or growing. Plus random days with 2-5 overlapping leaks and leak-free days."),
    ("", "4. Leak on", "Column 'Leak on' in training_raw_epanet.xlsx: 1 for every reading while a leak is active (leak start <= time < leak "
                       "end), 0 before the leak starts and after it ends. Example: leak 06:00-08:30 -> 0 until 05:45, 1 from 06:00 "
                       "to 08:15, 0 from 08:30."),
    ("", "5. Use", "The training days are only used to choose the detector settings (Tuning sheet); the detector never sees the test pipes."),
    ("", "Training_leaks", "Every training day and leak (Simulation number as in training_raw_epanet.xlsx): leak pipe, size, type, "
                           "leak start / end and when it was detected with the chosen settings."),
    ("training_raw_epanet.xlsx", "Pressure / Flow / Velocity", "Raw EPANET data of every simulated training day (48 normal days + all "
                                                                "training days), one row per 15-min reading. Columns: Simulation (day number), "
                                                                "Day (No leak / Leak day (n leaks)), Leak pipe, Emitter coef, Leak type, Leak "
                                                                "start, Leak end (several leaks separated by ';'), Demand factor (random demand "
                                                                "fluctuation at that reading), Time, Leak on (1 = a leak is active at that "
                                                                "reading, 0 = no leak), then the pressure at every junction / flow and velocity "
                                                                "in every pipe."),
    ("simulated_data.xlsx", "", "RAW EPANET output + sensor changes of every TEST day; 'Leak on' = 1 while a leak is active."),
    ("detection_results.xlsx", "Single_leak_results", "One row per test day with one leak: pipe, position, size, sudden / growing, leak time, "
                                                      "leak detected at, detection delay, result."),
    ("", "Multi_leak_results", "One row per leak on the multi-leak days (2, 3, 4 or 5 leaks at the same time): the same, plus how the "
                               "leak was detected."),
    ("", "Detection_summary / Detection_delay", "How many leaks were found, how many false detections, how fast the leaks were detected."),
    ("", "Delay_vs_leak_size / Leak_size_sweep_all", "Leak-size sweep on random test pipes: detection rate and delay vs leak size."),
    ("", "Confusion_matrix / Detection_timeline", "TP / FP / FN / TN; every reading of every test day: leak on, score, leak alarm on."),
    ("TERMS", "Emitter coef", "Leak size. Leak flow = coef x sqrt(pressure at the leak), e.g. 0.3 x sqrt(30 m) = 1.6 L/s."),
    ("", "Position", "Where on the pipe the leak is: 0 = start node, 1 = end node (as in the INP file)."),
    ("", "Detection delay", "Time the leak was detected - real start. A leak can only be detected when a new reading arrives."),
    ("", "False detection", "A leak reported when no (new) leak had started."),
    ("", "TP / FN / FP / TN", "Leak detected / leak missed / leak reported without a leak / no leak and nothing reported."),
]


# =====================================================================================================
# DATA SETS: 80 / 20 PIPE SPLIT -> TRAINING leaks (training pipes) and TEST leaks (test pipes only)
# =====================================================================================================
def leak_type(g):
    return "sudden" if not g else f"grows in {g:g} h"


def random_leak(rng, lo=0.0, hi=24.0, coef_range=None, durations=None):
    """Random duration, start minute (ending by 24:00), size (log-uniform) and sudden / growing."""
    d = float(rng.choice(durations or DURATIONS_H))
    on = min(np.floor(rng.uniform(max(lo, 0.25), max(lo, min(hi, 24 - d), 0.25)) * 60) / 60, 24 - d)
    return dict(start=hhmm(on), end=hhmm(on + d), coefficient=round(float(np.exp(rng.uniform(*np.log(coef_range or COEF_RANGE)))), 3),
                growth_h=float(min(rng.choice(GROWTH_CHOICES_H), d)))


def leak_counts(rng, n_days):
    """Number of leaks of every multi-leak day: drawn at random (independently for every day) from LEAKS_PER_MULTI_DAY."""
    return [int(k) for k in rng.choice(LEAKS_PER_MULTI_DAY, n_days)]


def multi_leak_day(rng, pipes, pos, k):
    """k leaks on different pipes, all starting within MULTI_START_SPREAD_H hours -> they overlap."""
    t0 = float(rng.uniform(0.5, 24 - MULTI_START_SPREAD_H - 4.5))
    out = []
    for p in rng.choice(pipes, min(k, len(pipes)), replace=False):
        d = float(rng.choice((1.0, 2.0, 3.0, 4.0))); on = np.floor((t0 + rng.uniform(0, MULTI_START_SPREAD_H)) * 60) / 60
        out.append(dict(pipe=str(p), position=pos(str(p)), start=hhmm(on), end=hhmm(min(on + d, 23.75)),
                        coefficient=round(float(np.exp(rng.uniform(*np.log(MULTI_COEF_RANGE)))), 3), growth_h=float(rng.choice((0, 0, 0, 1)))))
    return sorted(out, key=lambda l: hours(l["start"]))


def split_pipes(net):
    """80 / 20 split of the open pipes at random. User-defined leak pipes are always training pipes. Returns (train, test, seed)."""
    mine = set(USER_PIPES) | {l["pipe"] for l in USER_LEAKS} | {l["pipe"] for ls in USER_MULTI_LEAK_DAYS for l in ls}
    for p in mine:
        if p not in net.pipes: raise ValueError(f"{p} is not an open pipe in {INPUT_FILE.name}")
    seed = SPLIT_SEED if SPLIT_SEED is not None else int(np.random.SeedSequence().entropy % 1_000_000)
    rng = np.random.default_rng(seed); n_test = int(round((1 - TRAIN_FRACTION) * len(net.pipes)))
    test = {str(p) for p in rng.choice([p for p in net.pipes if p not in mine], n_test, replace=False)}
    return [p for p in net.pipes if p not in test], [p for p in net.pipes if p in test], seed


def build_training(net, train_pipes):
    rng = np.random.default_rng(SEED + 1); days = []
    rand_pos = lambda p: USER_PIPES[p]["position"] if p in USER_PIPES else round(float(rng.uniform(0.05, 0.95)), 2)
    for p, cfg in USER_PIPES.items():
        mine = [dict(l, coefficient=cfg["coefficient"]) for l in USER_LEAKS if l["pipe"] == p]
        n_extra = max(LEAKS_PER_USER_PIPE - len(mine), 0); block = 24 / max(n_extra, 1)
        for k, ev in enumerate(mine + [dict(random_leak(rng, k * block, (k + 1) * block), pipe=p) for k in range(n_extra)], 1):
            days.append(dict(kind="single", source="User-defined" if k <= len(mine) else "Random (on a user-defined pipe)",
                             leaks=[dict(ev, position=cfg["position"])]))
    for k, ls in enumerate(USER_MULTI_LEAK_DAYS, 1):
        days.append(dict(kind="multi", source="User-defined", leaks=[dict(l) for l in ls]))
    for p in train_pipes:                                        # every other training pipe: random leak(s)
        if p in USER_PIPES: continue
        for k in range(LEAKS_PER_TRAINING_PIPE):
            days.append(dict(kind="single", source="Random", leaks=[dict(random_leak(rng), pipe=p, position=rand_pos(p))]))
    for k, c in enumerate(leak_counts(rng, N_TRAINING_MULTI_LEAK_DAYS)):
        days.append(dict(kind="multi", source="Random", leaks=multi_leak_day(rng, train_pipes, rand_pos, c)))
    days += [dict(kind="none", source="-", leaks=[]) for k in range(N_TRAINING_NO_LEAK_DAYS)]
    for k, d in enumerate(days, 1):                              # professional day ID: TR-001 | pipe | leak time
        d["id"] = f"TR-{k:03d}"
        if d["kind"] == "single":
            l = d["leaks"][0]; d["name"] = f"TR-{k:03d} | {l['pipe']} | leak {l['start']}-{l['end']}"
        elif d["kind"] == "multi":
            d["name"] = f"TR-{k:03d} | {len(d['leaks'])} leaks (" + ", ".join(l["pipe"] for l in d["leaks"]) + ")"
        else:
            d["name"] = f"TR-{k:03d} | no leak"
    return days


def build_tests(net, test_pipes):
    """UNSEEN test days, ONLY on the test pipes: single leaks (random order), 2-5 leaks at once, a few leak-free days."""
    rng = np.random.default_rng(SEED + 7); days = []
    single = [p for p in test_pipes for _ in range(LEAKS_PER_TEST_PIPE)]
    for k, p in enumerate(rng.permutation(single), 1):          # random order, not grouped by pipe
        days.append(dict(kind="single", name=f"Test leak {k} ({p})",
                         leaks=[dict(random_leak(rng), pipe=str(p), position=round(float(rng.uniform(0.05, 0.95)), 2))]))
    for k, c in enumerate(leak_counts(rng, N_TEST_MULTI_LEAK_DAYS)):
        days.append(dict(kind="multi", name=f"Multi-leak day {k + 1} ({c} leaks)",
                         leaks=multi_leak_day(rng, test_pipes, lambda p: round(float(rng.uniform(0.05, 0.95)), 2), c)))
    days += [dict(kind="none", name=f"No-leak day {k + 1}", leaks=[]) for k in range(N_NO_LEAK_DAYS)]
    return days


def build_sweep(net, test_pipes):
    rng = np.random.default_rng(SEED + 3); out = []
    for p in rng.choice(test_pipes, min(N_SWEEP_PIPES, len(test_pipes)), replace=False):
        pos = round(float(rng.uniform(0.1, 0.9)), 2)
        for c in SWEEP_COEFS:
            for hr in SWEEP_START_HOURS:
                for g in SWEEP_GROWTH_H:
                    s = hhmm(hr + int(rng.integers(0, 60)) / 60)
                    out.append(dict(kind="sweep", name=f"{p} coef {c} {leak_type(g)} {s}", leaks=[dict(
                        pipe=str(p), position=pos, coefficient=c, start=s, end=hhmm(hours(s) + SWEEP_DURATION_H), growth_h=float(g))]))
    return out


# =====================================================================================================
# EVALUATION
# =====================================================================================================
def simulate_day(net, rng, leaks):
    dn = demand_noise(rng, 24.0, DEMAND_NOISE)
    if not leaks:
        h = net.open(); obs = net.run(h, 24.0, dnoise=dn); net.close(h)
    else:
        obs = net.simulate_leaks([dict(pipe=l["pipe"], position=l["position"], coefficient=l["coefficient"], on=hours(l["start"]),
                                       off=hours(l["end"]), growth_h=l.get("growth_h", 0.0)) for l in leaks], 24.0, 1.0, dn)
    obs["dn"] = dn
    return obs


def raw_tables(net, sim_no, leaks, obs, leak_on):
    """One simulated day -> rows of the Pressure, Flow and Velocity tables (same leading columns in all three)."""
    n = len(obs["t"]); j = lambda f: "; ".join(str(l[f]) for l in leaks) if leaks else "-"
    dn = obs.get("dn"); dn = np.ones(n) if dn is None else np.resize(np.asarray(dn, float), n)
    info = pd.DataFrame({"Simulation": sim_no,
                         "Day": "No leak" if not leaks else f"Leak day ({len(leaks)} leak" + ("s)" if len(leaks) > 1 else ")"),
                         "Leak pipe": j("pipe"), "Emitter coef": j("coefficient"),
                         "Leak type": "; ".join(leak_type(l.get("growth_h", 0)) for l in leaks) if leaks else "-",
                         "Leak start": j("start"), "Leak end": j("end"), "Demand factor": np.round(dn, 4),
                         "Time": [hhmm(x) for x in obs["t"]], "Leak on": np.asarray(leak_on, int)}, index=range(n))
    pu, qu, vu = net.units
    return (pd.concat([info, pd.DataFrame(obs["P"].round(4), columns=[f"{x} [{pu}]" for x in net.junctions])], axis=1),
            pd.concat([info, pd.DataFrame(obs["Q"].round(4), columns=[f"{x} [{qu}]" for x in net.pipes_all])], axis=1),
            pd.concat([info, pd.DataFrame(obs["V"].round(4), columns=[f"{x} [{vu}]" for x in net.pipes_all])], axis=1))


HOW = {"new alarm": "new alarm", "new leak while an alarm was active": "NEW leak recognised during an active alarm",
       "new leak right after a repair": "NEW leak recognised (sensor pattern changed while the alarm was on)",
       "together": "detected together with another leak (started between the same two readings)"}


def analyse_day(det, d, s):
    """Real-time monitoring of one day with settings s; every leak the detector found is matched with the
    induced leak that started just before it.  Returns per-leak results, false detections and detected windows."""
    obs, leaks = d["obs"], d["leaks"]; t = obs["t"]
    comb, alarms = det.monitor(t, d["R"], d["scores"], s["factor"], CONFIRM_READINGS, CLEAR_READINGS, s["jump"], s["rel"],
                               s["swap"], int(s["grace"]))
    W, rph = max(det.windows), 3600 / STEP_S
    span = [(int(np.ceil(hours(l["start"]) * rph - 1e-9)), int(np.ceil(hours(l["end"]) * rph - 1e-9))) for l in leaks]
    found = sorted([L for a in alarms for L in a["leaks"]], key=lambda L: L["i_det"])
    got, false = [None] * len(leaks), []
    for L in found:
        cand = [n for n, (a, b) in enumerate(span) if got[n] is None and a <= L["i_det"] <= b + W - 1]
        if not cand: false.append(L); continue
        if L["how"] != "new alarm":                       # a new pattern belongs to the leak(s) that started last
            m = max(span[n][0] for n in cand); cand = [n for n in cand if span[n][0] == m]
        for k, n in enumerate(sorted(cand, key=lambda n: span[n][0])): got[n] = (L, k > 0)
    per = []
    for n, l in enumerate(leaks):
        if got[n] is None:
            per.append(dict(status="missed", hidden=any(a["i0"] <= span[n][0] < a["i1"] for a in alarms))); continue
        L, together = got[n]; ie = L["i_end"] if L["i_end"] is not None else len(t)
        end_h = float(t[ie]) if ie < len(t) else 24.0
        per.append(dict(status="detected", time=float(t[L["i_det"]]), delay=round((t[L["i_det"]] - hours(l["start"])) * 60),
                        how="together" if together else L["how"], end_h=end_h, end_err=round((end_h - hours(l["end"])) * 60),
                        flow=float(obs["L"][L["i_det"]])))
    win = np.zeros(len(t), bool)
    for L in found: win[L["i_det"]:(L["i_end"] if L["i_end"] is not None else len(t))] = True
    return dict(comb=comb, alarms=alarms, per=per, false=[float(t[L["i_det"]]) for L in false], found=found, win=win)


def score_settings(det, days, s):
    """Score of one setting combination on a set of days (used for tuning)."""
    n = hit = fp = fp_free = 0; delays, ends = [], []
    for d in days:
        r = analyse_day(det, d, s); fp += len(r["false"]); fp_free += len(r["false"]) if d["kind"] == "none" else 0
        for p in r["per"]:
            n += 1
            if p["status"] == "detected": hit += 1; delays.append(p["delay"])
    prec = hit / (hit + fp) if hit + fp else 0.0; rec = hit / n if n else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(**s, **{"Leaks found": f"{hit}/{n}", "False detections": fp, "False on leak-free days": fp_free,
                        "Precision": round(prec, 3), "Recall": round(rec, 3), "F1": round(f1, 3),
                        "Median delay (min)": float(np.median(delays)) if delays else None,
                        "Score": round(f1 - (1.0 if fp_free else 0.0), 4)})


def tune(det, days):
    import itertools
    keys = list(TUNE_GRID); rows = [score_settings(det, days, dict(zip(keys, v))) for v in itertools.product(*TUNE_GRID.values())]
    tab = pd.DataFrame(rows).sort_values(["Score", "Median delay (min)"], ascending=[False, True]).reset_index(drop=True)
    best = {k: (int(tab.loc[0, k]) if k == "grace" else float(tab.loc[0, k])) for k in keys}
    return best, tab


def delay_summary(df):
    """How fast: rows = groups of leaks, columns = detection share within 5 ... 60 min, median / 90 % delay."""
    groups = [("All single leaks", df), ("Sudden leaks", df[df.Growth == 0]), ("Growing leaks", df[df.Growth > 0]),
              ("Small (coef < 0.1)", df[df["Emitter coef"] < 0.1]),
              ("Medium (0.1 - 0.3)", df[(df["Emitter coef"] >= 0.1) & (df["Emitter coef"] < 0.3)]),
              ("Large (coef >= 0.3)", df[df["Emitter coef"] >= 0.3])]
    out = []
    for name, g in groups:
        if not len(g): continue
        d = g.Delay.dropna()
        out.append({"Leaks": name, "Number": len(g), "Detected": f"{len(d)}/{len(g)}",
                    "Within 5 min": f"{(g.Delay <= 5).mean():.0%}", "Within 10 min": f"{(g.Delay <= 10).mean():.0%}",
                    "Within 15 min": f"{(g.Delay <= 15).mean():.0%}", "Within 30 min": f"{(g.Delay <= 30).mean():.0%}",
                    "Within 60 min": f"{(g.Delay <= 60).mean():.0%}",
                    "Median delay (min)": round(float(d.median()), 1) if len(d) else None,
                    "90 % detected within (min)": round(float(d.quantile(0.9)), 1) if len(d) else None,
                    "Longest delay (min)": round(float(d.max()), 1) if len(d) else None})
    return pd.DataFrame(out)


def window_summary(name, per, n_false, n_days):
    det_ = [p for p in per if p["status"] == "detected"]; dl = pd.Series([p["delay"] for p in det_], dtype=float)
    return {"Test days": name, "Days": n_days, "Leaks": len(per), "Leaks detected": f"{len(det_)}/{len(per)}",
            "Detected (%)": f"{len(det_) / max(len(per), 1):.0%}", "False detections": n_false,
            "Median detection delay (min)": round(float(dl.median()), 1) if len(dl) else None,
            "Detected within 15 min": f"{(dl <= 15).sum() / max(len(per), 1):.0%}",
            "Detected within 30 min": f"{(dl <= 30).sum() / max(len(per), 1):.0%}"}


def main():
    print("=" * 80 + "\nDRUMCHAPEL - REAL-TIME LEAK DETECTION WITH THE MATRIX PROFILE\n" + "=" * 80)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved in:\n  {OUTPUT_DIR.resolve()}\n")
    net, rng = Network(INPUT_FILE), np.random.default_rng(SEED)
    qu = net.units[1]
    train_pipes, test_pipes, split_seed = split_pipes(net)
    training = build_training(net, train_pipes)
    assert not {l["pipe"] for d in training for l in d["leaks"]} & set(test_pipes)
    tests = build_tests(net, test_pipes); sweep = build_sweep(net, test_pipes) if SIZE_SWEEP else []
    n_ts, n_tm = sum(d["kind"] == "single" for d in tests), sum(d["kind"] == "multi" for d in tests)
    print(f"Network      : {len(net.junctions)} junctions, {len(net.pipes)} open pipes")
    print(f"Sensors      : " + ", ".join(f"{len(names)} x {SENSOR_TEXT[f]}" for f, _, names in net.blocks()) +
          "  (head = pressure + elevation)")
    print(f"Readings     : every {READING_INTERVAL_MIN} min; leaks can start at any minute")
    print(f"Normal days  : {N_TRAIN_DAYS} leak-free days ({N_REFERENCE_DAYS} = Matrix-Profile reference, the rest = thresholds)")
    print(f"Pipe split   : {len(net.pipes)} open pipes -> {len(train_pipes)} TRAINING pipes ({TRAIN_FRACTION:.0%}) + {len(test_pipes)} "
          f"TEST pipes ({1 - TRAIN_FRACTION:.0%}), random split seed {split_seed}")
    print(f"Test pipes   : {', '.join(test_pipes)}")
    print(f"Training     : {sum(len(d['leaks']) for d in training)} leaks on {len(training)} days, only on training pipes "
          f"(user-defined + random leaks) -> used to tune the detector settings")
    print(f"Test (unseen): {n_ts} days with one leak + {n_tm} days with {MULTI_TXT} leaks, ONLY on the "
          f"test pipes + {N_NO_LEAK_DAYS} leak-free days")
    # ---- normal behaviour: leak-free days ----
    h, train_obs, train_dn = net.open(), [], []
    for _ in range(N_TRAIN_DAYS):
        train_dn.append(demand_noise(rng, 24.0, DEMAND_NOISE)); train_obs.append(net.run(h, 24.0, dnoise=train_dn[-1]))
        train_obs[-1]["dn"] = train_dn[-1]
    net.close(h)
    train_R = [residual(net, o, net.expected(o), rng, PRESSURE_NOISE) for o in train_obs]
    ref, nul = train_R[:N_REFERENCE_DAYS], train_R[N_REFERENCE_DAYS:]
    det = (MatrixProfileDetector(sensor_scale(net, ref), MP_WINDOW, DEFAULT_SETTINGS["factor"]).fit(ref, nul)
           .fit_online(nul, MP_WINDOWS, TOP_K_SENSORS, ref).fit_new_leak(nul))

    def simulate(days, label):
        print(f"  simulating {len(days)} {label} ...")
        for d in days:
            d["obs"] = simulate_day(net, rng, d["leaks"]); d["R"] = residual(net, d["obs"], net.expected(d["obs"]), rng, PRESSURE_NOISE)
            d["scores"] = det.online_scores(d["R"])
    print(f"\n[1] Training leaks ({len(train_pipes)} training pipes)"); simulate(training, "training days")
    if AUTO_TUNE:
        best, tuning = tune(det, training)
        print(f"    tried {len(tuning)} setting combinations on the training leaks -> best: " +
              ", ".join(f"{SETTING_NAMES[k][0]} = {v}" for k, v in best.items()))
    else:
        best = dict(DEFAULT_SETTINGS); tuning = pd.DataFrame([score_settings(det, training, best)])
    det.factor = best["factor"]
    print("\n[2] Unseen test days"); simulate(tests, "test days"); simulate(sweep, "leak-size sweep days")

    # ---------------- training leaks with the chosen settings ----------------
    tr_rows = []
    for k, d in enumerate(training, N_TRAIN_DAYS + 1):        # Simulation number = same as in training_raw_epanet.xlsx
        r = analyse_day(det, d, best)
        day = "No leak" if not d["leaks"] else f"Leak day ({len(d['leaks'])} leak" + ("s)" if len(d["leaks"]) > 1 else ")")
        for j, (l, p) in enumerate(zip(d["leaks"], r["per"]), 1):
            tr_rows.append({"Simulation": k, "Day": day, "Leak pipe": l["pipe"], "Emitter coef": l["coefficient"],
                            "Leak type": leak_type(l.get("growth_h", 0)), "Leak start": l["start"], "Leak end": l["end"],
                            "Leak detected at": hhmm(p["time"]) if p["status"] == "detected" else "not detected",
                            "Detection delay (min)": p.get("delay")})
        if d["kind"] == "none":
            tr_rows.append({"Simulation": k, "Day": day,
                            "Leak detected at": ", ".join(f"false detection {hhmm(x)}" for x in r["false"]) or "no leak reported (correct)"})
    training_table = pd.DataFrame(tr_rows)

    # ---------------- evaluation of the unseen test days ----------------
    single_rows, multi_rows, js, steps, single, timeline = [], [], [], [], [], {}
    for n, d in enumerate(tests, 1):
        obs, t, leaks = d["obs"], d["obs"]["t"], d["leaks"]
        r = analyse_day(det, d, best); d.update(n=n, **r)
        actual = np.zeros(len(t), bool)
        for l in leaks: actual |= (t >= hours(l["start"])) & (t < hours(l["end"]))
        d["actual"] = actual; steps.append((actual, r["win"]))
        fa_txt = ", ".join(hhmm(x) for x in r["false"])
        if d["kind"] == "single":
            l, p = leaks[0], r["per"][0]; full = float(obs["L"][actual].max()) if actual.any() else 0.0
            dem = float(obs["D"][actual].mean()) if actual.any() else 1.0; ok = p["status"] == "detected"
            single_rows.append({"No.": n, "Test": d["name"], "Leak pipe": l["pipe"], "Position": l["position"], "Emitter coef": l["coefficient"],
                                "Leak type": leak_type(l.get("growth_h", 0)), "Leak active (induced)": f"{l['start']} - {l['end']}",
                                "Leak detected at": hhmm(p["time"]) if ok else "-", "Detection delay (min)": p.get("delay"),
                                f"Leak flow at detection [{qu}]": round(p["flow"], 3) if ok else None,
                                "Leak size (% of demand)": round(100 * full / max(dem, 1e-9), 1),
                                "Result": ("DETECTED" if ok else "MISSED") + (f" + FALSE DETECTION {fa_txt}" if fa_txt else "")})
            single.append({"Emitter coef": l["coefficient"], "Growth": l.get("growth_h", 0), "Delay": p["delay"] if ok else np.nan})
        elif d["kind"] == "multi":
            for j, (l, p) in enumerate(zip(leaks, r["per"]), 1):
                ok = p["status"] == "detected"
                how = HOW[p["how"]] if ok else ("MISSED - hidden by the other leaks" if p["hidden"] else "MISSED - too small to detect")
                multi_rows.append({"No.": n, "Day": d["name"] if j == 1 else "", "Leak": j, "Leak pipe": l["pipe"], "Position": l["position"],
                                   "Emitter coef": l["coefficient"], "Leak active (induced)": f"{l['start']} - {l['end']}",
                                   "Leak detected": "YES" if ok else "NO", "Leak detected at": hhmm(p["time"]) if ok else "-",
                                   "Detection delay (min)": p.get("delay"),
                                   "False detections (day)": (fa_txt or "none") if j == 1 else "", "How it was detected": how})
        else:
            single_rows.append({"No.": n, "Test": d["name"], "Leak pipe": "none", "Leak detected at": fa_txt or "-",
                                "Result": "NO LEAK REPORTED (correct)" if not fa_txt else f"FALSE DETECTION {fa_txt}"})
        tag = d["name"]
        timeline[f"{tag} - Leak on"] = actual.astype(int); timeline[f"{tag} - Score"] = np.round(r["comb"], 3)
        timeline[f"{tag} - Leak alarm on"] = r["win"].astype(int)
        dets = [[p["time"], f"leak {k + 1} ({leaks[k]['pipe']}) detected {hhmm(p['time'])}, +{p['delay']} min"] for k, p in enumerate(r["per"])
                if p["status"] == "detected"] + [[x, f"false detection {hhmm(x)}"] for x in r["false"]]
        found_n = sum(p["status"] == "detected" for p in r["per"])
        info = (f"{found_n}/{len(leaks)} leaks detected" if leaks else "no leak") + (f", false detection {fa_txt}" if fa_txt else "")
        js.append(dict(label=f"{n}. {d['name']}", info=info, t=t, score=np.round(np.maximum(r["comb"], 1e-3), 3),
                       leaks=[[hours(l["start"]), hours(l["end"])] for l in leaks],
                       alarms=[[a["raise_h"], a["clear_h"]] for a in r["alarms"]], dets=dets))
    single_table, multi_table = pd.DataFrame(single_rows), pd.DataFrame(multi_rows)

    # ---------------- leak-size sweep ----------------
    sw_rows = []
    for d in sweep:
        l = d["leaks"][0]; p = analyse_day(det, d, best)["per"][0]
        t = d["obs"]["t"]; act = (t >= hours(l["start"])) & (t < hours(l["end"]))
        full = float(d["obs"]["L"][act].max()); dem = float(d["obs"]["D"][act].mean())
        sw_rows.append({"Pipe": l["pipe"], "Emitter coef": l["coefficient"], "Leak type": "sudden" if not l["growth_h"] else "growing",
                        "Growth (h)": l["growth_h"], "Leak start": l["start"], "Start period": f"{l['start'][:2]} h",
                        "Detected": p["status"] == "detected", "Delay": p["delay"] if p["status"] == "detected" else np.nan,
                        "Leak size (% of demand)": round(100 * full / dem, 1)})
    sweep_df = pd.DataFrame(sw_rows)

    # ---------------- metrics ----------------
    sd = [d for d in tests if d["kind"] == "single"]; md = [d for d in tests if d["kind"] == "multi"]; nd = [d for d in tests if d["kind"] == "none"]
    single = pd.DataFrame(single)
    tp = sum(d["per"][0]["status"] == "detected" for d in sd); fp = sum(bool(d["false"]) for d in nd)
    cm_event = metrics(tp, fp, len(sd) - tp, len(nd) - fp)
    a = np.concatenate([s_[0] for s_ in steps]); p_ = np.concatenate([s_[1] for s_ in steps])
    cm_step = metrics(int((a & p_).sum()), int((~a & p_).sum()), int((a & ~p_).sum()), int((~a & ~p_).sum()))
    cms = {"Per leak event (days)": cm_event, "Per sensor reading": cm_step}
    wsum = pd.DataFrame([window_summary("days with one leak", [d["per"][0] for d in sd], sum(len(d["false"]) for d in sd), len(sd)),
                         window_summary(f"days with {MULTI_TXT} leaks (all)", [p for d in md for p in d["per"]],
                                        sum(len(d["false"]) for d in md), len(md))] +
                        [window_summary(f"   - days with {c} leaks", [p for d in md if len(d["leaks"]) == c for p in d["per"]],
                                        sum(len(d["false"]) for d in md if len(d["leaks"]) == c), sum(len(d["leaks"]) == c for d in md))
                         for c in sorted({len(d["leaks"]) for d in md})] + [
                         {"Test days": "leak-free days", "Days": len(nd), "Leaks": 0, "False detections": sum(len(d["false"]) for d in nd)}])
    dsum = delay_summary(single)
    sweep_pivot = pd.DataFrame()
    if len(sweep_df):
        sweep_pivot = sweep_df.groupby(["Emitter coef", "Leak type"]).agg(
            **{"Leak size (% of demand)": ("Leak size (% of demand)", "mean"), "Detected": ("Detected", "mean"),
               "Median delay (min)": ("Delay", "median"), "Longest delay (min)": ("Delay", "max")}).reset_index()
        sweep_pivot["Detected"] = (100 * sweep_pivot["Detected"]).round(0).astype(int).astype(str) + " %"
        sweep_pivot["Leak size (% of demand)"] = sweep_pivot["Leak size (% of demand)"].round(1)
    cm_table = pd.DataFrame([{"Level": k, **{m: (round(v, 3) if isinstance(v, float) else v) for m, v in d.items()}} for k, d in cms.items()])
    mp = [p for d in md for p in d["per"]]; m_ok = sum(p["status"] == "detected" for p in mp)
    all_det = [p for d in sd + md for p in d["per"] if p["status"] == "detected"]
    cards = {"Single leaks detected": f"{tp} / {len(sd)}",
             f"Leaks detected on {MULTI_TXT}-leak days": f"{m_ok} / {len(mp)}",
             "False detections": f"{fp} / {len(nd)} leak-free days, {sum(len(d['false']) for d in sd + md)} on leak days",
             "Median detection delay": f"{np.median([p['delay'] for p in all_det]):.0f} min" if all_det else "-"}
    chosen = pd.DataFrame([{"Setting": SETTING_NAMES[k][0], "Chosen value": v, "What it means": SETTING_NAMES[k][1],
                            "Name in the code": k} for k, v in best.items()])
    tuning_out = tuning.rename(columns={k: v[0] for k, v in SETTING_NAMES.items()})

    # ---------------- save ----------------
    print("\nWriting the output files ...")
    kx = max(1, int(round(RAW_EXCEL_EVERY_MIN / READING_INTERVAL_MIN)))
    thin = lambda o: {k_: (v[::kx] if k_ in ("t", "P", "Q", "V") else v) for k_, v in o.items()}
    for d in training:
        d["actual"] = np.zeros(len(d["obs"]["t"]), bool)
        for l in d["leaks"]: d["actual"] |= (d["obs"]["t"] >= hours(l["start"])) & (d["obs"]["t"] < hours(l["end"]))
    readme = pd.DataFrame(README, columns=["File / topic", "Sheet / term", "Meaning"])
    n_lk = {}
    for is_test, ds in ((False, training), (True, tests)):
        for d in ds:
            for l in d["leaks"]: n_lk[(l["pipe"], is_test)] = n_lk.get((l["pipe"], is_test), 0) + 1
    split_df = pd.DataFrame([{"Pipe": p, "Set": "TEST" if p in test_pipes else "TRAINING",
                              "User-defined leaks": "yes" if p in USER_PIPES else "", "Training leaks": n_lk.get((p, False), 0),
                              "Test leaks": n_lk.get((p, True), 0)} for p in net.pipes])
    files = [safe_save(OUTPUT_DIR / "training_data.xlsx", {
        "README": readme, "Pipe_split": split_df, "Training_leaks": training_table, "Chosen_settings": chosen, "Tuning": tuning_out})]
    if SAVE_RAW_TRAINING_DATA:                      # 3 sheets: Pressure, Flow, Velocity of every normal and training day
        print("  writing training_raw_epanet.xlsx (Pressure / Flow / Velocity of every normal and training day) ...")
        days_ = [([], o, np.zeros(len(o["t"]), bool)) for o in train_obs] + [(d["leaks"], d["obs"], d["actual"]) for d in training]
        tabs = [raw_tables(net, k + 1, lk, o, on) for k, (lk, o, on) in enumerate(days_)]
        files.append(safe_save(OUTPUT_DIR / "training_raw_epanet.xlsx",
                               {name: pd.concat([t_[i] for t_ in tabs], ignore_index=True)
                                for i, name in enumerate(("Pressure", "Flow", "Velocity"))}))
    timeline = pd.DataFrame({"Time": [hhmm(x) for x in tests[0]["obs"]["t"]], **timeline})
    files += [safe_save(OUTPUT_DIR / "detection_results.xlsx",
                        {"README": readme, "Pipe_split": split_df, "Detection_summary": wsum, "Single_leak_results": single_table,
                         "Multi_leak_results": multi_table, "Detection_delay": dsum, "Delay_vs_leak_size": sweep_pivot,
                         "Leak_size_sweep_all": sweep_df.rename(columns={"Delay": "Detection delay (min)"}),
                         "Confusion_matrix": cm_table, "Detection_timeline": timeline}),
              safe_save(OUTPUT_DIR / "simulated_data.xlsx", {"README": readme, "Junction_elevations": network_info(net)[0], "Pipe_data": network_info(net)[1],
                                                              **epanet_sheets(net, [f"{d['n']}. {d['name']}" for d in tests],
                                                                              [thin(d["obs"]) for d in tests], [d["R"][::kx] for d in tests],
                                                                              [d["actual"][::kx] for d in tests])})]
    save_confusion_png(OUTPUT_DIR / "confusion_matrix.png", cms); files.append(OUTPUT_DIR / "confusion_matrix.png")
    save_delay_png(OUTPUT_DIR / "detection_delay.png", single, sweep_df); files.append(OUTPUT_DIR / "detection_delay.png")
    ok_p = [d["leaks"][0]["pipe"] for d in sd if d["per"][0]["status"] == "detected"]
    miss_p = [d["leaks"][0]["pipe"] for d in sd if d["per"][0]["status"] != "detected"]
    multi_p = sorted({l["pipe"] for d in md for l in d["leaks"]} - set(ok_p) - set(miss_p))
    train_p = list(train_pipes)
    info = {p: [f"TRAINING pipe ({TRAIN_FRACTION:.0%} part)" + (" - user-defined leaks" if p in USER_PIPES else "") + ": leaks used only to tune the detector"]
            for p in train_p}
    for p in test_pipes: info[p] = [f"TEST pipe ({1 - TRAIN_FRACTION:.0%} part)"]
    res = lambda r: f"detected at {hhmm(r['time'])}" if r.get("status") == "detected" else "MISSED"
    for d in sd:
        l, r = d["leaks"][0], d["per"][0]
        info.setdefault(l["pipe"], []).append(f"single test leak {l['start']}-{l['end']} (coef {l['coefficient']}): {res(r)}")
    for d in md:
        for l, r in zip(d["leaks"], d["per"]):
            info.setdefault(l["pipe"], []).append(f"{d['name']}: leak {l['start']}-{l['end']} (coef {l['coefficient']}): {res(r)}")
    info = {p: v[:6] + ([f"... and {len(v) - 6} more"] if len(v) > 6 else []) for p, v in info.items()}
    save_result_map_png(OUTPUT_DIR / "test_leaks_map.png", net, ok_p, miss_p, multi_p, train_p); files.append(OUTPUT_DIR / "test_leaks_map.png")
    tb = lambda df: df.to_html(index=False, classes="tbl", border=0, na_rep="")
    delay_html = ("<h3>Summary</h3>" + tb(wsum) + img_tag(OUTPUT_DIR / "detection_delay.png") + "<h3>Detection delay (days with one leak)</h3>" +
                  tb(dsum) + (f"<h3>Leak-size sweep ({N_SWEEP_PIPES} random test pipes, start at a random minute after "
                              f"{' / '.join(f'{h:02d}:00' for h in SWEEP_START_HOURS)})</h3>" + tb(sweep_pivot) if len(sweep_pivot) else ""))
    short = {"factor": "how far above normal the anomaly score must rise before a leak is reported",
             "jump": "how big a sudden change between two readings must be to count as a new leak",
             "rel": "that sudden change must also be at least this share of the current leak signal",
             "swap": "below this pattern similarity, a sudden drop = one leak stopped and a new one started",
             "grace": "readings to wait after a detection before looking for the next new leak"}
    settings_box = ("<div class='card' style='max-width:1100px;margin:6px 0 10px'><b style='font-size:14px'>The 5 detector settings in "
                    "short</b><ul style='margin:6px 0 2px 18px;padding:0'>" +
                    "".join(f"<li><b style='display:inline;font-size:13px'>{SETTING_NAMES[k][0]}</b> ({best[k]:g}): {v}</li>" for k, v in short.items()) +
                    "</ul></div>")
    tuning_html = settings_box + (f"<div class='note'>Every combination of the settings below was tried on the training leaks ({len(training)} days on the "
                   f"{len(train_pipes)} training pipes = {TRAIN_FRACTION:.0%} of the pipes); the one with the highest "
                   "score (F1 of the leak detections = balance of leaks found and false detections, and no false detection on "
                   "leak-free days) is used for the unseen test.</div><h3>Chosen settings</h3>" + tb(chosen) + "<h3>Top 10 of the tried combinations</h3>" + tb(tuning_out.head(10)))
    single_show = ["No.", "Test", "Leak pipe", "Emitter coef", "Leak type", "Leak active (induced)", "Leak detected at",
                   "Detection delay (min)", "Result"]
    files.append(write_detection_html(OUTPUT_DIR / "leak_detection_report.html", net, "Drumchapel - real-time leak detection with the Matrix Profile",
                                      cards, cms, js, single_table[[c for c in single_show if c in single_table]], multi_table,
                                      delay_html, dict(ok=ok_p, miss=miss_p, multi=multi_p, train=train_p, info=info), tuning_html))

    # ---------------- console report ----------------
    print(f"\nSETTINGS CHOSEN ON THE TRAINING LEAKS ({len(train_pipes)} training pipes)"); print_table(chosen[["Setting", "Chosen value"]])
    print("\nTUNING - best 5 combinations (training leaks)"); print_table(tuning_out.head(5))
    print(f"\nTEST - DAYS WITH ONE LEAK (unseen, {len(test_pipes)} test pipes, random order)"); print_table(single_table.loc[single_table["Leak pipe"] != "none", single_show])
    print(f"\nTEST - DAYS WITH {MULTI_TXT} LEAKS (unseen, test pipes)")
    print_table(multi_table[["No.", "Day", "Leak", "Leak pipe", "Emitter coef", "Leak active (induced)", "Leak detected at",
                             "Detection delay (min)", "How it was detected"]])
    print("\nTEST - LEAK-FREE DAYS"); print_table(single_table.loc[single_table["Leak pipe"] == "none", ["No.", "Test", "Leak detected at", "Result"]])
    print("\nDETECTION - SUMMARY"); print_table(wsum)
    print("\nHOW FAST ARE SINGLE LEAKS DETECTED?"); print_table(dsum)
    if len(sweep_pivot): print(f"\nLEAK-SIZE SWEEP ({N_SWEEP_PIPES} random test pipes)"); print_table(sweep_pivot)
    for k, m in cms.items():
        print(f"\nCONFUSION MATRIX - {k.upper()}")
        print(f"                     Detected LEAK   Detected NO LEAK")
        print(f"  Actual LEAK        TP = {m['TP']:<10}  FN = {m['FN']}")
        print(f"  Actual NO LEAK     FP = {m['FP']:<10}  TN = {m['TN']}")
        print(f"  Accuracy {m['Accuracy']:.1%} | Precision {m['Precision']:.1%} | Recall {m['Recall']:.1%} | F1 {m['F1']:.3f}")
    print("\nFILES WRITTEN:")
    what = {"training_data.xlsx": "training: 80/20 pipe split, training leaks, chosen settings, tuning",
            "training_raw_epanet.xlsx": "training: raw EPANET data of every normal and training day (Pressure / Flow / Velocity)",
            "simulated_data.xlsx": f"test data: raw EPANET results + sensor changes of all {len(tests)} test days",
            "detection_results.xlsx": "test results: single leaks, multi-leak days, detection summary, delay, sweep, confusion matrices",
            "detection_delay.png": "how fast leaks are detected (time to detection, delay vs leak size)"}
    for f in files: print(f"  {Path(f).resolve()}" + (f"\n      -> {what[Path(f).name]}" if Path(f).name in what else ""))
    open_file(files[-1])
    if OPEN_OUTPUT_FOLDER: open_file(OUTPUT_DIR)


if __name__ == "__main__":
    main()
