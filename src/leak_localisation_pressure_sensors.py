#!/usr/bin/env python3
"""
main.exe - LEAK DETECTION AND LOCALISATION WITH A REDUCED NUMBER OF PRESSURE SENSORS   (ONE file, Linux)
=====================================================================================================
HOW TO RUN (Linux workstation; python3 needed; internet needed only on the first start):
    chmod +x main.exe
    ./main.exe all        -> ALL 6 networks, one after another (recommended)
    ./main.exe            -> menu: choose networks (e.g. 3, 2 4, or A = all; add F for the fast setting)
    ./main.exe 2 fast     -> quick check with fewer test days (not for the final results)
ONE PARTICULAR TOWN FIRST - type its number (the .inp files are INSIDE this file):
    ./main.exe 1          -> Modena            -> Modena.html          (or: python3 Modena.py)
    ./main.exe 2          -> L-Town            -> L-Town.html          (or: python3 L-Town.py)
    ./main.exe 3          -> L-Town v2 Model   -> L-Town_v2_Model.html (or: python3 L-Town_v2_Model.py)
    ./main.exe 4          -> L-Town v2 Real    -> L-Town_v2_Real.html  (or: python3 L-Town_v2_Real.py)
    ./main.exe 5          -> Hanoi             -> Hanoi.html           (or: python3 Hanoi.py)
    ./main.exe 6          -> D-Town            -> D-Town.html          (or: python3 D-Town.py)
SEVERAL TOWNS IN YOUR OWN ORDER - numbers separated by spaces, run left to right:
    ./main.exe 6 1        -> D-Town first, then Modena         ./main.exe 2 3 4 -> the three L-Town networks
    (L-Town v2 Real is the original 177 MB file; its year-long demand patterns are read only for the first week,
     in memory, because every simulated day starts at 0:00 - the file itself is not changed.)
FIRST START: missing packages (epyt = EPANET Python Toolkit incl. the EPANET engine, numpy, pandas, openpyxl,
matplotlib, plotly; PyTorch if there is an NVIDIA GPU, about 2-3 GB) are installed automatically into the folder
.leak_env next to main.exe.  If it stops with "install python3-venv":  sudo apt install python3-venv  and run again.
RESULTS (next to main.exe): one report per network (Modena.html, L-Town.html, ... - open in a browser) + one results
folder per network (<network>_pressure_sensor_study: Excel tables, figures, trained models).
main.exe also writes one script per network: Modena.py, L-Town.py, L-Town_v2_Model.py, L-Town_v2_Real.py, Hanoi.py,
D-Town.py  (python3 Modena.py = ./main.exe 1).
EVERY NETWORK is run with the same method and the same full settings (80/20 split by scenarios, 40 % single-leak /
40 % multi-leak (half overlapping, half non-overlapping) / 20 % leak-free days in training, validation AND test,
sensors halved step by step, encoder-decoder + hydraulic check) and gets the same report (results summary, training /
validation / test tables, confusion matrices, key findings on sensors vs multiple leaks, maps, every test day).
HARDWARE / RUN TIME: EPANET runs on all CPU cores in parallel (more cores = faster); the encoder-decoder is trained on
the GPU if there is one (exactly the same results as on the CPU).  Approximate time per network (grows with the number
of pipes, shrinks with more CPU cores):  Hanoi (34 pipes) under 1 h;  Modena (317 pipes) and D-Town (443 pipes, 7 tanks,
11 pumps) several hours;  L-Town, L-Town v2 Model, L-Town v2 Real (about 905 pipes each) the longest, many hours each.
A network that has finished is not affected if a later one is stopped; trained models are cached, so a stopped
network restarts faster.
(This file is a shell script and a Python program at the same time: the shell part above prepares Python and
 then runs the rest of this same file with it.)
"""
# (the description of the study follows)

# =====================================================================================================================
# 0. REQUIREMENTS - installed automatically the first time (EPyT = EPANET Python Toolkit, includes the EPANET engine)
# =====================================================================================================================
import sys as _sys, importlib.util as _ilu, subprocess as _sp
_REQUIRED = {"epyt": "epyt", "numpy": "numpy", "pandas": "pandas", "openpyxl": "openpyxl", "matplotlib": "matplotlib",
             "plotly": "plotly"}
_OPTIONAL = {"plotly"}          # without plotly the HTML report loads its charts from the internet
if not getattr(_sys, "frozen", False):
    for _mod, _pip in _REQUIRED.items():
        if _ilu.find_spec(_mod) is None:
            print(f"Installing the required package {_pip} ...")
            if _sp.call([_sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                         "--no-warn-script-location", _pip]) != 0:
                if _mod in _OPTIONAL:
                    print(f"  ! {_pip} could not be installed - continuing without it (the report then needs internet)")
                else:
                    _sys.exit(f"  ! {_pip} could not be installed - install it with:  {_sys.executable} -m pip install {_pip}")

import re

# =====================================================================================================
# PART A - HELPERS, EPANET NETWORK AND SIMULATION
# =====================================================================================================
from pathlib import Path
import os, sys, json, heapq, pickle, time, contextlib, tempfile, webbrowser, warnings, shutil, importlib, importlib.util
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


# chart library (plotly.js) when the python package plotly is not installed: loaded when the report is opened (internet
# needed then); three sources are tried one after another; a note is shown if none can be reached.
PLOTLY_CDN_TAG = (
    '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
    '<script>window.Plotly||document.write(\'<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"><\\/script>\')</script>'
    '<script>window.Plotly||document.write(\'<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"><\\/script>\')</script>'
    '<script>window.Plotly||document.addEventListener("DOMContentLoaded",function(){var d=document.createElement("div");'
    'd.style.cssText="background:#fde8e8;border:1px solid #d33;padding:10px;margin:10px 0;font:15px sans-serif";'
    'd.textContent="The charts could not be drawn: the chart library (plotly.js) is loaded from the internet when this '
    'report is opened. Open the report on a computer with internet (or install the python package plotly and re-run).";'
    'document.body.insertBefore(d,document.body.firstChild)});</script>')


_TRIM_REPORTED = set()


def trim_long_patterns(lines, name="", keep=2016):
    """Very long demand patterns (e.g. L-TOWN_v2_Real: one value every 5 min for a whole year, 177 MB) are used here
    only for their first week (2016 values): the study simulates single days from 0:00, so the rest is never read.
    Done in memory only - the .inp file itself stays the original.  (Without this every one of the thousands of
    simulations would re-read 177 MB.)"""
    out, sec, seen, cut = [], None, {}, 0
    for line in lines:
        s = line.strip()
        if s.startswith("["): sec = s.upper(); out.append(line); continue
        if sec == "[PATTERNS]" and s and not s.startswith(";"):
            tok = s.split(";")[0].split(); pid, vals = tok[0], tok[1:]; n = seen.get(pid, 0)
            if n >= keep: cut += len(vals); continue
            if n + len(vals) <= keep: seen[pid] = n + len(vals); out.append(line); continue   # unchanged line
            cut += n + len(vals) - keep; vals = vals[:keep - n]
            seen[pid] = n + len(vals); out.append(" " + pid + "\t" + "\t".join(vals)); continue
        out.append(line)
    if cut and name not in _TRIM_REPORTED:
        _TRIM_REPORTED.add(name)
        print(f"    {name}: demand patterns longer than one week -> the first week is used (the study simulates single days)")
    return out


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
        if self.inp.stat().st_size < 50e6:
            self.lines = trim_long_patterns(self.inp.read_text(encoding="utf-8", errors="ignore").splitlines(), self.inp.name)
        else:                                # very large file: read it line by line (keeps the memory use small)
            with open(self.inp, encoding="utf-8", errors="ignore") as fh:
                self.lines = trim_long_patterns((l.rstrip("\r\n") for l in fh), self.inp.name)
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
            self.valves = list(G.getLinkValveNameID()) if G.getLinkValveCount() else []
            self.pipes_all = list(G.getLinkPipeNameID())
            try: self.link_type = {l: str(t).upper() for l, t in zip(lid, G.getLinkType())}
            except Exception: self.link_type = {}
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
        self.vertices, out, sec, ctrl = {}, [], None, set()
        for line in self.lines:     # read vertices + write the SCADA-driven expected model
            s = line.split(";")[0].strip(); tok = s.split()
            if s.startswith("["):
                sec = s.upper(); out.append("[RESERVOIRS]" if sec == "[TANKS]" else line); continue
            if sec == "[VERTICES]" and len(tok) >= 3:
                self.vertices.setdefault(tok[0], []).append((float(tok[1]), float(tok[2])))
            if sec == "[TANKS]" and len(tok) >= 3:
                out.append(f"{tok[0]} {float(tok[1]) + float(tok[2]):.4f}"); continue
            if sec in ("[CONTROLS]", "[RULES]") and tok:      # links switched by controls / rules (e.g. a bypass pipe)
                for a_, b_ in zip(tok[:-1], tok[1:]):
                    if a_.upper() == "LINK" and b_ in self.ends: ctrl.add(b_)
                continue
            out.append(line)
        # links whose open / closed status is copied from the measurements into the expected model:
        # pumps, valves and every link switched by a control or rule (general for any network)
        # Pressure / flow CONTROL valves (PRV, PSV, PBV, FCV) are NOT copied: their status read from EPANET is only
        # open / closed, and forcing "open" would switch off their pressure control (the valve would open fully and the
        # expected model would differ from the real network by many metres, e.g. D-Town).  They keep working by
        # themselves in the expected model (same settings as the real network).
        ctl_valves = {v for v in self.valves if self.link_type.get(v, "") in ("PRV", "PSV", "PBV", "FCV")}
        # + every link at a TANK: EPANET closes it while the tank is full (or empty); in the expected model the tank is
        # a fixed-head node that never fills, so this closure must be copied too (e.g. D-Town, 7 tanks).  Check-valve
        # pipes are left out (forcing them open would switch off the check valve).
        tank_links = [l for l in self.ends if set(self.ends[l]) & set(self.tanks) and self.link_type.get(l, "") != "CVPIPE"]
        self.status_links = [l for l in dict.fromkeys(self.pumps + self.valves + sorted(ctrl, key=lambda l: list(self.ends).index(l))
                                                      + tank_links) if l not in ctl_valves]
        self.scada_lines = out                       # expected (SCADA) model: also used with a candidate leak
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
            adj = {}       # all links (also pumps, valves and closed pipes) so that every part of the network is reachable
            for p in self.pipes_all + self.pumps + getattr(self, "valves", []):
                a, b = self.ends[p]; w = self.length.get(p, 0.0) if p in self.pipes_all else 0.0
                adj.setdefault(a, []).append((b, w)); adj.setdefault(b, []).append((a, w))
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
    def leak_inp(self, leaks, scada=False):
        """Copy of the INP (scada=True: of the SCADA expected model) with every (pipe, pos) in `leaks` split at pos (0-1)
        and a junction LEAK_<pipe> inserted."""
        leaks = dict(leaks); lines_j = []
        for pipe, pos in leaks.items():
            if not 0 < pos < 1: raise ValueError("leak position must be between 0 and 1")
            a, b = self.ends[pipe]
            lines_j.append(f"LEAK_{pipe}\t{self.elev[a] + pos * (self.elev[b] - self.elev[a]):.4f}")   # interpolated elevation
        out, sec = [], None
        for line in (self.scada_lines if scada else self.lines):
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
        path = self.tmp / f"{'scada_' if scada else ''}leak_{abs(hash(tuple(sorted(leaks)))) % 10 ** 8}.inp"
        path.write_text("\n".join(out), encoding="utf-8")
        return path

    def open(self, pipe=None, pos=None, scada=False, leaks=None):
        """Load the real network (pipe=None), a leak model (one leak: pipe + pos; several: leaks=[(pipe, pos), ...])
        or the SCADA-driven expected model.  EPyT keeps ONE model open at a time -> close() before the next."""
        leaks = list(leaks or ([(pipe, pos)] if pipe is not None else []))
        path = self.leak_inp(leaks, scada) if leaks else (self.scada_inp if scada else self.work_inp)
        with quiet(): G = epanet(str(path))
        G.setTimeHydraulicStep(STEP_S); G.setTimeReportingStep(STEP_S)
        if self.daily_pattern is not None:
            G.setPattern(1, list(self.daily_pattern))
        lid, lp = list(G.getLinkNameID()), [p for p, _ in leaks]
        return dict(G=G, j=[G.getNodeIndex(n) - 1 for n in self.junctions],
                    la=[lid.index(f"{p}_A" if p in lp else p) for p in self.pipes_all],
                    lb=[lid.index(f"{p}_B" if p in lp else p) for p in self.pipes_all],
                    tk=[G.getNodeIndex(n) for n in self.tanks],
                    pu=[G.getLinkIndex(f"{p}_A" if p in lp else p) for p in self.status_links],
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
        z --CLASSIFIER--> softmax over the pipes -> leaks on the same pipe get similar embeddings

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
        dev = gpu_device()
        if dev is not None:                       # GPU available -> the SAME network is trained with PyTorch on the GPU
            exact = bool(globals().get("CFG", {}).get("GPU_EXACT", True))
            try:
                if exact and not self._gpu_matches_cpu(dev, X, y, Xv, yv, lr, batch, weight_decay, input_noise, seed):
                    raise RuntimeError("GPU training does not reproduce the CPU training")
                return self._fit_torch(dev, X, y, Xv, yv, epochs, lr, batch, weight_decay, patience, input_noise, seed, log,
                                       exact)
            except Exception as e:                # anything wrong on the GPU -> the tested NumPy training on the CPU
                print(f"    ! GPU training not used ({type(e).__name__}: {str(e)[:120]}) -> training on the CPU (NumPy)")
                self.history = []
        return self._fit_numpy(X, y, Xv, yv, epochs, lr, batch, weight_decay, patience, input_noise, seed, log)

    def _gpu_matches_cpu(self, dev, X, y, Xv, yv, lr, batch, weight_decay, input_noise, seed):
        """Safety check before training on the GPU: ONE epoch on the CPU (NumPy) and ONE epoch on the GPU from the same
        start must give the same weights (relative difference < 1e-6).  Only then the GPU is used."""
        import copy
        a, b = copy.copy(self), copy.copy(self)
        a.W = {k: v.copy() for k, v in self.W.items()}; b.W = {k: v.copy() for k, v in self.W.items()}; a.history, b.history = [], []
        a._fit_numpy(X, y, Xv, yv, 1, lr, batch, weight_decay, 10 ** 6, input_noise, seed, None)
        b._fit_torch(dev, X, y, Xv, yv, 1, lr, batch, weight_decay, 10 ** 6, input_noise, seed, None, True)
        d = max(float(np.max(np.abs(a.W[k] - b.W[k])) / (np.max(np.abs(a.W[k])) + 1e-12)) for k in a.W)
        ok = d < 1e-6
        print(f"    GPU check: 1 epoch GPU vs CPU, largest relative weight difference {d:.1e} -> " +
              ("identical, training on the GPU" if ok else "NOT identical"))
        return ok

    def _fit_numpy(self, X, y, Xv, yv, epochs, lr, batch, weight_decay, patience, input_noise, seed, log):
        """The original (tested) NumPy training on the CPU."""
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
        self.device = "CPU (NumPy)"
        return self

    def _fit_torch(self, dev, X, y, Xv, yv, epochs, lr, batch, weight_decay, patience, input_noise, seed, log, exact=True):
        """GPU training (PyTorch).  exact=True (GPU_EXACT): the SAME computation as the NumPy training - same start
        weights, same random numbers (batch order, input noise from the same NumPy generator), same loss, same AdamW
        update, float64 - so the trained model and all results are the same as on the CPU, only faster for big networks.
        exact=False: float32 + PyTorch random numbers (faster, results equivalent but not identical).
        The trained weights are copied back into self.W (NumPy), so everything after training works as before."""
        import torch
        dt = torch.float64 if exact else torch.float32
        torch.manual_seed(seed); r = np.random.default_rng(seed)
        T = lambda a: torch.tensor(np.asarray(a), dtype=dt, device=dev)
        P = {k: T(v).requires_grad_(True) for k, v in self.W.items()}
        M = {k: torch.zeros_like(v) for k, v in P.items()}; V = {k: torch.zeros_like(v) for k, v in P.items()}
        Xg, Xvg = T(X), T(Xv); yg = torch.tensor(np.asarray(y), dtype=torch.long, device=dev)
        yvg = torch.tensor(np.asarray(yv), dtype=torch.long, device=dev)
        sub_np = r.choice(len(X), min(len(X), 5000), replace=False); sub = torch.tensor(sub_np, device=dev)

        def forward(x):
            h1 = torch.tanh(x @ P["W1"] + P["b1"]); z = h1 @ P["W2"] + P["b2"]
            xh = torch.tanh(z @ P["W3"] + P["b3"]) @ P["W4"] + P["b4"]; return xh, z @ P["Wc"] + P["bc"]

        def evaluate(x, yy):                      # same formula as the NumPy evaluate()
            xh, lg = forward(x); pr = torch.softmax(lg, 1)
            loss = ((xh - x) ** 2).mean() - torch.log(pr[torch.arange(len(yy), device=dev), yy] + 1e-12).mean()
            return float(loss), float((pr.argmax(1) == yy).to(dt).mean())

        step, best, best_W, wait = 0, np.inf, None, 0
        for ep in range(epochs):
            idx_np = r.permutation(len(X))
            for s0 in range(0, len(X), batch):
                b_np = idx_np[s0:s0 + batch]; b = torch.tensor(b_np, device=dev); x0 = Xg[b]
                noise = T(r.normal(0, input_noise, (len(b_np), X.shape[1]))) if exact else input_noise * torch.randn_like(x0)
                xh, lg = forward(x0 + noise)
                loss = ((xh - x0) ** 2).mean() + torch.nn.functional.cross_entropy(lg, yg[b])
                G = torch.autograd.grad(loss, list(P.values()))
                step += 1
                with torch.no_grad():                # AdamW exactly as in the NumPy training
                    for (k, p), g in zip(P.items(), G):
                        M[k].mul_(0.9).add_(0.1 * g); V[k].mul_(0.999).add_(0.001 * g * g)
                        p.sub_(lr * (M[k] / (1 - 0.9 ** step)) / (torch.sqrt(V[k] / (1 - 0.999 ** step)) + 1e-8))
                        if k[0] == "W": p.sub_(lr * weight_decay * p)
            with torch.no_grad():
                tl, ta = evaluate(Xg[sub], yg[sub]); vl, va = evaluate(Xvg, yvg)
            self.history.append(dict(epoch=ep + 1, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=va))
            if vl < best - 1e-4: best, best_W, wait = vl, {k: p.detach().cpu().numpy().astype(float).copy() for k, p in P.items()}, 0
            else: wait += 1
            if log and (ep + 1) % 25 == 0:
                log(f"   epoch {ep + 1:4d} | train loss {tl:.3f} acc {ta:.1%} | validation loss {vl:.3f} acc {va:.1%}  [GPU]")
            if wait >= patience:
                if log: log(f"   early stopping at epoch {ep + 1} (best validation loss at epoch {ep + 1 - wait})")
                break
        self.W = best_W if best_W is not None else {k: p.detach().cpu().numpy().astype(float) for k, p in P.items()}
        self.best_epoch = int(np.argmin([h["val_loss"] for h in self.history])) + 1
        self.device = f"GPU ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else f"GPU ({dev.type})"
        return self


_GPU = "unchecked"


def gpu_device():
    """The GPU to train on (CUDA), or None -> CPU.  USE_GPU = False in the SETTINGS switches the GPU off."""
    global _GPU
    if _GPU == "unchecked":
        _GPU = None
        if globals().get("CFG", {}).get("USE_GPU", True):
            try:
                import torch
                if torch.cuda.is_available(): _GPU = torch.device("cuda:0")
            except Exception:
                _GPU = None
        print(f"    encoder-decoder training on: {'GPU - ' + __import__('torch').cuda.get_device_name(0) if _GPU is not None else 'CPU (NumPy)'}")
    return _GPU


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

    def fit_position(self, X, pipes, positions, features="embedding", gamma=2.0, alpha=1e-3):
        """POSITION MODEL (one per pipe): kernel ridge regression  position = f(embedding)  learned from all
        simulated leaks of that pipe.  It interpolates smoothly between the training positions, so it is more
        precise than averaging the neighbours' positions."""
        F = self._pos_features(np.asarray(X, float), features); pipes = np.asarray(pipes); positions = np.asarray(positions, float)
        self.pos_model, self.pos_cfg, self.pos_alpha = {}, (features, gamma), alpha
        for p in np.unique(pipes):
            m = pipes == p; A = F[m]; mu = positions[m].mean()
            K = np.exp(-gamma * np.maximum((A ** 2).sum(1)[:, None] + (A ** 2).sum(1)[None, :] - 2 * A @ A.T, 0))
            self.pos_model[p] = (A, np.linalg.solve(K + alpha * np.eye(len(A)), positions[m] - mu), mu)
        return self

    def fit_zone_confidence(self, X, meta, cfg):
        """ZONE CONFIDENCE: how far is the localised position typically from the true one?  Learned on validation
        leaks at positions NEVER used for training (a localiser trained without them).  The mean absolute error of
        each pipe = spread b of the position (Laplace distribution around the localised position)."""
        va = split_by_position(meta); tr = ~va
        L = EmbeddingLocaliser(self.model, X[tr], meta.Pipe[tr], meta.Position[tr], self.k, self.tau)
        if getattr(self, "pos_model", None):
            L.fit_position(X[tr], meta.Pipe[tr], meta.Position[tr], self.pos_cfg[0], self.pos_cfg[1], getattr(self, "pos_alpha", 1e-3))
        err = {}
        for i in np.where(va)[0]:
            r = L.locate(X[i]); p = meta.Pipe.iloc[i]
            if r["pipe"] == p: err.setdefault(p, []).append(abs(r["pos"] - meta.Position.iloc[i]))
        self.pos_spread = {p: float(np.mean(v)) for p, v in err.items()}
        self.pos_spread_all = float(np.mean(np.concatenate([np.asarray(v) for v in err.values()]))) if err else 0.05
        return self

    def zone_confidence(self, pipe, pos):
        """Confidence (0-1) that the leak lies in each zone of the localised pipe: share of a Laplace distribution
        (centre = localised position, spread = typical position error of that pipe) that falls inside each zone."""
        b = max(getattr(self, "pos_spread", {}).get(pipe, getattr(self, "pos_spread_all", 0.05)), 1e-3)
        cdf = lambda x: 0.5 * np.exp(x / b) if x < 0 else 1 - 0.5 * np.exp(-x / b)
        E = np.asarray(ZONE_EDGES, float)
        m = np.array([cdf(hi - pos) - cdf(lo - pos) for lo, hi in zip(E[:-1], E[1:])])
        return m / m.sum()

    def _pos_features(self, X, features):
        return self.embed(X) if features == "embedding" else X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def locate(self, x):
        sim = self.Z @ self.embed(x)[0]; nn = np.argsort(-sim)[: self.k]
        w = np.exp(self.tau * (sim[nn] - sim[nn[0]]))
        votes = pd.Series(w).groupby(self.pipes[nn]).sum().sort_values(ascending=False)
        pipe = votes.index[0]; same = self.pipes[nn] == pipe
        if getattr(self, "zone_vote", False):           # zone = weighted vote of the neighbours on that pipe,
            zn = np.array([zone_index(p) for p in self.pos[nn]])
            zv = pd.Series(w[same]).groupby(zn[same]).sum()
            same = same & (zn == zv.idxmax())          # position = weighted mean of the neighbours in that zone
        pos = float(np.sum(w[same] * self.pos[nn][same]) / np.sum(w[same]))
        if getattr(self, "pos_model", None) and pipe in self.pos_model:      # kernel ridge position model
            A, wk, mu = self.pos_model[pipe]; f = self._pos_features(np.atleast_2d(np.asarray(x, float)), self.pos_cfg[0])[0]
            pos = float(np.clip(np.exp(-self.pos_cfg[1] * ((A - f) ** 2).sum(1)) @ wk + mu, 0.01, 0.99))
        zc = self.zone_confidence(pipe, pos) if hasattr(self, "pos_spread") else None
        return dict(pipe=pipe, pos=pos, confidence=float(votes.iloc[0] / votes.sum()), similarity=float(sim[nn[0]]), zone_conf=zc,
                    candidates=", ".join(f"{p} ({v / votes.sum():.0%})" for p, v in votes.head(3).items()))

    def reference_table(self):
        df = pd.DataFrame(self.Z, columns=[f"z{i + 1}" for i in range(self.Z.shape[1])])
        df.insert(0, "Pipe", self.pipes); df.insert(1, "Zone", [zone(p) for p in self.pos])
        g = df.groupby(["Pipe", "Zone"]); out = g.mean(); out.insert(0, "N_reference_leaks", g.size())
        return out.reset_index()


_NET_CACHE = {}


def training_positions(n_pos, rng):
    """Leak positions on one pipe: spread over every zone (number per zone proportional to the zone length,
    e.g. 8 per 20 % zone), each at a random place inside its own slot -> the whole pipe is covered."""
    E = np.asarray(ZONE_EDGES, float); w = np.diff(E); n = np.floor(n_pos * w / w.sum()).astype(int)
    n[np.argsort(-(n_pos * w / w.sum() - n))[: n_pos - n.sum()]] += 1      # distribute the remainder
    out = []
    for lo, hi, k in zip(E[:-1], E[1:], n):
        lo, hi = max(lo, 0.02), min(hi, 0.98)
        out.append(lo + (np.arange(k) + rng.uniform(0.05, 0.95, k)) / max(k, 1) * (hi - lo))
    return np.concatenate(out)


def split_by_position(meta, every=5):
    """Validation = every 5th training position of each pipe (whole positions held out, incl. all copies),
    so validation measures how well the model handles leak positions it has NEVER seen."""
    val = np.zeros(len(meta), bool)
    for _, g in meta.groupby("Pipe"):
        pos = np.sort(g.Position.unique()); held = set(pos[2::every])
        val[g.index[g.Position.isin(held)]] = True
    return val


def train_localiser(net, X, meta, cfg, curve_path=None, log=print):
    """Train the encoder-decoder with a train / validation split, report how well it generalises,
    then build the final reference embedding database from ALL simulated leaks."""
    lab = meta.Pipe if cfg.get("CLASS_LABEL", "pipe") == "pipe" else meta.Pipe + "|" + meta.Zone   # classifier output
    cls = lab.astype("category"); y = cls.cat.codes.to_numpy()
    va = split_by_position(meta); tr = ~va
    model = EncoderDecoder(X.shape[1], len(cls.cat.categories), cfg["HIDDEN_DIM"], cfg["LATENT_DIM"], cfg["SEED"])
    model.fit(X[tr], y[tr], X[va], y[va], cfg["MAX_EPOCHS"], weight_decay=cfg["WEIGHT_DECAY"],
              patience=cfg["PATIENCE"], seed=cfg["SEED"], log=log)
    # ---- generalisation check on the held-out positions (references = training positions only) ----
    loc_tr = EmbeddingLocaliser(model, X[tr], meta.Pipe[tr], meta.Position[tr])
    if cfg.get("POSITION_METHOD") == "kernel_ridge":
        loc_tr.fit_position(X[tr], meta.Pipe[tr], meta.Position[tr], **cfg.get("POSITION_KRR", {}))
    q = meta[va & (meta.Noise_copy > 0).to_numpy()]
    q = q.iloc[:: max(1, len(q) // 3000)]
    rows = []
    for i, m_ in q.iterrows():
        r = loc_tr.locate(X[i])
        rows.append(dict(Pipe=m_.Pipe, ok=r["pipe"] == m_.Pipe, adj=net.adjacent(r["pipe"], m_.Pipe),
                         zone_ok=r["pipe"] == m_.Pipe and zone(r["pos"]) == m_.Zone,
                         zone_near=r["pipe"] == m_.Pipe and abs(zone_index(r["pos"]) - zone_index(m_.Position)) <= 1,
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
        ("Validation: correct pipe AND correct zone", f"{v.zone_ok.mean():.1%}"),
        ("Validation: correct pipe AND correct or neighbouring zone", f"{v.zone_near.mean():.1%}"),
        (f"Validation: median localisation error ({u})", f"{v.dist.median():.1f}"),
        (f"Validation: 90 % of errors below ({u})", f"{v.dist.quantile(0.9):.1f}"),
        (f"Median pipe length ({u})", f"{np.median([net.length[p] for p in net.pipes]):.1f}"),
        ("Trained on", getattr(model, "device", "CPU (NumPy)")),
    ], columns=["Metric", "Value"])
    per_pipe = v.groupby("Pipe").agg(Validation_leaks=("ok", "size"), Pipe_correct=("ok", "mean"),
                                     Zone_correct=("zone_ok", "mean"), Median_error=("dist", "median")).reset_index()
    if curve_path: save_training_curve(h, model.best_epoch, curve_path)
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
    if os.environ.get("SLURM_JOB_ID"): return             # batch job on a server: nothing to open
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
# PART D - PRESSURE-SENSOR STUDY: training data, sensor selection, tests, evaluation
# =====================================================================================================
FEATURES = ("H",)          # ONLY junction pressures (the change of hydraulic head = the change of pressure)


def proj_demand(R, S, scale):
    """Remove from every reading the part of the pressure change that a uniform change of ALL demands would
    cause (direction S from the hydraulic model).  Normal days: the random demand fluctuation disappears;
    a leak changes the pressures in a LOCAL pattern, which stays -> the detector sees the leak, not the demand."""
    X, Ss = R / scale, S / scale
    c = (X * Ss).sum(1) / ((Ss * Ss).sum(1) + 1e-12)
    return R - c[:, None] * S


def leak_coef_range(net, pct):
    """Emitter coefficients that give a leak of pct[0]-pct[1] % of the average network demand (any network / units)."""
    h = net.open(); o = net.run(h, 24.0); net.close(h)
    d_mean = float(np.mean(o["D"])); p_mean = float(np.mean(np.clip(o["P"], 1.0, None)))
    return tuple(float(p / 100 * d_mean / np.sqrt(p_mean)) for p in pct), d_mean, p_mean


def training_plan(n_pipes, cfg):
    """Positions per pipe x runs per position, so that the total is about TRAINING_SIMULATIONS
    (small network: 40 positions x 4 runs per pipe, big network: at least 20 positions x 2 runs)."""
    lo = int(cfg.get("MIN_SIM_PER_PIPE", 40))
    per_pipe = int(np.clip(round(cfg["TRAINING_SIMULATIONS"] / n_pipes), lo, max(lo, 160)))
    n_pos = int(np.clip(per_pipe // 2, min(20, max(1, per_pipe // 2)), 40)); runs = max(1, int(round(per_pipe / n_pos)))
    return n_pos, runs


def _train_pipe_job(args):
    """Worker (runs in parallel): all training leaks of ONE pipe -> pressure-change windows (all junctions)."""
    inp, pipe, k, cfg, crange = args
    if inp not in _NET_CACHE: _NET_CACHE[inp] = Network(inp)
    net, rng = _NET_CACHE[inp], np.random.default_rng([cfg["SEED"], k])
    n_pos, runs = training_plan(len(net.pipes), cfg); sims = []
    for pos in training_positions(n_pos, rng):
        h = net.open(pipe, pos)
        try:
            for _ in range(runs):
                coef = float(rng.uniform(*crange)); dur = float(rng.choice(cfg["TRAIN_DURATIONS_H"]))
                on = rng.integers(0, int((24 - dur) * 4) + 1) / 4; dm = float(rng.choice(cfg["DEMAND_MULTIPLIERS"]))
                sims.append((pos, coef, on, on + dur, dm, net.run(h, on + dur, coef, on, on + dur, dm)))
        finally:
            net.close(h)
    out, h = [], net.open(scada=True)
    try:
        for pos, coef, on, off, dm, sim in sims:
            (exp, S), i0, i1 = demand_response(net, sim, dm, h), int(on * 4), int(off * 4)
            R = residual(net, sim, exp, None)
            r0 = max(0, i0 - 4)                                    # raw data kept: 1 h before the leak -> leak end
            out.append(dict(Pipe=pipe, Position=round(float(pos), 4), Zone=zone(pos), Coefficient=round(coef, 4),
                            Start=hhmm(on), End=hhmm(off), Demand_multiplier=dm, R=R[i0:i1].astype(np.float32),
                            S=S[i0:i1].astype(np.float32), raw_t=sim["t"][r0:i1], raw_P=sim["P"][r0:i1].astype(np.float32),
                            raw_on=(sim["t"][r0:i1] >= on).astype(int)))
    finally:
        net.close(h)
    return out


def _test_day_job(args):
    """Worker: one test day (no leak, one leak or several leaks) -> pressure changes of all junctions."""
    inp, leaks, seed, cfg = args
    if inp not in _NET_CACHE: _NET_CACHE[inp] = Network(inp)
    net, rng = _NET_CACHE[inp], np.random.default_rng(seed)
    dn = demand_noise(rng, 24.0, cfg["DEMAND_NOISE"])
    if leaks:
        obs = net.simulate_leaks([dict(pipe=l["pipe"], position=l["position"], coefficient=l["coefficient"], on=l["on"],
                                       off=l["off"]) for l in leaks], 24.0, 1.0, dn)
    else:
        h = net.open(); obs = net.run(h, 24.0, dnoise=dn); net.close(h)
    exp, S = demand_response(net, obs)
    R = residual(net, obs, exp, rng, cfg["PRESSURE_NOISE"])
    return dict(t=obs["t"], R=R.astype(np.float32), S=S.astype(np.float32), P=obs["P"].astype(np.float32),
                L=obs["L"], dn=dn if dn is not None else np.ones(len(obs["t"])),
                bcH=np.asarray(obs["H"], np.float32), bcS=np.asarray(obs["S"], np.float32), Pexp=np.asarray(exp["P"], np.float32))


def _proj(R, S):
    """Remove the uniform-demand-change direction S from every reading (as in the fingerprint)."""
    return R - (np.sum(R * S, 1) / (np.sum(S * S, 1) + 1e-12))[:, None] * S


def _refine_day_job(args):
    """Safe wrapper: an unexpected error on one test day keeps the encoder-decoder answer for that day (and is reported)
    instead of stopping the whole run."""
    try:
        return _refine_day_core(args)
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {str(e)[:200]}"}


def _refine_day_core(args):
    """Worker: HYDRAULIC CHECK of the localisation of every detected leak of ONE day (all sensor counts).
    The encoder-decoder gives the most likely pipes; the hydraulic model, run under the conditions MEASURED that day
    (tank levels, pump / valve status), simulates a leak at several points of each of these pipes.  The candidate whose
    simulated pressure-change pattern matches the measured one best (cosine similarity at the sensors) wins.
      stage 1: top pipes x 5 zone centres, leak size = middle of the range
      stage 2: leak size estimated from the measured amplitude, best pipes x 10 points with that size.
    MULTIPLE LEAKS (REFINE_JOINT): a leak that starts while other leaks of the same alarm are running is checked TOGETHER
    with those leaks (at the pipe, point and size found for them): the simulated change it ADDS to the running leaks is
    compared with the measured change since just before it started."""
    inp, cfg, crange, scale, day, tasks, Wn = args
    if inp not in _NET_CACHE: _NET_CACHE[inp] = Network(inp)
    net = _NET_CACHE[inp]; cmid = float(np.mean(crange)); cache = {}
    R, S, t, bcH, bcS, Pexp = day["R"], day["S"], day["t"], day["bcH"], day["bcS"], day["Pexp"]

    def run_window(h, i0, i1, sched):
        G = h["G"]; G.setTimePatternStart(int(round(t[i0] * 3600)))
        o = net.run(h, (i1 - i0 - 1) * STEP_S / 3600, bc=dict(t=t[i0:i1], H=bcH[i0:i1], S=bcS[i0:i1]), schedule=sched)
        P = o["P"][: i1 - i0]
        if len(P) < i1 - i0: P = np.vstack([P, np.repeat(P[-1:], i1 - i0 - len(P), 0)])
        return P

    def simulate(reqs):
        """reqs: (pipe, pos, coef, i0, i1, bg) with bg = ((pipe, pos, coef), ...) leaks already running (may be empty).
        cache[req] = pressure change the candidate leak ADDS (to the expected pressures / to the running leaks)."""
        groups = {}
        for q in reqs:
            if q not in cache: groups.setdefault((q[0], q[5]), []).append(q)
        base = {}
        for (pipe, bg), qs in groups.items():
            if bg:                                    # pressures with only the running leaks (once per window)
                for i0, i1 in {(q[3], q[4]) for q in qs}:
                    if (bg, i0, i1) in base: continue
                    hb = net.open(leaks=[(p_, z_) for p_, z_, _ in bg], scada=True)
                    try: base[(bg, i0, i1)] = run_window(hb, i0, i1, {p_: [(c_, 0.0, 1e6)] for p_, _, c_ in bg})
                    finally: net.close(hb)
            h = net.open(leaks=[(pipe, 0.5)] + [(p_, z_) for p_, z_, _ in bg], scada=True); G = h["G"]
            try:
                ia, ib, nd = G.getLinkIndex(f"{pipe}_A"), G.getLinkIndex(f"{pipe}_B"), G.getNodeIndex(f"LEAK_{pipe}")
                L = net.length[pipe]; a, b = net.ends[pipe]
                for q in qs:
                    _, pos, coef, i0, i1, _ = q
                    G.setLinkLength(ia, L * pos); G.setLinkLength(ib, L * (1 - pos))
                    G.setNodeElevations(nd, net.elev[a] + pos * (net.elev[b] - net.elev[a]))
                    sched = {pipe: [(coef, 0.0, 1e6)], **{p_: [(c_, 0.0, 1e6)] for p_, _, c_ in bg}}
                    P = run_window(h, i0, i1, sched)
                    cache[q] = (P - (base[(bg, i0, i1)] if bg else Pexp[i0:i1])).astype(np.float32)
            finally:
                net.close(h)

    def observed(tk):
        i0, i1, b0 = tk["win"]; idx = tk["idx"]; sc = scale[idx]; base = tk.get("base")
        Rw = R[i0:i1][:, idx].astype(float)
        if b0 is not None and b0 < i0: Rw = Rw - R[b0:i0][:, idx].astype(float).mean(0)
        elif base: Rw = Rw - R[base[0]:base[1]][:, idx].astype(float).mean(0)
        Sw = S[i0:i1][:, idx] / sc; return _proj(Rw / sc, Sw), Sw, sc

    def score(tk, q, xu, Sw, sc):
        f = _proj(cache[q][:, tk["idx"]] / sc, Sw).mean(0); x = xu.mean(0)
        W = Wn.get(tk["n"]) if Wn else None
        if W is not None: f, x = W @ f, W @ x            # whitened: sensors with correlated noise count less
        return float(f @ x / (np.linalg.norm(f) * np.linalg.norm(x) + 1e-12))

    Z1 = [(a + b) / 2 for a, b in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:])]
    Z2 = sorted({round(a + (b - a) * f, 4) for a, b in zip(ZONE_EDGES[:-1], ZONE_EDGES[1:]) for f in (0.25, 0.75)})
    out, todo = {}, sorted(tasks, key=lambda x: (x["n"], x["win"][0]))
    joint = cfg.get("REFINE_JOINT", True)

    def background(tk):
        """Leaks of the same alarm (same day, same number of sensors) found earlier -> simulated together."""
        if not joint or tk["win"][2] is None: return ()
        k_ = tk["key"]
        prev = [x for x in tasks if x["key"][1:3] == k_[1:3] and x["key"][4] == k_[4] and x["win"][0] < tk["win"][0]]
        return tuple(sorted((out[x["key"]]["pipe"], round(out[x["key"]]["pos"], 3), out[x["key"]]["coef"]) for x in prev))

    def ready(tk):
        k_ = tk["key"]
        return not joint or tk["win"][2] is None or all(x["key"] in out for x in tasks if x["key"][1:3] == k_[1:3]
                                                        and x["key"][4] == k_[4] and x["win"][0] < tk["win"][0])

    while todo:                       # first the leaks without running leaks, then the next leaks of each alarm, ...
        batch = [tk for tk in todo if ready(tk)] or todo[:1]
        todo = [tk for tk in todo if tk not in batch]
        bgs = {id(tk): background(tk) for tk in batch}
        cand = lambda tk, ps: [p for p in ps if p not in {b_[0] for b_ in bgs[id(tk)]}]
        simulate({(p, z, cmid, *tk["win"][:2], bgs[id(tk)]) for tk in batch for p in cand(tk, tk["top"]) for z in Z1})
        stage2 = []
        for tk in batch:
            xu, Sw, sc = observed(tk); i0, i1 = tk["win"][:2]; bg = bgs[id(tk)]
            s1 = sorted(((score(tk, (p, z, cmid, i0, i1, bg), xu, Sw, sc), p, z) for p in cand(tk, tk["top"]) for z in Z1), reverse=True)
            if not s1: continue
            f = _proj(cache[(s1[0][1], s1[0][2], cmid, i0, i1, bg)][:, tk["idx"]] / sc, Sw)
            amp = float((f * xu).sum() / ((f * f).sum() + 1e-12))                    # measured / simulated size
            cest = float(np.clip(cmid * amp, 0.5 * crange[0], 1.5 * crange[1])); cest = float(f"{cest:.2g}")
            stage2.append((tk, list(dict.fromkeys(p for _, p, _ in s1))[: cfg["REFINE_FINAL_PIPES"]], cest))
        simulate({(p, z, c, *tk["win"][:2], bgs[id(tk)]) for tk, best, c in stage2 for p in best for z in Z2})
        for tk, best, c in stage2:
            xu, Sw, sc = observed(tk); i0, i1 = tk["win"][:2]; bg = bgs[id(tk)]
            s2 = sorted(((score(tk, (p, z, c, i0, i1, bg), xu, Sw, sc), p, z) for p in best for z in Z2), reverse=True)
            sim_, pipe, pos = s2[0]
            # relative zone confidence on the chosen pipe: misfit (1 - similarity) of the best point of every zone
            mis = np.array([min((1 - s for s, p, z in s2 if p == pipe and zone_index(z) == k), default=np.nan) for k in range(len(Z1))])
            mis = np.where(np.isnan(mis), np.nanmax(mis) * 4 + 1e-9, mis); m0 = max(float(mis.min()), 1e-9)
            zc = np.exp(-3.0 * (mis / m0 - 1)); zc = zc / zc.sum()
            out[tk["key"]] = dict(pipe=pipe, pos=float(pos), zone_conf=zc, similarity=sim_, coef=c, refined=True,
                                  candidates=", ".join(dict.fromkeys(p for _, p, _ in s2[:6])))
    return out


def run_parallel(func, jobs, cfg, log, what):
    """Run jobs on several CPU cores (falls back to one core, e.g. in some Windows IDEs)."""
    if os.environ.get("SLURM_CPUS_PER_TASK"):            # on a Slurm cluster: exactly the CPUs the job was given
        cores = int(os.environ["SLURM_CPUS_PER_TASK"]) + 1
    else:
        try: cores = len(os.sched_getaffinity(0))        # Linux: CPUs this process may use
        except AttributeError: cores = os.cpu_count() or 2
    workers = cfg.get("WORKERS") or max(1, cores - 1)
    if os.name == "nt": workers = min(workers, 61)        # Windows allows at most 61 worker processes
    workers = min(workers, len(jobs))
    if workers > 1 and len(jobs) > 1:
        try:
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing as _mp
            ctx = _mp.get_context("fork") if os.name == "posix" and sys.platform != "darwin" else None
            with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
                out = list(ex.map(func, jobs, chunksize=max(1, len(jobs) // (8 * workers))))
            log(f"    {len(jobs)} {what} simulated on {workers} CPU cores")
            return out
        except Exception as e:
            log(f"    parallel run not possible here ({type(e).__name__}: {str(e)[:200]}) -> one CPU core "
                f"(tip: set WORKERS in the SETTINGS to a smaller number, e.g. 8)")
    out = [func(j) for j in jobs]; log(f"    {len(jobs)} {what} simulated"); return out


def node_distances(net):
    """Shortest distance along the pipes between every pair of nodes (for the localisation error)."""
    nodes = sorted({n for p in net.pipes_all + net.pumps + net.valves for n in net.ends[p]}); ix = {n: i for i, n in enumerate(nodes)}
    D = np.full((len(nodes), len(nodes)), np.inf)
    for n in nodes:
        for m, d in net._dijkstra(n).items():
            if m in ix: D[ix[n], ix[m]] = d
    return D, ix


def pair_errors(net, D, ix, pa, xa, pb, xb):
    """Matrix of localisation errors between leak points (pa, xa) [rows] and (pb, xb) [columns]."""
    La = np.array([net.length[p] for p in pa]); Lb = np.array([net.length[p] for p in pb])
    a0 = np.array([ix[net.ends[p][0]] for p in pa]); a1 = np.array([ix[net.ends[p][1]] for p in pa])
    b0 = np.array([ix[net.ends[p][0]] for p in pb]); b1 = np.array([ix[net.ends[p][1]] for p in pb])
    da = [np.asarray(xa) * La, (1 - np.asarray(xa)) * La]; db = [np.asarray(xb) * Lb, (1 - np.asarray(xb)) * Lb]
    best = np.full((len(pa), len(pb)), np.inf)
    for ea, na in ((da[0], a0), (da[1], a1)):
        for eb, nb in ((db[0], b0), (db[1], b1)):
            best = np.minimum(best, ea[:, None] + D[np.ix_(na, nb)] + eb[None, :])
    same = np.asarray(pa)[:, None] == np.asarray(pb)[None, :]
    return np.where(same, np.abs(np.asarray(xa)[:, None] - np.asarray(xb)[None, :]) * La[:, None], best).astype(np.float32)


def select_sensors(net, store, meta, sizes, scale, log):
    """SENSOR PLACEMENT by backward elimination.  Start with a sensor at every junction; repeatedly remove the
    sensor whose removal makes the localisation of the training leaks worst the least (test: every leak at a
    held-out position is matched with the most similar leak of the other positions; score = mean distance along
    the pipes between the true and the matched leak).  The sets are nested: the 23 sensors are part of the 46, ..."""
    rb = np.stack([r["R"].mean(0) for r in store]) / scale; sb = np.stack([r["S"].mean(0) for r in store]) / scale
    va = split_by_position(meta.reset_index(drop=True)); tr = ~va
    if not va.any() or not tr.any():          # very few positions per pipe: every 5th leak is held out instead
        va = np.zeros(len(meta), bool); va[2::5] = True; tr = ~va
    cap = globals().get("CFG", {}).get("SENSOR_SELECT_MAX")
    if cap:                                   # big networks: a fixed random subsample of the leaks (fast, little memory)
        rs = np.random.default_rng(12345)
        for m_, k_ in ((va, cap[0]), (tr, cap[1])):
            on = np.where(m_)[0]
            if len(on) > k_: m_[rs.choice(on, len(on) - k_, replace=False)] = False
    D, ix = node_distances(net)
    E = pair_errors(net, D, ix, meta.Pipe[va].tolist(), meta.Position[va].to_numpy(), meta.Pipe[tr].tolist(), meta.Position[tr].to_numpy())

    def score(idx):
        r, s = rb[:, idx], sb[:, idx]
        f = r - ((r * s).sum(1) / ((s * s).sum(1) + 1e-12))[:, None] * s
        f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
        nn = np.argmax(f[va] @ f[tr].T, axis=1)
        e = E[np.arange(len(nn)), nn]
        return float(np.mean(np.where(np.isfinite(e), e, np.nanmax(np.where(np.isfinite(E), E, np.nan)))))   # safety for unconnected parts
    keep = list(range(len(net.junctions))); sets = {len(keep): list(keep)}; targets = sorted(sizes, reverse=True)
    while len(keep) > min(targets):
        nxt = max(t for t in targets if t < len(keep))
        n_rm = min(len(keep) - nxt, max(1, len(keep) // 20))            # big networks: a few sensors per round
        sc = [score([k for k in keep if k != c]) for c in keep]
        for c in [keep[i] for i in np.argsort(sc)[:n_rm]]: keep.remove(c)
        if len(keep) in targets:
            sets[len(keep)] = list(keep)
            log(f"    {len(keep):4d} sensors: mean matching error of the training leaks {score(keep):8.1f} {net.len_unit}")
    return {n: [net.junctions[i] for i in sorted(sets[n])] for n in sorted(sets, reverse=True)}


SETTING_NAMES = {"factor": "Alarm threshold factor", "jump": "New-leak jump size", "rel": "New-leak jump share",
                 "swap": "Pattern-change similarity", "grace": "Wait after an event (readings)", "confirm": "Readings to confirm",
                 "clear": "Normal readings to end an alarm"}


def sensor_counts(n_junctions, cfg):
    """All junctions, then halve (rounded up) until MIN_SENSORS: 92 -> 46 -> 23 -> 12 -> 6 -> 3,  9 -> 5 -> 3."""
    out = [n_junctions]
    while out[-1] > cfg["MIN_SENSORS"]:
        out.append(max(cfg["MIN_SENSORS"], int(np.ceil(out[-1] / 2))))
    return out


def fingerprints(store, idx, scale, cfg, seed):
    """Fingerprints of the training leaks seen by the sensors idx (+ noisy copies)."""
    rng = np.random.default_rng(seed); X, rows = [], []
    for r in store:
        Rw, Sw, sc = r["R"][:, idx], r["S"][:, idx], scale[idx]
        for a in range(cfg["NOISE_COPIES"] + 1):
            Rn = Rw + rng.normal(0, cfg["PRESSURE_NOISE"], Rw.shape) if a else Rw
            X.append(signature(Rn / sc, 0, len(Rn), Sw / sc))
            rows.append(dict(Pipe=r["Pipe"], Position=r["Position"], Zone=r["Zone"], Coefficient=r["Coefficient"],
                             Start=r["Start"], End=r["End"], Demand_multiplier=r["Demand_multiplier"], Noise_copy=a))
    return np.array(X, np.float32), pd.DataFrame(rows)


def random_start(rng, lo, hi):
    return float(np.floor(rng.uniform(lo, hi) * 60) / 60)            # any minute (readings are every 15 min)


def build_test_days(net, cfg, n_single, crange, rng, source=None):
    """Days of single leaks (random pipe, position, size, start minute, duration), leak-free days and
    days with 2-3 OVERLAPPING leaks on different pipes that start at different times.
    source(rng, exclude) -> (pipe, position, coefficient): where the leaks come from (None = anywhere: test days;
    the training / validation days take their leaks from the training / validation leak positions).  The SAME
    construction is used for the training, validation and test days."""
    days, pos_ = [], lambda: round(float(rng.uniform(0.02, 0.98)), 3)
    if source is None:
        source = lambda rng_, ex: (str(rng_.choice([p for p in net.pipes if p not in ex])), pos_(), round(float(rng_.uniform(*crange)), 3))
    for k in range(n_single):
        d = float(rng.choice(cfg["TEST_DURATIONS_H"])); on = random_start(rng, 0.25, 24 - d - 0.25)
        p, z, c = source(rng, set())
        days.append(dict(kind="single", name=f"Single leak {k + 1}", leaks=[dict(pipe=p, position=z, coefficient=c, on=on, off=on + d)]))
    for k in range(cfg["N_NO_LEAK_DAYS"]):
        days.append(dict(kind="none", name=f"No-leak day {k + 1}", leaks=[]))
    n_ov = int(round(cfg.get("OVERLAP_SHARE", 1.0) * cfg["N_MULTI_LEAK_DAYS"]))
    for k in range(cfg["N_MULTI_LEAK_DAYS"]):
        n = int(rng.choice(cfg["LEAKS_PER_MULTI_DAY"])); ov = k < n_ov
        t, leaks = 0.0, []
        for j in range(n):
            d = float(rng.choice(cfg["TEST_DURATIONS_H"]))                          # SAME durations as single leaks
            if ov:                                                                  # OVERLAPPING: next leak starts
                if j:                                                               # while the previous one is running
                    pv = leaks[-1]; t = pv["on"] + max(0.25, round(float(rng.uniform(*cfg["OVERLAP_START_FRAC"])) * (pv["off"] - pv["on"]) * 60) / 60)
            else:                                                                   # NON-OVERLAPPING: next leak starts
                if j: t = leaks[-1]["off"] + random_start(rng, *cfg["SEQ_GAP_H"])   # after the previous one has ended
            if t >= 20.0: break
            p, z, c = source(rng, {l_["pipe"] for l_ in leaks})                   # always a different pipe
            leaks.append(dict(pipe=p, position=z, coefficient=c, on=t, off=t + d))
        # SAME start-time distribution as single leaks: the whole sequence is shifted to a random time of the day
        # (night = low, quiet demand vs day = high, noisy demand changes how easily a leak is seen)
        span = leaks[-1]["off"] - leaks[0]["on"]
        if span < 23.25:
            sh = random_start(rng, 0.25, 23.75 - span) - leaks[0]["on"]
            for l_ in leaks: l_["on"] += sh; l_["off"] += sh
        for l_ in leaks: l_["off"] = min(l_["off"], 23.75)
        leaks = [l_ for l_ in leaks if l_["on"] < 23.5]
        days.append(dict(kind="multi", overlap=ov, name=f"Multi-leak day {k + 1} ({len(leaks)} leaks, " +
                         ("overlapping" if ov else "one after another") + ")", leaks=leaks))
    return days


def analyse_day(det, t, Rp, leaks, s, scores=None):
    """Real-time monitoring of one day; every leak found by the detector is matched with the true leak
    that started just before it.  Returns per-leak results, FALSE alarms (no leak running), DUPLICATE alarms
    (an already detected leak reported again) and the found leaks."""
    comb, alarms = det.monitor(t, Rp, scores, s["factor"], s["confirm"], s["clear"], s["jump"], s["rel"], s["swap"], s["grace"])
    W = max(det.windows); span = [(int(np.ceil(l["on"] * 4 - 1e-9)), int(np.ceil(l["off"] * 4 - 1e-9))) for l in leaks]
    found = sorted([dict(L, alarm=a) for a in alarms for L in a["leaks"]], key=lambda L: L["i_det"])
    got, false, dup = [None] * len(leaks), [], []
    for L in found:
        cand = [n for n, (a, b) in enumerate(span) if got[n] is None and a <= L["i_det"] <= b + W - 1]
        if not cand:
            (dup if any(a <= L["i_det"] <= b + W - 1 for a, b in span) else false).append(L); continue
        if L["how"] != "new alarm":                   # a new pattern belongs to the leak(s) that started last
            m = max(span[n][0] for n in cand); cand = [n for n in cand if span[n][0] == m]
        for k, n in enumerate(sorted(cand, key=lambda n: span[n][0])): got[n] = (L, k > 0)
    return comb, got, false, dup, found


def tune_detector(det, tdays, idx, sc, grid, base):
    """Detector settings chosen on separate TUNING days (random leaks, never used for testing): every combination
    of the grid is tried; best = most leaks found (single leaks weighted double) with NO alarm on the leak-free
    tuning days; false / duplicate alarms on leak days only break ties.  Returns (best settings, table of all combinations)."""
    import itertools
    pre = []
    for d, sim in tdays:
        Rp = proj_demand(sim["R"][:, idx], sim["S"][:, idx], sc); pre.append((d, sim["t"], Rp, det.online_scores(Rp)))
    rows = []
    for vals in itertools.product(*grid.values()):
        s = dict(base, **dict(zip(grid, vals))); hit = n = fp = fp_free = hit1 = n1 = 0; delays = []
        for d, t, Rp, sco in pre:
            _, got, false, dup, _ = analyse_day(det, t, Rp, d["leaks"], s, sco)
            n += len(d["leaks"]); hit += sum(g is not None for g in got); fp += len(false) + len(dup)
            if d["kind"] == "single": n1 += 1; hit1 += got[0] is not None
            fp_free += len(false) if not d["leaks"] else 0
            delays += [(t[g[0]["i_det"]] - l["on"]) * 60 for l, g in zip(d["leaks"], got) if g is not None]
        prec = hit / (hit + fp) if hit + fp else 0.0; rec = hit / n if n else 0.0; f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rec1 = hit1 / n1 if n1 else 0.0
        # DETECTION FIRST: most leaks found (single leaks count double), NO alarm on the leak-free tuning days;
        # false / duplicate alarms on leak days only as a small tie-breaker
        score = (2 * rec1 + rec) / 3 - (1.0 if fp_free else 0.0) - 0.002 * fp
        rows.append(dict(s, **{"Leaks found": f"{hit}/{n}", "Single leaks found": f"{hit1}/{n1}", "False + duplicate alarms": fp,
                               "Alarms on leak-free days": fp_free, "F1": round(f1, 3),
                               "Median delay (min)": float(np.median(delays)) if delays else np.nan, "Score": round(score, 4)}))
    tab = pd.DataFrame(rows).sort_values(["Score", "Median delay (min)"], ascending=[False, True]).reset_index(drop=True)
    return {k: (type(base[k])(tab.loc[0, k]) if k in base else tab.loc[0, k]) for k in base}, tab


def leak_window(L, n_steps, max_readings, margin=0):
    """Readings used to localise a detected leak: from its detection until the next event (another leak
    detected / a leak ended) or max_readings; the readings before it (other leaks already running) = baseline.
    margin: stop this many readings BEFORE the next leak is reported (that leak usually started a little earlier than it
    was reported), as long as at least 3 readings remain."""
    ev = sorted({x["i_det"] for x in L["alarm"]["leaks"]} | {x["i_end"] for x in L["alarm"]["leaks"] if x["i_end"] is not None})
    nxt = [e for e in ev if e > L["i_det"]]
    if nxt and margin and nxt[0] - margin >= L["i_det"] + 3: nxt = [nxt[0] - margin]
    i1 = min([n_steps, L["i_det"] + max_readings] + nxt[:1]); prev = [e for e in ev if e < L["i_det"]]
    b0 = max(prev[-1] if prev else L["i_det"] - 1, L["i_det"] - 4) if L["how"] != "new alarm" else None
    return L["i_det"], max(i1, L["i_det"] + 1), b0


def pre_baseline(R, S, sc, i0, i1, b0, ratio):
    """A leak reported as a NEW alarm may still overlap an earlier leak that was never reported (or whose alarm already
    ended).  If the readings 1-2 h before the detection already show a clear pressure change (> ratio x the change in the
    leak window), that earlier change is used as baseline and removed.  Returns (a, b) or None."""
    if not ratio or b0 is not None or i0 < 8: return None
    a, b = i0 - 8, i0 - 4
    mb = _proj(R[a:b].astype(float) / sc, S[a:b] / sc).mean(0); ml = _proj(R[i0:i1].astype(float) / sc, S[i0:i1] / sc).mean(0)
    return (a, b) if np.linalg.norm(mb) > ratio * np.linalg.norm(ml) else None


def localise(loc, R, S, sc, L, n_steps, max_readings, top_k=0, refine_readings=None, pre_ratio=None, margin=0):
    i0, i1, b0 = leak_window(L, n_steps, max_readings, margin)
    Rw = R[i0:i1].astype(float); base = pre_baseline(R, S, sc, i0, i1, b0, pre_ratio)
    if b0 is not None and b0 < i0: Rw = Rw - R[b0:i0].astype(float).mean(0)        # remove the leaks already running
    elif base: Rw = Rw - R[base[0]:base[1]].astype(float).mean(0)                   # remove an unreported earlier leak
    x = signature(Rw / sc, 0, len(Rw), S[i0:i1] / sc); res = loc.locate(x); res["win"] = (i0, i1, b0)
    if top_k:              # most likely pipes (best similarity of their reference leaks) -> candidates for the refinement
        sim = loc.Z @ loc.embed(x)[0]
        res["top"] = list(pd.Series(sim).groupby(loc.pipes).max().sort_values(ascending=False).index[:top_k])
        res["win_ref"] = leak_window(L, n_steps, refine_readings or max_readings, margin)
        res["base_ref"] = pre_baseline(R, S, sc, *res["win_ref"], pre_ratio)
    return res


def summarise(rows, n_sensors, net):
    """Performance numbers of one sensor set."""
    df = pd.DataFrame(rows); s = df[df.Kind == "single"]; m = df[df.Kind == "multi"]; z = df[df.Kind == "none"]
    pct = lambda x: float(100 * np.mean(x)) if len(x) else np.nan
    det = lambda g: g.Detected == "YES"
    out = {"Sensors": n_sensors,
           "Single: detected (%)": pct(det(s)), "Single: median delay (min)": float(s.loc[det(s), "Delay (min)"].median()) if det(s).any() else np.nan,
           "Single: correct pipe (%)": pct(s["Pipe correct"] == "YES"),
           "Single: correct or neighbouring pipe (%)": pct(s["Pipe or neighbour"] == "YES"),
           "Single: correct pipe and zone (%)": pct(s["Zone correct"] == "YES"),
           f"Single: median error ({net.len_unit})": float(s["Error"].median()) if s["Error"].notna().any() else np.nan,
           "Multi: leaks detected (%)": pct(det(m)), "Multi: hidden by other leaks (%)": pct(m.Detected == "NO (hidden by other leaks)"),
           "Multi: correct pipe (%)": pct(m["Pipe correct"] == "YES"),
           "Multi: correct or neighbouring pipe (%)": pct(m["Pipe or neighbour"] == "YES"),
           f"Multi: median error ({net.len_unit})": float(m["Error"].median()) if m["Error"].notna().any() else np.nan,
           "No-leak days with a false alarm": f"{int((z['False alarms'] > 0).sum())} / {len(z)}",
           "False alarms on leak days (no leak running)": int(df.loc[df.Kind != "none", "False alarms"].sum()),
           "Duplicate alarms (leak reported again)": int(df["Duplicate alarms"].sum())}
    order = {}
    for k in sorted(m["Leak no."].dropna().unique()):
        g = m[m["Leak no."] == k]
        order[int(k)] = dict(detected=pct(det(g)), neighbour=pct(g["Pipe or neighbour"] == "YES"), pipe=pct(g["Pipe correct"] == "YES"), n=len(g))
    size = {}
    for k in sorted(m["Leaks that day"].dropna().unique()):
        g = m[m["Leaks that day"] == k]
        size[int(k)] = dict(detected=pct(det(g)), neighbour=pct(g["Pipe or neighbour"] == "YES"), pipe=pct(g["Pipe correct"] == "YES"), n=len(g))
    return out, order, size


MULTI_LABEL = "Multiple leaks (different pipes)"
COL_PIPE = "Leaks detected - correct pipe (%)"
COL_ZONE = "Leaks localised - correct pipe + zone (%)"
COL_RET = "Accuracy retained after removing sensors (% of all-sensor result)"
COL_CHG = "Change after halving the sensors (points)"


def results_summary(results, counts):
    """THE SUMMARY TABLE: sensors x leak type x leak pipe x where the leak was induced.
    Leaks detected - correct pipe = detected AND on the correct pipe;  leaks localised = also in the correct 20 % zone
    (share of ALL test leaks: an undetected leak counts as wrong).  No leak: share of leak-free days without an alarm."""
    r = results.copy()
    r["Induced zone"] = [zone(p / 100) if p == p else "-" for p in r.get("Position (%)", pd.Series(np.nan, index=r.index))]
    r["_pipe"] = r.get("Leak pipe", pd.Series("-", index=r.index)).astype(str).str.replace(r"\.0$", "", regex=True)
    rows = []
    if "Overlap" not in r: r["Overlap"] = None
    for kind, ov, label in (("single", None, "Single leak"), ("multi", True, "Multiple leaks, overlapping (different pipes)"),
                            ("multi", False, "Multiple leaks, non-overlapping (different pipes)"), ("none", None, "No leak (correct = no alarm)")):
        groups = [("All pipes", None, "All positions", None)]
        if kind != "none":
            groups += [("All pipes", None, f"{z} of the pipe", z) for z in zone_labels()]
            pipes = r.loc[(r.Kind == kind) & ((r.Overlap == ov) if ov is not None else True), "_pipe"].unique()
            groups += [(f"Pipe {q}", q, "All positions", None) for q in sorted(pipes, key=lambda x: (len(x), x))]
        for pname, q, gname, z in groups:
            prev, full = None, None
            for n in counts:
                g = r[(r.Sensors == n) & (r.Kind == kind)]
                if ov is not None: g = g[g.Overlap == ov]
                if q is not None: g = g[g._pipe == q]
                if z is not None: g = g[g["Induced zone"] == z]
                if not len(g): continue
                if kind == "none":
                    pc = np.nan; ok = 100.0 * float((g["False alarms"] == 0).mean())
                else:
                    pc = 100.0 * float((g["Pipe correct"] == "YES").mean()); ok = 100.0 * float((g["Zone correct"] == "YES").mean())
                full = ok if full is None else full
                rows.append({"Sensors": n, "Leak type": label, "Leak pipe": pname if kind != "none" else "-",
                             "Leak induced at (point of the pipe)": gname if kind != "none" else "-", "Test cases": len(g),
                             COL_PIPE: round(pc, 1) if pc == pc else "-", COL_ZONE: round(ok, 1),
                             COL_RET: int(round(100.0 * ok / full)) if full > 0 else "-",
                             COL_CHG: "-" if prev is None else round(ok - prev, 1)})
                prev = ok
    return pd.DataFrame(rows)


ERR_CATS = ["correct pipe + correct zone", "correct pipe, next zone", "correct pipe, zone further away",
            "neighbouring pipe (same junction)", "wrong pipe (not neighbouring)", "not detected"]


def error_breakdown(results, counts):
    """WHY a leak is not a 'correct result': every test leak falls in exactly one category (shares add up to 100 %)."""
    r = results[results.Kind != "none"].copy()
    tz = np.clip((r["Position (%)"] // 20).astype(float), 0, 4)
    pz = r["Most likely zone"].astype(str).str.extract(r"^([0-9]+)-")[0].astype(float) // 20
    cat = np.select([r.Detected != "YES", (r["Pipe correct"] != "YES") & (r["Pipe or neighbour"] == "YES"), r["Pipe correct"] != "YES",
                     r["Zone correct"] == "YES", (tz - pz).abs() == 1], [ERR_CATS[5], ERR_CATS[3], ERR_CATS[4], ERR_CATS[0], ERR_CATS[1]], ERR_CATS[2])
    r["Category"] = cat
    out = []
    for kind, label in (("single", "Single leak"), ("multi", "Multiple leaks")):
        for n in counts:
            g = r[(r.Kind == kind) & (r.Sensors == n)]
            if len(g): out.append({"Leak type": label, "Sensors": n, **{c: round(100.0 * float((g.Category == c).mean()), 1) for c in ERR_CATS}})
    return pd.DataFrame(out), r


def tank_distance(net):
    """Number of pipes between every pipe and the nearest tank / reservoir (0 = the pipe touches it)."""
    fixed = set(net.xy) - set(net.junctions)
    links = [l for l in net.ends if l in net.pipes_all or l in net.pumps or l in getattr(net, "valves", [])]
    hop, front = {n: 0 for n in fixed}, list(fixed)
    while front:
        nxt = []
        for n in front:
            for l in links:
                a, b = net.ends[l]
                for x, y in ((a, b), (b, a)):
                    if x == n and y not in hop: hop[y] = hop[n] + 1; nxt.append(y)
        front = nxt
    return {p: min(hop.get(net.ends[p][0], 99), hop.get(net.ends[p][1], 99)) for p in net.pipes_all}


def missed_by_size(results, counts, net=None):
    """Detection rate per leak-size quarter (emitter coefficient) and per distance from a tank / reservoir:
    shows WHICH leaks are missed (small leaks / leaks on pipes next to a tank or reservoir)."""
    r = results[results.Kind != "none"].copy()
    if "Emitter coef" not in r or not len(r): return pd.DataFrame()
    r["Size"] = pd.qcut(r["Emitter coef"].astype(float), 4, labels=["smallest 25 %", "25-50 %", "50-75 %", "largest 25 %"])
    det = lambda g: round(100.0 * float((g == "YES").mean()), 1)
    t = r.groupby(["Sensors", "Size"], observed=True).Detected.apply(det).unstack()
    t = t.reindex(counts); t.columns = [f"Detected (%) - {c} of the leaks" for c in t.columns]
    if net is not None:
        td = tank_distance(net)
        h = r["Leak pipe"].map(lambda q: td.get(str(q).split(".")[0] if str(q).endswith(".0") else str(q), 99))
        r["Where"] = np.select([h == 0, h == 1], ["pipe touching a tank / reservoir", "1 pipe away from it"], "2+ pipes away")
        w = r.groupby(["Sensors", "Where"]).Detected.apply(det).unstack().reindex(counts)
        w = w[[c for c in ("pipe touching a tank / reservoir", "1 pipe away from it", "2+ pipes away") if c in w.columns]]
        n_ = r[r.Sensors == counts[0]].Where.value_counts()
        w.columns = [f"Detected (%) - leak on {c} ({n_.get(c, 0)} leaks)" for c in w.columns]
        t = t.join(w)
    return t.reset_index()


def why_text(eb, cats, msz, counts, u):
    """Short bullet points: why a leak is not localised correctly."""
    out = []
    for kind in ("Single leak", "Multiple leaks"):
        e = eb[eb["Leak type"] == kind].set_index("Sensors")
        if not len(e): continue
        f, l = e.loc[counts[0]], e.loc[counts[-1]]
        out.append(f"<b>{kind if kind == 'Single leak' else MULTI_LABEL}, {counts[0]} sensors:</b> {f[ERR_CATS[0]]:.0f} % correct pipe + zone, "
                   f"{f[ERR_CATS[1]]:.0f} % next zone, {f[ERR_CATS[3]]:.0f} % neighbouring pipe, {f[ERR_CATS[4]]:.0f} % wrong pipe, "
                   f"{f[ERR_CATS[5]]:.0f} % not detected")
        out.append(f"<b>Removing sensors</b> mainly adds wrong pipes: {f[ERR_CATS[4]]:.0f} % &rarr; {l[ERR_CATS[4]]:.0f} % "
                   f"({counts[0]} &rarr; {counts[-1]} sensors, {kind.lower()})")
    s = cats[(cats.Kind == "single") & (cats.Sensors == counts[0])]
    if len(s):
        nb = s[s.Category == ERR_CATS[3]]; ends = lambda g: float(np.minimum(g["Position (%)"], 100 - g["Position (%)"]).median())
        if len(nb):
            out.append(f"<b>Neighbouring pipe</b> = leak close to a junction (median {ends(nb):.0f} % of the pipe length from it): "
                       "it looks like a leak on the other pipe at that junction")
    if len(msz):
        m = msz.set_index("Sensors").loc[counts[0]]
        sz = [c for c in m.index if "of the leaks" in c]; t0 = [c for c in m.index if "touching" in c]; far = [c for c in m.index if "2+" in c]
        out.append(f"<b>Not detected:</b> smallest leaks {100 - m[sz[0]]:.0f} % missed vs largest {100 - m[sz[-1]]:.0f} %" +
                   (f"; leaks on a pipe touching a tank / reservoir {100 - m[t0[0]]:.0f} % missed (pressure held by the tank) vs "
                    f"{100 - m[far[0]]:.0f} % elsewhere" if t0 and far and m[t0[0]] == m[t0[0]] else ""))
    out.append("<b>Zone</b> = the hardest part: the 5 zones of a pipe give almost the same pressure pattern")
    return out


def _set_metrics(L, noleak_flags, u):
    """Metrics of one set of leaks (rows of the result table) + its leak-free days (list of True = false alarm)."""
    pc = lambda x: round(100.0 * float(np.mean(x)), 1) if len(x) else "-"
    d = L[L.Detected == "YES"] if len(L) else L
    ec = [c for c in L.columns if c.startswith("Error")] if len(L) else []
    m = {"Leaks evaluated": len(L), "Leaks detected (%)": pc(L.Detected == "YES") if len(L) else "-",
         "Accuracy - pipe (%), detected leaks": pc(d["Pipe correct"] == "YES") if len(d) else "-",
         "Accuracy - zone (correct pipe + zone, %), detected leaks": pc(d["Zone correct"] == "YES") if len(d) else "-",
         "Correct or neighbouring pipe (%), detected leaks": pc(d["Pipe or neighbour"] == "YES") if len(d) else "-",
         f"Median localisation error ({u}), detected leaks": round(float(d[ec[0]].median()), 1) if len(d) and ec else "-"}
    if len(d) and "Enc-dec pipe correct" in d:
        m.update({"Encoder-decoder alone - pipe (%)": pc(d["Enc-dec pipe correct"] == "YES"),
                  "Encoder-decoder alone - zone (%)": pc(d["Enc-dec zone correct"] == "YES")})
    m.update({"Leak-free days": len(noleak_flags), "Leak-free days with a false alarm": int(np.sum(noleak_flags)) if len(noleak_flags) else 0})
    return m


def _set_rows(results, rows_ev, noleak, n):
    """(leak rows, leak-free-day flags) of the training / validation / test set for n sensors."""
    split = lambda g: (g[g.Kind != "none"], list(g[g.Kind == "none"]["False alarms"] > 0))
    ev = lambda k: rows_ev[(rows_ev.Sensors == n) & (rows_ev.Set == k)]
    return {"training": split(ev("training")), "validation": split(ev("validation")), "test": split(results[results.Sensors == n])}


def accuracy_tables(models, results, rows_ev, noleak, counts, u):
    """THREE SIMPLE TABLES (training / validation / test; rows = metric, columns = number of sensors): ONE accuracy over ALL
    leaks of the set, every value from the complete method (detection -> encoder-decoder -> hydraulic check); accuracy =
    share of the DETECTED leaks localised on the correct pipe / correct pipe + zone.  + the three FULL reports."""
    col = lambda n: f"{n} sensors"
    simple, full = {"training": {}, "validation": {}, "test": {}}, {"training": {}, "validation": {}, "test": {}}
    pc = lambda x: round(100.0 * float(np.mean(x)), 1) if len(x) else "-"
    for n in counts:
        rep = dict(zip(models[n]["report"].Metric, models[n]["report"].Value))
        nf = str(rep.get("Training fingerprints / validation fingerprints", "- / -")).split(" / ")
        g = results[results.Sensors == n]
        for k, (L, nl) in _set_rows(results, rows_ev, noleak, n).items():
            m = _set_metrics(L, nl, u); K = k.capitalize()
            nd = {"training": "Training", "validation": "Validation", "test": "Test"}[k]
            first = (f"{nd} leaks evaluated (single + multiple leaks; + {len(nl)} leak-free days)", len(L))
            simple[k][col(n)] = {first[0]: first[1], f"{K} accuracy - pipe (%)": m["Accuracy - pipe (%), detected leaks"],
                                 f"{K} accuracy - zone (correct pipe + zone, %)": m["Accuracy - zone (correct pipe + zone, %), detected leaks"]}
            f = {first[0]: first[1]}
            if k in ("training", "validation"):
                f[f"{nd} fingerprints the encoder-decoder used (simulated leaks incl. noisy copies)"] = nf[0] if k == "training" else nf[-1]
            if k == "training":
                f.update({"Best epoch (early stopping)": rep.get("Best epoch (early stopping)", "-"),
                          "Classifier accuracy on the training fingerprints": rep.get("Classifier accuracy - training", "-")})
            if k == "validation":
                f.update({"Encoder-decoder on the validation fingerprints - pipe": rep.get("Validation: correct pipe", "-"),
                          "Encoder-decoder on the validation fingerprints - zone": rep.get("Validation: correct pipe AND correct zone", "-")})
            if True:
                gk = _set_rows(results, rows_ev, noleak, n)[k][0]
                for kind, ov_, lab in (("single", None, "Single leaks"), ("multi", True, "Multiple leaks, overlapping"),
                                       ("multi", False, "Multiple leaks, non-overlapping")):
                    q = gk[gk.Kind == kind]
                    if ov_ is not None and "Overlap" in q: q = q[q.Overlap == ov_]
                    qd = q[q.Detected == "YES"]
                    f.update({f"{lab}: leaks / detected (%)": f"{len(q)} / {pc(q.Detected == 'YES')}",
                              f"{lab}: pipe / zone accuracy (%)": f"{pc(qd['Pipe correct'] == 'YES')} / {pc(qd['Zone correct'] == 'YES')}"})
            f.update({f"Full method: {a_}": b_ for a_, b_ in m.items()})
            full[k][col(n)] = f
    mk = lambda dd: pd.DataFrame(dd).rename_axis("Metric").reset_index()
    return tuple(mk(simple[k]) for k in ("training", "validation", "test")) + tuple(mk(full[k]) for k in ("training", "validation", "test"))


def scenario_table(days_by_set):
    """Composition of each set: days and leaks per scenario type (single / multiple overlapping / multiple non-overlapping
    / leak-free) and number of days with 2 / 3 leaks."""
    rows = []
    kinds = (("Single leak", lambda d: d["kind"] == "single"),
             ("Multiple leaks, overlapping", lambda d: d["kind"] == "multi" and d.get("overlap")),
             ("Multiple leaks, non-overlapping", lambda d: d["kind"] == "multi" and not d.get("overlap")),
             ("Leak-free", lambda d: d["kind"] == "none"))
    for set_, days in days_by_set.items():
        for lab, f in kinds:
            ds = [d for d in days if f(d)]
            rows.append({"Set": set_, "Scenario": lab, "Days": len(ds), "Leaks": sum(len(d["leaks"]) for d in ds),
                         "Days with 2 leaks": sum(len(d["leaks"]) == 2 for d in ds) if "Multiple" in lab else "-",
                         "Days with 3 leaks": sum(len(d["leaks"]) == 3 for d in ds) if "Multiple" in lab else "-"})
        rows.append({"Set": set_, "Scenario": "TOTAL", "Days": len(days), "Leaks": sum(len(d["leaks"]) for d in days),
                     "Days with 2 leaks": "", "Days with 3 leaks": ""})
    return pd.DataFrame(rows)


def confusion_data(results, rows_ev, noleak, counts):
    """Confusion matrices per set (training / validation / test) and number of sensors:
      pipe: actual (leak / no leak) x reported (leak on the correct pipe / leak where there is none / no leak reported)
      zone: true zone x localised zone on the correct pipe (+ neighbouring pipe / wrong pipe / not detected)."""
    zl = zone_labels(); zw = int(round(100 * (ZONE_EDGES[1] - ZONE_EDGES[0]))); out = {}
    for n in counts:
        out[str(n)] = {}
        for k, (r, nl) in _set_rows(results, rows_ev, noleak, n).items():
            M = np.zeros((len(zl), len(zl) + 3), int)
            for _, x in r.iterrows():
                t = zone_index(x["Position (%)"] / 100)
                if x.Detected != "YES": c = len(zl) + 2
                elif x["Pipe correct"] != "YES": c = len(zl) if x["Pipe or neighbour"] == "YES" else len(zl) + 1
                else: c = int(re.match(r"([0-9]+)", str(x["Most likely zone"])).group(1)) // zw
                M[t, min(c, len(zl) + 2)] += 1
            det = (r.Detected == "YES").to_numpy() if len(r) else np.zeros(0, bool)
            okp = (r["Pipe correct"] == "YES").to_numpy() if len(r) else np.zeros(0, bool)
            P = [[int((det & okp).sum()), int((det & ~okp).sum()), int((~det).sum())],
                 [0, int(np.sum(nl)), int(len(nl) - np.sum(nl))]]
            out[str(n)][k] = dict(zone=M.tolist(), pipe=P)
    return out


def optimum(summary, share):
    """Smallest number of sensors that keeps at least `share` of the full-sensor performance in detection
    (single + multi), in finding the exact pipe (single) and the correct or a neighbouring pipe (single + multi)."""
    keys = ["Single: detected (%)", "Multi: leaks detected (%)", "Single: correct pipe (%)",
            "Single: correct or neighbouring pipe (%)", "Multi: correct or neighbouring pipe (%)"]
    full = summary.iloc[0]; ok = []
    for _, r in summary.iterrows():
        if all((r[k] >= share * full[k]) if full[k] == full[k] else True for k in keys): ok.append(int(r.Sensors))
    return min(ok) if ok else int(full.Sensors), keys


# =====================================================================================================
# SETTINGS  (edit here)
# =====================================================================================================
try:
    HERE = Path(__file__).resolve().parent
except NameError:                      # running inside Jupyter
    HERE = Path.cwd()
INPUT_FILE = HERE / "Net1.inp"         # <-- any EPANET network, e.g. HERE / "Net3.inp" or a full path r"C:\...\Net3.inp"
OUTPUT_DIR = HERE / f"{INPUT_FILE.stem}_pressure_sensor_study"     # all results are written here
ZONE_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)                         # 5 zones of 20 % along every pipe

CFG = dict(
    SEED=42,
    # ---- sensors: ONLY pressure at the junctions ----
    MIN_SENSORS=3,                   # all junctions -> half -> half ... -> 3 sensors
    PRESSURE_NOISE=0.01,             # pressure sensor noise (in the pressure unit of the INP file: psi or m)
    DEMAND_NOISE=0.02,               # unknown random demand fluctuation (2 %)
    # ---- leaks ----
    LEAK_SIZE_PCT=(2.0, 7.0),        # leak flow = 2 - 7 % of the average network demand (works for any network size)
    # ---- training (encoder-decoder) ----
    TRAINING_SIMULATIONS=4000,       # about this many EPANET leak simulations (spread over all pipes and zones)
    MIN_SIM_PER_PIPE=40,             # at least this many training leaks per pipe (big networks -> more simulations)
    MAX_TEST_DAYS=None,              # None = 20 % of all scenarios; a number caps the test days (big networks)
    SENSOR_SELECT_MAX=None,          # None = all training leaks for the sensor placement; (n_val, n_train) = random
                                     # subsample (big networks: keeps the placement fast and the memory small)
    TRAIN_DURATIONS_H=(1.0, 2.0, 3.0),
    DEMAND_MULTIPLIERS=(0.9, 1.0, 1.1),
    NOISE_COPIES=3,                  # extra copies of every fingerprint with sensor noise
    HIDDEN_DIM=64, LATENT_DIM=16,    # encoder width and bottleneck (embedding) size
    WEIGHT_DECAY=1e-3, MAX_EPOCHS=400, PATIENCE=40,
    CLASS_LABEL="pipe",              # classifier on the embedding predicts the pipe; the position comes from
    POSITION_METHOD="kernel_ridge",  # a kernel ridge regression per pipe (-> zone + zone confidence)
    POSITION_KRR=dict(features="embedding", gamma=2.0, alpha=1e-3),
    ZONE_CONFIDENCE=True,
    # ---- detection (Matrix Profile, real time, on the pressure changes with the demand effect removed) ----
    N_NORMAL_DAYS=48, N_REFERENCE_DAYS=8,    # leak-free days: 8 = what normal looks like, 40 = alarm threshold
    MP_WINDOWS=(1, 2, 4),                    # last 15, 30, 60 min
    DETECTION=dict(factor=1.5, jump=1.0, rel=0.3, swap=0.5, grace=0, confirm=1, clear=2),   # start values
    TUNE_GRID=dict(factor=(1.1, 1.2, 1.3, 1.5, 1.7), jump=(1.0, 1.5, 2.0), swap=(0.3, 0.5), grace=(0, 1, 2)),   # tried per
                                     # sensor set; lower factors = more sensitive (only chosen if no false alarm on the
                                     # leak-free tuning days)
    TUNING_DAYS=dict(single=120, multi=30, none=40),   # separate tuning days (random leaks, not the test days)
    LOCALISE_READINGS=8,             # readings (2 h) after a detection used to localise the leak
    # ---- localisation: encoder-decoder + MODEL-BASED REFINEMENT ----
    USE_GPU=True,                    # train the encoder-decoder on the GPU (NVIDIA + PyTorch) if there is one,
                                     # otherwise automatically on the CPU (NumPy); EPANET always runs on the CPU cores
    GPU_EXACT=True,                  # True: the GPU repeats the CPU computation exactly (float64, same random numbers)
                                     # -> SAME results as on the CPU; checked automatically before training (1 epoch
                                     # GPU vs CPU); False: float32, faster, results equivalent but not identical
    MODEL_REFINEMENT=True,           # True: the hydraulic model, run under the conditions measured that day (tank levels,
                                     # pump status), checks the most likely pipes of the encoder-decoder and picks the
                                     # pipe + point whose simulated pressure pattern fits the measurement best
    REFINE_TOP_PIPES=8,              # pipes of the encoder-decoder that are checked (5 points each) with ALL sensors;
    REFINE_TOP_MAX=8,                # > 8: more pipes with fewer sensors, 8 x sqrt(all / sensors) (tested on Net1: no gain)
    WINDOW_MARGIN=0,                 # end the window this many readings before the next leak is reported (tested: no gain)
    REFINE_FINAL_PIPES=3,            # best pipes checked again with 10 points and the estimated leak size
    EVAL_DAYS=dict(training="all", validation="all"),  # "all" = EVERY training / validation scenario is used (a number =
                                                       # random sample of that many days, faster); days built EXACTLY like the test days
                                                       # (same TEST_MIX: single / multiple / leak-free), leaks taken from
                                                       # the training / validation leak positions
    REFINE_JOINT=True,               # multiple leaks: a new leak is checked together with the leaks already running
    PRE_BASELINE_RATIO=0,            # new alarm: remove an earlier unreported leak seen 1-2 h before (tested: lower accuracy, off)
    REFINE_READINGS=8,               # readings (2 h) after the detection used by the refinement (until the next event)
    REFINE_WHITEN=True,              # weight the sensors by the noise seen on leak-free days (whitening)
    # ---- tests ----
    TEST_FRACTION=0.20,              # test scenarios = 20 % of all scenarios (80 % = training leaks)
    TEST_MIX=dict(single=0.40, multi=0.40, none=0.20),   # the 20 % test: 40 % single-leak days, 40 % multi-leak days,
                                                          # 20 % days without a leak
    TEST_DURATIONS_H=(1.0, 2.0, 3.0, 4.0),
    LEAKS_PER_MULTI_DAY=(2, 3),      #   2 or 3 leaks on different pipes
    OVERLAP_START_FRAC=(0.25, 0.75), #   overlapping: the next leak starts after 25-75 % of the previous leak's duration
                                     #   every leak (single or multiple) lasts TEST_DURATIONS_H -> fair comparison
    OVERLAP_SHARE=0.5,               # multiple-leak days: half OVERLAPPING (leaks running at the same time), half
                                     # NON-OVERLAPPING (the next leak starts after the previous one has ended)
    SEQ_GAP_H=(0.5, 2.0),            #   non-overlapping: the next leak starts 0.5-2 h after the previous one ended
    TEST_SEED=None,                  # None = new random test days on every run (the seed is printed); a number = repeat
    OPTIMUM_SHARE=0.90,              # optimum = fewest sensors keeping >= 90 % of the all-sensor performance
    WORKERS=None,                    # CPU cores (None = automatic, 1 = no parallel)
    SAVE_RAW_DATA=True,              # raw EPANET pressures of the training leaks and test days (CSV)
    OPEN_OUTPUT_FOLDER=True,
)


def save_performance_png(path, summary, order, size, net_name, opt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    n = summary.Sensors.to_numpy(); fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    a = ax[0, 0]
    for k, c, l in (("Single: detected (%)", "#1f77b4", "single leaks detected"), ("Multi: leaks detected (%)", "#ff7f0e", "leaks detected on multi-leak days"),
                    ("Multi: hidden by other leaks (%)", "#d62728", "multi-leak days: leaks hidden by other leaks")):
        a.plot(n, summary[k], "o-", color=c, label=l)
    a.set_title("(a) Detection vs number of pressure sensors"); a.set_ylabel("%"); a.set_ylim(0, 105)
    a = ax[0, 1]
    for k, c, ls, l in (("Single: correct pipe (%)", "#1f77b4", "-", "single: correct pipe"),
                        ("Single: correct or neighbouring pipe (%)", "#1f77b4", "--", "single: correct or neighbouring pipe"),
                        ("Single: correct pipe and zone (%)", "#2ca02c", "-", "single: correct pipe and 20 % zone"),
                        ("Multi: correct pipe (%)", "#ff7f0e", "-", "multi: correct pipe"),
                        ("Multi: correct or neighbouring pipe (%)", "#ff7f0e", "--", "multi: correct or neighbouring pipe")):
        a.plot(n, summary[k], "o", ls=ls, color=c, label=l)
    a.set_title("(b) Localisation vs number of pressure sensors (share of ALL test leaks)"); a.set_ylabel("%"); a.set_ylim(0, 105)
    a = ax[1, 0]
    pal = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    for i, k in enumerate(sorted(order.get(int(n[0]), {}))):
        a.plot(n, [order[int(s)].get(k, {}).get("detected", np.nan) for s in n], "o-", color=pal[i % 4], label=f"leak {k} of the day - detected")
        a.plot(n, [order[int(s)].get(k, {}).get("neighbour", np.nan) for s in n], "s--", color=pal[i % 4], label=f"leak {k} of the day - correct or neighbouring pipe")
    a.set_title("(c) Multi-leak days: 1st / 2nd / 3rd leak of the day"); a.set_ylabel("%"); a.set_ylim(0, 105)
    a = ax[1, 1]
    for i, k in enumerate(sorted(size.get(int(n[0]), {}))):
        a.plot(n, [size[int(s)].get(k, {}).get("detected", np.nan) for s in n], "o-", color=pal[i % 4], label=f"days with {k} leaks - detected")
        a.plot(n, [size[int(s)].get(k, {}).get("neighbour", np.nan) for s in n], "s--", color=pal[i % 4], label=f"days with {k} leaks - correct or neighbouring pipe")
    a.set_title("(d) Multi-leak days with 2 vs 3 leaks"); a.set_ylabel("%"); a.set_ylim(0, 105)
    for a in ax.flat:
        a.set_xscale("log", base=2); a.set_xticks(n); a.set_xticklabels([str(int(v)) for v in n]); a.invert_xaxis()
        a.set_xlabel("number of pressure sensors"); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower left")
        a.axvline(opt, color="grey", ls=":", lw=1.5)
    fig.suptitle(f"{net_name}: performance when the number of pressure sensors is halved step by step "
                 f"(dotted line = optimum, {opt} sensors)", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def save_sensor_map_png(path, net, sets):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    k = len(sets); cols = min(3, k); rows_ = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows_, cols, figsize=(6 * cols, 5 * rows_)); axs = np.atleast_1d(axs).ravel()
    for a, (n, js) in zip(axs, sets.items()):
        for p in net.pipes_all:
            xy = np.array(net.polyline(p)); a.plot(xy[:, 0], xy[:, 1], color="#b0b7bf", lw=1)
        for j in net.junctions:
            x, y = net.xy[j]; on = j in js
            a.plot(x, y, "o", ms=7 if on else 3, color="#08519c" if on else "#9e9e9e", zorder=3)
            if on and n <= 25: a.annotate(j, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="#08306b")
        for t in net.tanks: a.plot(*net.xy[t], "s", ms=8, color="#6d4c41")
        a.set_title(f"{n} pressure sensors"); a.set_aspect("equal"); a.axis("off")
    for a in axs[k:]: a.axis("off")
    fig.suptitle("Where the pressure sensors are (blue) for every number of sensors", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    t_start = time.time()
    print("=" * 100 + "\nLEAK DETECTION + LOCALISATION WITH A REDUCED NUMBER OF PRESSURE SENSORS (encoder-decoder)\n" + "=" * 100)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True); log = print
    net = Network(INPUT_FILE); u = net.len_unit; cfg = CFG
    log(f"Network : {net.inp.name}: {len(net.junctions)} junctions, {len(net.pipes)} open pipes, {len(net.tanks)} tanks, "
        f"{len(net.pumps)} pumps  (units: pressure {net.units[0]}, flow {net.units[1]}, length {u})")
    log(f"Sensors : ONLY pressure at the junctions; results are written to\n          {OUTPUT_DIR.resolve()}")
    crange, d_mean, p_mean = leak_coef_range(net, cfg["LEAK_SIZE_PCT"])
    log(f"Leaks   : {cfg['LEAK_SIZE_PCT'][0]:g}-{cfg['LEAK_SIZE_PCT'][1]:g} % of the average demand ({d_mean:.0f} {net.units[1]}) "
        f"-> emitter coefficient {crange[0]:.2f}-{crange[1]:.2f}")
    counts = sensor_counts(len(net.junctions), cfg)
    log(f"Sensor counts tested: {' -> '.join(map(str, counts))}")

    # ---------------- [1] normal days ----------------
    log(f"\n[1] {cfg['N_NORMAL_DAYS']} leak-free days (what normal operation looks like + alarm thresholds)")
    rng = np.random.default_rng(cfg["SEED"])
    normal = run_parallel(_test_day_job, [(str(net.inp), [], int(rng.integers(1 << 31)), cfg) for _ in range(cfg["N_NORMAL_DAYS"])],
                          cfg, log, "normal days")
    base_sd = np.concatenate([d["R"] for d in normal[:cfg["N_REFERENCE_DAYS"]]]).std(0)
    scale_all = np.maximum(base_sd, 0.1 * np.median(base_sd) + 1e-9)

    # ---------------- [2] training data + sensor sets (cached) ----------------
    cache = OUTPUT_DIR / "trained_models_cache.pkl"
    skip = ("TEST_FRACTION", "TEST_DURATIONS_H", "TEST_MIX", "LEAKS_PER_MULTI_DAY", "OVERLAP_START_FRAC",
            "TUNE_GRID", "TUNING_DAYS", "TEST_SEED", "OPTIMUM_SHARE", "WORKERS", "SAVE_RAW_DATA", "OPEN_OUTPUT_FOLDER", "DETECTION",
            "LOCALISE_READINGS", "N_NORMAL_DAYS", "N_REFERENCE_DAYS", "MP_WINDOWS",
            "EVAL_DAYS", "MAX_TEST_DAYS", "USE_GPU", "GPU_EXACT", "MODEL_REFINEMENT", "REFINE_TOP_PIPES", "REFINE_FINAL_PIPES", "REFINE_READINGS", "REFINE_WHITEN", "REFINE_JOINT", "PRE_BASELINE_RATIO", "REFINE_TOP_MAX", "WINDOW_MARGIN", "OVERLAP_SHARE", "SEQ_GAP_H")
    key = repr({k: cfg[k] for k in sorted(cfg) if k not in skip}) + repr(ZONE_EDGES) + (net.inp.read_text(errors="ignore") if net.inp.stat().st_size < 50e6 else "\n".join(net.lines))
    C = None
    if cache.exists():
        try:
            with open(cache, "rb") as f: C = pickle.load(f)
            if C.get("key") != key: C = None
            else: log("\n[2]-[4] training data, sensor sets and trained models loaded from trained_models_cache.pkl "
                      "(settings unchanged; delete it to retrain)")
        except Exception:
            C = None
    if C is None:
        n_pos, runs = training_plan(len(net.pipes), cfg)
        log(f"\n[2] TRAINING DATA: leaks on all {len(net.pipes)} pipes x {n_pos} positions x {runs} runs "
            f"(random size, start time, duration, demand level)")
        res = run_parallel(_train_pipe_job, [(str(net.inp), p, k, cfg, crange) for k, p in enumerate(net.pipes)], cfg, log, "pipes")
        store = [r for part in res for r in part]
        meta = pd.DataFrame([{k: r[k] for k in ("Pipe", "Position", "Zone", "Coefficient", "Start", "End", "Demand_multiplier")} for r in store])
        log(f"    {len(store)} EPANET leak simulations")
        log("\n[3] SENSOR PLACEMENT: removing sensors one by one (the one whose loss matters least for telling leak locations apart)")
        sets = select_sensors(net, store, meta, counts, scale_all, log)
        log("\n[4] ENCODER-DECODER: one model per number of sensors (same architecture, trained on the pressures of those sensors)")
        models = {}
        for n, js in sets.items():
            idx = [net.junctions.index(j) for j in js]; t0 = time.time()
            X, m_ = fingerprints(store, idx, scale_all, cfg, cfg["SEED"] + n)
            loc, rep, per_pipe, hist = train_localiser(net, X, m_, cfg, OUTPUT_DIR / "training_curve.png" if n == counts[0] else None, log=None)
            loc.fit_position(X, m_.Pipe, m_.Position, **cfg["POSITION_KRR"])
            if cfg.get("ZONE_CONFIDENCE"): loc.fit_zone_confidence(X, m_, cfg)
            models[n] = dict(loc=loc, report=rep, per_pipe=per_pipe)
            log(f"    {n:4d} sensors: trained in {time.time() - t0:5.0f} s on {loc.model.device if hasattr(loc.model, 'device') else 'CPU'} | validation (positions never trained): correct pipe "
                f"{rep.Value[5]}, correct or neighbouring pipe {rep.Value[6]}, median error {rep.Value[9]} {u}")
        C = dict(key=key, store=store, meta=meta, sets=sets, models=models, crange=crange)
        with open(cache, "wb") as f: pickle.dump(C, f)
    store, meta, sets, models = C["store"], C["meta"], C["sets"], C["models"]

    # ---------------- [5] test days ----------------
    tseed = cfg["TEST_SEED"] if cfg["TEST_SEED"] is not None else int(np.random.SeedSequence().entropy % 1_000_000)
    trng = np.random.default_rng(tseed); f_ = cfg["TEST_FRACTION"]
    n_test = int(round(f_ / (1 - f_) * len(store))); mix = cfg["TEST_MIX"]            # 80 / 20 split of all scenarios
    if cfg.get("MAX_TEST_DAYS"): n_test = min(n_test, int(cfg["MAX_TEST_DAYS"]))   # big networks: cap
    n_single = int(round(mix["single"] * n_test)); n_multi = int(round(mix["multi"] * n_test)); n_none = max(n_test - n_single - n_multi, 0)
    days = build_test_days(net, dict(cfg, N_NO_LEAK_DAYS=n_none, N_MULTI_LEAK_DAYS=n_multi), n_single, crange, trng)
    log(f"\n[5] TEST DAYS (never used for training, test seed {tseed}): 80/20 split -> {len(store)} training leaks + {n_test} test "
        f"scenarios = {n_single} single-leak days ({mix['single']:.0%}), {n_multi} days with "
        f"{'/'.join(map(str, cfg['LEAKS_PER_MULTI_DAY']))} overlapping leaks ({mix['multi']:.0%}), {n_none} days without a leak ({mix['none']:.0%})")
    sims = run_parallel(_test_day_job, [(str(net.inp), d["leaks"], int(trng.integers(1 << 31)), cfg) for d in days], cfg, log, "test days")

    # ---------------- [6] detector settings, tuned per number of sensors on separate tuning days ----------------
    td = cfg["TUNING_DAYS"]; urng = np.random.default_rng(cfg["SEED"] + 99)
    tdays = build_test_days(net, dict(cfg, N_NO_LEAK_DAYS=td["none"], N_MULTI_LEAK_DAYS=td["multi"]), td["single"], crange, urng)
    log(f"\n[6] DETECTOR SETTINGS tuned for every number of sensors on {len(tdays)} separate tuning days "
        f"({td['single']} single leaks, {td['multi']} multi-leak days, {td['none']} days without a leak)")
    tsims = run_parallel(_test_day_job, [(str(net.inp), d["leaks"], int(urng.integers(1 << 31)), cfg) for d in tdays], cfg, log, "tuning days")

    # ---------------- [6b] training / validation days, built EXACTLY like the test days ----------------
    # same mix (TEST_MIX: single / multiple leaks on different pipes / leak-free), same timing and noise, same complete
    # method (detection -> encoder-decoder -> hydraulic check); only the leaks come from the training / validation positions
    er = np.random.default_rng(cfg["SEED"] + 7); vmask = np.asarray(split_by_position(meta.assign(Noise_copy=0)), bool)
    evdays = []
    for tag, m_ in (("training", ~vmask), ("validation", vmask)):
        rows_ = meta[m_].sample(frac=1.0, random_state=int(er.integers(1 << 31))).reset_index(drop=True)
        used = np.zeros(len(rows_), bool)
        def src(rng_, ex, rows_=rows_, used=used):
            """Next unused leak of this set on a pipe not used yet that day (every scenario of the set is used once)."""
            free = np.where(~used & ~rows_.Pipe.astype(str).isin(ex).to_numpy())[0]
            if not len(free): free = np.where(~rows_.Pipe.astype(str).isin(ex).to_numpy())[0]
            i = int(free[0]); used[i] = True; q = rows_.iloc[i]
            return str(q.Pipe), float(q.Position), float(q.Coefficient)
        N_ = cfg["EVAL_DAYS"][tag]
        if N_ == "all":                   # ENTIRE set: enough days (same mix as the test days) to use every scenario once
            lpd = mix["single"] * 1 + mix["multi"] * float(np.mean(cfg["LEAKS_PER_MULTI_DAY"]))     # leaks per day
            N_ = int(np.ceil(len(rows_) / max(lpd, 1e-9)))
        ns, nm = int(round(mix["single"] * N_)), int(round(mix["multi"] * N_)); nn = max(N_ - ns - nm, 0)
        dd = build_test_days(net, dict(cfg, N_NO_LEAK_DAYS=nn, N_MULTI_LEAK_DAYS=nm), ns, crange, er, src)
        if cfg["EVAL_DAYS"][tag] == "all" and (~used).any():       # the few scenarios left over -> single-leak days
            dd += build_test_days(net, dict(cfg, N_NO_LEAK_DAYS=0, N_MULTI_LEAK_DAYS=0), int((~used).sum()), crange, er, src)
        for d in dd:
            d["set"] = tag; d["name"] = f"{tag}: {d['name']}"; evdays.append(d)
        N_ = len(dd)
        log(f"    {tag}: {N_} days, {int(used.sum())} of the {len(rows_)} {tag} scenarios used")
    log(f"\n[6b] TRAINING / VALIDATION days built exactly like the test days ({mix['single']:.0%} single, {mix['multi']:.0%} multiple, "
        f"{mix['none']:.0%} leak-free): {sum(d['set'] == 'training' for d in evdays)} training + "
        f"{sum(d['set'] == 'validation' for d in evdays)} validation days")
    evsims = run_parallel(_test_day_job, [(str(net.inp), d["leaks"], int(er.integers(1 << 31)), cfg) for d in evdays], cfg, log, "training / validation days")

    # ---------------- [7] evaluation for every number of sensors ----------------
    log("\n[7] EVALUATION for every number of sensors (detection with the Matrix Profile -> localisation with the encoder-decoder)")
    rows_all, summ, order, size, cases, tuned, tuning_tabs = [], [], {}, {}, {}, {}, {}
    refine = bool(cfg.get("MODEL_REFINEMENT"))
    topk = {n: (min(cfg.get("REFINE_TOP_MAX", cfg["REFINE_TOP_PIPES"]), int(round(cfg["REFINE_TOP_PIPES"] * np.sqrt(counts[0] / n))))
                if refine else 0) for n in counts}          # fewer sensors -> the encoder-decoder is less sure -> check more pipes
    rows_n, tasks, patch, rows_ev, noleak = {}, {}, [], [], {}

    def fill_loc(row, res, l):
        ok = res["pipe"] == l["pipe"]; nb = ok or net.adjacent(res["pipe"], l["pipe"]); zc = res.get("zone_conf")
        row.update({"Localised pipe": res["pipe"], "Most likely zone": f"{zone(res['pos'])} ({100 * max(zc):.0f}%)" if zc is not None else zone(res["pos"]),
                    "Pipe correct": "YES" if ok else "NO", "Pipe or neighbour": "YES" if nb else "NO",
                    "Zone correct": "YES" if ok and zone_index(res["pos"]) == zone_index(l["position"]) else "NO",
                    "Error": round(net.leak_distance(l["pipe"], l["position"], res["pipe"], res["pos"]), 1)})
        if zc is not None: row.update({f"Confidence {lab}": f"{100 * q:.0f}%" for lab, q in zip(zone_labels(), zc)})

    def process_day(n, idx, sc, loc, det, s_det, tag, k, d, sim):
        """Detection + encoder-decoder localisation of one day; registers the hydraulic check (refinement) tasks."""
        R, S, t = sim["R"][:, idx], sim["S"][:, idx], sim["t"]; Rp = proj_demand(R, S, sc)
        comb, got, false, dup, found = analyse_day(det, t, Rp, d["leaks"], s_det)
        fa = ", ".join(hhmm(t[L["i_det"]]) for L in false); locs = {}; rows, cs = [], []
        base = {"Sensors": n, "No.": k, "Kind": d["kind"], "Overlap": d.get("overlap"), "Test day": d["name"], "False alarms": len(false),
                "False alarm at": fa or "-", "Duplicate alarms": len(dup)}
        if not d["leaks"]:
            rows.append(dict(base, **{"Result": "NO LEAK REPORTED (correct)" if not false else f"FALSE ALARM at {fa}"}))
            cs.append(dict(row=0, leaks=[], res=None, t=t, comb=comb, false=[float(t[L["i_det"]]) for L in false]))
            return rows, cs
        for j, (l, g) in enumerate(zip(d["leaks"], got), 1):
            row = dict(base, **({} if j == 1 else {"False alarms": 0, "Duplicate alarms": 0}), **{"Leak no.": j if d["kind"] == "multi" else None, "Leaks that day": len(d["leaks"]) if d["kind"] == "multi" else None,
                                "Leak pipe": l["pipe"], "Position (%)": round(100 * l["position"], 1), "Emitter coef": l["coefficient"],
                                "Leak start": hhmm(l["on"]), "Leak end": hhmm(l["off"])})
            res = None
            if g is None:
                hidden = any(a_["alarm"]["i0"] <= int(np.ceil(l["on"] * 4)) < a_["alarm"].get("i1", len(t)) for a_ in found)
                row.update({"Detected": "NO (hidden by other leaks)" if (hidden and d["kind"] == "multi") else "NO",
                            "Pipe correct": "NO", "Pipe or neighbour": "NO", "Zone correct": "NO"})
            else:
                L, together = g; key_ = (tag, n, k, L["i_det"], id(L["alarm"]))            # unique per day and alarm
                if key_ not in locs:
                    locs[key_] = localise(loc, R, S, sc, L, len(t), cfg["LOCALISE_READINGS"], topk[n], cfg.get("REFINE_READINGS"),
                                          cfg.get("PRE_BASELINE_RATIO"), cfg.get("WINDOW_MARGIN", 0))
                    if refine: tasks.setdefault((tag, k - 1), []).append(dict(n=n, key=key_, idx=idx, win=locs[key_]["win_ref"],
                                                                             base=locs[key_].get("base_ref"), top=locs[key_]["top"]))
                res = locs[key_]
                row.update({"Detected": "YES", "Detected at": hhmm(t[L["i_det"]]), "Delay (min)": round((t[L["i_det"]] - l["on"]) * 60),
                            "How": {"new alarm": "new alarm", "new leak while an alarm was active": "new leak during an alarm",
                                    "new leak right after a repair": "new leak (pattern changed)"}.get(L["how"], L["how"]) + (" (together)" if together else "")})
                fill_loc(row, res, l)
                if refine: row.update({"Enc-dec pipe": res["pipe"], "Enc-dec pipe correct": row["Pipe correct"], "Enc-dec zone correct": row["Zone correct"]})
            rows.append(row)
            cs.append(dict(row=len(rows) - 1, leaks=d["leaks"], j=j - 1, res=res, t=t, comb=comb, false=[float(t[L["i_det"]]) for L in false]))
            if res is not None and refine: patch.append((key_, row, cs[-1], l))
        return rows, cs

    for n, js in sets.items():
        idx = [net.junctions.index(j) for j in js]; sc = scale_all[idx]; loc = models[n]["loc"]
        nul = [proj_demand(d["R"][:, idx], d["S"][:, idx], sc) for d in normal]
        ref, thr = nul[:cfg["N_REFERENCE_DAYS"]], nul[cfg["N_REFERENCE_DAYS"]:]
        det = (MatrixProfileDetector(sc, max(cfg["MP_WINDOWS"]), cfg["DETECTION"]["factor"]).fit(ref, thr)
               .fit_online(thr, cfg["MP_WINDOWS"], 0, ref).fit_new_leak(thr))
        s_det, tuning_tabs[n] = tune_detector(det, list(zip(tdays, tsims)), idx, sc, cfg["TUNE_GRID"], cfg["DETECTION"])
        tuned[n] = s_det
        rows, cs = [], []
        for k, (d, sim) in enumerate(zip(days, sims), 1):
            r_, c_ = process_day(n, idx, sc, loc, det, s_det, "test", k, d, sim)
            for c__ in c_: c__["row"] += len(rows)
            rows += r_; cs += c_
        rows_n[n] = rows; cases[n] = cs
        for k, (d, sim) in enumerate(zip(evdays, evsims), 1):                 # training / validation leaks
            r_, _ = process_day(n, idx, sc, loc, det, s_det, d["set"], k, d, sim)
            for x in r_: x["Set"] = d["set"]
            rows_ev += r_

    # ---- hydraulic check of every localisation (hydraulic model under the conditions measured that day) ----
    if refine and tasks:
        log(f"    HYDRAULIC CHECK: {sum(map(len, tasks.values()))} localisations (test + training / validation leaks), the hydraulic model "
            f"checks the {'/'.join(str(topk[n]) for n in counts)} most likely pipes of the encoder-decoder for {'/'.join(map(str, counts))} sensors (5 points each), then 10 points on the best "
            f"{cfg['REFINE_FINAL_PIPES']} with the estimated leak size" +
            ("; a leak that starts while others are running is checked together with them" if cfg.get("REFINE_JOINT", True) else "") + " ...")
        keys = sorted(tasks, key=lambda x: (x[0], x[1])); Wn = {}
        if cfg.get("REFINE_WHITEN"):          # noise covariance of the fingerprint on the leak-free days -> whitening matrix
            for n, js in sets.items():
                ix_ = [net.junctions.index(j) for j in js]; sc_ = scale_all[ix_]
                E = np.concatenate([_proj(d["R"][:, ix_] / sc_, d["S"][:, ix_] / sc_) for d in normal])
                Cv = np.cov(E.T) if len(ix_) > 1 else np.atleast_2d(E.var()); Cv = Cv + 0.05 * np.trace(Cv) / len(ix_) * np.eye(len(ix_))
                w_, V_ = np.linalg.eigh(Cv); Wn[n] = (V_ / np.sqrt(w_)) @ V_.T
        simof = lambda key: sims[key[1]] if key[0] == "test" else evsims[key[1]]
        jobs = [(str(net.inp), cfg, crange, scale_all, {f: simof(k)[f] for f in ("R", "S", "t", "bcH", "bcS", "Pexp")}, tasks[k], Wn) for k in keys]
        ref_out = {}
        errs = []
        for o in run_parallel(_refine_day_job, jobs, cfg, log, "days re-checked with the hydraulic model"):
            if "__error__" in o: errs.append(o.pop("__error__"))
            ref_out.update(o)
        if errs: log(f"    note: the hydraulic check failed on {len(errs)} days (encoder-decoder answer kept there), e.g. {errs[0]}")
        for key_, row, case, l in patch:
            if key_ in ref_out: fill_loc(row, ref_out[key_], l); case["res"] = ref_out[key_]
    rows_ev = pd.DataFrame(rows_ev).rename(columns={"Error": f"Error ({u})"})

    for n in sets:
        rows = rows_n[n]; s_det = tuned[n]
        s_, o_, z_ = summarise(rows, n, net); summ.append(s_); order[n] = o_; size[n] = z_; rows_all += rows
        log(f"    {n:4d} sensors | settings " + ", ".join(f"{k}={s_det[k]:g}" for k in cfg["TUNE_GRID"]) + "\n"
            f"               | single: detected {s_['Single: detected (%)']:5.1f} %, correct or neighbouring pipe "
            f"{s_['Single: correct or neighbouring pipe (%)']:5.1f} % | multi: detected {s_['Multi: leaks detected (%)']:5.1f} %, "
            f"correct or neighbouring pipe {s_['Multi: correct or neighbouring pipe (%)']:5.1f} % | no-leak days with alarm "
            f"{s_['No-leak days with a false alarm']}")
    summary = pd.DataFrame(summ).round(1)
    opt, opt_keys = optimum(summary, cfg["OPTIMUM_SHARE"])
    mt = multi_table(summary, order, size, u)                       # multiple leaks vs number of sensors
    rel = relation_table(mt); r_an = relation_text(rel)              # THE RELATION: sensors x leaks at the same time
    rel["pipe + zone"] = relation_zone(pd.DataFrame(rows_all), counts)  # same, exact pipe + zone (share of all leaks)
    k_an, k_tab = key_findings(rel["pipe + zone"], opt)                 # key findings at the top of the report
    m_an, mopt = multi_analysis(mt, cfg["OPTIMUM_SHARE"], sum(1 for r in rows_all if r["Sensors"] == counts[0] and r["Kind"] == "multi"), u)

    # ---------------- [7] outputs ----------------
    log("\n[8] Writing results ...")
    results = pd.DataFrame(rows_all).rename(columns={"Error": f"Error ({u})"})
    mo = pd.DataFrame([dict(Sensors=n, **{f"Leak {k}: {m}": v[m] for k, v in o.items() for m in ("detected", "pipe", "neighbour")})
                       for n, o in order.items()]).round(1)
    ms = pd.DataFrame([dict(Sensors=n, **{f"Days with {k} leaks: {m}": v[m] for k, v in o.items() for m in ("detected", "pipe", "neighbour")})
                       for n, o in size.items()]).round(1)
    sens_tab = pd.DataFrame([{"Junction": j, **{f"{n} sensors": ("X" if j in js else "") for n, js in sets.items()}} for j in net.junctions])
    val = pd.concat([models[n]["report"].set_index("Metric").rename(columns={"Value": f"{n} sensors"}) for n in sets], axis=1).reset_index()
    rsum = results_summary(results, counts)                          # THE one summary table
    t_tab, v_tab, x_tab, ft_tab, fv_tab, fx_tab = accuracy_tables(models, results, rows_ev, noleak, counts, u)
    conf = confusion_data(results, rows_ev, noleak, counts)         # confusion matrices per set (pipe, zone)
    scen = scenario_table({"Training": [d for d in evdays if d["set"] == "training"],
                           "Validation": [d for d in evdays if d["set"] == "validation"], "Test": days})
    eb, cats = error_breakdown(results, counts)                       # why a leak is not a correct result
    msz = missed_by_size(results, counts, net)                             # detection per leak-size quarter
    why = why_text(eb, cats, msz, counts, u)
    allp = rsum["Leak pipe"].isin(["All pipes", "-"])
    readme = pd.DataFrame(README, columns=["File / topic", "Sheet / term", "Meaning"])
    results["Leak type"] = [{"single": "Single", "none": "No leak"}.get(k_, "Multiple, " + ("overlapping" if o_ else "non-overlapping"))
                            for k_, o_ in zip(results.Kind, results.get("Overlap", [None] * len(results)))]
    results["Leak induced at"] = [f"{p:.0f} % of pipe {q}" if p == p else "-" for p, q in zip(results.get("Position (%)", np.nan), results.get("Leak pipe", "-"))]
    results["Correct pipe and zone"] = results["Zone correct"].fillna("-")
    results["Leak time"] = [f"{a_}-{b_}" if isinstance(a_, str) else "-" for a_, b_ in zip(results.get("Leak start", np.nan), results.get("Leak end", np.nan))]
    show = ["Sensors", "No.", "Leak type", "Leak no.", "Leak induced at", "Leak time", "Detected", "Localised pipe", "Most likely zone",
            "Correct pipe and zone", f"Error ({u})", "Result"]
    files = [safe_save(OUTPUT_DIR / "sensor_study_results.xlsx", {
        "README": readme, "Results_summary": rsum[allp], "Results_by_pipe": rsum[~allp], "Why_not_correct": eb,
        "Detection_by_leak_size": msz, "Scenarios": scen, "Training_details": t_tab, "Validation_details": v_tab, "Test_details": x_tab,
        "Full_training_report": ft_tab, "Full_validation_report": fv_tab, "Full_test_report": fx_tab,
        "Performance_vs_sensors": summary,
        "Relation_localisation": rel["localised"].reset_index(), "Relation_detection": rel["detected"].reset_index(),
        "Relation_exact_pipe": rel["exact pipe"].reset_index(),
        "Relation_pipe_and_zone": rel["pipe + zone"].reset_index(), "Key_findings_change": k_tab, "Multi_leak_vs_sensors": mt, "Multi_leak_by_leak_order": mo, "Multi_leak_by_day_size": ms,
        "Sensor_sets": sens_tab,
        "Detector_settings": pd.DataFrame([dict(Sensors=n, **{SETTING_NAMES.get(k, k): v for k, v in s_.items()}) for n, s_ in tuned.items()]),
        "Detector_tuning": pd.concat([t_.head(10).assign(Sensors=n) for n, t_ in tuning_tabs.items()], ignore_index=True),
        "Test_results": results[[c for c in show if c in results]],
        "Test_results_all_columns": results.drop(columns=["Kind", "Leak type", "Leak induced at", "Correct pipe and zone", "Leak time"]),
        "Train_valid_leak_results": rows_ev,
        "Model_validation": val})]
    files.append(safe_save(OUTPUT_DIR / "training_data.xlsx", {
        "README": readme,
        "Training_leaks": meta.assign(Simulation=np.arange(1, len(meta) + 1))[["Simulation", "Pipe", "Position", "Zone", "Coefficient", "Start", "End", "Demand_multiplier"]],
        "Training_summary": meta.groupby("Pipe", sort=False).agg(Simulations=("Position", "size"), Positions=("Position", "nunique"),
                                                                    Min_coef=("Coefficient", "min"), Max_coef=("Coefficient", "max")).reset_index()}))
    if cfg["SAVE_RAW_DATA"]:
        pu = net.units[0]; cols = [f"{j} [{pu}]" for j in net.junctions]
        parts = [pd.concat([pd.DataFrame({"Simulation": k, "Leak pipe": r["Pipe"], "Position": r["Position"], "Emitter coef": r["Coefficient"],
                                          "Leak start": r["Start"], "Leak end": r["End"], "Demand multiplier": r["Demand_multiplier"],
                                          "Time": [hhmm(x) for x in r["raw_t"]], "Leak on": r["raw_on"]}),
                            pd.DataFrame(np.round(r["raw_P"], 4), columns=cols)], axis=1) for k, r in enumerate(store, 1)]
        pd.concat(parts, ignore_index=True).to_csv(OUTPUT_DIR / "training_raw_pressure.csv", index=False)
        parts = []
        for k, (d, sim) in enumerate(zip(days, sims), 1):
            on = np.zeros(len(sim["t"]), int)
            for l in d["leaks"]: on |= ((sim["t"] >= l["on"]) & (sim["t"] < l["off"])).astype(int)
            j_ = lambda f: "; ".join(hhmm(l[f]) if f in ("on", "off") else str(l[f]) for l in d["leaks"]) if d["leaks"] else "-"
            parts.append(pd.concat([pd.DataFrame({"No.": k, "Test day": d["name"], "Leak pipe": j_("pipe"), "Emitter coef": j_("coefficient"),
                                                  "Leak start": j_("on"), "Leak end": j_("off"), "Demand factor": np.round(np.resize(sim["dn"], len(sim["t"])), 4),
                                                  "Time": [hhmm(x) for x in sim["t"]], "Leak on": on}),
                                    pd.DataFrame(np.round(sim["P"], 4), columns=cols)], axis=1))
        pd.concat(parts, ignore_index=True).to_csv(OUTPUT_DIR / "test_raw_pressure.csv", index=False)
        files += [OUTPUT_DIR / "training_raw_pressure.csv", OUTPUT_DIR / "test_raw_pressure.csv"]
    save_performance_png(OUTPUT_DIR / "performance_vs_sensors.png", summary, order, size, net.inp.stem, opt)
    save_relation_png(OUTPUT_DIR / "relation_sensors_vs_leaks.png", rel, net.inp.stem)
    files.append(OUTPUT_DIR / "relation_sensors_vs_leaks.png")
    save_multileak_png(OUTPUT_DIR / "multi_leak_vs_sensors.png", mt, net.inp.stem, u, mopt)
    files.append(OUTPUT_DIR / "multi_leak_vs_sensors.png")
    save_sensor_map_png(OUTPUT_DIR / "sensor_layouts.png", net, sets)
    files += [OUTPUT_DIR / "performance_vs_sensors.png", OUTPUT_DIR / "sensor_layouts.png"]
    analysis = analysis_text(summary, order, size, opt, cfg["OPTIMUM_SHARE"], u)
    tuned_tab = pd.DataFrame([dict(Sensors=n, **{SETTING_NAMES.get(k, k): v for k, v in s_.items() if k in cfg["TUNE_GRID"]}) for n, s_ in tuned.items()])
    files.append(write_study_html(OUTPUT_DIR / "sensor_study_report.html", net, summary, order, size, sets, results, cases, days,
                                  opt, analysis, val, crange, dict(cfg, _NEV=(sum(d["set"] == "training" for d in evdays),
                                                                              sum(d["set"] == "validation" for d in evdays))), tseed, len(store), show, tuned_tab, mt, m_an, mopt, rel, r_an, rsum, (t_tab, v_tab, x_tab, ft_tab, fv_tab, fx_tab, k_an, k_tab, scen),
                                  dict(single=n_single, multi=n_multi, none=n_none), eb, msz, why, conf))

    # ---------------- console ----------------
    log("\n" + "=" * 100 + "\nRESULTS SUMMARY (detected = correct pipe; localised = correct pipe + zone; share of all test leaks; no leak = no alarm)\n" + "=" * 100)
    print_table(rsum[allp & rsum["Leak induced at (point of the pipe)"].isin(["All positions", "-"])])
    log("\nWHY A LEAK IS NOT A CORRECT RESULT (% of the test leaks)"); print_table(eb)
    log("\nDETECTION PER LEAK SIZE"); print_table(msz)
    for w in why: log(" - " + re.sub("<[^>]+>", "", w))
    for nm_, tb_ in (("TRAINING", t_tab), ("VALIDATION", v_tab), ("TEST", x_tab)): log(f"\n{nm_} DETAILS"); print_table(tb_)
    log("\n" + "=" * 100 + "\nPERFORMANCE vs NUMBER OF PRESSURE SENSORS\n" + "=" * 100)
    print_table(summary)
    log("\nMULTI-LEAK DAYS: 1st / 2nd / 3rd leak of the day"); print_table(mo)
    log("\nANALYSIS"); [log("  - " + re.sub("<[^>]+>", "", a)) for a in analysis]
    log("\n" + "=" * 100 + "\nMULTIPLE LEAKS vs NUMBER OF PRESSURE SENSORS\n" + "=" * 100)
    print_table(mt[[c for c in mt if c == "Sensors" or c.startswith(("Single: correct or", "Multi:", "Gap", "Leak"))
                    and ("detected" in c or "neighbouring" in c or "Gap" in c or "error" in c)]])
    log("\nRELATION: % of leaks localised (correct or neighbouring pipe) - rows = leaks at the same time, columns = pressure sensors")
    print_table(rel["localised"].round(0).reset_index())
    log("\nRELATION: % of leaks detected"); print_table(rel["detected"].round(0).reset_index())
    [log("  - " + re.sub("<[^>]+>", "", a)) for a in r_an]
    log("\nANALYSIS - MULTIPLE LEAKS"); [log("  - " + re.sub("<[^>]+>", "", a)) for a in m_an]
    log(f"\nOPTIMUM: {opt} pressure sensors (fewest sensors keeping >= {cfg['OPTIMUM_SHARE']:.0%} of the all-sensor performance)")
    log(f"\nFILES WRITTEN ({(time.time() - t_start) / 60:.1f} min):")
    for f in files: log(f"  {Path(f).resolve()}")
    open_file(files[-1])
    if cfg["OPEN_OUTPUT_FOLDER"]: open_file(OUTPUT_DIR)


def analysis_text(summary, order, size, opt, share, u):
    """Short bullet points: how the performance changes when sensors are removed."""
    f, l = summary.iloc[0], summary.iloc[-1]; N0, N1 = int(f.Sensors), int(l.Sensors); out = []
    out.append(f"<b>Detected:</b> {f['Single: detected (%)']:.0f} % single / {f['Multi: leaks detected (%)']:.0f} % multiple leaks with {N0} sensors "
               f"&rarr; {l['Single: detected (%)']:.0f} % / {l['Multi: leaks detected (%)']:.0f} % with {N1}")
    out.append(f"<b>Correct pipe + zone (single):</b> {f['Single: correct pipe and zone (%)']:.0f} % with {N0} sensors &rarr; "
               f"{l['Single: correct pipe and zone (%)']:.0f} % with {N1}")
    drop = summary["Single: correct pipe (%)"].diff().fillna(0).to_numpy()
    if len(drop) > 1:
        k = int(np.argmin(drop)); out.append(f"<b>Biggest loss</b> (correct pipe): {int(summary.Sensors[k - 1])} &rarr; {int(summary.Sensors[k])} "
                                             f"sensors ({drop[k]:+.0f} points)")
    out.append(f"<b>Optimum: {opt} sensors</b> (fewest keeping &ge; {share:.0%} of the all-sensor detection and localisation)")
    return out


# =====================================================================================================
# MULTIPLE LEAKS vs NUMBER OF SENSORS (dedicated table, figure and written analysis)
# =====================================================================================================
def multi_table(summary, order, size, u):
    """One row per number of sensors: single vs multi-leak performance, 1st / 2nd / 3rd leak, days with 2 / 3 leaks."""
    rows = []
    for _, r in summary.iterrows():
        n = int(r.Sensors); o, z = order.get(n, {}), size.get(n, {})
        row = {"Sensors": n, "Single: detected (%)": r["Single: detected (%)"], "Multi: detected (%)": r["Multi: leaks detected (%)"],
               "Multi: hidden by other leaks (%)": r["Multi: hidden by other leaks (%)"],
               "Single: exact pipe (%)": r["Single: correct pipe (%)"], "Multi: exact pipe (%)": r["Multi: correct pipe (%)"],
               "Single: correct or neighbouring pipe (%)": r["Single: correct or neighbouring pipe (%)"],
               "Multi: correct or neighbouring pipe (%)": r["Multi: correct or neighbouring pipe (%)"],
               "Gap single - multi (points)": r["Single: correct or neighbouring pipe (%)"] - r["Multi: correct or neighbouring pipe (%)"],
               f"Single: median error ({u})": r[f"Single: median error ({u})"], f"Multi: median error ({u})": r[f"Multi: median error ({u})"]}
        for k, v in sorted(o.items()):
            row.update({f"Leak {k} of the day: detected (%)": v["detected"], f"Leak {k} of the day: exact pipe (%)": v["pipe"],
                        f"Leak {k} of the day: correct or neighbouring pipe (%)": v["neighbour"]})
        for k, v in sorted(z.items()):
            row.update({f"Days with {k} leaks: detected (%)": v["detected"], f"Days with {k} leaks: exact pipe (%)": v["pipe"],
                        f"Days with {k} leaks: correct or neighbouring pipe (%)": v["neighbour"]})
        rows.append(row)
    return pd.DataFrame(rows).round(1)


def multi_analysis(mt, share, n_leaks, u):
    """Short bullet points: number of pressure sensors vs multiple leaks (on different pipes)."""
    f = mt.iloc[0]; N = mt.Sensors.astype(int).tolist(); out = []
    keep = lambda col: min([int(r.Sensors) for _, r in mt.iterrows() if r[col] >= share * f[col]] or [N[0]])
    d_ok, l_ok = keep("Multi: detected (%)"), keep("Multi: correct or neighbouring pipe (%)"); mopt = max(d_ok, l_ok)
    out.append(f"<b>Detection</b> of multiple leaks stays &ge; {share:.0%} of the all-sensor value down to <b>{d_ok} sensors</b>")
    out.append(f"<b>Localisation</b> (correct or neighbouring pipe) stays &ge; {share:.0%} down to <b>{l_ok} sensors</b>")
    gap = mt["Gap single - multi (points)"]
    out.append(f"<b>Multiple vs single:</b> on average {gap.mean():.0f} points lower (max {gap.max():.0f} with {int(mt.Sensors[gap.idxmax()])} sensors)")
    c1 = "Leak 1 of the day: correct or neighbouring pipe (%)"
    later = [c for c in mt if c.startswith("Leak ") and "correct or neighbouring" in c and not c.startswith("Leak 1 ")]
    if c1 in mt and later:
        lat = mt[later].mean(axis=1)
        out.append(f"<b>Later leaks</b> (start while another is running) are hardest: {lat.iloc[0]:.0f} % vs {mt[c1].iloc[0]:.0f} % for the "
                   f"1st leak ({N[0]} sensors)")
    out.append(f"<b>Optimum for multiple leaks: {mopt} sensors</b> ({n_leaks} test leaks; differences of about &plusmn;10 points can be chance)")
    return out, mopt


def save_multileak_png(path, mt, net_name, u, mopt):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    N = mt.Sensors.astype(int).to_numpy(); x = np.arange(len(N)); fig, ax = plt.subplots(2, 2, figsize=(16, 10.5))
    a = ax[0, 0]
    a.plot(x, mt["Single: correct or neighbouring pipe (%)"], "o-", color="#1f77b4", lw=2.2, label="single leaks: correct or neighbouring pipe")
    a.plot(x, mt["Multi: correct or neighbouring pipe (%)"], "o-", color="#d62728", lw=2.2, label="multi-leak days: correct or neighbouring pipe")
    a.fill_between(x, mt["Multi: correct or neighbouring pipe (%)"], mt["Single: correct or neighbouring pipe (%)"], color="#d62728", alpha=0.12,
                   label="loss caused by the other leaks")
    a.plot(x, mt["Single: detected (%)"], "s--", color="#1f77b4", alpha=0.6, label="single leaks: detected")
    a.plot(x, mt["Multi: detected (%)"], "s--", color="#d62728", alpha=0.6, label="multi-leak days: detected")
    a.set_title("(a) Single leak vs several leaks at the same time"); a.set_ylim(0, 105); a.set_ylabel("% of the test leaks")
    a = ax[0, 1]
    a.plot(x, mt[f"Single: median error ({u})"], "o-", color="#1f77b4", lw=2.2, label="single leaks")
    a.plot(x, mt[f"Multi: median error ({u})"], "o-", color="#d62728", lw=2.2, label="leaks on multi-leak days")
    a.set_yscale("log"); a.set_title("(b) Median localisation error (detected leaks)"); a.set_ylabel(f"error along the pipes ({u}, log scale)")
    a = ax[1, 0]; cols = [c for c in mt if c.startswith("Leak ") and "correct or neighbouring" in c]; w = 0.8 / max(len(cols), 1)
    pal = ["#2c7fb8", "#fd8d3c", "#d62728", "#756bb1"]
    for i, c in enumerate(cols):
        k = c.split()[1]
        a.bar(x + (i - (len(cols) - 1) / 2) * w, mt[c], w, color=pal[i % 4], label=f"leak {k} of the day: correct or neighbouring pipe")
        a.plot(x + (i - (len(cols) - 1) / 2) * w, mt[f"Leak {k} of the day: detected (%)"], "kD", ms=5, label="detected" if i == 0 else None)
    a.set_title("(c) 1st / 2nd / 3rd leak of the day (bars = localised, diamonds = detected)"); a.set_ylim(0, 105); a.set_ylabel("%")
    a = ax[1, 1]; ds = sorted({int(c.split()[2]) for c in mt if c.startswith("Days with ")}); w = 0.8 / max(len(ds), 1)
    for i, k in enumerate(ds):
        xo = x + (i - (len(ds) - 1) / 2) * w
        a.bar(xo, mt[f"Days with {k} leaks: correct or neighbouring pipe (%)"], w, color=pal[i % 4], alpha=0.45, label=f"days with {k} leaks: correct or neighbouring pipe")
        a.bar(xo, mt[f"Days with {k} leaks: exact pipe (%)"], w * 0.55, color=pal[i % 4], label=f"days with {k} leaks: exact pipe")
    a.set_title("(d) Days with 2 vs 3 leaks at the same time"); a.set_ylim(0, 105); a.set_ylabel("%")
    for i_, a in enumerate(ax.flat):
        a.set_xticks(x); a.set_xticklabels([str(n) for n in N]); a.set_xlabel("number of pressure sensors")
        a.grid(alpha=.3, axis="y")
        if i_ == 1: a.legend(fontsize=8, loc="upper left")
        else:                                            # legend above the data (free space at the top)
            a.set_ylim(0, 135); a.set_yticks(range(0, 101, 20)); a.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
        if mopt in N: a.axvline(list(N).index(mopt), color="grey", ls=":", lw=1.5)
    fig.suptitle(f"{net_name}: reduced number of pressure sensors vs MULTIPLE LEAKS (dotted line = optimum for multiple leaks, {mopt} sensors)",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def relation_table(mt):
    """THE RELATION: performance as a function of (number of pressure sensors) x (number of leaks at the same time).
    1 leak = single-leak test days, 2 / 3 leaks = multi-leak days with 2 / 3 overlapping leaks."""
    ks = sorted({int(c.split()[2]) for c in mt if c.startswith("Days with ")})
    out = {}
    for what, s_col, m_col in (("detected", "Single: detected (%)", "detected (%)"),
                               ("localised", "Single: correct or neighbouring pipe (%)", "correct or neighbouring pipe (%)"),
                               ("exact pipe", "Single: exact pipe (%)", "exact pipe (%)")):
        t = pd.DataFrame({"1 leak": mt[s_col].to_numpy(), **{f"{k} leaks (different pipes)": mt[f"Days with {k} leaks: {m_col}"].to_numpy() for k in ks}},
                         index=mt.Sensors.astype(int).to_numpy()).T
        t.index.name = "Leaks at the same time"; t.columns.name = "Pressure sensors"; out[what] = t
    return out


def save_relation_png(path, rel, net_name):
    """One figure: performance vs number of pressure sensors, one line per number of simultaneous leaks + heatmaps."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    loc, det = rel["localised"], rel["detected"]; N = list(loc.columns); x = np.arange(len(N))
    fig = plt.figure(figsize=(17, 10.5)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    pal = {"1 leak": "#1a9850", "2 leaks": "#fdae61", "3 leaks": "#d73027", "4 leaks": "#762a83"}
    for j, (tab, title) in enumerate(((loc, "LOCALISATION: leaks found on the correct or a neighbouring pipe"),
                                      (det, "DETECTION: leaks detected in real time"))):
        a = fig.add_subplot(gs[0, j])
        for k in tab.index:
            a.plot(x, tab.loc[k], "o-", lw=2.8, ms=8, color=pal.get(k, "grey"), label=f"{k} at the same time")
            for xi, v in zip(x, tab.loc[k]):
                if v == v: a.annotate(f"{v:.0f}", (xi, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=pal.get(k, "grey"))
        a.set_xticks(x); a.set_xticklabels([str(n) for n in N]); a.set_xlabel("number of pressure sensors  (halved at every step)")
        a.set_ylabel("% of the leaks"); a.set_ylim(0, 108); a.grid(alpha=.3); a.legend(loc="lower left", fontsize=10)
        a.set_title(f"({'ab'[j]}) {title}", fontsize=12)
    for j, (tab, title) in enumerate(((loc, "(c) Localisation (%): sensors x leaks at the same time"),
                                      (det, "(d) Detection (%): sensors x leaks at the same time"))):
        a = fig.add_subplot(gs[1, j]); im = a.imshow(tab.to_numpy(float), cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        a.set_xticks(x); a.set_xticklabels([str(n) for n in N]); a.set_yticks(range(len(tab.index))); a.set_yticklabels(tab.index)
        a.set_xlabel("number of pressure sensors"); a.set_ylabel("leaks at the same time"); a.set_title(title, fontsize=12)
        for (r, c), v in np.ndenumerate(tab.to_numpy(float)):
            if v == v: a.text(c, r, f"{v:.0f}", ha="center", va="center", fontsize=11, fontweight="bold", color="black")
        fig.colorbar(im, ax=a, fraction=0.025, pad=0.02, label="%")
    fig.suptitle(f"{net_name}: RELATION between the number of pressure sensors and the number of leaks at the same time\n"
                 "(the more leaks run at the same time, the more sensors are needed to localise them)", fontsize=14)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def relation_zone(r, counts):
    """Relation table, exact pipe + zone: rows = leaks at the same time, columns = sensors, value = % of ALL leaks."""
    out = {}
    for n in counts:
        g = r[r.Sensors == n]; col = {}
        s_ = g[g.Kind == "single"]; col["1 leak"] = 100.0 * float((s_["Zone correct"] == "YES").mean()) if len(s_) else np.nan
        m_ = g[g.Kind == "multi"]
        for ov, tag in ((True, "overlapping"), (False, "non-overlapping")):
            for k, q in m_[m_.Overlap == ov].groupby("Leaks that day"):
                col[f"{int(k)} leaks, {tag}"] = 100.0 * float((q["Zone correct"] == "YES").mean())
        out[n] = col
    t = pd.DataFrame(out).round(1); t.index.name = "Leaks at the same time"; t.columns.name = "Pressure sensors"
    return t


def key_findings(z, opt):
    """KEY FINDINGS (bullets + table): how localisation (correct pipe + zone) changes with the number of sensors and the
    number of simultaneous leaks."""
    N = list(z.columns); out = []; keep = {}
    for k in z.index:
        r = z.loc[k]; thr = 0.9 * r.iloc[0]; kp = N[0]
        for n in N[1:]:
            if r[n] >= thr: kp = n
            else: break
        keep[k] = kp
        out.append(f"<b>{k}:</b> " + " &rarr; ".join(f"{r[n]:.0f} % ({n})" for n in N) + f"; unaffected (&ge; 90 %) down to <b>{kp} sensors</b>")
    if len(z.index) > 1:
        v = [keep[k] for k in z.index]
        lab = ("More leaks at the same time need more sensors" if all(b >= a for a, b in zip(v, v[1:])) and v[-1] > v[0] else
               "Sensors needed")
        out.append(f"<b>{lab}:</b> unaffected down to " + " / ".join(f"{x}" for x in v) + " sensors for " +
                   " / ".join(k.split(" (")[0] for k in z.index))
        avg = z.mean(axis=1)
        for tag in ("overlapping", "non-overlapping"):
            ks = ["1 leak"] + [k for k in z.index if k.endswith(", " + tag)]
            ks = [k for k in ks if k in avg.index]
            if len(ks) > 1:
                out.append(f"<b>{tag.capitalize()}:</b> " + " &rarr; ".join(f"{avg[k]:.0f} % ({k.split(',')[0]})" for k in ks) +
                           " on average over all sensor counts")
    out.append(f"<b>Optimum overall: {opt} sensors</b>")
    ch = pd.DataFrame({f"{a} \u2192 {b} sensors": (z[b] - z[a]).round(1) for a, b in zip(N[:-1], N[1:])}).T
    ch.index.name = "Halving the sensors: change (points)"; ch.columns.name = None   # no empty extra column in the report
    return out, ch.reset_index()


def relation_text(rel):
    """Short bullet points: the relation number of sensors x number of simultaneous leaks (every sensor count)."""
    loc = rel["localised"]; N = list(loc.columns); out = []
    for k in loc.index:
        r = loc.loc[k]; thr = 0.9 * r.iloc[0]; keep = N[0]
        for n in N[1:]:
            if r[n] >= thr: keep = n
            else: break
        out.append(f"<b>{k}:</b> " + " &rarr; ".join(f"{r[n]:.0f} % ({n})" for n in N) +
                   f"; unaffected (&ge; 90 %) down to <b>{keep} sensors</b>")
    return out

def write_study_html(path, net, summary, order, size, sets, results, cases, days, opt, analysis, val, crange, cfg, tseed, n_train, show, tuned_tab,
                     mt, m_an, mopt, rel, r_an, rsum, accs, ntest, eb, msz, why, conf):
    """Interactive report: performance vs number of sensors, multi-leak analysis, sensor maps, every test day."""
    u = net.len_unit
    kind = lambda n: "J" if n in net.junctions else ("Tank" if n in net.tanks else "Reservoir")
    clean = lambda v: None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
    cols = [c for c in show if c in results.columns and c != "Sensors"]
    tables, cs_out = {}, {}
    for n in sets:
        r_ = results[results.Sensors == n]
        tables[n] = {k: [[clean(v) for v in row] for row in r_[r_.Kind == k][cols].astype(object).where(r_[r_.Kind == k][cols].notna(), None).values.tolist()]
                     for k in ("single", "multi", "none")}
        cs = []
        for c in cases[n]:
            res = c["res"]
            cs.append(dict(kind=results.loc[results.index[0], "Kind"] if False else None, leaks=[[l["pipe"], l["position"], l["on"], l["off"]] for l in c["leaks"]],
                           j=c.get("j"), pipe=res["pipe"] if res else None, pos=res["pos"] if res else None,
                           zc=[round(float(q), 4) for q in res["zone_conf"]] if res and res.get("zone_conf") is not None else None,
                           false=c["false"]))
        kinds = results[results.Sensors == n].Kind.tolist()
        for c, k in zip(cs, kinds): c["kind"] = k
        cs_out[n] = {k: [c for c in cs if c["kind"] == k] for k in ("single", "multi", "none")}
    # day score timeline (one per test day and sensor count)
    dayscore = {n: [] for n in sets}
    for n in sets:
        seen = set()
        for c in cases[n]:
            key = id(c["comb"])
            if key not in seen: seen.add(key); dayscore[n].append(np.round(np.maximum(c["comb"], 1e-3), 3).tolist())
    day_of = {}
    for n in sets:                                   # row -> day index
        r_ = results[results.Sensors == n]; ids = r_["No."].tolist()
        day_of[n] = {k: [int(i) - 1 for i in r_[r_.Kind == k]["No."]] for k in ("single", "multi", "none")}
    data = dict(geo={p: net.polyline(p) for p in net.pipes_all}, nodes={n: list(net.xy[n]) + [kind(n)] for n in net.xy},
                ends={p: net.ends[p] for p in net.pipes_all}, sets={str(n): js for n, js in sets.items()},
                counts=[int(n) for n in sets], summary=summary.to_dict(orient="list"),
                order={str(n): {str(k): v for k, v in o.items()} for n, o in order.items()},
                size={str(n): {str(k): v for k, v in o.items()} for n, o in size.items()},
                cols=cols, tables={str(n): t for n, t in tables.items()}, cases={str(n): c for n, c in cs_out.items()},
                dayscore={str(n): s for n, s in dayscore.items()}, dayof={str(n): d for n, d in day_of.items()},
                t=[float(x) for x in np.arange(97) * 0.25], zones=zone_labels(), edges=list(ZONE_EDGES), opt=int(opt),
                labelpipes=len(net.pipes) <= 40, unit=u, mt=mt.to_dict(orient="list"), mopt=int(mopt),
                eb={k: g.drop(columns=['Leak type']).set_index('Sensors').reindex([int(n) for n in sets]).to_dict(orient='list') for k, g in eb.groupby('Leak type')},
                msz=msz.to_dict(orient='list') if len(msz) else None, conf=conf,
                rel={k: dict(rows=list(v.index), cols=[int(c) for c in v.columns], z=np.round(v.to_numpy(float), 1).tolist()) for k, v in rel.items()})
    js = json.dumps(data, default=lambda o: o.tolist() if hasattr(o, "tolist") else (None if o is None else float(o)))
    js = js.replace("NaN", "null")
    try:
        import plotly
        plotly_tag = "<script>" + (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(encoding="utf-8") + "</script>"
    except Exception:
        plotly_tag = PLOTLY_CDN_TAG
    tb = lambda df: df.to_html(index=False, classes="tbl", border=0, na_rep="-")
    f0 = summary.iloc[0]
    cards = {"Network": f"{net.inp.stem}", "Junctions (max. pressure sensors)": len(net.junctions), "Open pipes": len(net.pipes),
             "Training leaks (80 %)": n_train, "Test scenarios (20 %)": sum(ntest.values()),
             f"Single-leak test days ({cfg['TEST_MIX']['single']:.0%})": ntest["single"],
             f"Multiple-leak test days, different pipes ({cfg['TEST_MIX']['multi']:.0%})": ntest["multi"],
             f"Leak-free test days ({cfg['TEST_MIX']['none']:.0%})": ntest["none"],
             "Optimum number of sensors": opt}
    html = """<!doctype html><html><head><meta charset="utf-8"><title>@@TITLE@@</title>@@PLOTLY@@
<style>body{font-family:Segoe UI,Arial,sans-serif;margin:18px;background:#f6f7f9;color:#222}h1{font-size:21px;margin:0 0 6px}
h2{font-size:17px;margin:26px 0 6px}.note{font-size:13px;color:#555;margin:2px 0 8px;max-width:1200px}
.cards{display:flex;gap:10px;flex-wrap:wrap}.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 14px;font-size:13px}
.card b{font-size:19px;display:block}.card.opt{border:2px solid #08519c}.tbl{border-collapse:collapse;font-size:12px;background:#fff}
.tbl th{background:#263238;color:#fff;padding:5px 7px;position:sticky;top:0;text-align:center!important;vertical-align:middle}.tbl td{padding:4px 7px;border-bottom:1px solid #e3e3e3;text-align:center}
.click tbody tr:hover td{background:#e3f2fd;cursor:pointer}.click tr.sel td{background:#fff3c4}.scroll{max-height:380px;overflow:auto;border:1px solid #ddd}
.row{display:flex;flex-wrap:wrap;gap:12px}.box{background:#fff}ul.an li{margin:4px 0;font-size:14px}
.ok{color:#1b5e20;font-weight:600}.bad{color:#b71c1c;font-weight:600}select{font-size:14px;padding:2px 6px}
#info{font-size:14px;margin:10px 0;padding:8px 12px;background:#fff;border-left:5px solid #08519c}</style></head><body>
<h1>@@TITLE@@</h1><ul class="an"><li>Only <b>pressure sensors at junctions</b>; sensors halved step by step: @@COUNTS@@</li>
<li>Leaks: @@LEAK@@; multiple leaks of one day are always on <b>different pipes</b>: half of these days <b>overlapping</b> (running at the same time), half <b>non-overlapping</b> (one after another)</li>
<li>Localisation = encoder-decoder (8 most likely pipes) + hydraulic-model check of those pipes; test seed @@SEED@@</li></ul>
<div class="cards">@@CARDS@@</div>
<h2>Key findings: number of pressure sensors vs multiple leaks</h2><ul class="an">@@KEYAN@@</ul>
<div class="row"><div id="g_k1" class="box" style="width:600px;height:330px"></div><div id="g_k2" class="box" style="width:600px;height:330px"></div></div>
<h3>How the result changes when the sensors are halved (correct pipe + zone, points)</h3><div class="scroll" style="max-height:none">@@KEYTAB@@</div>
<h2>1. Results summary</h2><ul class="an">
<li><b>Leaks detected - correct pipe</b> = alarm raised and the leak placed on the correct pipe</li>
<li><b>Leaks localised - correct pipe + zone</b> = also in the correct 20 % zone of that pipe</li>
<li>Both are shares of ALL test leaks (a missed leak counts as wrong); no leak: share of leak-free days without an alarm</li>
<li><b>Accuracy retained</b> = localised (%) &divide; localised with all sensors; <b>change</b> = vs twice as many sensors (points)</li>
<li>Per-pipe rows have few test leaks each: a drill-down only</li></ul>
<div class="note">Show: <select id="pickR"><option value="">all leak types</option><option value="Single">single leak</option>
<option value="Multiple">multiple leaks (both)</option><option value="Multiple leaks, overlapping">multiple leaks, overlapping</option>
<option value="Multiple leaks, non">multiple leaks, non-overlapping</option><option value="No leak">no leak</option></select>
<select id="pickL"><option value="">all pipes together</option></select>
<select id="pickP"><option value="All">all positions together</option><option value="">every 20 % zone of the pipe</option></select></div>
<div class="scroll" id="rsum" style="max-height:520px">@@RSUM@@</div>
<h2>2. Accuracy: training, validation and test</h2><ul class="an">
<li>All values: complete method (detection &rarr; encoder-decoder &rarr; hydraulic check), all leak types together, one value per sensor count</li>
<li>Accuracy = share of the <b>detected</b> leaks on the correct pipe / correct pipe + zone (detection itself: confusion matrices, section 4)</li>
<li><b>Training</b> = leaks the model learned from; <b>validation</b> = leak positions never used for training; <b>test</b> = unseen test days</li>
<li>Same process for all three sets: days built the same way (@@MIX@@ single / multiple leaks on different pipes / leak-free),
same noise, same complete method; only the leak positions differ (training / validation / unseen test positions)</li>
<li>Days per set: @@NEV@@ (training / validation / test)</li></ul>
<h3>Scenarios in each set (every scenario evaluated at every number of sensors)</h3><div class="scroll" style="max-height:none">@@SCEN@@</div>
<h3>Training details</h3><div class="scroll" style="max-height:none">@@TTAB@@</div>
<details><summary style="cursor:pointer;font-size:13px">full training report</summary><div class="scroll" style="max-height:none">@@FTTAB@@</div></details>
<h3>Validation details</h3><div class="scroll" style="max-height:none">@@VTAB@@</div>
<details><summary style="cursor:pointer;font-size:13px">full validation report</summary><div class="scroll" style="max-height:none">@@FVTAB@@</div></details>
<h3>Test details</h3><div class="scroll" style="max-height:none">@@XTAB@@</div>
<details><summary style="cursor:pointer;font-size:13px">full test report</summary><div class="scroll" style="max-height:none">@@FXTAB@@</div></details>
<h2>3. Every test day</h2><ul class="an"><li>Choose sensors and type of day; click a row: map (red = true leak, localised pipe split into
its 5 zones, darker = more likely) + detection score of that day</li></ul>
<div class="note">Number of sensors: <select id="pickN"></select> &nbsp; Test days: <select id="pickK"><option value="single">single leak</option>
<option value="multi">multiple leaks (different pipes)</option><option value="none">no leak</option></select></div>
<div class="click scroll" id="tab"></div><div id="info"></div>
<div class="row"><div id="g_case" class="box" style="flex:3;min-width:520px;height:560px"></div><div id="g_score" class="box" style="flex:2;min-width:420px;height:560px"></div></div>
<h2>4. Confusion matrices: training, validation and test set</h2><ul class="an">
<li><b>Leak pipe:</b> rows = actual (leak / no leak); columns = what was reported: leak on the correct pipe, a leak where there is none
(wrong pipe, or a false alarm on a leak-free day), no leak reported (missed leak / correct silence)</li>
<li><b>Position:</b> rows = true zone of the leak; columns = zone found on the correct pipe, or neighbouring pipe / wrong pipe / not detected</li>
<li>All leak types together; cells = number of cases and % of the row</li>@@WHY@@</ul>
<div class="note">Number of sensors: <select id="cmN"></select></div>
<div class="row"><div id="g_p_training" class="box" style="width:400px;height:330px"></div><div id="g_p_validation" class="box" style="width:400px;height:330px"></div><div id="g_p_test" class="box" style="width:400px;height:330px"></div></div>
<div class="row"><div id="g_z_training" class="box" style="width:400px;height:430px"></div><div id="g_z_validation" class="box" style="width:400px;height:430px"></div><div id="g_z_test" class="box" style="width:400px;height:430px"></div></div>
<h2>5. Number of pressure sensors vs multiple leaks</h2><ul class="an">@@RANALYSIS@@@@MANALYSIS@@</ul>
<div class="row"><div id="g_r1" class="box" style="width:600px;height:420px"></div><div id="g_r2" class="box" style="width:600px;height:420px"></div></div>
<div class="row"><div id="g_r3" class="box" style="width:600px;height:300px"></div><div id="g_r4" class="box" style="width:600px;height:300px"></div></div>
<div class="row"><div id="g_m1" class="box" style="width:600px;height:420px"></div><div id="g_m2" class="box" style="width:600px;height:420px"></div></div>
<details><summary style="cursor:pointer;font-size:13px">table of these numbers</summary><div class="scroll" style="max-height:none">@@MTABLE@@</div></details>
<h2>6. Performance vs number of pressure sensors</h2><ul class="an">@@ANALYSIS@@</ul>
<div class="row"><div id="g_det" class="box" style="width:600px;height:400px"></div><div id="g_loc" class="box" style="width:600px;height:400px"></div></div>
<div class="row"><div id="g_err" class="box" style="width:600px;height:360px"></div><div id="g_fa" class="box" style="width:600px;height:360px"></div></div>
<details><summary style="cursor:pointer;font-size:13px">table of these numbers</summary><div class="scroll" style="max-height:none">@@SUMMARY@@</div></details>
<h2>7. Where the pressure sensors are</h2><ul class="an"><li>Removed one at a time (the one that matters least); smaller sets are part of the bigger ones</li></ul>
<div class="note">Number of sensors: <select id="pickS"></select></div><div id="g_map" class="box" style="height:600px"></div>
<h2>8. Detector settings per number of sensors</h2><ul class="an"><li>Tuned on separate tuning days: most leaks found, fewest false alarms</li></ul>
<div class="scroll" style="max-height:none">@@TUNED@@</div>
<details><summary style="cursor:pointer;font-size:13px">full validation report of the encoder-decoder</summary><div class="scroll" style="max-height:none">@@VAL@@</div></details>
<script>
const D=@@DATA@@,S=D.summary,N=D.counts;
// sensor counts as CATEGORIES (every count gets a tick); a shape on a category axis is placed by the category INDEX
const NS=N.map(String),cfgx={type:'category',title:{text:'number of pressure sensors (halved at every step)'},automargin:true};
const vline=v=>[{type:'line',x0:N.indexOf(v),x1:N.indexOf(v),y0:0,y1:1,yref:'paper',line:{dash:'dot',color:'#757575'}}];
const optline=vline(D.opt);
const logticks=(lo,hi)=>{const t=[];for(let e=-2;e<7;e++)for(const m of [1,2,5]){const v=m*Math.pow(10,e);if(v>=lo/1.5&&v<=hi*1.5)t.push(v)}return t};
function lines(id,keys,title,ytitle,ymax,log){const all=keys.flatMap(([k])=>S[k]).filter(v=>v>0);
 Plotly.newPlot(id,keys.map(([k,n,c,d])=>({x:NS,y:S[k],mode:'lines+markers',name:n,line:{color:c,dash:d||'solid'}})),
 {title:{text:title,font:{size:14}},xaxis:cfgx,yaxis:Object.assign({title:{text:ytitle},automargin:true,range:ymax?[0,ymax]:undefined},
  log?{type:'log',tickvals:logticks(Math.min(...all),Math.max(...all))}:{}),shapes:optline,legend:{orientation:'h',y:-0.3},
  margin:{l:60,r:10,t:40,b:10}},{displaylogo:false})}
lines('g_det',[['Single: detected (%)','single leaks detected','#1f77b4'],['Multi: leaks detected (%)','multiple leaks detected','#ff7f0e'],
 ['Multi: hidden by other leaks (%)','multiple: hidden by other leaks','#d62728','dot']],'Detection','% of the test leaks',105);
lines('g_loc',[['Single: correct pipe (%)','single: correct pipe','#1f77b4'],['Single: correct or neighbouring pipe (%)','single: correct or neighbouring pipe','#1f77b4','dash'],
 ['Single: correct pipe and zone (%)','single: correct pipe + 20 % zone','#2ca02c'],['Multi: correct pipe (%)','multiple: correct pipe','#ff7f0e'],
 ['Multi: correct or neighbouring pipe (%)','multiple: correct or neighbouring pipe','#ff7f0e','dash']],'Localisation (share of all test leaks)','% of the test leaks',105);
lines('g_err',[['Single: median error ('+D.unit+')','single leaks','#1f77b4'],['Multi: median error ('+D.unit+')','multiple leaks','#ff7f0e']],
 'Median localisation error (detected leaks)','error along the pipes ('+D.unit+', log scale)',0,true);
lines('g_fa',[['Single: median delay (min)','median detection delay, single leaks (min)','#6a1b9a'],['False alarms on leak days (no leak running)','false alarms (no leak running)','#d62728','dot'],
 ['Duplicate alarms (leak reported again)','duplicate alarms (same leak reported again)','#8d6e63','dash']],'Detection delay, false and duplicate alarms','delay (minutes) / number of alarms');
// ---------- THE RELATION: sensors x leaks at the same time ----------
const RCL=['#1a9850','#fdae61','#d73027','#762a83'];
[['g_r1','localised','LOCALISATION vs number of sensors, per number of simultaneous leaks'],['g_r2','detected','DETECTION vs number of sensors, per number of simultaneous leaks']].forEach(([id,k,t])=>{
 const r=D.rel[k];Plotly.newPlot(id,r.rows.map((row,i)=>({x:r.cols.map(String),y:r.z[i],mode:'lines+markers+text',text:r.z[i].map(v=>v==null?'':v.toFixed(0)),textposition:'top center',
  textfont:{size:10,color:RCL[i%4]},name:row+' at the same time',line:{color:RCL[i%4],width:3},marker:{size:9}})),
  {title:{text:t,font:{size:13}},xaxis:{type:'category',title:'number of pressure sensors (halved at every step)'},yaxis:{title:{text:'% of the leaks'},range:[0,108]},
   legend:{orientation:'h',y:-0.3},margin:{l:55,r:10,t:40,b:10}},{displaylogo:false})});
[['g_r3','localised','Localisation (%): sensors x leaks at the same time'],['g_r4','detected','Detection (%): sensors x leaks at the same time']].forEach(([id,k,t])=>{
 const r=D.rel[k];Plotly.newPlot(id,[{type:'heatmap',z:r.z,x:r.cols.map(String),y:r.rows,zmin:0,zmax:100,colorscale:'RdYlGn',text:r.z.map(a=>a.map(v=>v==null?'':v.toFixed(0))),
  texttemplate:'%{text}',textfont:{size:13},hovertemplate:'%{y}, %{x} sensors: %{z:.0f} %<extra></extra>',colorbar:{title:'%',len:0.9}}],
  {title:{text:t,font:{size:13}},xaxis:{type:'category',title:{text:'number of pressure sensors'}},yaxis:{autorange:'reversed',title:{text:'leaks at the same time'},automargin:true},
   margin:{l:70,r:10,t:40,b:50}},{displaylogo:false})});
// ---------- key findings: heatmap + lines (correct pipe + zone) ----------
{const r=D.rel['pipe + zone'];
 Plotly.newPlot('g_k1',[{type:'heatmap',z:r.z,x:r.cols.map(String),y:r.rows,zmin:0,zmax:100,colorscale:'RdYlGn',text:r.z.map(a=>a.map(v=>v==null?'':v.toFixed(0)+'%')),
  texttemplate:'%{text}',textfont:{size:13},hovertemplate:'%{y}, %{x} sensors: %{z:.0f} %<extra></extra>',colorbar:{title:{text:'%'}}}],
  {title:{text:'Correct pipe + zone (% of all leaks)',font:{size:14}},xaxis:{type:'category',title:{text:'number of pressure sensors'}},
   yaxis:{autorange:'reversed',title:{text:'leaks at the same time'},automargin:true},margin:{l:70,r:10,t:40,b:50}},{displaylogo:false});
 Plotly.newPlot('g_k2',r.rows.map((row,i)=>({x:r.cols.map(String),y:r.z[i],mode:'lines+markers',name:row,line:{color:RCL[i%4],width:3},marker:{size:8}})),
  {title:{text:'Correct pipe + zone vs number of sensors',font:{size:14}},xaxis:{type:'category',title:{text:'number of pressure sensors (halved at every step)'}},
   yaxis:{title:{text:'% of all leaks'},range:[0,105]},legend:{orientation:'h',y:-0.3},margin:{l:55,r:10,t:40,b:10}},{displaylogo:false});}
// ---------- multiple leaks vs number of sensors ----------
const M=D.mt,cat=NS,mline=vline(D.mopt);
const catx=cfgx,pal=['#2c7fb8','#fd8d3c','#d62728','#756bb1'];
const mlay=(t,y,extra)=>Object.assign({title:{text:t,font:{size:14}},xaxis:catx,yaxis:y,shapes:mline,legend:{orientation:'h',y:-0.28},margin:{l:60,r:10,t:40,b:10}},extra||{});
Plotly.newPlot('g_m1',[
 {x:cat,y:M['Multi: correct or neighbouring pipe (%)'],mode:'lines+markers',name:'multiple leaks: correct or neighbouring pipe',line:{color:'#d62728',width:3}},
 {x:cat,y:M['Single: correct or neighbouring pipe (%)'],mode:'lines+markers',name:'single leaks: correct or neighbouring pipe',line:{color:'#1f77b4',width:3},
  fill:'tonexty',fillcolor:'rgba(214,39,40,0.12)'},
 {x:cat,y:M['Multi: detected (%)'],mode:'lines+markers',name:'multiple leaks: detected',line:{color:'#d62728',dash:'dash'}},
 {x:cat,y:M['Single: detected (%)'],mode:'lines+markers',name:'single leaks: detected',line:{color:'#1f77b4',dash:'dash'}}],
 mlay('Single vs multiple leaks (shaded = loss caused by the other leaks)',{title:{text:'% of the test leaks'},range:[0,105]}),{displaylogo:false});
Plotly.newPlot('g_m2',[{x:cat,y:M['Single: median error ('+D.unit+')'],mode:'lines+markers',name:'single leaks',line:{color:'#1f77b4',width:3}},
 {x:cat,y:M['Multi: median error ('+D.unit+')'],mode:'lines+markers',name:'multiple leaks',line:{color:'#d62728',width:3}}],
 mlay('Median localisation error (detected leaks)',{title:{text:'error along the pipes ('+D.unit+', log scale)'},type:'log',automargin:true,
  tickvals:logticks(Math.min(...M['Single: median error ('+D.unit+')'].filter(v=>v>0)),Math.max(...M['Multi: median error ('+D.unit+')'].filter(v=>v>0)))}),{displaylogo:false});
const LC=['#e31a1c','#ff7f00','#c51b8a','#8c510a'];          // one colour per leak of the day (map + detection chart)
// ---------- map helpers ----------
function ptAt(p,f){const g=D.geo[p],d=[];let L=0;for(let k=1;k<g.length;k++){const s=Math.hypot(g[k][0]-g[k-1][0],g[k][1]-g[k-1][1]);d.push(s);L+=s}
 let t=f*L;for(let k=1;k<g.length;k++){if(t<=d[k-1]||k==g.length-1){const r=d[k-1]?Math.min(t/d[k-1],1):0;return [g[k-1][0]+r*(g[k][0]-g[k-1][0]),g[k-1][1]+r*(g[k][1]-g[k-1][1])]}t-=d[k-1]}return g[g.length-1]}
function base(){const ps=Object.keys(D.geo),x=[],y=[];for(const p of ps){for(const q of D.geo[p]){x.push(q[0]);y.push(q[1])}x.push(null);y.push(null)}
 const tr=[{x,y,mode:'lines',line:{color:'#c3c9cf',width:1.5},name:'pipes',hoverinfo:'skip'}];
 const hv=ps.map(p=>ptAt(p,0.5));tr.push({x:hv.map(h=>h[0]),y:hv.map(h=>h[1]),text:ps.map(p=>'Pipe '+p),mode:D.labelpipes?'text':'markers',
  marker:{size:8,opacity:0},textfont:{size:10,color:'#78909c'},hoverinfo:'text',showlegend:false});return tr}
const nname=n=>{const k=D.nodes[n][2];return (k=='J'?'J ':k+' ')+n};
function sensorTraces(n){const js=new Set(D.sets[n]),all=Object.keys(D.nodes),on=all.filter(j=>js.has(j)),off=all.filter(j=>!js.has(j));
 return [{x:off.map(j=>D.nodes[j][0]),y:off.map(j=>D.nodes[j][1]),mode:'markers',marker:{size:off.map(j=>D.nodes[j][2]=='J'?5:10),
   symbol:off.map(j=>D.nodes[j][2]=='J'?'circle':'square'),color:'#9e9e9e'},text:off.map(nname),hoverinfo:'text',name:'junction without sensor / tank / reservoir'},
  {x:on.map(j=>D.nodes[j][0]),y:on.map(j=>D.nodes[j][1]),mode:'markers+text',marker:{size:11,color:'#08519c',line:{color:'#fff',width:1}},
   text:on.map(j=>j),textposition:'top right',textfont:{size:10,color:'#08306b'},hovertext:on.map(j=>'pressure sensor at '+nname(j)),hoverinfo:'text',name:'pressure sensor'}]}
const lay=(t,extra)=>Object.assign({title:{text:t,font:{size:14}},xaxis:{scaleanchor:'y',visible:false},yaxis:{visible:false},hovermode:'closest',
 legend:{orientation:'h',y:-0.02},margin:{l:10,r:10,t:40,b:10},plot_bgcolor:'#fff'},extra||{});
const pickS=document.getElementById('pickS');N.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n+(n==D.opt?' (optimum)':'');pickS.appendChild(o)});
function showSensors(){const n=pickS.value;Plotly.react('g_map',base().concat(sensorTraces(n)),lay(n+' pressure sensors'),{scrollZoom:true,displaylogo:false})}
pickS.onchange=showSensors;showSensors();
// ---------- test-day table + case view ----------
const pickN=document.getElementById('pickN'),pickK=document.getElementById('pickK');
N.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n+(n==D.opt?' (optimum)':'');pickN.appendChild(o)});
function shade(q){const a=[225,236,247],b=[8,48,107],t=Math.min(Math.max(q,0),1);return 'rgb('+a.map((v,k)=>Math.round(v+(b[k]-v)*Math.sqrt(t))).join(',')+')'}
function hh(h){const m=Math.round(h*60);return String(Math.floor(m/60)).padStart(2,'0')+':'+String(m%60).padStart(2,'0')}
function showTable(){const n=pickN.value,k=pickK.value,rows=D.tables[n][k],keep=D.cols.map((c,i)=>i).filter(i=>rows.some(r=>r[i]!==null&&r[i]!==''));
 let h='<table class="tbl"><thead><tr>'+keep.map(i=>'<th>'+D.cols[i]+'</th>').join('')+'</tr></thead><tbody>';
 rows.forEach((r,i)=>{h+='<tr data-i="'+i+'">'+keep.map(c=>{const v=r[c]===null?'-':r[c];const cl=/^(NO|FALSE)/.test(String(v))?' class="bad"':(/^(YES|NO LEAK)/.test(String(v))?' class="ok"':'');return '<td'+cl+'>'+v+'</td>'}).join('')+'</tr>'});
 document.getElementById('tab').innerHTML=h+'</tbody></table>';
 document.querySelectorAll('#tab tbody tr').forEach(r=>r.onclick=()=>showCase(+r.dataset.i));if(rows.length)showCase(0);}
function showCase(i){const n=pickN.value,k=pickK.value,c=D.cases[n][k][i];document.querySelectorAll('#tab tbody tr').forEach(r=>r.classList.toggle('sel',+r.dataset.i==i));
 const tr=base().concat(sensorTraces(n)),ann=[];let txt='';
 if(c.pipe&&c.zc){const best=Math.max(...c.zc);const E=D.edges;
  c.zc.forEach((q,z)=>{const f0=E[z]+0.006,f1=E[z+1]-0.006,pts=[...Array(12).keys()].map(s=>ptAt(c.pipe,f0+(f1-f0)*s/11));
   tr.push({x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),mode:'lines',line:{color:shade(q),width:13},name:'localised pipe '+c.pipe+' (zones coloured by confidence)',
    showlegend:z==0,legendgroup:'z',text:pts.map(()=>'zone '+D.zones[z]+': '+(100*q).toFixed(0)+'%'),hoverinfo:'text'});
   const m=ptAt(c.pipe,(E[z]+E[z+1])/2),a=ptAt(c.pipe,Math.max((E[z]+E[z+1])/2-0.03,0)),b=ptAt(c.pipe,Math.min((E[z]+E[z+1])/2+0.03,1));
   let nx=-(b[1]-a[1]),ny=b[0]-a[0];const l=Math.hypot(nx,ny)||1;nx/=l;ny/=l;if(nx<-1e-6||(Math.abs(nx)<=1e-6&&ny<0)){nx=-nx;ny=-ny}
   ann.push({x:m[0],y:m[1],showarrow:false,xshift:nx*16,yshift:ny*16,xanchor:nx>0.35?'left':(nx<-0.35?'right':'center'),yanchor:ny>0.35?'bottom':(ny<-0.35?'top':'middle'),
    text:(q==best?'<b>':'')+(100*q).toFixed(0)+'%'+(q==best?'</b>':''),font:{size:q==best?14:11,color:q==best?'#08306b':'#455a64'},
    bgcolor:'rgba(255,255,255,0.95)',bordercolor:q==best?'#08306b':'#b0bec5',borderwidth:q==best?2:1,borderpad:3})});
  const s0=ptAt(c.pipe,0);ann.push({x:s0[0],y:s0[1],showarrow:false,yshift:-14,text:nname(D.ends[c.pipe][0])+' (0 %)',font:{size:10,color:'#37474f'},bgcolor:'rgba(255,255,255,0.9)'});}
 c.leaks.forEach((l,j)=>{const p=ptAt(l[0],l[1]);const me=(c.j===j);
  tr.push({x:[p[0]],y:[p[1]],mode:'markers+text',marker:{size:me?22:16,symbol:'star',color:LC[j%4],line:{color:me?'#000':'#fff',width:1.5}},text:[c.leaks.length>1?String(j+1):''],
   textposition:'top left',textfont:{color:LC[j%4],size:13},name:me?'this leak (true position)':'other leak of the day',showlegend:j==(me?c.j:(c.j==0?1:0)),
   hovertext:['leak '+(j+1)+': pipe '+l[0]+' at '+(100*l[1]).toFixed(0)+'%, '+hh(l[2])+'-'+hh(l[3])],hoverinfo:'text'})});
 if(c.leaks.length){const l=c.leaks[c.j||0];txt='<b>Leak'+(c.leaks.length>1?' '+((c.j||0)+1)+' of '+c.leaks.length:'')+':</b> pipe <b>'+l[0]+'</b> at '+(100*l[1]).toFixed(0)+
  '% of its length, '+hh(l[2])+' - '+hh(l[3]);
  txt+=c.pipe?' &nbsp;|&nbsp; <b>Localised pipe:</b> <b style="color:#08306b">'+c.pipe+'</b>'+(c.zc?'<br><b>Confidence per zone:</b> '+D.zones.map((z,q)=>z+': <b>'+(100*c.zc[q]).toFixed(0)+'%</b>').join(' &nbsp; '):''):' &nbsp;|&nbsp; <b class="bad">not detected</b>';}
 else txt='<b>Day without a leak</b>'+(c.false.length?' - <b class="bad">false alarm at '+c.false.map(hh).join(', ')+'</b>':' - <b class="ok">no leak reported (correct)</b>');
 document.getElementById('info').innerHTML=txt;
 let xr,yr;if(c.pipe){const g=D.geo[c.pipe].concat(c.leaks.filter((l,j)=>j===(c.j||0)).map(l=>ptAt(l[0],l[1]))),xs=g.map(q=>q[0]),ys=g.map(q=>q[1]),el=document.getElementById('g_case');
  const W=Math.max(el.clientWidth-20,300),H=Math.max(el.clientHeight-110,250),cx=(Math.min(...xs)+Math.max(...xs))/2,cy=(Math.min(...ys)+Math.max(...ys))/2;
  const u=Math.max((Math.max(...xs)-Math.min(...xs))/(0.55*W),(Math.max(...ys)-Math.min(...ys))/(0.55*H),1e-9);xr=[cx-u*W/2,cx+u*W/2];yr=[cy-u*H/2,cy+u*H/2];}
 Plotly.react('g_case',tr,lay(n+' sensors: '+(c.pipe?'localised pipe '+c.pipe:'test day'),{annotations:ann,xaxis:{scaleanchor:'y',visible:false,range:xr,autorange:!xr},
  yaxis:{visible:false,range:yr,autorange:!yr}}),{scrollZoom:true,displaylogo:false});
 const day=D.dayof[n][k][i],sc=D.dayscore[n][day],sh=[];
 c.leaks.forEach((l,j)=>sh.push({type:'rect',x0:l[2],x1:l[3],y0:0,y1:1,yref:'paper',fillcolor:LC[j%4],opacity:j===(c.j||0)?0.30:0.14,
  line:{width:j===(c.j||0)?2:0,color:LC[j%4]}}));
 const lk=c.leaks.map((l,j)=>({x:[null],y:[null],mode:'lines',line:{color:LC[j%4],width:10},opacity:0.45,
  name:(c.leaks.length>1?'leak '+(j+1)+' (pipe '+l[0]+')':'leak (pipe '+l[0]+')')+' on: '+hh(l[2])+'-'+hh(l[3])}));
 Plotly.react('g_score',[{x:D.t,y:sc,mode:'lines',name:'real-time anomaly score / threshold',line:{color:'#263238'}},
  {x:[0,24],y:[1,1],mode:'lines',name:'alarm threshold',line:{dash:'dash',color:'#ff9800'}}].concat(lk),
  {title:{text:'Detection score of this day ('+n+' sensors)',font:{size:14}},shapes:sh,xaxis:{title:'time of day (h)',dtick:2,range:[0,24]},
   yaxis:{title:{text:'anomaly score / alarm threshold (log scale)'},type:'log',tickvals:[0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500],automargin:true},
   legend:{orientation:'h',y:-0.22},margin:{l:60,r:10,t:40,b:10}},{displaylogo:false});}
pickN.onchange=showTable;pickK.onchange=showTable;showTable();
// ---------- confusion matrices (training / validation / test) ----------
const ZL=D.zones,cmN=document.getElementById('cmN'),SETS=['training','validation','test'];
N.forEach(n=>{const o=document.createElement('option');o.value=n;o.textContent=n;cmN.appendChild(o)});
function hm(id,M,x,y,title,xt,yt,cs){const rs=M.map(r=>r.reduce((a,b)=>a+b,0)),P=M.map((r,i)=>r.map(v=>rs[i]?100*v/rs[i]:0));
 Plotly.react(id,[{type:'heatmap',z:P,x:x,y:y,colorscale:cs,zmin:0,zmax:100,showscale:false,text:M.map((r,i)=>r.map((v,j)=>v+'<br>'+P[i][j].toFixed(0)+'%')),
  texttemplate:'%{text}',textfont:{size:11},hovertemplate:'%{y} &rarr; %{x}: %{text}<extra></extra>'}],
  {title:{text:title,font:{size:13}},xaxis:{title:{text:xt},automargin:true,tickangle:-35},yaxis:{title:{text:yt},autorange:'reversed',automargin:true},
   margin:{l:80,r:10,t:40,b:10}},{displaylogo:false})}
function showCM(){const n=cmN.value;SETS.forEach(k=>{const c=D.conf[n][k],T=k[0].toUpperCase()+k.slice(1);
 hm('g_p_'+k,c.pipe,['leak, correct pipe','leak where there is none','no leak reported'],['leak','no leak'],T+' set: leak pipe ('+n+' sensors)','reported by the method','actual situation','Greens');
 hm('g_z_'+k,c.zone,ZL.concat(['neighb. pipe','wrong pipe','not detected']),ZL,T+' set: position ('+n+' sensors)','localised as (zone on the correct pipe / other)','true zone of the leak','Blues')})}
cmN.onchange=showCM;showCM();
// results summary filter (leak type, leak pipe, all positions / every zone)
const pickL=document.getElementById('pickL');[...new Set([...document.querySelectorAll('#rsum tbody tr')].map(r=>r.cells[2].textContent).filter(v=>v.startsWith('Pipe ')))]
 .sort((a,b)=>a.localeCompare(b,undefined,{numeric:true})).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent='leak on '+v.toLowerCase();pickL.appendChild(o)});
function filtR(){const t=document.getElementById('pickR').value,pz=document.getElementById('pickP').value,pl=pickL.value;
 document.querySelectorAll('#rsum tbody tr').forEach(r=>{const c=r.cells,pipe=c[2].textContent,pos=c[3].textContent;
  const okP=pl?pipe==pl:(pipe=='All pipes'||pipe=='-'),okZ=pl?true:(!pz||pos.startsWith('All')||pos=='-');
  r.style.display=((!t||c[1].textContent.startsWith(t))&&okP&&okZ)?'':'none';const k=parseFloat(c[7].textContent);
  c[7].className=isNaN(k)?'':(k>=90?'ok':(k<70?'bad':''))})}
document.getElementById('pickR').onchange=filtR;document.getElementById('pickP').onchange=filtR;pickL.onchange=filtR;filtR();
</script></body></html>"""
    rep = {"TITLE": f"{net.inp.stem} - leak detection and localisation with a reduced number of pressure sensors",
           "PLOTLY": plotly_tag, "COUNTS": " &rarr; ".join(str(n) for n in sets),
           "LEAK": f"{cfg['LEAK_SIZE_PCT'][0]:g}-{cfg['LEAK_SIZE_PCT'][1]:g} % of the average demand (emitter coefficient {crange[0]:.2f}-{crange[1]:.2f})",
           "SEED": str(tseed), "CARDS": "".join(f'<div class="card{" opt" if "Optimum" in k else ""}">{k}<b>{v}</b></div>' for k, v in cards.items()),
           "ANALYSIS": "".join(f"<li>{a}</li>" for a in analysis), "SUMMARY": tb(summary), "VAL": tb(val), "RSUM": tb(rsum),
           "KEYAN": "".join(f"<li>{a}</li>" for a in accs[6]), "KEYTAB": accs[7].to_html(index=False, classes="tbl", border=0, na_rep="-", escape=False),
           "SCEN": tb(accs[8]), "TTAB": tb(accs[0]), "VTAB": tb(accs[1]), "XTAB": tb(accs[2]), "FTTAB": tb(accs[3]), "FVTAB": tb(accs[4]), "FXTAB": tb(accs[5]),
           "NEV": f"{cfg.get('_NEV', ('-', '-'))[0]} / {cfg.get('_NEV', ('-', '-'))[1]} / {sum(ntest.values())}",
           "MIX": " / ".join(f"{v:.0%}" for v in cfg["TEST_MIX"].values()), "TUNED": tb(tuned_tab), "MOPT": str(int(mopt)),
           "MANALYSIS": "".join(f"<li>{a}</li>" for a in m_an),
           "RANALYSIS": "".join(f"<li>{a}</li>" for a in r_an), "MTABLE": tb(mt), "WHY": "".join(f"<li>{a}</li>" for a in why), "DATA": js}
    for k, v in rep.items():
        if k != "DATA": html = html.replace(f"@@{k}@@", v)
    html = html.replace("@@DATA@@", rep["DATA"])
    Path(path).write_text(html, encoding="utf-8")
    return path


README = [
    ("WHAT THIS PROGRAM DOES", "", "Leak detection (Matrix Profile, real time) and leak localisation (encoder-decoder embeddings) with ONLY "
                                   "pressure sensors at the junctions, for ANY EPANET network (change INPUT_FILE). The number of pressure "
                                   "sensors is halved step by step (all junctions -> ... -> 3) to see how the performance changes."),
    ("HOW", "1. Training data", "EPANET simulates leaks on every pipe (positions spread over the 5 zones, random size 2-7 % of the "
                                "average demand, start time, duration, demand level). Fingerprint = pressure change at the sensors while the "
                                "leak is on, with the effect of a uniform demand change removed, scaled to length 1."),
    ("", "2. Sensor placement", "Backward elimination: start with a sensor at every junction and remove, one at a time, the sensor whose "
                                "loss makes the training leaks least distinguishable. Sets are nested."),
    ("", "3. Encoder-decoder", "For every number of sensors: fingerprint -> encoder -> 16-number embedding (bottleneck) -> decoder rebuilds "
                               "the fingerprint; a classifier on the embedding learns the pipe. A new leak: most similar stored embeddings "
                               "-> pipe; kernel ridge regression on the embedding -> position -> zone + confidence of every zone."),
    ("", "3b. Model-based refinement", "MODEL_REFINEMENT=True: the hydraulic model, run under the conditions measured that day (tank "
                                     "levels, pump status), simulates a leak at 5 points of each of the most likely pipes of the "
                                     "encoder-decoder, estimates the leak size, then 10 points on the best 3 pipes; the pipe + point "
                                     "whose pressure pattern (whitened with the noise of the leak-free days) fits best is the result. "
                                     "Test_results_all_columns keeps the encoder-decoder answer (Enc-dec ...) for comparison."),
    ("", "4. Detection", "Pressure changes with the demand effect removed; real-time Matrix Profile against 8 leak-free reference days; "
                         "threshold = factor x the highest score on 40 other leak-free days. A new leak during an alarm = a new pressure "
                         "pattern; it is localised from the pressure change it adds. The detector settings are tuned for every number "
                         "of sensors on separate tuning days (sheet Detector_settings / Detector_tuning)."),
    ("", "5. Tests", "80 / 20 split: the test scenarios (20 % of all scenarios) are 40 % single-leak days, 40 % days with 2-3 "
                     "overlapping leaks on different pipes and 20 % days without a leak (TEST_MIX). Every test is evaluated with "
                     "every number of sensors."),
    ("sensor_study_results.xlsx", "Results_summary", "THE SUMMARY: sensors x leak type (single / multiple leaks on different "
                                                     "pipes / no leak) x where the leak was induced; leaks detected - correct pipe (%), "
                                                     "leaks localised - correct pipe + zone (%), accuracy retained after removing sensors, "
                                                     "change after halving the sensors (points)."),
    ("", "Results_by_pipe", "The same summary for every leak pipe separately (few test leaks per pipe: a drill-down)."),
    ("", "Why_not_correct", "Every test leak in one category: correct pipe + zone / correct pipe, next zone / zone further away / "
                            "neighbouring pipe / wrong pipe / not detected (% of the test leaks)."),
    ("", "Detection_by_leak_size", "Share of the leaks detected per leak-size quarter (emitter coefficient) and per distance of the leak pipe from a tank / reservoir."),
    ("", "Scenarios", "Days and leaks of every scenario type (single / multiple overlapping / multiple non-overlapping / "
                      "leak-free) in the training, validation and test set."),
    ("", "Training_details / Validation_details / Test_details", "Simple accuracy tables (rows = metric, columns = number of sensors): "
                                                                  "one pipe and one zone accuracy over all leaks of the set, complete method."),
    ("", "Full_training_report / Full_validation_report / Full_test_report", "All details of each set (detection, pipe, zone, "
                                                                              "neighbouring pipe, error, encoder-decoder alone, false alarms)."),
    ("", "Train_valid_leak_results", "One row per training / validation leak (re-simulated as a monitoring day) and number of sensors."),
    ("", "Performance_vs_sensors", "All detection and localisation numbers for every number of sensors."),
    ("", "Relation_localisation / _detection / _exact_pipe", "THE RELATION: rows = number of leaks at the same time (1 = single-leak "
                                                              "days, 2 / 3 = multi-leak days), columns = number of pressure sensors, "
                                                              "value = % of the leaks localised / detected (figure relation_sensors_vs_leaks.png)."),
    ("", "Multi_leak_vs_sensors", "MULTIPLE LEAKS vs number of sensors: single vs multi-leak performance, 1st / 2nd / 3rd leak of the "
                                  "day, days with 2 / 3 leaks (also figure multi_leak_vs_sensors.png and section 2 of the report)."),
    ("", "Multi_leak_by_leak_order / _by_day_size", "Multi-leak days: 1st / 2nd / 3rd leak of the day; days with 2 vs 3 leaks."),
    ("", "Sensor_sets", "X = junction has a pressure sensor in that set."),
    ("", "Test_results / Test_results_all_columns", "One row per test leak (or leak-free day) and number of sensors."),
    ("", "Model_validation", "Encoder-decoder validation (positions never used for training) for every number of sensors."),
    ("training_data.xlsx", "Training_leaks / Training_summary", "Every training leak simulation; number of leaks per pipe."),
    ("training_raw_pressure.csv", "", "Raw EPANET pressures of the training leaks (1 h before the leak until its end)."),
    ("test_raw_pressure.csv", "", "Raw EPANET pressures of every test day (00:00-24:00)."),
    ("TERMS", "Detected", "YES = reported in real time while the leak was on; 'hidden by other leaks' = an alarm was already on and the "
                          "new leak did not add a recognisable new pattern."),
    ("", "Correct or neighbouring pipe", "Localised pipe = leaking pipe, or a pipe directly connected to it."),
    ("", "Confidence 0-20% ... 80-100%", "Confidence that the leak is in that zone of the localised pipe (sums to 100 %)."),
    ("", "Error", "Distance along the pipes between the true and the localised leak point."),
    ("", "False alarm / duplicate alarm", "False alarm = a leak reported while no leak is running. Duplicate alarm = a leak that was already "
                                          "detected is reported again (e.g. when its pressure pattern shifts with the hourly demand)."),
    ("", "Optimum", "Fewest sensors that keep >= 90 % of the all-sensor detection (single + multi), exact-pipe localisation "
                    "(single) and correct-or-neighbouring-pipe localisation (single + multi)."),
]


def run_study(inp, report_name=None, out_dir=None, **settings):
    """Run the complete study for ONE network (used by program1/2/3.py and main.py).
    inp: path of the EPANET .inp file;  report_name: e.g. "program1.html" (a copy of the HTML report with this name is
    written next to the results folder);  out_dir: folder for all results (default: <network>_pressure_sensor_study
    next to the .inp file);  settings: any CFG setting to change, e.g. TEST_SEED=79995."""
    global INPUT_FILE, OUTPUT_DIR
    INPUT_FILE = Path(inp).resolve()
    if not INPUT_FILE.exists(): raise FileNotFoundError(f"network file not found: {INPUT_FILE}")
    OUTPUT_DIR = Path(out_dir).resolve() if out_dir else INPUT_FILE.parent / f"{INPUT_FILE.stem}_pressure_sensor_study"
    import copy
    global _CFG_DEFAULT
    if "_CFG_DEFAULT" not in globals(): _CFG_DEFAULT = copy.deepcopy(CFG)
    unknown = [k for k in settings if k not in _CFG_DEFAULT]
    if unknown: raise KeyError(f"unknown setting(s): {unknown}")
    CFG.clear(); CFG.update(copy.deepcopy(_CFG_DEFAULT)); CFG.update(copy.deepcopy(settings))   # every network: defaults + its own settings
    main()
    if report_name:
        src = OUTPUT_DIR / "sensor_study_report.html"; dst = INPUT_FILE.parent / report_name
        shutil.copyfile(src, dst); print(f"\nREPORT: {dst}")
        return dst
    return OUTPUT_DIR / "sensor_study_report.html"

# =====================================================================================================================
# LAUNCHER: the networks to study.  One line per network: (inp file, report name).
# The .inp file is taken from the folder of main.py; the benchmark networks listed here are copied automatically
# from the EPyT package if they are not in the folder.  Add your own networks (e.g. your guide's list) as new lines.
# =====================================================================================================================
BIG = dict(MAX_TEST_DAYS=1160, EVAL_DAYS=dict(training=1000, validation=1000), SENSOR_SELECT_MAX=(1500, 6000))
NETWORKS = [
    # (inp file, report, settings).  The .inp files are INSIDE this file (EMBEDDED at the end) and are written next to
    # main.exe on the first run - nothing else is needed.  Settings: see the SETTINGS section (CFG) above.
    ("Modena.inp",          "Modena.html",    BIG),      # 268 junctions, 317 pipes, 4 reservoirs      (Bragalli et al. 2008)
    ("L-TOWN.inp",          "L-Town.html",    BIG),      # 782 junctions, 905 pipes, 1 tank, 1 pump    (BattLeDIM 2020)
    ("L-TOWN_v2_Model.inp", "L-Town_v2_Model.html", BIG),  # L-Town v2, MODEL network (BattLeDIM dataset generator)
    ("L-TOWN_v2_Real.inp",  "L-Town_v2_Real.html",  BIG),  # L-Town v2, REAL network (original 177 MB file)
    ("Hanoi.inp",           "Hanoi.html",     dict()),   #  31 junctions,  34 pipes, gravity network   (Fujiwara & Khang 1990)
    ("D-Town.inp",          "D-Town.html",    BIG),      # 399 junctions, 443 pipes, 7 tanks, 11 pumps (Battle of Water Networks II)
]
# BIG networks: test days capped at 1160 (as Net3), 1000 training + 1000 validation days evaluated (instead of all),
# sensor placement on a subsample of the training leaks - otherwise one network would take days.  To evaluate EVERY
# training / validation scenario change EVAL_DAYS to dict(training="all", validation="all") (much longer).
FAST_SETTINGS = dict(EVAL_DAYS=dict(training=300, validation=300), MAX_TEST_DAYS=400)   # "fast": quick check

PROGRAM_TEMPLATE = '''"""{prog}.py - leak study for {inp}  ->  {report}   (created by main.exe)
Run:  python3 {prog}.py        (or: python3 {prog}.py fast)       same as:  ./{main} {no}
Needs {main} in the same folder (it contains the network file, the settings and the packages set-up).
Results: {report} + the folder {stem}_pressure_sensor_study next to {main}."""
import os, sys, runpy
from pathlib import Path

if __name__ == "__main__":
    main = str(Path(__file__).resolve().parent / "{main}")
    if os.name == "posix":            # Linux / macOS: main.exe prepares Python + packages, then runs
        os.execv("/bin/sh", ["/bin/sh", main, "{no}"] + sys.argv[1:])
    sys.argv = [main, "{no}"] + sys.argv[1:]     # Windows: run main.exe with this Python (packages installed by it)
    runpy.run_path(main, run_name="__main__")
'''


def base_dir():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def find_network(name):
    """The .inp file next to main.exe; if missing: written from the copy EMBEDDED in this file, or taken from the
    benchmark networks of the EPyT package."""
    p = base_dir() / name
    if p.exists(): return p
    for d in (base_dir() / "networks", base_dir().parent / "networks"):     # repository layout: networks/<name>
        if (d / name).exists(): return d / name
        if (d / (name + ".xz")).exists():                                  # large file stored compressed (L-TOWN_v2_Real)
            import lzma
            p.write_bytes(lzma.decompress((d / (name + ".xz")).read_bytes()))
            print(f"    {name} unpacked from networks/{name}.xz"); return p
    if name in EMBEDDED:
        import zlib, lzma, base64
        e = EMBEDDED[name]
        if isinstance(e, tuple) and e[0] == "lzma": p.write_bytes(lzma.decompress(base64.b64decode(e[1])))
        else: p.write_bytes(zlib.decompress(base64.b64decode("".join(e))))
        print(f"    {name} written next to main.exe (included in main.exe)"); return p
    import epyt
    for q in Path(epyt.__file__).parent.rglob(name):
        if "_temp" not in q.stem:
            shutil.copyfile(q, p); print(f"    {name} copied from the EPyT benchmark networks"); return p
    raise FileNotFoundError(f"{name} not found - put it next to main.exe")


def write_programs():
    """One small script per network: Modena.py, L-Town.py, ... (run one network with: python3 Modena.py)."""
    if getattr(sys, "frozen", False) or not Path(__file__).name.endswith(".exe"): return   # only main.exe writes them
    me = Path(__file__).name
    for no, (inp, report, _) in enumerate(NETWORKS, 1):
        prog = Path(report).stem
        (base_dir() / f"{prog}.py").write_text(PROGRAM_TEMPLATE.format(prog=prog, inp=inp, report=report, main=me, no=no, stem=Path(inp).stem),
                                               encoding="utf-8")


def run_all(chosen, fast=False):
    done = []
    for inp, report, sett in chosen:
        print("\n" + "#" * 100 + f"\n#  {inp}  ->  {report}\n" + "#" * 100); t0 = time.time()
        try:
            run_study(find_network(inp), report, **dict(sett, **(FAST_SETTINGS if fast else {})))
            done.append(f"{report} ({(time.time() - t0) / 60:.1f} min)")
        except Exception as e:
            print(f"\n!! {inp} stopped: {type(e).__name__}: {e}")
    print("\nFINISHED:", ", ".join(done) if done else "nothing")


def menu():
    print("LEAK DETECTION + LOCALISATION WITH A REDUCED NUMBER OF PRESSURE SENSORS\n")
    print("  No  Town               network file           report                  run it directly with")
    for i, (inp, report, _) in enumerate(NETWORKS, 1):
        here = (base_dir() / inp).exists() or inp in EMBEDDED
        town = Path(report).stem.replace("_", " ")
        print(f"  {i:2d}  {town:<18s} {inp:<22s} {report:<23s} ./main.exe {i}   or   python3 {Path(report).stem}.py"
              + ("" if here else "   (file missing - put it next to main.exe)"))
    print("   A  all towns, one after another (large networks take hours each)   = ./main.exe all\n   Q  quit")
    print("  (full instructions: head -70 main.exe)")
    c = input("\nChoice (e.g. 1, 2 4, or A; add F for the fast setting, e.g. 4 F): ").replace(",", " ").strip().upper().split()
    if not c or c == ["Q"]: return [], False
    fast = "F" in c; c = [x for x in c if x != "F"]
    return (list(NETWORKS) if "A" in c else [NETWORKS[int(x) - 1] for x in c if x.isdigit() and 0 < int(x) <= len(NETWORKS)]), fast


# =====================================================================================================================
# NETWORK FILES: in the repository they are in the folder networks/ (main.exe carries them inside, see tools/build_main_exe.py)
# =====================================================================================================================
EMBEDDED = {}


if __name__ == "__main__":
    import multiprocessing; multiprocessing.freeze_support()
    write_programs()
    args = [a.upper() for a in sys.argv[1:]]
    if args:      # python main.py all | python main.py 1 2 | python main.py 13 fast
        fast = "FAST" in args or "F" in args
        chosen = list(NETWORKS) if "ALL" in args else [NETWORKS[int(a) - 1] for a in args if a.isdigit() and 0 < int(a) <= len(NETWORKS)]
    else:
        chosen, fast = menu()
    if chosen: run_all(chosen, fast)
    elif args:
        print("Unknown choice " + " ".join(sys.argv[1:]) + " - type a town number, e.g.:")
        for no, (inp, report, _) in enumerate(NETWORKS, 1): print(f"    ./main.exe {no}   -> {Path(report).stem}")
        print("    ./main.exe all -> every town")
