#!/usr/bin/env python3
"""
4-AXIS PIPE & TUBE PROFILE CUTTER - G-CODE GENERATOR  (Streamlit)
=================================================================
Generates X / Y / Z / A G-code for a plasma pipe and tube cutting machine
(rotary chuck on A, torch on X-Y-Z), with a live Matplotlib preview and a
one-tap .gcode download. Works on phone and desktop browsers.

RUN LOCALLY
-----------
    pip install streamlit numpy matplotlib
    streamlit run cnc_4axis_pipe_tube_gcode_generator.py

    The app opens at http://localhost:8501.
    To use it from your phone on the same Wi-Fi:
        streamlit run cnc_4axis_pipe_tube_gcode_generator.py --server.address 0.0.0.0
    then browse to http://<computer-ip>:8501 on the phone.

DEPLOY FREE (Streamlit Community Cloud)
---------------------------------------
    1. Put this file and a requirements.txt (three lines: streamlit, numpy,
       matplotlib) in a GitHub repository.
    2. Go to share.streamlit.io -> New app -> pick the repo and this file.
    3. Open the app URL on your phone and use "Add to Home Screen".

COORDINATE MODEL
----------------
    X  longitudinal position along the tube axis (mm)
    Y  transverse shift of the torch (mm). Zero for round pipe; follows the
       contact point across flats and corners of square/rectangular tube
    Z  torch height above the machine datum: tube centre-line height
       + radial distance to the surface at the contact point + standoff
    A  chuck angle in degrees, 0 -> 360 over one full contour

    The torch always stays vertical. The chuck angle A is set to the angle
    of the local surface normal, so the torch is square to the surface at
    every point. On a round pipe Y stays 0 and Z is constant. On a
    rectangular tube A holds still on a flat while Y travels across it, and
    at each corner A, Y and Z move together around the corner radius.

    The cut line is the intersection of the tube with a mating surface:
      * Saddle: a run pipe of given diameter meeting the tube at the
        intersection angle (90 deg = T-joint).
      * Angled plane (miter): a flat plane at the intersection angle to the
        tube axis (90 deg = square cut).
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import streamlit as st
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)

MIN_CORNER_R = 0.5  # mm; a perfectly sharp corner would need an instant 90 deg A move


# ======================================================================
# 1. PARAMETERS
# ======================================================================
@dataclass(frozen=True)
class Params:
    # material
    shape: str = "round"          # "round" | "rect"
    od: float = 60.3
    width: float = 60.0
    height: float = 40.0
    wall: float = 3.0
    corner_r: float = 4.5
    # cut geometry
    cut_type: str = "saddle"      # "saddle" | "miter"
    angle: float = 90.0           # intersection angle, degrees
    run_od: float = 114.3
    tilt: float = 0.0             # cut orientation (A angle), degrees
    ref: float = 0.0              # fit-up depth fraction: 0 = OD, 0.5 = mid-wall, 1 = ID
    keep: int = -1                # -1 = chuck side kept (waste toward +X), +1 = free end kept
    x0: float = 150.0             # X of the longest point of the part
    # plasma
    kerf: float = 1.5
    bevel: float = 0.0
    bevel_sig: int = -1           # -1 weld prep (ID edge reaches past OD edge), +1 reverse
    cut_h: float = 3.0
    pierce_h: float = 4.5
    pierce_delay: float = 0.5
    lead_in: float = 3.0
    lead_out: float = 3.0
    overlap: float = 0.0
    torch_on: str = "M3"
    torch_off: str = "M5"
    # machine
    feed: float = 1500.0
    plunge: float = 2000.0
    corner_pct: float = 60.0
    a_max: float = 180.0          # deg/s
    feed_mode: str = "G93"        # "G93" inverse time | "G94" units per minute
    safe_clear: float = 25.0
    segments: int = 360
    tol: float = 0.02
    a_end: str = "rewind"         # "rewind" | "zero" | "leave"
    # calibration
    x_off: float = 0.0
    y_off: float = 0.0
    z_center: float = 0.0
    a_off: float = 0.0
    a_rev: bool = False
    a_scale: float = 1.0


# ======================================================================
# 2. TUBE CROSS-SECTION (rounded rectangle; round pipe is a = b = rc)
# ======================================================================
class Profile:
    """Outer cross-section parametrised by perimeter arc length p.

    p = 0 is the middle of the top flat and p grows toward +u (clockwise when
    looking along +X). psi is the outward-normal angle from +v toward +u in
    degrees, unwrapped so it keeps growing by 360 per lap.
    """

    def __init__(self, a: float, b: float, rc: float):
        rc = float(min(rc, a, b))
        self.a, self.b, self.rc = float(a), float(b), rc
        fa, fb = a - rc, b - rc
        arc = rc * math.pi / 2.0
        self.len = np.array([fa, arc, 2 * fb, arc, 2 * fa, arc, 2 * fb, arc, fa])
        self.s0 = np.concatenate([[0.0], np.cumsum(self.len)[:-1]])
        self.length = float(self.len.sum())
        self.is_arc = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], bool)
        # flat: start point; arc: centre
        self.ref = np.array([[0, b], [fa, fb], [a, fb], [fa, -fb], [fa, -b],
                             [-fa, -fb], [-a, -fb], [-fa, fb], [-fa, b]], float)
        self.dirv = np.array([[1, 0], [0, 0], [0, -1], [0, 0], [-1, 0],
                              [0, 0], [0, 1], [0, 0], [1, 0]], float)
        self.psi0 = np.array([0, 0, 90, 90, 180, 180, 270, 270, 360], float)

    def seg_index(self, p):
        p = np.asarray(p, float)
        pm = p - np.floor(p / self.length) * self.length
        return np.clip(np.searchsorted(self.s0, pm, side="right") - 1, 0, 8)

    def eval(self, p):
        """Return (u, v, psi_deg) of the outer surface at perimeter position p."""
        p = np.asarray(p, float)
        k = np.floor(p / self.length)
        pm = p - k * self.length
        idx = np.clip(np.searchsorted(self.s0, pm, side="right") - 1, 0, 8)
        s = pm - self.s0[idx]
        arc = self.is_arc[idx]
        psi = self.psi0[idx] + np.where(arc, np.degrees(s / max(self.rc, 1e-12)), 0.0)
        pr = np.radians(psi)
        u = np.where(arc, self.ref[idx, 0] + self.rc * np.sin(pr), self.ref[idx, 0] + self.dirv[idx, 0] * s)
        v = np.where(arc, self.ref[idx, 1] + self.rc * np.cos(pr), self.ref[idx, 1] + self.dirv[idx, 1] * s)
        return u, v, psi + 360.0 * k


# ======================================================================
# 3. MATHEMATICAL ENGINE
# ======================================================================
def compute_toolpath(P: Params) -> SimpleNamespace:
    """4-axis simultaneous kinematics for one full contour.

    Steps
      1. Cut curve x(p): where the tube surface meets the saddle / miter plane.
      2. Kerf: offset that curve into the waste, in the unrolled surface plane,
         by half the kerf (widened by 1/cos of the bevel seen across the path).
      3. Bevel: fixed-tilt torch in the X-Z plane -> shift nozzle X by the
         standoff and wall-depth terms.
      4. Pose: A = surface-normal angle, Y = lateral contact position,
         Z = contact height + standoff. Then apply axis calibration.
    """
    # ---- cross-section ------------------------------------------------
    if P.shape == "round":
        a = b = rc = P.od / 2.0
    else:
        a, b = P.width / 2.0, P.height / 2.0
        rc = min(max(P.corner_r, MIN_CORNER_R), a, b)
    t = P.wall
    if not (0.0 < t < min(a, b)):
        raise ValueError("Wall thickness must be above 0 and below half of the smallest dimension.")
    if P.feed <= 0 or P.a_max <= 0 or P.plunge <= 0:
        raise ValueError("Feed rates and the maximum A speed must be above 0.")
    prof = Profile(a, b, rc)
    L = prof.length
    r_max = math.hypot(a - rc, b - rc) + rc

    dirn = -1 if P.a_rev else 1          # physical chuck sense for increasing A
    sw = -P.keep                         # +1: waste toward +X
    saddle = P.cut_type == "saddle"
    gam = math.radians(P.angle)
    sing, cosg = math.sin(gam), math.cos(gam)
    Rr = P.run_od / 2.0 if saddle else 0.0
    phi = math.radians(dirn * P.tilt)    # cut orientation in the tube frame
    sin_t, cos_t = math.sin(phi), math.cos(phi)
    d_ref = P.ref * t
    beta = math.radians(P.bevel)
    tanb, sinb, cosb = math.tan(beta), math.sin(beta), math.cos(beta)
    sig = P.bevel_sig

    # ---- 1. cut curve -------------------------------------------------
    def cut_f(p_arr):
        u, v, psi = prof.eval(p_arr)
        pr = np.radians(psi)
        ur = u - np.sin(pr) * d_ref      # point on the reference surface
        vr = v - np.cos(pr) * d_ref
        s_t = ur * sin_t + vr * cos_t    # along the orientation direction
        s_n = ur * cos_t - vr * sin_t    # across it
        if saddle:
            arg = Rr * Rr - s_n * s_n
            if arg.min() < -1e-3 * max(Rr * Rr, 1.0):
                raise ValueError("The run pipe is too small: it must be wider than the branch tube "
                                 "measured across the cut orientation. Increase the run pipe diameter.")
            root = np.sqrt(np.maximum(arg, 0.0))
        else:
            root = 0.0
        return (s_t * cosg - root) / sing

    dense = cut_f(np.linspace(0.0, L, 2001))
    f_max, f_min = float(dense.max()), float(dense.min())

    def cut_x(p_arr):
        return P.x0 - sw * (f_max - cut_f(p_arr))

    # ---- sampling along the traversal distance q ---------------------
    q_end = L + max(P.overlap, 0.0)
    q_uni = np.linspace(0.0, q_end, max(int(round(P.segments * q_end / L)), 16) + 1)
    edges = np.append(prof.s0, L)
    extra = np.concatenate([np.linspace(edges[i], edges[i + 1], 17 if prof.is_arc[i] else 2) for i in range(9)])
    base = np.mod(dirn * extra, L)
    q = np.concatenate([q_uni, base, base + L, [0.0, L, q_end]])
    q = q[(q >= -1e-12) & (q <= q_end + 1e-9)]
    q = np.unique(np.round(q, 8))
    q = q[np.concatenate([[True], np.diff(q) > 1e-6])]
    q[0], q[-1] = 0.0, q_end
    p = dirn * q

    x_nom = cut_x(p)
    u_nom, v_nom, psi_nom = prof.eval(p)
    A_nom = dirn * psi_nom

    # ---- 2. kerf offset in the unrolled surface plane ----------------
    dxdp = np.gradient(x_nom, p)
    nrm = np.sqrt(1.0 + dxdp ** 2)
    cos_path = 1.0 / nrm
    beff = np.degrees(np.arctan(tanb * cos_path))                 # bevel normal to the path
    d_k = 0.5 * P.kerf * np.sqrt(1.0 + (tanb * cos_path) ** 2)
    nwp, nwx = -sw * dxdp / nrm, sw / nrm                         # unit normal into the waste
    p_k = p + d_k * nwp
    x_k = x_nom + d_k * nwx

    # ---- 3 + 4. torch pose --------------------------------------------
    def pose(pk, xk, h):
        u, v, psi = prof.eval(pk)
        pr = np.radians(psi)
        Yg = u * np.cos(pr) - v * np.sin(pr)                       # lateral position
        Zg = u * np.sin(pr) + v * np.cos(pr)                       # height above the axis
        Xn = xk + sw * sig * (tanb * d_ref + sinb * P.cut_h)        # nozzle X with bevel shift
        return Xn, Yg, Zg + h * cosb, dirn * psi, Zg, u, v, psi

    def calibrate(Xn, Yg, Zn, A_deg):
        return (P.x_off + Xn, P.y_off + Yg, P.z_center + Zn, P.a_scale * (A_deg + P.a_off))

    Xn, Yg, Zn, A_deg, Zg, uk, vk, psi_k = pose(p_k, x_k, P.cut_h)
    X, Y, Z, A = calibrate(Xn, Yg, Zn, A_deg)

    # pierce and lead-out points (straight, into the waste)
    pp, xp = p_k[0] + P.lead_in * nwp[0], x_k[0] + P.lead_in * nwx[0]
    po, xo = p_k[-1] + P.lead_out * nwp[-1], x_k[-1] + P.lead_out * nwx[-1]
    pi_hi = pose(np.array([pp]), np.array([xp]), P.pierce_h)
    pi_lo = pose(np.array([pp]), np.array([xp]), P.cut_h)
    po_lo = pose(np.array([po]), np.array([xo]), P.cut_h)
    c_hi, c_lo, c_out = (calibrate(*(float(r[0]) for r in s[:4])) for s in (pi_hi, pi_lo, po_lo))

    # ---- segment lengths, feed and time --------------------------------
    ds = np.hypot(np.diff(p_k), np.diff(x_k))                      # surface path length
    dA = np.abs(np.diff(A_deg))
    seg_idx = prof.seg_index(0.5 * (p[:-1] + p[1:]))
    corner = prof.is_arc[seg_idx] if P.shape == "rect" else np.zeros(len(ds), bool)
    f_seg = P.feed * np.where(corner, P.corner_pct / 100.0, 1.0)
    t_seg = np.maximum(np.maximum(ds / f_seg, dA / (P.a_max * 60.0)), 1e-9)   # minutes

    # ---- plot helpers ---------------------------------------------------
    zones = []
    if P.shape == "rect":
        for i in range(9):
            if prof.is_arc[i] and prof.len[i] > 1e-9:
                lo, hi = prof.s0[i], prof.s0[i] + prof.len[i]
                lo, hi = (lo, hi) if dirn > 0 else (L - hi, L - lo)
                for lap in (0.0, L):
                    if lo + lap < q_end:
                        zones.append((lo + lap, min(hi + lap, q_end)))
    marks = []
    for target in (0, 90, 180, 270, 360):
        m = np.abs(A_nom - target) < 1e-6
        qm = q[m].mean() if m.any() else float(np.interp(target, A_nom, q))
        marks.append((qm, target))

    warnings = []
    if P.safe_clear < P.pierce_h + 2.0:
        warnings.append("Safe Z clearance is very close to the pierce height. Increase it.")
    if P.bevel > 0 and beff.max() - beff.min() > 3.0:
        warnings.append(f"With a fixed-tilt torch the real bevel across the cut varies from {beff.min():.1f} to "
                        f"{beff.max():.1f} deg. A true constant bevel needs a tilt axis.")
    if P.shape == "rect" and P.corner_r < MIN_CORNER_R:
        warnings.append(f"Corner radius raised to {MIN_CORNER_R} mm (a sharp corner needs an instant 90 deg A move).")
    if P.shape == "rect" and P.ref > 0 and rc < t:
        warnings.append("Corner radius is smaller than the wall thickness; mid-wall and inner references are approximate at the corners.")

    return SimpleNamespace(
        prof=prof, sw=sw, dirn=dirn, r_max=r_max, L=L, f_max=f_max, f_min=f_min, cut_x=cut_x,
        q=q, p=p, x_nom=x_nom, u_nom=u_nom, v_nom=v_nom, A_nom=A_nom, p_k=p_k, x_k=x_k, uk=uk, vk=vk, psi_k=psi_k,
        Xn=Xn, Yg=Yg, Zg=Zg, Zn=Zn, A_deg=A_deg, X=X, Y=Y, Z=Z, A=A, beff=beff,
        pierce=dict(aim=(pp, xp), hi=c_hi, lo=c_lo), lead_out=dict(aim=(po, xo), pose=c_out),
        ds=ds, dA=dA, corner=corner, t_seg=t_seg, zones=zones, marks=marks, warnings=warnings,
        z_safe=P.z_center + r_max + P.safe_clear,
        cut_len=float(ds.sum()), total_min=float(t_seg.sum() + P.pierce_delay / 60.0),
        x_origin=P.x0 - sw * f_max, run_axis=(sw * cosg, sing * math.sin(phi), sing * math.cos(phi)),
    )


# ======================================================================
# 4. G-CODE
# ======================================================================
def rdp_keep(M: np.ndarray, tol: float, forced) -> np.ndarray:
    """Ramer-Douglas-Peucker thinning. Returns the indices of points to keep."""
    n = len(M)
    keep = np.zeros(n, bool)
    keep[[0, n - 1]] = True
    keep[np.asarray(forced, int)] = True
    idx = np.flatnonzero(keep)
    stack = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        seg = M[j] - M[i]
        d = M[i + 1:j] - M[i]
        l2 = float(seg @ seg)
        if l2 < 1e-18:
            dist = np.linalg.norm(d, axis=1)
        else:
            tp = np.clip(d @ seg / l2, 0.0, 1.0)
            dist = np.linalg.norm(d - np.outer(tp, seg), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return np.flatnonzero(keep)


def _cm(text: str) -> str:
    return "(" + text.replace("(", "[").replace(")", "]") + ")"


def build_gcode(P: Params, T: SimpleNamespace):
    """Return (gcode_text, number_of_G1_moves)."""
    n = len(T.q)
    inv = P.feed_mode == "G93"
    M = np.column_stack([T.X, T.Y, T.Z, T.A_deg * math.radians(1.0) * T.r_max])
    forced = np.flatnonzero(T.corner[1:] != T.corner[:-1]) + 1
    keep = rdp_keep(M, P.tol, forced) if P.tol > 0 else np.arange(n)
    cum_l = np.concatenate([[0.0], np.cumsum(T.ds)])
    cum_t = np.concatenate([[0.0], np.cumsum(T.t_seg)])

    out, last, last_f, n_g1 = [], {}, [None], [0]

    def words(vals):
        w = []
        for ax in "XYZA":
            if ax in vals:
                s = f"{vals[ax] + 0.0:.3f}"
                s = "0.000" if s == "-0.000" else s
                if last.get(ax) != s:
                    last[ax] = s
                    w.append(ax + s)
        return w

    def g0(**vals):
        w = words(vals)
        if w:
            out.append("G0 " + " ".join(w))

    def g1(length, minutes, **vals):
        w = words(vals)
        if not w:
            return
        if inv:
            w.append(f"F{1.0 / max(minutes, 1e-9):.4f}")
        else:
            fs = f"{length / max(minutes, 1e-9):.0f}"
            if last_f[0] != fs:
                last_f[0] = fs
                w.append("F" + fs)
        out.append("G1 " + " ".join(w))
        n_g1[0] += 1

    shape = (f"Round pipe OD {P.od:g} x {P.wall:g} wall" if P.shape == "round"
             else f"Rect tube {P.width:g} x {P.height:g} x {P.wall:g} wall, corner R {T.prof.rc:g}")
    cut = (f"Saddle onto {P.run_od:g} OD run pipe, intersection {P.angle:g} deg" if P.cut_type == "saddle"
           else f"Angled plane cut, intersection {P.angle:g} deg")
    out += [
        _cm("4-AXIS PIPE/TUBE PROFILE CUT - PLASMA"),
        _cm("Generated " + datetime.now().strftime("%Y-%m-%d %H:%M")),
        _cm(shape), _cm(cut),
        _cm(f"Kerf {P.kerf:g} mm, bevel {P.bevel:g} deg, cut height {P.cut_h:g}, pierce height {P.pierce_h:g}"),
        _cm("X=longitudinal Y=transverse Z=standoff height A=chuck degrees"),
        _cm("Feed G93 inverse time: F = 1/minutes per block" if inv else f"Feed G94: mm/min, cut feed {P.feed:g}"),
        _cm(f"Cut length {T.cut_len:.1f} mm, est. time {T.total_min * 60:.0f} s"),
        _cm("DRY-RUN WITH THE TORCH DISABLED BEFORE CUTTING"),
        f"G21 G90 G17 G40 G49 {'G93' if inv else 'G94'}",
    ]
    pi_hi, pi_lo, po = T.pierce["hi"], T.pierce["lo"], T.lead_out["pose"]
    out.append(_cm("Move to pierce point"))
    g0(Z=T.z_safe)
    g0(X=pi_hi[0], Y=pi_hi[1], A=pi_hi[3])
    g0(Z=pi_hi[2])
    out.append(P.torch_on)
    if P.pierce_delay > 0:
        out.append(f"G4 P{P.pierce_delay:.2f}")
    dz = pi_hi[2] - pi_lo[2]
    if dz > 1e-6:
        g1(dz, dz / P.plunge, Z=pi_lo[2])
    out.append(_cm("Cut contour"))
    first = (T.X[0], T.Y[0], T.Z[0], T.A[0])
    lead = math.hypot(T.pierce["aim"][0] - T.p_k[0], T.pierce["aim"][1] - T.x_k[0])
    g1(lead, lead / P.feed, X=first[0], Y=first[1], Z=first[2], A=first[3])
    for k0, k1 in zip(keep[:-1], keep[1:]):
        g1(cum_l[k1] - cum_l[k0], cum_t[k1] - cum_t[k0], X=T.X[k1], Y=T.Y[k1], Z=T.Z[k1], A=T.A[k1])
    lead_o = math.hypot(T.lead_out["aim"][0] - T.p_k[-1], T.lead_out["aim"][1] - T.x_k[-1])
    if lead_o > 1e-6:
        g1(lead_o, lead_o / P.feed, X=po[0], Y=po[1], Z=po[2], A=po[3])
    out.append(P.torch_off)
    g0(Z=T.z_safe)
    if P.a_end == "rewind":
        g0(A=P.a_scale * P.a_off)
    elif P.a_end == "zero":
        out.append(f"G92 A{P.a_scale * (((T.A_deg[-1]) % 360.0) + P.a_off):.3f}")
    out += ["G94", "M30"]
    return "\n".join(out) + "\n", n_g1[0]


# ======================================================================
# 5. PLOTS (Matplotlib)
# ======================================================================
def _new_fig(w: float, h: float) -> Figure:
    try:
        return Figure(figsize=(w, h), dpi=110, layout="constrained")
    except TypeError:  # older Matplotlib
        return Figure(figsize=(w, h), dpi=110, constrained_layout=True)


def fig_flat(P: Params, T: SimpleNamespace, i: int) -> Figure:
    fig = _new_fig(8.0, 4.6)
    ax = fig.subplots()
    lo, hi = min(T.x_nom.min(), T.x_k.min()), max(T.x_nom.max(), T.x_k.max())
    pad = 0.18 * (hi - lo) + 6.0
    edge = hi + pad if T.sw > 0 else lo - pad
    ax.fill_between(T.q, T.x_nom, edge, facecolor="0.93", edgecolor="0.75", hatch="///", lw=0, label="Waste")
    for k, (a0, a1) in enumerate(T.zones):
        ax.axvspan(a0, a1, color="tab:orange", alpha=0.14, lw=0, label="Corner radius" if k == 0 else None)
    ax.plot(T.q, T.x_nom, color="tab:blue", lw=2.2, label="Part edge")
    ax.plot(T.q, T.x_k, color="tab:red", lw=1.3, ls="--", label="Torch centre-line (kerf offset)")
    if P.bevel > 0:
        ax.plot(T.q, T.Xn, color="tab:green", lw=1.0, ls=":", label="Nozzle X (tilt shift)")
    q_pi = T.dirn * T.pierce["aim"][0]
    q_po = T.dirn * T.lead_out["aim"][0]
    ax.plot([q_pi, T.q[0]], [T.pierce["aim"][1], T.x_k[0]], color="tab:red", lw=1.3)
    ax.plot([T.q[-1], q_po], [T.x_k[-1], T.lead_out["aim"][1]], color="tab:red", lw=1.3)
    ax.plot(q_pi, T.pierce["aim"][1], "o", color="tab:red", ms=6, label="Pierce")
    ax.plot(T.q[i], T.x_k[i], "*", color="gold", mec="k", ms=15, zorder=5, label="Torch")
    ax.set_xlabel("Distance along the cut, unrolled perimeter (mm)")
    ax.set_ylabel("X along the tube (mm)")
    ax.set_xlim(T.q[0] - 0.02 * T.q[-1], T.q[-1] * 1.02)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=8, frameon=False)
    top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    top.set_xticks([m[0] for m in T.marks])
    top.set_xticklabels([f"A {m[1]}\u00b0" for m in T.marks], fontsize=8)
    ax.set_title("Flat pattern (tube unrolled, start at left)", fontsize=10, pad=24)
    return fig


def fig_3d(P: Params, T: SimpleNamespace, i: int, elev: float, azim: float) -> Figure:
    fig = _new_fig(7.2, 6.2)
    ax = fig.add_subplot(projection="3d")
    prof, L = T.prof, T.L
    D = 2 * max(prof.a, prof.b)
    ext = D + 10.0
    lo_c, hi_c = T.x_nom.min(), T.x_nom.max()
    if T.sw > 0:
        kept_far, waste_far = lo_c - ext, hi_c + 0.5 * ext
    else:
        kept_far, waste_far = hi_c + ext, lo_c - 0.5 * ext
    nl = 40
    pj = (np.arange(nl) + 0.5) * L / nl
    uj, vj, _ = prof.eval(pj)
    xj = T.cut_x(pj)
    for k in range(nl):
        ax.plot([kept_far, xj[k]], [uj[k]] * 2, [vj[k]] * 2, color="#4f81bd", lw=0.6, alpha=0.55)
        ax.plot([xj[k], waste_far], [uj[k]] * 2, [vj[k]] * 2, color="0.6", lw=0.5, alpha=0.4, ls=":")
    pr = np.linspace(0, L, 361)
    ur, vr, _ = prof.eval(pr)
    ax.plot(np.full_like(ur, kept_far), ur, vr, color="#4f81bd", lw=1.0)
    ax.plot(T.x_nom, T.u_nom, T.v_nom, color="crimson", lw=2.2, label="Part edge")
    ax.plot(T.x_k, T.uk, T.vk, color="darkorange", lw=1.0, ls="--", label="Torch path")
    ax.scatter([T.x_k[i]], [T.uk[i]], [T.vk[i]], marker="*", s=180, color="gold", edgecolor="k", zorder=10, label="Torch")
    m = 1.35 * max(prof.a, prof.b)
    x_lo = min(kept_far, waste_far, lo_c) - 0.05 * D
    x_hi = max(kept_far, waste_far, hi_c) + 0.05 * D
    if P.cut_type == "saddle":  # draw the run pipe the cut was calculated against
        rx, ry, rz = T.run_axis
        r = np.array([rx, ry, rz])
        helper = np.array([0.0, 1.0, 0.0]) if abs(ry) < 0.9 else np.array([0.0, 0.0, 1.0])
        e1 = np.cross(r, helper)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(r, e1)
        span = (D + P.run_od) / max(math.sin(math.radians(P.angle)), 0.2)
        tau = np.linspace(-span, span, 120)
        origin = np.array([T.x_origin, 0.0, 0.0])
        for th in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            pts = origin + tau[:, None] * r + (P.run_od / 2) * (math.cos(th) * e1 + math.sin(th) * e2)
            bad = (pts[:, 0] < x_lo) | (pts[:, 0] > x_hi) | (np.abs(pts[:, 1]) > m) | (np.abs(pts[:, 2]) > m)
            pts[bad] = np.nan
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="0.35", lw=0.7, alpha=0.5)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-m, m)
    ax.set_zlim(-m, m)
    ax.set_box_aspect((x_hi - x_lo, 2 * m, 2 * m))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (mm)", fontsize=8)
    ax.set_ylabel("u (mm)", fontsize=8)
    ax.set_zlabel("v (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("Tube at A = 0 (blue = part, grey dotted = waste)", fontsize=10)
    return fig


def fig_end(P: Params, T: SimpleNamespace, i: int) -> Figure:
    """View along the tube axis: the tube is rotated by A so the surface normal points up at the torch."""
    fig = _new_fig(6.2, 6.0)
    ax = fig.subplots()
    pr = np.linspace(0, T.L, 721)
    u, v, psi = T.prof.eval(pr)
    ps = np.radians(psi)
    ui, vi = u - np.sin(ps) * P.wall, v - np.cos(ps) * P.wall
    ang = math.radians(float(T.psi_k[i]))
    c, s = math.cos(ang), math.sin(ang)

    def rot(a_, b_):
        return a_ * c - b_ * s, a_ * s + b_ * c

    ax.fill(*rot(u, v), facecolor="#c9d6e8", edgecolor="#1f4e79", lw=1.6)
    ax.fill(*rot(ui, vi), facecolor="white", edgecolor="#1f4e79", lw=1.0)
    ax.plot([0], [0], "+", color="0.3", ms=10)
    y, zg = float(T.Yg[i]), float(T.Zg[i])
    zn = zg + P.cut_h * math.cos(math.radians(P.bevel))
    ax.plot([y, y], [zn, zn + 0.55 * T.r_max], lw=7, color="0.35", solid_capstyle="butt")
    ax.plot([y], [zn], "v", color="tab:red", ms=9)
    ax.plot([y], [zg], "o", color="gold", mec="k", ms=7, zorder=5)
    ax.axhline(T.z_safe - P.z_center, color="tab:green", ls=":", lw=1.2)
    ax.text(-1.45 * T.r_max, T.z_safe - P.z_center - 0.04 * T.r_max, "safe Z", color="tab:green", fontsize=8, va="top")
    lim = 1.5 * T.r_max
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim + 0.3 * T.r_max)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("Y (mm)")
    ax.set_ylabel("Height above tube axis (mm)")
    ax.set_title(f"End view  |  A {T.A_deg[i]:.1f}\u00b0   Y {T.Y[i]:.2f}   Z {T.Z[i]:.2f}", fontsize=10)
    return fig


def fig_axes(P: Params, T: SimpleNamespace, i: int) -> Figure:
    fig = _new_fig(8.0, 7.2)
    axs = fig.subplots(4, 1, sharex=True)
    series = [("X (mm)", T.X, "tab:blue"), ("Y (mm)", T.Y, "tab:green"),
              ("Z (mm)", T.Z, "tab:red"), ("A", T.A, "tab:purple")]
    for ax, (label, y, col) in zip(axs, series):
        ax.plot(T.q, y, color=col, lw=1.5)
        ax.plot(T.q[i], y[i], "o", color="k", ms=5)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    axs[-1].set_xlabel("Distance along the cut (mm)")
    axs[0].set_title("Axis positions along the cut", fontsize=10)
    return fig


# ======================================================================
# 6. STREAMLIT USER INTERFACE
# ======================================================================
_CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1100px;}
div[data-testid="stMetricValue"] {font-size: 1.3rem;}
@media (max-width: 640px) {.block-container {padding-left: .8rem; padding-right: .8rem;}}
</style>
"""


def _stretch() -> dict:
    """Make a widget fill its container on both old and new Streamlit versions."""
    try:
        if "width" in inspect.signature(st.download_button).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def sidebar_inputs() -> Params:
    sb = st.sidebar
    sb.header("Job setup")
    kw: dict = {}

    with sb.expander("1. Material and dimensions", expanded=True):
        shape_lbl = st.radio("Material shape", ["Round Pipe", "Square/Rectangular Tube"])
        if shape_lbl == "Round Pipe":
            kw["shape"] = "round"
            kw["od"] = st.number_input("Outer diameter (mm)", 6.0, 800.0, 60.3, 0.1, format="%.2f")
        else:
            kw["shape"] = "rect"
            kw["width"] = st.number_input("Width (mm)", 10.0, 600.0, 60.0, 0.5, format="%.2f")
            kw["height"] = st.number_input("Height (mm)", 10.0, 600.0, 40.0, 0.5, format="%.2f")
            kw["corner_r"] = st.number_input("Outer corner radius (mm)", 0.5, 150.0, 4.5, 0.1, format="%.2f",
                                             help="Typically 1.5 to 2.5 times the wall thickness.")
        kw["wall"] = st.number_input("Wall thickness (mm)", 0.5, 40.0, 3.0, 0.1, format="%.2f")

    with sb.expander("2. Cut geometry", expanded=True):
        cut_lbl = st.radio("Cut type", ["Saddle (branch onto a run pipe)", "Angled plane (miter)"])
        saddle = cut_lbl.startswith("Saddle")
        kw["cut_type"] = "saddle" if saddle else "miter"
        kw["angle"] = st.slider(
            "Intersection angle (deg)", 15.0, 165.0, 90.0, 0.5,
            help="Saddle: angle between this tube's axis and the run pipe's axis (90 = T-joint). "
                 "Miter: angle between the cut plane and the tube axis (90 = square cut).")
        if saddle:
            kw["run_od"] = st.number_input("Run pipe outer diameter (mm)", 10.0, 1500.0, 114.3, 0.1, format="%.2f")
        kw["tilt"] = st.slider(
            "Cut orientation (deg)", 0.0, 360.0, 0.0, 1.0,
            help="Rotates the cut around the tube. At 0 the start point lies in the plane that "
                 "contains both tube axes, on the side the run pipe axis tilts toward.")
        ref_lbl = st.radio("Fit-up reference surface", ["Outer", "Mid-wall", "Inner"], horizontal=True,
                           help="Which wall surface the intersection curve is calculated on.")
        kw["ref"] = {"Outer": 0.0, "Mid-wall": 0.5, "Inner": 1.0}[ref_lbl]
        keep_lbl = st.radio("Part you keep", ["Chuck side (offcut falls away)", "Free-end side"])
        kw["keep"] = -1 if keep_lbl.startswith("Chuck") else 1
        kw["x0"] = st.number_input(
            "X of the longest point of the part (mm)", -5000.0, 5000.0, 150.0, 1.0, format="%.2f",
            help="Distance from your X zero to the point of the cut that reaches furthest toward the waste.")

    with sb.expander("3. Plasma, kerf and bevel", expanded=False):
        kw["kerf"] = st.number_input("Kerf width (mm)", 0.0, 8.0, 1.5, 0.05, format="%.2f")
        kw["bevel"] = st.slider("Bevel angle (deg)", 0.0, 45.0, 0.0, 0.5,
                                help="Needs the torch fixed at this tilt in the X-Z plane (no tilt axis).")
        if kw["bevel"] > 0:
            b_lbl = st.radio("Bevel direction", ["Weld prep (inner edge reaches past outer edge)",
                                                 "Reverse (outer edge reaches past inner edge)"])
            kw["bevel_sig"] = -1 if b_lbl.startswith("Weld") else 1
        kw["cut_h"] = st.number_input("Cut standoff (mm)", 0.5, 30.0, 3.0, 0.1, format="%.2f")
        kw["pierce_h"] = st.number_input("Pierce standoff (mm)", 0.5, 40.0, 4.5, 0.1, format="%.2f")
        kw["pierce_delay"] = st.number_input("Pierce delay (s)", 0.0, 10.0, 0.5, 0.1, format="%.2f")
        kw["lead_in"] = st.number_input("Lead-in length (mm)", 0.0, 30.0, 3.0, 0.5, format="%.1f")
        kw["lead_out"] = st.number_input("Lead-out length (mm)", 0.0, 30.0, 3.0, 0.5, format="%.1f")
        kw["overlap"] = st.number_input("Overburn past the start (mm)", 0.0, 30.0, 0.0, 0.5, format="%.1f",
                                        help="Above 0, A runs slightly past 360 so the cut closes cleanly.")
        kw["torch_on"] = st.text_input("Torch on code", "M3")
        kw["torch_off"] = st.text_input("Torch off code", "M5")

    with sb.expander("4. Machine and feeds", expanded=False):
        kw["feed"] = st.number_input("Cut feed (mm/min)", 50.0, 20000.0, 1500.0, 50.0, format="%.0f")
        kw["plunge"] = st.number_input("Z plunge feed (mm/min)", 50.0, 20000.0, 2000.0, 50.0, format="%.0f")
        kw["corner_pct"] = st.slider("Feed in tube corners (% of cut feed)", 10, 100, 60, 5) * 1.0
        kw["a_max"] = st.number_input("Max A speed (deg/s)", 5.0, 3600.0, 180.0, 5.0, format="%.0f")
        mode_lbl = st.radio("Feed mode", ["G93 inverse time", "G94 units per minute"],
                            help="G93 gives the exact surface speed whatever your controller does with a rotary axis. "
                                 "Use G94 only if your controller lacks G93, and dry-run first.")
        kw["feed_mode"] = "G93" if mode_lbl.startswith("G93") else "G94"
        kw["safe_clear"] = st.number_input("Safe Z clearance above the tube (mm)", 5.0, 300.0, 25.0, 1.0, format="%.1f")
        kw["segments"] = st.slider("Path resolution (points per turn)", 90, 1440, 360, 30)
        kw["tol"] = st.number_input("G-code thinning tolerance (mm, 0 = off)", 0.0, 1.0, 0.02, 0.005, format="%.3f")
        end_lbl = st.selectbox("A axis after the cut", ["Rewind to A0", "Zero A with G92", "Leave at end angle"])
        kw["a_end"] = {"Rewind to A0": "rewind", "Zero A with G92": "zero", "Leave at end angle": "leave"}[end_lbl]

    with sb.expander("5. Axis calibration", expanded=False):
        kw["x_off"] = st.number_input("X offset (mm)", -5000.0, 5000.0, 0.0, 0.1, format="%.3f")
        kw["y_off"] = st.number_input("Y offset (mm)", -5000.0, 5000.0, 0.0, 0.1, format="%.3f")
        kw["z_center"] = st.number_input("Tube centre-line height above Z zero (mm)", -2000.0, 2000.0, 0.0, 0.5,
                                         format="%.3f", help="Z = this value + radius at the contact point + standoff.")
        kw["a_off"] = st.number_input("A zero offset (deg)", -360.0, 360.0, 0.0, 0.1, format="%.3f")
        kw["a_rev"] = st.checkbox("Reverse A direction", False,
                                  help="Tick if positive A turns the chuck the opposite way to your wiring assumption.")
        kw["a_scale"] = st.number_input("A scale (output units per degree)", 0.001, 1000.0, 1.0, 0.001, format="%.4f")
    return Params(**kw)


def _file_name(P: Params) -> str:
    size = f"od{P.od:g}" if P.shape == "round" else f"{P.width:g}x{P.height:g}"
    kind = f"saddle{P.run_od:g}" if P.cut_type == "saddle" else "miter"
    return f"pipecut_{P.shape}_{size}_{kind}_{P.angle:g}deg_4axis.gcode"


def main() -> None:
    st.set_page_config(page_title="4-axis pipe cutter", page_icon="\U0001F527", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.title("4-axis pipe and tube cutter")
    st.caption("G-code generator for plasma profile cuts. Set up the job in the sidebar, check the preview, then download.")

    P = sidebar_inputs()
    try:
        T = compute_toolpath(P)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    gcode, n_moves = build_gcode(P, T)

    for w in T.warnings:
        st.warning(w)

    st.download_button("Download G-code file", data=gcode, file_name=_file_name(P), mime="text/plain",
                       type="primary", **_stretch())
    st.caption("Dry-run with the torch disabled before the first cut.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cut length", f"{T.cut_len:.0f} mm")
    c2.metric("Est. cut time", f"{int(T.total_min * 60) // 60}:{int(T.total_min * 60) % 60:02d}")
    c3.metric("G1 moves", f"{n_moves}")
    c4, c5, c6 = st.columns(3)
    c4.metric("Y travel", f"{T.Y.max() - T.Y.min():.1f} mm")
    c5.metric("Z range", f"{T.Z.max() - T.Z.min():.1f} mm")
    c6.metric("Cut depth profile", f"{T.f_max - T.f_min:.1f} mm")

    pos = st.slider("Torch position along the cut", 0, 100, 0, 1, format="%d%%")
    i = int(round(pos / 100.0 * (len(T.q) - 1)))
    st.caption(f"A {T.A[i]:.2f}   X {T.X[i]:.2f}   Y {T.Y[i]:.2f}   Z {T.Z[i]:.2f}")

    t_flat, t_3d, t_end, t_axes, t_code = st.tabs(["Flat pattern", "3D view", "End view", "Axis motion", "G-code"])
    with t_flat:
        st.pyplot(fig_flat(P, T, i))
    with t_3d:
        e_col, a_col = st.columns(2)
        elev = e_col.slider("Elevation", 0, 90, 25)
        azim = a_col.slider("Rotation", -180, 180, -60)
        st.pyplot(fig_3d(P, T, i, elev, azim))
    with t_end:
        st.pyplot(fig_end(P, T, i))
    with t_axes:
        st.pyplot(fig_axes(P, T, i))
    with t_code:
        lines = gcode.splitlines()
        st.code("\n".join(lines[:300]), language="text")
        if len(lines) > 300:
            st.caption(f"Showing the first 300 of {len(lines)} lines. The download has all of them.")

    with st.expander("How this model works and its limits"):
        st.markdown(
            "- The torch stays vertical. A is set to the angle of the surface normal at the contact point, "
            "Y follows the contact point sideways and Z follows its height, so the torch is square to the surface everywhere.\n"
            "- The cut curve is the tube surface meeting the run pipe (saddle) or a flat plane (miter), "
            "computed on the outer, mid-wall or inner surface you choose.\n"
            "- Kerf is offset into the waste in the unrolled surface plane, so it stays correct on slanted cuts and corners.\n"
            "- Bevel needs the torch clamped at that tilt in the X-Z plane. The software shifts X to compensate "
            "for standoff and wall depth, but the real bevel shrinks where the cut runs along the tube. The warning above shows the range.\n"
            "- Lead-in and lead-out are straight moves into the waste. Pierce happens at the pierce standoff, then Z drops to cut standoff.\n"
            "- Check the sign of A and Y with a dry run, and use the calibration section to correct offsets or direction."
        )


if __name__ == "__main__":
    main()
