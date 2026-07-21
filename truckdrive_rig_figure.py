"""Top-down + side-elevation figure of TruckDrive camera extrinsics across fleets.

Shows the 5 training cameras positioned in the cab frame, with heading arrows,
overlaying the two calibration clusters (A = batches <=18, B = batches >=21) plus
the intermediate batch-40 rig. Shaded wedges encode the inter-fleet angular
difference; the front camera (biggest difference, ~14 deg, pitch-dominated) is
highlighted with a dedicated side-elevation panel.
"""
import json, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrow, Rectangle

VIEWS = ["forward_center_medium", "sideward_left_front_wide", "sideward_right_front_wide",
         "rearward_left_bottom_medium", "rearward_right_bottom_medium"]
SHORT = {"forward_center_medium": "front_center",
         "sideward_left_front_wide": "side_left_front",
         "sideward_right_front_wide": "side_right_front",
         "rearward_left_bottom_medium": "rear_left",
         "rearward_right_bottom_medium": "rear_right"}

def R_from_q(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def q_angle(q1, q2):
    d = abs(sum(a*b for a, b in zip(q1, q2))); d = min(1.0, d)
    return math.degrees(2*math.acos(d))

def load(fn, v):
    d = json.load(open(fn))["cab_camera_leopard_" + v]["transform"]
    t = d["translation"]; r = d["rotation"]
    T = np.array([t["x"], t["y"], t["z"]])
    q = [r["x"], r["y"], r["z"], r["w"]]
    axis = R_from_q(q) @ np.array([0, 0, 1.0])   # optical axis (+Z) in cab frame
    return T, axis, q

files = sorted(set(glob.glob("/tmp/tf_cmp/*.json") +
                   glob.glob("/tmp/tf_valtest/*.json") +
                   glob.glob("/tmp/tf_trainbatch/*.json")))

# reference front-cam quats for cluster classification
refA = load("/tmp/tf_cmp/scene_10_1.json", "forward_center_medium")[2]
refB = load("/tmp/tf_cmp/scene_20_1.json", "forward_center_medium")[2]

# collect per-cluster samples
clusters = {"A": [], "B": [], "mid": []}
for fn in files:
    try:
        _, _, qf = load(fn, "forward_center_medium")
    except KeyError:
        continue  # no front_center (batches 1-4)
    dA, dB = q_angle(qf, refA), q_angle(qf, refB)
    key = "mid" if min(dA, dB) > 4 else ("A" if dA < dB else "B")
    clusters[key].append(fn)

def cluster_stats(fns):
    """mean T and mean optical axis per view."""
    out = {}
    for v in VIEWS:
        Ts, Ax = [], []
        for fn in fns:
            try:
                T, a, _ = load(fn, v)
            except KeyError:
                continue
            Ts.append(T); Ax.append(a)
        if not Ts:
            continue
        Tm = np.mean(Ts, axis=0)
        am = np.mean(Ax, axis=0); am /= np.linalg.norm(am)
        out[v] = (Tm, am, np.array(Ax))
    return out

stA = cluster_stats(clusters["A"])
stB = cluster_stats(clusters["B"])
stM = cluster_stats(clusters["mid"]) if clusters["mid"] else {}

# ---------------- figure ----------------
fig = plt.figure(figsize=(15, 6.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
axB = fig.add_subplot(gs[0, 0])   # bird's-eye
axS = fig.add_subplot(gs[0, 1])   # side elevation (front cam)

COL = {"A": "#1f77b4", "B": "#d62728", "mid": "#2ca02c"}
LBL = {"A": f"Fleet A  (batches ≤18, {len(clusters['A'])} scenes)",
       "B": f"Fleet B  (batches ≥21, {len(clusters['B'])} scenes)",
       "mid": f"batch 40 ({len(clusters['mid'])} scenes)"}

# screen mapping for BEV: horizontal = y (left cams y<0 -> left), vertical = x (forward up)
def bev_xy(T):      # returns (screen_x, screen_y)
    return T[1], T[0]
def bev_dir(a):     # arrow components on screen
    return a[1], a[0]

# --- cab schematic ---
cab = Rectangle((-1.3, 1.9), 2.6, 1.5, facecolor="#eeeeee", edgecolor="#999999",
                lw=1.2, zorder=0)
axB.add_patch(cab)
axB.annotate("cab (schematic)", (0, 2.65), ha="center", va="center",
             color="#999999", fontsize=8, zorder=1)
axB.annotate("", xy=(0, 3.9), xytext=(0, 3.4),
             arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
axB.text(0, 4.0, "forward", ha="center", va="bottom", fontsize=9, color="#555")

ARROW_LEN = 0.9
for st, ckey in [(stA, "A"), (stB, "B")] + ([(stM, "mid")] if stM else []):
    c = COL[ckey]
    for v in VIEWS:
        if v not in st:
            continue
        Tm, am, allax = st[v]
        sx, sy = bev_xy(Tm)
        dx, dy = bev_dir(am)
        n = math.hypot(dx, dy) or 1.0
        axB.scatter([sx], [sy], s=55, color=c, edgecolor="k", lw=0.6,
                    zorder=5, label=LBL[ckey] if v == VIEWS[0] else None)
        axB.arrow(sx, sy, ARROW_LEN*dx/n, ARROW_LEN*dy/n, head_width=0.09,
                  head_length=0.12, fc=c, ec=c, lw=1.6, length_includes_head=True,
                  zorder=4, alpha=0.9)

# camera name labels (from fleet B positions)
for v in VIEWS:
    if v in stB:
        sx, sy = bev_xy(stB[v][0])
        off = (0.12, 0.16) if "left" in v or v == "forward_center_medium" else (0.12, -0.22)
        axB.annotate(SHORT[v], (sx, sy), xytext=(sx+off[0], sy+off[1]),
                     fontsize=8.5, color="#333", zorder=6)

# inter-fleet yaw wedge on the front camera
Tf = stB["forward_center_medium"][0]
sx, sy = bev_xy(Tf)
yawA = math.degrees(math.atan2(*bev_dir(stA["forward_center_medium"][1])[::-1]))
yawB = math.degrees(math.atan2(*bev_dir(stB["forward_center_medium"][1])[::-1]))
axB.set_title("Bird's-eye view  —  camera positions & ground-projected heading",
              fontsize=11, pad=10)
# FRU cab frame: +y = right, -y = left. Standard axis already puts left cams
# (y<0) on the left, right cams (y>0) on the right -- no inversion needed.
axB.set_xlabel("y  (m)", labelpad=2); axB.set_ylabel("x  (forward), m")
axB.set_aspect("equal"); axB.grid(alpha=0.25)
axB.set_xlim(-2.4, 2.4); axB.set_ylim(1.6, 4.2)
axB.text(-2.25, 1.72, "◄ LEFT", fontsize=8.5, color="#777", weight="bold", va="bottom")
axB.text(2.25, 1.72, "RIGHT ►", fontsize=8.5, color="#777", weight="bold",
         va="bottom", ha="right")
axB.text(0.0, 1.72, "front-cam difference is pitch → see right panel", fontsize=8,
         color="#8a6d00", style="italic", ha="center", va="bottom")
axB.legend(loc="upper center", fontsize=8.5, framealpha=0.9, ncol=3,
           bbox_to_anchor=(0.5, -0.10))

# ---------------- side elevation: front camera pitch ----------------
# horizontal = x (forward), vertical = z (up)
axS.set_title("Side elevation  —  FRONT camera optical axis (biggest difference)",
              fontsize=11, pad=10)
RAY = 1.6
for st, ckey in [(stA, "A"), (stB, "B")] + ([(stM, "mid")] if stM else []):
    Tm, am, allax = st["forward_center_medium"]
    x0, z0 = Tm[0], Tm[2]
    dx, dz = am[0], am[2]
    n = math.hypot(dx, dz) or 1.0
    pitch = math.degrees(math.atan2(dz, dx))
    axS.scatter([x0], [z0], s=70, color=COL[ckey], edgecolor="k", lw=0.6, zorder=5)
    axS.arrow(x0, z0, RAY*dx/n, RAY*dz/n, head_width=0.05, head_length=0.08,
              fc=COL[ckey], ec=COL[ckey], lw=2.2, length_includes_head=True, zorder=4,
              label=f"{LBL[ckey].split('(')[0].strip()}:  pitch {pitch:+.1f}°")
    # per-scene spread rays (thin, translucent)
    for a in allax:
        na = math.hypot(a[0], a[2]) or 1.0
        axS.plot([x0, x0+RAY*a[0]/na], [z0, z0+RAY*a[2]/na], color=COL[ckey],
                 lw=0.5, alpha=0.15, zorder=2)

# shaded wedge between fleet A and B optical axes = the inter-fleet difference
xf, zf = stA["forward_center_medium"][0][0], stA["forward_center_medium"][0][2]
aA = stA["forward_center_medium"][1]; aB = stB["forward_center_medium"][1]
pA = math.degrees(math.atan2(aA[2], aA[0])); pB = math.degrees(math.atan2(aB[2], aB[0]))
diff = q_angle(stA["forward_center_medium"][1].tolist()+[0],  # placeholder not used
               stB["forward_center_medium"][1].tolist()+[0]) if False else abs(pA-pB)
w = Wedge((xf, zf), RAY*0.72, min(pA, pB), max(pA, pB), facecolor="#ffcc00",
          alpha=0.35, edgecolor="none", zorder=1)
axS.add_patch(w)
axS.set_ylim(zf-0.18, zf+0.62)
axS.annotate(f"Δ ≈ {abs(pA-pB):.0f}° pitch  (fleet A points up, fleet B level)",
             (xf+0.02, zf+0.52), fontsize=10, color="#8a6d00", weight="bold",
             ha="left", va="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff7d6", ec="#e0c860", lw=0.8))
axS.axhline(zf, color="#bbb", lw=0.8, ls="--", zorder=0)
axS.annotate("", xy=(xf+RAY+0.3, zf), xytext=(xf, zf),
             arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.0))
axS.text(xf+RAY+0.32, zf-0.04, "forward", fontsize=8, color="#999", va="top")
axS.set_xlabel("x  (forward), m"); axS.set_ylabel("z  (up), m")
axS.set_aspect("equal"); axS.grid(alpha=0.25)
axS.legend(loc="lower left", fontsize=8.5, framealpha=0.9)

fig.suptitle("TruckDrive camera-rig extrinsics across calibration fleets  "
             "(val/test are 100% Fleet B)", fontsize=13, weight="bold", y=0.99)
out = "/tmp/claude-1001/-mnt-efs-users-rod-repos-alpamayo-recipes/99db5cf5-f60e-4c23-a7cd-9d932e36f6d3/scratchpad/rig_extrinsics.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
print("fleet counts:", {k: len(v) for k, v in clusters.items()})
print(f"front-cam mean pitch: A={pA:+.1f}deg  B={pB:+.1f}deg  diff={abs(pA-pB):.1f}deg")
