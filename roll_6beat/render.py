"""자유형 롤링과 6비트 킥 타이밍 설명 영상 렌더러.

간단한 3D 막대 인형(팔은 손 젓기 모델의 경로 재사용)을 뒤에서 본 어깨·골반 단면과 옆에서 본 시점으로
그리고, 팔 단계·롤링 각도·킥 6박자를 타임라인으로 정렬해 보여준다.

python3 render.py  ->  roll_6beat_timing.mp4
PREVIEW=10,20 python3 render.py  ->  _build/ 에 해당 시각 프레임 PNG만 저장
"""
import contextlib
import os
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Polygon, Circle
from matplotlib.animation import FFMpegWriter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "freestyle_pull"))
import pull_model as pm  # noqa: E402

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

FPS = 30
W, H = 12.8, 7.2
OUT = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(OUT, "_build")
os.makedirs(TMP, exist_ok=True)

SKIN = "#f2c9a0"
SKIN_D = "#d99a6c"
SUIT = "#1b2a49"
CAP = "#d62828"
C_R = "#ff8c42"        # 오른쪽 (팔·다리)
C_L = "#4fc3ff"        # 왼쪽
C_ROLL = "#ffd166"
SURF = pm.SURFACE_Y

# ----------------------------------------------------------------- 동작 모델
ROLL_SH = np.radians(40)      # 어깨 롤링 최대각
HIP_RATIO = 0.6               # 골반 롤링 = 어깨의 60%
ROLL_PEAK = 0.15              # 오른쪽으로 가장 많이 기운 위상 (오른손이 앞으로 뻗어 캐치할 때)
ENTRY_S = 0.12                # 손이 입수해 앞으로 뻗는 시점 (팔 위상)
BEATS = [(ENTRY_S + k / 6) % 1.0 for k in range(6)]          # 6박자 다운킥 시점
BEAT_LEG = ["L", "R", "L", "R", "L", "R"]                     # 오른손 입수 = 왼발 강한 킥
BEAT_MAJOR = [True, False, False, True, False, False]
SH_Y, HIP_Y_, HIP_X = 0.20, 0.09, -0.55
KICK_AMP = 0.10


def wrap(d):
    return (d + 0.5) % 1.0 - 0.5


def roll_angle(s):
    """어깨 롤링 각 (+: 오른쪽 어깨가 내려감)."""
    return ROLL_SH * np.cos(2 * np.pi * (s - ROLL_PEAK))


def rot_x(p, ang):
    """x축(진행 방향) 기준 회전. +ang 이면 +y(오른쪽)가 아래로."""
    y, z = p[..., 1], p[..., 2]
    c, s = np.cos(ang), np.sin(ang)
    out = p.copy()
    out[..., 1] = y * c + z * s
    out[..., 2] = -y * s + z * c
    return out


def arm_points(s, side):
    """한쪽 팔 (어깨, 팔꿈치, 손목, 손끝) 3D (몸 좌표, 롤링 전). side=+1 오른쪽."""
    S, E, Wr, T = pm.arm(s / pm.FREQ)
    ph = s % 1.0
    # 옆으로의 움직임: 물속에서는 몸 아래로 모였다가(풀) 바깥으로(푸시), 리커버리는 옆으로 크게
    if ph < pm.S_OUT:
        q = np.clip((ph - pm.S_IN) / (pm.S_OUT - pm.S_IN), 0, 1)
        lat_w = 0.03 - 0.17 * np.sin(np.pi * q) ** 2 + 0.10 * q
    else:
        v = (ph - pm.S_OUT) / (1 - pm.S_OUT)
        lat_w = 0.13 + 0.08 * np.sin(np.pi * v) - 0.10 * v
    lat_e = 0.5 * lat_w + 0.06
    base = np.array([0.0, side * SH_Y, 0.0])
    pts = [base,
           base + [E[0], side * lat_e, E[1]],
           base + [Wr[0], side * lat_w, Wr[1]],
           base + [T[0], side * lat_w, T[1]]]
    return np.array(pts)


def leg_disp(s, leg):
    """다리의 위아래 변위 (+: 아래). 각 다리는 한 주기에 3번 다운킥."""
    beats = [b for b, l in zip(BEATS, BEAT_LEG) if l == leg]
    major = [b for b, l, mj in zip(BEATS, BEAT_LEG, BEAT_MAJOR) if l == leg and mj][0]
    amp = KICK_AMP * (1 + 0.7 * np.exp(-(wrap(s - major + 1 / 12) ** 2) / (2 * 0.08 ** 2)))
    return amp * np.sin(2 * np.pi * 3 * (s - beats[0]))     # 다운킥 시점에 아래로 가장 빠르게


def leg_points(s, side, leg):
    d = leg_disp(s, leg)
    hip = np.array([HIP_X, side * HIP_Y_, -0.02])
    knee = np.array([HIP_X - 0.45, side * HIP_Y_, -0.06 - 0.45 * d])
    ankle = np.array([HIP_X - 0.88, side * HIP_Y_, -0.04 - d])
    toe = ankle + [-0.18, 0.0, 0.03 - 0.25 * d]
    return np.array([hip, knee, ankle, toe])


def body(s):
    """롤링을 적용한 3D 점들."""
    phi = roll_angle(s)
    hip_phi = HIP_RATIO * phi
    out = {}
    out["armR"] = rot_x(arm_points(s, +1), phi)
    out["armL"] = rot_x(arm_points(s + 0.5, -1), phi)
    out["legR"] = rot_x(leg_points(s, +1, "R"), hip_phi)
    out["legL"] = rot_x(leg_points(s, -1, "L"), hip_phi)
    sh = np.array([[0.05, SH_Y + 0.02, 0.07], [0.05, -SH_Y - 0.02, 0.07],
                   [0.05, SH_Y, -0.13], [0.05, -SH_Y, -0.13]])
    hp = np.array([[HIP_X, 0.17, 0.06], [HIP_X, -0.17, 0.06], [HIP_X, 0.15, -0.10], [HIP_X, -0.15, -0.10]])
    out["torso_sh"] = rot_x(sh, phi)
    out["torso_hip"] = rot_x(hp, hip_phi)
    out["head"] = rot_x(np.array([[0.27, 0.0, 0.05]]), phi)[0]
    out["phi"] = phi; out["hip_phi"] = hip_phi
    return out


# ----------------------------------------------------------------- 투영
def side_view(p):
    """오른쪽 옆에서 본 정사영 (진행 방향 = 화면 오른쪽)."""
    p = np.atleast_2d(p)
    return np.c_[p[:, 0], p[:, 2]]


def hull(pts):
    pts = sorted(map(tuple, pts))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


# ----------------------------------------------------------------- 렌더 도우미
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, ramp=0.6):
    return smooth((t - t0) / ramp) * smooth((t1 - t) / ramp)


class _PreviewWriter:
    def __init__(self, times):
        self.frames = {int(round(float(x) * FPS)) for x in times}
        self.i = 0

    def saving(self, fig, path, dpi):
        self.fig, self.path = fig, path
        return contextlib.nullcontext()

    def grab_frame(self):
        if self.i in self.frames:
            self.fig.savefig(f"{self.path}_{self.i / FPS:05.1f}.png", dpi=100)
        self.i += 1


def writer():
    if os.environ.get("PREVIEW"):
        return _PreviewWriter(os.environ["PREVIEW"].split(","))
    return FFMpegWriter(fps=FPS, codec="libx264", bitrate=5000,
                        extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])


def water_bg(ax, xl, xr, yb, yt, surf):
    sky = np.linspace(0, 1, 32)[:, None]
    ax.imshow(np.dstack([0.75 + 0.15 * sky, 0.86 + 0.1 * sky, 0.97 + 0 * sky]),
              extent=(xl, xr, surf, yt), origin="lower", aspect="auto", zorder=0)
    deep = np.linspace(0, 1, 64)[:, None]
    ax.imshow(np.dstack([0.02 + 0.07 * deep, 0.12 + 0.24 * deep, 0.25 + 0.32 * deep]),
              extent=(xl, xr, yb, surf), origin="lower", aspect="auto", zorder=0)
    ax.plot([xl, xr], [surf, surf], color="white", lw=2, alpha=0.85, zorder=1)


# ----------------------------------------------------------------- 시나리오
SCENES = [  # (시작, 끝, 배속, 머리글)
    (0.0, 5.0, 0.35, ""),
    (5.0, 21.0, 0.3, "① 롤링 : 몸통이 좌우로 구른다"),
    (21.0, 39.0, 0.3, "② 6비트 킥 : 양팔 한 번씩 젓는 동안 발차기 6번"),
    (39.0, 58.0, 0.22, "③ 대각선 타이밍 : 손 입수 ↔ 반대쪽 발 다운킥"),
    (58.0, 69.0, 0.35, ""),
]
END = 69.0
CAPTIONS = [
    (5.0, 10.5, "뒤에서 본 어깨·골반 단면(왼쪽)과 옆모습(오른쪽). 주황 = 오른쪽 팔·다리, 파랑 = 왼쪽\n오른손이 앞으로 뻗을 때 몸이 오른쪽으로, 왼손이 뻗을 때 왼쪽으로 기운다"),
    (10.5, 16.0, "어깨는 약 40°, 골반은 그보다 적게(약 25°) 돈다. 머리는 숨쉴 때 말고는 돌지 않는다\n롤링은 몸통(코어)이 주도하고, 다리는 골반을 따라 같이 기울어진다"),
    (16.0, 21.0, "몸이 기울면 발차기도 비스듬히(대각선으로) 찬다 — 골반 단면의 발 궤적 선을 보자\n발은 몸통 폭 안에서, 다리가 옆으로 벌어지지 않게"),
    (21.0, 27.0, "6비트 킥 : 오른팔·왼팔이 한 번씩 젓는 동안(1주기) 발차기 6번\n오른발·왼발이 번갈아 3번씩 → 아래 타임라인의 숫자 1~6"),
    (27.0, 33.0, "6번이 모두 같은 세기는 아니다 : 큰 동그라미 2번이 강한 킥\n손이 입수하는 순간에 맞춰 강하게, 나머지는 리듬을 유지하는 작은 킥"),
    (33.0, 39.0, "롤링이 한쪽 끝에 이를 때마다 강한 킥이 들어가 몸의 회전을 밀어주고\n나머지 작은 킥들이 하체가 가라앉지 않게 받쳐 준다"),
    (39.0, 45.5, "오른손이 입수해 앞으로 뻗는 순간 → 왼발이 강하게 다운킥 (노란 연결선)\n왼손이 입수할 때 → 오른발이 강하게 다운킥"),
    (45.5, 52.0, "대각선으로 짝을 이룬 손과 발이 동시에 힘을 쓰면서\n몸이 비틀리지 않고 일직선으로 유지된다 (손과 반대쪽 발의 균형)"),
    (52.0, 58.0, "리듬으로 세어 보기 : '하나'(오른손 입수 + 왼발 킥) - 둘 - 셋\n'넷'(왼손 입수 + 오른발 킥) - 다섯 - 여섯"),
]
SUMMARY = [
    "정리",
    "1.  롤링은 몸통이 주도 — 어깨 약 40°, 골반은 그보다 적게",
    "2.  다리는 골반을 따라 기울며 비스듬히 찬다 (발은 몸통 폭 안에서)",
    "3.  6비트 킥 = 한 주기(양팔 한 번씩)에 발차기 6번, 그중 2번이 강한 킥",
    "4.  손 입수 ↔ 반대쪽 발 강한 다운킥 : 대각선 짝으로 균형과 회전",
]


def scene_at(T):
    for sc in SCENES:
        if sc[0] <= T < sc[1]:
            return sc
    return SCENES[-1]


ARM_PHASES = pm.PHASES


def render(path):
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("#06182e")

    # 뒤에서 본 단면 두 개 (정사영: 화면 오른쪽 = 수영자의 오른쪽)
    def section_axes(rect, title):
        ax = fig.add_axes(rect)
        hw = 0.46
        ht = 2 * hw * (rect[3] * H) / (rect[2] * W)          # 가로세로 비율을 1:1로
        ax.set_xlim(-hw, hw); ax.set_ylim(-ht * 0.62, ht * 0.38); ax.axis("off")
        yb, yt = ax.get_ylim()
        water_bg(ax, -hw, hw, yb, yt, SURF)
        ax.text(-hw + 0.02, yt - 0.02, title, color="#0d2240", fontsize=13, weight="bold", va="top", zorder=40)
        return ax

    cax = section_axes([0.01, 0.37, 0.245, 0.55], "어깨·가슴 (뒤에서 본 단면)")
    hax = section_axes([0.262, 0.37, 0.245, 0.55], "골반·다리 (뒤에서 본 단면)")
    th = np.linspace(0, 2 * np.pi, 60)
    chest = Polygon(np.zeros((3, 2)), closed=True, color=SKIN, zorder=10); cax.add_patch(chest)
    head_c = Circle((0, 0), 0.09, color=CAP, zorder=11); cax.add_patch(head_c)
    arm_c = {k: cax.plot([], [], color=c, lw=8, solid_capstyle="round", zorder=12)[0]
             for k, c in (("armR", C_R), ("armL", C_L))}
    sh_line, = cax.plot([], [], color=C_ROLL, lw=2.5, ls="--", zorder=20)
    roll_txt = cax.text(0, 0, "", color=C_ROLL, fontsize=15, weight="bold", zorder=30, ha="center", va="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.5), ec="none"))
    pelvis = Polygon(np.zeros((3, 2)), closed=True, color=SUIT, zorder=10); hax.add_patch(pelvis)
    hip_line, = hax.plot([], [], color=C_ROLL, lw=2.5, ls="--", zorder=20)
    hip_txt = hax.text(0, 0, "", color=C_ROLL, fontsize=15, weight="bold", zorder=30, ha="center", va="top",
                       bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.5), ec="none"))
    trails = {k: hax.plot([], [], color=c, lw=2, alpha=0.6, zorder=11)[0] for k, c in (("legR", C_R), ("legL", C_L))}
    feet = {k: hax.scatter([0], [0], s=420, color=c, edgecolor="white", linewidths=1.5, zorder=14)
            for k, c in (("legR", C_R), ("legL", C_L))}
    thighs = {k: hax.plot([], [], color=c, lw=10, solid_capstyle="round", alpha=0.85, zorder=12)[0]
              for k, c in (("legR", C_R), ("legL", C_L))}
    foot_hist = {"legR": [], "legL": []}
    diag_txt = fig.text(0.255, 0.385, "", color="#fff176", fontsize=16, weight="bold", ha="center",
                        zorder=41, bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.6), ec="none"))
    beat_big = hax.text(0.43, hax.get_ylim()[1] - 0.09, "", color="white", fontsize=40, weight="bold",
                        ha="right", va="top", zorder=40)
    glow = {k: hax.scatter([0], [0], s=1600, color="#fff176", alpha=0, zorder=13) for k in ("legR", "legL")}

    # 옆에서 본 시점
    sax = fig.add_axes([0.515, 0.37, 0.475, 0.55])
    SXL, SXR = -1.75, 0.95
    SYB = -0.80
    SYT = SYB + (SXR - SXL) * (0.55 * H) / (0.475 * W)
    sax.set_xlim(SXL, SXR); sax.set_ylim(SYB, SYT); sax.axis("off")
    water_bg(sax, SXL, SXR, SYB, SYT, SURF)
    sax.text(SXL + 0.03, SYT - 0.04, "옆에서 본 모습 (오른쪽 옆)", color="#0d2240", fontsize=14,
             weight="bold", va="top", zorder=40)
    torso_s = Polygon(np.zeros((3, 2)), closed=True, color=SKIN, zorder=10)
    hips_s = Polygon(np.zeros((3, 2)), closed=True, color=SUIT, zorder=11)
    sax.add_patch(torso_s); sax.add_patch(hips_s)
    head_s = Circle((0, 0), 0.11, color=CAP, zorder=12)
    sax.add_patch(head_s)
    arm_s = {k: sax.plot([], [], color=c, lw=lw, solid_capstyle="round", zorder=z)[0]
             for k, c, lw, z in (("armR", C_R, 9, 14), ("armL", C_L, 7, 8))}
    leg_s = {k: sax.plot([], [], color=c, lw=lw, solid_capstyle="round", zorder=z)[0]
             for k, c, lw, z in (("legR", C_R, 11, 13), ("legL", C_L, 9, 7))}
    diag_s, = sax.plot([], [], color="#fff176", lw=3, zorder=25, alpha=0)

    # 타임라인
    tax = fig.add_axes([0.07, 0.135, 0.90, 0.215])
    tax.set_facecolor((0, 0, 0, 0.35))
    tax.set_xlim(0, 1); tax.set_ylim(0, 4)
    tax.set_yticks([0.5, 1.5, 2.5, 3.5]); tax.set_yticklabels(["발차기", "롤링", "왼팔", "오른팔"], color="white", fontsize=13)
    tax.set_xticks([])
    for sp in tax.spines.values():
        sp.set_color((1, 1, 1, 0.3))
    for row, off, col in ((3, 0.0, C_R), (2, 0.5, C_L)):
        for a, b, name, pc in ARM_PHASES:
            for sh in (0, 1, -1):
                x0 = (a - off) % 1.0 + sh
                tax.add_patch(Rectangle((x0, row + 0.12), b - a, 0.76, color=pc, alpha=0.55, lw=0))
                if name != "리커버리" and 0 <= x0 + (b - a) / 2 <= 1:
                    tax.text(x0 + (b - a) / 2, row + 0.5, name.split(" ")[0], ha="center", va="center",
                             fontsize=10, color="#0d2240", clip_on=True)
                elif 0 <= x0 + (b - a) / 2 <= 1:
                    tax.text(x0 + (b - a) / 2, row + 0.5, "리커버리", ha="center", va="center",
                             fontsize=10, color="#0d2240", clip_on=True)
        ent = (ENTRY_S - off) % 1.0
        tax.plot([ent, ent], [row + 0.05, row + 0.95], color="white", lw=2)
    xs = np.linspace(0, 1, 300)
    rl = np.array([roll_angle(x) for x in xs]) / ROLL_SH
    tax.plot(xs, 1.5 + 0.4 * rl, color=C_ROLL, lw=2.5)
    tax.axhline(1.5, color="white", lw=0.6, alpha=0.4)
    tax.text(0.003, 1.93, "오른쪽", color=C_ROLL, fontsize=9, va="top")
    tax.text(0.003, 1.07, "왼쪽", color=C_ROLL, fontsize=9, va="bottom")
    beat_art = []
    for i, (b, l, mj) in enumerate(zip(BEATS, BEAT_LEG, BEAT_MAJOR)):
        c = C_R if l == "R" else C_L
        circ = tax.scatter([b], [0.5], s=520 if mj else 220, color=c, edgecolor="white",
                           linewidths=2 if mj else 1, zorder=5, alpha=0)
        txt = tax.text(b, 0.5, str(i + 1), ha="center", va="center", fontsize=13 if mj else 10,
                       weight="bold", color="#0d2240", zorder=6, alpha=0)
        beat_art.append((circ, txt))
    links = []
    for ent, beat in ((ENTRY_S, BEATS[0]), ((ENTRY_S - 0.5) % 1.0, BEATS[3])):
        row = 3 if ent == ENTRY_S else 2
        ln, = tax.plot([ent, beat], [row + 0.05, 0.9], color="#fff176", lw=2.5, ls="--", alpha=0, zorder=4)
        links.append(ln)
    cursor, = tax.plot([0, 0], [0, 4], color="white", lw=2.5, zorder=10)
    legend_t = fig.text(0.07, 0.355, "흰 세로선 = 손 입수,  큰 원 = 강한 킥,  주황 R = 오른발 / 파랑 L = 왼발",
                        color="#bfe6ff", fontsize=11, zorder=40)

    header = fig.text(0.015, 0.96, "", fontsize=22, weight="bold", color="white", va="center", zorder=40)
    slowmo = fig.text(0.985, 0.96, "", fontsize=13, color="#bfe6ff", ha="right", va="center", zorder=40)
    cap = fig.text(0.015, 0.012, "", fontsize=15, color="white", va="bottom", ha="left", linespacing=1.4,
                   zorder=50, bbox=dict(boxstyle="round,pad=0.45", fc=(0, 0, 0, 0.65), ec="none"))

    overlay = fig.add_axes([0, 0, 1, 1], zorder=60); overlay.axis("off")
    overlay.set_xlim(0, 1); overlay.set_ylim(0, 1)
    ov = overlay.add_patch(Rectangle((0, 0), 1, 1, color=(0.02, 0.07, 0.15), alpha=0))
    title1 = overlay.text(0.5, 0.58, "자유형 롤링과 6비트 킥 타이밍", fontsize=44, weight="bold",
                          color="white", ha="center", va="center")
    title2 = overlay.text(0.5, 0.45, "몸통의 회전, 팔 젓기, 발차기는 어떻게 맞물리는가", fontsize=22,
                          color="#bfe6ff", ha="center", va="center")
    sums = [overlay.text(0.07, 0.80 - i * 0.12, s, fontsize=34 if i == 0 else 21,
                         weight="bold" if i == 0 else "normal", color="#ffd166" if i == 0 else "white",
                         va="center") for i, s in enumerate(SUMMARY)]
    note = overlay.text(0.07, 0.12, "* 코칭에서 쓰는 전형적인 패턴을 단순화한 막대 인형 애니메이션입니다. 각도와 타이밍은 사람마다 다릅니다.",
                        fontsize=13, color="#9fb3c8")

    s = 0.55          # 시작 위상
    w_ = writer()
    with w_.saving(fig, path, dpi=100):
        for fi in range(int(END * FPS)):
            T = fi / FPS
            sc = scene_at(T)
            s += sc[2] * pm.FREQ / FPS
            ph = s % 1.0
            B = body(s)

            # ---- 뒤에서 본 단면
            ys = lambda p: np.atleast_2d(p)[:, [1, 2]]
            ell = np.c_[0.25 * np.cos(th), 0.12 * np.sin(th) - 0.03]
            ph_ = B["phi"]
            R = np.array([[np.cos(ph_), np.sin(ph_)], [-np.sin(ph_), np.cos(ph_)]])
            chest.set_xy(ell @ R.T)
            head_c.center = (0.0, 0.05)        # 머리는 롤링을 따라 돌지 않고 거의 고정 (숨쉴 때만 돌림)
            for k in ("armR", "armL"):
                a2 = ys(B[k])
                arm_c[k].set_data(a2[:, 0], a2[:, 1])
                arm_c[k].set_zorder(12 if B[k][2, 2] > -0.05 else 9)
            a_roll = fade(T, 6.0, 58.0, 0.6)
            e = np.array([[-0.36, 0.0], [0.36, 0.0]]) @ R.T
            sh_line.set_data(e[:, 0], e[:, 1]); sh_line.set_alpha(a_roll)
            deg = np.degrees(ph_)
            yb_c = cax.get_ylim()[0]
            roll_txt.set_text(f"어깨 {abs(deg):.0f}° " + ("오른쪽" if deg > 0 else "왼쪽"))
            roll_txt.set_position((0, yb_c + 0.15)); roll_txt.set_alpha(a_roll)
            roll_txt.get_bbox_patch().set_alpha(0.5 * a_roll)
            hph = B["hip_phi"]
            Rh = np.array([[np.cos(hph), np.sin(hph)], [-np.sin(hph), np.cos(hph)]])
            pelvis.set_xy(np.c_[0.19 * np.cos(th), 0.09 * np.sin(th) - 0.02] @ Rh.T)
            eh = np.array([[-0.30, 0.0], [0.30, 0.0]]) @ Rh.T + [0, -0.02]
            hip_line.set_data(eh[:, 0], eh[:, 1]); hip_line.set_alpha(a_roll)
            hip_txt.set_text(f"골반 {abs(deg) * HIP_RATIO:.0f}°")
            hip_txt.set_position((0, hax.get_ylim()[0] + 0.15)); hip_txt.set_alpha(a_roll)
            hip_txt.get_bbox_patch().set_alpha(0.5 * a_roll)
            for k in ("legR", "legL"):
                hipj = ys(B[k][0])[0]; ank = ys(B[k][2])[0]
                feet[k].set_offsets([ank])
                thighs[k].set_data([hipj[0], ank[0]], [hipj[1], ank[1]])
                foot_hist[k] = (foot_hist[k] + [ank])[-int(1.0 / (sc[2] * pm.FREQ / FPS)):]
                fh = np.array(foot_hist[k])
                trails[k].set_data(fh[:, 0], fh[:, 1])
            # ---- 옆에서 본 시점
            st = side_view(np.vstack([B["torso_sh"], B["torso_hip"], B["head"][None, :] + [0.0, 0, 0]]))
            torso_s.set_xy(hull(st[:-1]))
            hips_s.set_xy(hull(np.vstack([side_view(B["torso_hip"]),
                                          side_view(B["torso_hip"]) + [0.14, 0]])))
            head_s.center = tuple(side_view(B["head"])[0])
            for k in ("armR", "armL"):
                a2 = side_view(B[k]); arm_s[k].set_data(a2[:, 0], a2[:, 1])
            for k in ("legR", "legL"):
                l2 = side_view(B[k]); leg_s[k].set_data(l2[:, 0], l2[:, 1])

            # ---- 타임라인
            cursor.set_data([ph, ph], [0, 4])
            a_beats = fade(T, 21.3, 70.0, 0.6)
            cur_beat = None
            for i, (b, (circ, txt)) in enumerate(zip(BEATS, beat_art)):
                near = abs(wrap(ph - b)) < 0.035
                circ.set_alpha(a_beats * (1.0 if near else 0.55))
                circ.set_sizes([(620 if BEAT_MAJOR[i] else 260) * (1.35 if near else 1.0)])
                txt.set_alpha(a_beats)
                if near:
                    cur_beat = i
            beat_big.set_alpha(a_beats if cur_beat is not None else 0)
            if cur_beat is not None:
                beat_big.set_text(str(cur_beat + 1))
                beat_big.set_color(C_R if BEAT_LEG[cur_beat] == "R" else C_L)
            a_diag = fade(T, 39.3, 70.0, 0.6) if T < 58.0 else 0.0
            for ln in links:
                ln.set_alpha(0.9 * a_diag)
            # 대각선 짝 강조
            on_pair = None
            for side_arm, leg, beat in (("armR", "legL", BEATS[0]), ("armL", "legR", BEATS[3])):
                if abs(wrap(ph - beat)) < 0.06:
                    on_pair = (side_arm, leg)
            for k in glow:
                glow[k].set_alpha(0)
            if on_pair and a_diag > 0.01:
                ank = ys(B[on_pair[1]][2])[0]
                glow[on_pair[1]].set_offsets([ank]); glow[on_pair[1]].set_alpha(0.35 * a_diag)
                hs = side_view(B[on_pair[0]][3])[0]; fs = side_view(B[on_pair[1]][2])[0]
                diag_s.set_data([hs[0], fs[0]], [hs[1], fs[1]]); diag_s.set_alpha(a_diag)
                diag_txt.set_text("오른손 입수 + 왼발 강한 킥" if on_pair[0] == "armR" else "왼손 입수 + 오른발 강한 킥")
                diag_txt.set_alpha(a_diag); diag_txt.get_bbox_patch().set_alpha(0.6 * a_diag)
            else:
                diag_s.set_alpha(0); diag_txt.set_alpha(0)
                diag_txt.get_bbox_patch().set_alpha(0)

            # ---- 텍스트
            header.set_text(sc[3]); header.set_alpha(fade(T, sc[0], sc[1], 0.5) if sc[3] else 0)
            slowmo.set_text(f"슬로모션 ×{sc[2] * pm.FREQ / 0.7:.2f}" if sc[3] else "")
            cap.set_visible(False)
            for a0, a1, txt in CAPTIONS:
                if a0 <= T < a1:
                    al = fade(T, a0, a1, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.65 * al)
                    cap.set_visible(True)
            a_title = 1 - smooth((T - 3.6) / 1.2)
            a_sum = smooth((T - 58.3) / 1.0)
            ov.set_alpha(0.8 * max(a_title, a_sum))
            title1.set_alpha(a_title); title2.set_alpha(a_title)
            for i, st_ in enumerate(sums):
                st_.set_alpha(smooth((T - 58.6 - 0.9 * i) / 0.6))
            note.set_alpha(smooth((T - 63.0) / 0.8))

            w_.grab_frame()
            if fi % 300 == 0:
                print(f"{T:5.1f}s / {END}s", flush=True)
    plt.close(fig)


def main():
    if os.environ.get("PREVIEW"):
        render(os.path.join(TMP, "preview"))
        return
    raw = os.path.join(TMP, "raw.mp4")
    out = os.path.join(OUT, "roll_6beat_timing.mp4")
    render(raw)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print("done:", out)


if __name__ == "__main__":
    main()
