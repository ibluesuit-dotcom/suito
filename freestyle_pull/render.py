"""자유형 손 젓기(풀) 추진 원리 설명 영상 렌더러 (옆에서 본 시점).

python3 render.py  ->  freestyle_pull_propulsion.mp4
PREVIEW=10,20 python3 render.py  ->  _build/ 에 해당 시각 프레임 PNG만 저장
"""
import contextlib
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse
from matplotlib.animation import FFMpegWriter

import pull_model as m

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

FPS = 30
W, H = 12.8, 7.2
XL, XR = -2.3, 1.25
YT = 0.80
YB = YT - (XR - XL) * 9 / 16
SY = m.SURFACE_Y
OUT = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(OUT, "_build")
os.makedirs(TMP, exist_ok=True)

SKIN = "#f2c9a0"
SKIN_D = "#d99a6c"
SUIT = "#1b2a49"
CAP = "#d62828"
PALM = "#ffd166"
C_FORCE = "#ff5a4e"
C_WATER = "#4fc3ff"
C_THRUST = "#5dff8a"
TC = 1 / m.FREQ


# ----------------------------------------------------------------- 공통 도우미
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, ramp=0.6):
    return smooth((t - t0) / ramp) * smooth((t1 - t) / ramp)


def background(ax, xl, xr, yb, yt):
    sky = np.linspace(0, 1, 64)[:, None]
    ax.imshow(np.dstack([0.75 + 0.15 * sky, 0.86 + 0.1 * sky, 0.97 + 0 * sky]),
              extent=(xl, xr, SY, yt), origin="lower", aspect="auto", zorder=0)
    deep = np.linspace(0, 1, 128)[:, None]
    ax.imshow(np.dstack([0.02 + 0.07 * deep, 0.12 + 0.24 * deep, 0.25 + 0.32 * deep]),
              extent=(xl, xr, yb, SY), origin="lower", aspect="auto", zorder=0)
    ax.plot([xl, xr], [SY, SY], color="white", lw=2, alpha=0.85, zorder=3)


class Swimmer:
    """옆에서 본 수영자 (엎드린 자세, +x 방향 전진). 가까운 팔과 먼 팔(반 주기 차이)."""

    def __init__(self, ax, style="high", z=10):
        self.style = style
        xs = np.linspace(-0.62, 0.12, 60)
        top = np.interp(xs, [-0.62, -0.5, -0.2, 0.0, 0.12], [0.13, 0.16, 0.17, 0.17, 0.12])
        bot = np.interp(xs, [-0.62, -0.5, -0.25, -0.05, 0.12], [-0.06, -0.08, -0.07, -0.12, -0.06])
        ax.fill_between(xs, bot, top, color=SKIN, zorder=z, lw=0)
        sel = xs < -0.42
        ax.fill_between(xs[sel], bot[sel], top[sel], color=SUIT, zorder=z + 0.1, lw=0)
        ax.add_patch(Ellipse((0.25, 0.07), 0.25, 0.2, color=SKIN, zorder=z))
        ax.add_patch(Ellipse((0.27, 0.11), 0.22, 0.15, color=CAP, zorder=z + 0.2))
        self.legs = [ax.plot([], [], color=SKIN, lw=lw, solid_capstyle="round", zorder=z - 0.2 + 0.05 * i)[0]
                     for i, lw in enumerate((26, 18, 11))]
        # 먼 팔 (몸 뒤쪽, 흐리게)
        self.far = [ax.plot([], [], color=SKIN_D, lw=lw, solid_capstyle="round", alpha=0.55, zorder=z - 0.5)[0]
                    for lw in (14, 11, 9)]
        # 가까운 팔
        self.near = [ax.plot([], [], color=c, lw=lw, solid_capstyle="round", zorder=z + 0.4)[0]
                     for c, lw in ((SKIN, 15), (SKIN, 12), (SKIN_D, 10))]
        self.palm, = ax.plot([], [], color=PALM, lw=4, solid_capstyle="round", zorder=z + 0.45)
        self.jdots = ax.scatter([], [], s=22, c="#7a4a2a", zorder=z + 0.5)

    def update(self, t):
        P = m.arm(t, self.style)
        for ln, a, b in zip(self.near, P[:-1], P[1:]):
            ln.set_data([a[0], b[0]], [a[1], b[1]])
        # 손바닥 = 손의 뒤쪽(진행 반대쪽)을 향한 면
        Wr, Tp = P[2], P[3]
        d = (Tp - Wr) / np.linalg.norm(Tp - Wr)
        n = np.array([-d[1], d[0]])
        if n[0] > 0:
            n = -n
        o = n * 0.022
        self.palm.set_data([Wr[0] + o[0], Tp[0] + o[0]], [Wr[1] + o[1], Tp[1] + o[1]])
        self.palm.set_alpha(1.0 if Wr[1] < SY else 0.35)
        Pf = m.arm(t + 0.5 * TC, self.style)
        for ln, a, b in zip(self.far, Pf[:-1], Pf[1:]):
            ln.set_data([a[0] + 0.03, b[0] + 0.03], [a[1] + 0.02, b[1] + 0.02])
        # 다리: 작은 자유형 발차기
        ph = 2 * np.pi * 3 * m.FREQ * t
        hip = np.array([-0.58, 0.03])
        a1 = 0.10 * np.sin(ph); a2 = a1 - 0.18 * max(0, np.cos(ph + 0.4)); a3 = a2 - 0.25
        K = hip + 0.42 * np.array([-np.cos(a1), -np.sin(a1)])
        A = K + 0.40 * np.array([-np.cos(a2), -np.sin(a2)])
        Tt = A + 0.18 * np.array([-np.cos(a3), -np.sin(a3)])
        for ln, a, b in zip(self.legs, (hip, K, A), (K, A, Tt)):
            ln.set_data([a[0], b[0]], [a[1], b[1]])
        self.jdots.set_offsets(np.array([P[1], P[2]]))
        return P


def arrow(ax, color, z=20, lw=3.5, ms=22):
    a = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z)
    ax.add_patch(a)
    return a


def caption_box(fig):
    return fig.text(0.025, 0.03, "", fontsize=17, color="white", va="bottom", ha="left",
                    linespacing=1.45, zorder=50,
                    bbox=dict(boxstyle="round,pad=0.6", fc=(0, 0, 0, 0.6), ec="none"))


class PhaseRibbon:
    """한 스트로크의 단계 띠 + 현재 위치 표시."""

    def __init__(self, fig, rect):
        ax = fig.add_axes(rect)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        self.rects = []
        for a, b, name, col in m.PHASES:
            r = Rectangle((a, 0.15), b - a, 0.7, color=col, alpha=0.35, lw=0)
            ax.add_patch(r)
            tx = ax.text((a + b) / 2, 0.5, name, ha="center", va="center", fontsize=12,
                         color="#0d2240", weight="bold")
            self.rects.append((r, tx))
        self.marker = ax.plot([0, 0], [0.0, 1.0], color="#0d2240", lw=3)[0]
        self.ax = ax

    def update(self, s, alpha):
        s = s % 1.0
        for (a, b, _, _), (r, tx) in zip(m.PHASES, self.rects):
            on = a <= s < b
            r.set_alpha((0.95 if on else 0.3) * alpha)
            tx.set_alpha(alpha)
        self.marker.set_data([s, s], [0.0, 1.0]); self.marker.set_alpha(alpha)
        self.ax.set_visible(alpha > 0.01)


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
    return FFMpegWriter(fps=FPS, codec="libx264", bitrate=6000,
                        extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])


PEAK = {st: max(m.total_fx(t, st) for t in np.linspace(0, TC, 800)) for st in ("high", "dropped")}


def draw_thrust_graph(gax, ht, hf, alpha, window=1.3 * TC, title="추진력 (앞쪽 성분, 상대값)",
                      ylim=(-2.2, 8.2), small=False):
    t = np.array(ht); f = np.array(hf)
    sel = t > t[-1] - window
    t, f = t[sel], f[sel]
    gax.set_facecolor((0, 0, 0, 0.6 * alpha))
    for sp in gax.spines.values():
        sp.set_color((1, 1, 1, 0.4 * alpha))
    x = t - t[-1]
    ph = np.array([m.phase_of(tt * m.FREQ)[0] for tt in t])
    for a, b, name, col in m.PHASES:
        if name == "리커버리":
            continue
        gax.fill_between(x, ylim[0], ylim[1], where=ph == name, color=col,
                         alpha=0.16 * alpha, lw=0, step="mid")
    gax.axhline(0, color=(1, 1, 1, 0.5 * alpha), lw=1)
    gax.fill_between(x, 0, f, where=f > 0, color=C_THRUST, alpha=0.5 * alpha, lw=0)
    gax.fill_between(x, 0, f, where=f < 0, color=C_FORCE, alpha=0.5 * alpha, lw=0)
    gax.plot(x, f, color="white", lw=2, alpha=alpha)
    gax.plot([x[-1]], [f[-1]], "o", color="white", ms=7, alpha=alpha)
    gax.set_xlim(-window, 0.05 * window); gax.set_ylim(*ylim)
    gax.set_xticks([]); gax.set_yticks([])
    fs = 11 if small else 13
    gax.text(0.02, 0.95, title, transform=gax.transAxes, color="white", fontsize=fs, va="top", alpha=alpha)
    gax.text(0.98, 0.95, "+ 앞으로", transform=gax.transAxes, color=C_THRUST, fontsize=fs - 1,
             ha="right", va="top", alpha=alpha)
    gax.text(0.98, 0.04, "- 브레이크", transform=gax.transAxes, color=C_FORCE, fontsize=fs - 1,
             ha="right", alpha=alpha)


def part_shares(style="high"):
    ts = np.linspace(0, TC, 800, endpoint=False)
    pos = np.zeros(3)
    for t in ts:
        pos += np.maximum(m.segment_forces(t, style)[0][:, 0], 0)
    return pos / pos.sum()


# ----------------------------------------------------------------- 메인 장면
SCENES = {
    "title":   (0.0, 5.0, 0.5),
    "s1":      (5.0, 22.0, 0.45),
    "s2":      (22.0, 44.0, 0.3),
    "s3":      (44.0, 62.0, 0.45),
    "summary": (62.0, 73.0, 0.5),
}
MAIN_END = 73.0
S3_END = SCENES["s3"][1]

CAPTIONS = [
    (5.0, 11.0, "옆에서 본 자유형 한 팔 스트로크 (앞쪽의 진한 팔)\n엔트리 → 캐치 → 풀(당기기) → 푸시(밀기) → 리커버리"),
    (11.0, 16.5, "캐치 : 팔꿈치는 높게 둔 채 손목·아래팔을 먼저 세워 '물을 잡는다'\n풀·푸시 : 손바닥(노랑)과 아래팔을 뒤로 향한 채 가슴 아래 → 허벅지까지 점점 빠르게"),
    (16.5, 22.0, "흰 점선 = 몸 기준 손의 경로,  노란 선 = 물 기준 손의 궤적\n몸이 앞으로 나아가는 동안 손은 물을 뒤로 밀며 물속에 작은 고리를 그린다"),
    (22.0, 28.0, "파란 화살표 : 손·아래팔이 물을 미는 힘\n빨간 화살표 : 물이 되미는 힘 (같은 크기·반대 방향) → 그 앞쪽 성분이 추진력"),
    (28.0, 35.5, "추진력은 캐치 후 점점 커져 푸시 후반, 손이 허리 → 엉덩이로 밀려 올라갈 때 최대!\n손이 가장 빠르고, 손바닥이 뒤를 향한 채 물을 뒤로 밀어내기 때문"),
    (35.5, 44.0, "부위별 추진력 : 손바닥이 약 {hand}%, 아래팔이 약 {fore}%, 위팔은 거의 0\n→ 손과 아래팔을 하나의 '노'처럼 세워 쓰는 것이 핵심"),
    (44.0, 50.0, "점 하나하나가 물 입자.  밝을수록 손 젓기로 빨라진 물\n(카메라가 몸을 따라가므로 물은 오른쪽 → 왼쪽으로 흘러간다)"),
    (50.0, 56.0, "손이 지나간 길을 따라 물이 뒤로 밀려나며 큰 소용돌이가 생겨 뒤로 흘러간다\n빨강 = 반시계 회전,  파랑 = 시계 회전"),
    (56.0, 62.0, "흰 화살표 = 물 자체의 움직임 : 손이 지나간 깊이에서 물이 뒤로 흐른다\n물에 '뒤쪽' 운동량을 준 만큼, 몸은 '앞쪽'으로 나아간다"),
]

HEADERS = {
    "s1": "① 손 젓기 동작 : 다섯 단계",
    "s2": "② 추진력은 언제, 어디서 가장 클까?",
    "s3": "③ 물의 흐름 : 뒤로 밀려나는 물과 소용돌이",
}

SUMMARY = [
    "정리",
    "1.  추진력은 손과 아래팔로 물을 '뒤로' 미는 반작용 — 손바닥 약 {hand}%, 아래팔 약 {fore}%",
    "2.  가장 강한 구간은 푸시 후반 : 손이 허리 → 엉덩이로 가속하며 밀 때",
    "3.  캐치에서 팔꿈치를 높게, 아래팔을 일찍 세워야 추진 구간이 길어진다",
    "4.  엔트리·리커버리는 추진이 아닌 준비 — 물을 가르듯 부드럽게",
]


def scene_at(T):
    for k, (a, b, _) in SCENES.items():
        if a <= T < b:
            return k
    return "summary"


def recycle(px, py, rng):
    out = (px < XL - 0.2) | (py > SY - 0.01) | (py < YB - 0.1)
    n = out.sum()
    if n:
        px[out] = rng.uniform(XR + 0.02, XR + 0.25, n)
        py[out] = rng.uniform(YB, SY - 0.02, n)
    return px, py


def vorticity_rgba(wake, gx, gy, alpha):
    sel = wake.vx > XL - 0.2
    vx, vy, g = wake.vx[sel], wake.vy[sel], wake.g[sel]
    d2 = (wake.delta * 1.3) ** 2
    X = gx.ravel()[:, None]; Y = gy.ravel()[:, None]
    r2 = (X - vx[None, :]) ** 2 + (Y - vy[None, :]) ** 2
    om = (g[None, :] / (np.pi * d2) * np.exp(-r2 / d2)).sum(1).reshape(gx.shape)
    s = np.clip(om / 10.0, -1, 1)
    rgba = np.zeros(gx.shape + (4,))
    rgba[s > 0] = (1.0, 0.28, 0.28, 0)
    rgba[s <= 0] = (0.3, 0.6, 1.0, 0)
    rgba[..., 3] = np.abs(s) ** 0.8 * 0.85 * alpha
    return rgba


def render_main(path):
    shares = part_shares("high")
    pct = dict(hand=int(round(shares[2] * 100)), fore=int(round(shares[1] * 100)))
    caps = [(a, b, txt.format(**pct)) for a, b, txt in CAPTIONS]

    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.axis("off")
    background(ax, XL, XR, YB, YT)

    gx, gy = np.meshgrid(np.linspace(XL, XR, 180), np.linspace(YB, SY, 90))
    vort_img = ax.imshow(np.zeros(gx.shape + (4,)), extent=(XL, XR, YB, SY),
                         origin="lower", aspect="auto", zorder=4, interpolation="bilinear")

    rng = np.random.default_rng(1)
    NP = 2400
    px = rng.uniform(XL - 0.1, XR + 0.1, NP)
    py = rng.uniform(YB, SY - 0.02, NP)
    tracers = ax.scatter(px, py, s=5, c="white", zorder=5, lw=0)

    qx, qy = np.meshgrid(np.arange(-2.2, 0.8, 0.16), np.arange(-1.05, 0.01, 0.14))
    qx, qy = qx.ravel(), qy.ravel()
    quiv = ax.quiver(qx, qy, 0 * qx, 0 * qy, color="white", scale=9, width=0.0024,
                     headwidth=4, zorder=8, alpha=0)

    sw = Swimmer(ax)
    path_body = ax.plot([], [], color="white", lw=1.5, ls="--", alpha=0, zorder=6)[0]
    trail = LineCollection([], linewidths=3, zorder=7)
    ax.add_collection(trail)

    a_force = [arrow(ax, C_FORCE), arrow(ax, C_FORCE, lw=2.5, ms=16)]
    a_water = [arrow(ax, C_WATER), arrow(ax, C_WATER, lw=2.5, ms=16)]
    a_thrust = arrow(ax, C_THRUST, lw=5, ms=28)
    l_force = ax.text(0, 0, "물 → 손", color=C_FORCE, fontsize=15, weight="bold", zorder=31)
    l_water = ax.text(0, 0, "손 → 물", color=C_WATER, fontsize=15, weight="bold", zorder=31)
    l_thrust = ax.text(0, 0, "추진력", color=C_THRUST, fontsize=16, weight="bold", zorder=31)
    l_peak = ax.text(0, 0, "최대 추진!", color="#fff176", fontsize=20, weight="bold", zorder=32,
                     bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.55), ec="none"))
    l_palm = ax.text(0, 0, "손바닥", color=PALM, fontsize=14, weight="bold", zorder=31)

    header = fig.text(0.02, 0.945, "", fontsize=25, weight="bold", color="#0d2240", va="center", zorder=40)
    slowmo = fig.text(0.975, 0.945, "", fontsize=14, color="#0d2240", ha="right", va="center", zorder=40)
    ribbon = PhaseRibbon(fig, [0.30, 0.83, 0.67, 0.06])
    legend1 = fig.text(0.02, 0.74, "● 반시계 회전", fontsize=14, color="#ff6b6b", zorder=40)
    legend2 = fig.text(0.13, 0.74, "● 시계 회전", fontsize=14, color="#6ea8ff", zorder=40)
    cap = caption_box(fig)

    gax = fig.add_axes([0.025, 0.20, 0.38, 0.23])
    bax = fig.add_axes([0.025, 0.455, 0.38, 0.05])

    overlay = fig.add_axes([0, 0, 1, 1], zorder=60); overlay.axis("off")
    overlay.set_xlim(0, 1); overlay.set_ylim(0, 1)
    ov_rect = overlay.add_patch(Rectangle((0, 0), 1, 1, color=(0.02, 0.07, 0.15), alpha=0))
    title1 = overlay.text(0.5, 0.58, "자유형 손 젓기의 추진 원리", fontsize=46, weight="bold",
                          color="white", ha="center", va="center")
    title2 = overlay.text(0.5, 0.45, "추진력은 어떻게 생기고, 어느 구간에서 가장 강할까",
                          fontsize=22, color="#bfe6ff", ha="center", va="center")
    sum_txts = [overlay.text(0.08, 0.80 - i * 0.12, s.format(**pct), fontsize=34 if i == 0 else 21,
                             weight="bold" if i == 0 else "normal",
                             color="#ffd166" if i == 0 else "white", va="center")
                for i, s in enumerate(SUMMARY)]
    note = overlay.text(0.08, 0.12, "* 옆에서 본 평면만 다룬 단순화한 2차원 물리 모델(압력 항력 + 와류법)의 시각화이며, 힘의 크기는 상대값입니다.",
                        fontsize=13, color="#9fb3c8")

    # 한 스트로크의 손목 경로 (몸 기준)
    ss = np.linspace(0, TC, 300)
    body_path = np.array([m.arm(t)[2] for t in ss])

    wake = m.Wake()
    pre_dt = 0.5 / FPS
    wake.t = 0.05 * TC - 2.5 * TC
    for _ in range(int(2.5 * TC / pre_dt)):
        u, v = wake.velocity(px, py)
        px += (u - m.U) * pre_dt; py += v * pre_dt
        wake.step(pre_dt)
        px, py = recycle(px, py, rng)
    hist_t, hist_f = [], []
    hand_pts = []

    w = writer()
    with w.saving(fig, path, dpi=100):
        for fi in range(int(MAIN_END * FPS)):
            T = fi / FPS
            sc = scene_at(T)
            spd = SCENES[sc][2]
            dts = spd / FPS
            u, v = wake.velocity(px, py)
            px += (u - m.U) * dts; py += v * dts
            wake.step(dts)
            px, py = recycle(px, py, rng)
            t = wake.t
            s = t * m.FREQ
            P = sw.update(t)
            F, C = m.segment_forces(t)
            fx = F[:, 0].sum()
            hist_t.append(t); hist_f.append(fx)
            c_hand = 0.5 * (P[2] + P[3])
            hand_pts = [(x - m.U * dts, y) for x, y in hand_pts] + [tuple(c_hand)]
            hand_pts = [p for p in hand_pts if p[0] > XL - 0.2]

            # --- 물 입자
            u, v = wake.velocity(px, py)
            sp = np.hypot(u, v)
            if sc == "s3":
                a_tr = 0.25 + 0.75 * smooth((T - 44.0) / 1.0)
            elif sc == "summary":
                a_tr = 1.0 - 0.65 * smooth((T - 62.0) / 1.0)
            else:
                a_tr = 0.3
            cval = np.clip(sp / 0.35, 0, 1)
            cols = np.empty((len(px), 4))
            cols[:, 0] = 0.35 + 0.65 * cval; cols[:, 1] = 0.6 + 0.4 * cval; cols[:, 2] = 1.0
            cols[:, 3] = a_tr * (0.35 + 0.65 * cval)
            tracers.set_offsets(np.c_[px, py]); tracers.set_facecolors(cols)
            tracers.set_sizes(4 + 14 * cval)

            # --- 소용돌이 / 물 흐름 화살표 (s3)
            a_v = fade(T, 49.5, 62.6, 1.0)
            if a_v > 0.01:
                vort_img.set_data(vorticity_rgba(wake, gx, gy, a_v)); vort_img.set_visible(True)
            else:
                vort_img.set_visible(False)
            legend1.set_alpha(a_v); legend2.set_alpha(a_v)
            a_q = fade(T, 55.5, 62.6, 1.0)
            if a_q > 0.01:
                uq, vq = wake.velocity(qx, qy, delta=0.1)
                quiv.set_UVC(uq, vq); quiv.set_alpha(a_q)
            else:
                quiv.set_alpha(0)

            # --- 손 경로 (s1)
            a_bp = fade(T, 7.0, 16.5, 0.8)
            path_body.set_data(body_path[:, 0], body_path[:, 1]); path_body.set_alpha(0.6 * a_bp)
            a_trail = fade(T, 16.0, 22.3, 0.8)
            pts = np.array(hand_pts)
            if a_trail > 0.01 and len(pts) > 1:
                segs = np.stack([pts[:-1], pts[1:]], 1)
                inw = (pts[1:, 1] < SY).astype(float)
                al = np.linspace(0, 1, len(segs)) * a_trail * (0.15 + 0.85 * inw)
                trail.set_segments(segs)
                trail.set_color([(1, 0.82, 0.4, a) for a in al])
            else:
                trail.set_segments([])
            a_palm = fade(T, 11.0, 16.5, 0.5)
            l_palm.set_position((P[3][0] + 0.04, P[3][1] - 0.06))
            l_palm.set_alpha(a_palm if P[2][1] < SY else 0); l_palm.set_visible(a_palm > 0.01)

            # --- 힘 화살표 (s2)
            a_f = fade(T, 22.3, 44.0, 0.8)
            vis = a_f > 0.01
            sc_f = 0.09
            for k, seg in enumerate((2, 1)):
                Ff, c = F[seg], C[seg]
                on = vis and np.linalg.norm(Ff) * sc_f > 0.015
                a_force[k].set_visible(on); a_water[k].set_visible(on)
                a_force[k].set_alpha(a_f); a_water[k].set_alpha(a_f)
                if on:
                    a_force[k].set_positions(tuple(c), tuple(c + Ff * sc_f))
                    a_water[k].set_positions(tuple(c), tuple(c - Ff * sc_f))
            Fh, ch = F[2], C[2]
            on_l = vis and np.linalg.norm(Fh) * sc_f > 0.05
            l_force.set_position(tuple(ch + Fh * sc_f + [0.02, 0.02]))
            l_water.set_position(tuple(ch - Fh * sc_f + [-0.2, -0.05]))
            for lab in (l_force, l_water):
                lab.set_visible(on_l); lab.set_alpha(a_f)
            base = np.array([0.05, 0.42])
            fxa = fx * 0.06
            a_thrust.set_positions(tuple(base), tuple(base + [fxa, 0]))
            a_thrust.set_color(C_THRUST if fx >= 0 else C_FORCE)
            show_t = vis and abs(fxa) > 0.02
            a_thrust.set_visible(show_t); a_thrust.set_alpha(a_f)
            l_thrust.set_text("추진력" if fx >= 0 else "브레이크")
            l_thrust.set_color(C_THRUST if fx >= 0 else C_FORCE)
            l_thrust.set_position((base[0] - 0.02, base[1] + 0.06))
            l_thrust.set_visible(show_t); l_thrust.set_alpha(a_f)
            peak_on = vis and fx > 0.85 * PEAK["high"] and T > 28.0
            l_peak.set_visible(peak_on)
            l_peak.set_position((ch[0] - 0.25, ch[1] - 0.22))

            # --- 그래프 + 부위별 비율 (s2)
            a_g = fade(T, 22.3, 44.0, 0.8)
            gax.clear()
            if a_g > 0.01:
                draw_thrust_graph(gax, hist_t, hist_f, a_g)
                gax.set_visible(True)
            else:
                gax.set_visible(False)
            a_b = fade(T, 35.5, 44.0, 0.6)
            bax.clear()
            if a_b > 0.01:
                bax.set_visible(True)
                bax.set_xlim(0, 1); bax.set_ylim(0, 1); bax.axis("off")
                x0 = 0
                for frac, name, col in ((shares[2], "손바닥", PALM), (shares[1], "아래팔", SKIN_D),
                                        (shares[0], "", "#888888")):
                    bax.add_patch(Rectangle((x0, 0), frac, 1, color=col, alpha=a_b, lw=0))
                    if frac > 0.08:
                        bax.text(x0 + frac / 2, 0.5, f"{name} {frac * 100:.0f}%", ha="center", va="center",
                                 fontsize=13, weight="bold", color="#0d2240", alpha=a_b)
                    x0 += frac
            else:
                bax.set_visible(False)

            # --- 텍스트
            ribbon.update(s, fade(T, 5.3, 62.0, 0.5))
            if sc in HEADERS:
                a0, a1, _ = SCENES[sc]
                al = fade(T, a0, a1, 0.5)
                header.set_text(HEADERS[sc]); header.set_alpha(al)
                slowmo.set_text(f"슬로모션 ×{spd:g}"); slowmo.set_alpha(al)
            else:
                header.set_alpha(0); slowmo.set_alpha(0)
            cap.set_visible(False)
            for c0_, c1_, txt in caps:
                if c0_ <= T < c1_:
                    al = fade(T, c0_, c1_, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.6 * al)
                    cap.set_visible(True)

            a_title = 1 - smooth((T - 3.6) / 1.2)
            a_sum = smooth((T - 62.3) / 1.0)
            ov_rect.set_alpha(0.75 * max(a_title, a_sum))
            title1.set_alpha(a_title); title2.set_alpha(a_title)
            for i, st in enumerate(sum_txts):
                st.set_alpha(smooth((T - 62.6 - 0.9 * i) / 0.6))
            note.set_alpha(smooth((T - 67.0) / 0.8))

            w.grab_frame()
            if fi % 150 == 0:
                print(f"main {T:5.1f}s / {MAIN_END}s", flush=True)
    plt.close(fig)


# ----------------------------------------------------------------- 비교 장면
COMP_LEN = 16.0
COMP_SPEED = 0.3


def render_compare(path):
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("#06182e")
    fig.text(0.025, 0.945, "④ 높은 팔꿈치 vs 팔꿈치가 떨어진 스트로크", fontsize=26, weight="bold",
             color="white", va="center")
    fig.text(0.975, 0.945, f"슬로모션 ×{COMP_SPEED:g}", fontsize=15, color="#bfe6ff", ha="right", va="center")
    rows = [("high", "높은 팔꿈치 : 아래팔을 일찍 세운다", 0.50),
            ("dropped", "팔꿈치가 떨어짐 : 팔꿈치가 먼저 가라앉는다", 0.135)]
    xl, xr, yt = -1.6, 1.4, 0.46
    yb = yt - (xr - xl) * (0.355 * H) / (0.52 * W)
    panels = []
    for style, label, y0 in rows:
        ax = fig.add_axes([0.01, y0, 0.52, 0.355])
        ax.set_xlim(xl, xr); ax.set_ylim(yb, yt); ax.axis("off")
        background(ax, xl, xr, yb, yt)
        sw = Swimmer(ax, style=style)
        af = arrow(ax, C_FORCE, ms=18, lw=3); aw = arrow(ax, C_WATER, ms=18, lw=3)
        at = arrow(ax, C_THRUST, ms=22, lw=4.5)
        ax.text(xl + 0.03, yt - 0.04, label, fontsize=15, weight="bold", color="#0d2240", va="top", zorder=40)
        gax = fig.add_axes([0.55, y0 + 0.02, 0.30, 0.32])
        bax = fig.add_axes([0.875, y0 + 0.02, 0.095, 0.32])
        panels.append(dict(style=style, sw=sw, af=af, aw=aw, at=at, gax=gax, bax=bax, ht=[], hf=[]))
    cap = caption_box(fig)
    cap.set_position((0.025, 0.012)); cap.set_fontsize(15)
    caps = [
        (0.0, 8.0, "위 : 팔꿈치를 높게 두고 아래팔을 세우면 캐치부터 손바닥이 뒤를 향해 물을 뒤로 민다\n아래 : 팔꿈치가 먼저 가라앉으면 손바닥이 아래를 향해 물을 '누르기'만 하고, 푸시도 약해진다"),
        (8.0, COMP_LEN, "오른쪽 막대 = 한 스트로크 평균 추진력 (상대값)\n같은 팔 젓기라도 손·아래팔이 '어느 쪽을 향하느냐'에 따라 추진력이 크게 달라진다"),
    ]
    ts = np.linspace(0, TC, 800, endpoint=False)
    means = {p["style"]: np.mean([m.total_fx(tt, p["style"]) for tt in ts]) for p in panels}
    ymax = max(means.values()) * 1.5
    t = 0.1 * TC
    step = COMP_SPEED / FPS
    for p in panels:
        for k in range(int(1.3 * TC / step)):
            tt = t - 1.3 * TC + k * step
            p["ht"].append(tt); p["hf"].append(m.total_fx(tt, p["style"]))

    w = writer()
    with w.saving(fig, path, dpi=100):
        for fi in range(int(COMP_LEN * FPS)):
            T = fi / FPS
            t += step
            for p in panels:
                P = p["sw"].update(t)
                F, C = m.segment_forces(t, p["style"])
                Ff = F[2] + F[1]; c = C[2]
                on = np.linalg.norm(Ff) * 0.05 > 0.012
                p["af"].set_visible(on); p["aw"].set_visible(on)
                if on:
                    p["af"].set_positions(tuple(c), tuple(c + 0.05 * Ff))
                    p["aw"].set_positions(tuple(c), tuple(c - 0.05 * Ff))
                fx = F[:, 0].sum()
                base = np.array([0.1, 0.36])
                p["at"].set_positions(tuple(base), tuple(base + [0.05 * fx, 0]))
                p["at"].set_color(C_THRUST if fx >= 0 else C_FORCE)
                p["at"].set_visible(abs(fx) > 0.3)
                p["ht"].append(t); p["hf"].append(fx)
                p["gax"].clear()
                draw_thrust_graph(p["gax"], p["ht"], p["hf"], 1.0, title="추진력", small=True)
                b = p["bax"]; b.clear()
                mv = means[p["style"]]
                b.set_facecolor((0, 0, 0, 0.55))
                b.bar([0], [mv], color=C_THRUST if mv > 0 else C_FORCE, width=0.6)
                b.axhline(0, color="white", lw=1)
                b.set_xlim(-0.6, 0.6); b.set_ylim(-0.15 * ymax, ymax)
                b.set_xticks([]); b.set_yticks([])
                for sp in b.spines.values():
                    sp.set_color((1, 1, 1, 0.4))
                b.text(0, 0.97 * ymax, "평균", color="white", ha="center", va="top", fontsize=12)
                b.text(0, mv + 0.04 * ymax, f"{mv:+.2f}", color="white", ha="center", va="bottom",
                       fontsize=13, weight="bold")
            for c0, c1, txt in caps:
                if c0 <= T < c1:
                    al = fade(T, c0, c1, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.6 * al)
            w.grab_frame()
            if fi % 150 == 0:
                print(f"compare {T:5.1f}s", flush=True)
    plt.close(fig)


# ----------------------------------------------------------------- 조립
def main():
    main_mp4 = os.path.join(TMP, "main.mp4")
    comp_mp4 = os.path.join(TMP, "compare.mp4")
    if os.environ.get("PREVIEW"):
        if os.environ.get("ONLY") != "compare":
            render_main(main_mp4)
        if os.environ.get("ONLY") != "main":
            render_compare(comp_mp4)
        return
    render_main(main_mp4)
    render_compare(comp_mp4)
    a = os.path.join(TMP, "a.mp4"); c = os.path.join(TMP, "c.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", main_mp4, "-t", str(S3_END),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", a], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(S3_END), "-i", main_mp4,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", c], check=True)
    lst = os.path.join(TMP, "list.txt")
    with open(lst, "w") as fh:
        for f in (a, comp_mp4, c):
            fh.write(f"file '{f}'\n")
    out = os.path.join(OUT, "freestyle_pull_propulsion.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print("done:", out)


if __name__ == "__main__":
    main()
