"""평영 발차기 추진 원리 설명 영상 렌더러 (위에서 내려다본 시점).

python3 render.py  ->  breaststroke_kick_propulsion.mp4
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

import kick_model as m

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

FPS = 30
W, H = 12.8, 7.2
XL, XR = -5.4, 2.6
YB = -(XR - XL) * 9 / 16 * 0.55
YT = YB + (XR - XL) * 9 / 16
OUT = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(OUT, "_build")
os.makedirs(TMP, exist_ok=True)

SKIN = "#f2c9a0"
SKIN_D = "#d99a6c"
SUIT = "#1b2a49"
CAP = "#d62828"
SOLE = "#ffd166"
C_FORCE = "#ff5a4e"     # 물이 발을 미는 힘
C_WATER = "#4fc3ff"     # 발이 물을 미는 힘
C_THRUST = "#5dff8a"    # 추진력 (앞 성분)
TC = 1 / m.FREQ         # 한 주기 (시뮬레이션 시간)


# ----------------------------------------------------------------- 공통 도우미
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, ramp=0.6):
    return smooth((t - t0) / ramp) * smooth((t1 - t) / ramp)


def pool_background(ax, x0, x1, y0, y1):
    yy = np.linspace(0, 1, 128)[:, None]
    xx = np.linspace(0, 1, 2)[None, :]
    g = 0.5 + 0.5 * np.cos((yy - 0.5) * np.pi) + 0 * xx
    ax.imshow(np.dstack([0.03 + 0.05 * g, 0.20 + 0.14 * g, 0.36 + 0.16 * g]),
              extent=(x0, x1, y0, y1), origin="lower", aspect="auto", zorder=0)
    # 수영장 바닥 레인 표시선
    ax.add_patch(Rectangle((x0, -0.12), x1 - x0, 0.24, color=(0.02, 0.08, 0.18), alpha=0.35, zorder=0.5, lw=0))


class Swimmer:
    """위에서 본 수영자: 몸통·머리·앞으로 뻗은 팔(고정) + 두 다리(매 프레임 갱신)."""

    def __init__(self, ax, z=10):
        xs = np.linspace(-0.3, 1.62, 60)
        half = np.interp(xs, [-0.3, 0.0, 0.5, 1.1, 1.4, 1.62], [0.30, 0.34, 0.29, 0.40, 0.45, 0.22])
        ax.fill_between(xs, -half, half, color=SKIN, zorder=z, lw=0)
        sel = xs < 0.35
        ax.fill_between(xs[sel], -half[sel], half[sel], color=SUIT, zorder=z + 0.1, lw=0)
        ax.add_patch(Ellipse((1.92, 0), 0.62, 0.48, color=SKIN, zorder=z))
        ax.add_patch(Ellipse((1.98, 0), 0.52, 0.47, color=CAP, zorder=z + 0.2))
        for sd in (+1, -1):     # 앞으로 뻗은 팔 (킥 드릴 자세)
            ax.plot([1.35, 3.6], [sd * 0.40, sd * 0.10], color=SKIN, lw=16,
                    solid_capstyle="round", zorder=z - 0.1)
        self.legs = []
        for sd in (+1, -1):
            thigh, = ax.plot([], [], color=SKIN, lw=30, solid_capstyle="round", zorder=z + 0.05)
            shank, = ax.plot([], [], color=SKIN, lw=21, solid_capstyle="round", zorder=z + 0.3)
            foot, = ax.plot([], [], color=SKIN_D, lw=14, solid_capstyle="round", zorder=z + 0.4)
            sole, = ax.plot([], [], color=SOLE, lw=6, solid_capstyle="round", zorder=z + 0.45)
            self.legs.append((sd, thigh, shank, foot, sole))
        self.jdots = ax.scatter([], [], s=26, c="#7a4a2a", zorder=z + 0.5)

    def update(self, t, ankle="flex"):
        dots = []
        for sd, thigh, shank, foot, sole in self.legs:
            Hh, K, A, T, flex = m.leg(t, sd, ankle)
            thigh.set_data([Hh[0], K[0]], [Hh[1], K[1]])
            shank.set_data([K[0], A[0]], [K[1], A[1]])
            foot.set_data([A[0], T[0]], [A[1], T[1]])
            d = (T - A) / np.linalg.norm(T - A)
            n = np.array([-d[1], d[0]])
            if n[0] > 0:            # 발바닥 쪽 = 뒤쪽을 향하는 면
                n = -n
            o = n * 0.09
            sole.set_data([A[0] + o[0], T[0] + o[0]], [A[1] + o[1], T[1] + o[1]])
            sole.set_alpha(smooth(flex * 1.3))
            dots += [K, A]
        self.jdots.set_offsets(np.array(dots))


def arrow(ax, color, z=20, lw=3.5, ms=22):
    a = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z)
    ax.add_patch(a)
    return a


def caption_box(fig):
    return fig.text(0.025, 0.03, "", fontsize=17, color="white", va="bottom", ha="left",
                    linespacing=1.45, zorder=50,
                    bbox=dict(boxstyle="round,pad=0.6", fc=(0, 0, 0, 0.6), ec="none"))


class AnkleInset:
    """옆에서 본 발목 모양 (포인트 ↔ 플렉스)."""

    def __init__(self, fig, rect):
        ax = fig.add_axes(rect)
        ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.75, 0.75); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((-0.75, -0.75), 1.5, 1.5, color=(0, 0, 0, 0.5), zorder=0))
        ax.text(0, 0.62, "옆에서 본 발목", color="white", fontsize=12, ha="center", va="center")
        self.shank, = ax.plot([0, 0], [0.45, -0.05], color=SKIN, lw=9, solid_capstyle="round")
        self.foot, = ax.plot([], [], color=SKIN_D, lw=7, solid_capstyle="round")
        self.sole, = ax.plot([], [], color=SOLE, lw=3)
        self.label = ax.text(0, -0.62, "", color="white", fontsize=13, ha="center", va="center", weight="bold")
        self.ax = ax

    def update(self, flex, alpha=1.0):
        # 포인트: 발이 정강이와 일직선(아래), 플렉스: 발이 정강이와 직각(앞쪽)
        ang = -np.pi / 2 + flex * (np.pi / 2 - 0.15)
        A = np.array([0, -0.05])
        T = A + 0.42 * np.array([np.cos(ang), np.sin(ang)])
        self.foot.set_data([A[0], T[0]], [A[1], T[1]])
        n = np.array([np.sin(ang), -np.cos(ang)]) * 0.06
        self.sole.set_data([A[0] + n[0], T[0] + n[0]], [A[1] + n[1], T[1] + n[1]])
        self.sole.set_alpha(alpha)
        self.label.set_text("플렉스" if flex > 0.5 else "포인트")
        self.label.set_color(SOLE if flex > 0.5 else "white")
        for a in self.ax.get_children():
            if hasattr(a, "set_alpha") and a is not self.sole:
                try:
                    a.set_alpha(alpha)
                except Exception:
                    pass
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


def leg_forces(t, ankle="flex"):
    """양 다리 합산: 발(+정강이 절반)에 걸리는 힘과 위치, 전체 앞쪽 힘."""
    out = []
    for sd in (+1, -1):
        F, C = m.segment_forces(t, sd, ankle)
        out.append((F[2] + 0.5 * F[1], C[2]))
    return out, m.total_fx(t, ankle)


def draw_thrust_graph(gax, ht, hf, alpha, window=1.6 * TC, title="추진력 (앞쪽 성분, 상대값)",
                      ylim=(-12.5, 12.5), small=False):
    t = np.array(ht); f = np.array(hf)
    sel = t > t[-1] - window
    t, f = t[sel], f[sel]
    gax.set_facecolor((0, 0, 0, 0.6 * alpha))
    for sp in gax.spines.values():
        sp.set_color((1, 1, 1, 0.4 * alpha))
    x = t - t[-1]
    ph = np.array([m.phase_name(tt * m.FREQ)[0] for tt in t])
    for name, col in (("리커버리", "#c7a6ff"), ("킥", "#ffb347")):
        gax.fill_between(x, ylim[0], ylim[1], where=ph == name, color=col,
                         alpha=0.15 * alpha, lw=0, step="mid")
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
    gax.text(0.02, 0.04, "보라=리커버리  주황=킥", transform=gax.transAxes, color="#e6d9ff",
             fontsize=fs - 1, alpha=alpha)


# ----------------------------------------------------------------- 메인 장면
SCENES = {
    "title":   (0.0, 5.0, 0.8),
    "s1":      (5.0, 21.0, 0.6),
    "s2":      (21.0, 41.0, 0.35),
    "s3":      (41.0, 62.0, 0.6),
    "summary": (62.0, 73.0, 0.8),
}
MAIN_END = 73.0
S3_END = SCENES["s3"][1]

CAPTIONS = [
    (5.0, 10.5, "위에서 내려다본 평영 발차기 (팔은 앞으로 뻗은 킥 드릴 자세)\n① 리커버리 : 무릎을 굽혀 뒤꿈치를 엉덩이 쪽으로 끌어온다"),
    (10.5, 16.0, "② 킥 : 발목을 당겨(플렉스) 발끝을 바깥으로 돌리고, 노란 발바닥으로\n물을 '바깥 → 뒤 → 안쪽'으로 반원을 그리듯 밀어낸다"),
    (16.0, 21.0, "③ 글라이드 : 두 다리를 모아 발끝을 쭉 펴고 미끄러진다\n노란 선 = 발바닥이 물속에 남긴 궤적 : 킥 때 발이 물을 '붙잡고' 뒤로 민다"),
    (21.0, 27.5, "파란 화살표 : 발바닥이 물을 미는 힘\n빨간 화살표 : 물이 발을 되미는 힘 (같은 크기·반대 방향, 뉴턴 제3법칙)"),
    (27.5, 34.0, "킥 : 뒤를 향한 발바닥이 물을 뒤로 밀어 → 몸은 앞으로 (초록 = 추진력)\n발이 안쪽으로 모일 때도 비스듬한 발바닥이 물을 뒤로 보내 추진력이 이어진다"),
    (34.0, 41.0, "그래프 : 킥에서 큰 추진력, 리커버리에서는 다리가 물을 앞으로 밀어 브레이크\n→ 리커버리는 작고 부드럽게, 킥은 빠르고 강하게, 끝나면 글라이드"),
    (41.0, 47.5, "점 하나하나가 물 입자.  밝을수록 발차기로 빨라진 물\n(카메라가 몸을 따라가므로 물은 오른쪽 → 왼쪽으로 흘러간다)"),
    (47.5, 55.0, "킥 때 각 발바닥이 물 덩어리를 뒤로 밀어내며, 발 양옆에 서로 반대로 도는 소용돌이 쌍이 생긴다\n(빨강·파랑 = 회전 방향, 입체로 보면 발마다 도넛 모양의 소용돌이 고리)"),
    (55.0, 62.0, "흰 화살표 = 물 자체의 움직임 : 소용돌이 쌍 사이로 물이 뒤로 뿜어진다 (두 갈래 제트)\n가운데는 리커버리 때 다리가 앞으로 끌고 온 물 (브레이크) → 작게 접어야 하는 이유"),
]

HEADERS = {
    "s1": "① 평영 발차기 동작 : 리커버리 → 킥 → 글라이드",
    "s2": "② 작용·반작용 : 발바닥으로 물을 뒤로 밀기",
    "s3": "③ 물의 흐름 : 뒤로 밀려나가는 물과 소용돌이 고리",
}

SUMMARY = [
    "정리",
    "1.  추진력의 근원은 발바닥으로 '물을 뒤로 보내는 것' — 작용·반작용",
    "2.  발목을 당겨(플렉스) 발끝을 바깥으로 → 발바닥이 뒤를 향해야 물이 잡힌다",
    "3.  바깥 → 뒤 → 안쪽으로 반원을 그리며 끝까지 모아야 추진이 이어진다",
    "4.  리커버리는 브레이크 — 작고 부드럽게 접고, 킥 뒤에는 글라이드",
]


def scene_at(T):
    for k, (a, b, _) in SCENES.items():
        if a <= T < b:
            return k
    return "summary"


def recycle(px, py, rng):
    out = (px < XL - 0.6) | (py > YT + 0.3) | (py < YB - 0.3)
    n = out.sum()
    if n:
        px[out] = rng.uniform(XR + 0.1, XR + 0.6, n)
        py[out] = rng.uniform(YB, YT, n)
    return px, py


def vorticity_rgba(wake, gx, gy, alpha):
    sel = wake.vx > XL - 0.5
    vx, vy, g = wake.vx[sel], wake.vy[sel], wake.g[sel]
    d2 = (wake.delta * 1.3) ** 2
    X = gx.ravel()[:, None]; Y = gy.ravel()[:, None]
    r2 = (X - vx[None, :]) ** 2 + (Y - vy[None, :]) ** 2
    om = (g[None, :] / (np.pi * d2) * np.exp(-r2 / d2)).sum(1).reshape(gx.shape)
    s = np.clip(om / 1.6, -1, 1)
    rgba = np.zeros(gx.shape + (4,))
    rgba[s > 0] = (1.0, 0.28, 0.28, 0)
    rgba[s <= 0] = (0.3, 0.6, 1.0, 0)
    rgba[..., 3] = np.abs(s) ** 0.8 * 0.85 * alpha
    return rgba


def render_main(path):
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.axis("off")
    pool_background(ax, XL, XR, YB, YT)

    gx, gy = np.meshgrid(np.linspace(XL, XR, 160), np.linspace(YB, YT, 90))
    vort_img = ax.imshow(np.zeros(gx.shape + (4,)), extent=(XL, XR, YB, YT),
                         origin="lower", aspect="auto", zorder=4, interpolation="bilinear")

    rng = np.random.default_rng(1)
    NP = 2400
    px = rng.uniform(XL - 0.5, XR + 0.5, NP)
    py = rng.uniform(YB, YT, NP)
    tracers = ax.scatter(px, py, s=5, c="white", zorder=5, lw=0)

    qx, qy = np.meshgrid(np.arange(-5.2, -0.9, 0.35), np.arange(-1.6, 1.65, 0.27))
    qx, qy = qx.ravel(), qy.ravel()
    quiv = ax.quiver(qx, qy, 0 * qx, 0 * qy, color="white", scale=6, width=0.0026,
                     headwidth=4, zorder=8, alpha=0)

    sw = Swimmer(ax)
    trails = [LineCollection([], linewidths=3, zorder=7) for _ in range(2)]
    for tr in trails:
        ax.add_collection(tr)

    arrows = [(arrow(ax, C_FORCE), arrow(ax, C_WATER)) for _ in range(2)]
    a_thrust = arrow(ax, C_THRUST, lw=5, ms=28)
    l_force = ax.text(0, 0, "물 → 발", color=C_FORCE, fontsize=15, weight="bold", zorder=31)
    l_water = ax.text(0, 0, "발 → 물", color=C_WATER, fontsize=15, weight="bold", zorder=31)
    l_thrust = ax.text(0, 0, "추진력", color=C_THRUST, fontsize=16, weight="bold", zorder=31)
    l_sole = ax.text(0, 0, "발바닥", color=SOLE, fontsize=14, weight="bold", zorder=31)

    phase_txt = fig.text(0.975, 0.875, "", fontsize=24, weight="bold", ha="right", va="center", zorder=40,
                         bbox=dict(boxstyle="round,pad=0.3", fc=(0.03, 0.1, 0.2, 0.85), ec="none"))
    header = fig.text(0.02, 0.945, "", fontsize=26, weight="bold", color="white", va="center", zorder=40,
                      bbox=dict(boxstyle="round,pad=0.3", fc=(0.03, 0.1, 0.2, 0.75), ec="none"))
    slowmo = fig.text(0.975, 0.945, "", fontsize=15, color="#bfe6ff", ha="right", va="center", zorder=40)
    legend1 = fig.text(0.02, 0.86, "● 반시계 회전", fontsize=15, color="#ff6b6b", zorder=40)
    legend2 = fig.text(0.14, 0.86, "● 시계 회전", fontsize=15, color="#6ea8ff", zorder=40)
    cap = caption_box(fig)
    inset = AnkleInset(fig, [0.865, 0.62, 0.11, 0.2])

    gax = fig.add_axes([0.62, 0.19, 0.355, 0.23])

    overlay = fig.add_axes([0, 0, 1, 1], zorder=60); overlay.axis("off")
    overlay.set_xlim(0, 1); overlay.set_ylim(0, 1)
    ov_rect = overlay.add_patch(Rectangle((0, 0), 1, 1, color=(0.02, 0.07, 0.15), alpha=0))
    title1 = overlay.text(0.5, 0.58, "평영 발차기의 추진 원리", fontsize=46, weight="bold",
                          color="white", ha="center", va="center")
    title2 = overlay.text(0.5, 0.45, "발바닥은 물을 어떻게 붙잡고, 물은 몸을 어떻게 앞으로 보내는가",
                          fontsize=22, color="#bfe6ff", ha="center", va="center")
    sum_txts = [overlay.text(0.10, 0.80 - i * 0.12, s, fontsize=34 if i == 0 else 22,
                             weight="bold" if i == 0 else "normal",
                             color="#ffd166" if i == 0 else "white", va="center")
                for i, s in enumerate(SUMMARY)]
    note = overlay.text(0.10, 0.12, "* 위에서 본 수평면만 다룬 단순화한 2차원 물리 모델(압력 항력 + 와류법)의 시각화이며, 힘의 크기는 상대값입니다.",
                        fontsize=13, color="#9fb3c8")

    # ---- 사전 시뮬레이션 (흔적이 이미 발달한 상태에서 시작)
    wake = m.Wake()
    pre_dt = 1 / FPS
    wake.t = 0.8 * TC - 3 * TC
    for _ in range(int(3 * TC / pre_dt)):
        u, v = wake.velocity(px, py)
        px += (u - m.U) * pre_dt; py += v * pre_dt
        wake.step(pre_dt)
        px, py = recycle(px, py, rng)
    hist_t, hist_f = [], []
    foot_pts = [[], []]

    nframes = int(MAIN_END * FPS)
    w = writer()
    with w.saving(fig, path, dpi=100):
        for fi in range(nframes):
            T = fi / FPS
            sc = scene_at(T)
            spd = SCENES[sc][2]
            dts = spd / FPS
            u, v = wake.velocity(px, py)
            px += (u - m.U) * dts; py += v * dts
            wake.step(dts)
            px, py = recycle(px, py, rng)
            t = wake.t
            sw.update(t)
            lf, fx_tot = leg_forces(t)
            hist_t.append(t); hist_f.append(fx_tot)
            for k, sd in enumerate((+1, -1)):
                Hh, K, A, Tt, fl = m.leg(t, sd)
                c = 0.5 * (A + Tt)
                foot_pts[k] = [(x - m.U * dts, y, f) for x, y, f in foot_pts[k]] + [(c[0], c[1], fl)]
                foot_pts[k] = [p for p in foot_pts[k] if p[0] > XL - 0.5]

            # --- 물 입자
            u, v = wake.velocity(px, py)
            sp = np.hypot(u, v)
            if sc == "s3":
                a_tr = 0.25 + 0.75 * smooth((T - 41.0) / 1.0)
            elif sc == "summary":
                a_tr = 1.0 - 0.65 * smooth((T - 62.0) / 1.0)
            else:
                a_tr = 0.3
            cval = np.clip(sp / 0.6, 0, 1)
            cols = np.empty((len(px), 4))
            cols[:, 0] = 0.35 + 0.65 * cval; cols[:, 1] = 0.6 + 0.4 * cval; cols[:, 2] = 1.0
            cols[:, 3] = a_tr * (0.35 + 0.65 * cval)
            tracers.set_offsets(np.c_[px, py]); tracers.set_facecolors(cols)
            tracers.set_sizes(4 + 14 * cval)

            # --- 소용돌이 / 제트 (s3)
            a_v = fade(T, 47.0, 62.6, 1.0)
            if a_v > 0.01:
                vort_img.set_data(vorticity_rgba(wake, gx, gy, a_v)); vort_img.set_visible(True)
            else:
                vort_img.set_visible(False)
            legend1.set_alpha(a_v); legend2.set_alpha(a_v)
            a_q = fade(T, 54.5, 62.6, 1.0)
            if a_q > 0.01:
                uq, vq = wake.velocity(qx, qy, delta=0.3)
                quiv.set_UVC(uq, vq); quiv.set_alpha(a_q)
            else:
                quiv.set_alpha(0)

            # --- 발바닥 궤적 (s1 후반)
            a_trail = fade(T, 15.5, 21.3, 0.8)
            for k in range(2):
                pts = np.array(foot_pts[k])
                if a_trail > 0.01 and len(pts) > 1:
                    segs = np.stack([pts[:-1, :2], pts[1:, :2]], 1)
                    al = np.linspace(0, 1, len(segs)) * a_trail * (0.25 + 0.75 * pts[1:, 2])
                    trails[k].set_segments(segs)
                    trails[k].set_color([(1, 0.82, 0.4, a) for a in al])
                else:
                    trails[k].set_segments([])

            # --- 힘 화살표 (s2)
            a_f = fade(T, 21.3, 41.0, 0.8)
            vis = a_f > 0.01
            sc_f = 0.13
            for (af, aw), (Ff, c) in zip(arrows, lf):
                af.set_visible(vis); aw.set_visible(vis); af.set_alpha(a_f); aw.set_alpha(a_f)
                if vis:
                    af.set_positions(tuple(c), tuple(c + Ff * sc_f))
                    aw.set_positions(tuple(c), tuple(c - Ff * sc_f))
            Ff0, c0 = lf[0]
            l_force.set_position(tuple(c0 + Ff0 * sc_f + [0.05, 0.08]))
            l_water.set_position(tuple(c0 - Ff0 * sc_f + [-0.75, 0.1]))
            fxa = fx_tot * 0.11
            base = np.array([0.55, 0.0])
            a_thrust.set_positions(tuple(base), tuple(base + [fxa, 0]))
            a_thrust.set_color(C_THRUST if fx_tot >= 0 else C_FORCE)
            show_t = vis and abs(fxa) > 0.04
            a_thrust.set_visible(show_t); a_thrust.set_alpha(a_f)
            l_thrust.set_text("추진력" if fx_tot >= 0 else "브레이크")
            l_thrust.set_color(C_THRUST if fx_tot >= 0 else C_FORCE)
            l_thrust.set_position((base[0] + max(fxa, 0) + 0.1, base[1] + 0.25))
            l_thrust.set_visible(show_t); l_thrust.set_alpha(a_f)
            for lab in (l_force, l_water):
                lab.set_visible(vis); lab.set_alpha(a_f)

            # --- 발바닥 라벨 (s1 킥 장면)
            a_sole = fade(T, 10.5, 16.0, 0.5)
            Hh, K, A, Tt, fl = m.leg(t, +1)
            l_sole.set_position((Tt[0] + 0.12, Tt[1] + 0.12))
            l_sole.set_alpha(a_sole * smooth(fl * 1.5)); l_sole.set_visible(a_sole > 0.01)

            # --- 그래프 (s2)
            a_g = fade(T, 21.3, 41.0, 0.8)
            gax.clear()
            if a_g > 0.01:
                draw_thrust_graph(gax, hist_t, hist_f, a_g)
                gax.set_visible(True)
            else:
                gax.set_visible(False)

            # --- 텍스트
            name, col = m.phase_name(t * m.FREQ)
            a_ph = fade(T, 5.3, 62.0, 0.5)
            phase_txt.set_text(name); phase_txt.set_color(col)
            phase_txt.set_alpha(a_ph); phase_txt.get_bbox_patch().set_alpha(0.85 * a_ph)
            inset.update(m.leg(t, +1)[4], a_ph)
            if sc in HEADERS:
                a0, a1, _ = SCENES[sc]
                al = fade(T, a0, a1, 0.5)
                header.set_text(HEADERS[sc]); header.set_alpha(al); header.get_bbox_patch().set_alpha(0.75 * al)
                slowmo.set_text(f"슬로모션 ×{spd:g}"); slowmo.set_alpha(al)
            else:
                header.set_alpha(0); header.get_bbox_patch().set_alpha(0); slowmo.set_alpha(0)
            cap.set_visible(False)
            for c0_, c1_, txt in CAPTIONS:
                if c0_ <= T < c1_:
                    al = fade(T, c0_, c1_, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.6 * al)
                    cap.set_visible(True)

            # --- 오버레이
            a_title = 1 - smooth((T - 3.6) / 1.2)
            a_sum = smooth((T - 62.3) / 1.0)
            ov_rect.set_alpha(0.75 * max(a_title, a_sum))
            title1.set_alpha(a_title); title2.set_alpha(a_title)
            for i, s in enumerate(sum_txts):
                s.set_alpha(smooth((T - 62.6 - 0.9 * i) / 0.6))
            note.set_alpha(smooth((T - 67.0) / 0.8))

            w.grab_frame()
            if fi % 150 == 0:
                print(f"main {T:5.1f}s / {MAIN_END}s", flush=True)
    plt.close(fig)


# ----------------------------------------------------------------- 발목 비교 장면
COMP_LEN = 16.0
COMP_SPEED = 0.35


def render_compare(path):
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("#06182e")
    fig.text(0.025, 0.945, "④ 발목을 당기느냐(플렉스) vs 펴느냐(포인트)", fontsize=26, weight="bold",
             color="white", va="center")
    fig.text(0.975, 0.945, f"슬로모션 ×{COMP_SPEED:g}", fontsize=15, color="#bfe6ff", ha="right", va="center")
    rows = [("flex", "발목 플렉스 : 발바닥이 뒤를 향한다", 0.50),
            ("pointed", "발목 포인트 : 발끝이 늘 뒤를 향한다", 0.135)]
    xl, xr, yb, yt = -2.9, 1.2, -1.15, 1.15
    panels = []
    for ankle, label, y0 in rows:
        ax = fig.add_axes([0.01, y0, 0.52, 0.355])
        ax.set_xlim(xl, xr); ax.set_ylim(yb, yt); ax.axis("off")
        pool_background(ax, xl, xr, yb, yt)
        sw = Swimmer(ax)
        arr = [(arrow(ax, C_FORCE, ms=18, lw=3), arrow(ax, C_WATER, ms=18, lw=3)) for _ in range(2)]
        at = arrow(ax, C_THRUST, ms=22, lw=4.5)
        ax.text(xl + 0.08, yt - 0.12, label, fontsize=16, weight="bold", color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.5), ec="none"), zorder=40)
        gax = fig.add_axes([0.55, y0 + 0.02, 0.30, 0.32])
        bax = fig.add_axes([0.875, y0 + 0.02, 0.095, 0.32])
        panels.append(dict(ankle=ankle, sw=sw, arr=arr, at=at, gax=gax, bax=bax, ht=[], hf=[]))
    cap = caption_box(fig)
    cap.set_position((0.03, 0.012)); cap.set_fontsize(16)
    caps = [
        (0.0, 8.0, "위 : 플렉스한 발바닥(노랑)이 물을 뒤로 밀어 큰 추진력\n아래 : 포인트한 발은 물을 '썰고' 지나가 추진력이 작고, 리커버리 브레이크만 남는다"),
        (8.0, COMP_LEN, "오른쪽 막대 = 한 주기 평균 추진력 (상대값)\n평영 킥은 발목을 당겨 발바닥으로 물을 잡는 것이 핵심 (자유형과 반대!)"),
    ]
    ts = np.linspace(0, TC, 600, endpoint=False)
    means = {p["ankle"]: np.mean([m.total_fx(tt, p["ankle"]) for tt in ts]) for p in panels}
    t = 0.05 * TC
    step = COMP_SPEED / FPS
    for p in panels:
        for k in range(int(1.6 * TC / step)):
            tt = t - 1.6 * TC + k * step
            p["ht"].append(tt); p["hf"].append(m.total_fx(tt, p["ankle"]))

    w = writer()
    with w.saving(fig, path, dpi=100):
        for fi in range(int(COMP_LEN * FPS)):
            T = fi / FPS
            t += step
            for p in panels:
                p["sw"].update(t, p["ankle"])
                lf, fx = leg_forces(t, p["ankle"])
                for (af, aw), (Ff, c) in zip(p["arr"], lf):
                    af.set_positions(tuple(c), tuple(c + 0.1 * Ff))
                    aw.set_positions(tuple(c), tuple(c - 0.1 * Ff))
                base = np.array([0.5, 0.0])
                p["at"].set_positions(tuple(base), tuple(base + [0.08 * fx, 0]))
                p["at"].set_color(C_THRUST if fx >= 0 else C_FORCE)
                p["at"].set_visible(abs(fx) > 0.3)
                p["ht"].append(t); p["hf"].append(fx)
                p["gax"].clear()
                draw_thrust_graph(p["gax"], p["ht"], p["hf"], 1.0, title="추진력", small=True)
                b = p["bax"]; b.clear()
                mv = means[p["ankle"]]
                b.set_facecolor((0, 0, 0, 0.55))
                b.bar([0], [mv], color=C_THRUST if mv > 0 else C_FORCE, width=0.6)
                b.axhline(0, color="white", lw=1)
                b.set_xlim(-0.6, 0.6); b.set_ylim(-0.9, 0.9)
                b.set_xticks([]); b.set_yticks([])
                for sp in b.spines.values():
                    sp.set_color((1, 1, 1, 0.4))
                b.text(0, 0.84, "평균", color="white", ha="center", va="top", fontsize=12)
                b.text(0, mv + (0.08 if mv > 0 else -0.08), f"{mv:+.2f}", color="white",
                       ha="center", va="bottom" if mv > 0 else "top", fontsize=13, weight="bold")
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
    out = os.path.join(OUT, "breaststroke_kick_propulsion.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print("done:", out)


if __name__ == "__main__":
    main()
