"""자유형 발차기 추진 원리 설명 영상 렌더러.

python3 render.py  ->  freestyle_kick_propulsion.mp4
"""
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
XL, XR = -6.0, 4.3
YT = 1.95
YB = YT - (XR - XL) * 9 / 16
SY = m.SURFACE_Y
OUT = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(OUT, "_build")
os.makedirs(TMP, exist_ok=True)

SKIN = "#f2c9a0"
SUIT = "#1b2a49"
C_FORCE = "#ff5a4e"     # 물이 발을 미는 힘
C_WATER = "#4fc3ff"     # 발이 물을 미는 힘
C_THRUST = "#5dff8a"    # 추진력 (앞 성분)
C_DOWN = "#ffb347"
C_UP = "#9be7ff"


# ----------------------------------------------------------------- 공통 도우미
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, ramp=0.6):
    """[t0, t1] 구간에서 1, 양 끝은 ramp 초 동안 페이드."""
    return smooth((t - t0) / ramp) * smooth((t1 - t) / ramp)


def background(ax, surface=True):
    sky = np.linspace(0, 1, 64)[:, None]
    ax.imshow(np.dstack([0.75 + 0.15 * sky, 0.86 + 0.1 * sky, 0.97 + 0 * sky]),
              extent=(XL, XR, SY, YT), origin="lower", aspect="auto", zorder=0)
    deep = np.linspace(0, 1, 128)[:, None]
    ax.imshow(np.dstack([0.02 + 0.06 * deep, 0.12 + 0.22 * deep, 0.25 + 0.30 * deep]),
              extent=(XL, XR, YB, SY), origin="lower", aspect="auto", zorder=0)


class Swimmer:
    """몸통/머리(고정) + 다리(매 프레임 갱신)."""

    def __init__(self, ax, hip=(0.0, 0.0), z=10):
        hx, hy = hip
        self.hip = hip
        xs = np.linspace(-0.15, 3.3, 40)
        top = hy + 0.30 + 0.06 * np.sin(np.clip((xs - 0.2) / 3.1, 0, 1) * np.pi)
        bot = hy - 0.28 - 0.08 * np.sin(np.clip((xs - 0.4) / 2.6, 0, 1) * np.pi)
        ax.fill_between(xs + hx, bot, top, color=SKIN, zorder=z, lw=0)
        ax.fill_between(xs[xs < 1.0] + hx, bot[xs < 1.0], top[xs < 1.0], color=SUIT, zorder=z + 0.1, lw=0)
        ax.add_patch(Ellipse((hx + 3.75, hy + 0.08), 0.85, 0.62, color=SKIN, zorder=z))
        ax.add_patch(Ellipse((hx + 3.85, hy + 0.2), 0.8, 0.45, color="#d62828", zorder=z + 0.2))  # 수모
        # 팔 (앞으로 뻗은 팔)
        ax.plot([hx + 3.2, hx + 4.8], [hy - 0.1, hy - 0.35], color=SKIN, lw=11,
                solid_capstyle="round", zorder=z - 0.1)
        self.thigh, = ax.plot([], [], color=SUIT, lw=20, solid_capstyle="round", zorder=z + 0.3)
        self.shank, = ax.plot([], [], color=SKIN, lw=15, solid_capstyle="round", zorder=z + 0.2)
        self.foot, = ax.plot([], [], color="#d99a6c", lw=10, solid_capstyle="round", zorder=z + 0.2)
        self.jdots = ax.scatter([], [], s=28, c="#7a4a2a", zorder=z + 0.5)

    def update(self, P):
        Hh, K, A, T = P
        self.thigh.set_data([Hh[0], K[0]], [Hh[1], K[1]])
        self.shank.set_data([K[0], A[0]], [K[1], A[1]])
        self.foot.set_data([A[0], T[0]], [A[1], T[1]])
        self.jdots.set_offsets(np.array([K, A]))


def arrow(ax, color, z=20, lw=3.5, ms=22):
    a = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=ms,
                        color=color, lw=lw, zorder=z)
    ax.add_patch(a)
    return a


def toe_down(t, ankle="flexible", h=0.08):
    return m.joints(t + h, ankle)[3][1] < m.joints(t - h, ankle)[3][1]


def total_fx(t, ankle="flexible"):
    return m.segment_forces(t, ankle)[0][:, 0].sum()


def caption_box(fig):
    return fig.text(0.03, 0.045, "", fontsize=19, color="white", va="bottom", ha="left",
                    linespacing=1.45, zorder=50,
                    bbox=dict(boxstyle="round,pad=0.6", fc=(0, 0, 0, 0.55), ec="none"))


class _PreviewWriter:
    """PREVIEW="t1,t2,..." 환경변수가 있으면 해당 시각 프레임만 PNG로 저장."""

    def __init__(self, times):
        self.frames = {int(round(float(x) * FPS)) for x in times}
        self.i = 0

    def saving(self, fig, path, dpi):
        self.fig, self.path = fig, path
        import contextlib
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


# ----------------------------------------------------------------- 메인 장면
# (시작, 끝, 시뮬레이션 배속)
SCENES = {
    "title":   (0.0, 5.0, 1.0),
    "s1":      (5.0, 18.0, 0.8),
    "s2":      (18.0, 36.0, 0.45),
    "s3":      (36.0, 56.0, 0.8),
    "summary": (56.0, 66.0, 1.0),
}
MAIN_END = 66.0

CAPTIONS = [
    ("s1", 5.0, 10.0, "고관절(허벅지)에서 시작된 움직임이 무릎 → 발목 → 발끝으로\n차례로 전달되며 점점 빨라진다  (채찍 효과)"),
    ("s1", 10.0, 14.5, "다운킥 ▼ : 무릎이 살짝 굽었다가 '탁' 펴지며 발등으로 물을 누른다\n업킥 ▲ : 다리를 곧게 편 채 부드럽게 들어 올린다"),
    ("s1", 14.5, 18.0, "노란 선 = 발끝이 물속에 남긴 궤적\n몸이 앞으로 나아가므로, 물 입장에서 발은 '위아래 + 뒤로' 움직인다"),
    ("s2", 18.0, 24.0, "파란 화살표 : 발이 물을 미는 힘\n빨간 화살표 : 물이 발을 되미는 힘  (같은 크기·반대 방향, 뉴턴 제3법칙)"),
    ("s2", 24.0, 30.0, "발끝을 펴서 발등이 비스듬히 '뒤'를 향하므로, 되미는 힘이 앞쪽으로 기운다\n그 앞쪽 성분(초록 화살표)이 바로 추진력"),
    ("s2", 30.0, 36.0, "그래프 : 추진력은 다운킥 때 크게 나오고, 업킥 때는 작다\n→ 빠르고 강한 다운킥 + 유연하게 펴진 발목이 핵심"),
    ("s3", 36.0, 42.0, "점 하나하나가 물 입자.  밝게 빛날수록 발차기로 빨라진 물\n(카메라가 몸을 따라가므로 물은 오른쪽 → 왼쪽으로 흘러간다)"),
    ("s3", 42.0, 49.0, "발끝이 방향을 바꿀 때마다 소용돌이가 떨어져 나간다\n빨강 = 반시계 회전,  파랑 = 시계 회전  →  위아래로 엇갈린 두 줄"),
    ("s3", 49.0, 56.0, "흰 화살표 = 물 자체의 움직임 : 두 줄 소용돌이 사이로 물이 뒤로 뿜어진다 (제트)\n물에 '뒤쪽' 운동량을 준 만큼, 몸은 '앞쪽' 운동량(추진력)을 얻는다"),
]

HEADERS = {
    "s1": "① 발차기 동작 : 채찍처럼 휘두르기",
    "s2": "② 작용·반작용 : 물을 뒤로 밀면 몸은 앞으로",
    "s3": "③ 물의 흐름 : 소용돌이와 뒤로 뿜어지는 물줄기",
}

SUMMARY = [
    "정리",
    "1.  추진력의 근원은 '물을 뒤로 보내는 것' — 작용·반작용",
    "2.  허벅지 → 무릎 → 발끝으로 속도를 키우는 채찍 동작, 핵심은 다운킥",
    "3.  발끝을 펴야(발목 유연성) 물이 되미는 힘이 앞쪽으로 기운다",
    "4.  떨어져 나간 소용돌이 사이로 뒤로 향하는 물줄기(제트)가 생긴다",
]


def scene_at(T):
    for k, (a, b, _) in SCENES.items():
        if a <= T < b:
            return k
    return "summary"


def render_main(path):
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT); ax.axis("off")
    background(ax)
    sx = np.linspace(XL, XR, 300)
    surf, = ax.plot(sx, SY + 0 * sx, color="white", lw=2, alpha=0.8, zorder=3)

    # 와도(소용돌이) 필드
    gx, gy = np.meshgrid(np.linspace(XL, XR, 170), np.linspace(YB, SY, 100))
    vort_img = ax.imshow(np.zeros(gx.shape + (4,)), extent=(XL, XR, YB, SY),
                         origin="lower", aspect="auto", zorder=4, interpolation="bilinear")

    # 물 입자
    rng = np.random.default_rng(1)
    NP = 2200
    px = rng.uniform(XL - 0.5, XR + 0.5, NP)
    py = rng.uniform(YB, SY - 0.05, NP)
    tracers = ax.scatter(px, py, s=5, c="white", zorder=5, lw=0)

    # 제트 화살표 격자
    qx, qy = np.meshgrid(np.arange(-5.8, -2.4, 0.4), np.arange(-1.4, 0.5, 0.3))
    qx, qy = qx.ravel(), qy.ravel()
    quiv = ax.quiver(qx, qy, 0 * qx, 0 * qy, color="white", scale=7, width=0.0028,
                     headwidth=4, zorder=8, alpha=0)

    sw = Swimmer(ax)
    trail = LineCollection([], linewidths=3, zorder=7)
    ax.add_collection(trail)
    jlabels = [ax.text(0, 0, s, fontsize=14, color="white", zorder=30, ha="center",
                       bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.5), ec="none"))
               for s in ("고관절", "무릎", "발목", "발끝")]

    a_force = arrow(ax, C_FORCE)
    a_water = arrow(ax, C_WATER)
    a_thrust = arrow(ax, C_THRUST, lw=4.5, ms=26)
    l_force = ax.text(0, 0, "물 → 발", color=C_FORCE, fontsize=15, weight="bold", zorder=31)
    l_water = ax.text(0, 0, "발 → 물", color=C_WATER, fontsize=15, weight="bold", zorder=31)
    l_thrust = ax.text(0, 0, "추진력", color=C_THRUST, fontsize=16, weight="bold", zorder=31)

    phase_txt = fig.text(0.975, 0.855, "", fontsize=24, weight="bold", ha="right", va="center", zorder=40,
                         bbox=dict(boxstyle="round,pad=0.3", fc=(0.03, 0.1, 0.2, 0.8), ec="none"))
    header = fig.text(0.025, 0.93, "", fontsize=27, weight="bold", color="#0d2240", va="center", zorder=40)
    slowmo = fig.text(0.975, 0.93, "", fontsize=15, color="#0d2240", ha="right", va="center", zorder=40)
    cap = caption_box(fig)

    gax = fig.add_axes([0.60, 0.19, 0.37, 0.22])
    legend_txt = fig.text(0.025, 0.775, "", fontsize=15, color="#ff6b6b", ha="left", zorder=40,
                          bbox=dict(boxstyle="round,pad=0.35", fc=(0, 0, 0, 0.45), ec="none"))
    legend_txt2 = fig.text(0.20, 0.775, "", fontsize=15, color="#6ea8ff", ha="left", zorder=41)

    overlay = fig.add_axes([0, 0, 1, 1], zorder=60); overlay.axis("off")
    overlay.set_xlim(0, 1); overlay.set_ylim(0, 1)
    ov_rect = overlay.add_patch(Rectangle((0, 0), 1, 1, color=(0.02, 0.07, 0.15), alpha=0))
    title1 = overlay.text(0.5, 0.58, "자유형 발차기의 추진 원리", fontsize=46, weight="bold",
                          color="white", ha="center", va="center")
    title2 = overlay.text(0.5, 0.45, "발은 물을 어떻게 밀고, 물은 몸을 어떻게 앞으로 보내는가",
                          fontsize=22, color="#bfe6ff", ha="center", va="center")
    sum_txts = [overlay.text(0.12, 0.78 - i * 0.12, s, fontsize=34 if i == 0 else 22,
                             weight="bold" if i == 0 else "normal",
                             color="#ffd166" if i == 0 else "white", va="center")
                for i, s in enumerate(SUMMARY)]
    note = overlay.text(0.12, 0.12, "* 단순화한 2차원 물리 모델(압력 항력 + 와류법)의 시각화이며, 힘의 크기는 상대값입니다.",
                        fontsize=13, color="#9fb3c8")

    # ---- 사전 시뮬레이션 (흔적이 이미 발달한 상태에서 시작)
    wake = m.Wake()
    for _ in range(12 * FPS):
        u, v = wake.velocity(px, py)
        px += (u - m.U) / FPS; py += v / FPS
        wake.step(1 / FPS)
        px, py = recycle(px, py, rng)
    hist_t, hist_f, hist_down = [], [], []
    toe_pts = []

    nframes = int(MAIN_END * FPS)
    w = writer()
    with w.saving(fig, path, dpi=100):
        for fi in range(nframes):
            T = fi / FPS
            sc = scene_at(T)
            spd = SCENES[sc][2]
            dts = spd / FPS
            # --- 물리 업데이트
            u, v = wake.velocity(px, py)
            px += (u - m.U) * dts; py += v * dts
            wake.step(dts)
            px, py = recycle(px, py, rng)
            t = wake.t
            P = m.joints(t)
            sw.update(P)
            F, C = m.segment_forces(t)
            down = toe_down(t)
            hist_t.append(t); hist_f.append(F[:, 0].sum()); hist_down.append(down)
            toe_pts = [(x - m.U * dts, y) for x, y in toe_pts] + [tuple(P[3])]
            toe_pts = [p for p in toe_pts if p[0] > XL - 0.5]

            # --- 물 입자 표현
            u, v = wake.velocity(px, py)
            sp = np.hypot(u, v)
            a_tr = {"title": 0.35, "s1": 0.3, "s2": 0.25, "s3": 1.0, "summary": 0.35}[sc]
            if sc == "s3":
                a_tr = 0.25 + 0.75 * smooth((T - 36.0) / 1.0)
            elif sc == "summary":
                a_tr = 1.0 - 0.65 * smooth((T - 56.0) / 1.0)
            cval = np.clip(sp / 0.8, 0, 1)
            cols = np.empty((len(px), 4))
            cols[:, 0] = 0.35 + 0.65 * cval; cols[:, 1] = 0.6 + 0.4 * cval; cols[:, 2] = 1.0
            cols[:, 3] = a_tr * (0.35 + 0.65 * cval)
            tracers.set_offsets(np.c_[px, py]); tracers.set_facecolors(cols)
            tracers.set_sizes(4 + 14 * cval)

            # --- 소용돌이 필드 (s3)
            a_v = fade(T, 41.0, 56.6, 1.0) if sc in ("s3", "summary") else 0.0
            if a_v > 0.01:
                vort_img.set_data(vorticity_rgba(wake, gx, gy, a_v))
                vort_img.set_visible(True)
            else:
                vort_img.set_visible(False)
            a_q = fade(T, 48.5, 56.6, 1.0)
            if a_q > 0.01:
                uq, vq = wake.velocity(qx, qy, delta=0.5)
                quiv.set_UVC(uq, vq); quiv.set_alpha(a_q)
            else:
                quiv.set_alpha(0)
            legend_txt.set_text("● 반시계 회전          " if a_v > 0.01 else "")
            legend_txt2.set_text("● 시계 회전" if a_v > 0.01 else "")
            legend_txt.set_alpha(a_v); legend_txt2.set_alpha(a_v)
            legend_txt.get_bbox_patch().set_alpha(0.45 * a_v)

            # --- 발끝 궤적 (s1)
            a_trail = fade(T, 14.0, 18.3, 0.8)
            if a_trail > 0.01 and len(toe_pts) > 1:
                pts = np.array(toe_pts)
                segs = np.stack([pts[:-1], pts[1:]], 1)
                al = np.linspace(0, 1, len(segs)) * a_trail
                trail.set_segments(segs)
                trail.set_color([(1, 0.85, 0.2, a) for a in al])
            else:
                trail.set_segments([])

            # --- 관절 라벨 (s1 초반)
            a_j = fade(T, 5.5, 10.5, 0.6)
            offs = [(0.0, -0.75), (0.05, -0.62), (-0.1, -0.62), (-0.55, 0.25)]
            for lab, p, o in zip(jlabels, P, offs):
                lab.set_position((p[0] + o[0], p[1] + o[1])); lab.set_alpha(a_j)
                lab.get_bbox_patch().set_alpha(0.5 * a_j)
                lab.set_visible(a_j > 0.01)

            # --- 힘 화살표 (s2)
            a_f = fade(T, 18.3, 36.0, 0.8)
            Ff = F[2] + F[1] * 0.5            # 발 + 정강이 아래쪽
            c = C[2]
            sc_f = 0.2
            vis = a_f > 0.01
            for art in (a_force, a_water, a_thrust, l_force, l_water, l_thrust):
                art.set_visible(vis); art.set_alpha(a_f)
            if vis:
                tip = c + Ff * sc_f
                a_force.set_positions(tuple(c), tuple(tip))
                a_water.set_positions(tuple(c), tuple(c - Ff * sc_f))
                fx = max(F[:, 0].sum(), 0) * 0.65
                base = np.array([0.3, -0.85])
                a_thrust.set_positions(tuple(base), tuple(base + [fx, 0]))
                a_thrust.set_visible(vis and fx > 0.03)
                l_thrust.set_visible(vis and fx > 0.03)
                l_force.set_position(tuple(tip + [0.08, 0.05]))
                l_water.set_position(tuple(c - Ff * sc_f + [-0.75, -0.2]))
                l_thrust.set_position((base[0] + fx + 0.1, base[1] - 0.1))

            # --- 추진력 그래프 (s2)
            a_g = fade(T, 18.3, 36.0, 0.8)
            gax.clear()
            if a_g > 0.01:
                draw_thrust_graph(gax, hist_t, hist_f, hist_down, a_g)
                gax.set_visible(True)
            else:
                gax.set_visible(False)

            # --- 텍스트
            phase_txt.set_text("▼ 다운킥" if down else "▲ 업킥")
            phase_txt.set_color(C_DOWN if down else C_UP)
            a_ph = fade(T, 9.5, 56.0, 0.5)
            phase_txt.set_alpha(a_ph); phase_txt.get_bbox_patch().set_alpha(0.8 * a_ph)
            if sc in HEADERS:
                a0, a1, _ = SCENES[sc]
                header.set_text(HEADERS[sc]); header.set_alpha(fade(T, a0, a1, 0.5))
                slowmo.set_text(f"슬로모션 ×{spd:g}"); slowmo.set_alpha(fade(T, a0, a1, 0.5))
            else:
                header.set_alpha(0); slowmo.set_alpha(0)
            cap.set_visible(False)
            for s, c0, c1, txt in CAPTIONS:
                if c0 <= T < c1:
                    al = fade(T, c0, c1, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.55 * al)
                    cap.set_visible(True)

            # --- 오버레이 (타이틀/정리)
            a_title = 1 - smooth((T - 3.6) / 1.2)
            a_sum = smooth((T - 56.3) / 1.0)
            ov_rect.set_alpha(0.72 * max(a_title, a_sum))
            title1.set_alpha(a_title); title2.set_alpha(a_title)
            for i, s in enumerate(sum_txts):
                s.set_alpha(smooth((T - 56.6 - 0.9 * i) / 0.6))
            note.set_alpha(smooth((T - 61.0) / 0.8))

            w.grab_frame()
            if fi % 150 == 0:
                print(f"main {T:5.1f}s / {MAIN_END}s", flush=True)
    plt.close(fig)


def recycle(px, py, rng):
    out = (px < XL - 0.6) | (py > SY - 0.02) | (py < YB - 0.2)
    n = out.sum()
    if n:
        px[out] = rng.uniform(XR + 0.1, XR + 0.6, n)
        py[out] = rng.uniform(YB, SY - 0.05, n)
    return px, py


def vorticity_rgba(wake, gx, gy, alpha):
    d2 = wake.delta ** 2
    X = gx.ravel()[:, None]; Y = gy.ravel()[:, None]
    r2 = (X - wake.vx[None, :]) ** 2 + (Y - wake.vy[None, :]) ** 2
    om = (wake.g[None, :] / (np.pi * d2) * np.exp(-r2 / d2)).sum(1).reshape(gx.shape)
    s = np.clip(om / 2.5, -1, 1)
    rgba = np.zeros(gx.shape + (4,))
    pos = s > 0
    rgba[pos] = (1.0, 0.25, 0.25, 0)
    rgba[~pos] = (0.25, 0.55, 1.0, 0)
    rgba[..., 3] = np.abs(s) ** 0.8 * 0.85 * alpha
    return rgba


def draw_thrust_graph(gax, ht, hf, hd, alpha, window=2.0 / m.FREQ, title="추진력 (앞쪽 성분, 상대값)",
                      ylim=(-2.4, 2.6)):
    t = np.array(ht); f = np.array(hf); d = np.array(hd)
    sel = t > t[-1] - window
    t, f, d = t[sel], f[sel], d[sel]
    gax.set_facecolor((0, 0, 0, 0.55 * alpha))
    for sp in gax.spines.values():
        sp.set_color((1, 1, 1, 0.4 * alpha))
    x = t - t[-1]
    # 다운킥 구간 음영
    gax.fill_between(x, ylim[0], ylim[1], where=d, color=C_DOWN, alpha=0.16 * alpha, lw=0, step="mid")
    gax.axhline(0, color=(1, 1, 1, 0.5 * alpha), lw=1)
    gax.fill_between(x, 0, f, where=f > 0, color=C_THRUST, alpha=0.5 * alpha, lw=0)
    gax.fill_between(x, 0, f, where=f < 0, color=C_FORCE, alpha=0.5 * alpha, lw=0)
    gax.plot(x, f, color="white", lw=2, alpha=alpha)
    gax.plot([x[-1]], [f[-1]], "o", color="white", ms=7, alpha=alpha)
    gax.set_xlim(-window, 0.05 * window); gax.set_ylim(*ylim)
    gax.set_xticks([]); gax.set_yticks([])
    gax.text(0.02, 0.92, title, transform=gax.transAxes, color="white",
             fontsize=13, va="top", alpha=alpha)
    gax.text(0.02, 0.08, "주황 배경 = 다운킥", transform=gax.transAxes, color=C_DOWN,
             fontsize=12, alpha=alpha)
    gax.text(0.98, 0.92, "+ 앞으로", transform=gax.transAxes, color=C_THRUST, fontsize=12,
             ha="right", va="top", alpha=alpha)
    gax.text(0.98, 0.08, "- 브레이크", transform=gax.transAxes, color=C_FORCE, fontsize=12,
             ha="right", alpha=alpha)


# ----------------------------------------------------------------- 발목 비교 장면
COMP_LEN = 15.0
COMP_SPEED = 0.45


def render_compare(path):
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("#06182e")
    header = fig.text(0.025, 0.945, "④ 발목 유연성의 차이", fontsize=27, weight="bold",
                      color="white", va="center")
    fig.text(0.975, 0.945, f"슬로모션 ×{COMP_SPEED:g}", fontsize=15, color="#bfe6ff", ha="right", va="center")
    rows = [("flexible", "유연한 발목 : 발끝을 쭉 편다", 0.50),
            ("stiff", "뻣뻣한 발목 : 발이 아래로 꺾여 있다", 0.13)]
    panels = []
    for ankle, label, y0 in rows:
        ax = fig.add_axes([0.0, y0, 0.58, 0.36])
        ax.set_xlim(-3.0, 2.4); ax.set_ylim(-1.75, 1.45); ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((-3.0, -1.75), 5.4, 1.75 + SY, color="#123a63", zorder=0))
        ax.add_patch(Rectangle((-3.0, SY), 5.4, 1.0, color="#bcd9f2", zorder=0))
        ax.plot([-3.0, 2.4], [SY, SY], color="white", lw=1.5, zorder=1)
        sw = Swimmer(ax)
        af = arrow(ax, C_FORCE, ms=18, lw=3)
        aw = arrow(ax, C_WATER, ms=18, lw=3)
        at = arrow(ax, C_THRUST, ms=20, lw=4)
        ax.text(-2.9, 1.25, label, fontsize=17, weight="bold",
                color="#0d2240", va="center")
        gax = fig.add_axes([0.60, y0 + 0.03, 0.25, 0.30])
        bax = fig.add_axes([0.88, y0 + 0.03, 0.08, 0.30])
        panels.append(dict(ankle=ankle, sw=sw, af=af, aw=aw, at=at, gax=gax, bax=bax,
                           ht=[], hf=[], hd=[]))
    cap = caption_box(fig)
    cap.set_position((0.03, 0.015))
    cap.set_fontsize(17)
    caps = [
        (0.0, 7.5, "위 : 발등이 뒤를 향해 물을 뒤로 민다 → 되미는 힘이 앞으로 기울어 추진력(초록)\n아래 : 꺾인 발이 물을 앞쪽으로 떠밀어, 되미는 힘이 뒤로 기운다 → 브레이크"),
        (7.5, COMP_LEN, "오른쪽 막대 = 한 주기 평균 추진력 (상대값)\n발목이 유연할수록 같은 발차기로 더 많은 추진력을 얻는다"),
    ]
    means = {}
    ts = np.linspace(0, 1 / m.FREQ, 400, endpoint=False)
    for p in panels:
        means[p["ankle"]] = np.mean([total_fx(t, p["ankle"]) for t in ts])

    t = 6.0
    for p in panels:   # 그래프 기록 미리 채우기
        for k in range(int(2 / m.FREQ / (COMP_SPEED / FPS))):
            tt = t - 2 / m.FREQ + k * COMP_SPEED / FPS
            p["ht"].append(tt); p["hf"].append(total_fx(tt, p["ankle"])); p["hd"].append(toe_down(tt, p["ankle"]))

    w = writer()
    nframes = int(COMP_LEN * FPS)
    with w.saving(fig, path, dpi=100):
        for fi in range(nframes):
            T = fi / FPS
            t += COMP_SPEED / FPS
            a_all = fade(T, 0, COMP_LEN, 0.5)
            for p in panels:
                P = m.joints(t, p["ankle"])
                p["sw"].update(P)
                F, C = m.segment_forces(t, p["ankle"])
                Ff = F[2] + 0.5 * F[1]; c = C[2]
                p["af"].set_positions(tuple(c), tuple(c + 0.13 * Ff))
                p["aw"].set_positions(tuple(c), tuple(c - 0.13 * Ff))
                fx = F[:, 0].sum()
                base = np.array([0.2, -1.3])
                p["at"].set_positions(tuple(base), tuple(base + [0.3 * fx, 0]))
                p["at"].set_color(C_THRUST if fx >= 0 else C_FORCE)
                p["ht"].append(t); p["hf"].append(fx); p["hd"].append(toe_down(t, p["ankle"]))
                p["gax"].clear()
                draw_thrust_graph(p["gax"], p["ht"], p["hf"], p["hd"], 1.0, title="추진력", ylim=(-6.0, 2.6))
                b = p["bax"]; b.clear()
                mv = means[p["ankle"]]
                b.set_facecolor((0, 0, 0, 0.55))
                b.bar([0], [mv], color=C_THRUST if mv > 0 else C_FORCE, width=0.6)
                b.axhline(0, color="white", lw=1)
                b.set_xlim(-0.6, 0.6); b.set_ylim(-1.3, 1.3)
                b.set_xticks([]); b.set_yticks([])
                for sp in b.spines.values():
                    sp.set_color((1, 1, 1, 0.4))
                b.text(0, 1.18, "평균", color="white", ha="center", va="top", fontsize=12)
                b.text(0, mv + (0.12 if mv > 0 else -0.12), f"{mv:+.2f}", color="white",
                       ha="center", va="bottom" if mv > 0 else "top", fontsize=13, weight="bold")
            for c0, c1, txt in caps:
                if c0 <= T < c1:
                    al = fade(T, c0, c1, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.55 * al)
            w.grab_frame()
            if fi % 150 == 0:
                print(f"compare {T:5.1f}s", flush=True)
    plt.close(fig)


# ----------------------------------------------------------------- 조립
def main():
    main_mp4 = os.path.join(TMP, "main.mp4")
    comp_mp4 = os.path.join(TMP, "compare.mp4")
    if os.environ.get("PREVIEW"):
        render_main(main_mp4) if os.environ.get("ONLY") != "compare" else None
        render_compare(comp_mp4) if os.environ.get("ONLY") != "main" else None
        return
    render_main(main_mp4)
    render_compare(comp_mp4)
    cut = SCENES["s3"][1]
    a = os.path.join(TMP, "a.mp4"); c = os.path.join(TMP, "c.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", main_mp4, "-t", str(cut),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", a], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(cut), "-i", main_mp4,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", c], check=True)
    lst = os.path.join(TMP, "list.txt")
    with open(lst, "w") as fh:
        for f in (a, comp_mp4, c):
            fh.write(f"file '{f}'\n")
    out = os.path.join(OUT, "freestyle_kick_propulsion.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print("done:", out)


if __name__ == "__main__":
    main()
