"""스타트 후 수중 돌핀킥 설명 영상 렌더러.

python3 render.py  ->  underwater_dolphin_kick.mp4
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
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.animation import FFMpegWriter

import dolphin_model as m

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
C_FORCE = "#ff5a4e"
C_WATER = "#4fc3ff"
C_THRUST = "#5dff8a"
C_DOWN = "#ffb347"
C_UP = "#9be7ff"
PH_COL = {"출발·비행": "#bfe6ff", "글라이드": "#c7a6ff", "돌핀킥": "#ffb347", "자유형": "#5dff8a"}
TC = 1 / m.FREQ


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, ramp=0.6):
    return smooth((t - t0) / ramp) * smooth((t1 - t) / ramp)


def arrow(ax, color, z=20, lw=3.5, ms=22):
    a = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z)
    ax.add_patch(a)
    return a


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


class Body:
    """돌핀킥 몸 (마디 사슬) + 머리."""

    LW = (7, 9, 11, 26, 22, 16, 11)
    COL = (SKIN, SKIN, SKIN, SKIN, SKIN, SKIN, SKIN_D)

    def __init__(self, ax, scale=1.0, z=10):
        self.segs = [ax.plot([], [], color=c, lw=lw * scale, solid_capstyle="round", zorder=z + 0.01 * i)[0]
                     for i, (c, lw) in enumerate(zip(self.COL, self.LW))]
        self.suit, = ax.plot([], [], color=SUIT, lw=24 * scale, solid_capstyle="butt", zorder=z + 0.2)
        self.head = Circle((0, 0), 0.12, color=CAP, zorder=z + 0.3)
        ax.add_patch(self.head)

    def update(self, P, alpha=1.0):
        for i, ln in enumerate(self.segs):
            ln.set_data(P[i:i + 2, 0], P[i:i + 2, 1]); ln.set_alpha(alpha)
        hip = P[4]; d = (P[3] - P[4]) / np.linalg.norm(P[3] - P[4])
        q = hip + d * 0.2
        self.suit.set_data([hip[0] - d[0] * 0.05, q[0]], [hip[1] - d[1] * 0.05, q[1]]); self.suit.set_alpha(alpha)
        # 머리: 어깨 앞, 두 팔 사이
        n = np.array([-d[1], d[0]])
        self.head.center = tuple(P[3] + d * 0.17 + n * 0.02); self.head.set_alpha(alpha)


def pose_streamline(t):
    """물결 없이 일직선 (글라이드·비행)."""
    P = np.zeros((len(m.S), 2))
    P[:, 0] = -m.S
    return P


def place(P, origin, ang, scale=1.0):
    """몸 좌표(손끝 = 원점, -x 가 발 쪽)를 회전·이동."""
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return origin + (P * scale) @ R.T


# ----------------------------------------------------------------- 스타트 경로 (수영장 옆모습)
PROFILE = m.start_profile()
PR = np.array([(a, b, c) for a, b, c, _ in PROFILE])
PR_PH = [d for *_, d in PROFILE]
ALT = m.start_profile(breakout_x=0.0)
AR = np.array([(a, b, c) for a, b, c, _ in ALT])
BREAK_X = 13.0
BLOCK_H = 0.75
DEPTH = -0.9


def path_y(x):
    """손끝이 지나가는 경로 높이 (수면 = 0)."""
    if x < 3.3:
        k = x / 3.3
        return BLOCK_H + 0.6 * np.sin(np.pi * k * 0.8) * (1 - k) * 2.0 - BLOCK_H * k ** 1.8
    if x < 6.0:
        return DEPTH * smooth((x - 3.3) / 2.7)
    if x < 9.5:
        return DEPTH
    if x < BREAK_X:
        return DEPTH * (1 - smooth((x - 9.5) / (BREAK_X - 9.5)))
    return 0.0


def t_at(arr, x):
    i = np.argmax(arr[:, 1] >= x)
    return arr[i, 0]


T15 = t_at(PR, 15.0)
T15_ALT = t_at(AR, 15.0)

# ----------------------------------------------------------------- 장면
SCENES = [  # (시작, 끝, 이름, 배속)
    (0.0, 5.0, "title", 0.12),
    (5.0, 24.0, "s1", 0.0),
    (24.0, 41.0, "s2", 0.12),
    (41.0, 58.0, "s3", 0.09),
    (58.0, 74.0, "s4", 0.12),
    (74.0, 86.0, "tips", 0.12),
    (86.0, 96.0, "summary", 0.12),
]
END = 96.0
S1_RUN = (6.5, 21.5)          # 스타트 재생 구간 (영상 시각)
HEADERS = {
    "s1": "① 스타트 전체 흐름 : 가장 빠른 순간은 입수 직후",
    "s2": "② 수중 돌핀킥 동작 : 가슴에서 발끝으로 전달되는 물결",
    "s3": "③ 추진력은 어디서? : 발등·발바닥이 오리발처럼",
    "s4": "④ 물의 흐름 : 소용돌이 쌍과 뒤로 뿜어지는 물",
    "tips": "⑤ 잘하는 방법",
}
CAPTIONS = [
    (5.0, 10.0, "출발대를 차고 날아 입수하는 순간 속도가 가장 빠르다 (약 4.8 m/s, 수영 속도의 2배 이상)\n이후는 이 속도를 얼마나 덜 잃느냐의 싸움"),
    (10.0, 15.0, "입수 후 유선형 자세로 잠깐 글라이드 → 속도가 수영 속도 가까이 떨어지기 전에 돌핀킥 시작\n물속 약 0.8~1 m 깊이에서 차야 수면 물결(조파 저항)을 피한다"),
    (15.0, 20.0, "물속 돌핀킥은 물결을 만들지 않아, 같은 힘으로도 수면 수영보다 빠르다\n15 m 이전(규정)에 떠올라 첫 스트로크로 자연스럽게 이어간다"),
    (20.0, 24.0, "15 m 도착 : 돌핀킥 잠영 {t1:.2f}초 vs 바로 떠올라 자유형 {t2:.2f}초 (모델 계산)\n→ 잠영 구간을 잘 쓰면 약 {dt:.1f}초 이득"),
    (24.0, 30.0, "팔은 머리 위로 쭉 펴서 귀 뒤로 조이고, 머리는 팔 사이에 고정 (유선형)\n움직임은 가슴에서 시작해 골반 → 무릎 → 발끝으로 '물결'처럼 전달된다"),
    (30.0, 35.5, "진폭은 가슴에서 가장 작고 발끝에서 가장 크다 (채찍 끝)\n무릎은 다운킥 직전에만 살짝 굽혔다가, 발끝까지 쭉 펴며 내려찬다"),
    (35.5, 41.0, "노란 선 = 발끝이 물속에 남긴 궤적 : 몸이 나아가며 물결 모양이 된다\n작고 빠르게 — 크게 차면 몸 전체가 위아래로 흔들려 저항만 커진다"),
    (41.0, 47.0, "빨간 화살표 = 물이 발·정강이를 되미는 힘, 파란 = 발이 물을 미는 힘\n발등(다운킥)과 발바닥(업킥)이 오리발처럼 물을 뒤로 민다"),
    (47.0, 53.0, "추진력은 다운킥에서 크게(약 {down}%), 업킥에서도 나온다(약 {up}%)\n자유형 발차기보다 업킥의 몫이 커서, 위로 찰 때도 힘을 빼지 말아야 한다"),
    (53.0, 58.0, "부위별 : 발 약 {foot}%, 정강이 약 {shank}%, 허벅지 약 {thigh}%\n→ 발목을 쭉 펴서(포인트) 발을 넓은 '꼬리지느러미'처럼 쓰는 것이 핵심"),
    (58.0, 64.0, "점 = 물 입자, 밝을수록 빨라진 물 (카메라가 몸을 따라가므로 물은 오른쪽 → 왼쪽)"),
    (64.0, 69.0, "발끝이 방향을 바꿀 때마다 서로 반대로 도는 소용돌이 쌍이 떨어져 나간다\n빨강 = 반시계 회전,  파랑 = 시계 회전"),
    (69.0, 74.0, "흰 화살표 = 물 자체의 움직임 : 소용돌이 쌍 사이로 물이 위·아래로 번갈아 비스듬히 뒤쪽으로 밀려난다\n돌고래·물고기와 같은 원리로, 물을 뒤로 보낸 만큼 몸이 앞으로 나아간다"),
]
TIPS = [
    "유선형 : 손을 포개고 팔꿈치를 귀 뒤로 조인다, 머리는 팔 사이에 (고개 들지 않기)",
    "깊이 : 수면 아래 약 0.8~1 m — 너무 얕으면 물결 저항, 너무 깊으면 올라오는 거리 손해",
    "타이밍 : 입수 후 잠깐 글라이드, 속도가 떨어지기 전에 킥 시작",
    "킥 : 가슴부터 물결, 작고 빠르게, 다운킥·업킥 모두 힘 있게, 발목은 쭉 펴서",
    "브레이크아웃 : 15 m 전에 비스듬히 떠오르며 첫 스트로크로 연결",
    "연습 : 벽 차고 스트림라인 글라이드 → 옆으로 누워 돌핀킥 → 오리발 돌핀킥 → 수직 돌핀킥",
]
WARN = "[안전] 잠영 전에 숨을 여러 번 크게 몰아쉬지 마세요(과호흡). 숨이 차다는 신호가 늦게 와서\n물속에서 갑자기 의식을 잃을 수 있습니다. 혼자 길게 잠영하지 말고, 한 번에 10~15 m 정도로."
SUMMARY = [
    "정리",
    "1.  가장 빠른 순간은 입수 직후 — 잠영은 그 속도를 오래 지키는 구간",
    "2.  물속에서는 물결 저항이 없어 같은 힘으로 수면보다 빠르다",
    "3.  돌핀킥 = 가슴 → 골반 → 무릎 → 발끝으로 전달되는 물결, 작고 빠르게",
    "4.  추진은 발등(다운킥)과 발바닥(업킥) — 발이 꼬리지느러미 역할",
]


def scene_at(T):
    for sc in SCENES:
        if sc[0] <= T < sc[1]:
            return sc
    return SCENES[-1]


def thrust_stats():
    ts = np.linspace(0, TC, 600, endpoint=False)
    fx = np.array([m.segment_forces(t)[0][:, 0] for t in ts])
    tot = fx.sum(1)
    down = np.array([m.toe_down(t) for t in ts])
    pos = np.maximum(tot, 0)
    seg = np.maximum(fx, 0).sum(0)
    seg = seg / seg.sum()
    return dict(down=int(round(100 * pos[down].sum() / pos.sum())),
                up=int(round(100 * pos[~down].sum() / pos.sum())),
                foot=int(round(seg[6] * 100)), shank=int(round(seg[5] * 100)), thigh=int(round(seg[4] * 100)),
                peak=tot.max())


def render(path):
    st = thrust_stats()
    caps = [(a, b, txt.format(t1=T15, t2=T15_ALT, dt=T15_ALT - T15, **st)) for a, b, txt in CAPTIONS]

    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("#06182e")

    # ---------- 장면 ①: 수영장 옆모습 + 속도 그래프
    pax = fig.add_axes([0.0, 0.40, 1.0, 0.52])
    PXL, PXR = -0.8, 16.2
    PYT = 1.9
    PYB = PYT - (PXR - PXL) * (0.52 * H) / (1.0 * W)
    pax.set_xlim(PXL, PXR); pax.set_ylim(PYB, PYT); pax.axis("off")
    sky = np.linspace(0, 1, 32)[:, None]
    pax.imshow(np.dstack([0.75 + 0.15 * sky, 0.86 + 0.1 * sky, 0.97 + 0 * sky]), extent=(PXL, PXR, 0, PYT),
               origin="lower", aspect="auto", zorder=0)
    deep = np.linspace(0, 1, 64)[:, None]
    pax.imshow(np.dstack([0.02 + 0.07 * deep, 0.12 + 0.24 * deep, 0.25 + 0.32 * deep]),
               extent=(PXL, PXR, PYB, 0), origin="lower", aspect="auto", zorder=0)
    pax.plot([PXL, PXR], [0, 0], color="white", lw=2, zorder=1)
    pax.add_patch(Rectangle((-0.8, 0), 0.75, BLOCK_H, color="#5c677d", zorder=2))
    pax.add_patch(Rectangle((-0.8, PYB), 0.35, -PYB, color="#8d99ae", zorder=2))
    for xm in (5, 10, 15):
        pax.plot([xm, xm], [PYB, 0.0], color="white", lw=1, ls=":", alpha=0.5, zorder=1)
        pax.text(xm, PYB + 0.08, f"{xm} m", color="white", ha="center", fontsize=12, zorder=3)
    pax.plot([15, 15], [PYB, PYT], color="#ff6b6b", lw=2, ls="--", zorder=1)
    pax.text(15.05, PYT - 0.1, "15 m (잠영 제한)", color="#ff6b6b", fontsize=12, va="top", zorder=3)
    path_x = np.linspace(0, 15.5, 400)
    pax.plot(path_x, [path_y(x) for x in path_x], color="white", lw=1, ls="--", alpha=0.35, zorder=2)
    big = Body(pax, scale=0.45)
    ph_lab = pax.text(0, 0, "", fontsize=15, weight="bold", ha="center", zorder=30,
                      bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, 0.55), ec="none"))
    spd_lab = pax.text(0, 0, "", fontsize=14, color="white", ha="center", zorder=30)
    surf_waves, = pax.plot([], [], color="white", lw=2, zorder=3)

    vax = fig.add_axes([0.07, 0.15, 0.88, 0.22])

    # ---------- 장면 ②~④: 몸 확대 (몸 기준)
    bax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    BXL, BXR = -5.6, 0.6
    BYC = 0.0
    bh = (BXR - BXL) * H / W
    BYB, BYT = -bh * 0.56, bh * 0.44
    bax.set_xlim(BXL, BXR); bax.set_ylim(BYB, BYT); bax.axis("off")
    bax.imshow(np.dstack([0.02 + 0.07 * deep, 0.12 + 0.24 * deep, 0.25 + 0.32 * deep]),
               extent=(BXL, BXR, BYB, BYT), origin="lower", aspect="auto", zorder=0)
    gx, gy = np.meshgrid(np.linspace(BXL, BXR, 200), np.linspace(BYB, BYT, 110))
    vort_img = bax.imshow(np.zeros(gx.shape + (4,)), extent=(BXL, BXR, BYB, BYT), origin="lower",
                          aspect="auto", zorder=1, interpolation="bilinear")
    rng = np.random.default_rng(2)
    NP = 2200
    px = rng.uniform(BXL - 0.2, BXR + 0.2, NP); py = rng.uniform(BYB, BYT, NP)
    tracers = bax.scatter(px, py, s=5, c="white", zorder=2, lw=0)
    qx, qy = np.meshgrid(np.arange(-5.4, -1.6, 0.3), np.arange(-1.0, 0.9, 0.22))
    qx, qy = qx.ravel(), qy.ravel()
    quiv = bax.quiver(qx, qy, 0 * qx, 0 * qy, color="white", scale=7, width=0.0024, headwidth=4,
                      zorder=8, alpha=0)
    body = Body(bax)
    trail = LineCollection([], linewidths=3, zorder=7); bax.add_collection(trail)
    crest = bax.scatter([0], [0], s=300, color="#fff176", alpha=0, zorder=15, edgecolor="none")
    crest_lab = bax.text(0, 0, "물결", color="#fff176", fontsize=13, weight="bold", zorder=16, ha="center")
    jl = {n: bax.text(0, 0, n, color="white", fontsize=12, ha="center", zorder=30,
                      bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.5), ec="none"))
          for n in ("어깨", "골반", "무릎", "발끝")}
    a_force = [arrow(bax, C_FORCE), arrow(bax, C_FORCE, lw=2.5, ms=16)]
    a_water = [arrow(bax, C_WATER), arrow(bax, C_WATER, lw=2.5, ms=16)]
    a_thrust = arrow(bax, C_THRUST, lw=5, ms=28)
    l_thrust = bax.text(0, 0, "추진력", color=C_THRUST, fontsize=16, weight="bold", zorder=31)
    phase_txt = fig.text(0.975, 0.86, "", fontsize=24, weight="bold", ha="right", va="center", zorder=40,
                         bbox=dict(boxstyle="round,pad=0.3", fc=(0.03, 0.1, 0.2, 0.85), ec="none"))
    legend1 = fig.text(0.02, 0.86, "● 반시계 회전", fontsize=14, color="#ff6b6b", zorder=40)
    legend2 = fig.text(0.13, 0.86, "● 시계 회전", fontsize=14, color="#6ea8ff", zorder=40)
    gax = fig.add_axes([0.025, 0.17, 0.38, 0.22])
    sbax = fig.add_axes([0.025, 0.42, 0.38, 0.05])
    dbax = fig.add_axes([0.025, 0.49, 0.38, 0.05])

    header = fig.text(0.015, 0.955, "", fontsize=22, weight="bold", color="white", va="center", zorder=70)
    slowmo = fig.text(0.985, 0.955, "", fontsize=13, color="#bfe6ff", ha="right", va="center", zorder=40)
    cap = fig.text(0.015, 0.015, "", fontsize=16, color="white", va="bottom", ha="left", linespacing=1.4,
                   zorder=50, bbox=dict(boxstyle="round,pad=0.45", fc=(0, 0, 0, 0.65), ec="none"))

    overlay = fig.add_axes([0, 0, 1, 1], zorder=60); overlay.axis("off")
    overlay.set_xlim(0, 1); overlay.set_ylim(0, 1)
    ov = overlay.add_patch(Rectangle((0, 0), 1, 1, color=(0.02, 0.07, 0.15), alpha=0))
    title1 = overlay.text(0.5, 0.58, "스타트 후 수중 돌핀킥", fontsize=46, weight="bold", color="white",
                          ha="center", va="center")
    title2 = overlay.text(0.5, 0.45, "가장 빠른 구간은 어떻게 만들어지고, 추진력은 어디서 나오는가",
                          fontsize=22, color="#bfe6ff", ha="center", va="center")
    tip_txts = [overlay.text(0.06, 0.82 - i * 0.085, f"{i + 1}.  {s}", fontsize=18, color="white", va="center")
                for i, s in enumerate(TIPS)]
    warn_txt = overlay.text(0.06, 0.20, WARN, fontsize=16, color="#ffd166", va="center", linespacing=1.5,
                            bbox=dict(boxstyle="round,pad=0.5", fc=(0.35, 0.1, 0.05, 0.6), ec="none"))
    sums = [overlay.text(0.07, 0.80 - i * 0.12, s, fontsize=34 if i == 0 else 21,
                         weight="bold" if i == 0 else "normal", color="#ffd166" if i == 0 else "white",
                         va="center") for i, s in enumerate(SUMMARY)]
    note = overlay.text(0.07, 0.12, "* 단순화한 2차원 물리 모델(압력 항력 + 와류법)과 1차원 운동 방정식의 시각화입니다. 수치는 모델 값이며 사람마다 다릅니다.",
                        fontsize=13, color="#9fb3c8")

    wake = m.Wake(gain=1.5)
    pre = 0.25 / FPS
    wake.t = -2.0 * TC
    for _ in range(int(2.0 * TC / pre)):
        u, v = wake.velocity(px, py)
        px += (u - m.U) * pre; py += v * pre
        wake.step(pre)
        out = px < BXL - 0.2
        px[out] = rng.uniform(BXR, BXR + 0.2, out.sum()); py[out] = rng.uniform(BYB, BYT, out.sum())
    hist_t, hist_f, hist_d = [], [], []
    toe_pts = []

    def body_layer_visible(vis):
        bax.set_visible(vis)
        for a in (gax, sbax, dbax):
            a.set_visible(False)

    w_ = writer()
    with w_.saving(fig, path, dpi=100):
        for fi in range(int(END * FPS)):
            T = fi / FPS
            sc = scene_at(T)
            name = sc[2]
            in_s1 = name == "s1"
            pax.set_visible(in_s1); vax.set_visible(in_s1)
            body_layer_visible(not in_s1)

            if in_s1:
                # 스타트 재생: 실제 시간의 0.6배 속도
                k = np.clip((T - S1_RUN[0]) / (S1_RUN[1] - S1_RUN[0]), 0, 1)
                tr = k * 7.0
                i = min(np.searchsorted(PR[:, 0], tr), len(PR) - 1)
                x, v = PR[i, 1], PR[i, 2]; ph = PR_PH[i]
                y = path_y(x)
                y2 = path_y(x + 0.05)
                ang = np.arctan2(y2 - y, 0.05)
                if ph == "돌핀킥":
                    P = m.joints(tr)
                else:
                    P = pose_streamline(tr)
                big.update(place(P, np.array([x, y]), ang, scale=0.95))
                ph_lab.set_text(ph); ph_lab.set_color(PH_COL[ph])
                ph_lab.set_position((x - 1.0, y + 0.55))
                spd_lab.set_text(f"{v:.1f} m/s"); spd_lab.set_position((x - 1.0, y + 0.3))
                if ph == "자유형":
                    wx = np.linspace(x - 2.5, x + 0.3, 40)
                    surf_waves.set_data(wx, 0.04 * np.sin((wx - x) * 6) * np.exp((wx - x) / 1.5))
                else:
                    surf_waves.set_data([], [])
                vax.clear()
                vax.set_facecolor((0, 0, 0, 0.4))
                for sp in vax.spines.values():
                    sp.set_color((1, 1, 1, 0.3))
                vax.tick_params(colors="white", labelsize=11)
                alt_mask = np.array([p == "자유형" for *_, p in ALT])
                vax.plot(np.where(alt_mask, AR[:, 1], np.nan), np.where(alt_mask, AR[:, 2], np.nan), color="#adb5bd", lw=2, ls="--", label="글라이드 후 바로 떠올라 자유형")
                sel = PR[:, 0] <= tr
                xs, vs = PR[sel, 1], PR[sel, 2]
                phs = np.array(PR_PH)[sel]
                for pn, col in PH_COL.items():
                    mk = phs == pn
                    if mk.any():
                        vax.plot(np.where(mk, xs, np.nan), np.where(mk, vs, np.nan), color=col, lw=3.5, label=pn)
                vax.plot([x], [v], "o", color="white", ms=8)
                vax.set_xlim(0, 16); vax.set_ylim(0, 5.3)
                vax.set_xlabel("거리 (m)", color="white", fontsize=12); vax.set_ylabel("속도 (m/s)", color="white", fontsize=12)
                vax.axvline(15, color="#ff6b6b", lw=1, ls="--")
                vax.legend(loc="upper right", fontsize=10, facecolor=(0, 0, 0, 0.4), edgecolor="none",
                           labelcolor="white", ncol=5)
                if k >= 1.0:
                    vax.text(15.1, 4.2, f"15 m 도착\n잠영 {T15:.2f}초\n바로 수면 {T15_ALT:.2f}초", color="white",
                             fontsize=11, va="top")
                phase_txt.set_alpha(0); phase_txt.get_bbox_patch().set_alpha(0)
                legend1.set_alpha(0); legend2.set_alpha(0)
            else:
                spd = sc[3]
                dts = spd / FPS
                u, v = wake.velocity(px, py)
                px += (u - m.U) * dts; py += v * dts
                wake.step(dts)
                out = (px < BXL - 0.2) | (py > BYT + 0.2) | (py < BYB - 0.2)
                px[out] = rng.uniform(BXR, BXR + 0.2, out.sum()); py[out] = rng.uniform(BYB, BYT, out.sum())
                t = wake.t
                P = m.joints(t)
                body.update(P)
                F, C = m.segment_forces(t)
                fx = F[:, 0].sum()
                down = m.toe_down(t)
                hist_t.append(t); hist_f.append(fx); hist_d.append(down)
                toe_pts = [(xx - m.U * dts, yy) for xx, yy in toe_pts] + [tuple(P[-1])]
                toe_pts = [p for p in toe_pts if p[0] > BXL - 0.2]

                # 물 입자
                u, v = wake.velocity(px, py)
                sp_ = np.hypot(u, v)
                a_tr = 0.25 + 0.75 * smooth((T - 58.0) / 1.0) if name == "s4" else 0.3
                cval = np.clip(sp_ / 0.5, 0, 1)
                cols = np.empty((len(px), 4))
                cols[:, 0] = 0.35 + 0.65 * cval; cols[:, 1] = 0.6 + 0.4 * cval; cols[:, 2] = 1.0
                cols[:, 3] = a_tr * (0.35 + 0.65 * cval)
                tracers.set_offsets(np.c_[px, py]); tracers.set_facecolors(cols); tracers.set_sizes(4 + 14 * cval)
                a_v = fade(T, 63.5, 74.6, 1.0)
                if a_v > 0.01:
                    sel = wake.vx > BXL - 0.2
                    d2 = (wake.delta * 1.3) ** 2
                    r2 = (gx.ravel()[:, None] - wake.vx[sel]) ** 2 + (gy.ravel()[:, None] - wake.vy[sel]) ** 2
                    om = (wake.g[sel] / (np.pi * d2) * np.exp(-r2 / d2)).sum(1).reshape(gx.shape)
                    s_ = np.clip(om / 6.0, -1, 1)
                    rgba = np.zeros(gx.shape + (4,))
                    rgba[s_ > 0] = (1.0, 0.28, 0.28, 0); rgba[s_ <= 0] = (0.3, 0.6, 1.0, 0)
                    rgba[..., 3] = np.abs(s_) ** 0.8 * 0.85 * a_v
                    vort_img.set_data(rgba); vort_img.set_visible(True)
                else:
                    vort_img.set_visible(False)
                legend1.set_alpha(a_v); legend2.set_alpha(a_v)
                a_q = fade(T, 68.5, 74.6, 1.0)
                if a_q > 0.01:
                    uq, vq = wake.velocity(qx, qy, delta=0.1)
                    quiv.set_UVC(uq, vq); quiv.set_alpha(a_q)
                else:
                    quiv.set_alpha(0)

                # 물결 마루 (s2)
                a_c = fade(T, 24.5, 35.5, 0.6)
                ss = np.linspace(0.8, 2.45, 200)
                yy = m.centerline_y(ss, t)
                j = np.argmax(yy)
                ci = np.interp(ss[j], m.S, np.arange(len(m.S)))
                i0 = int(min(ci, len(m.S) - 2)); fr = ci - i0
                cp = P[i0] * (1 - fr) + P[i0 + 1] * fr
                crest.set_offsets([cp]); crest.set_alpha(0.6 * a_c)
                crest_lab.set_position((cp[0], cp[1] + 0.2)); crest_lab.set_alpha(a_c)
                a_j = fade(T, 24.5, 30.0, 0.5)
                for n, idx in (("어깨", 3), ("골반", 4), ("무릎", 5), ("발끝", 7)):
                    jl[n].set_position((P[idx][0], P[idx][1] - 0.32)); jl[n].set_alpha(a_j)
                    jl[n].get_bbox_patch().set_alpha(0.5 * a_j); jl[n].set_visible(a_j > 0.01)
                a_tl = fade(T, 35.0, 41.3, 0.7)
                pts = np.array(toe_pts)
                if a_tl > 0.01 and len(pts) > 1:
                    segs = np.stack([pts[:-1], pts[1:]], 1)
                    al = np.linspace(0, 1, len(segs)) * a_tl
                    trail.set_segments(segs); trail.set_color([(1, 0.85, 0.2, a) for a in al])
                else:
                    trail.set_segments([])

                # 힘 (s3)
                a_f = fade(T, 41.3, 58.0, 0.8)
                vis = a_f > 0.01
                scf = 1.3
                for kk, seg in enumerate((6, 5)):
                    on = vis and np.linalg.norm(F[seg]) * scf > 0.03
                    a_force[kk].set_visible(on); a_water[kk].set_visible(on)
                    a_force[kk].set_alpha(a_f); a_water[kk].set_alpha(a_f)
                    if on:
                        a_force[kk].set_positions(tuple(C[seg]), tuple(C[seg] + F[seg] * scf))
                        a_water[kk].set_positions(tuple(C[seg]), tuple(C[seg] - F[seg] * scf))
                base = np.array([-1.0, 0.55])
                a_thrust.set_positions(tuple(base), tuple(base + [fx * 1.2, 0]))
                a_thrust.set_visible(vis and fx * 1.2 > 0.04); a_thrust.set_alpha(a_f)
                l_thrust.set_position((base[0] + fx * 1.2 + 0.08, base[1] - 0.05))
                l_thrust.set_visible(vis); l_thrust.set_alpha(a_f)
                a_g = fade(T, 41.3, 58.0, 0.8)
                gax.clear()
                if a_g > 0.01:
                    gax.set_visible(True)
                    tt = np.array(hist_t); ff = np.array(hist_f); dd = np.array(hist_d)
                    sel = tt > tt[-1] - 2 * TC
                    xg = tt[sel] - tt[-1]
                    gax.set_facecolor((0, 0, 0, 0.6))
                    for spn in gax.spines.values():
                        spn.set_color((1, 1, 1, 0.4))
                    gax.fill_between(xg, -0.2, 1.0, where=dd[sel], color=C_DOWN, alpha=0.16, lw=0, step="mid")
                    gax.axhline(0, color=(1, 1, 1, 0.5), lw=1)
                    gax.fill_between(xg, 0, ff[sel], color=C_THRUST, alpha=0.5, lw=0)
                    gax.plot(xg, ff[sel], color="white", lw=2)
                    gax.plot([0], [ff[-1]], "o", color="white", ms=7)
                    gax.set_xlim(-2 * TC, 0.05 * TC); gax.set_ylim(-0.1, st["peak"] * 1.25)
                    gax.set_xticks([]); gax.set_yticks([])
                    gax.text(0.02, 0.95, "추진력 (앞쪽 성분, 상대값)", transform=gax.transAxes, color="white",
                             fontsize=13, va="top")
                    gax.text(0.98, 0.95, "주황 배경 = 다운킥", transform=gax.transAxes, color=C_DOWN,
                             fontsize=11, va="top", ha="right")
                a_b = fade(T, 47.0, 58.0, 0.6)
                for axb, items in ((dbax, ((st["down"] / 100, f"다운킥 {st['down']}%", C_DOWN),
                                           (st["up"] / 100, f"업킥 {st['up']}%", C_UP))),
                                   (sbax, ((st["foot"] / 100, f"발 {st['foot']}%", SKIN_D),
                                           (st["shank"] / 100, f"정강이 {st['shank']}%", SKIN),
                                           (max(1 - st["foot"] / 100 - st["shank"] / 100, 0), "", "#888888")))):
                    axb.clear()
                    show = a_b > 0.01 and (axb is dbax or T >= 53.0)
                    axb.set_visible(show)
                    if show:
                        al = a_b if axb is dbax else fade(T, 53.0, 58.0, 0.5)
                        axb.set_xlim(0, 1); axb.set_ylim(0, 1); axb.axis("off")
                        x0 = 0
                        for frac, lab, col in items:
                            axb.add_patch(Rectangle((x0, 0), frac, 1, color=col, alpha=al, lw=0))
                            if frac > 0.1:
                                axb.text(x0 + frac / 2, 0.5, lab, ha="center", va="center", fontsize=13,
                                         weight="bold", color="#0d2240", alpha=al)
                            x0 += frac
                ph_on = fade(T, 24.3, 74.0, 0.5)
                phase_txt.set_text("▼ 다운킥" if down else "▲ 업킥")
                phase_txt.set_color(C_DOWN if down else C_UP)
                phase_txt.set_alpha(ph_on); phase_txt.get_bbox_patch().set_alpha(0.85 * ph_on)

            # ---- 텍스트
            if name in HEADERS:
                al = fade(T, sc[0], sc[1], 0.5)
                header.set_text(HEADERS[name]); header.set_alpha(al)
                if name == "s1":
                    slowmo.set_text("슬로모션 ×0.47")
                else:
                    slowmo.set_text(f"슬로모션 ×{sc[3]:g}")
                slowmo.set_alpha(al if name != "tips" else 0)
            else:
                header.set_alpha(0); slowmo.set_alpha(0)
            cap.set_visible(False)
            for a0, a1, txt in caps:
                if a0 <= T < a1:
                    al = fade(T, a0, a1, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.65 * al)
                    cap.set_visible(True)

            a_title = 1 - smooth((T - 3.6) / 1.2)
            a_tips = fade(T, 74.0, 86.0, 0.8)
            a_sum = smooth((T - 86.3) / 1.0)
            ov.set_alpha(0.82 * max(a_title, a_tips, a_sum))
            title1.set_alpha(a_title); title2.set_alpha(a_title)
            for i, t_ in enumerate(tip_txts):
                t_.set_alpha(a_tips * smooth((T - 74.5 - 0.7 * i) / 0.5))
            warn_txt.set_alpha(a_tips * smooth((T - 79.5) / 0.6))
            warn_txt.get_bbox_patch().set_alpha(0.6 * a_tips * smooth((T - 79.5) / 0.6))
            for i, s_ in enumerate(sums):
                s_.set_alpha(smooth((T - 86.6 - 0.9 * i) / 0.6))
            note.set_alpha(smooth((T - 91.0) / 0.8))

            w_.grab_frame()
            if fi % 300 == 0:
                print(f"{T:5.1f}s / {END}s", flush=True)
    plt.close(fig)


def main():
    if os.environ.get("PREVIEW"):
        render(os.path.join(TMP, "preview"))
        return
    raw = os.path.join(TMP, "raw.mp4")
    out = os.path.join(OUT, "underwater_dolphin_kick.mp4")
    render(raw)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-crf", "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print("done:", out)


if __name__ == "__main__":
    main()
