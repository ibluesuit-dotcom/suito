"""손바닥 기울기에 따른 힘 방향 설명 영상 렌더러.

python3 render.py  ->  hand_pitch_force.mp4
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
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.animation import FFMpegWriter

import hand_model as hm

font_manager.fontManager.addfont("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

FPS = 30
W, H = 12.8, 7.2
OUT = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(OUT, "_build")
os.makedirs(TMP, exist_ok=True)

SKIN = "#f2c9a0"
PALM = "#ffd166"
C_D = "#ffa94d"       # 항력
C_L = "#c77dff"       # 양력
C_F = "#ff5a4e"       # 합력
C_X = "#5dff8a"       # 앞쪽 성분
C_Y = "#adb5bd"       # 옆 성분
C_V = "#ffffff"       # 손 움직임
C_WATER = "#4fc3ff"

# 메인 그림 영역 (왼쪽), 좌표 범위
MAIN = [0.0, 0.0, 0.62, 1.0]
MX = 2.4
MY = MX * (MAIN[3] * H) / (MAIN[2] * W)
CHORD = 1.0
A_J = CHORD / 4             # 주코프스키 원 반지름
FSCALE = 0.75               # 힘 화살표 길이 배율


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def fade(t, t0, t1, ramp=0.6):
    return smooth((t - t0) / ramp) * smooth((t1 - t) / ramp)


def lerp(a, b, k):
    return a + (b - a) * k


# ----------------------------------------------------------------- 시나리오
# 각 시각의 (손 움직임 방향 beta[도], 손 속도 크기, 손바닥 기울기 tilt[도]).
#  beta: 0 = 정확히 뒤, 90 = 정확히 아래(옆). tilt: 0 = 손바닥이 정확히 뒤, +면 손이 움직이는 쪽으로 기울임
def scenario(T):
    if T < 5.0:                       # 타이틀
        return 0.0, 1.0, 0.0
    if T < 19.0:                      # ① 똑바로 뒤로
        k = T - 10.0
        if k < 0:
            tilt = 0.0
        elif k < 7.0:
            tilt = 50 * np.sin(np.pi * k / 3.5) * (1 if k < 3.5 else 1)
        else:
            tilt = 0.0
        return 0.0, 1.0, tilt
    if T < 43.0:                      # ② 비스듬히 (뒤 + 아래)
        beta = 45.0 * smooth((T - 19.0) / 1.2)
        if T < 25.5:
            tilt = 45.0 * smooth((T - 19.0) / 1.2)          # 손바닥이 손 움직임 쪽 정면
        elif T < 31.5:
            tilt = lerp(45.0, 0.0, smooth((T - 25.5) / 1.2))  # 손바닥을 정확히 뒤로
        elif T < 38.0:
            k = (T - 31.5) / 6.5                             # 0 → 75 → 최적값 훑기
            tilt = 75 * np.sin(np.pi * min(k * 1.15, 1)) if k * 1.15 < 0.5 else None
            if tilt is None:
                tilt = lerp(75.0, BEST_TILT, smooth((k - 0.5 / 1.15) / (1 - 0.5 / 1.15)))
        else:
            tilt = BEST_TILT
        return beta, 1.0, tilt
    if T < 62.0:                      # ③ 스컬링: 손을 위아래(옆)로만
        k = T - 43.0
        if k < 1.5:
            beta = lerp(45.0, 90.0, smooth(k / 1.5))
            return beta, 1.0, lerp(BEST_TILT, 45.0, smooth(k / 1.5))
        ph = (k - 1.5) / SCULL_PERIOD
        c = np.cos(2 * np.pi * ph)                           # +: 아래로, -: 위로
        speed = abs(c) ** 0.6
        beta = 90.0 if c >= 0 else -90.0
        return beta, speed, 45.0
    return 0.0, 1.0, 0.0


SCULL_PERIOD = 3.2


def hand_state(beta_deg, speed, tilt_deg):
    b = np.radians(beta_deg)
    v = speed * np.array([-np.cos(b), -np.sin(b)])         # 물 기준 손 속도 (뒤 + 아래)
    sgn = 1.0 if beta_deg >= 0 else -1.0
    psi = -sgn * np.radians(tilt_deg)                      # 손 움직임 쪽으로 기울이기
    f = hm.facing(psi)
    return v, f


def forward_curve(beta_deg, tilts):
    out = []
    for td in tilts:
        v, f = hand_state(beta_deg, 1.0, td)
        D, L, _ = hm.hand_force(v, f)
        out.append((D + L))
    return np.array(out)


TILTS = np.linspace(0, 90, 181)
_curve45 = forward_curve(45.0, TILTS)
BEST_TILT = float(TILTS[np.argmax(_curve45[:, 0])])


# ----------------------------------------------------------------- 물 흐름 (판 주위 퍼텐셜 흐름)
def flow_velocity(pts, w, f):
    """평판(손) 주위 흐름. w: 손이 느끼는 물의 상대 속도, f: 손바닥 방향."""
    sp = np.linalg.norm(w)
    if sp < 1e-6:
        return np.zeros_like(pts)
    chord_dir = np.array([f[1], -f[0]])                    # 손바닥에 수직인 방향 = 판 방향
    th = np.arctan2(chord_dir[1], chord_dir[0])
    z = (pts[:, 0] + 1j * pts[:, 1]) * np.exp(-1j * th)
    af = np.arctan2(w[1], w[0]) - th
    a = A_J
    # 역 주코프스키 사상 (|ζ| > a 인 가지)
    root = np.sqrt(z * z - 4 * a * a + 0j)
    zeta = (z + root) / 2
    bad = np.abs(zeta) < a
    zeta[bad] = (z[bad] - root[bad]) / 2
    # 쿠타 조건 순환 × (손 모형 양력계수 / 얇은 날개 이론값)
    alpha = np.arcsin(np.clip(abs(np.sin(af)), 0, 1))
    cl_model = hm.coeffs(alpha)[1]
    k = cl_model / max(2 * np.pi * np.sin(alpha), 1e-6)
    gam = (-4 * np.pi * a * sp * np.sin(af)) if np.cos(af) >= 0 else (4 * np.pi * a * sp * np.sin(af))
    gam *= k
    dF = sp * (np.exp(-1j * af) - a * a * np.exp(1j * af) / zeta ** 2) - 1j * gam / (2 * np.pi * zeta)
    dz = 1 - a * a / zeta ** 2
    vel = dF / np.where(np.abs(dz) < 0.05, 0.05, dz)
    vel = np.conj(vel) * np.exp(1j * th)
    out = np.c_[vel.real, vel.imag]
    s = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(s > 2.5 * sp, out / s * 2.5 * sp, out)


class Particles:
    def __init__(self, n=1400, seed=3):
        self.rng = np.random.default_rng(seed)
        R = np.hypot(MX, MY) + 0.15
        r = R * np.sqrt(self.rng.uniform(0, 1, n)); a = self.rng.uniform(0, 2 * np.pi, n)
        self.p = np.c_[r * np.cos(a), r * np.sin(a)]

    def step(self, w, f, dt):
        v = flow_velocity(self.p, w, f)
        self.p = self.p + v * dt
        sp = np.linalg.norm(w)
        if sp > 1e-3:
            wh = w / sp
            R = np.hypot(MX, MY) + 0.15
            d = self.p @ wh
            out = (np.linalg.norm(self.p, axis=1) > R) & (d > 0)
            # 하류로 빠져나간 입자는 흐름 방향으로 대칭 이동해 상류에서 다시 들어온다 (밀도 유지)
            self.p[out] = self.p[out] - 2 * d[out, None] * wh[None, :]
        return v


# ----------------------------------------------------------------- 렌더
def arrow(ax, color, lw=4, ms=24, z=20, ls="-"):
    a = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=ms, color=color,
                        lw=lw, zorder=z, linestyle=ls)
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
    return FFMpegWriter(fps=FPS, codec="libx264", bitrate=6000,
                        extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])


END = 73.0
HEADERS = [
    (5.0, 19.0, "① 손을 똑바로 뒤로 밀 때"),
    (19.0, 43.0, "② 손이 비스듬히(뒤 + 아래·안쪽) 움직일 때"),
    (43.0, 62.0, "③ 스컬링 : 손을 옆으로만 움직여도 앞으로 가는 힘"),
]
CAPTIONS = [
    (5.0, 10.0, "손을 단면(판)으로 본 모습. 오른쪽이 진행 방향(앞), 노란 면이 손바닥\n손이 뒤로 움직이면 물은 손바닥으로 밀려 들어온다 (파란 점 = 물)"),
    (10.0, 15.0, "물이 손에 주는 힘 = 항력(주황, 물이 흐르는 방향) + 양력(보라, 그에 수직)\n두 힘을 합친 빨간 화살표 중 초록 = 앞으로 가는 힘, 회색 = 옆으로 새는 힘"),
    (15.0, 19.0, "손이 똑바로 뒤로 움직일 때는 손바닥을 정확히 뒤로 향할 때 앞으로 가는 힘이 가장 크다\n기울이면 힘이 옆으로 새어 나간다"),
    (19.0, 25.5, "실제 손은 뒤로 가면서 아래·안쪽으로도 움직인다 (흰 화살표)\n손바닥을 손이 움직이는 쪽 정면으로 향하면 → 힘도 비스듬히, 옆으로 많이 샌다"),
    (25.5, 31.5, "손바닥을 정확히 뒤로 향하면 → 항력은 비스듬하지만 양력이 앞쪽으로 더해져\n앞으로 가는 힘이 더 커진다"),
    (31.5, 38.0, "손바닥 기울기를 바꿔 보면 (오른쪽 그래프)\n'정확히 뒤'와 '손 움직임 정면' 사이의 각도에서 앞으로 가는 힘이 가장 크다"),
    (38.0, 43.0, "최적 : 손바닥은 거의 뒤를 향하되, 손이 움직이는 쪽으로 약 {best:.0f}°만 살짝 기울인다\n→ 항력과 양력이 함께 앞을 향하도록"),
    (43.0, 50.0, "손을 위아래(또는 안팎)로만 움직여도, 손바닥을 비스듬히 기울이면\n양력이 앞쪽을 향해 몸을 앞으로 보낸다 (프로펠러 날개와 같은 원리)"),
    (50.0, 56.5, "방향을 바꿀 때마다 손바닥 기울기도 뒤집어야 양력이 계속 앞을 향한다\n옆으로 새는 힘은 왕복하며 서로 상쇄되고, 앞으로 가는 힘만 쌓인다"),
    (56.5, 62.0, "자유형 스트로크의 S자 궤적에서도 손은 계속 이렇게 각도를 조절한다\n→ 스컬링 연습으로 손바닥 각도 감각을 익히자"),
]
SUMMARY = [
    "정리",
    "1.  손에 걸리는 힘 = 물을 미는 항력 + 날개처럼 생기는 양력",
    "2.  손이 똑바로 뒤로 갈 때 → 손바닥은 정확히 뒤로",
    "3.  손이 비스듬히 갈 때 → 손바닥은 거의 뒤로, 움직이는 쪽으로 살짝 기울이기",
    "4.  스컬링 감각 = 손바닥 각도로 힘의 방향을 조절하는 능력",
]


def render(path):
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor("#06182e")
    ax = fig.add_axes(MAIN)
    ax.set_xlim(-MX, MX); ax.set_ylim(-MY, MY); ax.set_aspect("equal"); ax.axis("off")
    g = np.linspace(0, 1, 128)[:, None]
    ax.imshow(np.dstack([0.03 + 0.05 * g, 0.16 + 0.15 * g, 0.30 + 0.18 * g]),
              extent=(-MX, MX, -MY, MY), origin="lower", aspect="auto", zorder=0)

    parts = Particles()
    streaks = ax.quiver(parts.p[:, 0], parts.p[:, 1], 0 * parts.p[:, 0], 0 * parts.p[:, 0],
                        color=C_WATER, angles="xy", scale_units="xy", scale=1, width=0.003,
                        headwidth=0, headlength=0, headaxislength=0, alpha=0.75, zorder=3)

    hand_back, = ax.plot([], [], color=SKIN, lw=22, solid_capstyle="round", zorder=10)
    hand_palm, = ax.plot([], [], color=PALM, lw=7, solid_capstyle="round", zorder=11)
    t_palm = ax.text(0, 0, "손바닥", color=PALM, fontsize=14, weight="bold", ha="center", va="center", zorder=12)
    t_back = ax.text(0, 0, "손등", color=SKIN, fontsize=14, weight="bold", ha="center", va="center", zorder=12)

    a_v = arrow(ax, C_V, lw=3, ms=20)
    a_D = arrow(ax, C_D)
    a_L = arrow(ax, C_L)
    a_F = arrow(ax, C_F, lw=6, ms=30, z=22)
    a_X = arrow(ax, C_X, lw=5, ms=26, z=23)
    a_Y = arrow(ax, C_Y, lw=3, ms=20, z=21)
    lab = {k: ax.text(0, 0, s, color=c, fontsize=15, weight="bold", zorder=30, ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.45), ec="none"))
           for k, s, c in (("v", "손의 움직임", C_V), ("D", "항력", C_D), ("L", "양력", C_L),
                           ("F", "물이 손에 주는 힘", C_F), ("X", "앞으로", C_X), ("Y", "옆으로", C_Y))}
    a_fwd = arrow(ax, "#bfe6ff", lw=2.5, ms=18, z=5)
    a_fwd.set_positions((-MX + 0.25, MY - 0.55), (-MX + 1.25, MY - 0.55))
    ax.text(-MX + 1.35, MY - 0.55, "진행 방향 (앞)", color="#bfe6ff", fontsize=14, va="center", zorder=5)

    header = fig.text(0.02, 0.955, "", fontsize=23, weight="bold", color="white", va="center", zorder=40)
    cap = fig.text(0.02, 0.025, "", fontsize=16, color="white", va="bottom", ha="left", linespacing=1.45,
                   zorder=50, bbox=dict(boxstyle="round,pad=0.5", fc=(0, 0, 0, 0.65), ec="none"))

    # 오른쪽 패널: 그래프 + 수치
    gax = fig.add_axes([0.675, 0.42, 0.30, 0.42])
    info = fig.text(0.675, 0.31, "", fontsize=15, color="white", va="top", linespacing=1.6, zorder=40)

    overlay = fig.add_axes([0, 0, 1, 1], zorder=60); overlay.axis("off")
    overlay.set_xlim(0, 1); overlay.set_ylim(0, 1)
    ov = overlay.add_patch(Rectangle((0, 0), 1, 1, color=(0.02, 0.07, 0.15), alpha=0))
    title1 = overlay.text(0.5, 0.58, "손바닥 기울기와 힘의 방향", fontsize=44, weight="bold",
                          color="white", ha="center", va="center")
    title2 = overlay.text(0.5, 0.45, "손바닥은 정확히 뒤를 향해야 할까?", fontsize=24, color="#bfe6ff",
                          ha="center", va="center")
    sums = [overlay.text(0.07, 0.80 - i * 0.12, s, fontsize=34 if i == 0 else 21,
                         weight="bold" if i == 0 else "normal", color="#ffd166" if i == 0 else "white",
                         va="center") for i, s in enumerate(SUMMARY)]
    note = overlay.text(0.07, 0.12, "* 손을 얇은 판으로 단순화한 2차원 모델. 양력·항력 계수는 손 모형 실험값의 모양을 따른 근사이며, 힘은 상대값입니다.",
                        fontsize=13, color="#9fb3c8")

    caps = [(a, b, t.format(best=BEST_TILT)) for a, b, t in CAPTIONS]
    hist = []          # 스컬링 기록 (시각, Fx, Fy)

    w_ = writer()
    with w_.saving(fig, path, dpi=100):
        for fi in range(int(END * FPS)):
            T = fi / FPS
            beta, speed, tilt = scenario(T)
            v, f = hand_state(beta, speed, tilt)
            wrel = -v
            D, L, alpha = hm.hand_force(v, f)
            F = D + L

            # --- 물 입자
            vel = parts.step(wrel, f, 1.1 / FPS)
            streaks.set_offsets(parts.p)
            streaks.set_UVC(-vel[:, 0] * 0.12, -vel[:, 1] * 0.12)

            # --- 손 (판)
            cd = np.array([f[1], -f[0]]) * CHORD / 2
            hand_back.set_data([-cd[0], cd[0]], [-cd[1], cd[1]])
            off = f * 0.045
            hand_palm.set_data([-cd[0] + off[0], cd[0] + off[0]], [-cd[1] + off[1], cd[1] + off[1]])
            t_palm.set_position(tuple(f * 0.32 + cd * 0.9))
            t_back.set_position(tuple(-f * 0.30 + cd * 0.9))

            # --- 화살표
            a_vis = fade(T, 5.5, 62.0, 0.5)
            a_force = fade(T, 10.0, 62.0, 0.5)
            vh = v * 0.9
            a_v.set_positions((0.0, 0.0), tuple(vh))
            lab["v"].set_position(tuple(vh * 1.18 + [0, -0.12]))
            for art in (a_v, lab["v"]):
                art.set_alpha(a_vis)
            s = FSCALE
            for key, vec, art in (("D", D, a_D), ("L", L, a_L), ("F", F, a_F)):
                art.set_positions((0.0, 0.0), tuple(vec * s))
                art.set_alpha(a_force)
                art.set_visible(np.linalg.norm(vec) * s > 0.04)
                lab[key].set_position(tuple(vec * s * 1.12 + {"D": (0.0, -0.13), "L": (0.18, 0.0), "F": (0.0, 0.13)}[key]))
                lab[key].set_alpha(a_force); lab[key].get_bbox_patch().set_alpha(0.45 * a_force)
                lab[key].set_visible(np.linalg.norm(vec) * s > 0.15)
            fx, fy = F[0] * s, F[1] * s
            base = np.array([0.0, -MY + 1.15])
            a_X.set_positions(tuple(base), tuple(base + [fx, 0]))
            a_Y.set_positions(tuple(base + [fx, 0]), tuple(base + [fx, fy]))
            a_X.set_alpha(a_force); a_Y.set_alpha(a_force)
            a_X.set_color(C_X if fx >= 0 else C_F)
            a_X.set_visible(abs(fx) > 0.03); a_Y.set_visible(abs(fy) > 0.03)
            lab["X"].set_position(tuple(base + [fx / 2, 0.16])); lab["X"].set_alpha(a_force)
            lab["Y"].set_position(tuple(base + [fx + 0.42, fy / 2])); lab["Y"].set_alpha(a_force)
            lab["X"].set_visible(abs(fx) > 0.2); lab["Y"].set_visible(abs(fy) > 0.15)
            for k in ("X", "Y"):
                lab[k].get_bbox_patch().set_alpha(0.45 * a_force)

            # --- 오른쪽 그래프
            gax.clear()
            a_g = fade(T, 10.0, 62.0, 0.5)
            gax.set_visible(a_g > 0.01)
            if a_g > 0.01:
                gax.set_facecolor((0, 0, 0, 0.5))
                for sp in gax.spines.values():
                    sp.set_color((1, 1, 1, 0.4))
                gax.tick_params(colors="white", labelsize=11)
                if T < 43.0:
                    b_now = 0.0 if T < 19.0 else 45.0
                    cur = forward_curve(b_now, TILTS)
                    gax.plot(TILTS, cur[:, 0], color=C_X, lw=3, alpha=a_g, label="앞으로 가는 힘")
                    gax.plot(TILTS, np.abs(cur[:, 1]), color=C_Y, lw=2, ls="--", alpha=a_g, label="옆으로 새는 힘")
                    i = np.argmax(cur[:, 0])
                    gax.axvline(TILTS[i], color=C_X, lw=1, ls=":", alpha=a_g)
                    gax.plot([abs(tilt)], [F[0]], "o", color="white", ms=10, alpha=a_g)
                    gax.set_xlim(0, 90); gax.set_ylim(0, 1.45)
                    gax.set_xlabel("손바닥 기울기 (°)  0 = 정확히 뒤", color="white", fontsize=12)
                    gax.set_title("손바닥 기울기에 따른 힘" + (" (손이 똑바로 뒤로)" if T < 19 else " (손이 45° 비스듬히)"),
                                  color="white", fontsize=13)
                    if T >= 19.0:
                        gax.axvline(45, color=C_V, lw=1, ls=":", alpha=0.6 * a_g)
                        gax.text(45.5, 1.38, "손 움직임 정면", color=C_V, fontsize=10, va="top", alpha=a_g)
                    gax.text(TILTS[i] + 1, 0.05, f"최대 {TILTS[i]:.0f}°", color=C_X, fontsize=11, alpha=a_g)
                    leg = gax.legend(loc="upper right", fontsize=11, facecolor=(0, 0, 0, 0.4), edgecolor="none",
                                     labelcolor="white", bbox_to_anchor=(1.0, 0.9))
                else:
                    hist.append((T, F[0], F[1]))
                    hh = np.array(hist)
                    sel = hh[:, 0] > T - 2 * SCULL_PERIOD
                    hh = hh[sel]
                    x = hh[:, 0] - T
                    gax.axhline(0, color="white", lw=1, alpha=0.5)
                    gax.plot(x, hh[:, 1], color=C_X, lw=3, label="앞으로 가는 힘")
                    gax.plot(x, hh[:, 2], color=C_Y, lw=2, ls="--", label="옆으로 새는 힘")
                    allh = np.array(hist)
                    gax.axhline(allh[:, 1].mean(), color=C_X, lw=1, ls=":")
                    gax.set_xlim(-2 * SCULL_PERIOD, 0.1); gax.set_ylim(-1.0, 1.3)
                    gax.set_xticks([])
                    gax.set_title("스컬링 중 시간에 따른 힘", color="white", fontsize=13)
                    gax.legend(loc="lower left", fontsize=11, facecolor=(0, 0, 0, 0.4), edgecolor="none",
                               labelcolor="white")
            # 수치
            info.set_alpha(a_g)
            if a_g > 0.01:
                if T < 43.0:
                    info.set_text(f"손바닥 기울기 : {tilt:4.0f}°\n받음각 (물 흐름과 손바닥) : {np.degrees(alpha):4.0f}°\n"
                                  f"앞으로 가는 힘 : {F[0]:5.2f}\n옆으로 새는 힘 : {abs(F[1]):5.2f}")
                else:
                    allh = np.array(hist)
                    info.set_text(f"평균 앞으로 가는 힘 : {allh[:, 1].mean():5.2f}\n"
                                  f"평균 옆으로 새는 힘 : {allh[:, 2].mean():+5.2f}\n(왕복하며 옆 힘은 상쇄)")
            else:
                info.set_text("")

            # --- 텍스트
            header.set_text("")
            for a0, a1, txt in HEADERS:
                if a0 <= T < a1:
                    header.set_text(txt); header.set_alpha(fade(T, a0, a1, 0.5))
            cap.set_visible(False)
            for a0, a1, txt in caps:
                if a0 <= T < a1:
                    al = fade(T, a0, a1, 0.4)
                    cap.set_text(txt); cap.set_alpha(al); cap.get_bbox_patch().set_alpha(0.65 * al)
                    cap.set_visible(True)

            a_title = 1 - smooth((T - 3.6) / 1.2)
            a_sum = smooth((T - 62.3) / 1.0)
            ov.set_alpha(0.8 * max(a_title, a_sum))
            title1.set_alpha(a_title); title2.set_alpha(a_title)
            for i, st in enumerate(sums):
                st.set_alpha(smooth((T - 62.6 - 0.9 * i) / 0.6))
            note.set_alpha(smooth((T - 67.0) / 0.8))

            w_.grab_frame()
            if fi % 300 == 0:
                print(f"{T:5.1f}s / {END}s", flush=True)
    plt.close(fig)


def main():
    if os.environ.get("PREVIEW"):
        render(os.path.join(TMP, "preview"))
        return
    raw = os.path.join(TMP, "raw.mp4")
    out = os.path.join(OUT, "hand_pitch_force.mp4")
    render(raw)
    # 물 입자 때문에 용량이 커서 화질 기준(CRF)으로 다시 인코딩
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-crf", "24",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    print("done:", out)


if __name__ == "__main__":
    main()
