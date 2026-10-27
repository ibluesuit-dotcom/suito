"""수중 돌핀킥 2D 모델 (옆에서 본 시점, 단위: m, s).

몸 전체를 손끝(s=0)부터 발끝(s=L)까지 이어진 사슬로 보고,
머리 쪽에서 발끝 쪽으로 전달되는 물결(진행파) y = A(s)·sin(k·s − ω·t) 로 움직인다.
진폭은 가슴에서 가장 작고 발끝에서 가장 크다.
좌표: 수영자는 +x 로 전진, 카메라가 몸을 따라가므로 물은 −x 로 U 속도로 흐른다.
"""
import numpy as np

U = 2.2              # 수중 돌핀킥 유영 속도 (엘리트 수준, m/s)
FREQ = 2.2           # 킥 주파수 (Hz)
OMEGA = 2 * np.pi * FREQ
WAVELEN = 2.3        # 몸을 따라 전달되는 물결의 파장 (m)
K = 2 * np.pi / WAVELEN

# 몸의 마디 (손끝부터 거리 s, m)
NAMES = ["손끝", "손목", "팔꿈치", "어깨", "골반", "무릎", "발목", "발끝"]
S = np.array([0.0, 0.19, 0.47, 0.80, 1.35, 1.80, 2.23, 2.45])
SEG_LEN = np.diff(S)
# 각 마디의 진폭 (m): 가슴 근처가 가장 작고, 골반 → 무릎 → 발끝으로 커진다
AMP_S = np.array([0.0, 0.19, 0.47, 0.80, 1.05, 1.35, 1.80, 2.23, 2.45])
AMP_A = np.array([0.035, 0.03, 0.025, 0.02, 0.02, 0.06, 0.12, 0.21, 0.25])
# 분절별 물을 받는 폭 계수 (손·팔은 유선형으로 모아 작음, 두 발을 모은 발이 가장 큼)
WIDTH = np.array([0.15, 0.15, 0.2, 0.55, 0.7, 0.55, 1.5])
SEG_NAMES = ["손", "아래팔", "위팔", "몸통", "허벅지", "정강이", "발"]


def amp(s):
    return np.interp(s, AMP_S, AMP_A)


WARP = 0.2          # 다운킥을 업킥보다 빠르고 짧게 (시간 비대칭)


def centerline_y(s, t):
    th = OMEGA * t + WARP * np.sin(OMEGA * t - K * 2.23)
    return amp(s) * np.sin(K * s - th)


def joints(t):
    """마디 위치 (손끝→발끝). 분절 길이를 지키도록 물결의 기울기를 따라 이어 붙인다.
    어깨(s=0.80)를 기준점으로 앞뒤로 쌓는다."""
    ia = 3
    P = np.zeros((len(S), 2))
    P[ia] = [-S[ia], centerline_y(S[ia], t)]
    for i in range(ia, len(S) - 1):          # 어깨 → 발끝
        sm = 0.5 * (S[i] + S[i + 1])
        dy = (centerline_y(S[i + 1], t) - centerline_y(S[i], t)) / SEG_LEN[i]
        ang = np.arctan(dy)
        if i == len(S) - 2:                    # 발: 발끝을 쭉 편(포인트) 채 물결을 따라 채찍처럼
            ang = ang - 0.05
        P[i + 1] = P[i] + SEG_LEN[i] * np.array([-np.cos(ang), np.sin(ang)])
    for i in range(ia, 0, -1):                 # 어깨 → 손끝
        dy = (centerline_y(S[i], t) - centerline_y(S[i - 1], t)) / SEG_LEN[i - 1]
        ang = np.arctan(dy)
        P[i - 1] = P[i] + SEG_LEN[i - 1] * np.array([np.cos(ang), -np.sin(ang)])
    return P


def segment_forces(t, dt=1e-3, nsamp=6):
    """물이 각 분절에 가하는 힘 (압력 항력, ρ/2·Cd 를 1 로 둔 상대값, N 비례)."""
    P0, P1, P = joints(t - dt), joints(t + dt), joints(t)
    F, C = [], []
    for i in range(len(S) - 1):
        a, b = P[i], P[i + 1]
        va = (P1[i] - P0[i]) / (2 * dt); vb = (P1[i + 1] - P0[i + 1]) / (2 * dt)
        d = b - a; ln = np.linalg.norm(d); n = np.array([-d[1], d[0]]) / ln
        f = np.zeros(2)
        for k in range(nsamp):
            u = (k + 0.5) / nsamp
            v = va * (1 - u) + vb * u
            vrel = np.array([-U, 0.0]) - v
            vn = vrel @ n
            f += WIDTH[i] * abs(vn) * vn * n * (ln / nsamp)
        F.append(f); C.append((a + b) / 2)
    return np.array(F), np.array(C)


def toe_down(t, h=0.01):
    return joints(t + h)[-1][1] < joints(t - h)[-1][1]


def kernel(px, py, vx, vy, g, delta):
    dx = px[:, None] - vx[None, :]
    dy = py[:, None] - vy[None, :]
    r2 = dx * dx + dy * dy + delta * delta
    f = g[None, :] / (2 * np.pi * r2)
    return (-dy * f).sum(1), (dx * f).sum(1)


class Wake:
    """정강이·발이 물에 준 운동량만큼 와류쌍을 넣는 와류법 (임펄스 보존)."""

    SHED_DT = 1 / 60

    def __init__(self, gain=0.02, delta=0.05, sep=0.10):
        self.gain, self.delta, self.sep = gain, delta, sep
        self.vx = np.zeros(0); self.vy = np.zeros(0); self.g = np.zeros(0)
        self.acc = {}; self.acc_t = 0.0; self.t = 0.0

    def velocity(self, px, py, delta=None):
        if len(self.g) == 0:
            return np.zeros_like(px), np.zeros_like(py)
        return kernel(px, py, self.vx, self.vy, self.g, delta or self.delta)

    def step(self, dt):
        t1 = self.t + dt
        if len(self.g):
            u, v = self.velocity(self.vx, self.vy)
            self.vx = self.vx + (u - U) * dt
            self.vy = self.vy + v * dt
        F, C = segment_forces(t1)
        for seg in (5, 6):
            imp, _ = self.acc.get(seg, (np.zeros(2), C[seg]))
            self.acc[seg] = (imp - F[seg] * self.gain * dt, C[seg])
        self.acc_t += dt
        nx, ny, ng = [], [], []
        if self.acc_t >= self.SHED_DT:
            for imp, c in self.acc.values():
                mag = np.linalg.norm(imp)
                if mag < 1e-6:
                    continue
                sdir = np.array([-imp[1], imp[0]]) / mag
                gam = mag / self.sep
                for sign in (+1, -1):
                    p = c + sign * 0.5 * self.sep * sdir
                    nx.append(p[0]); ny.append(p[1]); ng.append(sign * gam)
            self.acc = {}; self.acc_t = 0.0
        self.vx = np.append(self.vx, nx); self.vy = np.append(self.vy, ny); self.g = np.append(self.g, ng)
        self.t = t1
        keep = self.vx > -6.5
        self.vx, self.vy, self.g = self.vx[keep], self.vy[keep], self.g[keep]


# ----------------------------------------------------------------- 스타트 구간 속도 (1차원 운동 방정식)
M_EFF = 90.0          # 몸무게 75 kg + 함께 가속되는 물 (부가 질량)
K_GLIDE = 15.0        # 유선형 글라이드 항력 계수 (N·s²/m²), 2 m/s 에서 약 60 N
K_DOLPHIN = 19.0      # 돌핀킥 중 (몸을 움직여 항력이 조금 커짐)
K_SURFACE = 26.0      # 수면 수영 (조파 저항 = 물결을 만드는 저항이 더해짐)
T_DOLPHIN = K_DOLPHIN * U ** 2          # 수중 돌핀킥 평균 추진력 (정상 속도 U)
T_FREE = K_SURFACE * 2.0 ** 2           # 수면 자유형 평균 추진력 (정상 속도 2.0 m/s)
ENTRY_V = 4.8         # 입수 속도 (m/s)


def start_profile(dt=0.002, t_end=8.0, breakout_x=13.0, glide_time=0.45, dive_x=3.3, dive_t=1.05):
    """스타트 → 입수 → 글라이드 → 돌핀킥 → 브레이크아웃 → 자유형. (시각, 거리, 속도, 단계)."""
    out = []
    t = 0.0
    push_t = 0.65
    while t < dive_t:                       # 출발대에서 밀기 → 비행 (입수 직전 속도 약 ENTRY_V)
        if t < push_t:
            k = t / push_t
            x = 0.4 * k * k
            v = 0.95 * ENTRY_V * (3 * k * k - 2 * k ** 3)
        else:
            k = (t - push_t) / (dive_t - push_t)
            x = 0.4 + (dive_x - 0.4) * k
            v = ENTRY_V * (0.95 + 0.05 * k)
        out.append((t, x, v, "출발·비행")); t += dt
    x, v = dive_x, ENTRY_V
    t_entry = t
    while t < t_end:
        if t - t_entry < glide_time:
            ph, a = "글라이드", -K_GLIDE * v * v / M_EFF
        elif x < breakout_x:
            ph, a = "돌핀킥", (T_DOLPHIN - K_DOLPHIN * v * v) / M_EFF
        else:
            ph, a = "자유형", (T_FREE - K_SURFACE * v * v) / M_EFF
        v += a * dt; x += v * dt; t += dt
        out.append((t, x, v, ph))
    return out
