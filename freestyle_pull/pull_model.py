"""자유형 손 젓기(풀) 2D 모델 (옆에서 본 시점).

좌표계: 수영자는 +x 방향(오른쪽)으로 전진. 카메라가 몸을 따라가므로 물은 -x 로 U 속도로 흐른다.
원점 = 어깨, y 는 위쪽이 +. 관절 위치는 옆에서 본 투영 좌표다.
"""
import numpy as np

U = 0.7            # 유영 속도 (몸 기준 물의 흐름)
FREQ = 0.6         # 한 팔 스트로크 주기 (슬로모션 기준 Hz)
SURFACE_Y = 0.25
HAND_LEN = 0.19
TRUE_LEN = (0.33, 0.30, HAND_LEN)   # 위팔, 아래팔, 손 (실제 길이)
WIDTH = (0.35, 0.8, 1.7)            # 물을 받는 폭 계수: 손바닥이 가장 넓다

S_IN, S_OUT = 0.12, 0.62     # 물속 구간 (위상)

PHASES = [  # (시작, 끝, 이름, 색)
    (0.00, 0.16, "엔트리·뻗기", "#9be7ff"),
    (0.16, 0.30, "캐치", "#c7a6ff"),
    (0.30, 0.45, "풀 (당기기)", "#ffd166"),
    (0.45, 0.62, "푸시 (밀기)", "#ffb347"),
    (0.62, 1.00, "리커버리", "#a0aec0"),
]


def phase_of(s):
    s = s % 1.0
    for a, b, name, col in PHASES:
        if a <= s < b:
            return name, col
    return PHASES[-1][2], PHASES[-1][3]


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _underwater(q, style):
    """물속 경로 (q: 0=캐치 시작 앞, 1=출수). 손목·팔꿈치 위치."""
    c, sn = np.cos(np.pi * q), np.sin(np.pi * q)
    rise_w = 0.32 * _ss((q - 0.70) / 0.30)
    rise_e = 0.25 * _ss((q - 0.75) / 0.25)
    if style == "high":
        # 높은 팔꿈치: 팔꿈치는 어깨 높이 가까이 머물고 아래팔·손이 일찍 수직으로 선다
        W = np.array([0.05 + 0.58 * c, -0.52 * sn ** 1.1 + rise_w])
        E = np.array([0.04 + 0.30 * c, -0.04 - 0.12 * sn + rise_e])
    else:
        # 팔꿈치가 떨어짐: 팔꿈치가 먼저 가라앉고 아래팔이 누워 손바닥이 아래를 향한다
        W = np.array([0.05 + 0.58 * c, -0.44 * sn ** 1.6 + rise_w])
        E = np.array([0.04 + 0.30 * c, -0.04 - 0.30 * sn ** 0.8 + rise_e])
    return E, W


ENTRY_W = np.array([0.42, 0.33]); ENTRY_E = np.array([0.22, 0.36])


def _pose(s, style):
    s = s % 1.0
    if s < S_IN:                                   # 엔트리: 손끝부터 비스듬히 들어감
        v = _ss(s / S_IN)
        E0, W0 = _underwater(0.0, style)
        return ENTRY_E + (E0 - ENTRY_E) * v, ENTRY_W + (W0 - ENTRY_W) * v, "entry"
    if s < S_OUT:                                  # 물속: 천천히 잡고 점점 가속
        u = (s - S_IN) / (S_OUT - S_IN)
        q = u ** 1.6
        E, W = _underwater(q, style)
        return E, W, "water"
    v = (s - S_OUT) / (1 - S_OUT)                  # 리커버리 (물 밖)
    E1, W1 = _underwater(1.0, style)
    k = _ss(v)
    W = W1 + (ENTRY_W - W1) * k + np.array([0, 0.10 * np.sin(np.pi * v)])
    E = E1 + (ENTRY_E - E1) * k + np.array([0, 0.22 * np.sin(np.pi * v)])
    return E, W, "air"


def arm(t, style="high"):
    """어깨, 팔꿈치, 손목, 손끝."""
    s = t * FREQ
    E, W, where = _pose(s, style)
    fa = np.arctan2(W[1] - E[1], W[0] - E[0])       # 아래팔 방향
    th = fa
    if where == "air":
        # 출수 직후 세운 손을 리커버리 동안 천천히 아래팔 방향으로 되돌린다
        v = (s % 1.0 - S_OUT) / (1 - S_OUT)
        if v < 0.25:
            th_exit = arm(S_OUT / FREQ - 1e-6, style)
            d0 = th_exit[3] - th_exit[2]
            a0 = np.arctan2(d0[1], d0[0])
            th = a0 + np.angle(np.exp(1j * (fa - a0))) * _ss(v / 0.25)
        elif v > 0.75:
            # 엔트리 직전: 손끝을 앞·아래로 세워 입수 준비
            th = fa + np.angle(np.exp(1j * (-0.55 - fa))) * _ss((v - 0.75) / 0.25)
    if where != "air":
        # 엔트리: 손끝부터 비스듬히 앞·아래로, 출수: 손목이 먼저 나오고 손끝은 아래로 뒤따른다
        if where == "entry":
            blend = 1.0 - _ss(s % 1.0 / S_IN)
            target = -0.55
        else:
            q = ((s % 1.0 - S_IN) / (S_OUT - S_IN)) ** 1.6
            blend = _ss((q - 0.88) / 0.12)
            target = -1.85
        d = np.angle(np.exp(1j * (target - th)))
        th = th + blend * d
    S = np.zeros(2)
    T = W + HAND_LEN * np.array([np.cos(th), np.sin(th)])
    return S, E, W, T


def segment_forces(t, style="high", dt=1e-3, nsamp=8):
    """물이 각 분절(위팔, 아래팔, 손)에 가하는 힘 (압력 항력, 상대값). 물 밖은 0."""
    P0, P1, P = arm(t - dt, style), arm(t + dt, style), arm(t, style)
    forces, centers = [], []
    for s in range(3):
        a, b = P[s], P[s + 1]
        va = (P1[s] - P0[s]) / (2 * dt)
        vb = (P1[s + 1] - P0[s + 1]) / (2 * dt)
        d = b - a
        ln = max(np.linalg.norm(d), 1e-6)
        n = np.array([-d[1], d[0]]) / ln
        proj = 1.0 if s == 2 else min(ln / TRUE_LEN[s], 1.0)
        F = np.zeros(2)
        for k in range(nsamp):
            u = (k + 0.5) / nsamp
            p = a * (1 - u) + b * u
            if p[1] > SURFACE_Y:
                continue
            v = va * (1 - u) + vb * u
            vrel = np.array([-U, 0.0]) - v
            vn = vrel @ n
            f_plane = abs(vn) * vn * n
            f_bluff = 0.5 * np.linalg.norm(vrel) * vrel   # 옆으로 벌어져 짧게 보이는 분절
            F += WIDTH[s] * (proj * f_plane + (1 - proj) * f_bluff) * (TRUE_LEN[s] / nsamp)
        forces.append(F)
        centers.append((a + b) / 2)
    return np.array(forces), np.array(centers)


def total_fx(t, style="high"):
    return segment_forces(t, style)[0][:, 0].sum()


def kernel(px, py, vx, vy, g, delta):
    dx = px[:, None] - vx[None, :]
    dy = py[:, None] - vy[None, :]
    r2 = dx * dx + dy * dy + delta * delta
    f = g[None, :] / (2 * np.pi * r2)
    return (-dy * f).sum(1), (dx * f).sum(1)


class Wake:
    """손·아래팔이 물에 가한 운동량만큼 와류쌍을 넣는 와류법 (임펄스 보존)."""

    SHED_DT = 1 / 30

    def __init__(self, style="high", gain=0.12, delta=0.06, sep=0.12):
        self.style = style
        self.gain, self.delta, self.sep = gain, delta, sep
        self.vx = np.zeros(0); self.vy = np.zeros(0); self.g = np.zeros(0)
        self.acc = {}; self.acc_t = 0.0
        self.t = 0.0

    def velocity(self, px, py, delta=None):
        if len(self.g) == 0:
            return np.zeros_like(px), np.zeros_like(py)
        # 수면을 평평한 뚜껑으로 보고 거울상 와류(부호 반대)를 더한다
        vx = np.concatenate([self.vx, self.vx])
        vy = np.concatenate([self.vy, 2 * SURFACE_Y - self.vy])
        g = np.concatenate([self.g, -self.g])
        return kernel(px, py, vx, vy, g, delta or self.delta)

    def step(self, dt):
        t1 = self.t + dt
        if len(self.g):
            u, v = self.velocity(self.vx, self.vy)
            self.vx = self.vx + (u - U) * dt
            self.vy = np.minimum(self.vy + v * dt, SURFACE_Y - 0.03)
        F, C = segment_forces(t1, self.style)
        for seg in (1, 2):
            if C[seg][1] > SURFACE_Y:
                continue
            imp, _ = self.acc.get(seg, (np.zeros(2), C[seg]))
            self.acc[seg] = (imp - F[seg] * self.gain * dt, C[seg])
        self.acc_t += dt
        nx, ny, ng = [], [], []
        if self.acc_t >= self.SHED_DT:
            for imp, c in self.acc.values():
                mag = np.linalg.norm(imp)
                if mag < 1e-5:
                    continue
                sdir = np.array([-imp[1], imp[0]]) / mag
                gam = mag / self.sep
                for sign in (+1, -1):
                    p = c + sign * 0.5 * self.sep * sdir
                    nx.append(p[0]); ny.append(min(p[1], SURFACE_Y - 0.03)); ng.append(sign * gam)
            self.acc = {}; self.acc_t = 0.0
        self.vx = np.append(self.vx, nx); self.vy = np.append(self.vy, ny); self.g = np.append(self.g, ng)
        self.t = t1
        keep = self.vx > -2.6
        self.vx, self.vy, self.g = self.vx[keep], self.vy[keep], self.g[keep]
