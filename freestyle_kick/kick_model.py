"""자유형 발차기 2D 모델: 다리 운동학, 수력(압력 항력) 추정, 와류 흔적(vortex blob) 시뮬레이션.

좌표계: 수영자는 +x 방향(오른쪽)으로 전진. 카메라는 몸을 따라가므로 물은 -x 방향으로 U 속도로 흐른다.
y는 위쪽이 +. 엉덩이(고관절)는 원점.
"""
import numpy as np

L1, L2, L3 = 1.0, 0.95, 0.45   # 허벅지, 정강이, 발 길이
U = 1.3                         # 유영 속도 (몸 기준 물의 흐름 속도)
FREQ = 0.6                      # 발차기 주파수 (슬로모션 기준 Hz)
OMEGA = 2 * np.pi * FREQ
SURFACE_Y = 0.9

# 분절별 (폭 계수) — 발은 면적이 넓어 물을 잘 민다
SEG_WIDTH = (0.6, 0.55, 1.0)


def _thigh_shank(phi):
    a1 = 0.32 * np.sin(phi)
    w = np.angle(np.exp(1j * (phi + 0.4)))        # (-pi, pi]로 감싼 위상
    knee = 0.3 * np.maximum(0.0, np.cos(w * np.pi / 2 / 1.2)) ** 2
    return a1, a1 - knee


def joint_angles(t, ankle="flexible"):
    """각 분절의 절대 각도 (-x 축 기준, +면 아래로 기울어짐)."""
    phi = OMEGA * t
    a1, a2 = _thigh_shank(phi)
    h = 1e-3
    da2 = (_thigh_shank(phi + h)[1] - _thigh_shank(phi - h)[1]) / (2 * h)   # 정강이 각속도 (위상 기준)
    if ankle == "flexible":
        # 발끝을 편 상태. 물의 압력에 밀려 수동적으로 휜다:
        # 정강이가 내려가면(다운킥) 발등에 압력 -> 발끝이 위로, 올라가면(업킥) 발바닥에 압력 -> 아래로
        rel = -0.15 - 0.8 * np.tanh(1.5 * da2)
    else:
        rel = 0.85 - 0.10 * np.tanh(1.6 * da2)    # 발목이 뻣뻣해 발이 아래로 꺾여 있음
    return a1, a2, a2 + rel


def joints(t, ankle="flexible", hip=(0.0, 0.0)):
    a1, a2, a3 = joint_angles(t, ankle)
    H = np.array(hip, dtype=float)
    K = H + L1 * np.array([-np.cos(a1), -np.sin(a1)])
    A = K + L2 * np.array([-np.cos(a2), -np.sin(a2)])
    T = A + L3 * np.array([-np.cos(a3), -np.sin(a3)])
    return H, K, A, T


def segment_forces(t, ankle="flexible", dt=1e-3, nsamp=8):
    """각 분절에 물이 가하는 힘 (압력 항력 모델, 상대값).

    반환: forces[3] (각 분절 합력 벡터), centers[3] (분절 중심)
    """
    P0 = joints(t - dt, ankle)
    P1 = joints(t + dt, ankle)
    P = joints(t, ankle)
    forces, centers = [], []
    for s in range(3):
        a, b = P[s], P[s + 1]
        va = (P1[s] - P0[s]) / (2 * dt)
        vb = (P1[s + 1] - P0[s + 1]) / (2 * dt)
        d = b - a
        ln = np.linalg.norm(d)
        n = np.array([-d[1], d[0]]) / ln
        F = np.zeros(2)
        for k in range(nsamp):
            u = (k + 0.5) / nsamp
            v = va * (1 - u) + vb * u
            vrel = np.array([-U, 0.0]) - v        # 분절이 느끼는 물의 상대속도
            vn = vrel @ n
            F += SEG_WIDTH[s] * abs(vn) * vn * n * (ln / nsamp)
        forces.append(F)
        centers.append((a + b) / 2)
    return np.array(forces), np.array(centers)


def kernel(px, py, vx, vy, g, delta):
    """Rosenhead-Moore 정규화 점와류가 (px,py)에 유도하는 속도."""
    dx = px[:, None] - vx[None, :]
    dy = py[:, None] - vy[None, :]
    r2 = dx * dx + dy * dy + delta * delta
    f = g[None, :] / (2 * np.pi * r2)
    return (-dy * f).sum(1), (dx * f).sum(1)


class Wake:
    """발끝에서 떨어져 나가는 와류를 추적하는 간단한 와류법 시뮬레이션."""

    def __init__(self, ankle="flexible", k_gamma=0.2, delta=0.24):
        self.ankle = ankle
        self.k = k_gamma
        self.delta = delta
        self.vx = np.zeros(0); self.vy = np.zeros(0); self.g = np.zeros(0)
        self.gb_prev = None
        self.t = 0.0

    def bound(self, t):
        F, C = segment_forces(t, self.ankle)
        # 발에 걸리는 수직 하중 ~ 결합 순환 (Kutta-Joukowski: 왼쪽 흐름에서 위쪽 힘 <-> 반시계 순환)
        gb = self.k * (F[2, 1] + 0.4 * F[1, 1])
        return gb, C[2]

    def velocity(self, px, py, include_bound=True, delta=None):
        vx, vy, g = self.vx, self.vy, self.g
        if include_bound:
            gb, c = self.bound(self.t)
            vx = np.append(vx, c[0]); vy = np.append(vy, c[1]); g = np.append(g, gb)
        if len(g) == 0:
            return np.zeros_like(px), np.zeros_like(py)
        return kernel(px, py, vx, vy, g, delta or self.delta)

    def step(self, dt):
        t1 = self.t + dt
        gb, _ = self.bound(t1)
        if self.gb_prev is None:
            self.gb_prev = gb
        dg = -(gb - self.gb_prev)
        self.gb_prev = gb
        _, _, _, T = joints(t1, self.ankle)
        # 기존 와류 이동 (자유류 + 상호 유도)
        if len(self.g):
            u, v = self.velocity(self.vx, self.vy)
            self.vx = self.vx + (u - U) * dt
            self.vy = np.minimum(self.vy + v * dt, SURFACE_Y - 0.15)   # 수면 위로는 못 나감
        # 새 와류 방출 (발끝 바로 뒤)
        self.vx = np.append(self.vx, T[0] - 0.05)
        self.vy = np.append(self.vy, T[1])
        self.g = np.append(self.g, dg)
        self.t = t1
        keep = self.vx > -10.5
        self.vx, self.vy, self.g = self.vx[keep], self.vy[keep], self.g[keep]
