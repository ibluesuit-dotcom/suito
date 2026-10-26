"""평영 발차기 2D 모델 (위에서 내려다본 수평면).

좌표계: 수영자는 +x 방향(오른쪽)으로 전진, 카메라가 몸을 따라가므로 물은 -x 로 U 속도로 흐른다.
y 는 몸의 좌우 방향. 두 다리는 몸 중심선(y=0)에 대해 대칭.
관절 위치는 위에서 본 투영 좌표이므로, 무릎을 굽혀 다리가 아래로 향하면 투영 길이가 짧아진다.
"""
import numpy as np

U = 1.0            # 유영 속도
FREQ = 0.45        # 발차기 주기 (슬로모션 기준 Hz)
HIP_Y = 0.17
L1, L2, L3 = 1.0, 0.95, 0.42   # 실제 분절 길이 (투영과 무관한 3D 길이)

# 키프레임: (주기 위상, 무릎 x, 무릎 y, 발목 x, 발목 y, 발 방향각, 발목 꺾임)
#  발 방향각: 뒤꿈치→발끝 방향 (pi = 뒤쪽을 향함, pi/2 = 바깥쪽을 향함)
#  발목 꺾임: 0 = 포인트(쭉 편 발), 1 = 플렉스(발등을 정강이 쪽으로 당김)
KEYS = [
    (0.00, -1.00, 0.10, -1.97, 0.07, np.pi, 0.0),          # 글라이드: 다리 모으고 쭉
    (0.12, -0.98, 0.14, -1.80, 0.12, np.pi, 0.0),          # 리커버리 시작
    (0.26, -0.87, 0.29, -1.25, 0.29, 3.05, 0.05),          # 리커버리: 뒤꿈치를 엉덩이 쪽으로
    (0.34, -0.79, 0.37, -0.96, 0.40, 2.40, 0.45),          # 리커버리 끝: 발을 천천히 바깥으로 돌림
    (0.42, -0.76, 0.41, -0.88, 0.50, 1.45, 1.0),           # 캐치: 발목 플렉스, 발끝 바깥
    (0.48, -0.84, 0.48, -1.20, 0.82, 1.30, 1.0),           # 킥 1: 바깥·뒤로
    (0.56, -0.94, 0.42, -1.66, 0.78, 1.85, 0.95),          # 킥 2: 뒤로, 안쪽으로 돌기 시작
    (0.64, -1.00, 0.22, -1.93, 0.30, 2.55, 0.6),           # 킥 3: 안쪽으로 모으기
    (0.72, -1.00, 0.10, -1.97, 0.07, np.pi, 0.0),          # 다리 모음 → 글라이드
]


def phase_name(s):
    s = s % 1.0
    if 0.12 <= s < 0.42:
        return "리커버리", "#c7a6ff"
    if 0.42 <= s < 0.72:
        return "킥", "#ffb347"
    return "글라이드", "#9be7ff"


def _interp(s, ankle):
    """주기적 Catmull-Rom 보간."""
    keys = np.array(KEYS)
    if ankle == "pointed":                 # 발목을 당기지 못하는 경우: 발이 늘 뒤를 향함
        keys = keys.copy()
        keys[:, 5] = np.pi - 0.25 * (np.pi - keys[:, 5])
        keys[:, 6] = 0.1 * keys[:, 6]
    ts = keys[:, 0]
    n = len(ts)
    s = s % 1.0
    i = np.searchsorted(ts, s, side="right") - 1
    i0, i1 = i % n, (i + 1) % n
    im, ip = (i - 1) % n, (i + 2) % n
    t0 = ts[i0]
    t1 = ts[i1] + (1.0 if i1 <= i0 else 0.0)
    u = (s - t0) / (t1 - t0)
    P0, P1, P2, P3 = keys[im, 1:], keys[i0, 1:], keys[i1, 1:], keys[ip, 1:]
    u2, u3 = u * u, u * u * u
    return 0.5 * (2 * P1 + (-P0 + P2) * u + (2 * P0 - 5 * P1 + 4 * P2 - P3) * u2
                  + (-P0 + 3 * P1 - 3 * P2 + P3) * u3)


def leg(t, side=+1, ankle="flex"):
    """한쪽 다리 (side=+1 위쪽, -1 아래쪽): 고관절, 무릎, 발목(뒤꿈치), 발끝, 꺾임 정도."""
    kx, ky, ax_, ay, th, flex = _interp(t * FREQ, ankle)
    H = np.array([0.0, side * HIP_Y])
    K = np.array([kx, side * ky])
    A = np.array([ax_, side * ay])
    T = A + L3 * np.array([np.cos(th), side * np.sin(th)])
    return H, K, A, T, float(np.clip(flex, 0, 1))


def segment_forces(t, side=+1, ankle="flex", dt=1e-3, nsamp=8):
    """물이 각 분절(허벅지, 정강이, 발)에 가하는 힘 (압력 항력, 상대값)."""
    P0 = leg(t - dt, side, ankle)
    P1 = leg(t + dt, side, ankle)
    P = leg(t, side, ankle)
    true_len = (L1, L2, L3)
    forces, centers = [], []
    for s in range(3):
        a, b = P[s], P[s + 1]
        va = (P1[s] - P0[s]) / (2 * dt)
        vb = (P1[s + 1] - P0[s + 1]) / (2 * dt)
        d = b - a
        ln = max(np.linalg.norm(d), 1e-6)
        n = np.array([-d[1], d[0]]) / ln
        if s == 2:
            # 발: 플렉스해 발바닥이 뒤를 향할수록 수평면에서 물을 받는 면적이 커진다
            width = 0.25 + 1.35 * P[4]
            proj = 1.0
        else:
            width = 0.6 if s == 0 else 0.5
            proj = min(ln / true_len[s], 1.0)   # 투영 비율: 짧으면 분절이 아래로 서 있음
        F = np.zeros(2)
        for k in range(nsamp):
            u = (k + 0.5) / nsamp
            v = va * (1 - u) + vb * u
            vrel = np.array([-U, 0.0]) - v
            vn = vrel @ n
            f_plane = abs(vn) * vn * n                    # 수평면에 누운 판/원통
            f_bluff = np.linalg.norm(vrel) * vrel         # 아래로 선 원통: 모든 방향 항력
            F += width * (proj * f_plane + (1 - proj) * f_bluff) * (true_len[s] / nsamp)
        forces.append(F)
        centers.append((a + b) / 2)
    return np.array(forces), np.array(centers)


def kick_force(t, side=+1, ankle="flex"):
    """다리의 상대 운동만으로 생기는 힘 (몸이 그냥 미끄러질 때의 항력은 뺀다)."""
    F, C = segment_forces(t, side, ankle)
    return F, C


def total_fx(t, ankle="flex"):
    return sum(segment_forces(t, sd, ankle)[0][:, 0].sum() for sd in (+1, -1)) - glide_drag()


_GLIDE = None


def glide_drag():
    """다리를 모은 글라이드 자세에서의 기준 항력 (그래프 0점 = 글라이드)."""
    global _GLIDE
    if _GLIDE is None:
        t0 = 0.85 / FREQ
        _GLIDE = sum(segment_forces(t0, sd)[0][:, 0].sum() for sd in (+1, -1))
    return _GLIDE


def kernel(px, py, vx, vy, g, delta):
    dx = px[:, None] - vx[None, :]
    dy = py[:, None] - vy[None, :]
    r2 = dx * dx + dy * dy + delta * delta
    f = g[None, :] / (2 * np.pi * r2)
    return (-dy * f).sum(1), (dx * f).sum(1)


class Wake:
    """발(과 정강이)이 물에 가한 힘만큼 운동량을 물에 주입하는 와류법.

    매 스텝, 발이 물에 가하는 힘 Fw 에 대해 Γ·(s_y, -s_x) = Fw·dt 를 만족하는
    작은 와류쌍(+Γ, -Γ, 간격 s)을 발 위치에 놓는다 -> 2D 와류계의 임펄스 = 물이 받은 운동량.
    """

    SHED_DT = 1 / 30      # 이 시간마다 쌓인 운동량을 와류쌍으로 방출

    def __init__(self, ankle="flex", gain=0.09, delta=0.16, sep=0.3):
        self.ankle = ankle
        self.acc = {}
        self.acc_t = 0.0
        self.gain = gain
        self.delta = delta
        self.sep = sep
        self.vx = np.zeros(0); self.vy = np.zeros(0); self.g = np.zeros(0)
        self.t = 0.0

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
        for sd in (+1, -1):
            F, C = segment_forces(t1, sd, self.ankle)
            for seg in (1, 2):
                c = C[seg] if seg == 1 else leg(t1, sd, self.ankle)[2] * 0.4 + C[2] * 0.6
                imp, _ = self.acc.get((sd, seg), (np.zeros(2), c))
                self.acc[(sd, seg)] = (imp - F[seg] * self.gain * dt, c)   # 물이 받는 운동량
        self.acc_t += dt
        nx, ny, ng = [], [], []
        if self.acc_t >= self.SHED_DT:
            for (imp, c) in self.acc.values():
                mag = np.linalg.norm(imp)
                if mag < 1e-4:
                    continue
                sdir = np.array([-imp[1], imp[0]]) / mag      # Γ(s_y,-s_x) = 운동량
                gam = mag / self.sep
                p1 = c + 0.5 * self.sep * sdir
                p2 = c - 0.5 * self.sep * sdir
                nx += [p1[0], p2[0]]; ny += [p1[1], p2[1]]; ng += [gam, -gam]
            self.acc = {}
            self.acc_t = 0.0
        self.vx = np.append(self.vx, nx); self.vy = np.append(self.vy, ny); self.g = np.append(self.g, ng)
        self.t = t1
        keep = self.vx > -7.5
        self.vx, self.vy, self.g = self.vx[keep], self.vy[keep], self.g[keep]
