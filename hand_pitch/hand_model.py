"""손바닥을 얇은 판(날개 단면)으로 본 힘 모델.

좌표계: x = 진행 방향(앞), y = 손이 비스듬히 움직이는 방향(옆/아래).
손의 속도 v(물 기준), 손바닥이 향하는 방향 f(단위벡터).
물이 손에 주는 힘 = 항력 D(물의 상대 흐름 방향) + 양력 L(그에 수직, 손등 쪽).
계수는 손 모형 실험(Schleihauf 1979 등)의 모양을 따른 근사:
    CL(α) = 1.0·sin 2α,  CD(α) = 0.1 + 1.1·sin²α   (α = 받음각, 0~90°)
"""
import numpy as np


def coeffs(alpha):
    return 0.1 + 1.1 * np.sin(alpha) ** 2, 1.0 * np.sin(2 * alpha)


def hand_force(v, f):
    """v: 물 기준 손 속도, f: 손바닥이 향하는 방향. 반환: (D, L, 받음각)."""
    v = np.asarray(v, float); f = np.asarray(f, float) / np.linalg.norm(f)
    w = -v                                    # 손이 느끼는 물의 흐름
    sp = np.linalg.norm(w)
    if sp < 1e-9:
        return np.zeros(2), np.zeros(2), 0.0
    wh = w / sp
    # 물이 손바닥 쪽으로 들어와야 한다 (f·w < 0)
    s = -(wh @ f)
    alpha = np.arcsin(np.clip(abs(s), 0, 1))
    cd, cl = coeffs(alpha)
    q = sp * sp
    D = cd * q * wh
    perp = np.array([-wh[1], wh[0]])
    if perp @ (-f) < 0:                       # 양력은 손바닥 → 손등 쪽 (판에 수직한 압력 방향)
        perp = -perp
    if s < 0:                                 # 물이 손등 쪽으로 들어오면 양력 방향도 뒤집힌다
        perp = -perp
    L = cl * q * perp
    return D, L, alpha


def facing(psi):
    """손바닥 방향: psi=0 이면 정확히 뒤(-x)를 향함. +psi 는 +y 쪽으로 돌림."""
    return np.array([-np.cos(psi), np.sin(psi)])
