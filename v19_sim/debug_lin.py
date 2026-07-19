# -*- coding: utf-8 -*-
"""debug_lin.py — MuJoCo 6D 수치 선형화 vs model_v19 Ac 비교 (부호/동역학 정합 진단)"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v19_bringup"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mujoco
from sim_engine import SimEngine

eng = SimEngine()
eng.real.update(servo_on=False, sensor_on=False, f_phi=0.0, f_ankle=0.0, f_hip=0.0,
                armature=0.0, b_gear=0.0, cur_deadband=0)
eng.x0 = dict(phi0=0.0, alpha0=0.0, theta0=0.0)
eng.build()
m, d = eng.model, eng.data

# 수치 선형화: q̈ = f(q, q̇, u) 를 유한차분
def accel(q, v, u):
    d.qpos[:] = q; d.qvel[:] = v; d.ctrl[0] = u
    mujoco.mj_forward(m, d)
    return d.qacc.copy()

q0 = np.zeros(3); v0 = np.zeros(3); h = 1e-6
A_q = np.zeros((3, 3)); A_v = np.zeros((3, 3)); B_u = np.zeros(3)
for i in range(3):
    dq = np.zeros(3); dq[i] = h
    A_q[:, i] = (accel(q0+dq, v0, 0) - accel(q0-dq, v0, 0)) / (2*h)
    A_v[:, i] = (accel(q0, v0+dq, 0) - accel(q0, v0-dq, 0)) / (2*h)
B_u = (accel(q0, v0, h) - accel(q0, v0, -h)) / (2*h)

# 좌표 변환: 모델 x=[φ,α,θ] = T q,  q=[q_phi,q_ankle,q_hip]
T = np.array([[-1, 0, 0], [1, 1, 0], [1, 1, 1]], float)
Ti = np.linalg.inv(T)
Aq_m = T @ A_q @ Ti
Av_m = T @ A_v @ Ti
Bu_m = T @ B_u

Ac = eng.g["mdl"]["Ac"]; Bc = eng.g["mdl"]["Bc"]
np.set_printoptions(precision=3, suppress=True, linewidth=140)
print("MuJoCo  d(q̈)/dq (모델좌표):"); print(Aq_m)
print("model_v19 Ac[3:,:3]:");        print(Ac[3:, :3])
print("MuJoCo  d(q̈)/dq̇:");           print(Av_m)
print("model_v19 Ac[3:,3:]:");        print(Ac[3:, 3:])
print("MuJoCo  B:", Bu_m)
print("model   B:", Bc[3:, 0])
ev_mj = np.linalg.eigvals(np.block([[np.zeros((3,3)), np.eye(3)], [Aq_m, Av_m]]))
ev_md = np.linalg.eigvals(Ac)
print("고유값 MuJoCo:", np.sort_complex(ev_mj))
print("고유값 model :", np.sort_complex(ev_md))
