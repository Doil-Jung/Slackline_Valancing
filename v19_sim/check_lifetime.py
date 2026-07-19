# -*- coding: utf-8 -*-
"""check_lifetime.py — fwe_preset 동일 조건 생존시간: 기준점 + 미세 초기조건 분포
사용:  python check_lifetime.py            # 기준점(0.5°) 1회 — 내 PC의 결정론 값 확인
       python check_lifetime.py dist       # 미세 지터 20케이스 분포 (오래 걸림)
       python check_lifetime.py 3 7        # 케이스 3~6 만 (병렬 분할용)
"""
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v19_bringup"))
sys.path.insert(0, HERE)
from sim_engine import SimEngine, MODE_FWE3

CAP = 120.0

# 기준 0.5° 주변 ±0.03° 지터 20케이스 (재현 가능하게 고정 수열)
JIT = [0.0, 0.005, -0.005, 0.010, -0.010, 0.015, -0.015, 0.020, -0.020,
       0.025, -0.025, 0.030, -0.030, 0.008, -0.008, 0.012, -0.012,
       0.018, -0.018, 0.022]


def run(a0, t0):
    e = SimEngine()
    e.nominal.update(c_phi=0.008, RHO=0.95)
    e.real.update(enc_use_gyro=False, fwe_settle_s=0.0, fwe_adapt_T=False,
                  fwe_ff=True, fwe_fast_L=False, w_noload_dps=462.0,
                  fwe_online=True)
    e.kp_servo, e.kd_servo = 80.0, 1.3
    e.estimator = "enc"
    e.ctrl_mode = MODE_FWE3
    e.x0 = dict(phi0=0.0, alpha0=a0, theta0=t0)
    e.build()
    e.real["fwe_gamma_cal"] = e.g["mdl"]["fwe"]["GAMMA_CYCLE"] * (18.6/28.91)
    e.reset()
    surv = CAP
    for i in range(int(CAP/e.p["DT"])):
        e.control_step()
        if e.fallen:
            surv = e.data.time
            break
    return surv, e.fwe["cycles"]


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:                      # 기준점 1회
        s, c = run(0.5, 0.5)
        print(f"기준점(0.5°,0.5°): 생존 {s:.2f}s cycles {c}   (클라우드 참조값 56.45s/133)")
    else:
        lo, hi = (0, len(JIT)) if args[0] == "dist" else (int(args[0]), int(args[1]))
        for k in range(lo, min(hi, len(JIT))):
            a0 = 0.5 + JIT[k]
            s, c = run(a0, a0)
            print(f"case{k:02d} a0={a0:.3f}: 생존 {s:6.2f}s cycles {c}", flush=True)
