# -*- coding: utf-8 -*-
"""
sweep_timing.py — FWE 사이클 시간표(T_fold, T_wait, T_extend, T_rest, 트리거) 스윕
====================================================================================
목적: params_v19 의 직관값(80/60/80/50ms, 0.4°)을 전물리 생존 지도로 대체.
공정성: 각 타이밍 점마다 이론 γ(T) 재계산 × 실행갭 보정비(γ*/γ_공칭 = 18.6/28.91)
        + 온라인 RLS 미세조정. 조건: enc 추정, 전현실화, c_φ=0.008, vmax=462.
사용: python sweep_timing.py axis          # 축별 스캔 (빠름)
      python sweep_timing.py grid out.csv  # 격자 스윕 (오래 걸림, CSV 저장)
"""
import sys, os, itertools, csv
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v19_bringup"))
sys.path.insert(0, HERE)
from sim_engine import SimEngine, MODE_FWE3

EXEC_GAP = 18.6 / 28.91          # 기준 타이밍에서 실측한 실행갭 보정비
BASE = dict(servo_on=True, sensor_on=True, enc_use_gyro=False,
            fwe_settle_s=0.0, fwe_adapt_T=False, fwe_ff=True, fwe_fast_L=False,
            w_noload_dps=462.0, fwe_online=True)


def run_point(tf, tw, te, tr, trig, a0=0.5, cap=15.0):
    e = SimEngine()
    e.real.update(**BASE)
    e.ctrl_mode = MODE_FWE3
    e.estimator = "enc"
    e.kp_servo, e.kd_servo = 80, 1.3
    e.nominal.update(c_phi=0.008, RHO=0.95, A_TRIGGER_DEG=trig,
                     T_FOLD=tf, T_WAIT=tw, T_EXTEND=te, T_REST=tr)
    e.x0 = dict(phi0=0.0, alpha0=a0, theta0=a0)
    e.build()
    gam_nom = e.g["mdl"]["fwe"]["GAMMA_CYCLE"]
    e.real["fwe_gamma_cal"] = gam_nom * EXEC_GAP     # 점별 γ (이론×실행갭)
    e.reset()
    t_surv, Am = cap, 0.0
    n = int(cap / e.p["DT"])
    for i in range(n):
        e.control_step()
        if i > 200:
            Am = max(Am, abs(e.risk_A(e.get_state())))
        if e.fallen:
            t_surv = e.data.time
            break
    return dict(tf=tf, tw=tw, te=te, tr=tr, trig=trig, a0=a0,
                surv=round(t_surv, 2), cyc=e.fwe["cycles"],
                Amax=round(float(np.rad2deg(Am)), 2),
                gam=round(gam_nom * EXEC_GAP, 1))


def axis_scan():
    b = dict(tf=0.08, tw=0.06, te=0.08, tr=0.05, trig=0.4)
    print(f"기준점: {b}  (생존 상한 15s)")
    r0 = run_point(**b)
    print(f"  기준: 생존 {r0['surv']}s cycles {r0['cyc']} |A|max {r0['Amax']}° γ={r0['gam']}")
    axes = dict(
        tf=[0.04, 0.06, 0.11, 0.15],
        tw=[0.02, 0.04, 0.107, 0.16],       # 0.107 = 이론 후보 λT_w=ln2
        te=[0.06, 0.11, 0.15, 0.20],        # 펴기 완료 여지
        tr=[0.01, 0.03, 0.10, 0.15],
        trig=[0.2, 0.8, 1.2],
    )
    for ax, vals in axes.items():
        print(f"--- {ax} 축 ---")
        for v in vals:
            kw = dict(b); kw[ax] = v
            r = run_point(**kw)
            print(f"  {ax}={v}: 생존 {r['surv']:5.2f}s cycles {r['cyc']:3d} "
                  f"|A|max {r['Amax']:6.2f}° γ={r['gam']}")


def grid_scan(out_path):
    grid = list(itertools.product(
        [0.04, 0.06, 0.08],          # tf  (축스캔: 짧을수록 좋음)
        [0.02, 0.04, 0.06],          # tw  (짧을수록 좋음)
        [0.09, 0.11, 0.13],          # te  (최적점 ~110ms 주변)
        [0.04, 0.06],                # tr  (최적점 ~50ms 주변)
        [0.4, 0.6],                  # trig
        [0.5],                       # a0
    ))
    print(f"격자 {len(grid)}점 시작 → {out_path}")
    with open(out_path, "w", newline="") as f:
        wtr = None
        for k, (tf, tw, te, tr, trig, a0) in enumerate(grid):
            r = run_point(tf, tw, te, tr, trig, a0=a0, cap=20.0)
            if wtr is None:
                wtr = csv.DictWriter(f, fieldnames=list(r.keys()))
                wtr.writeheader()
            wtr.writerow(r)
            f.flush()
            if k % 20 == 0:
                print(f"  [{k}/{len(grid)}] tf={tf} tw={tw} te={te} tr={tr} "
                      f"trig={trig} a0={a0} → {r['surv']}s")
    print("완료")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "axis"
    if mode == "axis":
        axis_scan()
    else:
        grid_scan(sys.argv[2] if len(sys.argv) > 2 else "sweep_timing.csv")
