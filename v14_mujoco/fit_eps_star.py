# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""MuJoCo 수치 스캔으로 eps 최적점 찾기 → 해석 eps*와 비교"""
import numpy as np, mujoco, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (
    generate_mjcf, get_state,
    M1, M2_EFF, LC1, LC2_EFF, I2_EFF, L1, GRAV, DT
)

g=GRAV; m2=M2_EFF; Mt=M1+m2; I1=M1*L1**2/12
p1=M1*LC1+m2*L1; p2=m2*LC2_EFF; p3=m2*L1*LC2_EFF; ps=p1+p2; h=ps/Mt
T_inv=np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
J_aa=M1*LC1**2+I1+m2*L1**2; J_tt=m2*LC2_EFF**2+I2_EFF; J_at=p3
beta0=np.radians(2)

def eps_star_analytical(R):
    M=np.array([[Mt*R**2,R*p1,R*p2],[R*p1,J_aa,J_at],[R*p2,J_at,J_tt]])
    G=np.diag([-Mt*g*R, p1*g, p2*g])
    Mp=T_inv.T@M@T_inv; Gp=T_inv.T@G@T_inv
    A2=np.linalg.solve(Mp[:2,:2], Gp[:2,:2])
    ev,V=np.linalg.eig(A2)
    idx=np.argsort(ev.real); V=V[:,idx].real
    r=V[0,0]/V[1,0]
    return -h/(r*R+h)

def run_sim(R, eps, sim_t=2.0):
    xml=generate_mjcf(R)
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    ra=model.jnt_qposadr[model.joint('phi_Y').id]
    aa=model.jnt_qposadr[model.joint('ankle').id]
    ha=model.jnt_qposadr[model.joint('hip').id]
    try: rb=model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb=None
    phi0=-h*(1+eps)*beta0/R; bn=-eps*beta0
    data.qpos[ra]=phi0; data.qpos[aa]=bn-phi0; data.qpos[ha]=0
    if rb: data.qpos[rb]=phi0
    data.qvel[:]=0; mujoco.mj_forward(model,data)
    xmax=0
    for i in range(int(sim_t/DT)):
        x=get_state(model,data)
        if np.any(np.isnan(x)) or abs(x[1])>1.4: return 9999.0
        phi=x[0];alpha=x[1];theta=x[2]
        delta=theta-alpha; delta_d=x[5]-x[4]
        data.ctrl[0]=np.clip(10000*(0-delta)-200*delta_d, -50000, 50000)
        beta_v=(p1*alpha+p2*theta)/(Mt*h)
        xcom=abs(-R*phi+h*beta_v)
        if xcom>xmax: xmax=xcom
        mujoco.mj_step(model,data)
    return xmax

print("=" * 70)
print("  MuJoCo eps scan vs analytical eps*")
print("=" * 70)
print("  balance: min max|x_CoM| over 2s")
print("  eps scan: step 0.05 (coarse) -> 0.02 (fine)")
print()
hdr = f"  {'R':>4} | {'eps*_an':>8} | {'eps*_MJ':>8} | {'err':>7} | {'xmax(an)':>9} | {'xmax(MJ)':>9}"
print(hdr)
print(f"  {'-'*65}")

for R in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    es_an = eps_star_analytical(R)
    
    # coarse scan
    eps_range1 = np.arange(0.5, 4.0, 0.1)
    results1 = [(e, run_sim(R, e)) for e in eps_range1]
    best1 = min(results1, key=lambda x: x[1])
    
    # fine scan
    lo = max(0.1, best1[0]-0.15)
    hi = best1[0]+0.16
    eps_range2 = np.arange(lo, hi, 0.02)
    results2 = [(e, run_sim(R, e)) for e in eps_range2]
    best2 = min(results2, key=lambda x: x[1])
    
    eps_mj = best2[0]
    xmax_mj = best2[1]*1000
    xmax_an = run_sim(R, es_an)*1000
    
    err = (eps_mj - es_an) / es_an * 100
    print(f"  {R:4.1f} | {es_an:8.3f} | {eps_mj:8.3f} | {err:+6.1f}% | {xmax_an:7.1f}mm | {xmax_mj:7.1f}mm")

print()
print("  done!")
