# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
MuJoCo eps 스캔: 비정지 초기조건에서 ε*_gen 검증
================================================
ε를 넓은 범위로 스캔하여 x_CoM 최대값이 최소인 ε을 찾고,
해석적 ε*_gen과 비교한다. + 스캔 그래프 (fit_eps_star.py 형태)
"""
import numpy as np, mujoco, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (
    generate_mjcf, get_state,
    M1, M2_EFF, LC1, LC2_EFF, I2_EFF, L1, GRAV, DT
)

g=GRAV; m2=M2_EFF; Mt=M1+m2; I1=M1*L1**2/12
p1=M1*LC1+m2*L1; p2=m2*LC2_EFF; p3=m2*L1*LC2_EFF; ps=p1+p2; h=ps/Mt
T_inv=np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
J_aa=M1*LC1**2+I1+m2*L1**2; J_tt=m2*LC2_EFF**2+I2_EFF; J_at=p3

def mode_analysis(R):
    M=np.array([[Mt*R**2,R*p1,R*p2],[R*p1,J_aa,J_at],[R*p2,J_at,J_tt]])
    G=np.diag([-Mt*g*R, p1*g, p2*g])
    Mp=T_inv.T@M@T_inv; Gp=T_inv.T@G@T_inv
    A2=np.linalg.solve(Mp[:2,:2], Gp[:2,:2])
    ev,V=np.linalg.eig(A2)
    idx=np.argsort(ev.real); ev=ev[idx].real; V=V[:,idx].real
    Vi=np.linalg.inv(V)
    return np.sqrt(abs(ev[0])), np.sqrt(abs(ev[1])), V[:,0], V[:,1], Vi[0], Vi[1]

def eps_star_stat(R):
    _,_,v1,_,_,_=mode_analysis(R)
    r=v1[0]/v1[1]; return -h/(r*R+h)

def eps_star_gen(R, phi_i, beta_i, pdot_i, bdot_i):
    _,lam,_,_,_,u2=mode_analysis(R)
    q_pred=np.array([phi_i+h*beta_i/R+pdot_i/lam, bdot_i/lam])
    e=np.array([h/R,-1.0])
    num=u2@q_pred; den=beta_i*(u2@e)
    if abs(den)<1e-15: return np.nan
    return -num/den

def run_sim_nonstat(R, eps, phi_i, beta_i, pdot_i, bdot_i, sim_t=2.0):
    """비정지 초기조건에서 Wait 시뮬레이션, xmax 반환"""
    xml=generate_mjcf(R)
    model=mujoco.MjModel.from_xml_string(xml)
    data=mujoco.MjData(model)
    ra=model.jnt_qposadr[model.joint('phi_Y').id]
    aa=model.jnt_qposadr[model.joint('ankle').id]
    ha=model.jnt_qposadr[model.joint('hip').id]
    try: rb=model.jnt_qposadr[model.joint('rope_b_Y').id]
    except: rb=None
    va=model.jnt_dofadr[model.joint('phi_Y').id]
    va_a=model.jnt_dofadr[model.joint('ankle').id]
    va_h=model.jnt_dofadr[model.joint('hip').id]
    try: vb=model.jnt_dofadr[model.joint('rope_b_Y').id]
    except: vb=None

    # 접기 후 위치 (가설→MuJoCo)
    phi_after = phi_i + h*(1+eps)*beta_i/R
    beta_after = -eps*beta_i
    phi_mj = -phi_after
    alpha_after = beta_after  # δ=0

    data.qpos[ra] = phi_mj
    data.qpos[aa] = alpha_after - phi_mj
    data.qpos[ha] = 0
    if rb: data.qpos[rb] = phi_mj

    # 속도
    phi_dot_mj = -pdot_i
    alpha_dot = bdot_i
    data.qvel[va] = phi_dot_mj
    data.qvel[va_a] = alpha_dot - phi_dot_mj
    data.qvel[va_h] = 0
    if vb: data.qvel[vb] = phi_dot_mj

    mujoco.mj_forward(model, data)

    xmax = 0
    for i in range(int(sim_t/DT)):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > 1.4:
            return 9999.0
        phi=x[0]; alpha=x[1]; theta=x[2]
        delta=theta-alpha; delta_d=x[5]-x[4]
        data.ctrl[0] = np.clip(10000*(0-delta)-200*delta_d, -50000, 50000)
        beta_v = (p1*alpha+p2*theta)/(Mt*h)
        xcom = abs(-R*phi + h*beta_v)
        if xcom > xmax: xmax = xcom
        mujoco.mj_step(model, data)
    return xmax

# ============================================================
# 테스트 케이스 (소각 범위 내, 작은 초기조건)
# ============================================================
test_cases = [
    # (label, R, phi_deg, beta_deg, pdot_deg_s, bdot_deg_s)
    ("R=0.7, phi=0.3, bdot=1",    0.7, 0.3, 1.5, 0.0, 1.0),
    ("R=0.7, pdot=2, bdot=1",     0.7, 0.0, 1.5, 2.0, 1.0),
    ("R=1.0, phi=0.3, bdot=1",    1.0, 0.3, 1.5, 0.0, 1.0),
    ("R=1.0, pdot=2, bdot=1",     1.0, 0.0, 1.5, 2.0, 1.0),
]

print("=" * 70)
print("  MuJoCo ε 스캔 — 비정지 ε*_gen 검증")
print("=" * 70)

fig, axes = plt.subplots(len(test_cases), 1, figsize=(14, 5*len(test_cases)))
if len(test_cases) == 1: axes = [axes]

for tc_idx, (label, R, pd, bd, pdd, bdd) in enumerate(test_cases):
    pi=np.radians(pd); bi=np.radians(bd)
    pdi=np.radians(pdd); bdi=np.radians(bdd)
    
    es = eps_star_stat(R)
    eg = eps_star_gen(R, pi, bi, pdi, bdi)
    
    print(f"\n  [{tc_idx+1}] {label}")
    print(f"    ε*_stat={es:.4f}, ε*_gen={eg:.4f}")
    
    # 스캔 범위: ε*_gen 주변 넓게
    center = eg
    eps_range = np.arange(max(0.2, center-2.0), center+2.5, 0.05)
    
    print(f"    스캔 범위: [{eps_range[0]:.2f}, {eps_range[-1]:.2f}], {len(eps_range)}개")
    
    results = []
    for e_scan in eps_range:
        xm = run_sim_nonstat(R, e_scan, pi, bi, pdi, bdi, sim_t=2.0)
        results.append((e_scan, xm * 1000))  # mm
    
    eps_arr = [r[0] for r in results]
    xmax_arr = [r[1] for r in results]
    
    # 수치 최적
    best = min(results, key=lambda x: x[1])
    eps_mj = best[0]
    xmax_mj = best[1]
    
    # ε*_gen에서의 xmax
    xmax_at_gen = run_sim_nonstat(R, eg, pi, bi, pdi, bdi) * 1000
    xmax_at_stat = run_sim_nonstat(R, es, pi, bi, pdi, bdi) * 1000
    
    err_gen = (eps_mj - eg) / eg * 100 if eg != 0 else 999
    
    print(f"    수치 최적: ε_MJ={eps_mj:.3f}, xmax={xmax_mj:.1f}mm")
    print(f"    ε*_gen에서: xmax={xmax_at_gen:.1f}mm")
    print(f"    ε*_stat에서: xmax={xmax_at_stat:.1f}mm")
    print(f"    오차: {err_gen:+.1f}%")
    
    # 그래프
    ax = axes[tc_idx]
    
    # 발산 표시: xmax > 500mm → 빨간 배경
    for i in range(len(eps_arr)):
        if xmax_arr[i] > 500:
            ax.axvspan(eps_arr[i]-0.025, eps_arr[i]+0.025,
                       color='#FFCDD2', alpha=0.3, zorder=0)
    
    # xmax 스캔 결과
    ax.bar(eps_arr, [min(x, 600) for x in xmax_arr], width=0.04,
           color=['#E63946' if x > 500 else '#457B9D' for x in xmax_arr],
           alpha=0.7, zorder=1)
    
    # ε*_gen 수직선
    ax.axvline(eg, color='#E63946', lw=2.5, ls='-', zorder=3,
               label=f'ε*_gen = {eg:.3f} (해석)')
    # ε*_stat 수직선
    ax.axvline(es, color='#264653', lw=2, ls='--', zorder=3,
               label=f'ε*_stat = {es:.3f} (정지)')
    # MuJoCo 최적
    ax.axvline(eps_mj, color='#2A9D8F', lw=2, ls=':', zorder=3,
               label=f'ε_MJ = {eps_mj:.3f} (수치 최적)')
    
    # 500mm 한계선
    ax.axhline(500, color='gray', lw=0.8, ls=':', alpha=0.5)
    ax.text(eps_arr[-1], 520, '발산 한계', fontsize=8, ha='right', color='gray')
    
    ax.set_ylabel('max|x_CoM| [mm]', fontsize=11)
    ax.set_ylim(0, 650)
    ax.set_title(f'{label}  |  ε*_gen={eg:.3f}, ε_MJ={eps_mj:.3f}, 오차={err_gen:+.1f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2, axis='y')
    
    if tc_idx == len(test_cases) - 1:
        ax.set_xlabel('ε (접기 비율)', fontsize=12)

fig.suptitle('MuJoCo ε 스캔: 비정지 ε*_gen 검증\n'
             '파란 막대: 안정 (2초 생존), 빨간 막대: 발산 (쓰러짐)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'eps_gen_scan_mujoco.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n  그래프 저장: {save_path}")
print("  완료!")
