# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
"""
ε 스캔: x_CoM(t) 시계열 전체를 보여주는 그래프
히트맵 + 대표 궤적 오버레이
"""
import numpy as np, mujoco, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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
    _,_,v1,_,_,_=mode_analysis(R); r=v1[0]/v1[1]; return -h/(r*R+h)

def eps_star_gen(R, phi_i, beta_i, pdot_i, bdot_i):
    _,lam,_,_,_,u2=mode_analysis(R)
    q_pred=np.array([phi_i+h*beta_i/R+pdot_i/lam, bdot_i/lam])
    e=np.array([h/R,-1.0])
    num=u2@q_pred; den=beta_i*(u2@e)
    if abs(den)<1e-15: return np.nan
    return -num/den

def run_sim_trace(R, eps, phi_i, beta_i, pdot_i, bdot_i, sim_t=2.0, rec_dt=0.005):
    """x_CoM 시계열 전체 반환"""
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

    phi_after = phi_i + h*(1+eps)*beta_i/R
    beta_after = -eps*beta_i
    phi_mj = -phi_after; alpha_after = beta_after
    data.qpos[ra]=phi_mj; data.qpos[aa]=alpha_after-phi_mj; data.qpos[ha]=0
    if rb: data.qpos[rb]=phi_mj
    phi_dot_mj=-pdot_i; alpha_dot=bdot_i
    data.qvel[va]=phi_dot_mj; data.qvel[va_a]=alpha_dot-phi_dot_mj
    data.qvel[va_h]=0
    if vb: data.qvel[vb]=phi_dot_mj
    mujoco.mj_forward(model,data)

    rec_step = max(1, int(rec_dt/DT))
    n=int(sim_t/DT)
    times=[]; xcoms=[]
    for i in range(n):
        x=get_state(model,data)
        if np.any(np.isnan(x)) or abs(x[1])>1.4: break
        phi=x[0]; alpha=x[1]; theta=x[2]
        delta=theta-alpha; delta_d=x[5]-x[4]
        data.ctrl[0]=np.clip(10000*(0-delta)-200*delta_d,-50000,50000)
        if i%rec_step==0:
            beta_v=(p1*alpha+p2*theta)/(Mt*h)
            xcom=(-R*phi+h*beta_v)*1000
            times.append(data.time); xcoms.append(xcom)
        mujoco.mj_step(model,data)
    return np.array(times), np.array(xcoms)

# ============================================================
# 2개 대표 케이스
# ============================================================
cases = [
    ("R=0.7, phi=0.3deg, bdot=1deg/s", 0.7, 0.3, 1.5, 0.0, 1.0),
    ("R=1.0, pdot=2deg/s, bdot=1deg/s", 1.0, 0.0, 1.5, 2.0, 1.0),
]

for case_idx, (label, R, pd, bd, pdd, bdd) in enumerate(cases):
    pi=np.radians(pd); bi=np.radians(bd)
    pdi=np.radians(pdd); bdi=np.radians(bdd)
    es = eps_star_stat(R)
    eg = eps_star_gen(R, pi, bi, pdi, bdi)
    
    print(f"[{case_idx+1}] {label}: eps*_gen={eg:.3f}, eps*_stat={es:.3f}")
    
    # ε 스캔 범위
    eps_lo = max(0.3, eg - 1.5)
    eps_hi = eg + 1.5
    eps_scan = np.arange(eps_lo, eps_hi, 0.05)
    n_eps = len(eps_scan)
    
    # 시계열 수집
    sim_t = 2.0; rec_dt = 0.005
    n_time = int(sim_t / rec_dt)
    heatmap = np.full((n_eps, n_time), np.nan)
    
    for i, e in enumerate(eps_scan):
        t, xc = run_sim_trace(R, e, pi, bi, pdi, bdi, sim_t, rec_dt)
        L = min(len(xc), n_time)
        heatmap[i, :L] = xc[:L]
        # 발산 후 NaN → 마지막 값으로 채움 (또는 큰 값)
        if L < n_time:
            heatmap[i, L:] = 500 if xc[-1] > 0 else -500
    
    # 그래프
    fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                              gridspec_kw={'height_ratios': [2, 1]})
    
    # --- (1) 히트맵 ---
    ax = axes[0]
    t_arr = np.arange(n_time) * rec_dt
    
    vmax = 200
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(heatmap, aspect='auto', origin='lower',
                   extent=[0, sim_t, eps_scan[0], eps_scan[-1]],
                   cmap='RdBu_r', norm=norm, interpolation='bilinear')
    
    # ε*_gen 수평선
    ax.axhline(eg, color='#E63946', lw=2.5, ls='-',
               label=f'eps*_gen = {eg:.3f} (해석)')
    ax.axhline(es, color='#264653', lw=2, ls='--',
               label=f'eps*_stat = {es:.3f} (정지)')
    
    cb = plt.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('x_CoM [mm]', fontsize=11)
    
    ax.set_ylabel('epsilon (접기 비율)', fontsize=12)
    ax.set_title(f'{label}\nx_CoM(t) 히트맵 — eps*_gen 주변 스캔',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right',
              facecolor='white', edgecolor='gray', framealpha=0.9)
    
    # --- (2) 대표 궤적 오버레이 ---
    ax2 = axes[1]
    
    # 선택할 ε 값들
    picks = [
        (eg,           '#E63946', 2.5, '-',  f'eps*_gen={eg:.3f}'),
        (es,           '#264653', 2.0, '--', f'eps*_stat={es:.3f}'),
        (eg - 0.3,     '#F4A261', 1.5, '-',  f'eps={eg-0.3:.2f} (gen-0.3)'),
        (eg + 0.3,     '#2A9D8F', 1.5, '-',  f'eps={eg+0.3:.2f} (gen+0.3)'),
        (eg - 0.7,     '#E76F51', 1.2, ':',  f'eps={eg-0.7:.2f} (gen-0.7)'),
        (eg + 0.7,     '#457B9D', 1.2, ':',  f'eps={eg+0.7:.2f} (gen+0.7)'),
    ]
    
    for e_pick, col, lw, ls, lbl in picks:
        t, xc = run_sim_trace(R, e_pick, pi, bi, pdi, bdi, sim_t, rec_dt)
        ax2.plot(t, xc, color=col, lw=lw, ls=ls, label=lbl)
    
    ax2.axhline(0, color='k', lw=0.3)
    ax2.axhline(200, color='gray', lw=0.5, ls=':')
    ax2.axhline(-200, color='gray', lw=0.5, ls=':')
    ax2.set_ylim(-350, 350)
    ax2.set_xlabel('시간 [s]', fontsize=12)
    ax2.set_ylabel('x_CoM [mm]', fontsize=12)
    ax2.set_title('대표 epsilon 궤적 비교', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right', ncol=2)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__), '..',
                             'docs', f'eps_gen_scan_heatmap_{case_idx+1}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  저장: {save_path}")
    plt.close()

print("완료!")
