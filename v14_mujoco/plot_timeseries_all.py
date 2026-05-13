# -*- coding: utf-8 -*-
"""
모든 실험 케이스의 시계열 그래프 생성
R별로 한 페이지, beta0별로 한 행
각 그래프: beta(CoM 기울기), delta(접힘), phi(줄), tau(토크)
"""
import sys, os, pickle
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# --- 배치 실험 모듈에서 함수 가져오기 ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from batch_experiment import (generate_mjcf, compute_lqr, get_state,
                               M1, M2_EFF, LC1, LC2_EFF, L1, DT, TAU_MAX, GRAV)

Mt = M1 + M2_EFF
p1 = M1*LC1 + M2_EFF*L1
p2 = M2_EFF*LC2_EFF
h_com = (p1 + p2) / Mt

# ============================================================
#  시뮬레이션 실행 (시계열 전체 기록)
# ============================================================
def run_sim_full(R, beta0_deg, sim_time=5.0, K_lqr=None):
    """Run simulation and return full time series."""
    xml = generate_mjcf(R)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    if K_lqr is None:
        K_lqr = compute_lqr(R)
    
    beta0 = np.radians(beta0_deg)
    hip_qadr = model.jnt_qposadr[model.joint('hip').id]
    ankle_qadr = model.jnt_qposadr[model.joint('ankle').id]
    
    data.qpos[ankle_qadr] = beta0
    data.qpos[hip_qadr] = 0
    mujoco.mj_forward(model, data)
    
    n_steps = int(sim_time / DT)
    sample_every = 5  # 5ms마다 기록
    
    ts = []; phis = []; alphas = []; thetas = []; taus = []
    
    failed = False
    for i in range(n_steps):
        x = get_state(model, data)
        if np.any(np.isnan(x)) or abs(x[1]) > np.radians(80):
            failed = True
            break
        
        tau = -float(K_lqr @ x)
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        data.ctrl[0] = tau
        
        if i % sample_every == 0:
            ts.append(data.time)
            phis.append(x[0])
            alphas.append(x[1])
            thetas.append(x[2])
            taus.append(tau)
        
        mujoco.mj_step(model, data)
    
    ts = np.array(ts); phis = np.array(phis)
    alphas = np.array(alphas); thetas = np.array(thetas); taus = np.array(taus)
    
    # 유도 변수
    betas = (p1*alphas + p2*thetas) / (p1+p2)
    deltas = thetas - alphas
    
    return {
        't': ts, 'phi': phis, 'alpha': alphas, 'theta': thetas,
        'tau': taus, 'beta': betas, 'delta': deltas,
        'failed': failed, 'R': R, 'beta0_deg': beta0_deg
    }


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'timeseries')
    os.makedirs(out_dir, exist_ok=True)
    
    R_list = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    beta_list = [1, 2, 3, 5, 7, 10, 12, 15]
    
    print("=" * 60)
    print("  시계열 그래프 생성")
    print("=" * 60)
    
    # LQR 사전 계산
    K_cache = {}
    for R in R_list:
        K_cache[R] = compute_lqr(R)
    
    for R in R_list:
        print(f"\n  === R = {R:.1f} m ===")
        
        n_betas = len(beta_list)
        fig, axes = plt.subplots(n_betas, 1, figsize=(14, 3*n_betas), sharex=True)
        
        for idx, b0 in enumerate(beta_list):
            print(f"    beta0={b0}...", end='', flush=True)
            res = run_sim_full(R, b0, sim_time=5.0, K_lqr=K_cache[R])
            print(f" {'FAIL' if res['failed'] else 'OK'}")
            
            ax = axes[idx]
            t = res['t']
            
            if len(t) > 0:
                # 왼쪽 축: 각도
                l1, = ax.plot(t, np.degrees(res['beta']), '-', color='#e74c3c',
                              linewidth=2, label='beta (CoM)')
                l2, = ax.plot(t, np.degrees(res['delta']), '-', color='#3498db',
                              linewidth=2, label='delta (fold)')
                l3, = ax.plot(t, np.degrees(res['phi']), '-', color='#2ecc71',
                              linewidth=1.5, label='phi (rope)')
                ax.axhline(y=0, color='black', linewidth=0.3)
                ax.set_ylabel('[deg]', fontsize=9)
                
                # 오른쪽 축: 토크
                ax2 = ax.twinx()
                l4, = ax2.plot(t, res['tau'], '-', color='#95a5a6',
                               linewidth=0.8, alpha=0.5, label='tau')
                ax2.set_ylabel('tau [Nm]', fontsize=8, color='gray')
                ax2.tick_params(axis='y', labelcolor='gray', labelsize=7)
                
                # 범례 (첫 행만)
                if idx == 0:
                    lines = [l1, l2, l3, l4]
                    labels = [l.get_label() for l in lines]
                    ax.legend(lines, labels, loc='upper right', fontsize=8, ncol=4)
                
                # 제목
                status = "FAIL" if res['failed'] else "OK"
                d_peak = np.degrees(np.max(np.abs(res['delta'])))
                ax.set_title(f"b0={b0} deg  |  d_peak={d_peak:.1f} deg  |  {status}",
                             fontsize=10, loc='left', fontweight='bold',
                             color='red' if res['failed'] else 'black')
            else:
                ax.text(0.5, 0.5, f'b0={b0}: IMMEDIATE FAIL', transform=ax.transAxes,
                        ha='center', fontsize=12, color='red')
            
            ax.grid(True, alpha=0.2)
        
        axes[-1].set_xlabel('Time [s]', fontsize=11)
        
        fig.suptitle(f'R = {R:.1f} m  |  Time Series (beta, delta, phi, tau)',
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        save_path = os.path.join(out_dir, f'timeseries_R{R:.1f}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {save_path}")
    
    print(f"\n  All done! -> {out_dir}")


if __name__ == '__main__':
    main()
