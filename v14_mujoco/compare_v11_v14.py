"""
V11 vs V14 비교 테스트 (머리 없음, 균일 봉 모델)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_are, expm

# ============================================================
#  Common Parameters (NO HEAD)
# ============================================================
M1 = 0.1;  M2 = 0.25
L1 = 0.2;  L2 = 0.2
BD = 0.1
R_TGT = 0.3
B_THETA = 0.01
GRAV = 9.81
DT = 0.001
THETA0 = np.radians(8.6)
ALPHA0 = 0.0
TAU_MAX = 2.2
Q_DIAG = [1, 50, 50, 0.1, 5, 5]
R_COST = 0.001
T_END = 3.0

# Derived
LC1 = L1/2;  LC2 = L2/2
I1 = M1*L1**2/12;  I2 = M2*L2**2/12

# ============================================================
#  LQR Gain — V14 방식 그대로 (v14_mujoco.py 복사)
# ============================================================
def compute_lqr_v14():
    """v14_mujoco.py의 compute_lqr() 그대로 복사"""
    R = R_TGT
    m2e = M2; lc2e = LC2; i2e = I2  # no head
    p1 = M1*LC1 + m2e*L1; p2 = m2e*lc2e; p3 = m2e*L1*lc2e; Mt = M1+m2e
    print(f"[LQR] M2_eff={m2e}, LC2_eff={lc2e}, I2_eff={i2e:.6f}")
    print(f"[LQR] p1={p1}, p2={p2}, p3={p3}, Mt={Mt}")
    
    M_eq = np.array([
        [Mt*R**2,               R*p1,           R*p2],
        [R*p1, M1*LC1**2+I1+m2e*L1**2,         p3],
        [R*p2,                  p3,    m2e*lc2e**2+i2e]])
    G_mat = np.array([[-Mt*GRAV*R,0,0],[0,p1*GRAV,0],[0,0,p2*GRAV]])
    D_mat = np.diag([0, 0, -B_THETA])
    E_mat = np.array([[0],[-1],[1]])
    
    Mi = np.linalg.inv(M_eq); n = 6
    print(f"[LQR] M_inv @ E = {(Mi @ E_mat).flatten()}")
    
    Ac = np.zeros((n,n)); Bc = np.zeros((n,1))
    Ac[:3,3:] = np.eye(3)
    Ac[3:,:3] = Mi @ G_mat; Ac[3:,3:] = Mi @ D_mat; Bc[3:,:] = Mi @ E_mat
    
    Z = np.zeros((n+1,n+1)); Z[:n,:n]=Ac*DT; Z[:n,n:]=Bc*DT
    eZ = expm(Z); Ad=eZ[:n,:n]; Bd=eZ[:n,n:n+1]
    Q = np.diag(Q_DIAG); Rc = np.array([[R_COST]])
    P = solve_discrete_are(Ad, Bd, Q, Rc)
    K = np.linalg.inv(Rc + Bd.T@P@Bd) @ (Bd.T@P@Ad)
    
    # 안정성 체크 (phi flip 전)
    eigs_raw = sorted(abs(e) for e in np.linalg.eigvals(Ad - Bd@K))
    print(f"[LQR] K_raw (before phi flip) = [{', '.join(f'{k:.4f}' for k in K[0])}]")
    print(f"[LQR] Raw eigs: {[f'{e:.4f}' for e in eigs_raw]}")
    
    K[0,0] *= -1; K[0,3] *= -1  # phi convention
    print(f"[LQR] K_v14 (after phi flip) = [{', '.join(f'{k:.4f}' for k in K[0])}]")
    return K[0], Ad, Bd

K_v14, Ad, Bd = compute_lqr_v14()

# V11용: phi flip을 하지 않은 원본 K
K_v11 = K_v14.copy()
K_v11[0] *= -1;  K_v11[3] *= -1  # undo phi flip → raw K
print(f"\n[V11] K = [{', '.join(f'{k:.4f}' for k in K_v11)}]")

# 초기 토크 확인
x0_v11 = np.array([0, 0, THETA0, 0, 0, 0])
tau0_v11 = -float(K_v11 @ x0_v11)
print(f"[V11] Initial tau = -K@x0 = {tau0_v11:.4f} N·m")
print(f"  → theta0={np.degrees(THETA0):.1f}°, 양수 tau = 상체 오른쪽 가속")
print(f"  → 올바른 방향: theta>0이면 tau<0 (왼쪽으로 복원)")
if tau0_v11 > 0 and THETA0 > 0:
    print("  ⚠ 토크 방향이 반대! E 행렬 부호 확인 필요")

# ============================================================
#  V11: Lagrangian RK4 Simulation
# ============================================================
def run_v11():
    m1, m2 = M1, M2
    l1, l2 = L1, L2
    R = R_TGT; g = GRAV
    ell1 = l1/2; ell2 = l2/2
    i1 = m1*l1**2/12; i2 = m2*l2**2/12
    Mt = m1 + m2
    
    def derivs(state, tau):
        phi, alpha, theta, pd, ad, td = state
        ca = np.cos(alpha); sa = np.sin(alpha)
        ct = np.cos(theta); st = np.sin(theta)
        cat = np.cos(alpha - theta)
        p1 = m1*ell1 + m2*l1; p2 = m2*ell2; p3 = m2*l1*ell2
        
        M = np.array([
            [Mt*R**2, R*p1*ca, R*p2*ct],
            [R*p1*ca, m1*ell1**2+i1+m2*l1**2, p3*cat],
            [R*p2*ct, p3*cat, m2*ell2**2+i2]])
        
        f1 = -Mt*g*R*np.sin(phi) - R*p1*sa*ad**2 - R*p2*st*td**2
        f2 = p1*g*sa - p3*np.sin(alpha-theta)*td**2 - tau
        f3 = p2*g*st + p3*np.sin(alpha-theta)*ad**2 + tau - B_THETA*td
        
        accel = np.linalg.solve(M, [f1, f2, f3])
        return np.array([pd, ad, td, accel[0], accel[1], accel[2]])
    
    def rk4(state, tau, dt):
        k1 = derivs(state, tau)
        k2 = derivs(state + 0.5*dt*k1, tau)
        k3 = derivs(state + 0.5*dt*k2, tau)
        k4 = derivs(state + dt*k3, tau)
        return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    N = int(T_END / DT)
    state = np.array([0.0, ALPHA0, THETA0, 0.0, 0.0, 0.0])
    log = np.zeros((N, 8))
    
    for i in range(N):
        tau = -float(K_v11 @ state)
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        log[i] = [i*DT, *state, tau]
        state = rk4(state, tau, DT)
        if np.any(np.isnan(state)) or np.any(np.abs(state[:3]) > 2*np.pi):
            print(f"V11 diverged at t={i*DT:.3f}s"); log = log[:i+1]; break
    return log

# ============================================================
#  V14: MuJoCo Simulation (v14_mujoco.py의 XML 그대로)
# ============================================================
def run_v14():
    import mujoco
    
    HALF_POLE = 0.8 / 2.0
    ROPE_HORIZ = HALF_POLE - BD/2.0
    ROPE_VERT = R_TGT
    POLE_H = 1.0
    FOOT_Z = POLE_H - ROPE_VERT
    lw2 = (L1/4)/2; uw2 = (L2/4)/2; bd2 = BD/2
    rh = ROPE_HORIZ; rv = ROPE_VERT; hp = HALF_POLE
    
    xml = f"""<?xml version="1.0"?>
<mujoco model="slackline_v14_nohead">
  <option gravity="0 0 -{GRAV}" timestep="{DT}" iterations="200" tolerance="1e-10">
    <flag contact="disable"/>
  </option>
  <default>
    <geom contype="0" conaffinity="0"/>
    <joint damping="0" armature="0.001"/>
  </default>
  <worldbody>
    <geom type="plane" size="5 5 0.01" rgba="0.3 0.3 0.35 1" contype="1" conaffinity="1"/>
    <geom name="pole_a" type="cylinder" pos="0 {hp} {POLE_H/2}" size="0.01 {POLE_H/2}"/>
    <geom name="pole_b" type="cylinder" pos="0 {-hp} {POLE_H/2}" size="0.01 {POLE_H/2}"/>
    <body name="rope_a_mount" pos="0 {hp} {POLE_H}">
      <joint name="phi_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_a_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_a" type="capsule" size="0.005"
            fromto="0 0 0 0 {-rh} {-rv}" rgba="1 0.8 0 1" mass="0.01"/>
      <body name="lower_body" pos="0 {-rh} {-rv}">
        <joint name="ankle" type="hinge" axis="0 1 0"/>
        <geom name="lower_geom" type="box" size="{lw2} {bd2} {L1/2}"
              pos="0 {-bd2} {L1/2}" mass="{M1}" rgba="0.02 0.84 0.63 0.8"/>
        <body name="ankle_b_target" pos="0 {-BD} 0"/>
        <geom name="hip_cyl" type="cylinder" size="0.008 0.008"
              pos="0 {-bd2} {L1}" euler="90 0 0" rgba="1 0.75 0.04 1" mass="0.01"/>
        <body name="upper_body" pos="0 {-bd2} {L1}">
          <joint name="hip" type="hinge" axis="0 1 0" damping="{B_THETA}"/>
          <geom name="upper_geom" type="box" size="{uw2} {bd2} {L2/2}"
                pos="0 0 {L2/2}" mass="{M2}" rgba="0.51 0.22 0.93 0.8"/>
        </body>
      </body>
    </body>
    <body name="rope_b_mount" pos="0 {-hp} {POLE_H}">
      <joint name="rope_b_Y" type="hinge" axis="0 1 0"/>
      <joint name="rope_b_X" type="hinge" axis="1 0 0"/>
      <geom name="rope_b" type="capsule" size="0.005"
            fromto="0 0 0 0 {rh} {-(POLE_H - FOOT_Z)}" rgba="1 0.8 0 1" mass="0.01"/>
      <body name="rope_b_end" pos="0 {rh} {-(POLE_H - FOOT_Z)}"/>
    </body>
  </worldbody>
  <equality>
    <connect body1="rope_b_end" body2="ankle_b_target" anchor="0 0 0"
             solref="0.001 1" solimp="0.999 0.999 0.0001"/>
  </equality>
  <actuator>
    <motor name="hip_motor" joint="hip" ctrlrange="{-TAU_MAX} {TAU_MAX}" ctrllimited="true" gear="1"/>
  </actuator>
</mujoco>"""
    
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    
    # 초기 조건 설정
    hip_jid = model.joint('hip').id
    ankle_jid = model.joint('ankle').id
    data.qpos[model.jnt_qposadr[hip_jid]] = THETA0 - ALPHA0
    data.qpos[model.jnt_qposadr[ankle_jid]] = ALPHA0
    mujoco.mj_forward(model, data)
    
    # 초기 상태 확인
    x0 = get_mj_state(model, data)
    print(f"\n[V14] Initial state: phi={np.degrees(x0[0]):.2f}° alpha={np.degrees(x0[1]):.2f}° theta={np.degrees(x0[2]):.2f}°")
    tau0 = -float(K_v14 @ x0)
    print(f"[V14] Initial tau = {tau0:.4f} N·m")
    
    N = int(T_END / DT)
    log = np.zeros((N, 8))
    
    for i in range(N):
        x = get_mj_state(model, data)
        tau = -float(K_v14 @ x)
        tau = np.clip(tau, -TAU_MAX, TAU_MAX)
        data.ctrl[0] = tau
        log[i] = [data.time, *x, tau]
        mujoco.mj_step(model, data)
    
    return log

def get_mj_state(model, data):
    phi = data.joint('phi_Y').qpos[0]
    phi_dot = data.joint('phi_Y').qvel[0]
    lb_id = model.body('lower_body').id
    xmat = data.xmat[lb_id].reshape(3,3)
    alpha = np.arctan2(-xmat[2,0], xmat[2,2])
    ub_id = model.body('upper_body').id
    xmat_u = data.xmat[ub_id].reshape(3,3)
    theta = np.arctan2(-xmat_u[2,0], xmat_u[2,2])
    alpha_dot = data.cvel[lb_id][1]
    theta_dot = data.cvel[ub_id][1]
    return np.array([phi, alpha, theta, phi_dot, alpha_dot, theta_dot])

# ============================================================
#  Run & Plot
# ============================================================
print("\n=== Running V11 (Lagrangian RK4, no head) ===")
log11 = run_v11()
print(f"V11: {len(log11)} steps completed")

print("\n=== Running V14 (MuJoCo, no head) ===")
log14 = run_v14()
print(f"V14: {len(log14)} steps completed")

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle('V11 (Lagrangian RK4) vs V14 (MuJoCo) — No Head\n'
             f'm1={M1}, m2={M2}, L1={L1}, L2={L2}, R={R_TGT}, θ₀={np.degrees(THETA0):.1f}°',
             fontsize=13)

labels = ['φ (rope)', 'α (lower)', 'θ (upper)', 'τ (torque)']
indices = [1, 2, 3, 7]
ylabels = ['deg', 'deg', 'deg', 'N·m']

for ax, label, idx, yl in zip(axes, labels, indices, ylabels):
    d14 = log14[:, idx]; d11 = log11[:, idx]
    if yl == 'deg': d14 = np.degrees(d14); d11 = np.degrees(d11)
    ax.plot(log14[:, 0], d14, 'b-', alpha=0.8, lw=1.2, label='V14 (MuJoCo)')
    ax.plot(log11[:, 0], d11, 'r--', alpha=0.8, lw=1.2, label='V11 (Lagrangian)')
    ax.set_ylabel(f'{label} [{yl}]')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time [s]')
plt.tight_layout()
plt.savefig('compare_v11_v14.png', dpi=150)
print(f"\nSaved: compare_v11_v14.png")
plt.close()
