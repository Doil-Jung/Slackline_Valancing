# 예측변위 q_pred = q + qdot/lambda 의 정확성 검토 (선형 결합모델, 사람스케일 R=0.7)
# 결론: 예측변위의 u2 사영 = 성장지수 e^{lam t}의 계수 2A. 선형모델에서 근사가 아니라 정확한 모드 대수.
# E1: 임의 4D 상태 + eps*_gen 접기 -> A=0 기계정밀도 확인 (속도 포함 정확성)
# E2: 속도 무시(정지공식 eps*)로 접으면 잔여 A -> 낙하시간 (docs의 1.1~1.6s 낙하 관찰과 정합)
# E3: c2(t) = A e^{lt} + B e^{-lt} 정확 분해 확인 (RK4 대조)
# E4: 상태 추정오차 -> 잔여 A 민감도 (재개입 여유시간)
# 상세: docs/예측변위_정확성_검토_20260705.md
import numpy as np

Mt=75.0; h=0.96; ps=72.0; R=0.7; g=9.81; Jb=89.03
M22=np.array([[Mt*R**2,R*ps],[R*ps,Jb]]); G22=np.array([[-Mt*g*R,0],[0,ps*g]])
A2=np.linalg.inv(M22)@G22
w,V=np.linalg.eig(A2); si=np.argmin(w); ui=np.argmax(w)
omega=np.sqrt(-w[si]); lam=np.sqrt(w[ui]); Vinv=np.linalg.inv(V)
u2=Vinv[ui,:]; u1=Vinv[si,:]; e=np.array([h/R,-1.0])
print(f"omega_eff={omega:.4f} lam_eff={lam:.4f} (R={R})  1/lam={1/lam:.3f}s  Tosc={2*np.pi/omega:.3f}s")

def modal_prop(q,qd,t):
    """선형계 정확해 (모드 닫힌형)"""
    eta=Vinv@q; etad=Vinv@qd
    e1=eta[si]*np.cos(omega*t)+etad[si]/omega*np.sin(omega*t)
    e1d=-eta[si]*omega*np.sin(omega*t)+etad[si]*np.cos(omega*t)
    Ac=0.5*(eta[ui]+etad[ui]/lam); Bc=0.5*(eta[ui]-etad[ui]/lam)
    e2=Ac*np.exp(lam*t)+Bc*np.exp(-lam*t); e2d=lam*(Ac*np.exp(lam*t)-Bc*np.exp(-lam*t))
    et=np.zeros(2); etd=np.zeros(2); et[si],et[ui]=e1,e2; etd[si],etd[ui]=e1d,e2d
    return V@et, V@etd

def Acoef(q,qd): eta=Vinv@q; etad=Vinv@qd; return 0.5*(eta[ui]+etad[ui]/lam)

def fold(q,qd,eps):
    """순간접기: beta -> -eps*beta_i, CoM 등고선 따라 이동, 속도 보존"""
    dq=(1+eps)*q[1]*e
    return q+dq, qd.copy()

def eps_gen(q,qd):
    qpred=q+qd/lam
    return -(u2@(qpred+q[1]*e))/(q[1]*(u2@e))

rng=np.random.default_rng(0)
# --- E1 ---
worstA=0.0; worst_amp=0.0
for _ in range(100):
    q=rng.uniform(-1,1,2)*np.deg2rad([8.0,4.0])
    qd=rng.uniform(-1,1,2)*np.deg2rad([20.0,10.0])
    if abs(q[1])<1e-4: continue
    qn,qdn=fold(q,qd,eps_gen(q,qd))
    worstA=max(worstA,abs(Acoef(qn,qdn)))
    qf,qdf=modal_prop(qn,qdn,3.0)
    worst_amp=max(worst_amp,abs(np.rad2deg(qf[1])))
print(f"[E1] 100개 임의 4D 상태(속도 최대 20deg/s): 접기 직후 |A|max = {worstA:.2e} (기계정밀도)")
print(f"     3초 후 |beta|max = {worst_amp:.2f} deg (유계 진동 — 단, 소진폭 아님에 주의)")

# --- E2 ---
vs=V[:,si]; r=vs[0]/vs[1]; eps_rest=-h/(r*R+h)
def fall_time(q,qd,thr=np.deg2rad(15)):
    for t in np.linspace(0,5,5001):
        qf,_=modal_prop(q,qd,t)
        if abs(qf[1])>thr: return t
    return np.inf
cases=[(np.deg2rad([0.0,2.0]),np.deg2rad([0.0,6.0])),
       (np.deg2rad([3.0,2.0]),np.deg2rad([-10.0,4.0])),
       (np.deg2rad([-2.0,3.0]),np.deg2rad([15.0,-5.0]))]
print(f"[E2] 정지공식 eps*={eps_rest:.3f} (r={r:.3f}) 를 움직이는 상태에 쓰면:")
for q,qd in cases:
    qn,qdn=fold(q,qd,eps_rest); A=Acoef(qn,qdn); tf=fall_time(qn,qdn)
    qn2,qdn2=fold(q,qd,eps_gen(q,qd)); A2c=Acoef(qn2,qdn2)
    print(f"     상태 q={np.rad2deg(q).round(1)} qd={np.rad2deg(qd).round(1)}deg(/s): "
          f"잔여 A={A:+.2e} -> beta>15deg 도달 {tf:.2f}s | eps*_gen이면 A={A2c:.1e}")

# --- E3 ---
q0=np.deg2rad([2.0,3.0]); qd0=np.deg2rad([5.0,-8.0])
A0=Acoef(q0,qd0); eta0=Vinv@q0; etad0=Vinv@qd0
B0=0.5*(eta0[ui]-etad0[ui]/lam)
def rk4(q,qd,T,n=20000):
    dt=T/n; y=np.concatenate([q,qd]); out=[]
    def f(y): return np.concatenate([y[2:],A2@y[:2]])
    for i in range(n+1):
        out.append(y.copy())
        k1=f(y);k2=f(y+dt/2*k1);k3=f(y+dt/2*k2);k4=f(y+dt*k3)
        y=y+dt/6*(k1+2*k2+2*k3+k4)
    return np.array(out)
T=1.0; Y=rk4(q0,qd0,T); ts=np.linspace(0,T,len(Y))
c2_num=np.array([u2@y[:2] for y in Y])
c2_pred=A0*np.exp(lam*ts)+B0*np.exp(-lam*ts)
print(f"[E3] c2(t) 닫힌형 vs RK4 최대오차 = {np.max(np.abs(c2_num-c2_pred)):.2e} (A={A0:.4f}, B={B0:.4f})")

# --- E4 ---
g_phid=0.5*u2[0]/lam; g_betad=0.5*u2[1]/lam; g_phi=0.5*u2[0]; g_beta=0.5*u2[1]
print(f"[E4] 잔여 A 민감도: dA/dphi={g_phi:.3f} dA/dbeta={g_beta:.3f} "
      f"dA/dphidot={g_phid:.4f}/(rad/s) dA/dbetadot={g_betad:.4f}/(rad/s)")
beta_gain=abs(V[1,ui])
for sb in [0.5,1.0,2.0,5.0]:
    Ae=abs(g_betad)*np.deg2rad(sb)
    t15=np.log(np.deg2rad(15)/(Ae*beta_gain))/lam
    print(f"     betadot 오차 {sb:>4.1f} deg/s -> 잔여A={Ae:.2e} -> beta 15deg까지 {t15:.2f}s (재개입 여유)")
for sp in [0.5,1.0,2.0]:
    Ae=abs(g_phi)*np.deg2rad(sp)
    t15=np.log(np.deg2rad(15)/(Ae*beta_gain))/lam
    print(f"     phi 추정오차 {sp:>4.1f} deg   -> 잔여A={Ae:.2e} -> beta 15deg까지 {t15:.2f}s")
