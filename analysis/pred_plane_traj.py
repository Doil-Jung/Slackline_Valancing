# (phi,beta) "원위치 평면" vs "예측점 평면(q_pred = q + qdot/lam)" 비교
# 핵심: 안정모드 직선 = {u2^T x = 0} (u2 ⊥ v1). 따라서 u2^T q_pred = 2A 이므로
#   예측점이 직선 위 <=> A=0 (속도 포함 정확한 판정). Wait(자유운동) 중 A=0이면
#   예측점은 직선을 따라 미끄러지고, A!=0이면 직선에서 순수 e^{lam t}로 멀어진다.
# 검증: (1) 순간 eps* 접기 후 예측점의 직선이탈거리 ~ 0 (기계정밀도)
#       (2) 완전사이클(과접기-대기-펴기)에서 Wait 중 이탈거리 = 2A e^{lam t} 정확 일치, 펴기 후 0
# 그림: docs/예측점평면_궤적_R0.7.png
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Mt=75.0; h=0.96; ps=72.0; R=0.7; g=9.81; Jb=89.03; mu=4.33
beta_i=np.deg2rad(2.0); Tf=0.15; Te=0.15

M22=np.array([[Mt*R**2,R*ps],[R*ps,Jb]]); G22=np.array([[-Mt*g*R,0],[0,ps*g]])
Mi=np.linalg.inv(M22); A2=Mi@G22
w,V=np.linalg.eig(A2); si=np.argmin(w); ui=np.argmax(w)
omega=np.sqrt(-w[si]); lam=np.sqrt(w[ui]); Tosc=2*np.pi/omega
v_s=V[:,si]; r=v_s[0]/v_s[1]; Vinv=np.linalg.inv(V); u2=Vinv[ui,:]
Tw=1*Tosc
print(f"omega={omega:.3f} lam={lam:.3f} r={r:.3f}  u2@v1={u2@v_s:.1e} (직선 = u2의 영직선)")

def sig(u):
    u=np.clip(u,0,1); return np.where(u<=0.5,2*u**2,1-2*(1-u)**2)
def sig_ddu(u): return np.where((u>0)&(u<=0.5),4.0,np.where((u>0.5)&(u<1.0),-4.0,0.0))
def ddelta(t,df):
    if t<Tf: return df*sig_ddu(t/Tf)/Tf**2
    elif t<Tf+Tw: return 0.0
    else: return -df*sig_ddu((t-Tf-Tw)/Te)/Te**2
def rhs(t,y,df):
    q=y[:2]; qd=y[2:]; qdd=Mi@(G22@q+np.array([0.0,-mu*ddelta(t,df)])); return [qd[0],qd[1],qdd[0],qdd[1]]
def rhs_free(t,y):
    q=y[:2]; qd=y[2:]; qdd=Mi@(G22@q); return [qd[0],qd[1],qdd[0],qdd[1]]
def Acoef(q,qd):
    eta=Vinv@q; etad=Vinv@qd; return 0.5*(eta[ui]+etad[ui]/lam)
def integ(df,n=4000):
    T=Tf+Tw+Te; ts=np.linspace(0,T,n)
    s=solve_ivp(rhs,[0,T],[0,beta_i,0,0],args=(df,),t_eval=ts,max_step=T/n,rtol=1e-10,atol=1e-12)
    return ts,s.y,Acoef(s.y[:2,-1],s.y[2:,-1])

_,_,A0=integ(0.0); _,_,A1=integ(0.1); df_star=-A0*0.1/(A1-A0)
ts,Y,Ae=integ(df_star)
print(f"완전사이클 df*={np.rad2deg(df_star):.2f}deg, A_final={Ae:.1e}")

def qpred(Y): return Y[:2]+Y[2:]/lam
QP=qpred(Y)
# 사이클 후 자유운동
Tpost=2*Tosc
sp=solve_ivp(rhs_free,[0,Tpost],list(Y[:,-1]),t_eval=np.linspace(0,Tpost,2000),max_step=Tpost/2000,rtol=1e-10,atol=1e-12)
QPp=qpred(sp.y)
# 순간 eps* 접기(단발): A=0로 착지, 예측점은 직선 위를 미끄러짐
eps=-h/(r*R+h)
qn=np.array([h*(1+eps)*beta_i/R, -eps*beta_i]); qdn=np.zeros(2)
sd=solve_ivp(rhs_free,[0,2*Tosc],[qn[0],qn[1],0,0],t_eval=np.linspace(0,2*Tosc,2000),max_step=1e-3,rtol=1e-10,atol=1e-12)
QPd=qpred(sd.y)

# --- 수치 검증 ---
dist=lambda QP: (u2@QP)/np.linalg.norm(u2)          # 직선까지 부호거리(사영 스칼라)
d_single=np.max(np.abs(dist(QPd)))
print(f"[검증1] 순간 eps* 접기 후 예측점의 직선이탈 최대 = {d_single:.2e} (직선 위를 미끄러짐)")
i_f=np.searchsorted(ts,Tf); i_w=np.searchsorted(ts,Tf+Tw)
Af=Acoef(Y[:2,i_f],Y[2:,i_f])   # 접기 종료 시점의 잔여(의도된 과접기) A
dw=dist(QP[:,i_f:i_w+1]); tw=ts[i_f:i_w+1]-ts[i_f]
model=2*Af*np.exp(lam*tw)/np.linalg.norm(u2)*np.linalg.norm(V[:,ui])  # 2A e^{lt} * |v2| 성분의 수직거리
# 수직거리 = u2@qpred/|u2| = 2A e^{lt} /|u2| (u2@v2=1 정규화 기준)
model=2*Af*np.exp(lam*tw)/np.linalg.norm(u2)
print(f"[검증2] Wait 중 이탈거리 vs 2A·e^(lam t)/|u2| 최대오차 = {np.max(np.abs(dw-model)):.2e}")
print(f"[검증3] 펴기 종료 후 예측점 직선이탈 최대 = {np.max(np.abs(dist(QPp))):.2e}")

# --- 그림: 왼쪽 원위치 평면(그림자), 오른쪽 예측점 평면 ---
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.6,6.4))
bb=np.linspace(-25,25,60)
for a in (ax,ax2):
    for C in np.linspace(-0.30,0.30,31): a.plot(bb,np.rad2deg((C-h*np.deg2rad(bb))/R),color="0.90",lw=0.8,zorder=0)
    a.plot(bb,r*bb,color="seagreen",lw=2.2,ls="--",zorder=2,label="stable-mode line φ=r·β")
    a.axhline(0,color="0.6",lw=0.6); a.axvline(0,color="0.6",lw=0.6); a.grid(alpha=0.12)
    a.set_xlabel("β [deg]")
ax.set_xlim(-12,9); ax.set_ylim(-12,13)
ax2.set_xlim(-17,13); ax2.set_ylim(-17,20)
D=np.rad2deg
b,p=D(Y[1]),D(Y[0]); bp,pp=D(sp.y[1]),D(sp.y[0])
ax.plot(b[:i_f+1],p[:i_f+1],color="#d1495b",lw=3.0,label="Fold")
ax.plot(b[i_f:i_w+1],p[i_f:i_w+1],color="#2e86de",lw=3.0,label="Wait")
ax.plot(b[i_w:],p[i_w:],color="#e1a100",lw=3.0,label="Extend")
ax.plot(bp,pp,color="#8e44ad",lw=1.5,label="post-cycle")
ax.plot(b[0],p[0],"ko",ms=8)
ax.set_ylabel("φ [deg]")
ax.set_title("(a) raw (β, φ): shadow of 4D — velocities hidden,\n'on the line' is NOT a stability test")
ax.legend(loc="lower left",fontsize=8.5,framealpha=0.95)
qb,qp_=D(QP[1]),D(QP[0]); qbp,qpp=D(QPp[1]),D(QPp[0]); qbd,qpd=D(QPd[1]),D(QPd[0])
ax2.plot(qb[:i_f+1],qp_[:i_f+1],color="#d1495b",lw=3.0,label="Fold")
ax2.plot(qb[i_f:i_w+1],qp_[i_f:i_w+1],color="#2e86de",lw=3.0,label="Wait: leaves line as 2A·e^(λt)")
ax2.plot(qb[i_w:],qp_[i_w:],color="#e1a100",lw=3.0,label="Extend: lands ON line")
ax2.plot(qbp,qpp,color="#8e44ad",lw=1.7,label="post-cycle: slides ALONG line")
ax2.plot(qbd,qpd,color="0.45",lw=1.4,ls=":",label="single instant ε* fold: on line forever")
ax2.plot(qb[0],qp_[0],"ko",ms=8)
ax2.plot(qb[-1],qp_[-1],"*",color="#e1a100",ms=16,zorder=6)
def _fit(a,xs,ys,padx=0.07,pady=0.11):
    xs=np.concatenate([np.atleast_1d(np.asarray(v,float)) for v in xs])
    ys=np.concatenate([np.atleast_1d(np.asarray(v,float)) for v in ys])
    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max()
    cx0,cx1=a.get_xlim(); cy0,cy1=a.get_ylim()
    dx=(x1-x0)*padx+0.5; dy=(y1-y0)*pady+0.5
    a.set_xlim(min(cx0,x0-dx),max(cx1,x1+dx)); a.set_ylim(min(cy0,y0-dy),max(cy1,y1+dy))
_fit(ax,[b,bp],[p,pp])
_fit(ax2,[qb,qbp,qbd],[qp_,qpp,qpd])
ax2.set_title("(b) predicted point q+q̇/λ: distance to stable line = 2A/|u₂|\n→ 'on the line' IS the exact stability test (velocities included)")
ax2.legend(loc="lower left",fontsize=8.5,framealpha=0.95)
plt.tight_layout(); out="docs/예측점평면_궤적_R0.7.png"; plt.savefig(out,dpi=140); print("saved",out)
