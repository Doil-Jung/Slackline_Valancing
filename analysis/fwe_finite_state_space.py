"""
FWE 유한시간 접기-대기-펴기의 (phi, beta) 상태공간 궤적 + 시계열
모델: Wait 동역학 (phi,beta) 2x2 + delta(t) 구동.  M22 q'' = G22 q + (0,-mu*delta'')^T
사람스케일, R=0.7.  완전사이클 df는 A_final=0 (root-find, df에 affine).
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Mt=75.0; h=0.96; ps=72.0; p1=51.0; p2=21.0
R=0.7; g=9.81; Jb=89.03; mu=4.33
beta_i=np.deg2rad(2.0); Tf=0.15; Te=0.15

M22=np.array([[Mt*R**2,R*ps],[R*ps,Jb]]); G22=np.array([[-Mt*g*R,0],[0,ps*g]])
Mi=np.linalg.inv(M22); A2=Mi@G22
w,V=np.linalg.eig(A2); si=np.argmin(w); ui=np.argmax(w)
omega=np.sqrt(-w[si]); lam=np.sqrt(w[ui]); Tosc=2*np.pi/omega
v_s=V[:,si]; r=v_s[0]/v_s[1]; Vinv=np.linalg.inv(V); u2=Vinv[ui,:]
kappa2=(u2@Mi)[1]; k=0.5*kappa2*mu
print(f"omega={omega:.3f} lam={lam:.3f} Tosc={Tosc:.4f} r={r:.4f} k={k:.5f}")
Tw=1*Tosc

def sig(u):
    u=np.clip(u,0,1); return np.where(u<=0.5,2*u**2,1-2*(1-u)**2)
def sig_ddu(u): return np.where((u>0)&(u<=0.5),4.0,np.where((u>0.5)&(u<1.0),-4.0,0.0))
def Bx(x): return 4*(np.exp(x/2)-1)**2/x**2
def delta_t(t,df):
    if t<Tf: return df*sig(t/Tf)
    elif t<Tf+Tw: return df
    else: return df*(1-sig((t-Tf-Tw)/Te))
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
A0c=Acoef(np.array([0,beta_i]),np.array([0,0])); xf,xw,xe=lam*Tf,lam*Tw,lam*Te
df_closed=np.exp(xf+xw+xe)*A0c/(k*(np.exp(xe+xw)*Bx(xf)-Bx(xe)))
print(f"df_star(num)={np.rad2deg(df_star):.3f}deg A_final={Ae:.2e} | df_closed={np.rad2deg(df_closed):.3f}deg ({100*(df_closed-df_star)/df_star:+.3f}%)")

i_f=np.searchsorted(ts,Tf); i_w=np.searchsorted(ts,Tf+Tw)
b=np.rad2deg(Y[1]); p=np.rad2deg(Y[0])
# post-cycle bounded
Tpost=2*Tosc
sp=solve_ivp(rhs_free,[0,Tpost],[Y[0,-1],Y[1,-1],Y[2,-1],Y[3,-1]],t_eval=np.linspace(0,Tpost,2000),max_step=Tpost/2000,rtol=1e-10,atol=1e-12)
bp=np.rad2deg(sp.y[1]); pp=np.rad2deg(sp.y[0])
# single instant eps* fold: bounded (A=0) but body stays folded (delta!=0)
eps=-h/(r*R+h); dfe=(1+eps)*beta_i*(Jb-Mt*h**2)/mu
bn=beta_i-mu*dfe/(Jb-Mt*h**2); pn=h*(beta_i-bn)/R
sd=solve_ivp(rhs_free,[0,1.25*Tosc],[pn,bn,0,0],t_eval=np.linspace(0,1.25*Tosc,1500),max_step=1e-3,rtol=1e-10,atol=1e-12)
bd=np.rad2deg(sd.y[1]); pd=np.rad2deg(sd.y[0])
# teleport fold (same df)
dbq=-mu*df_star/(Jb-Mt*h**2); bf=beta_i+dbq; pf=0+h*(beta_i-bf)/R
tb=np.rad2deg([beta_i,bf]); tp=np.rad2deg([0,pf])

fig=plt.figure(figsize=(13.6,6.4)); ax=fig.add_subplot(1,2,1); ax2=fig.add_subplot(1,2,2)
bb=np.linspace(-14,14,60)
for C in np.linspace(-0.14,0.14,19): ax.plot(bb,np.rad2deg((C-h*np.deg2rad(bb))/R),color="0.88",lw=0.8,zorder=0)
ax.plot([],[],color="0.88",lw=0.8,label="CoM contours (slope −h/R)")
ax.plot(bb,r*bb,color="seagreen",lw=2.0,ls="--",label=f"stable-mode line φ=r·β (r={r:.2f})",zorder=2)
ax.plot(bd,pd,color="0.55",lw=1.5,ls=":",zorder=2,label="single instant ε* fold (bounded, stays folded)")
ax.plot(b[:i_f+1],p[:i_f+1],color="#d1495b",lw=3.0,label="Fold (finite-time)",zorder=4)
ax.plot(b[i_f:i_w+1],p[i_f:i_w+1],color="#2e86de",lw=3.0,label="Wait (over-fold grows)",zorder=4)
ax.plot(b[i_w:],p[i_w:],color="#e1a100",lw=3.0,label="Extend (finite-time)",zorder=4)
ax.plot(bp,pp,color="#8e44ad",lw=1.7,label="post-cycle (δ=0): bounded",zorder=3)
ax.annotate("",xy=(tb[1],tp[1]),xytext=(tb[0],tp[0]),arrowprops=dict(arrowstyle="-|>",color="0.45",ls=(0,(5,3)),lw=1.8),zorder=3)
ax.text((tb[0]+tb[1])/2-2.0,(tp[0]+tp[1])/2+0.5,"instant fold\n(teleport, same δ_f)",color="0.4",fontsize=8,ha="center")
ax.plot(b[0],p[0],"ko",ms=8,zorder=6); ax.text(b[0]+0.3,p[0]-1.0,"start β=2°, rest",fontsize=9)
ax.plot(b[i_f],p[i_f],"s",color="#d1495b",ms=7,zorder=6); ax.plot(b[i_w],p[i_w],"s",color="#2e86de",ms=7,zorder=6)
ax.plot(b[-1],p[-1],"*",color="#e1a100",ms=16,zorder=6); ax.text(b[-1]+0.3,p[-1]-0.2,"end: δ=0, A≈0",fontsize=8.5)
ax.axhline(0,color="0.6",lw=0.6); ax.axvline(0,color="0.6",lw=0.6)
ax.set_xlabel("β  weighted CoM tilt [deg]"); ax.set_ylabel("φ  rope/foot angle [deg]")
ax.set_title(f"(a) FWE full cycle in (β, φ) state space   R={R} m, T_w=1·T_osc, δ_f*={np.rad2deg(df_star):.1f}°")
ax.legend(loc="lower left",fontsize=7.8,framealpha=0.95); ax.set_xlim(-12,9); ax.set_ylim(-12,13); ax.grid(alpha=0.12)

tt=np.concatenate([ts,ts[-1]+sp.t])
phit=np.concatenate([p,pp]); betat=np.concatenate([b,bp])
delt=np.concatenate([np.array([np.rad2deg(delta_t(t,df_star)) for t in ts]),np.zeros_like(sp.t)])
ax2.axvspan(0,Tf,color="#d1495b",alpha=0.08); ax2.axvspan(Tf,Tf+Tw,color="#2e86de",alpha=0.06); ax2.axvspan(Tf+Tw,Tf+Tw+Te,color="#e1a100",alpha=0.10)
ax2.plot(tt,phit,color="#1f6f8b",lw=2.0,label="φ (rope)"); ax2.plot(tt,betat,color="#c0392b",lw=2.0,label="β (CoM tilt)")
ax2.plot(tt,delt,color="#555",lw=1.6,ls="--",label="δ (hip fold, input)"); ax2.axhline(0,color="0.6",lw=0.6)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("angle [deg]"); ax2.set_title("(b) time series — δ folds then returns to 0; φ,β bounded")
ax2.legend(loc="upper right",fontsize=9); ax2.grid(alpha=0.15)
yl=ax2.get_ylim()[1]
ax2.text(Tf/2,yl*0.92,"Fold",color="#d1495b",ha="center",fontsize=8); ax2.text(Tf+Tw/2,yl*0.92,"Wait",color="#2e86de",ha="center",fontsize=8); ax2.text(Tf+Tw+Te/2,yl*0.92,"Extend",color="#e1a100",ha="center",fontsize=8)
plt.tight_layout(); out="docs/FWE_상태공간궤적_R0.7.png"; plt.savefig(out,dpi=140); print("saved",out)
