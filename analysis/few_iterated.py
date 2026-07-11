import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

Mt=75.0; h=0.96; ps=72.0; p1=51.0; p2=21.0
R=0.7; g=9.81; Jb=89.03; mu=4.33
M22=np.array([[Mt*R**2,R*ps],[R*ps,Jb]]); G22=np.array([[-Mt*g*R,0],[0,ps*g]])
Mi=np.linalg.inv(M22); A2=Mi@G22
w,V=np.linalg.eig(A2); si=np.argmin(w); ui=np.argmax(w)
omega=np.sqrt(-w[si]); lam=np.sqrt(w[ui]); Tosc=2*np.pi/omega
v_s=V[:,si]; r=v_s[0]/v_s[1]; Vinv=np.linalg.inv(V); u2=Vinv[ui,:]
k=0.5*(u2@Mi)[1]*mu
def B(x): return 4*(np.exp(x/2)-1)**2/x**2
def Acoef(q,qd): eta=Vinv@q; etad=Vinv@qd; return 0.5*(eta[ui]+etad[ui]/lam)
def sig(u): u=np.clip(u,0,1); return np.where(u<=0.5,2*u**2,1-2*(1-u)**2)
def sig_ddu(u): return np.where((u>0)&(u<=0.5),4.0,np.where((u>0.5)&(u<1.0),-4.0,0.0))
def integ_cycle(y0,df,Th,Tw,n=400):
    T=2*Th+Tw
    def dd(t):
        if t<Th: return df*sig_ddu(t/Th)/Th**2
        elif t<Th+Tw: return 0.0
        else: return -df*sig_ddu((t-Th-Tw)/Th)/Th**2
    def dfun(t):
        if t<Th: return df*sig(t/Th)
        elif t<Th+Tw: return df
        else: return df*(1-sig((t-Th-Tw)/Th))
    def rhs(t,y):
        q=y[:2]; qd=y[2:]; qdd=Mi@(G22@q+np.array([0.0,-mu*dd(t)])); return [qd[0],qd[1],qdd[0],qdd[1]]
    ts=np.linspace(0,T,n); s=solve_ivp(rhs,[0,T],y0,t_eval=ts,max_step=T/n,rtol=1e-9,atol=1e-11)
    return ts,s.y,np.array([dfun(t) for t in ts])
def cyc_coef(Th,Tw): G=np.exp(lam*(2*Th+Tw)); gg=k*(np.exp(lam*(Th+Tw))*B(lam*Th)-B(lam*Th)); return G,gg

Th=0.08; Tw=0.12; rho=1.15; dfmax=np.deg2rad(55)
G,gg=cyc_coef(Th,Tw); gamma=rho*G/gg

def run_iter(beta_i_deg,err=0.0,N=8):
    y=np.array([0.0,np.deg2rad(beta_i_deg),0.0,0.0]); segs=[]; An=[]; dfn=[]; t0=0.0
    for c in range(N):
        A=Acoef(y[:2],y[2:]); An.append(A)
        df=np.clip(gamma*A,-dfmax,dfmax); dfn.append(df)
        ts,ys,dv=integ_cycle(y,df*(1+err),Th,Tw)
        segs.append((ts+t0,ys.copy(),dv)); t0+=ts[-1]; y=ys[:,-1]
    An.append(Acoef(y[:2],y[2:]))
    return segs,np.array(An),np.array(dfn)
def run_single(beta_i_deg,err=0.0,N=8):
    Gs,ggs=cyc_coef(Th,Tosc); y=np.array([0.0,np.deg2rad(beta_i_deg),0,0]); An=[]; t0=0; segs=[]
    for c in range(N):
        A=Acoef(y[:2],y[2:]); An.append(A)
        if c==0: df=(Gs/ggs)*A; ts,ys,dv=integ_cycle(y,df*(1+err),Th,Tosc)
        else: ts,ys,dv=integ_cycle(y,0.0,Th,0.0)   # 무개입 자유전파
        segs.append((ts+t0,ys.copy(),dv)); t0+=ts[-1]; y=ys[:,-1]
    An.append(Acoef(y[:2],y[2:]))
    return segs,np.array(An)

segs,An,dfn=run_iter(1.5,err=0.0,N=8)
_,An_e,_=run_iter(1.5,err=0.2,N=8)
_,Ans=run_single(1.5,err=0.2,N=8)
print("iter A_n:", " ".join(f"{a:+.1e}" for a in An))
print("iter df(deg):", " ".join(f"{np.rad2deg(d):+.0f}" for d in dfn))

# ===== FIGURE =====
fig=plt.figure(figsize=(14.5,6.6))
gs=fig.add_gridspec(2,2,width_ratios=[1.35,1],height_ratios=[1,1],wspace=0.24,hspace=0.42)
ax=fig.add_subplot(gs[:,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,1])

# (a) state space
bb=np.linspace(-6,6,40)
for C in np.linspace(-0.06,0.06,15): ax.plot(bb,np.rad2deg((C-h*np.deg2rad(bb))/R),color="0.9",lw=0.7,zorder=0)
ax.plot([],[],color="0.9",lw=0.7,label="CoM contours (slope -h/R)")
ax.plot(bb,r*bb,color="seagreen",lw=2.2,ls="--",label=f"stable-mode line  phi=r*beta  (r={r:.2f})",zorder=2)
ncyc=len(segs); cmap=cm.viridis
for i,(ts,ys,dv) in enumerate(segs):
    col=cmap(i/(ncyc-1))
    ax.plot(np.rad2deg(ys[1]),np.rad2deg(ys[0]),color=col,lw=2.0,zorder=4,alpha=0.9)
    ax.plot(np.rad2deg(ys[1,0]),np.rad2deg(ys[0,0]),'o',color=col,ms=4,zorder=5)  # 사이클 시작(δ=0, A측정점)
ax.plot(np.rad2deg(segs[0][1][1,0]),np.rad2deg(segs[0][1][0,0]),'ko',ms=9,zorder=6)
ax.text(np.rad2deg(segs[0][1][1,0])+0.15,np.rad2deg(segs[0][1][0,0]),"start  beta=1.5deg",fontsize=9)
ax.plot(np.rad2deg(segs[-1][1][1,-1]),np.rad2deg(segs[-1][1][0,-1]),'*',color='#d1495b',ms=17,zorder=6)
ax.text(0.2,-0.3,"converge (chatter near line)",fontsize=8.5,color='#d1495b')
ax.axhline(0,color="0.6",lw=0.5); ax.axvline(0,color="0.6",lw=0.5)
sm=cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(1,ncyc)); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,pad=0.01,fraction=0.045); cb.set_label("cycle #",fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_xlabel("beta  weighted CoM tilt [deg]"); ax.set_ylabel("phi  rope/foot angle [deg]")
ax.set_title(f"(a) Iterated FEW in (beta,phi) — alternating folds converge to stable line\n     R={R}m, Th={Th*1000:.0f}/Tw={Tw*1000:.0f}ms, rho={rho}  (Tcyc < doubling time)")
ax.legend(loc="upper left",fontsize=8,framealpha=0.95); ax.grid(alpha=0.12); _ab=np.concatenate([np.rad2deg(s[1][1]) for s in segs]); _ap=np.concatenate([np.rad2deg(s[1][0]) for s in segs])
ax.set_xlim(_ab.min()-1.0,_ab.max()+1.0); ax.set_ylim(_ap.min()-1.0,_ap.max()+1.0); ax.set_aspect("auto")

# (b) time series
tall=np.concatenate([s[0] for s in segs]); phall=np.concatenate([np.rad2deg(s[1][0]) for s in segs])
beall=np.concatenate([np.rad2deg(s[1][1]) for s in segs]); deall=np.concatenate([np.rad2deg(s[2]) for s in segs])
axb.plot(tall,beall,color="#c0392b",lw=1.8,label="beta (CoM tilt)")
axb.plot(tall,phall,color="#1f6f8b",lw=1.6,label="phi (rope)")
axb.plot(tall,deall,color="#555",lw=1.3,ls="--",label="delta (hip fold, input)")
axb.axhline(0,color="0.6",lw=0.5); axb.set_xlabel("time [s]"); axb.set_ylabel("angle [deg]")
axb.set_title("(b) time series — delta alternates front/back, beta & phi decay"); axb.legend(fontsize=7.5,ncol=3,loc="lower right"); axb.grid(alpha=0.15)

# (c) A_n convergence / robustness
cyc=np.arange(len(An))
axc.semilogy(cyc,np.abs(An),'o-',color="#2e86de",lw=1.8,ms=5,label="iterated FEW (no error)")
axc.semilogy(np.arange(len(An_e)),np.abs(An_e),'s--',color="#27ae60",lw=1.6,ms=4,label="iterated FEW (+20% fold err)")
axc.semilogy(np.arange(len(Ans)),np.abs(Ans),'^:',color="#d1495b",lw=1.8,ms=5,label="single FWE (+20% fold err)")
axc.set_xlabel("cycle n"); axc.set_ylabel("|A|  unstable-mode amplitude")
axc.set_title("(c) robustness — iterated converges w/ error; single FWE diverges"); axc.legend(fontsize=7.8,loc="upper left"); axc.grid(alpha=0.2,which="both")

plt.tight_layout()
out="docs/FEW_반복_상태공간_R0.7.png"; plt.savefig(out,dpi=145,bbox_inches="tight"); print("saved",out)
