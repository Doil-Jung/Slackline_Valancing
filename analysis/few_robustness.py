import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
Mt=75.0; h=0.96; ps=72.0; R=0.7; g=9.81; Jb=89.03; mu=4.33
M22=np.array([[Mt*R**2,R*ps],[R*ps,Jb]]); G22=np.array([[-Mt*g*R,0],[0,ps*g]])
Mi=np.linalg.inv(M22); w,V=np.linalg.eig(Mi@G22); si=np.argmin(w); ui=np.argmax(w)
lam=np.sqrt(w[ui]); Vinv=np.linalg.inv(V); k=0.5*(Vinv[ui,:]@Mi)[1]*mu
def B(x): return 4*(np.exp(x/2)-1)**2/x**2
Th=0.08; Tw=0.12
G=np.exp(lam*(2*Th+Tw)); gg=k*(np.exp(lam*(Th+Tw))*B(lam*Th)-B(lam*Th)); gdead=G/gg
invG=1.0/G
print(f"G={G:.3f}  1/G={invG:.3f}  안정창 rho(1+e) in ({1-invG:.3f},{1+invG:.3f})")

# ---- (a) 해석: 수렴영역 (rho, e) + 로버스트니스 곡선 ----
rho=np.linspace(0.55,1.5,400)
elow=(1-invG)/rho-1; ehigh=(1+invG)/rho-1
# 최악방향 대칭 견딤 |e| (0이 창 안일 때)
robust=np.where((elow<0)&(ehigh>0),np.minimum(-elow,ehigh),np.nan)

# ---- (b) 몬테카를로 생존확률 vs rho ----
def surv_prob(rho,esd,N=300,trials=1500,seed=1):
    rng=np.random.default_rng(seed); A=np.full(trials,0.004); dfmax=np.deg2rad(55)
    alive=np.ones(trials,bool)
    for n in range(N):
        Ameas=A*(1+rng.normal(0,0.04,trials))+rng.normal(0,8e-4,trials)
        df=np.clip(rho*gdead*Ameas,-dfmax,dfmax)*(1+rng.normal(0,esd,trials))
        A=G*A-gg*df+rng.normal(0,8e-4,trials)
        alive&=np.abs(A)<0.15
    return alive.mean()
rgrid=np.linspace(0.7,1.35,22)
curves={}
for esd in [0.15,0.30,0.45]:
    curves[esd]=[surv_prob(r,esd) for r in rgrid]
    print(f"esd={esd}: peak rho={rgrid[int(np.argmax(curves[esd]))]:.2f} surv={max(curves[esd]):.2f}")

fig,(ax,ax2)=plt.subplots(1,2,figsize=(13,5.2))
# (a)
ax.fill_between(rho,elow*100,ehigh*100,where=(ehigh>elow),color="#a9d6a9",alpha=0.5,label="converges (|m|<1)")
ax.plot(rho,elow*100,color="0.4",lw=1); ax.plot(rho,ehigh*100,color="0.4",lw=1)
ax.plot(rho,robust*100,color="#c0392b",lw=2.5,label="worst-case tolerable |e| (robustness)")
imax=np.nanargmax(robust); ax.plot(rho[imax],robust[imax]*100,'o',color="#c0392b",ms=9)
ax.annotate(f"peak at rho=1.00\n±{robust[imax]*100:.0f}% (=1/G)",xy=(rho[imax],robust[imax]*100),
            xytext=(1.12,robust[imax]*100-2),fontsize=9,color="#c0392b",
            arrowprops=dict(arrowstyle="->",color="#c0392b"))
ax.axvline(1.0,color="0.5",ls=":",lw=1); ax.axhline(0,color="0.6",lw=0.6)
ax.set_xlabel("rho  (control gain / deadbeat)"); ax.set_ylabel("fold error e  [%]")
ax.set_title("(a) SYSTEMATIC fold error tolerated (worst-case)\nanalytic m=G[1-rho(1+e)] -> widest at rho=1")
ax.legend(loc="upper right",fontsize=9); ax.grid(alpha=0.15); ax.set_ylim(-70,90); ax.set_xlim(0.6,1.5)
# (b)
cols={0.15:"#2e86de",0.30:"#e1a100",0.45:"#d1495b"}
for esd in [0.15,0.30,0.45]:
    ax2.plot(rgrid,curves[esd],'o-',color=cols[esd],lw=1.8,ms=4,label=f"random fold err sd={esd:.0%}")
ax2.axvline(1.0,color="0.5",ls=":",lw=1.2)
ax2.set_xlabel("rho"); ax2.set_ylabel("survival probability (300 cycles)")
ax2.set_title("(b) RANDOM per-cycle error: survival vs rho\n(noise gain ~G*rho -> peaks slightly below 1)")
ax2.legend(fontsize=9,loc="lower center"); ax2.grid(alpha=0.2); ax2.set_ylim(-0.03,1.03)
plt.tight_layout(); fig.suptitle("Iterated FEW robustness vs control gain rho  (both criteria favor rho in [0.8,1.0]; over-fold rho>1.1 worse)",fontsize=10.5,y=1.02)
out="docs/FEW_로버스트니스_곡선_R0.7.png"; plt.savefig(out,dpi=145,bbox_inches="tight"); print("saved",out)
