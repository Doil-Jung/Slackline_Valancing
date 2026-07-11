# 안정라인 기준 회전좌표에서 FWE 곡선 분리 시각화 + 두 축의 (phi,beta,phidot,betadot) 선형결합 주석
#   좌: 기존 예측점 평면(곡선이 안정라인 방향으로 눌려 겹침)
#   우: 라인방향 s / 라인수직 이탈거리 n(=2A/|u2|) 로 회전 -> n축 독립 확대로 분리
# u2 ⊥ v1 이므로 n은 예측점의 '안정라인까지 유클리드 수직거리' = 2A/|u2| 와 일치.
# n,s 축은 모두 (phi,beta,phidot,betadot)의 선형결합(계수는 회전벡터 nhat,dhat와 1/lam에서 유도).
# 유한시간 완전사이클(사람스케일 R=0.7).  ->  docs/예측점평면_회전확대_R0.7.png
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def _set_korean_font():
    import os
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "C:/Windows/Fonts/malgun.ttf"]:
        if os.path.exists(p):
            try: fm.fontManager.addfont(p)
            except Exception: pass
    for name in ["Malgun Gothic","NanumGothic","Noto Sans CJK KR","Noto Sans CJK JP","AppleGothic"]:
        if any(name==f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"]=name; break
    plt.rcParams["axes.unicode_minus"]=False
_set_korean_font()

Mt=75.0;h=0.96;ps=72.0;R=0.7;g=9.81;Jb=89.03;mu=4.33
beta_i=np.deg2rad(2.0);Tf=0.15;Te=0.15
M22=np.array([[Mt*R**2,R*ps],[R*ps,Jb]]);G22=np.array([[-Mt*g*R,0],[0,ps*g]])
Mi=np.linalg.inv(M22);A2=Mi@G22
w,V=np.linalg.eig(A2);si=np.argmin(w);ui=np.argmax(w)
omega=np.sqrt(-w[si]);lam=np.sqrt(w[ui]);Tosc=2*np.pi/omega
v_s=V[:,si];r=v_s[0]/v_s[1];Vinv=np.linalg.inv(V);u2=Vinv[ui,:]
Tw=1*Tosc

def sig(u):u=np.clip(u,0,1);return np.where(u<=0.5,2*u**2,1-2*(1-u)**2)
def sig_ddu(u):return np.where((u>0)&(u<=0.5),4.0,np.where((u>0.5)&(u<1.0),-4.0,0.0))
def ddelta(t,df):
    if t<Tf:return df*sig_ddu(t/Tf)/Tf**2
    elif t<Tf+Tw:return 0.0
    else:return -df*sig_ddu((t-Tf-Tw)/Te)/Te**2
def rhs(t,y,df):q=y[:2];qd=y[2:];qdd=Mi@(G22@q+np.array([0.0,-mu*ddelta(t,df)]));return[qd[0],qd[1],qdd[0],qdd[1]]
def rhs_free(t,y):return[y[2],y[3],*(Mi@(G22@y[:2]))]
def Ac(q,qd):et=Vinv@q;etd=Vinv@qd;return 0.5*(et[ui]+etd[ui]/lam)
def integ(df,n=4000):
    T=Tf+Tw+Te;ts=np.linspace(0,T,n)
    s=solve_ivp(rhs,[0,T],[0,beta_i,0,0],args=(df,),t_eval=ts,max_step=T/n,rtol=1e-10,atol=1e-12)
    return ts,s.y,Ac(s.y[:2,-1],s.y[2:,-1])
_,_,A0=integ(0.0);_,_,A1=integ(0.1);dfs=-A0*0.1/(A1-A0)
ts,Y,Ae=integ(dfs)
def qpred(Y):return Y[:2]+Y[2:]/lam
QP=qpred(Y)
sp=solve_ivp(rhs_free,[0,2*Tosc],list(Y[:,-1]),t_eval=np.linspace(0,2*Tosc,2000),rtol=1e-10,atol=1e-12)
QPp=qpred(sp.y)
eps=-h/(r*R+h)
qn=np.array([h*(1+eps)*beta_i/R,-eps*beta_i])
sd=solve_ivp(rhs_free,[0,2*Tosc],[qn[0],qn[1],0,0],t_eval=np.linspace(0,2*Tosc,2000),rtol=1e-10,atol=1e-12)
QPd=qpred(sd.y)
i_f=np.searchsorted(ts,Tf);i_w=np.searchsorted(ts,Tf+Tw)
D=np.rad2deg
dhat=np.array([1.0,r]);dhat/=np.linalg.norm(dhat)   # (beta,phi) 라인방향
nhat=np.array([-r,1.0]);nhat/=np.linalg.norm(nhat)  # (beta,phi) 수직방향
def rot(QP):
    x=D(QP[1]);y=D(QP[0])
    return x*dhat[0]+y*dhat[1], x*nhat[0]+y*nhat[1]
CF,CW,CE,CP="#d1495b","#2e86de","#e1a100","#8e44ad"

# 축 = (phi,beta,phidot,betadot) 선형결합 계수 (그림 rot()과 동일 부호)
cn=[nhat[1], nhat[0], nhat[1]/lam, nhat[0]/lam]
cs=[dhat[1], dhat[0], dhat[1]/lam, dhat[0]/lam]
NM=["φ","β","φ̇","β̇"]
def fmt(cs):return "  ".join("%+.3f·%s"%(c,nm) for c,nm in zip(cs,NM))

fig,(axL,axR)=plt.subplots(1,2,figsize=(14.2,6.6))
bb=np.linspace(-40,40,80)
axL.plot(bb,r*bb,color="seagreen",lw=2.2,ls="--",label="안정모드 라인 (A=0)")
axL.plot(D(QP[1,:i_f+1]),D(QP[0,:i_f+1]),color=CF,lw=3,label="Fold 접기")
axL.plot(D(QP[1,i_f:i_w+1]),D(QP[0,i_f:i_w+1]),color=CW,lw=3,label="Wait 대기")
axL.plot(D(QP[1,i_w:]),D(QP[0,i_w:]),color=CE,lw=3,label="Extend 펴기")
axL.plot(D(QPp[1]),D(QPp[0]),color=CP,lw=1.8,label="post-cycle 이후")
axL.plot(D(QPd[1]),D(QPd[0]),color="0.45",lw=1.4,ls=":",label="단발 eps* 접기")
axL.plot(D(QP[1,0]),D(QP[0,0]),"ko",ms=8)
axL.set_xlim(-17,13);axL.set_ylim(-17,20);axL.grid(alpha=0.15)
axL.axhline(0,color="0.6",lw=0.5);axL.axvline(0,color="0.6",lw=0.5)
axL.set_xlabel("beta_pred [deg]");axL.set_ylabel("phi_pred [deg]")
axL.set_title("(a) 기존 예측점 평면\n곡선들이 안정라인 방향으로 눌려 겹쳐 보임")
axL.legend(fontsize=8.2,loc="lower left")
sF,nF=rot(QP[:,:i_f+1]);sW,nW=rot(QP[:,i_f:i_w+1]);sE,nE=rot(QP[:,i_w:])
sP,nP=rot(QPp);sD,nD=rot(QPd);s0,n0=rot(QP[:,:1])
axR.axhline(0,color="seagreen",lw=2.4,ls="--",label="안정모드 라인 (n=0)")
axR.plot(sF,nF,color=CF,lw=3,label="Fold 접기: 라인 넘어감")
axR.plot(sW,nW,color=CW,lw=3,label="Wait 대기: n=2A e^(λt)로 이탈")
axR.plot(sE,nE,color=CE,lw=3,label="Extend 펴기: 라인으로 복귀")
axR.plot(sP,nP,color=CP,lw=2.2,label="post-cycle: 라인 위 (n≈0)")
axR.plot(sD,nD,color="0.45",lw=1.8,ls=":",label="단발 eps* 접기: 항상 라인 위")
axR.plot(s0,n0,"ko",ms=8);axR.annotate("start",(s0[0],n0[0]),textcoords="offset points",xytext=(8,6),fontsize=9)
axR.grid(alpha=0.15)
sr=np.ptp(np.concatenate([sF,sW,sE]));nr=np.ptp(np.concatenate([nF,nW,nE]))
axR.set_xlabel("s : 안정라인 방향 좌표 [deg]")
axR.set_ylabel("n : 라인과 수직인 이탈거리  2A/|u2| [deg]")
axR.set_title("(b) 안정라인 기준 회전 + 수직방향 확대 (약 %d배)\n두 축은 (φ, β, φ̇, β̇)의 선형결합"%(round(sr/nr)))
axR.legend(fontsize=8.0,loc="upper left")
box=("n축 (수직·불안정, =0이면 포획):\n   n = %s\n"
     "s축 (수평·안정):\n   s = %s\n"
     "(각도 deg, 각속도 deg/s, λ=%.2f/s;  예: φ_pred=φ+φ̇/λ)")%(fmt(cn),fmt(cs),lam)
axR.text(0.015,0.02,box,transform=axR.transAxes,fontsize=8.6,va="bottom",ha="left",
         bbox=dict(boxstyle="round",fc="#fffdf5",ec="0.55",alpha=0.95))
plt.tight_layout();out="docs/예측점평면_회전확대_R0.7.png";plt.savefig(out,dpi=140)
print("saved",out)
print("n:",fmt(cn)); print("s:",fmt(cs))
