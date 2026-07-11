# 예측점 평면(q_pred = q + qdot/lam)에서 본 FWE 시나리오 5종 그림 생성
#   S1 순간 단일접기 (eps*, 정지출발)         -> docs/예측점평면_S1_순간단일접기.png
#   S2 순간 완전사이클 (과접기-대기-펴기)     -> docs/예측점평면_S2_순간완전사이클.png
#   S3 유한시간 단일접기 (gamma 보정)         -> docs/예측점평면_S3_유한시간접기.png
#   S4 유한시간 완전사이클                    -> docs/예측점평면_S4_유한시간완전사이클.png
#   S5 반복 사이클 (실행오차 15% 흡수)        -> docs/예측점평면_S5_반복.png
# 규약: 안정모드 직선(초록 파선) = 판정선(직선이탈거리 = 2A/|u2|), 회색 = CoM 등고선(기울기 -h/R),
#       Fold 빨강 / Wait 파랑 / Extend 금색 / 자유운동 보라. 사람스케일 R=0.7, beta_i=2deg.
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

Mt=75.0; h=0.96; ps=72.0; R=0.7; g=9.81; Jb=89.03; mu=4.33
beta_i=np.deg2rad(2.0)
M22=np.array([[Mt*R**2,R*ps],[R*ps,Jb]]); G22=np.array([[-Mt*g*R,0],[0,ps*g]])
Mi=np.linalg.inv(M22); A2=Mi@G22
w,V=np.linalg.eig(A2); si=np.argmin(w); ui=np.argmax(w)
omega=np.sqrt(-w[si]); lam=np.sqrt(w[ui]); Tosc=2*np.pi/omega
v_s=V[:,si]; r=v_s[0]/v_s[1]; Vinv=np.linalg.inv(V); u2=Vinv[ui,:]
e=np.array([h/R,-1.0]); Jbh=Jb-Mt*h**2   # =19.91
D=np.rad2deg
print(f"omega={omega:.3f} lam={lam:.3f} Tosc={Tosc:.3f} r={r:.3f} Jb-Mt h^2={Jbh:.2f}")

def Acoef(q,qd): eta=Vinv@q; etad=Vinv@qd; return 0.5*(eta[ui]+etad[ui]/lam)
def qpred(Y): return Y[:2]+Y[2:]/lam
def rhs_free(t,y): return [y[2],y[3],*(Mi@(G22@y[:2]))]
def free(y0,T,n=1500):
    s=solve_ivp(rhs_free,[0,T],list(y0),t_eval=np.linspace(0,T,n),max_step=T/n,rtol=1e-10,atol=1e-12)
    return s.t,s.y
def sig(u): u=np.clip(u,0,1); return np.where(u<=0.5,2*u**2,1-2*(1-u)**2)
def sig_ddu(u): return np.where((u>0)&(u<=0.5),4.0,np.where((u>0.5)&(u<1.0),-4.0,0.0))
def jump_eps(q,eps):     # 순간접기: beta -> -eps*beta, CoM 등고선 이동, 속도 보존
    return q+(1+eps)*q[1]*e
def dd_fold(t,df,Tf): return df*sig_ddu(t/Tf)/Tf**2

# ---------- 공통 플롯 ----------
def plane(ax,xlim,ylim):
    bb=np.linspace(-40,40,80)
    for C in np.linspace(-0.5,0.5,51):
        ax.plot(bb,D((C-h*np.deg2rad(bb))/R),color="0.91",lw=0.7,zorder=0)
    ax.plot(bb,r*bb,color="seagreen",lw=2.2,ls="--",zorder=2,label="stable-mode line (A=0)")
    ax.axhline(0,color="0.6",lw=0.6); ax.axvline(0,color="0.6",lw=0.6); ax.grid(alpha=0.10)
    ax.set_xlabel("β_pred [deg]"); ax.set_ylabel("φ_pred [deg]")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
CF,CW,CE,CP="#d1495b","#2e86de","#e1a100","#8e44ad"

def fit_lims(ax,xs,ys,padx=0.07,pady=0.11):
    # 그려진 궤적 데이터를 모두 담도록 축 범위를 '확장만' 한다(기존 범위를 줄이지 않음).
    xs=np.concatenate([np.atleast_1d(np.asarray(a,dtype=float)) for a in xs])
    ys=np.concatenate([np.atleast_1d(np.asarray(a,dtype=float)) for a in ys])
    x0,x1=np.min(xs),np.max(xs); y0,y1=np.min(ys),np.max(ys)
    cx0,cx1=ax.get_xlim(); cy0,cy1=ax.get_ylim()
    dx=(x1-x0)*padx+0.5; dy=(y1-y0)*pady+0.5
    ax.set_xlim(min(cx0,x0-dx),max(cx1,x1+dx)); ax.set_ylim(min(cy0,y0-dy),max(cy1,y1+dy))

def tsax(ax2,segs,delta_segs,title):
    for t,b,ph,c in segs: ax2.plot(t,b,color=c,lw=2.0); ax2.plot(t,ph,color=c,lw=1.2,alpha=0.45)
    for t,d in delta_segs: ax2.plot(t,d,color="0.35",lw=1.5,ls="--")
    ax2.plot([],[],color="k",lw=2.0,label="β (bold) / φ (thin)")
    ax2.plot([],[],color="0.35",lw=1.5,ls="--",label="δ (hip)")
    ax2.axhline(0,color="0.6",lw=0.6); ax2.grid(alpha=0.15)
    ax2.set_xlabel("time [s]"); ax2.set_ylabel("angle [deg]"); ax2.set_title(title); ax2.legend(fontsize=8.5)

# ================= S1 순간 단일접기 =================
eps_s=-h/(r*R+h); q0=np.array([0.0,beta_i]); qd0=np.zeros(2)
q1=jump_eps(q0,eps_s); df1=(1+eps_s)*beta_i*Jbh/mu
t1,Y1=free(np.concatenate([q1,qd0]),3*Tosc,3000)
A1=Acoef(Y1[:2,0],Y1[2:,0]); QP1=qpred(Y1)
print(f"[S1] eps*={eps_s:.3f} delta_f={D(df1):.2f}deg A={A1:.1e}")
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.2,6.0))
plane(ax,(-7,4),(-8,9))
ax.annotate("",xy=(D(q1[1]),D(q1[0])),xytext=(D(q0[1]),D(q0[0])),
            arrowprops=dict(arrowstyle="-|>",color=CF,lw=2.4),zorder=5)
ax.text(D((q0[1]+q1[1])/2)+0.25,D((q0[0]+q1[0])/2),"instant fold ε*\n(along CoM contour)",color=CF,fontsize=9)
ax.plot(D(QP1[1]),D(QP1[0]),color=CP,lw=2.4,zorder=4,label="after fold: slides ALONG line forever")
ax.plot(D(q0[1]),D(q0[0]),"ko",ms=8,zorder=6); ax.text(D(q0[1])+0.15,D(q0[0])+0.3,"start (rest, β=2°)",fontsize=9)
ax.plot(D(q1[1]),D(q1[0]),"s",color=CF,ms=7,zorder=6)
fit_lims(ax,[D(QP1[1]),[D(q0[1]),D(q1[1])]],[D(QP1[0]),[D(q0[0]),D(q1[0])]])
ax.set_title(f"S1  instant single fold ε*={eps_s:.2f} (δ_f={D(df1):.1f}°) — lands ON line, |A|={abs(A1):.0e}")
ax.legend(loc="lower left",fontsize=8.5)
tsax(ax2,[(t1,D(Y1[1]),D(Y1[0]),CP)],[(t1,np.full_like(t1,D(df1)))],
     "bounded oscillation, never extends (δ stays folded)")
plt.tight_layout(); plt.savefig("docs/예측점평면_S1_순간단일접기.png",dpi=140); plt.close()

# ================= S2 순간 완전사이클 =================
Tw=1*Tosc
def cycle_instant(epsv):
    qA=jump_eps(q0,epsv); dq=qA-q0
    t,Y=free(np.concatenate([qA,qd0]),Tw,1500)
    qB=Y[:2,-1]-dq; qdB=Y[2:,-1]     # 펴기 = 접기 킥의 정확한 역벡터
    return qA,dq,t,Y,qB,qdB,Acoef(qB,qdB)
A0=cycle_instant(0.0)[-1]; Aa=cycle_instant(1.0)[-1]; eps_c=-A0/(Aa-A0)*1.0
eps_closed=(1+eps_s)/(1-np.exp(-lam*Tw))-1
qA,dq,tw,Yw,qB,qdB,Ac=cycle_instant(eps_c)
tp,Yp=free(np.concatenate([qB,qdB]),2*Tosc,2000)
QPw=qpred(Yw); QPp=qpred(Yp); qpB=qB+qdB/lam
print(f"[S2] eps_FWE(num)={eps_c:.3f} vs closed {eps_closed:.3f} ({100*(eps_closed-eps_c)/eps_c:+.2f}%), A_final={Ac:.1e}")
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.2,6.0))
plane(ax,(-16,10),(-13,18))
ax.annotate("",xy=(D(qA[1]),D(qA[0])),xytext=(D(q0[1]),D(q0[0])),
            arrowprops=dict(arrowstyle="-|>",color=CF,lw=2.4),zorder=5)
ax.text(D(qA[1])-4.2,D(qA[0])+0.4,f"over-fold ε={eps_c:.2f}\nlands PAST the line",color=CF,fontsize=9)
ax.plot(D(QPw[1]),D(QPw[0]),color=CW,lw=2.6,zorder=4,label="Wait: leaves line as 2A·e^(λt)")
ax.annotate("",xy=(D(qpB[1]),D(qpB[0])),xytext=(D(QPw[1,-1]),D(QPw[0,-1])),
            arrowprops=dict(arrowstyle="-|>",color=CE,lw=2.4),zorder=5)
ax.text(D(qpB[1])+0.4,D(qpB[0])-1.6,"instant extend\n(reverse kick) → ON line",color=CE,fontsize=9)
ax.plot(D(QPp[1]),D(QPp[0]),color=CP,lw=2.0,zorder=3,label="post-cycle (δ=0): slides ALONG line")
ax.plot(D(q0[1]),D(q0[0]),"ko",ms=8,zorder=6)
ax.plot(D(qpB[1]),D(qpB[0]),"*",color=CE,ms=16,zorder=6)
fit_lims(ax,[D(QPw[1]),D(QPp[1]),[D(q0[1]),D(qA[1]),D(qpB[1])]],
            [D(QPw[0]),D(QPp[0]),[D(q0[0]),D(qA[0]),D(qpB[0])]])
ax.set_title(f"S2  instant fold–wait–extend, T_w=1·T_osc, ε_FWE={eps_c:.2f} (closed-form {eps_closed:.2f})")
ax.legend(loc="lower left",fontsize=8.5)
dfc=(1+eps_c)*beta_i*Jbh/mu
tsax(ax2,[(tw,D(Yw[1]),D(Yw[0]),CW),(tw[-1]+tp,D(Yp[1]),D(Yp[0]),CP)],
     [(tw,np.full_like(tw,D(dfc))),(tw[-1]+tp,np.zeros_like(tp))],
     "over-fold grows during Wait, extend cancels it (δ→0)")
plt.tight_layout(); plt.savefig("docs/예측점평면_S2_순간완전사이클.png",dpi=140); plt.close()

# ================= S3 유한시간 단일접기 =================
Tf=0.25
def rhs_f(t,y,df,Tf):
    q=y[:2]; qd=y[2:]; qdd=Mi@(G22@q+np.array([0.0,-mu*dd_fold(t,df,Tf)])); return [qd[0],qd[1],qdd[0],qdd[1]]
def foldA(df):
    s=solve_ivp(rhs_f,[0,Tf],[0,beta_i,0,0],args=(df,Tf),t_eval=np.linspace(0,Tf,600),max_step=Tf/600,rtol=1e-10,atol=1e-12)
    return s,Acoef(s.y[:2,-1],s.y[2:,-1])
_,Af0=foldA(0.0); _,Af1=foldA(0.3); df3=-Af0*0.3/(Af1-Af0)
s3,A3=foldA(df3); gam=df3/df1
t3,Y3=free(s3.y[:,-1],3*Tosc,3000)
QP3f=qpred(s3.y); QP3=qpred(Y3)
print(f"[S3] Tf={Tf}s delta_f={D(df3):.2f}deg (instant {D(df1):.2f} -> gamma={gam:.3f}) A={A3:.1e}")
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.2,6.0))
plane(ax,(-13,6),(-9,17))
ax.plot(D(QP3f[1]),D(QP3f[0]),color=CF,lw=2.8,zorder=5,label=f"finite-time fold T_f={Tf}s (velocity spike ×1/λ)")
ax.plot(D(QP3[1]),D(QP3[0]),color=CP,lw=2.2,zorder=4,label="after fold: slides ALONG line")
ax.plot(D(QP3f[1,0]),D(QP3f[0,0]),"ko",ms=8,zorder=6)
ax.plot(D(QP3f[1,-1]),D(QP3f[0,-1]),"*",color=CF,ms=15,zorder=6)
fit_lims(ax,[D(QP3f[1]),D(QP3[1])],[D(QP3f[0]),D(QP3[0])])
ax.set_title(f"S3  finite-time single fold — must over-fold by γ={gam:.2f} (δ_f {D(df1):.1f}°→{D(df3):.1f}°), |A|={abs(A3):.0e}")
ax.legend(loc="lower left",fontsize=8.5)
tsax(ax2,[(s3.t,D(s3.y[1]),D(s3.y[0]),CF),(Tf+t3,D(Y3[1]),D(Y3[0]),CP)],
     [(s3.t,D(df3*sig(s3.t/Tf))),(Tf+t3,np.full_like(t3,D(df3)))],
     "fold over 0.25 s then hold — bounded")
plt.tight_layout(); plt.savefig("docs/예측점평면_S3_유한시간접기.png",dpi=140); plt.close()

# ================= S4 유한시간 완전사이클 =================
Tf4,Te4=0.15,0.15
def dd_cyc(t,df):
    if t<Tf4: return df*sig_ddu(t/Tf4)/Tf4**2
    elif t<Tf4+Tw: return 0.0
    else: return -df*sig_ddu((t-Tf4-Tw)/Te4)/Te4**2
def rhs_c(t,y,df):
    q=y[:2]; qd=y[2:]; qdd=Mi@(G22@q+np.array([0.0,-mu*dd_cyc(t,df)])); return [qd[0],qd[1],qdd[0],qdd[1]]
def cycA(df,n=3000):
    T=Tf4+Tw+Te4
    s=solve_ivp(rhs_c,[0,T],[0,beta_i,0,0],args=(df,),t_eval=np.linspace(0,T,n),max_step=T/n,rtol=1e-10,atol=1e-12)
    return s,Acoef(s.y[:2,-1],s.y[2:,-1])
_,Ac0=cycA(0.0); _,Ac1=cycA(0.3); df4=-Ac0*0.3/(Ac1-Ac0)
s4,A4=cycA(df4); t4p,Y4p=free(s4.y[:,-1],2*Tosc,2000)
QP4=qpred(s4.y); QP4p=qpred(Y4p)
i_f=np.searchsorted(s4.t,Tf4); i_w=np.searchsorted(s4.t,Tf4+Tw)
print(f"[S4] delta_f={D(df4):.2f}deg A_final={A4:.1e}")
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.2,6.0))
plane(ax,(-17,13),(-17,20))
ax.plot(D(QP4[1,:i_f+1]),D(QP4[0,:i_f+1]),color=CF,lw=2.8,zorder=5,label="Fold (finite)")
ax.plot(D(QP4[1,i_f:i_w+1]),D(QP4[0,i_f:i_w+1]),color=CW,lw=2.6,zorder=4,label="Wait: leaves line as 2A·e^(λt)")
ax.plot(D(QP4[1,i_w:]),D(QP4[0,i_w:]),color=CE,lw=2.8,zorder=5,label="Extend (finite): lands ON line")
ax.plot(D(QP4p[1]),D(QP4p[0]),color=CP,lw=1.9,zorder=3,label="post-cycle: slides ALONG line")
ax.plot(D(QP4[1,0]),D(QP4[0,0]),"ko",ms=8,zorder=6)
ax.plot(D(QP4[1,-1]),D(QP4[0,-1]),"*",color=CE,ms=16,zorder=6)
fit_lims(ax,[D(QP4[1]),D(QP4p[1])],[D(QP4[0]),D(QP4p[0])])
ax.set_title(f"S4  finite-time full cycle (T_f={Tf4}s, T_w=1·T_osc, T_e={Te4}s), δ_f*={D(df4):.1f}°")
ax.legend(loc="lower left",fontsize=8.5)
tsax(ax2,[(s4.t,D(s4.y[1]),D(s4.y[0]),CW),(s4.t[-1]+t4p,D(Y4p[1]),D(Y4p[0]),CP)],
     [(s4.t,D(df4*np.array([sig(min(t,Tf4)/Tf4)-(sig((t-Tf4-Tw)/Te4) if t>=Tf4+Tw else 0) for t in s4.t])))],
     "full cycle then free — bounded, δ returns to 0")
plt.tight_layout(); plt.savefig("docs/예측점평면_S4_유한시간완전사이클.png",dpi=140); plt.close()

# ================= S5 반복 — 실행오차와 루프이득 조건 =================
# 접기 실행오차 비율 rho는 펴기 시점까지 e^{lam*Tw} 배로 증폭되고(대기 중 A(t)=A0 e^{lam t}),
# 다음 개입까지 e^{lam*Trest} 더 자란다 -> 반복이 유계이려면 루프이득 rho*e^{lam(Tw+Trest)} < 1.
# 같은 rho=5%: 빠른 스케줄(Tw=0.5 Tosc, Trest=0.25s)은 유계, 느슨한 스케줄(Tw=Tosc, Trest=Tosc)은 발산.
RHO=1.05; NCYC=6
def dd_cyc2(t,df,Twv):
    if t<Tf4: return df*sig_ddu(t/Tf4)/Tf4**2
    elif t<Tf4+Twv: return 0.0
    else: return -df*sig_ddu((t-Tf4-Twv)/Te4)/Te4**2
def rhs_c2(t,y,df,Twv):
    q=y[:2]; qd=y[2:]; qdd=Mi@(G22@q+np.array([0.0,-mu*dd_cyc2(t,df,Twv)])); return [qd[0],qd[1],qdd[0],qdd[1]]
def cyc_from(y0,df,Twv,n=2000):
    T=Tf4+Twv+Te4
    s=solve_ivp(rhs_c2,[0,T],list(y0),args=(df,Twv),t_eval=np.linspace(0,T,n),max_step=T/n,rtol=1e-10,atol=1e-12)
    return s
def df_exact(y0,Twv):
    sA=cyc_from(y0,0.0,Twv); sB=cyc_from(y0,0.3,Twv)
    A0=Acoef(sA.y[:2,-1],sA.y[2:,-1]); A1=Acoef(sB.y[:2,-1],sB.y[2:,-1])
    return -A0*0.3/(A1-A0)
def repeat_run(Twv,Trest,ncyc):
    segs=[]; y=np.array([0,beta_i,0,0.0]); tcur=0.0; dfs=[]
    for k in range(ncyc):
        df=df_exact(y,Twv)*RHO; dfs.append(df)
        s=cyc_from(y,df,Twv)
        dprof=D(df*np.array([sig(min(t,Tf4)/Tf4)-(sig((t-Tf4-Twv)/Te4) if t>=Tf4+Twv else 0) for t in s.t]))
        segs.append(("cyc",k,tcur+s.t,s.y.copy(),dprof)); tcur+=s.t[-1]; y=s.y[:,-1]
        if abs(y[1])>np.deg2rad(30): break
        tr,Yr=free(y,Trest,600)
        segs.append(("rest",k,tcur+tr,Yr,np.zeros_like(tr))); tcur+=Trest; y=Yr[:,-1]
    return segs,dfs
TwF,TrF=0.3*Tosc,0.15; TwS,TrS=1*Tosc,1*Tosc
gF=(RHO-1)*np.exp(lam*(TwF+TrF)); gS=(RHO-1)*np.exp(lam*(TwS+TrS))
segF,dfF=repeat_run(TwF,TrF,NCYC); segS,dfS=repeat_run(TwS,TrS,4)
bmaxF=max(np.max(np.abs(D(Y[1]))) for _,_,_,Y,_ in segF)
bmaxS=max(np.max(np.abs(D(Y[1]))) for _,_,_,Y,_ in segS)
print(f"[S5] rho=+5%: loop gain fast={gF:.2f} (Tw=0.5Tosc,Tr=0.25s) beta_max={bmaxF:.1f}deg | "
      f"slow={gS:.1f} (Tw=Tosc,Tr=Tosc) beta_max={bmaxS:.1f}deg (발산)")
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.2,6.0))
cols=plt.cm.viridis(np.linspace(0.05,0.85,NCYC))
nrm=np.linalg.norm(u2)
for kind,k,t,Y,dp in segF:
    dd=np.abs(u2@qpred(Y))/nrm
    ax.semilogy(t,D(dd)+1e-6,color=cols[k],lw=2.0 if kind=="cyc" else 1.3,
                alpha=1.0 if kind=="cyc" else 0.7,zorder=4,
                label=(f"cycle {k+1}" if kind=="cyc" else None))
for kind,k,t,Y,dp in segS:
    dd=np.abs(u2@qpred(Y))/nrm
    ax.semilogy(t,D(dd)+1e-6,color="0.55",lw=1.4,ls=":",zorder=3)
ax.semilogy([],[],color="0.55",lw=1.4,ls=":",label="slow schedule (diverges)")
tg=np.linspace(0,1.2,50)
ax.semilogy(3.0+tg,0.2*np.exp(lam*tg),color="k",lw=1.0,ls="--")
ax.text(3.55,0.9,"slope e^(λt)",fontsize=8.5,rotation=38)
ax.set_xlabel("time [s]"); ax.set_ylabel("distance of predicted point to stable line  2A/|u₂|  [deg]")
ax.grid(alpha=0.2,which="both"); ax.set_ylim(1e-3,1e3)
ax.set_title(f"S5  repetition with +5% fold execution error, fast schedule\n"
             f"(T_w=0.3·T_osc, T_rest=0.15 s): loop gain {gF:.2f} < 1 → re-captured each fold")
ax.legend(loc="upper left",fontsize=8.2,ncol=2)
for kind,k,t,Y,dp in segF:
    ax2.plot(t,D(Y[1]),color="#c0392b",lw=1.8); ax2.plot(t,dp,color="0.35",lw=1.2,ls="--")
for kind,k,t,Y,dp in segS:
    ax2.plot(t,D(Y[1]),color="0.55",lw=1.5,ls=":")
# 발산하는 느슨한 스케줄(dotted)은 프레임을 벗어나도 무방하나, 유계인 빠른 스케줄(β·δ)은 다 보이게 확장
_fast=np.concatenate([np.concatenate([np.atleast_1d(D(Y[1])),np.atleast_1d(dp)]) for _,_,_,Y,dp in segF])
_m=max(32.0,np.max(np.abs(_fast))*1.12)
ax2.set_ylim(-_m,_m)
ax2.plot([],[],color="#c0392b",lw=1.8,label="β, fast schedule (bounded)")
ax2.plot([],[],color="0.55",lw=1.5,ls=":",label=f"β, slow schedule T_w=T_osc, T_rest=T_osc\n(loop gain {gS:.0f} → diverges)")
ax2.plot([],[],color="0.35",lw=1.2,ls="--",label="δ (hip)")
ax2.axhline(0,color="0.6",lw=0.6); ax2.grid(alpha=0.15); ax2.set_xlabel("time [s]"); ax2.set_ylabel("angle [deg]")
ax2.set_title("same +5% error: schedule decides survival\n(error amplifies e^(λ(T_w+T_rest)) between interventions)")
ax2.legend(fontsize=8.3,loc="lower left")
plt.tight_layout(); plt.savefig("docs/예측점평면_S5_반복.png",dpi=140); plt.close()
print("saved 5 figures to docs/")
