"""4-Phase check: Robot with R=0.6m"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import time as _t

g=9.81
M1=0.1;m2=0.25;L1=0.2;L2=0.2;R=0.6
LC1=L1/2;LC2=L2/2;Mt=M1+m2
I1=M1*L1**2/12;I2=m2*L2**2/12
p1=M1*LC1+m2*L1;p2=m2*LC2;ps=p1+p2
h=(M1*LC1+m2*(L1+LC2))/Mt;cf=p2/ps
Jaa=M1*LC1**2+I1+m2*L1**2;Jtt=m2*LC2**2+I2
Jat=m2*L1*LC2;Jt=Jaa+Jtt+2*Jat
Csd=(-Jaa*p2+Jtt*p1+Jat*(p1-p2))/ps**2
M11=Mt*R**2;M12=R;M22=Jt/ps**2;dM=M11*M22-M12**2
i11=M22/dM;i12=-M12/dM;i22=M11/dM
gMR=g*Mt*R;gps=g/ps
we=np.sqrt(abs(i11*gMR));le=np.sqrt(abs(i22*gps))
Tq=np.pi/(2*we)

print(f"R=0.6m: w_eff={we:.2f} lam={le:.2f} 1/lam={1/le*1000:.0f}ms Tq={Tq*1000:.0f}ms")
print(f"  h={h:.4f} cf={cf:.4f}")
b0=np.radians(5)
for e in [4,6,8,10,12]:
    dd=np.degrees(h*(1+e)*b0/cf)
    tf_min = 2*dd/360*1000 if dd>0 else 0
    print(f"  eps={e}: d_fold={dd:.1f}deg  Tf_min(360d/s)={tf_min:.0f}ms  vs 1/lam={1/le*1000:.0f}ms")

def run(eps,Tf,tw1):
    s0=Mt*h*b0;df=h*(1+eps)*b0/cf;a=4*df/Tf**2
    t2=Tf+tw1;t3=t2+Tf;Te=t3+3*Tq
    def rhs(t,y):
        p,dp,s,ds=y
        if t<Tf/2:dd=+a
        elif t<Tf:dd=-a
        elif t<t2:dd=0
        elif t<t2+Tf/2:dd=-a
        elif t<t3:dd=+a
        else:dd=0
        return[dp,i11*(-gMR*p)+i12*(gps*s-Csd*dd),ds,i12*(-gMR*p)+i22*(gps*s-Csd*dd)]
    sol=solve_ivp(rhs,(0,Te),[0,0,s0,0],method='RK45',rtol=1e-8,atol=1e-10,max_step=Tf/20,dense_output=True)
    tw2=np.linspace(t3+0.002,Te,500)
    pw2=[sol.sol(t)[0] for t in tw2]
    for k in range(len(pw2)-1):
        if pw2[k]*pw2[k+1]<0:
            try:
                tz=brentq(lambda t:sol.sol(t)[0],tw2[k],tw2[k+1],xtol=1e-8)
                bf=sol.sol(tz)[2]/(Mt*h);return bf/b0,tz,np.degrees(df),np.degrees(2*df/Tf)
            except:pass
    return None,None,None,None

t0=_t.time()
conv=[]
for eps in np.arange(2,15,1):
    for Tf in [0.02,0.05,0.08,0.10,0.15]:
        for tw1 in np.arange(0.01,2.0,0.05):
            r,tt,dd,vp=run(eps,Tf,tw1)
            if r is not None and abs(r)<1:
                conv.append((eps,Tf,tw1,r,tt,dd,vp,vp<=360))
    print(f"  eps={eps:.0f} [{_t.time()-t0:.1f}s] conv={len(conv)}")

print(f"\nTotal: {len(conv)} conv")
sok=[c for c in conv if c[7]]
print(f"Servo-OK: {len(sok)}")

for label,lst in [("ALL",conv),("SERVO-OK",sok)]:
    if not lst: continue
    lst.sort(key=lambda x:abs(x[3]))
    print(f"\n--- {label} top 15 ---")
    for e,tf,tw1,r,tt,d,v,s in lst[:15]:
        tag = "OK" if s else ""
        print(f"  eps={e:.0f} Tf={tf*1000:.0f}ms Tw1={tw1*1000:.0f}ms | r={r:.4f} T={tt:.2f}s | d={d:.0f}d v={v:.0f}d/s {tag}")
print(f"\nDone in {_t.time()-t0:.1f}s")
