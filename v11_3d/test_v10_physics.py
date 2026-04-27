# -*- coding: utf-8 -*-
import math, os

F = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_out.txt')
O = open(F, 'w', encoding='utf-8')

try:
    m1,m2 = 0.10, 0.32
    L1,lc1,lc2 = 0.232, 0.182, 0.095
    I1,I2 = 0.0003, 0.0009
    R,g = 0.336, 9.81
    bp,ba,bt = 0.001, 0.001, 0.05
    Mt=m1+m2; p1=m1*lc1+m2*L1; p2=m2*lc2; p3=m2*L1*lc2
    M11=Mt*R**2; M22=m1*lc1**2+I1+m2*L1**2; M33=m2*lc2**2+I2
    tmax=3.0

    # Pre-computed K from DARE (240Hz)
    K240 = [-2.86, -3.48, -1.09, -0.43, -0.38, -0.12]
    # Pre-computed K from DARE (1000Hz) - approximate same
    K1k = [-2.86, -3.48, -1.09, -0.43, -0.38, -0.12]

    def deriv(s,tau):
        phi,al,th,pd,ad,td=s
        sPA=math.sin(phi+al);cPA=math.cos(phi+al)
        sPT=math.sin(phi+th);cPT=math.cos(phi+th)
        sAT=math.sin(al-th);cAT=math.cos(al-th)
        M=[[M11,R*p1*cPA,R*p2*cPT],[R*p1*cPA,M22,p3*cAT],[R*p2*cPT,p3*cAT,M33]]
        f1=R*p1*sPA*ad**2+R*p2*sPT*td**2-Mt*g*R*math.sin(phi)-bp*pd
        f2=R*p1*sPA*pd**2-p3*sAT*td**2+p1*g*math.sin(al)-tau-ba*ad
        f3=R*p2*sPT*pd**2+p3*sAT*ad**2+p2*g*math.sin(th)+tau-bt*td
        a,b,c=M[0][0],M[0][1],M[0][2]
        d,e,f=M[1][1],M[1][2],M[2][2]
        det=a*(d*f-e*e)-b*(b*f-e*c)+c*(b*e-d*c)
        iv=1.0/det
        A11=d*f-e*e;A12=-(b*f-c*e);A13=b*e-c*d
        A22=a*f-c*c;A23=-(a*e-b*c);A33=a*d-b*b
        return [pd,ad,td,
                (A11*f1+A12*f2+A13*f3)*iv,
                (A12*f1+A22*f2+A23*f3)*iv,
                (A13*f1+A23*f2+A33*f3)*iv]

    def rk4(s,tau,dt):
        k1=deriv(s,tau)
        s2=[s[i]+.5*dt*k1[i] for i in range(6)]
        k2=deriv(s2,tau)
        s3=[s[i]+.5*dt*k2[i] for i in range(6)]
        k3=deriv(s3,tau)
        s4=[s[i]+dt*k3[i] for i in range(6)]
        k4=deriv(s4,tau)
        return [s[i]+(dt/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i]) for i in range(6)]

    def dot(K,s):
        return sum(K[i]*s[i] for i in range(6))

    def run(label,dt,th0,K):
        O.write(f"\n=== {label} ===\n")
        s=[0,0,math.radians(th0),0,0,0]
        t=0.0; steps=int(3.0/dt); interval=max(1,int(0.05/dt))
        O.write("  t     | phi    alpha  theta  | phid   alpd   thed   | tau     raw\n")
        for step in range(steps):
            tr=-dot(K,s)
            tc=max(-tmax,min(tmax,tr))
            if step<10 or step%interval==0:
                if t<=2.0:
                    d=math.degrees
                    O.write(f" {t:5.3f} | {d(s[0]):+6.2f} {d(s[1]):+6.2f} {d(s[2]):+6.2f} | {d(s[3]):+7.1f} {d(s[4]):+7.1f} {d(s[5]):+7.1f} | {tc:+.3f} {tr:+.3f}\n")
            s=rk4(s,tc,dt)
            t+=dt
            if abs(s[1])>math.radians(90) or abs(s[2])>math.radians(90):
                O.write(f" FALLEN at t={t:.3f}s (a={d(s[1]):+.1f} th={d(s[2]):+.1f})\n")
                return
        O.write(f" SURVIVED 3s! phi={d(s[0]):+.2f} a={d(s[1]):+.2f} th={d(s[2]):+.2f}\n")

    run("RK4@240Hz th=3",  1/240, 3.0, K240)
    run("RK4@1000Hz th=3", 0.001, 3.0, K1k)
    run("RK4@240Hz th=8.6",  1/240, 8.6, K240)
    run("RK4@1000Hz th=8.6", 0.001, 8.6, K1k)

    # Also test FREE FALL (no control) to verify physics speed
    O.write("\n=== FREE FALL (no control) th=10 ===\n")
    s=[0,0,math.radians(10),0,0,0]
    t=0.0
    for step in range(240):
        d=math.degrees
        if step%12==0:
            O.write(f" {t:5.3f} | {d(s[0]):+6.2f} {d(s[1]):+6.2f} {d(s[2]):+6.2f} | {d(s[3]):+7.1f} {d(s[4]):+7.1f} {d(s[5]):+7.1f}\n")
        s=rk4(s,0,1/240)
        t+=1/240

except Exception as e:
    import traceback
    O.write(f"\nERROR: {e}\n{traceback.format_exc()}")
finally:
    O.close()
