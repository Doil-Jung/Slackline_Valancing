# -*- coding: utf-8 -*-
import csv, math, os

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compare_out.txt')
O = open(out_path, 'w', encoding='utf-8')

log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_log.csv')
with open(log_path, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

O.write(f"Total: {len(rows)} rows\n\n")
O.write("=== PyBullet Free-Fall (from sim_log.csv) ===\n")
O.write("  t     | phi    alpha  theta  | phid   alpd   thed   | tau\n")
O.write("-" * 72 + "\n")

for r in rows:
    t = float(r['t'])
    step = int(round(t * 240))
    if (step < 10 or step % 12 == 0) and t <= 1.0:
        d = math.degrees
        phi = d(float(r['phi']))
        a = d(float(r['alpha']))
        th = d(float(r['theta']))
        pd = d(float(r['phi_dot']))
        ad = d(float(r['alpha_dot']))
        td = d(float(r['theta_dot']))
        tau = float(r['tau'])
        O.write(f" {t:5.3f} | {phi:+6.2f} {a:+6.2f} {th:+6.2f} | {pd:+7.1f} {ad:+7.1f} {td:+7.1f} | {tau:+.3f}\n")

# RK4 comparison
O.write("\n=== RK4 Free-Fall (same params) ===\n")
m1,m2 = 0.10, 0.32
L1,lc1,lc2 = 0.232, 0.182, 0.095
I1,I2 = 0.0003, 0.0009
g = 9.81
L_pole, L0, pole_height = 0.80, 1.00, 1.00
bd = 0.060
rope_horiz = L_pole/2 - bd/2
rope_len = L0/2
rope_vert = math.sqrt(rope_len**2 - rope_horiz**2)
R = rope_vert
Mt=m1+m2; p1=m1*lc1+m2*L1; p2=m2*lc2; p3=m2*L1*lc2
M11=Mt*R**2; M22=m1*lc1**2+I1+m2*L1**2; M33=m2*lc2**2+I2

def deriv(s):
    phi,al,th,pd,ad,td=s
    sPA=math.sin(phi+al);cPA=math.cos(phi+al)
    sPT=math.sin(phi+th);cPT=math.cos(phi+th)
    sAT=math.sin(al-th);cAT=math.cos(al-th)
    M=[[M11,R*p1*cPA,R*p2*cPT],[R*p1*cPA,M22,p3*cAT],[R*p2*cPT,p3*cAT,M33]]
    f1=R*p1*sPA*ad**2+R*p2*sPT*td**2-Mt*g*R*math.sin(phi)
    f2=R*p1*sPA*pd**2-p3*sAT*td**2+p1*g*math.sin(al)
    f3=R*p2*sPT*pd**2+p3*sAT*ad**2+p2*g*math.sin(th)
    a,b,c=M[0][0],M[0][1],M[0][2]
    dd,e,f=M[1][1],M[1][2],M[2][2]
    det=a*(dd*f-e*e)-b*(b*f-e*c)+c*(b*e-dd*c)
    iv=1.0/det
    A11=dd*f-e*e;A12=-(b*f-c*e);A13=b*e-c*dd
    A22=a*f-c*c;A23=-(a*e-b*c);A33=a*dd-b*b
    return [pd,ad,td,(A11*f1+A12*f2+A13*f3)*iv,(A12*f1+A22*f2+A23*f3)*iv,(A13*f1+A23*f2+A33*f3)*iv]

def rk4(s,dt):
    k1=deriv(s)
    s2=[s[i]+.5*dt*k1[i] for i in range(6)]
    k2=deriv(s2)
    s3=[s[i]+.5*dt*k2[i] for i in range(6)]
    k3=deriv(s3)
    s4=[s[i]+dt*k3[i] for i in range(6)]
    k4=deriv(s4)
    return [s[i]+(dt/6)*(k1[i]+2*k2[i]+2*k3[i]+k4[i]) for i in range(6)]

O.write("  t     | phi    alpha  theta  | phid   alpd   thed\n")
O.write("-" * 65 + "\n")
s=[0,0,math.radians(10),0,0,0]
dt=1.0/240.0
for step in range(240):
    d=math.degrees
    if step<10 or step%12==0:
        t=step*dt
        O.write(f" {t:5.3f} | {d(s[0]):+6.2f} {d(s[1]):+6.2f} {d(s[2]):+6.2f} | {d(s[3]):+7.1f} {d(s[4]):+7.1f} {d(s[5]):+7.1f}\n")
    s=rk4(s,dt)

O.close()
