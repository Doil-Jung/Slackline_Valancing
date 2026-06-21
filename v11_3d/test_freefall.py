# -*- coding: utf-8 -*-
"""PyBullet vs RK4 free-fall - no numpy"""
import os, math, traceback

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'freefall_out.txt')
O = open(out_path, 'w', encoding='utf-8')
def P(s=''):
    O.write(str(s) + '\n'); O.flush()

try:
    import pybullet as pb
    P("pybullet imported OK")
    
    m1, m2 = 0.10, 0.32
    L1 = 0.232
    lc1, lc2 = 0.182, 0.095
    I1, I2 = 0.0003, 0.0009
    g = 9.81
    L_pole, L0, pole_height = 0.80, 1.00, 1.00
    bd = 0.060
    half_pole = L_pole / 2
    rope_horiz = half_pole - bd / 2
    rope_len = L0 / 2
    rope_vert = math.sqrt(rope_len**2 - rope_horiz**2)
    foot_z = pole_height - rope_vert
    theta0 = math.radians(10.0)
    
    ra_end = [0, -rope_horiz, -rope_vert]
    ra_mid = [0, -rope_horiz/2, -rope_vert/2]
    com1 = [0, 0, lc1]
    com2 = [0, 0, lc2]
    
    P(f"rope_vert={rope_vert:.4f} theta0=10deg")
    
    cid = pb.connect(pb.DIRECT)
    pb.setGravity(0, 0, -g)
    pb.setTimeStep(1.0/240.0)
    
    pole_A_top = [0, half_pole, pole_height]
    
    robot_id = pb.createMultiBody(
        baseMass=0,
        basePosition=pole_A_top,
        baseOrientation=[0,0,0,1],
        linkMasses=[0.001, m1, m2],
        linkCollisionShapeIndices=[-1, -1, -1],
        linkVisualShapeIndices=[-1, -1, -1],
        linkPositions=[ra_end, [0,0,0], [0,0,L1]],
        linkOrientations=[[0,0,0,1]]*3,
        linkInertialFramePositions=[ra_mid, com1, com2],
        linkInertialFrameOrientations=[[0,0,0,1]]*3,
        linkParentIndices=[0, 1, 2],
        linkJointTypes=[pb.JOINT_REVOLUTE]*3,
        linkJointAxis=[[0,1,0]]*3,
    )
    
    ROPE_A, ANKLE, HIP = 0, 1, 2
    
    pb.changeDynamics(robot_id, ANKLE, mass=m1,
                      localInertiaDiagonal=[I1, I1, I1*0.1],
                      jointDamping=0.0)
    pb.changeDynamics(robot_id, HIP, mass=m2,
                      localInertiaDiagonal=[I2, I2, I2*0.1],
                      jointDamping=0.0)
    pb.changeDynamics(robot_id, ROPE_A, jointDamping=0.0)
    
    for j in range(3):
        pb.setJointMotorControl2(robot_id, j, pb.VELOCITY_CONTROL, force=0)
    
    pb.resetJointState(robot_id, HIP, theta0, 0)
    
    # Print dynamics info
    for j in range(3):
        dyn = pb.getDynamicsInfo(robot_id, j)
        P(f"  Link {j}: mass={dyn[0]:.4f} inertia={dyn[2]} localCoM={dyn[3]}")
    P()
    
    P("=== PyBullet Free-Fall (theta0=10, tau=0) ===")
    P("  t     | phi    alpha  theta  | phid   alpd   thed")
    P("-" * 65)
    
    for step in range(240):
        phi = pb.getJointState(robot_id, ROPE_A)[0]
        phi_dot = pb.getJointState(robot_id, ROPE_A)[1]
        
        lower_info = pb.getLinkState(robot_id, ANKLE,
                                     computeForwardKinematics=True,
                                     computeLinkVelocity=True)
        upper_info = pb.getLinkState(robot_id, HIP,
                                     computeForwardKinematics=True,
                                     computeLinkVelocity=True)
        
        lower_euler = pb.getEulerFromQuaternion(lower_info[5])
        upper_euler = pb.getEulerFromQuaternion(upper_info[5])
        alpha_abs = lower_euler[1]
        theta_abs = upper_euler[1]
        alpha_dot = lower_info[7][1]
        theta_dot = upper_info[7][1]
        
        d = math.degrees
        if step < 10 or step % 12 == 0:
            t = step / 240.0
            P(f" {t:5.3f} | {d(phi):+6.2f} {d(alpha_abs):+6.2f} {d(theta_abs):+6.2f} | "
              f"{d(phi_dot):+7.1f} {d(alpha_dot):+7.1f} {d(theta_dot):+7.1f}")
        
        pb.stepSimulation()
    
    pb.disconnect()
    P()
    
    # === RK4 comparison ===
    R = rope_vert
    Mt = m1 + m2
    p1v = m1*lc1 + m2*L1
    p2v = m2*lc2
    p3v = m2*L1*lc2
    M11 = Mt*R**2
    M22 = m1*lc1**2 + I1 + m2*L1**2
    M33 = m2*lc2**2 + I2
    
    def deriv(s):
        phi,al,th,pd,ad,td = s
        sPA=math.sin(phi+al);cPA=math.cos(phi+al)
        sPT=math.sin(phi+th);cPT=math.cos(phi+th)
        sAT=math.sin(al-th);cAT=math.cos(al-th)
        M=[[M11,R*p1v*cPA,R*p2v*cPT],[R*p1v*cPA,M22,p3v*cAT],[R*p2v*cPT,p3v*cAT,M33]]
        f1=R*p1v*sPA*ad**2+R*p2v*sPT*td**2-Mt*g*R*math.sin(phi)
        f2=R*p1v*sPA*pd**2-p3v*sAT*td**2+p1v*g*math.sin(al)
        f3=R*p2v*sPT*pd**2+p3v*sAT*ad**2+p2v*g*math.sin(th)
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
    
    P("=== RK4 Free-Fall (same params, same dt) ===")
    P("  t     | phi    alpha  theta  | phid   alpd   thed")
    P("-" * 65)
    s = [0, 0, float(theta0), 0, 0, 0]
    dt = 1.0/240.0
    for step in range(240):
        d = math.degrees
        if step < 10 or step % 12 == 0:
            t = step*dt
            P(f" {t:5.3f} | {d(s[0]):+6.2f} {d(s[1]):+6.2f} {d(s[2]):+6.2f} | "
              f"{d(s[3]):+7.1f} {d(s[4]):+7.1f} {d(s[5]):+7.1f}")
        s = rk4(s, dt)

except Exception as e:
    P(f"\nERROR: {e}")
    P(traceback.format_exc())
finally:
    O.close()
