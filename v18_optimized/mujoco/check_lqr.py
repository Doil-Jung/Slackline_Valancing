# -*- coding: utf-8 -*-
"""Compare LQR gains: v14 big robot vs v18 small robot (same code)"""
import numpy as np
from scipy.linalg import solve_discrete_are, expm

def compute_lqr(M1, M2, L1, L2, M_HEAD, HEAD_R, R, B_THETA, label):
    GRAV=9.81; DT=0.001
    LC1=L1/2; I1=M1*L1**2/12
    m_tot=M2+M_HEAD; d2=L2/2; d_h=L2+HEAD_R
    lc2=(M2*d2+M_HEAD*d_h)/m_tot
    i2_body=M2*L2**2/12+M2*(d2-lc2)**2
    i2_head=2/5*M_HEAD*HEAD_R**2+M_HEAD*(d_h-lc2)**2
    M2E=m_tot; LC2E=lc2; I2E=i2_body+i2_head

    p1=M1*LC1+M2E*L1; p2=M2E*LC2E; p3=M2E*L1*LC2E; Mt=M1+M2E
    M_eq=np.array([[Mt*R**2,R*p1,R*p2],[R*p1,M1*LC1**2+I1+M2E*L1**2,p3],[R*p2,p3,M2E*LC2E**2+I2E]])
    G_mat=np.array([[-Mt*GRAV*R,0,0],[0,p1*GRAV,0],[0,0,p2*GRAV]])
    D_mat=np.diag([0,0,-B_THETA])
    E_mat=np.array([[0],[-1],[1]])
    Mi=np.linalg.inv(M_eq); n=6
    Ac=np.zeros((n,n)); Bc=np.zeros((n,1))
    Ac[:3,3:]=np.eye(3)
    Ac[3:,:3]=Mi@G_mat; Ac[3:,3:]=Mi@D_mat; Bc[3:,:]=Mi@E_mat
    Z=np.zeros((n+1,n+1)); Z[:n,:n]=Ac*DT; Z[:n,n:]=Bc*DT
    eZ=expm(Z); Ad=eZ[:n,:n]; Bd=eZ[:n,n:n+1]
    Q=np.diag([1,50,50,0.1,5,5]); Rc=np.array([[0.001]])
    P=solve_discrete_are(Ad,Bd,Q,Rc)
    K=np.linalg.inv(Rc+Bd.T@P@Bd)@(Bd.T@P@Ad)
    
    # Before sign flip
    eigs_raw = sorted(abs(e) for e in np.linalg.eigvals(Ad - Bd@K))
    
    # Apply sign flip (v14 original)
    K[0,0] *= -1; K[0,3] *= -1
    eigs_flip = sorted(abs(e) for e in np.linalg.eigvals(Ad - Bd@K))
    
    # Controllability
    C_mat = np.hstack([np.linalg.matrix_power(Ad, i) @ Bd for i in range(n)])
    rank = np.linalg.matrix_rank(C_mat, tol=1e-8)
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  M1={M1} M2={M2} L1={L1} L2={L2} R={R} HEAD={M_HEAD}")
    print(f"  B_THETA={B_THETA}")
    print(f"{'='*60}")
    print(f"  Controllability rank = {rank}/6")
    print(f"  K = [{', '.join(f'{k:.1f}' for k in K[0])}]")
    print(f"  Before flip: max|eig| = {max(eigs_raw):.6f} {'STABLE' if max(eigs_raw)<1 else 'UNSTABLE'}")
    print(f"  After  flip: max|eig| = {max(eigs_flip):.6f} {'STABLE' if max(eigs_flip)<1 else 'UNSTABLE'}")
    
    x0 = np.array([0,0,0.15,0,0,0])
    tau0 = -float(K @ x0)
    print(f"  theta0=8.6deg: tau = {tau0:.2f} N*m")

# V14 original (big robot)
compute_lqr(M1=30.0, M2=40.0, L1=0.85, L2=0.80, M_HEAD=5.0, HEAD_R=0.20, 
            R=1.5, B_THETA=3.0, label="V14 BIG ROBOT (original)")

# V18 small robot
compute_lqr(M1=0.80, M2=1.70, L1=0.24, L2=0.56, M_HEAD=0.50, HEAD_R=0.14,
            R=0.50, B_THETA=0.01, label="V18 SMALL ROBOT")

# V18 small robot with higher damping (like v14)
compute_lqr(M1=0.80, M2=1.70, L1=0.24, L2=0.56, M_HEAD=0.50, HEAD_R=0.14,
            R=0.50, B_THETA=0.1, label="V18 SMALL + B_THETA=0.1")
