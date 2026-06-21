# -*- coding: utf-8 -*-
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, mujoco
GRAV=9.81; DT=0.001

def make_mjcf(L1, L2, M1, m2, LC2, R, HEAD_R, M_head, BD):
    LP = 1.0; HP = LP/2
    POLE_H = R + L1 + L2 + HEAD_R*2 + 0.3
    rh = HP - BD/2; rv = R
    M2b = m2 - M_head - 0.10
    return f"""<mujoco model='small_test'>
  <option gravity='0 0 -{GRAV}' timestep='{DT}'/>
  <default><joint damping='0' armature='0'/></default>
  <worldbody>
    <light pos='0 -2 3' dir='0 1 -0.5'/>
    <geom type='cylinder' pos='-{HP} 0 {POLE_H/2}' size='0.01 {POLE_H/2}' rgba='.4 .4 .4 1'/>
    <geom type='cylinder' pos=' {HP} 0 {POLE_H/2}' size='0.01 {POLE_H/2}' rgba='.4 .4 .4 1'/>
    <body name='rope_a' pos='-{HP} 0 {POLE_H}'>
      <joint name='rope_a_Y' type='hinge' axis='0 1 0'/>
      <geom type='capsule' fromto='0 0 0 {rh} 0 -{rv}' size='0.003' mass='0.005'/>
    </body>
    <body name='rope_b' pos='{HP} 0 {POLE_H}'>
      <joint name='rope_b_Y' type='hinge' axis='0 1 0'/>
      <geom type='capsule' fromto='0 0 0 -{rh} 0 -{rv}' size='0.003' mass='0.005'/>
    </body>
    <body name='foot' pos='0 0 {POLE_H - rv}'>
      <joint name='phi_Y' type='hinge' axis='0 1 0'/>
      <geom type='sphere' size='0.005' mass='0.005' rgba='.1 .1 .1 1'/>
      <body name='lower'>
        <joint name='ankle' type='hinge' axis='0 1 0'/>
        <geom type='box' pos='0 0 {L1/2}' size='0.02 0.02 {L1/2}' mass='{M1}' rgba='.3 .5 .8 1'/>
        <body name='upper' pos='0 0 {L1}'>
          <joint name='hip' type='hinge' axis='0 1 0' limited='true' range='-120 120'/>
          <geom type='box' pos='0 0 {LC2}' size='0.02 0.02 {L2/2}' mass='{max(0.001, M2b+0.10)}' rgba='.8 .3 .3 1'/>
          <body name='head' pos='0 0 {L2+HEAD_R}'>
            <geom type='sphere' size='{HEAD_R}' mass='{M_head}' rgba='1 .85 .7 1'/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect body1='rope_a' body2='foot' anchor='{rh} 0 -{rv}' solref='-10000 -200'/>
    <connect body1='rope_b' body2='foot' anchor='-{rh} 0 -{rv}' solref='-10000 -200'/>
  </equality>
  <actuator>
    <motor name='hip_motor' joint='hip' ctrllimited='true' ctrlrange='-500 500' gear='1'/>
  </actuator>
</mujoco>"""

RHO=0.30; M_SRV=0.067; M_ELC=0.10

def get_robot(L1,L2):
    M1=RHO*L1+M_SRV; M2b=RHO*L2+M_SRV; Mh=0.03*(L2/0.25); HR=L2/4
    m2=M2b+Mh+M_ELC; Mt=M1+m2; LC1=L1/2; I1=M1*L1**2/12
    db=L2/2; dh=L2+HR; de=L2*0.6
    LC2=(M2b*db+Mh*dh+M_ELC*de)/m2
    I2=M2b*L2**2/12+M2b*(db-LC2)**2+2/5*Mh*HR**2+Mh*(dh-LC2)**2+M_ELC*(de-LC2)**2
    p1=M1*LC1+m2*L1; p2=m2*LC2; ps=p1+p2; h=ps/Mt
    Jaa=M1*LC1**2+I1+m2*L1**2; Jtt=m2*LC2**2+I2; Jat=m2*L1*LC2
    BD=min(L1*0.4,0.20)
    return dict(M1=M1,m2=m2,Mt=Mt,LC1=LC1,LC2=LC2,h=h,p1=p1,p2=p2,ps=ps,
                Jaa=Jaa,Jtt=Jtt,Jat=Jat,HR=HR,Mh=Mh,BD=BD,L1=L1,L2=L2)

def get_mode(r, R):
    Mt=r['Mt']; h=r['h']; p1=r['p1']; p2=r['p2']; ps=r['ps']
    Tinv=np.array([[1,0,0],[0,Mt*h/ps,-p2/ps],[0,Mt*h/ps,p1/ps]])
    M_mat=np.array([[Mt*R**2,R*p1,R*p2],[R*p1,r['Jaa'],r['Jat']],[R*p2,r['Jat'],r['Jtt']]])
    G_mat=np.diag([-Mt*GRAV*R,p1*GRAV,p2*GRAV])
    Mp=Tinv.T@M_mat@Tinv; Gp=Tinv.T@G_mat@Tinv
    A2=np.linalg.solve(Mp[:2,:2],Gp[:2,:2])
    ev,V=np.linalg.eig(A2); idx=np.argsort(ev.real); ev=ev[idx].real; V=V[:,idx].real
    Vi=np.linalg.inv(V)
    return np.sqrt(abs(ev[0])),np.sqrt(abs(ev[1])),V,Vi

cases = [
    (0.40, 0.40, 0.40),
    (0.40, 0.40, 0.55),
    (0.50, 0.50, 0.50),
    (0.50, 0.50, 0.60),
]

print("=" * 80)
print("  MuJoCo V-rope 모델: 80cm/100cm 로봇 테스트")
print("=" * 80)

for L1,L2,R in cases:
    r = get_robot(L1,L2)
    om,lam,V,Vi = get_mode(r,R)
    v1=V[:,0]; ratio=v1[0]/v1[1]; es=-r['h']/(ratio*R+r['h'])
    bi=np.radians(2)
    phi_af=r['h']*(1+es)*bi/R; beta_af=-es*bi
    
    xml = make_mjcf(L1,L2,r['M1'],r['m2'],r['LC2'],R,r['HR'],r['Mh'],r['BD'])
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception as e:
        print(f"  L={L1+L2:.1f}m R={R}: MODEL FAIL: {e}")
        continue
    
    phi_mj = -phi_af
    ra=model.jnt_qposadr[model.joint('phi_Y').id]
    aa=model.jnt_qposadr[model.joint('ankle').id]
    ha=model.jnt_qposadr[model.joint('hip').id]
    rb=model.jnt_qposadr[model.joint('rope_b_Y').id]
    
    Mt=r['Mt']; h_r=r['h']; p1=r['p1']; p2=r['p2']
    
    print(f"\n  L1+L2={L1+L2:.1f}m R={R:.2f}m | Mt={Mt*1000:.0f}g lam={lam:.2f} eps*={es:.3f}")
    
    for Kp,Kd,tmax in [(5000,100,500),(10000,200,500),(20000,500,500),(50000,1000,500)]:
        data2 = mujoco.MjData(model)
        data2.qpos[ra]=phi_mj; data2.qpos[aa]=beta_af-phi_mj; data2.qpos[ha]=0
        data2.qpos[rb]=phi_mj
        mujoco.mj_forward(model,data2)
        survived=True; max_xcom=0; t_die=2.0; max_delta=0
        for i in range(2000):
            phi=data2.qpos[ra]; ank=data2.qpos[aa]; hip=data2.qpos[ha]
            alpha=ank+phi; theta=hip+alpha; delta=theta-alpha
            if np.any(np.isnan([phi,ank,hip])) or abs(alpha)>1.4:
                survived=False; t_die=i*DT; break
            deltad = data2.qvel[model.jnt_dofadr[model.joint('hip').id]]
            data2.ctrl[0]=np.clip(Kp*(0-delta)-Kd*deltad, -tmax, tmax)
            max_delta = max(max_delta, abs(np.degrees(delta)))
            beta_v=(p1*alpha+p2*theta)/(Mt*h_r)
            xcom=abs(-R*phi+h_r*beta_v)*1000
            max_xcom=max(max_xcom,xcom)
            mujoco.mj_step(model,data2)
        st='ALIVE' if survived else f'DEAD@{t_die:.3f}'
        print(f"    Kp={Kp:6d} Kd={Kd:4d} -> {st:12s} max_d={max_delta:.2f}d xcom={max_xcom:.1f}mm")

print("\n  완료!")
