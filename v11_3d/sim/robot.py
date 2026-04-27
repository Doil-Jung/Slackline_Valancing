"""
Slackline Balance Simulator V11 — Robot Model

로봇 구조:
  발판 (world.py의 platform link 위에 배치)
    └─ lower_body (하체) — revolute joint (발목, 자유)
        └─ upper_body (상체) — revolute joint (힙, 토크 τ 인가)

일반화 좌표 추출:
  φ  = 발판의 Rolling 각도 (x방향 변위에서 추정)
  α  = 하체의 수직 기준 기울기
  θ  = 상체의 수직 기준 기울기

추가 3D 상태:
  pitch = 발판의 Pitching (x축 회전)
  yaw   = 발판의 Yawing (z축 회전)
"""
import pybullet as p
import numpy as np
from config import V11Config, RobotParams


class SlacklineRobot:
    """줄 위의 2-링크 로봇 (하체 + 상체)"""

    def __init__(self, config: V11Config):
        self.config = config
        self.robot_params = config.robot
        self.robot_id = -1

        # Joint indices
        self.ankle_joint_idx = 0   # 발판-하체 연결
        self.hip_joint_idx = 1     # 하체-상체 연결 (토크 인가)

    def build(self, world):
        """
        로봇을 월드의 발판 위에 생성

        발판 (world.rope_body_id, link 1) 위에 로봇을 constraint로 연결
        """
        rp = self.robot_params
        rope_params = self.config.rope

        # 발판 위치 가져오기
        platform_state = world.get_platform_state()
        if platform_state is None:
            # 정적 초기 위치 사용
            foot_z = rope_params.foot_height
            platform_pos = [0, 0, foot_z]
        else:
            platform_pos = platform_state['position']

        # --- 로봇 멀티바디 생성 ---
        # base = foot (발판에 부착될 점)
        # link 0 = lower_body (하체)
        # link 1 = upper_body (상체)

        # 하체 치수
        lower_half = [rp.lower_width / 2, rp.body_depth / 2, rp.L1 / 2]
        upper_half = [rp.upper_width / 2, rp.body_depth / 2, rp.L2 / 2]

        # 하체 형상
        lower_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=lower_half)
        lower_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=lower_half,
                                        rgbaColor=[0.024, 0.839, 0.627, 0.7])  # #06d6a0

        # 상체 형상
        upper_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=upper_half)
        upper_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=upper_half,
                                        rgbaColor=[0.514, 0.220, 0.925, 0.7])  # #8338ec

        # 관성 계산 (균일 직육면체)
        lower_inertia = self._box_inertia(rp.m1, rp.lower_width, rp.body_depth, rp.L1)
        upper_inertia = self._box_inertia(rp.m2, rp.upper_width, rp.body_depth, rp.L2)

        # 힙 관절 시각적 표현 (원통)
        hip_vis = p.createVisualShape(p.GEOM_CYLINDER,
                                      radius=0.03,
                                      length=rp.body_depth * 1.5,
                                      rgbaColor=[1.0, 0.745, 0.043, 1.0])  # #ffbe0b

        # --- 로봇 MultiBody ---
        # base = 발 접촉점 (질량 매우 작음)
        # base에서 하체 CoM까지: z방향으로 L1/2
        # 하체 상단(힙)에서 상체 CoM까지: z방향으로 L2/2

        base_pos = [platform_pos[0], platform_pos[1],
                    platform_pos[2] + 0.02]  # 발판 위 약간 위

        # 초기 하체 기울기 α₀ 적용
        alpha0 = rp.alpha0
        theta0 = rp.theta0

        self.robot_id = p.createMultiBody(
            baseMass=0.01,  # 발 접촉점 (매우 가벼움)
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=base_pos,
            baseOrientation=[0, 0, 0, 1],

            linkMasses=[rp.m1, rp.m2],
            linkCollisionShapeIndices=[lower_col, upper_col],
            linkVisualShapeIndices=[lower_vis, upper_vis],

            # 링크 위치 (부모 프레임 기준)
            linkPositions=[
                [0, 0, rp.L1 / 2],          # 하체 CoM: base에서 L1/2 위
                [0, 0, rp.L1 / 2],           # 상체 CoM: 하체 CoM에서 L1/2 + L2/2 위
                                              # (힙은 하체 CoM에서 L1/2 위)
            ],
            linkOrientations=[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
            ],

            linkInertialFramePositions=[[0, 0, 0], [0, 0, 0]],
            linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1]],

            linkParentIndices=[0, 1],   # lower→base, upper→lower

            linkJointTypes=[p.JOINT_REVOLUTE, p.JOINT_REVOLUTE],
            linkJointAxis=[[0, 1, 0], [0, 1, 0]],  # y축 회전 (Rolling 평면)
        )

        # 초기 각도 설정
        if abs(alpha0) > 1e-6:
            p.resetJointState(self.robot_id, self.ankle_joint_idx, alpha0)
        if abs(theta0) > 1e-6:
            # θ는 절대 기울기이므로, 힙 조인트 각도 = θ - α (상대각)
            hip_angle = theta0 - alpha0
            p.resetJointState(self.robot_id, self.hip_joint_idx, hip_angle)

        # --- 발판과 로봇 연결 ---
        # 로봇 base를 월드의 발판(platform)에 고정
        self.platform_constraint = p.createConstraint(
            parentBodyUniqueId=world.rope_body_id,
            parentLinkIndex=1,  # platform
            childBodyUniqueId=self.robot_id,
            childLinkIndex=-1,  # robot base
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.02],  # 발판 상면
            childFramePosition=[0, 0, 0],
        )

        # --- 조인트 동역학 설정 ---
        # 발목: 자유 회전 (토크 없음, 감쇠만)
        p.setJointMotorControl2(
            self.robot_id, self.ankle_joint_idx,
            p.VELOCITY_CONTROL, force=0)  # 모터 비활성화
        p.changeDynamics(self.robot_id, 0,
                         jointDamping=self.robot_params.b_alpha)

        # 힙: 토크 모드 (LQR에서 제어)
        p.setJointMotorControl2(
            self.robot_id, self.hip_joint_idx,
            p.VELOCITY_CONTROL, force=0)  # 기본 모터 비활성화
        p.changeDynamics(self.robot_id, 1,
                         jointDamping=self.robot_params.b_theta)

        # 로봇-줄 충돌 비활성화
        for i in range(-1, 2):
            for j in range(-1, 3):
                p.setCollisionFilterPair(
                    self.robot_id, world.rope_body_id,
                    i, j, enableCollision=0)

        print(f"[Robot] Built: m1={rp.m1}kg, m2={rp.m2}kg, "
              f"L1={rp.L1}m, L2={rp.L2}m, "
              f"α₀={np.degrees(alpha0):.1f}°, θ₀={np.degrees(theta0):.1f}°")

        return self.robot_id

    def apply_hip_torque(self, tau: float):
        """힙 토크 τ 인가 (V10의 τ: α에 -τ, θ에 +τ)"""
        tau = np.clip(tau, -self.robot_params.tau_max, self.robot_params.tau_max)
        p.setJointMotorControl2(
            self.robot_id, self.hip_joint_idx,
            p.TORQUE_CONTROL, force=tau)

    def get_state(self, world=None):
        """
        로봇의 일반화 좌표 상태 반환

        Returns:
            dict with keys:
                # === Rolling (주제어 대상) ===
                phi, phi_dot        : 발 위치각 (원호 상에서)
                alpha, alpha_dot    : 하체 절대 기울기
                theta, theta_dot    : 상체 절대 기울기
                hip_angle           : 힙 상대각 (θ - α)

                # === 3D 추가 ===
                pitch, pitch_dot    : Pitching (x축 회전)
                yaw, yaw_dot        : Yawing (z축 회전)

                # === 위치 ===
                foot_pos            : 발 위치 [x, y, z]
                hip_pos             : 힙 위치 [x, y, z]
                head_pos            : 머리 위치 [x, y, z]
                com_pos             : 전체 CoM 위치 [x, y, z]
        """
        if self.robot_id < 0:
            return None

        # 발목 조인트 (α 관련)
        ankle_state = p.getJointState(self.robot_id, self.ankle_joint_idx)
        ankle_angle = ankle_state[0]     # α (하체 기울기)
        ankle_vel = ankle_state[1]       # α̇

        # 힙 조인트 (상대각 θ-α)
        hip_state = p.getJointState(self.robot_id, self.hip_joint_idx)
        hip_rel_angle = hip_state[0]     # θ - α (힙 상대각)
        hip_rel_vel = hip_state[1]       # θ̇ - α̇

        # 절대 각도 변환
        alpha = ankle_angle
        alpha_dot = ankle_vel
        theta = alpha + hip_rel_angle    # θ = α + (θ - α)
        theta_dot = alpha_dot + hip_rel_vel

        # 발판 상태 (φ, pitch, yaw 추출)
        if world is not None:
            platform_state = world.get_platform_state()
            if platform_state is not None:
                euler = platform_state['euler']
                ang_vel = platform_state['angular_velocity']
                pos = platform_state['position']

                # φ 추정: 발판의 x위치로부터 원호각 추정
                # V10의 x_foot = R * sin(φ) → φ ≈ arcsin(x / R)
                # 여기서는 줄의 기하학에서 R이 동적이므로, x 자체를 추적
                # 또는 발판의 y축 회전각을 직접 사용
                phi = euler[1]  # y축 회전 = Rolling 방향
                phi_dot = ang_vel[1]

                pitch = euler[0]   # x축 회전 = Pitching
                pitch_dot = ang_vel[0]

                yaw = euler[2]     # z축 회전 = Yawing
                yaw_dot = ang_vel[2]

                foot_pos = np.array(pos)
            else:
                phi = phi_dot = pitch = pitch_dot = yaw = yaw_dot = 0.0
                foot_pos = np.zeros(3)
        else:
            phi = phi_dot = pitch = pitch_dot = yaw = yaw_dot = 0.0
            base_state = p.getBasePositionAndOrientation(self.robot_id)
            foot_pos = np.array(base_state[0])

        # 링크 위치 계산
        lower_state = p.getLinkState(self.robot_id, 0)
        upper_state = p.getLinkState(self.robot_id, 1)
        hip_pos = np.array(lower_state[0]) + np.array([0, 0, self.robot_params.L1 / 2])
        head_pos = np.array(upper_state[0]) + np.array([0, 0, self.robot_params.L2 / 2])

        # 전체 CoM
        lower_com = np.array(lower_state[0])
        upper_com = np.array(upper_state[0])
        m1, m2 = self.robot_params.m1, self.robot_params.m2
        com_pos = (m1 * lower_com + m2 * upper_com) / (m1 + m2)

        return {
            # Rolling (주제어)
            'phi': phi, 'phi_dot': phi_dot,
            'alpha': alpha, 'alpha_dot': alpha_dot,
            'theta': theta, 'theta_dot': theta_dot,
            'hip_angle': hip_rel_angle,

            # 3D 추가
            'pitch': pitch, 'pitch_dot': pitch_dot,
            'yaw': yaw, 'yaw_dot': yaw_dot,

            # 위치
            'foot_pos': foot_pos,
            'hip_pos': hip_pos,
            'head_pos': head_pos,
            'com_pos': com_pos,

            # 에너지 (추후 계산)
            'time': 0.0,
        }

    def get_com_error(self, state=None):
        """CoM 수평 오차 (x방향) = x_com - x_foot"""
        if state is None:
            state = self.get_state()
        return state['com_pos'][0] - state['foot_pos'][0]

    def apply_perturbation(self, impulse, axis='theta'):
        """
        외부 교란 (각속도 충격)

        Args:
            impulse: 각속도 충격 크기 (rad/s)
            axis: 'theta' (Rolling, 기본), 'pitch', 'yaw'
        """
        if axis == 'theta':
            # 상체에 Rolling 방향 각속도 충격
            state = p.getJointState(self.robot_id, self.hip_joint_idx)
            current_vel = state[1]
            p.resetJointState(self.robot_id, self.hip_joint_idx,
                              state[0], current_vel + impulse)
        elif axis == 'pitch':
            # 발판에 Pitching 충격 (x축 토크)
            p.applyExternalTorque(self.robot_id, -1,
                                 [impulse * 10, 0, 0],
                                 p.LINK_FRAME)
        elif axis == 'yaw':
            # 발판에 Yawing 충격 (z축 토크)
            p.applyExternalTorque(self.robot_id, -1,
                                 [0, 0, impulse * 10],
                                 p.LINK_FRAME)

    def _box_inertia(self, mass, w, d, h):
        """균일 직육면체의 관성 모멘트 [Ixx, Iyy, Izz]"""
        return [
            mass * (d**2 + h**2) / 12.0,
            mass * (w**2 + h**2) / 12.0,
            mass * (w**2 + d**2) / 12.0,
        ]
