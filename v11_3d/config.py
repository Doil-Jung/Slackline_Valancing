"""
Slackline Balance Simulator V11 — Configuration & Parameters

좌표계: x = 줄에 수직 (흔들림 방향), y = 줄 방향 (기둥), z = 수직 (위)
용어:
  - Rolling  = x방향 흔들림 (주제어 대상, φ/α/θ)
  - Pitching = x축 회전 (줄 방향 기울임)
  - Yawing   = z축 회전 (수평 비틀림)

V10 파라미터를 Python으로 이식 + 3D 확장
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RobotParams:
    """로봇 물리 파라미터 (V10 params.js 이식)"""
    # === 하체 (다리) ===
    m1: float = 30.0        # 하체 질량 (kg)
    L1: float = 0.85        # 하체 길이 — 발에서 힙 (m)

    # === 상체 (몸통) ===
    m2: float = 40.0        # 상체 질량 (kg)
    L2: float = 0.80        # 상체 길이 — 힙에서 머리 (m)

    # === 렌더링/3D 치수 ===
    lower_width: float = 0.18   # 하체 폭 (x 방향, m)
    upper_width: float = 0.30   # 상체 폭 (x 방향, m)
    body_depth: float = 0.25    # 몸체 깊이 = 두 발 사이 간격 (y 방향, m)

    # === 감쇠 ===
    b_phi: float = 2.0      # 줄 감쇠 (N·m·s/rad) — rolling 방향
    b_alpha: float = 1.0    # 하체 회전 감쇠 (N·m·s/rad)
    b_theta: float = 0.5    # 상체 회전 감쇠 (N·m·s/rad)

    # === 중력 ===
    g: float = 9.81

    # === 서보 스펙 (STS3215) ===
    tau_max: float = 1500.0   # 최대 힙 토크 (N·m) — 사람 모델용 기본값
    # STS3215 로봇용: tau_max = 1.47 N·m

    # === 초기 조건 ===
    phi0: float = 0.0
    alpha0: float = 0.0
    theta0: float = 0.15
    phi_dot0: float = 0.0
    alpha_dot0: float = 0.0
    theta_dot0: float = 0.0

    # === 파생 상수 ===
    @property
    def ell1(self) -> float:
        """하체 CoM까지 거리"""
        return self.L1 / 2.0

    @property
    def ell2(self) -> float:
        """상체 CoM까지 거리"""
        return self.L2 / 2.0

    @property
    def I1(self) -> float:
        """하체 관성모멘트 (균일 막대)"""
        return self.m1 * self.L1**2 / 12.0

    @property
    def I2(self) -> float:
        """상체 관성모멘트 (균일 막대)"""
        return self.m2 * self.L2**2 / 12.0

    @property
    def p1(self) -> float:
        return self.m1 * self.ell1 + self.m2 * self.L1

    @property
    def p2(self) -> float:
        return self.m2 * self.ell2

    @property
    def p3(self) -> float:
        return self.m2 * self.L1 * self.ell2

    @property
    def M_total(self) -> float:
        return self.m1 + self.m2


@dataclass
class RopeParams:
    """줄 (슬랙라인) 파라미터"""
    L_pole: float = 4.0     # 기둥 간 거리 (y 방향, m)
    L0: float = 5.0         # 줄 전체 길이 (m) — L0 > L_pole이면 줄이 처짐
    pole_height: float = 4.5  # 기둥 높이 (바닥에서, m)
    rope_mass: float = 0.1  # 줄 질량 (kg) — 거의 무시
    _body_depth: float = 0.25  # 두 발 사이 간격 (RobotParams.body_depth와 동기화)

    @property
    def sag(self) -> float:
        """줄 처짐량: 줄은 기둥상단 → 하체 ±y 모서리이므로 수평거리 = L_pole/2 - bd/2"""
        half_rope = self.L0 / 2.0
        horiz = self.L_pole / 2.0 - self._body_depth / 2.0
        if half_rope <= horiz:
            return 0.0
        return np.sqrt(half_rope**2 - horiz**2)

    @property
    def foot_height(self) -> float:
        """발 위치의 높이 (바닥 기준) = 기둥 상단 - 처짐"""
        return self.pole_height - self.sag


@dataclass
class SimParams:
    """시뮬레이션 파라미터"""
    dt: float = 1.0 / 240.0     # PyBullet 기본 타임스텝 (240Hz)
    control_dt: float = 0.001    # 제어 루프 주기 (1kHz, V10과 동일)
    render_fps: float = 30.0     # 렌더링 FPS
    gravity: float = 9.81
    mode: str = 'A'              # 'A' = 힌지 줄, 'B' = 사다리꼴 강체
    use_gui: bool = True         # PyBullet GUI 사용 여부


@dataclass
class NoiseParams:
    """센서 노이즈 파라미터 (V10 noise.js 이식)"""
    angle_noise_sigma: float = 0.0    # 각도 노이즈 σ (rad)
    gyro_noise_sigma: float = 0.0     # 각속도 노이즈 σ (rad/s)
    gyro_bias_drift: float = 0.0      # 자이로 바이어스 드리프트 (rad/s²)


@dataclass
class MismatchParams:
    """모델 불확실성 (V10 params_v9.js 이식) — 비율로 표현"""
    m1: float = 0.0   # +0.1 = +10%
    m2: float = 0.0
    L1: float = 0.0
    L2: float = 0.0


@dataclass
class ControlParams:
    """LQR 제어 파라미터"""
    controller_on: bool = True
    gain_scale: float = 1.0
    r_scale: float = 1.0         # R 스케일 (LQR 튜닝)
    q_scale: float = 1.0         # 칼만 Q 스케일
    estimation_mode: str = 'ideal'  # 'ideal', 'kinematic', 'observer'
    total_delay_ms: float = 0.0
    augmented: bool = False       # 상태증강 LQR


@dataclass
class V11Config:
    """전체 설정"""
    robot: RobotParams = field(default_factory=RobotParams)
    rope: RopeParams = field(default_factory=RopeParams)
    sim: SimParams = field(default_factory=SimParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    mismatch: MismatchParams = field(default_factory=MismatchParams)
    control: ControlParams = field(default_factory=ControlParams)

    def get_plant_params(self) -> RobotParams:
        """플랜트 파라미터 (편차 적용)"""
        import copy
        plant = copy.deepcopy(self.robot)
        plant.m1 *= (1.0 + self.mismatch.m1)
        plant.m2 *= (1.0 + self.mismatch.m2)
        plant.L1 *= (1.0 + self.mismatch.L1)
        plant.L2 *= (1.0 + self.mismatch.L2)
        return plant

    def get_nominal_params(self) -> RobotParams:
        """공칭 파라미터 (편차 없음)"""
        import copy
        return copy.deepcopy(self.robot)
