"""
FWE 4-Phase Controller
======================
Fold-Wait₁-Extend-Wait₂ state machine for slackline balancing.
Outputs desired hip angle (delta_target) at each timestep.
"""
import numpy as np
from enum import Enum, auto


class Phase(Enum):
    IDLE    = auto()
    FOLD    = auto()
    WAIT1   = auto()
    EXTEND  = auto()
    WAIT2   = auto()


class FWEController:
    """
    4-Phase FWE controller.

    Parameters
    ----------
    eps : float       접기 강도 스케일
    T_f : float       FOLD/EXTEND 소요 시간 [s]
    T_w1 : float      WAIT₁ 대기 시간 [s]
    h_com : float     CoM 높이 [m]
    c_foot : float    발 이동 비율 (p2/ps)
    beta_threshold : float   FOLD 트리거 임계값 [rad]
    p1, p2 : float    일반화 질량 계수
    Mt : float        전체 질량 [kg]
    g, R, m1, m2, L1, L2 : float 로봇 물리 파라미터 (제4식 계산용)
    """

    def __init__(self, eps, T_f, T_w1, h_com, c_foot, p1, p2, Mt,
                 beta_threshold=np.radians(1.0),
                 max_delta=np.radians(30),
                 T_pred=0.1,
                 g=9.81, R=0.3, m1=0.1, m2=0.25, L1=0.2, L2=0.2,
                 c_pred=1.0,
                 c_beta_dot=0.3):
        self.eps = eps
        self.T_f = T_f
        self.T_w1 = T_w1
        self.h_com = h_com
        self.c_foot = c_foot
        self.p1 = p1
        self.p2 = p2
        self.Mt = Mt
        self.beta_threshold = beta_threshold
        self.max_delta = max_delta
        self.T_pred = T_pred
        self.c_pred = c_pred
        self.c_beta_dot = c_beta_dot
        self.T_w2_min = 0.100      # 100ms minimum wait to avoid double-triggering
        self.skipWait2Entirely = True

        # 물리 파라미터 저장 (제4식 계산용)
        self.g = g
        self.R = R
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2

        # State
        self.phase = Phase.IDLE
        self.t_phase = 0.0          # time since phase start
        self.d_fold = 0.0           # current fold target
        self.fold_sign = 1.0        # +1 or -1
        self.delta_target = 0.0     # current output
        self.beta_prev = None       # previous beta for numerical diff

        # W2 tracking
        self.phi_prev = None
        self.phi_crossed = False

        # Cycle counter
        self.cycle_count = 0

        # Log
        self.phase_log = []

    def compute_beta(self, phi, alpha, theta):
        """CoM absolute tilt angle (from vertical)."""
        x_com = self.R * np.sin(phi) + (self.p1 * np.sin(alpha) + self.p2 * np.sin(theta)) / self.Mt
        return x_com / self.h_com

    def _fold_profile(self, t_local):
        """
        Triangle acceleration profile: position at t_local.
        Goes from 0 to d_fold over T_f.
        """
        Tf = self.T_f
        d = abs(self.d_fold)
        if Tf <= 0:
            return d
        a = 4 * d / Tf**2
        half = Tf / 2

        if t_local <= 0:
            return 0.0
        elif t_local < half:
            return 0.5 * a * t_local**2
        elif t_local < Tf:
            dt = t_local - half
            return 0.5 * a * half**2 + a * half * dt - 0.5 * a * dt**2
        else:
            return d

    def _initiate_fold(self, beta, beta_dot, from_phase=""):
        # 가상 역진자 쓰러짐율 lambda 및 관성모멘트 J_tot 연산
        ell1 = self.L1 / 2
        ell2 = self.L2 / 2
        
        I1 = self.m1 * self.L1 * self.L1 / 12
        I2 = self.m2 * self.L2 * self.L2 / 12
        J_aa = self.m1 * ell1 * ell1 + I1 + self.m2 * self.L1 * self.L1
        J_tt = self.m2 * ell2 * ell2 + I2
        J_at = self.m2 * self.L1 * ell2
        J_tot = J_aa + J_tt + 2 * J_at
        
        lambda_val = np.sqrt((self.Mt * self.g * self.h_com) / J_tot)
        
        Tf = self.T_f
        Tw1 = self.T_w1
        T_total = 2 * Tf + Tw1
        T_fw = Tf + Tw1
        
        # k_1 = 1 기준의 f_bar_beta tilde
        f_bar_beta = (4.0 * self.c_foot) / self.h_com
        
        # B_pos, B_vel (k1 = 1 기준 tilde 항)
        B_pos = -(f_bar_beta / (lambda_val * lambda_val * Tf * Tf)) * (np.cosh(lambda_val * Tf) - 2 * np.cosh(lambda_val * Tf / 2) + 1)
        B_vel = -(f_bar_beta / (lambda_val * Tf * Tf)) * (np.sinh(lambda_val * Tf) - 2 * np.sinh(lambda_val * Tf / 2))
        
        # d_bar_beta_tilde
        term1 = (B_pos * np.cosh(lambda_val * Tw1) + (B_vel / lambda_val) * np.sinh(lambda_val * Tw1)) * np.cosh(lambda_val * Tf)
        term2 = (B_pos * lambda_val * np.sinh(lambda_val * Tw1) + B_vel * np.cosh(lambda_val * Tw1)) * (np.sinh(lambda_val * Tf) / lambda_val)
        d_bar_beta_tilde = term1 + term2 - B_pos
        
        # 분모 보호
        denom = d_bar_beta_tilde
        if abs(denom) < 1e-6:
            denom = np.sign(denom) * 1e-6
            if denom == 0:
                denom = -1e-6

        # 진짜 제4식 접기 공식 적용 (d_fold = - (1 + eps) * num / denom)
        beta_dot_scaled = beta_dot * getattr(self, 'c_beta_dot', 1.0)
        num = beta * np.cosh(lambda_val * T_total) + (beta_dot_scaled / lambda_val) * np.sinh(lambda_val * T_total) + beta * np.cosh(lambda_val * Tf)
        d_fold_val = - (1.0 + self.eps) * num / denom
        
        # 방향 및 크기 할당
        self.fold_sign = np.sign(d_fold_val)
        if self.fold_sign == 0:
            self.fold_sign = 1.0
        self.d_fold = abs(d_fold_val)

        if self.d_fold > self.max_delta:
            self.d_fold = self.max_delta

        self.phase = Phase.FOLD
        self.t_phase = 0.0
        self.cycle_count += 1
        self.delta_target = self.fold_sign * self._fold_profile(0.0)
        
        loc = f" (from {from_phase})" if from_phase else ""
        self._log(f"FOLD start{loc}: beta={np.degrees(beta):.1f}d d_fold={np.degrees(d_fold_val):.1f}d")

    def step(self, dt, phi, alpha, theta):
        """
        Advance one timestep.

        Parameters
        ----------
        dt : float       timestep [s]
        phi : float      rope angle [rad] (estimated)
        alpha : float    lower body angle [rad]
        theta : float    upper body angle [rad]

        Returns
        -------
        delta_target : float   desired hip angle [rad]
        """
        beta = self.compute_beta(phi, alpha, theta)
        
        # Calculate beta_dot via numerical differentiation
        if self.beta_prev is None:
            beta_dot = 0.0
        else:
            beta_dot = (beta - self.beta_prev) / dt
        self.beta_prev = beta

        # Predictive CoM angle
        T_pred = min(0.08, self.c_pred * abs(beta))
        beta_eff = beta + beta_dot * T_pred

        if self.phi_prev is None:
            self.phi_prev = phi
        phi_crossed_zero = (self.phi_prev is not None and np.sign(self.phi_prev) != np.sign(phi) and self.phi_prev != phi)

        self.t_phase += dt

        is_falling = (np.sign(beta) == np.sign(beta_dot)) or (abs(beta_dot) < 1e-5)
        need_fold = abs(beta_eff) > self.beta_threshold and is_falling

        # Retrigger condition: prevent same-direction retriggering and apply 40ms cooldown
        need_fold_retrigger = need_fold and (np.sign(beta_eff) != self.fold_sign) and (self.t_phase > 0.04)

        if self.phase == Phase.IDLE:
            self.delta_target = 0.0
            if need_fold:
                self._initiate_fold(beta, beta_dot, "IDLE")

        elif self.phase == Phase.FOLD:
            pos = self._fold_profile(self.t_phase)
            self.delta_target = self.fold_sign * pos
            if self.t_phase >= self.T_f:
                self.delta_target = self.fold_sign * abs(self.d_fold)
                self.phase = Phase.WAIT1
                self.t_phase = 0.0
                self._log(f"WAIT1 start: hold delta={np.degrees(self.delta_target):.0f}d")

        elif self.phase == Phase.WAIT1:
            # Hold at d_fold
            self.delta_target = self.fold_sign * abs(self.d_fold)
            if need_fold_retrigger:
                self.phase = Phase.EXTEND
                self.t_phase = 0.0
                self._log("WAIT1 early cut (need_fold_retrigger) -> EXTEND start")
            elif self.t_phase >= self.T_w1:
                self.phase = Phase.EXTEND
                self.t_phase = 0.0
                self._log("EXTEND start")

        elif self.phase == Phase.EXTEND:
            pos = self._fold_profile(self.t_phase)
            # Reverse: go from d_fold back to 0
            self.delta_target = self.fold_sign * (abs(self.d_fold) - pos)
            
            if phi_crossed_zero:
                self._log(f"EXTEND early cut: phi crossed zero. Transitioning directly to IDLE. beta={np.degrees(beta):.2f}d")
                self.delta_target = 0.0
                self.phase = Phase.IDLE
                self.t_phase = 0.0
                self.beta_prev = None
            elif self.t_phase >= self.T_f:
                if self.skipWait2Entirely:
                    self.delta_target = 0.0
                    self.phase = Phase.IDLE
                    self.t_phase = 0.0
                    self.beta_prev = None
                    self._log("EXTEND complete: skipping WAIT2. Transitioning directly to IDLE.")
                else:
                    self.delta_target = 0.0
                    self.phase = Phase.WAIT2
                    self.t_phase = 0.0
                    self.phi_crossed = False
                    self._log(f"WAIT2 start: phi={np.degrees(phi):.2f}d")

        elif self.phase == Phase.WAIT2:
            self.delta_target = 0.0
            
            if need_fold_retrigger:
                self._initiate_fold(beta, beta_dot, "WAIT2 (need_fold)")
            elif self.t_phase >= self.T_w2_min and phi_crossed_zero:
                self.phi_crossed = True
                self._log(f"WAIT2 complete (phi crossed zero): beta={np.degrees(beta):.2f}d")
                self.phase = Phase.IDLE
                self.t_phase = 0.0
                self.beta_prev = None
            elif self.t_phase > 2.0:
                self._log(f"IDLE: W2 timeout (phi={np.degrees(phi):.2f}d)")
                self.phase = Phase.IDLE
                self.t_phase = 0.0
                self.beta_prev = None

        self.phi_prev = phi
        return self.delta_target

    def _log(self, msg):
        self.phase_log.append(msg)
        print(f"  [FWE #{self.cycle_count}] {msg}")

    @property
    def phase_name(self):
        return self.phase.name
