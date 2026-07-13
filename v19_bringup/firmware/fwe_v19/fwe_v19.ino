/*
 * ==================================================================
 *  fwe_v19 — FWE 균형 시연 펌웨어 (재설계 로봇, 줄 관성·감쇠 반영)
 *  OpenCR 1.0 + XM430-W210-R + 상단 φ 절대엔코더   (2026-07-12)
 * ==================================================================
 *  fwe_demo_standalone 을 '대체'한다 (구버전은 자이로 단위버그 보유 — 사용 금지).
 *
 *  이론 대응 (연구정리 7/11 + 반복FEW 7/01 + 예측점 7/05):
 *   · 위험도  A = W4ᵀ[φ, β, φ̇, β̇]  — 감쇠 포함 4D 불안정 좌고유벡터 사영.
 *     (β성분=1 정규화 → A는 'β 환산 각도'. 무감쇠 극한 = u₂ᵀq_pred 예측변위)
 *   · 접기량  δf = ρ·γ·A   (γ_fold: 단일접기, γ_cycle: 완전사이클 — gains_v19.h,
 *     줄 관성·감쇠 포함 수치 맵 캘리브레이션으로 산출)
 *   · 반복 FEW: 매 사이클 A 재측정 → 등비 수축. ρ∈[0.8,1.0] (과대접기 금지)
 *
 *  δ 구동: 전류모드 유지 + δ 프로파일 PD 추종 (모드 전환 없음 → 칼만의 τ 입력이
 *          접기 중에도 정확 — v2 칼만을 그대로 신뢰할 수 있는 이유)
 *
 *  [시리얼] z:0점  1:단일접기(접고 유지)  2:완전사이클 1회  3:반복FEW  r:펴기
 *           s:정지(관찰)  t:상태  x:비상정지
 *
 *  ⚠ 순서: lqr_balance_v19 로 LQR 균형이 먼저 확인된 뒤에 FWE를 올린다.
 *     δ_fold 실행오차가 10% 넘으면(7/05 관측) 모드 3(반복)이 기본값이다 —
 *     단일접기(1)는 오차에 취약한 것이 '이론적으로 정상'이다.
 */
#include <Dynamixel2Arduino.h>
#include <IMU.h>
#include "phi_encoder.h"
#include "gains_v19.h"

const int MOTOR_DIR = +1;     // ★브링업 확정값 (lqr_balance_v19 와 동일하게)
const int IMU_DIR   = +1;
const int PHI_DIR_SET = +1;

// δ 프로파일 PD (전류모드 추종) — 진동하면 KD↑ 또는 KP↓, 느리면 KP↑
float KP_DELTA = 25.0f;       // [N·m/rad]
float KD_DELTA = 0.6f;        // [N·m·s/rad]

#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;
cIMU imu;

const float DEG2RAD = PI / 180.0f, RAD2DEG = 180.0f / PI;
const uint32_t DT_US = 2000;
int CUR_SAFE = CUR_SAFE_U;

// === FWE 상태머신 ===
enum Phase { PH_IDLE, PH_ARMED, PH_FOLD, PH_HOLD, PH_WAIT, PH_EXTEND, PH_REST };
Phase phase = PH_IDLE;
int mode = 0;                 // 1:단일접기 2:사이클1회 3:반복FEW
float xHat[6] = {0};
float delta_ref = 0, delta_ref_dot = 0;
float delta_target = 0;       // 이번 접기각 δf
float phase_t0 = 0;           // phase 시작 시각 [s]
int   cycle_cnt = 0;
const int MAX_CYCLES = 25;

int32_t servo_offset = 0;
float imu_mount_offset = 0, gyro_bias_y = 0, accel_ref_sq = 0;
float prev_tau = 0;
uint32_t next_ctrl_us = 0, dxl_fail_cnt = 0;
bool safety_stopped = false;
const float SAFETY_ANGLE_DEG = 80.0f;

// ============================================================
//  센서·모터 (lqr_balance_v19 와 동일 — 검증된 v2 계열)
// ============================================================
float getThetaAccel() {
  float ax = imu.accData[0], az = imu.accData[2];
  return IMU_DIR * ((-90.0f * DEG2RAD - atan2(ax, az)) - imu_mount_offset);
}
float getThetaDot() { return IMU_DIR * (imu.gy * DEG2RAD - gyro_bias_y); }

bool readDeltaBoth(float &d, float &dd) {
  uint8_t buf[8];
  int32_t n = dxl.read(DXL_ID, 128, 8, buf, sizeof(buf), 10);
  if (n < 8) {
    dxl_fail_cnt++;
    int32_t pos = dxl.getPresentPosition(DXL_ID);
    float rpm = dxl.getPresentVelocity(DXL_ID, UNIT_RPM);
    d = MOTOR_DIR * (float)(pos - servo_offset) * (2.0f * PI / 4096.0f);
    dd = MOTOR_DIR * rpm * 2.0f * PI / 60.0f;
    return false;
  }
  int32_t vel = (int32_t)((uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
                          ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24));
  int32_t pos = (int32_t)((uint32_t)buf[4] | ((uint32_t)buf[5] << 8) |
                          ((uint32_t)buf[6] << 16) | ((uint32_t)buf[7] << 24));
  d = MOTOR_DIR * (float)(pos - servo_offset) * (2.0f * PI / 4096.0f);
  dd = MOTOR_DIR * (vel * 0.229f) * 2.0f * PI / 60.0f;
  return true;
}
float commandTorque(float tau) {
  tau = constrain(tau, -TAU_MAX_NM, TAU_MAX_NM);
  float u = MOTOR_DIR * tau * TAU2UNIT;
  int unit = (int)(u + (u >= 0 ? 0.5f : -0.5f));
  unit = constrain(unit, -CUR_SAFE, CUR_SAFE);
  dxl.setGoalCurrent(DXL_ID, unit);
  return MOTOR_DIR * unit / TAU2UNIT;
}
void matVec6(const float A[6][6], const float x[6], float out[6]) {
  for (int i = 0; i < 6; i++) { out[i] = 0; for (int j = 0; j < 6; j++) out[i] += A[i][j] * x[j]; }
}

// 칼만 v19 (lqr_balance_v19 와 동일)
void kalmanStep(float tau, float z_phi, float z_thAcc, float z_thDot,
                float z_delta, float z_deltaDot, bool accel_ok) {
  float xPred[6];
  matVec6(Ad_mat, xHat, xPred);
  for (int i = 0; i < 6; i++) xPred[i] += Bd_vec[i] * tau;
  float innov[5];
  innov[0] = z_phi - xPred[0];
  float z1p = D1_FEED * tau;
  for (int j = 0; j < 6; j++) z1p += C1_acc[j] * xPred[j];
  innov[1] = z_thAcc - z1p;
  innov[2] = z_thDot - xPred[5];
  innov[3] = z_delta - (xPred[2] - xPred[1]);
  innov[4] = z_deltaDot - (xPred[5] - xPred[4]);
  if (!accel_ok) innov[1] = 0.0f;
  const float lim[5] = {0.10f, 0.15f, 5.0f, 0.3f, 5.0f};
  for (int i = 0; i < 5; i++) innov[i] = constrain(innov[i], -lim[i], lim[i]);
  for (int i = 0; i < 6; i++) {
    xHat[i] = xPred[i];
    for (int j = 0; j < 5; j++) xHat[i] += L_kf[i][j] * innov[j];
  }
  for (int i = 0; i < 3; i++) xHat[i] = constrain(xHat[i], -PI / 2, PI / 2);
  for (int i = 3; i < 6; i++) xHat[i] = constrain(xHat[i], -30.0f, 30.0f);
}

// ============================================================
//  위험도 A = W4ᵀ[φ, β, φ̇, β̇]   (β = P1R·α + P2R·θ)
// ============================================================
float riskA() {
  float beta    = P1R * xHat[1] + P2R * xHat[2];
  float betaDot = P1R * xHat[4] + P2R * xHat[5];
  return W4_risk[0] * xHat[0] + W4_risk[1] * beta
       + W4_risk[2] * xHat[3] + W4_risk[3] * betaDot;
}

// 삼각 가감속 프로파일: 0→dmax (T초).  out: (δref, δ̇ref)
void triProfile(float t, float T, float dmax, float &dr, float &drd) {
  if (t <= 0)      { dr = 0;    drd = 0; return; }
  if (t >= T)      { dr = dmax; drd = 0; return; }
  float a = 4.0f * dmax / (T * T);
  if (t < T / 2)   { dr = 0.5f * a * t * t;             drd = a * t; }
  else             { float tr = T - t; dr = dmax - 0.5f * a * tr * tr; drd = a * tr; }
}

const char* phName(Phase p) {
  switch (p) { case PH_IDLE: return "IDLE"; case PH_ARMED: return "ARMED";
    case PH_FOLD: return "FOLD"; case PH_HOLD: return "HOLD"; case PH_WAIT: return "WAIT";
    case PH_EXTEND: return "EXTEND"; case PH_REST: return "REST"; }
  return "?";
}

// ============================================================
//  접속·캘리브 (v2 계열)
// ============================================================
bool pingRetry(int n) { for (int i = 0; i < n; i++) { if (dxl.ping(DXL_ID)) return true; delay(200);} return false; }
bool connectDXL() {
  dxl.begin(1000000); dxl.setPortProtocolVersion(2.0);
  if (pingRetry(3)) return true;
  dxl.begin(57600);
  if (!pingRetry(3)) return false;
  dxl.torqueOff(DXL_ID); delay(50);
  dxl.writeControlTableItem(BAUD_RATE, DXL_ID, 3); delay(300);
  dxl.begin(1000000);
  return pingRetry(5);
}
void motorInitCurrent() {
  dxl.torqueOff(DXL_ID); delay(50); dxl.torqueOff(DXL_ID); delay(50);
  dxl.setOperatingMode(DXL_ID, OP_CURRENT); delay(50);
  dxl.writeControlTableItem(CURRENT_LIMIT, DXL_ID, CUR_LIMIT_U); delay(20);
  servo_offset = dxl.getPresentPosition(DXL_ID);
  dxl.torqueOn(DXL_ID); dxl.setGoalCurrent(DXL_ID, 0);
}
void calibrateAll() {
  float gs = 0; int cnt = 0;
  for (int i = 0; i < 500; i++) { imu.update();
    if (i >= 100) { gs += imu.gy * DEG2RAD; cnt++; } delay(4); }
  gyro_bias_y = gs / cnt;
  float off = 0, ref = 0;
  for (int i = 0; i < 200; i++) { imu.update();
    float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
    off += -90.0f * DEG2RAD - atan2(ax, az);
    ref += ax*ax + ay*ay + az*az; delay(2); }
  imu_mount_offset = off / 200; accel_ref_sq = ref / 200;
}

void setup() {
  Serial.begin(115200);
  imu.begin(500);
  ENC_PHI_DIR = PHI_DIR_SET;
  if (!connectDXL()) { Serial.println("!!! Motor not found"); while (1) delay(1000); }
  if (!encoderBegin()) Serial.println("!!! phi encoder 없음 — FWE는 φ 필수. 중단.");
  calibrateAll(); motorInitCurrent();
  Serial.println("==========================================");
  Serial.println("  FWE v19 — A=w'x 위험도 / rho·gamma·A 접기");
#if !GAINS_MEASURED
  Serial.println("  ⚠⚠ 미실측 게인 — 예행연습 전용!");
#endif
  Serial.print("  gamma_fold="); Serial.print(GAMMA_FOLD, 1);
  Serial.print(" gamma_cyc="); Serial.print(GAMMA_CYCLE, 1);
  Serial.print(" rho="); Serial.print(RHO_GAIN, 2);
  Serial.print(" trig="); Serial.print(A_TRIG_RAD * RAD2DEG, 2); Serial.println("deg");
  Serial.println("  z:zero 1:단일접기 2:사이클 3:반복FEW r:펴기 s:정지 t:상태 x:STOP");
  Serial.println("==========================================");
}

void startFold(float gamma) {
  float A = riskA();
  delta_target = RHO_GAIN * gamma * A;
  delta_target = constrain(delta_target, -DELTA_MAX_RAD, DELTA_MAX_RAD);
  phase = PH_FOLD; phase_t0 = micros() * 1e-6f;
  Serial.print("[FWE] FOLD: A="); Serial.print(A * RAD2DEG, 2);
  Serial.print("deg -> delta_f="); Serial.print(delta_target * RAD2DEG, 1); Serial.println("deg");
}

void loop() {
  imu.update();

  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'z') {
      motorInitCurrent(); calibrateAll(); encoderZeroHere();
      memset(xHat, 0, sizeof(xHat));
      prev_tau = 0; delta_ref = 0; delta_ref_dot = 0;
      phase = PH_IDLE; mode = 0; cycle_cnt = 0; safety_stopped = false;
      next_ctrl_us = micros();
      Serial.println(">>> ZERO (직립·정지 상태였어야 함)");
    }
    else if (c == '1' || c == '2' || c == '3') {
      if (safety_stopped) { Serial.println("!!! 'z' first"); return; }
      mode = c - '0'; cycle_cnt = 0;
      xHat[0] = encoderPhiRad(); xHat[2] = getThetaAccel();
      float d, dd; readDeltaBoth(d, dd);
      xHat[1] = xHat[2] - d; xHat[3] = 0; xHat[4] = 0; xHat[5] = getThetaDot();
      phase = PH_ARMED; next_ctrl_us = micros();
      Serial.print(">>> MODE "); Serial.print(mode);
      Serial.println(" ARMED — |A|>trig 이면 접는다 (외란을 살짝 주거나 기울여 놓기)");
    }
    else if (c == 'r') {           // 단일접기 후 수동 펴기
      if (phase == PH_HOLD) { phase = PH_EXTEND; phase_t0 = micros() * 1e-6f;
        Serial.println(">>> EXTEND"); }
    }
    else if (c == 's') { phase = PH_IDLE; mode = 0; dxl.setGoalCurrent(DXL_ID, 0);
      Serial.println(">>> STOP (0 torque 관찰)"); }
    else if (c == 't') {
      Serial.print("[ST] ph="); Serial.print(phName(phase));
      Serial.print(" A="); Serial.print(riskA() * RAD2DEG, 2);
      Serial.print("deg phi="); Serial.print(encoderPhiRad() * RAD2DEG, 1);
      Serial.print(" dl_ref="); Serial.print(delta_ref * RAD2DEG, 1);
      Serial.print(" cyc="); Serial.println(cycle_cnt);
    }
    else if (c == 'x') { dxl.setGoalCurrent(DXL_ID, 0); dxl.torqueOff(DXL_ID);
      phase = PH_IDLE; mode = 0; Serial.println(">>> E-STOP"); }
  }

  if (phase == PH_IDLE) { delay(1); return; }

  // ================= 500Hz 제어 블록 =================
  uint32_t now_us = micros();
  if ((int32_t)(now_us - next_ctrl_us) < 0) return;
  next_ctrl_us += DT_US;
  if ((int32_t)(now_us - next_ctrl_us) > 10000) next_ctrl_us = now_us + DT_US;
  float tnow = now_us * 1e-6f;

  // --- 측정 + 칼만 ---
  float z_phi = encoderPhiRad();
  float z_thAcc = getThetaAccel();
  float z_thDot = getThetaDot();
  float z_delta, z_deltaDot;
  readDeltaBoth(z_delta, z_deltaDot);
  float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
  float ratio = (accel_ref_sq > 0) ? (ax*ax + ay*ay + az*az) / accel_ref_sq : 1.0f;
  kalmanStep(prev_tau, z_phi, z_thAcc, z_thDot, z_delta, z_deltaDot,
             ratio > 0.64f && ratio < 1.56f);

  // --- 안전 ---
  if (fabs(xHat[1]) > SAFETY_ANGLE_DEG * DEG2RAD ||
      fabs(xHat[2]) > SAFETY_ANGLE_DEG * DEG2RAD ||
      fabs(z_thAcc) > SAFETY_ANGLE_DEG * DEG2RAD) {
    dxl.setGoalCurrent(DXL_ID, 0);
    phase = PH_IDLE; mode = 0; safety_stopped = true;
    Serial.println("\n!!! SAFETY STOP — 'z'"); return;
  }

  // --- 상태머신 ---
  float dt_ph = tnow - phase_t0;
  switch (phase) {
    case PH_ARMED:
      delta_ref = 0; delta_ref_dot = 0;
      if (fabs(riskA()) > A_TRIG_RAD)
        startFold(mode == 1 ? GAMMA_FOLD : GAMMA_CYCLE);
      break;
    case PH_FOLD:
      triProfile(dt_ph, T_FOLD_S, delta_target, delta_ref, delta_ref_dot);
      if (dt_ph >= T_FOLD_S) {
        phase = (mode == 1) ? PH_HOLD : PH_WAIT;
        phase_t0 = tnow;
        Serial.print("[FWE] "); Serial.print(mode == 1 ? "HOLD" : "WAIT");
        Serial.print(" A="); Serial.print(riskA() * RAD2DEG, 3); Serial.println("deg");
      }
      break;
    case PH_HOLD:            // 모드1: 접힌 채 유지 — "기다림의 상한 없음" 시연. 'r'로 펴기
      delta_ref = delta_target; delta_ref_dot = 0;
      break;
    case PH_WAIT:
      delta_ref = delta_target; delta_ref_dot = 0;
      if (dt_ph >= T_WAIT_S) { phase = PH_EXTEND; phase_t0 = tnow; }
      break;
    case PH_EXTEND: {
      float dr, drd;
      triProfile(dt_ph, T_EXTEND_S, delta_target, dr, drd);
      delta_ref = delta_target - dr; delta_ref_dot = -drd;   // 역재생
      if (dt_ph >= T_EXTEND_S) {
        delta_ref = 0; delta_ref_dot = 0;
        cycle_cnt++;
        Serial.print("[FWE] cycle "); Serial.print(cycle_cnt);
        Serial.print(" done. A="); Serial.print(riskA() * RAD2DEG, 3); Serial.println("deg");
        if (mode == 3 && cycle_cnt < MAX_CYCLES) { phase = PH_REST; phase_t0 = tnow; }
        else { phase = PH_ARMED; if (mode != 3) mode = 0, phase = PH_IDLE, dxl.setGoalCurrent(DXL_ID, 0); }
      }
      break;
    }
    case PH_REST:            // 반복FEW: 짧은 휴지 후 재측정·재접기
      delta_ref = 0; delta_ref_dot = 0;
      if (dt_ph >= T_REST_S) {
        if (fabs(riskA()) > A_TRIG_RAD) startFold(GAMMA_CYCLE);
        else { Serial.println("[FWE] |A|<trig — 균형 근방, ARMED 유지"); phase = PH_ARMED; }
      }
      break;
    default: break;
  }

  // --- δ 프로파일 PD 추종 (전류모드) ---
  float tau = KP_DELTA * (delta_ref - z_delta) + KD_DELTA * (delta_ref_dot - z_deltaDot);
  prev_tau = commandTorque(tau);

  // --- 로그 20Hz ---
  static unsigned long lp = 0;
  if (millis() - lp >= 50) {
    lp = millis();
    Serial.print("[FWE] "); Serial.print(phName(phase));
    Serial.print(" A="); Serial.print(riskA() * RAD2DEG, 2);
    Serial.print(" ph="); Serial.print(z_phi * RAD2DEG, 1);
    Serial.print(" dl="); Serial.print(z_delta * RAD2DEG, 1);
    Serial.print("/"); Serial.print(delta_ref * RAD2DEG, 1);
    Serial.print(" tau="); Serial.println(prev_tau, 2);
  }
}
