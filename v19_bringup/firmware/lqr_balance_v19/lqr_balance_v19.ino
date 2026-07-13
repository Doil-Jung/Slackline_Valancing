/*
 * ==================================================================
 *  lqr_balance_v19 — 재설계 로봇 LQR 균형 (φ 직접측정 + 전류제어)
 *  OpenCR 1.0 + XM430-W210-R + 상단 φ 절대엔코더   (2026-07-12)
 * ==================================================================
 *  lqr_balance_current_v2 를 계승하되 v19 재설계 반영:
 *
 *  [신규1] ★φ 직접측정 — 상단 엔코더가 칼만의 z0 로 들어간다.
 *          측정 z = [φ_enc, θ_acc(선형가속 보상), θ̇_gyro, δ_enc, δ̇] (5행)
 *          v2의 φ 간접 초기추정(C1_PHI/C2_PHI)은 폐기(불필요).
 *  [신규2] ★게인이 줄 관성 I_r·S_r·감쇠 c_φ 포함 플랜트에서 계산됨
 *          (gains_v19.h — compute_gains_v19.py 가 생성. 직접 수정 금지)
 *  [유지]  v2의 검증된 처리 전부: imu.gy(자이로 단위버그 회피), 1Mbps 승격,
 *          vel+pos 8바이트 단일 read, 2ms 페이싱, MOTOR_DIR 의 δ·δ̇ 반영,
 *          200샘플 캘리브, 선형가속 보상 innovation, 게이팅, 안전정지.
 *
 *  [시리얼]  z:0점(자세+φ)  c:제어ON/OFF  +/-:gain  t:상태  d:진단  x:비상정지
 *
 *  ⚠ 브링업 순서 (v19):
 *   0) encoder_phi / imu_axis_check / torque_cal 로 ENC_PHI_DIR·IMU_DIR·MOTOR_DIR 확정
 *      → 이 파일 상단 상수 3개에 반영.
 *   1) 실측 → params_v19.py 갱신 → compute_gains_v19.py → sim_check_v19.py 전부 PASS
 *   2) 부팅 로그: 1Mbps, [RATE] ~500Hz, GAINS_MEASURED 경고 없는지.
 *   3) 'd' 진단으로 φ·θ 부호 최종 확인 (로봇을 +x로 밀면 phi +).
 *   4) 로봇을 손으로 잡고 'z'→'c', gain 0.3 시작. CUR_SAFE 600 → 안정 시 855.
 */
#include <Dynamixel2Arduino.h>
#include <IMU.h>
#include "phi_encoder.h"
#include "gains_v19.h"

// === 방향 상수 — ★브링업에서 확정 ===
const int MOTOR_DIR = +1;     // torque_cal: +전류 → +δ(앞접기)
const int IMU_DIR   = +1;     // imu_axis_check: 앞기울임 → thAcc +
// ENC_PHI_DIR 은 phi_encoder.h 전역 — setup()에서 설정
const int PHI_DIR_SET = +1;   // encoder_phi: +x로 밀면 phi +

// === 다이나믹셀 ===
#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
const uint32_t DXL_BAUD_FAST = 1000000;
const uint32_t DXL_BAUD_SLOW = 57600;
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;
cIMU imu;

const float DEG2RAD = PI / 180.0f;
const float RAD2DEG = 180.0f / PI;
const uint32_t DT_US = 2000;      // 500Hz (gains_v19.h 의 DT_MODEL 과 일치해야 함)

int CUR_SAFE = CUR_SAFE_U;        // 브링업 상한 → 안정 후 855

// === 상태 ===
enum State { ST_IDLE, ST_READY, ST_RUNNING };
State state = ST_IDLE;
bool ctrl_on = false, safety_stopped = false, diag_mode = false;
float xHat[6] = {0};
int32_t servo_offset = 0;
float imu_mount_offset = 0, gyro_bias_y = 0, accel_ref_sq = 0;
float prev_tau = 0;
float gain_scale = 0.30f;
uint32_t next_ctrl_us = 0, dxl_fail_cnt = 0;
const float SAFETY_ANGLE_DEG = 80.0f;

// ============================================================
//  센서 (v2 검증 방식 유지)
// ============================================================
float getThetaAccel() {
  float ax = imu.accData[0], az = imu.accData[2];
  return IMU_DIR * ((-90.0f * DEG2RAD - atan2(ax, az)) - imu_mount_offset);
}
float getThetaDot() { return IMU_DIR * (imu.gy * DEG2RAD - gyro_bias_y); }

bool readDeltaBoth(float &delta_rad, float &delta_dot) {
  uint8_t buf[8];
  int32_t n = dxl.read(DXL_ID, 128, 8, buf, sizeof(buf), 10);
  if (n < 8) {
    dxl_fail_cnt++;
    int32_t pos = dxl.getPresentPosition(DXL_ID);
    float rpm = dxl.getPresentVelocity(DXL_ID, UNIT_RPM);
    delta_rad = MOTOR_DIR * (float)(pos - servo_offset) * (2.0f * PI / 4096.0f);
    delta_dot = MOTOR_DIR * rpm * 2.0f * PI / 60.0f;
    return false;
  }
  int32_t vel = (int32_t)((uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
                          ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24));
  int32_t pos = (int32_t)((uint32_t)buf[4] | ((uint32_t)buf[5] << 8) |
                          ((uint32_t)buf[6] << 16) | ((uint32_t)buf[7] << 24));
  delta_rad = MOTOR_DIR * (float)(pos - servo_offset) * (2.0f * PI / 4096.0f);
  delta_dot = MOTOR_DIR * (vel * 0.229f) * 2.0f * PI / 60.0f;
  return true;
}
float getDeltaRad() { float d, dd; readDeltaBoth(d, dd); return d; }

float commandTorque(float tau_model) {
  tau_model = constrain(tau_model, -TAU_MAX_NM, TAU_MAX_NM);
  float u = MOTOR_DIR * tau_model * TAU2UNIT;
  int unit = (int)(u + (u >= 0 ? 0.5f : -0.5f));
  unit = constrain(unit, -CUR_SAFE, CUR_SAFE);
  dxl.setGoalCurrent(DXL_ID, unit);
  return MOTOR_DIR * unit / TAU2UNIT;
}

void matVec6(const float A[6][6], const float x[6], float out[6]) {
  for (int i = 0; i < 6; i++) { out[i] = 0; for (int j = 0; j < 6; j++) out[i] += A[i][j] * x[j]; }
}

// ============================================================
//  칼만 v19 — z = [φ_enc, θ_acc(보상), θ̇, δ, δ̇]  (5행)
// ============================================================
void kalmanStep(float tau, float z_phi, float z_thAcc, float z_thDot,
                float z_delta, float z_deltaDot, bool accel_ok) {
  float xPred[6];
  matVec6(Ad_mat, xHat, xPred);
  for (int i = 0; i < 6; i++) xPred[i] += Bd_vec[i] * tau;

  float innov[5];
  innov[0] = z_phi - xPred[0];                       // ★φ 직접측정
  float z1p = D1_FEED * tau;
  for (int j = 0; j < 6; j++) z1p += C1_acc[j] * xPred[j];
  innov[1] = z_thAcc - z1p;                          // θ_acc (선형가속 보상)
  innov[2] = z_thDot - xPred[5];
  innov[3] = z_delta - (xPred[2] - xPred[1]);
  innov[4] = z_deltaDot - (xPred[5] - xPred[4]);
  if (!accel_ok) innov[1] = 0.0f;                    // 모델 밖 충격 게이팅

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
//  모터 접속 (v2 방식) / 캘리브
// ============================================================
bool pingRetry(int tries) {
  for (int i = 0; i < tries; i++) { if (dxl.ping(DXL_ID)) return true; delay(200); }
  return false;
}
bool connectDXL() {
  dxl.begin(DXL_BAUD_FAST); dxl.setPortProtocolVersion(2.0);
  if (pingRetry(3)) { Serial.println("  DXL @ 1Mbps OK"); return true; }
  dxl.begin(DXL_BAUD_SLOW);
  if (!pingRetry(3)) return false;
  Serial.println("  DXL @ 57600 → 1Mbps 승격...");
  dxl.torqueOff(DXL_ID); delay(50);
  dxl.writeControlTableItem(BAUD_RATE, DXL_ID, 3); delay(300);
  dxl.begin(DXL_BAUD_FAST);
  return pingRetry(5);
}
void motorInitCurrent() {
  dxl.torqueOff(DXL_ID); delay(50); dxl.torqueOff(DXL_ID); delay(50);
  dxl.setOperatingMode(DXL_ID, OP_CURRENT); delay(50);
  dxl.writeControlTableItem(CURRENT_LIMIT, DXL_ID, CUR_LIMIT_U); delay(20);
  servo_offset = dxl.getPresentPosition(DXL_ID);
  dxl.torqueOn(DXL_ID);
  dxl.setGoalCurrent(DXL_ID, 0);
}
void calibratePose(int n_samp) {
  float off = 0, ref = 0;
  for (int i = 0; i < n_samp; i++) {
    imu.update();
    float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
    off += -90.0f * DEG2RAD - atan2(ax, az);
    ref += ax * ax + ay * ay + az * az;
    delay(2);
  }
  imu_mount_offset = off / n_samp;
  accel_ref_sq = ref / n_samp;
}
void calibrateIMU() {
  float gs = 0; int cnt = 0;
  for (int i = 0; i < 500; i++) {
    imu.update();
    if (i >= 100) { gs += imu.gy * DEG2RAD; cnt++; }
    delay(4);
  }
  gyro_bias_y = gs / cnt;
  calibratePose(200);
}

void setup() {
  Serial.begin(115200);
  imu.begin(500);
  ENC_PHI_DIR = PHI_DIR_SET;

  Serial.println("Connecting DXL...");
  if (!connectDXL()) { Serial.println("!!! Motor not found"); while (1) delay(1000); }
  if (!encoderBegin()) Serial.println("!!! phi encoder 응답 없음 — 배선 확인 (제어 시작 금지)");

  Serial.print("IMU calib..."); calibrateIMU(); motorInitCurrent(); Serial.println(" done");

  Serial.println("==========================================");
  Serial.println("  LQR v19 — phi ENCODER + CURRENT + Kalman5");
#if !GAINS_MEASURED
  Serial.println("  ⚠⚠ gains_v19.h 가 '미실측 추정치' 기준 — 예행연습 전용!");
  Serial.println("     실측 → compute_gains_v19.py → sim_check PASS 후 재빌드하라.");
#endif
  Serial.print("  MOTOR_DIR="); Serial.print(MOTOR_DIR);
  Serial.print(" IMU_DIR="); Serial.print(IMU_DIR);
  Serial.print(" PHI_DIR="); Serial.println(PHI_DIR_SET);
  Serial.println("  z:zero c:ctrl +/-:gain t:state d:diag x:STOP");
  Serial.println("==========================================");
  state = ST_IDLE;
}

void loop() {
  imu.update();

  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'z') {
      motorInitCurrent();
      calibratePose(200);
      encoderZeroHere();                       // ★φ 영점 (정지·직립 상태에서!)
      memset(xHat, 0, sizeof(xHat));
      prev_tau = 0; ctrl_on = false; safety_stopped = false; state = ST_READY;
      Serial.print(">>> ZERO. thAcc="); Serial.print(getThetaAccel() * RAD2DEG, 2);
      Serial.print(" phi="); Serial.print(encoderPhiRad() * RAD2DEG, 2);
      Serial.print(" dl="); Serial.println(getDeltaRad() * RAD2DEG, 2);
    }
    else if (c == 'c') {
      if (state < ST_READY) { Serial.println("!!! 'z' first"); return; }
      ctrl_on = !ctrl_on;
      if (ctrl_on) {
        // ★v19: φ는 엔코더 실측으로 초기화 (v2의 간접추정 폐기)
        xHat[0] = encoderPhiRad();
        xHat[2] = getThetaAccel();
        xHat[1] = xHat[2] - getDeltaRad();
        xHat[3] = 0; xHat[4] = 0; xHat[5] = getThetaDot();
        prev_tau = 0; state = ST_RUNNING; next_ctrl_us = micros();
        Serial.print(">>> ON (gain="); Serial.print(gain_scale, 2); Serial.println(")");
      } else {
        dxl.setGoalCurrent(DXL_ID, 0); state = ST_READY;
        Serial.println(">>> OFF");
      }
    }
    else if (c == '+' || c == '=') { gain_scale = constrain(gain_scale + 0.1f, 0.0f, 1.0f);
      Serial.print("gain="); Serial.println(gain_scale, 2); }
    else if (c == '-') { gain_scale = constrain(gain_scale - 0.1f, 0.0f, 1.0f);
      Serial.print("gain="); Serial.println(gain_scale, 2); }
    else if (c == 't') {
      float d, dd; readDeltaBoth(d, dd);
      Serial.print("[ST] phiEnc="); Serial.print(encoderPhiRad() * RAD2DEG, 2);
      Serial.print(" thAcc="); Serial.print(getThetaAccel() * RAD2DEG, 2);
      Serial.print(" dl="); Serial.print(d * RAD2DEG, 2);
      Serial.print(" | xHat ph="); Serial.print(xHat[0] * RAD2DEG, 1);
      Serial.print(" a="); Serial.print(xHat[1] * RAD2DEG, 1);
      Serial.print(" th="); Serial.print(xHat[2] * RAD2DEG, 1);
      Serial.print(" | encErr="); Serial.print(enc_err_cnt);
      Serial.print(" dxlFail="); Serial.println(dxl_fail_cnt);
    }
    else if (c == 'x') {
      dxl.setGoalCurrent(DXL_ID, 0); dxl.torqueOff(DXL_ID);
      ctrl_on = false; state = ST_IDLE; Serial.println(">>> E-STOP");
    }
    else if (c == 'd') {
      diag_mode = !diag_mode;
      if (diag_mode) { dxl.setGoalCurrent(DXL_ID, 0); ctrl_on = false; state = ST_IDLE;
        Serial.println(">>> DIAG (0 torque): ①+x로 밀면 phi+ ②앞기울임 thAcc+ ③앞접기 dl+"); }
    }
  }

  if (diag_mode) {
    static unsigned long dp = 0;
    if (millis() - dp >= 50) {
      dp = millis();
      float d, dd; readDeltaBoth(d, dd);
      Serial.print("[DIAG] phi="); Serial.print(encoderPhiRad() * RAD2DEG, 2);
      Serial.print(" thAcc="); Serial.print(getThetaAccel() * RAD2DEG, 1);
      Serial.print(" thDot="); Serial.print(getThetaDot() * RAD2DEG, 1);
      Serial.print(" dl="); Serial.println(d * RAD2DEG, 1);
    }
  }

  if (state == ST_RUNNING && ctrl_on) {
    uint32_t now_us = micros();
    if ((int32_t)(now_us - next_ctrl_us) < 0) return;
    next_ctrl_us += DT_US;
    if ((int32_t)(now_us - next_ctrl_us) > 10000) next_ctrl_us = now_us + DT_US;

    static uint32_t lc = 0; static unsigned long rt = 0;
    lc++;
    if (rt == 0) rt = millis();
    if (millis() - rt >= 1000) {
      float hz = lc * 1000.0f / (millis() - rt); lc = 0; rt = millis();
      Serial.print("[RATE] "); Serial.print(hz, 0);
      if (hz < 450) Serial.print(" !!! <450Hz");
      Serial.println();
    }

    // --- 측정 ---
    float z_phi = encoderPhiRad();                 // ★직접측정
    float z_thAcc = getThetaAccel();
    float z_thDot = getThetaDot();
    float z_delta, z_deltaDot;
    readDeltaBoth(z_delta, z_deltaDot);
    float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
    float ratio = (accel_ref_sq > 0) ? (ax*ax + ay*ay + az*az) / accel_ref_sq : 1.0f;
    bool accel_ok = (ratio > 0.64f && ratio < 1.56f);

    kalmanStep(prev_tau, z_phi, z_thAcc, z_thDot, z_delta, z_deltaDot, accel_ok);

    float tau = 0;
    for (int i = 0; i < 6; i++) tau -= K_lqr[i] * xHat[i];
    tau *= gain_scale;

    bool over = (fabs(xHat[2]) > SAFETY_ANGLE_DEG * DEG2RAD) ||
                (fabs(xHat[1]) > SAFETY_ANGLE_DEG * DEG2RAD) ||
                (fabs(z_thAcc) > SAFETY_ANGLE_DEG * DEG2RAD);
    if (over) {
      dxl.setGoalCurrent(DXL_ID, 0);
      ctrl_on = false; state = ST_READY; safety_stopped = true;
      Serial.println("\n!!! SAFETY STOP (>80deg) — 'z' to re-zero");
      return;
    }

    prev_tau = commandTorque(tau);

    static unsigned long lp = 0;
    if (millis() - lp >= 100) {
      lp = millis();
      Serial.print("[CUR] ph="); Serial.print(z_phi * RAD2DEG, 1);
      Serial.print(" th^="); Serial.print(xHat[2] * RAD2DEG, 1);
      Serial.print(" a^="); Serial.print(xHat[1] * RAD2DEG, 1);
      Serial.print(" dl="); Serial.print(z_delta * RAD2DEG, 1);
      Serial.print(" tau="); Serial.print(prev_tau, 2);
      Serial.println(accel_ok ? "" : " [gated]");
    }
    return;
  }

  if (state == ST_IDLE) {
    static unsigned long lp = 0;
    if (millis() - lp >= 2000) { lp = millis();
      Serial.print("[IDLE] phi="); Serial.print(encoderPhiRad() * RAD2DEG, 1);
      Serial.print(" thAcc="); Serial.print(getThetaAccel() * RAD2DEG, 1);
      Serial.println("  -- 'z'"); }
  }
  delay(1);
}
