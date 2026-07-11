/*
 * ==================================================================
 *  슬랙라인 밸런싱 v2 — 전류제어 + 단일칼만(가속도계 선형가속 보상)
 *  OpenCR 1.0 + Dynamixel XM430-W210-R  (2026-07-02)
 * ==================================================================
 *
 *  ★ lqr_balance_current.ino(v1) 대비 수정 5가지 (v1은 보존, 이 파일이 정본):
 *
 *  [버그1·확정] 자이로 단위 16.4배 — imu.gyroData[]는 deg/s가 아니라
 *    raw LSB(2000dps/32768≈0.061dps/LSB)였다. θ̇가 16.4배 과대
 *    → 구버전들에서 "θ가 엄청 튀는" 직접 원인.
 *    수정: 라이브러리가 dps로 변환해 두는 imu.gy 사용.
 *
 *  [버그2·확정] 57600 baud로는 500Hz 불가 — pos·vel 읽기+전류쓰기
 *    3 트랜잭션 ≈ 12~14ms → 실제 ~70Hz인데 Ad/Bd/K/L은 2ms 가정.
 *    수정: (a) 1Mbps 자동 승격(57600으로 발견되면 BAUD_RATE=3 기록 후 재접속),
 *          (b) Present Velocity(128)+Position(132) 인접 레지스터를
 *              8바이트 단일 read로 → 트랜잭션 2회→1회,
 *          (c) 제어 블록을 micros() 기준 정확히 2ms 페이싱,
 *          (d) [RATE]<450Hz면 경고.
 *
 *  [버그3·잠재] 엔코더·속도에 MOTOR_DIR 미적용 — Dynamixel은 +전류와
 *    +위치가 같은 회전방향이라 MOTOR_DIR=-1로 뒤집으면 δ·δ̇도 뒤집어야 함.
 *    수정: getDelta/getDeltaDot에 MOTOR_DIR 곱.
 *
 *  [버그4·경미] 마운트 오프셋·가속도 기준이 단일 샘플 → 200샘플 평균.
 *    commandTorque 반올림을 곱 결과 부호 기준으로.
 *
 *  [개선5·센싱★] 가속도계 측정모델 확장 — 이 로봇은 정상 동작 중에도
 *    발이 원호 위에서 흔들려 IMU에 상시 선형가속. v1은 1g 게이팅으로
 *    θ_acc를 자주 버렸다. v2 측정모델: z0 = θ − ẍ_imu/g (선형!)
 *      innov0 = z_thAcc − (C0_acc·xPred + D0_FEED·τ)
 *    → 흔들림·구동 중에도 가속도계 정보 사용. 게이팅은 모델 밖 충격
 *    (베어링 스틱션 등) 대비 안전장치로 유지.
 *
 *  상수 출처: compute_K_measured_v2.py (v11_small 예시값, DT=0.002, R=0.30,
 *            ELL_IMU=0.13 ★실측 후 전부 재계산·교체할 것)
 *  부호 규약(v17): θ,α 연직기준 시계+, δ=θ−α(앞접기+), τ→[0,−1,+1], τ=−K·x
 *
 *  [시리얼 명령]  z:0점  c:제어ON/OFF  +/-:gain  g:표시  t:상태  d:진단  x:비상정지
 *
 *  ⚠ 브링업(순서대로):
 *   0) 부팅 로그에서 baud 1Mbps 접속 확인. [RATE] 500Hz 근처 확인.
 *   1) 'd' 진단: 로봇을 ~1초에 90° 돌려 thDot이 ~90(±20) deg/s인지 확인
 *      (수백~천 단위면 자이로 단위 문제 재발 — imu.gy 확인).
 *      앞으로 기울이면 thAcc +, δ 앞접기 방향 +인지 확인.
 *   2) MOTOR_DIR: torque_cal로 +전류→+δ(앞접기) 개루프 확인. 아니면 -1로.
 *      (v2는 δ·δ̇에도 자동 반영됨)
 *   3) 폐루프: gain 0.30에서 'z'→'c'. 개루프 불안정 플랜트라 gain<0.2는 발산 정상.
 *   4) CUR_SAFE 600(≈1.6A) 시작 → 안정 시 855.
 *   5) ★실물 줄(베어링 진자) 위에서 (모델은 발이 원호 위 가정).
 */

#include <Dynamixel2Arduino.h>
#include <IMU.h>

// === 다이나믹셀 ===
#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
const uint32_t DXL_BAUD_FAST = 1000000;  // 목표 baud (BAUD_RATE=3)
const uint32_t DXL_BAUD_SLOW = 57600;    // 공장 기본 (발견 시 1M로 승격)

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

cIMU imu;

// === 상수 ===
const float DEG2RAD = PI / 180.0f;
const float RAD2DEG = 180.0f / PI;
const float DT      = 0.002f;          // 500Hz
const uint32_t DT_US = 2000;

// ============================================================
//  전류제어 변환 (compute_K_measured_v2.py)
// ============================================================
const float TAU_MAX   = 3.0f;
const float TAU2UNIT  = 285.0822f;     // τ[N·m] → GoalCurrent[unit]
const int   CUR_LIMIT = 1193;          // 하드 상한 (3.21A)
int   CUR_SAFE        = 600;           // 브링업 안전 상한 → 안정 후 855
const int   MOTOR_DIR = +1;            // torque_cal로 검증. δ·δ̇에도 자동 적용됨(v2)
const int   IMU_DIR   = +1;            // 'd' 진단으로 검증

const float R_ARC  = 0.30f;
const float C1_PHI = 0.175000f;        // φ 초기추정용
const float C2_PHI = 0.093750f;

// ============================================================
//  LQR 게인 (v1과 동일 — 플랜트 불변)
// ============================================================
const float K_lqr[6] = {
  -18.5941f, -18.3188f, -10.3211f, -2.7182f, -2.1860f, -1.4159f
};

// ============================================================
//  ★칼만 v2 — 측정 z=[θ_acc, θ̇_gyro, δ, δ̇]
//  z0 측정행: z0 = θ − ẍ_imu/g  (선형가속 보상, ELL_IMU=0.13 가정)
//    innov0 = z_thAcc − (C0_acc·xPred + D0_FEED·τ)
//  z1: e6(θ̇)   z2: e3−e2(δ)   z3: e6−e5(δ̇)
// ============================================================
const float C0_acc[6] = {
  -0.440000f, -1.155000f, 0.715000f, 0.000000f, 0.000000f, 0.007747f
};
const float D0_FEED = 0.907238f;

const float L_kf[6][4] = {
  {-0.067695f, -0.152261f,  0.167658f, -0.152828f},  // phi
  {-0.004901f, -0.014676f, -0.165846f, -0.012635f},  // alpha
  {-0.002787f, -0.005110f,  0.189134f, -0.012267f},  // theta
  {-0.170285f, -0.480906f,  0.508374f, -0.319823f},  // phiDot
  {-0.054343f, -0.224994f,  0.058933f, -0.539743f},  // alphaDot
  { 0.006885f,  0.300758f,  0.095657f,  0.052575f}   // thetaDot
};

const float Ad_mat[6][6] = {
  { 0.99914986f, -0.00085811f,  0.00007336f,  0.00199943f, -0.00000057f, -0.00000195f},
  { 0.00147104f,  1.00180110f, -0.00033004f,  0.00000098f,  0.00200120f,  0.00000875f},
  {-0.00023476f, -0.00061608f,  1.00038132f, -0.00000016f, -0.00000041f,  0.00198989f},
  {-0.85015100f, -0.85804627f,  0.07327674f,  0.99914986f, -0.00085811f, -0.00191853f},
  { 1.47093646f,  1.80054465f, -0.32957612f,  0.00147104f,  1.00180110f,  0.00862887f},
  {-0.23448556f, -0.61520875f,  0.38071807f, -0.00023476f, -0.00061608f,  0.99003221f}
};
const float Bd_vec[6] = {
  0.00144903f, -0.00352000f, 0.00193372f, 1.44871051f, -3.51791670f, 1.93080310f
};

// === 상태머신 ===
enum State { ST_IDLE, ST_READY, ST_RUNNING };
State state = ST_IDLE;
bool ctrl_on = false;
bool safety_stopped = false;
bool diag_mode = false;

float xHat[6] = {0};

int32_t encoder_offset = 0;
float imu_mount_offset = 0.0f;
float gyro_bias_y = 0.0f;        // rad/s (imu.gy 기준)
float accel_ref_sq = 0.0f;

float prev_tau = 0.0f;           // 직전 '실제 적용' τ (예측·보상에 사용)
float gain_scale = 0.30f;        // ★개루프 불안정: gain<0.2는 발산이 정상
uint32_t next_ctrl_us = 0;       // 2ms 페이싱
uint32_t dxl_fail_cnt = 0;

const float SAFETY_ANGLE_DEG = 80.0f;
const float DELTA_MAX_DEG    = 60.0f;

// ============================================================
//  센서 읽기
// ============================================================
float getThetaAccel() {              // atan2는 스케일 무관 → accData(raw) OK
  float ax = imu.accData[0], az = imu.accData[2];
  float raw = -90.0f * DEG2RAD - atan2(ax, az);
  return IMU_DIR * (raw - imu_mount_offset);
}
float getThetaDot() {                // ★버그1 수정: imu.gy = 라이브러리 dps 변환값
  // (구버전 라이브러리에서 imu.gy가 없어 컴파일 실패 시:
  //  imu.gyroData[1] * (2000.0f/32768.0f) 로 대체 — raw LSB→dps)
  return IMU_DIR * (imu.gy * DEG2RAD - gyro_bias_y);
}

// ★버그2 수정: Present Velocity(128,4B)+Position(132,4B) 8바이트 단일 read
//   ★버그3 수정: MOTOR_DIR을 δ·δ̇에도 적용
bool readDeltaBoth(float &delta_rad, float &delta_dot) {
  uint8_t buf[8];
  int32_t n = dxl.read(DXL_ID, 128, 8, buf, sizeof(buf), 10);
  if (n < 8) {  // 실패 시 개별 read 폴백
    dxl_fail_cnt++;
    int32_t pos = dxl.getPresentPosition(DXL_ID);
    float rpm   = dxl.getPresentVelocity(DXL_ID, UNIT_RPM);
    delta_rad = MOTOR_DIR * (float)(pos - encoder_offset) * (2.0f * PI / 4096.0f);
    delta_dot = MOTOR_DIR * rpm * 2.0f * PI / 60.0f;
    return false;
  }
  int32_t vel_raw = (int32_t)((uint32_t)buf[0] | ((uint32_t)buf[1] << 8) |
                              ((uint32_t)buf[2] << 16) | ((uint32_t)buf[3] << 24));
  int32_t pos_raw = (int32_t)((uint32_t)buf[4] | ((uint32_t)buf[5] << 8) |
                              ((uint32_t)buf[6] << 16) | ((uint32_t)buf[7] << 24));
  delta_rad = MOTOR_DIR * (float)(pos_raw - encoder_offset) * (2.0f * PI / 4096.0f);
  delta_dot = MOTOR_DIR * (vel_raw * 0.229f) * 2.0f * PI / 60.0f;  // 0.229 rpm/unit
  return true;
}
float getDeltaRad() { float d, dd; readDeltaBoth(d, dd); return d; }

// ============================================================
//  전류(토크) 명령 — 반환: 실제 적용된 모델 τ
//  ★버그4 수정: 반올림을 곱 결과(u) 부호 기준으로
// ============================================================
float commandTorque(float tau_model) {
  tau_model = constrain(tau_model, -TAU_MAX, TAU_MAX);
  float u = MOTOR_DIR * tau_model * TAU2UNIT;
  int unit = (int)(u + (u >= 0 ? 0.5f : -0.5f));
  unit = constrain(unit, -CUR_SAFE, CUR_SAFE);
  dxl.setGoalCurrent(DXL_ID, unit);
  return MOTOR_DIR * unit / TAU2UNIT;
}

void matVec6(const float A[6][6], const float x[6], float out[6]) {
  for (int i = 0; i < 6; i++) {
    out[i] = 0;
    for (int j = 0; j < 6; j++) out[i] += A[i][j] * x[j];
  }
}

// ============================================================
//  칼만 1스텝 — ★개선5: z0에 선형가속 보상 측정행 C0_acc + D0_FEED·τ
// ============================================================
void kalmanStep(float tau, float z_thAcc, float z_thDot,
                float z_delta, float z_deltaDot, bool accel_ok) {
  float xPred[6];
  matVec6(Ad_mat, xHat, xPred);
  for (int i = 0; i < 6; i++) xPred[i] += Bd_vec[i] * tau;

  float innov[4];
  float z0_pred = D0_FEED * tau;
  for (int j = 0; j < 6; j++) z0_pred += C0_acc[j] * xPred[j];
  innov[0] = z_thAcc    - z0_pred;                   // θ_acc (보상 모델)
  innov[1] = z_thDot    -  xPred[5];                 // θ̇
  innov[2] = z_delta    - (xPred[2] - xPred[1]);     // δ
  innov[3] = z_deltaDot - (xPred[5] - xPred[4]);     // δ̇

  if (!accel_ok) innov[0] = 0.0f;                    // 모델 밖 충격 게이팅

  const float innovLim[4] = {0.15f, 5.0f, 0.3f, 5.0f};  // θ_acc 클램프 0.5→0.15
  for (int i = 0; i < 4; i++) innov[i] = constrain(innov[i], -innovLim[i], innovLim[i]);

  for (int i = 0; i < 6; i++) {
    xHat[i] = xPred[i];
    for (int j = 0; j < 4; j++) xHat[i] += L_kf[i][j] * innov[j];
  }
  const float angleLim = PI / 2.0f, velLim = 30.0f;
  for (int i = 0; i < 3; i++) xHat[i] = constrain(xHat[i], -angleLim, angleLim);
  for (int i = 3; i < 6; i++) xHat[i] = constrain(xHat[i], -velLim, velLim);
}

// ============================================================
//  모터 접속 — 1Mbps 우선, 57600 발견 시 BAUD_RATE=3 기록 후 승격
// ============================================================
bool pingRetry(int tries) {
  for (int i = 0; i < tries; i++) { if (dxl.ping(DXL_ID)) return true; delay(200); }
  return false;
}
bool connectDXL() {
  dxl.begin(DXL_BAUD_FAST);
  dxl.setPortProtocolVersion(2.0);
  if (pingRetry(3)) { Serial.println("  DXL @ 1Mbps OK"); return true; }

  dxl.begin(DXL_BAUD_SLOW);
  if (!pingRetry(3)) return false;
  Serial.println("  DXL @ 57600 발견 → BAUD_RATE=3(1Mbps)로 승격...");
  dxl.torqueOff(DXL_ID); delay(50);
  dxl.writeControlTableItem(BAUD_RATE, DXL_ID, 3);   // EEPROM (torque off 필요)
  delay(300);
  dxl.begin(DXL_BAUD_FAST);
  if (pingRetry(5)) { Serial.println("  승격 성공: DXL @ 1Mbps"); return true; }
  Serial.println("  !!! 승격 실패 — Wizard로 baud 확인 필요");
  return false;
}

void motorInitCurrent() {
  dxl.torqueOff(DXL_ID);  delay(50);
  dxl.torqueOff(DXL_ID);  delay(50);
  dxl.setOperatingMode(DXL_ID, OP_CURRENT);
  delay(50);
  dxl.writeControlTableItem(CURRENT_LIMIT, DXL_ID, CUR_LIMIT);
  delay(20);
  encoder_offset = dxl.getPresentPosition(DXL_ID);
  dxl.torqueOn(DXL_ID);
  dxl.setGoalCurrent(DXL_ID, 0);
}

// ============================================================
//  캘리브 — ★버그4 수정: 마운트 오프셋·가속도 기준 200샘플 평균
// ============================================================
void calibratePose(int n_samp) {
  float off_sum = 0, ref_sum = 0;
  for (int i = 0; i < n_samp; i++) {
    imu.update();
    float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
    off_sum += -90.0f * DEG2RAD - atan2(ax, az);
    ref_sum += ax*ax + ay*ay + az*az;
    delay(2);
  }
  imu_mount_offset = off_sum / n_samp;
  accel_ref_sq     = ref_sum / n_samp;
}
void calibrateIMU() {   // 부팅 시: 자이로 바이어스(정지 상태) + 자세
  float gyro_sum = 0; int cnt = 0;
  for (int i = 0; i < 500; i++) {
    imu.update();
    if (i >= 100) { gyro_sum += imu.gy * DEG2RAD; cnt++; }  // ★imu.gy (dps)
    delay(4);
  }
  gyro_bias_y = gyro_sum / cnt;
  calibratePose(200);
}

void setup() {
  Serial.begin(115200);
  imu.begin(500);        // ★내부 갱신주기를 제어주기(500Hz)에 맞춤

  Serial.println("Connecting DXL...");
  if (!connectDXL()) { Serial.println("!!! Motor not found (12V/RS-485 확인)"); while (1) delay(1000); }

  Serial.print("IMU stabilizing + calib...");
  calibrateIMU();
  motorInitCurrent();
  Serial.println(" done");
  Serial.print("  Gyro bias = "); Serial.print(gyro_bias_y * RAD2DEG, 3); Serial.println(" deg/s");
  Serial.print("  Accel ref = "); Serial.println(sqrt(accel_ref_sq), 1);

  Serial.println("==========================================");
  Serial.println("  Slackline LQR v2 — CURRENT + accel-comp Kalman");
  Serial.println("  gyro=imu.gy(dps)  bus=1Mbps  pacing=2ms");
  Serial.print  ("  CUR_SAFE="); Serial.print(CUR_SAFE);
  Serial.print  (" MOTOR_DIR="); Serial.print(MOTOR_DIR);
  Serial.print  (" IMU_DIR=");   Serial.println(IMU_DIR);
  Serial.println("  z:zero c:ctrl +/-:gain g:show t:state d:diag x:STOP");
  Serial.println("==========================================");
  Serial.print("  Gain scale = "); Serial.println(gain_scale, 2);

  state = ST_IDLE;
}

void loop() {
  imu.update();

  // === 시리얼 명령 ===
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'z' || c == 'Z') {
      motorInitCurrent();
      calibratePose(200);      // ★200샘플 평균 (약 0.4s — 그동안 가만히 잡기)
      memset(xHat, 0, sizeof(xHat));
      prev_tau = 0; ctrl_on = false; safety_stopped = false; state = ST_READY;
      Serial.print(">>> ZERO SET. thAcc="); Serial.print(getThetaAccel()*RAD2DEG, 2);
      Serial.print(" delta="); Serial.println(getDeltaRad()*RAD2DEG, 2);
      Serial.println("    Press 'c' to start.");
    }
    else if (c == 'c' || c == 'C') {
      if (state < ST_READY) { Serial.println("!!! Press 'z' first"); return; }
      ctrl_on = !ctrl_on;
      if (ctrl_on) {
        float theta_now = getThetaAccel();
        float delta_now = getDeltaRad();
        float alpha_now = theta_now - delta_now;
        xHat[0] = -(C1_PHI*alpha_now + C2_PHI*theta_now) / R_ARC;
        xHat[1] = alpha_now; xHat[2] = theta_now;
        xHat[3] = 0; xHat[4] = 0; xHat[5] = getThetaDot();
        prev_tau = 0; state = ST_RUNNING; next_ctrl_us = micros();
        Serial.print(">>> CONTROL ON (gain="); Serial.print(gain_scale, 2); Serial.println(")");
      } else {
        dxl.setGoalCurrent(DXL_ID, 0);
        state = ST_READY;
        Serial.println(">>> CONTROL OFF (0 torque, free)");
      }
    }
    else if (c == '+' || c == '=') { gain_scale = constrain(gain_scale+0.1f,0.0f,1.0f);
      Serial.print(">>> gain="); Serial.println(gain_scale,2); }
    else if (c == '-' || c == '_') { gain_scale = constrain(gain_scale-0.1f,0.0f,1.0f);
      Serial.print(">>> gain="); Serial.println(gain_scale,2); }
    else if (c == 'g' || c == 'G') { Serial.print(">>> gain="); Serial.println(gain_scale,2); }
    else if (c == 't' || c == 'T') {
      float d, dd; readDeltaBoth(d, dd);
      Serial.print("[STATE] thAcc="); Serial.print(getThetaAccel()*RAD2DEG,2);
      Serial.print(" thDot="); Serial.print(getThetaDot()*RAD2DEG,1);
      Serial.print(" dl="); Serial.print(d*RAD2DEG,2);
      Serial.print(" dlDot="); Serial.print(dd*RAD2DEG,1);
      Serial.print(" | xHat phi="); Serial.print(xHat[0]*RAD2DEG,1);
      Serial.print(" a="); Serial.print(xHat[1]*RAD2DEG,1);
      Serial.print(" th="); Serial.print(xHat[2]*RAD2DEG,1);
      Serial.print(" | dxlFail="); Serial.print(dxl_fail_cnt);
      Serial.println();
    }
    else if (c == 'x' || c == 'X') {
      dxl.setGoalCurrent(DXL_ID, 0); dxl.torqueOff(DXL_ID);
      ctrl_on = false; state = ST_IDLE; Serial.println(">>> EMERGENCY STOP!");
    }
    else if (c == 'd' || c == 'D') {
      diag_mode = !diag_mode;
      if (diag_mode) { dxl.setGoalCurrent(DXL_ID,0); ctrl_on=false; state=ST_IDLE;
        Serial.println(">>> DIAG ON (0 torque)");
        Serial.println("    ①1초에 90° 돌려 thDot≈90 확인(수백이면 단위버그) ②앞기울임=thAcc+ ③앞접기=delta+"); }
      else Serial.println(">>> DIAG OFF");
    }
  }

  // === 진단 출력 (50Hz) ===
  if (diag_mode) {
    static unsigned long dp = 0;
    if (millis() - dp >= 20) {
      dp = millis();
      float ax=imu.accData[0],ay=imu.accData[1],az=imu.accData[2];
      float amag = sqrt(ax*ax+ay*ay+az*az);
      float d, dd; readDeltaBoth(d, dd);
      Serial.print("[DIAG] thAcc="); Serial.print(getThetaAccel()*RAD2DEG,1);
      Serial.print(" thDot="); Serial.print(getThetaDot()*RAD2DEG,1);
      Serial.print(" delta="); Serial.print(d*RAD2DEG,1);
      Serial.print(" dDot="); Serial.print(dd*RAD2DEG,1);
      Serial.print(" | ratio="); Serial.println(accel_ref_sq>0 ? amag*amag/accel_ref_sq : 1.0f, 2);
    }
  }

  // === 제어 루프 — ★정확히 2ms 페이싱 ===
  if (state == ST_RUNNING && ctrl_on) {
    uint32_t now_us = micros();
    if ((int32_t)(now_us - next_ctrl_us) < 0) return;   // 아직 2ms 안 됨
    next_ctrl_us += DT_US;
    if ((int32_t)(now_us - next_ctrl_us) > 10000) next_ctrl_us = now_us + DT_US;  // 과지연 리싱크

    // 루프율 계측 (Ad의 500Hz 가정 검증)
    static uint32_t loop_cnt = 0; static unsigned long rate_t0 = 0;
    loop_cnt++;
    if (rate_t0 == 0) rate_t0 = millis();
    if (millis() - rate_t0 >= 1000) {
      float hz = loop_cnt * 1000.0f / (millis() - rate_t0);
      loop_cnt = 0; rate_t0 = millis();
      Serial.print("[RATE] "); Serial.print(hz, 0); Serial.print(" Hz");
      if (hz < 450) Serial.print("  !!! <450Hz: baud/출력량 점검 (모델은 500Hz 가정)");
      Serial.println();
    }

    // --- 물리 측정 (pos+vel 단일 read) ---
    float z_thAcc = getThetaAccel();
    float z_thDot = getThetaDot();
    float z_delta, z_deltaDot;
    readDeltaBoth(z_delta, z_deltaDot);

    // 가속도 게이팅 (모델 밖 충격 대비 안전장치)
    float ax=imu.accData[0],ay=imu.accData[1],az=imu.accData[2];
    float ratio = (accel_ref_sq>0) ? (ax*ax+ay*ay+az*az)/accel_ref_sq : 1.0f;
    bool accel_ok = (ratio > 0.64f && ratio < 1.56f);

    // --- 칼만 (입력 = 직전 실제 적용 τ) ---
    kalmanStep(prev_tau, z_thAcc, z_thDot, z_delta, z_deltaDot, accel_ok);

    // --- LQR ---
    float tau = 0;
    for (int i = 0; i < 6; i++) tau -= K_lqr[i] * xHat[i];
    tau *= gain_scale;

    // --- 안전 체크 ---
    bool over = (fabs(xHat[2]) > SAFETY_ANGLE_DEG*DEG2RAD) ||
                (fabs(xHat[1]) > SAFETY_ANGLE_DEG*DEG2RAD) ||
                (fabs(z_thAcc) > SAFETY_ANGLE_DEG*DEG2RAD);
    if (over) {
      dxl.setGoalCurrent(DXL_ID, 0);
      ctrl_on = false; state = ST_READY; safety_stopped = true;
      Serial.println("\n!!! SAFETY STOP (>80deg)");
      Serial.print("  thAcc="); Serial.print(z_thAcc*RAD2DEG,1);
      Serial.print(" xHat th="); Serial.print(xHat[2]*RAD2DEG,1);
      Serial.print(" a="); Serial.print(xHat[1]*RAD2DEG,1);
      Serial.print(" phi="); Serial.print(xHat[0]*RAD2DEG,1);
      Serial.print(" tau="); Serial.println(tau,3);
      Serial.println("Press 'z' to re-zero.");
      return;
    }

    prev_tau = commandTorque(tau);

    // --- 출력 10Hz ---
    static unsigned long lp = 0;
    if (millis() - lp >= 100) {
      lp = millis();
      Serial.print("[CUR] thA="); Serial.print(z_thAcc*RAD2DEG,1);
      Serial.print(" th^="); Serial.print(xHat[2]*RAD2DEG,1);
      Serial.print(" a^="); Serial.print(xHat[1]*RAD2DEG,1);
      Serial.print(" ph^="); Serial.print(xHat[0]*RAD2DEG,1);
      Serial.print(" dl="); Serial.print(z_delta*RAD2DEG,1);
      Serial.print(" tau="); Serial.print(prev_tau,2);
      Serial.print(accel_ok?"":" [gated]");
      Serial.println();
    }
    return;   // 제어 중엔 delay 없이 페이싱으로 주기 유지
  }

  if (state == ST_IDLE) {
    static unsigned long lp=0;
    if (millis()-lp>=2000){ lp=millis();
      Serial.print("[IDLE] thAcc="); Serial.print(getThetaAccel()*RAD2DEG,1);
      Serial.println("  -- 'z' to zero-set"); }
  }
  if (state == ST_READY && !ctrl_on && !safety_stopped) {
    static unsigned long lp=0;
    if (millis()-lp>=1000){ lp=millis();
      Serial.print("[READY] thAcc="); Serial.print(getThetaAccel()*RAD2DEG,1);
      Serial.print(" dl="); Serial.print(getDeltaRad()*RAD2DEG,1);
      Serial.print("  -- 'c' (gain="); Serial.print(gain_scale,2); Serial.println(")"); }
  }

  delay(1);
}
