/*
 * ==================================================================
 *  슬랙라인 밸런싱 — 전류제어(OP_CURRENT) + 단일 모델기반 칼만
 *  OpenCR 1.0 + Dynamixel XM430-W210-R  (2026-06-22)
 * ==================================================================
 *
 *  ★ lqr_balance.ino(위치제어+KP_EQUIV) 대체본. θ폭주 근본해결:
 *    (1) 진짜 토크: τ를 GoalCurrent로 직접 명령 → 칼만 예측 Bd·τ가 실제와 일치
 *        (위치제어의 '가짜 토크'가 칼만 예측을 어긋나게 해 양의피드백을
 *         만들던 고리를 끊는다.)
 *    (2) 단일 모델기반 칼만: 상보필터(CF) 제거. IMU·엔코더 '물리 측정'을
 *        그대로 칼만에 넣는다(직렬 중첩·α 오염 제거).
 *          z0 = θ_acc   (가속도계 중력기울기, 1g 근방 게이팅)
 *          z1 = θ̇_gyro
 *          z2 = δ        (엔코더 위치 = θ − α)
 *          z3 = δ̇        (PRESENT_VELOCITY = θ̇ − α̇)
 *        α·φ 는 측정 안 함 → 모델 커플링(Ad)+엔코더 구속으로 칼만이 복원.
 *
 *  부호 규약(v17): θ,α 연직기준 시계+ , δ=θ−α (앞접기+), τ→[0,−1,+1], τ=−K·x
 *  상수 출처: compute_K_measured.py (v11_small 예시값, DT=0.002, R=0.30).
 *            ★실측(m,ℓ,I,Kt) 완료 후 재계산해 교체할 것.
 *
 *  [시리얼 명령]
 *    z : 0점 세팅 (직립으로 손으로 잡고)  c : 제어 ON/OFF
 *    + / - : gain_scale ±0.1   g : gain 표시   t : 상태 1회
 *    d : 센서 진단 토글   x : 비상정지(토크 OFF)
 *
 *  ⚠ 브링업(중요 — 순서대로):
 *    ★이 플랜트는 개루프 불안정 → '낮은 게인부터'가 아니다. gain<0.2면 발산.
 *    1) 센서: 'd'로 손으로 기울여 thAcc·δ·δ̇ 부호·크기 확인.
 *    2) MOTOR_DIR: 폐루프로 방향 찾지 말 것(낮은 게인=발산). 먼저 `torque_cal` 스케치로
 *       작은 +전류가 힙을 +δ(앞접기) 방향으로 미는지 개루프 확인 → 아니면 MOTOR_DIR 뒤집기.
 *    3) 폐루프: gain_scale=0.30(안정 하한 근처)에서 'z'→'c'. 잘 잡으면 0.5→0.7→1.0로 ↑.
 *    4) CUR_SAFE 600(≈1.6A) 시작, 안정 확인 후 855까지. 80° 초과 시 자동 안전정지.
 *    5) ★실물 줄(베어링 진자) 위에서 할 것 — 모델은 발이 줄 위(φ 원호)에 있다고 가정.
 */

#include <Dynamixel2Arduino.h>
#include <IMU.h>

// === 다이나믹셀 ===
#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     57600     // ⚠ 500Hz에서 pos+vel 읽기+전류쓰기 대역 빠듯하면
                               //   모터 baud를 1Mbps로 올리고 이 값도 교체 권장.
Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

cIMU imu;

// === 상수 ===
const float DEG2RAD = PI / 180.0f;
const float RAD2DEG = 180.0f / PI;
const float DT      = 0.002f;     // 500Hz

// ============================================================
//  전류제어 변환 (compute_K_measured.py)
// ============================================================
const float TAU_MAX   = 3.0f;       // N·m (연속운전 2.0~2.5 권장)
const float TAU2UNIT  = 285.0822f;  // τ[N·m] → GoalCurrent[unit]
const int   CUR_LIMIT = 1193;       // 하드 상한 (= 3.21A)
int   CUR_SAFE        = 600;        // ★브링업 안전 상한(unit). 안정확인 후 855까지 ↑
const int   MOTOR_DIR = +1;         // 전류부호 ↔ 모델 τ 방향. torque_cal로 검증됨(+τ→앞접기 OK)
const int   IMU_DIR   = +1;         // IMU θ·θ̇ 부호. 실물검증: 앞으로 기울면 thAcc + (정상)

const float R_ARC  = 0.30f;
const float C1_PHI = 0.175000f;     // φ 보조추정 (제어 시작 시 초기화용)
const float C2_PHI = 0.093750f;

// ============================================================
//  LQR 게인 K (6×1) — τ = −K·x
// ============================================================
const float K_lqr[6] = {
  -18.5941f, -18.3188f, -10.3211f, -2.7182f, -2.1860f, -1.4159f
};

// ============================================================
//  단일 칼만 게인 L (6×4)  — 측정 z=[θ_acc, θ̇_gyro, δ, δ̇]
//    C row0: θ        C row1: θ̇
//    C row2: δ=θ−α    C row3: δ̇=θ̇−α̇
// ============================================================
const float L_kf[6][4] = {
  {-0.018827f, -0.171852f,  0.162912f, -0.172294f},  // phi
  { 0.019463f, -0.004901f, -0.186718f, -0.001742f},  // alpha
  { 0.020869f,  0.004972f,  0.168658f, -0.001074f},  // theta
  { 0.002568f, -0.391297f,  0.375370f, -0.225330f},  // phiDot
  { 0.001309f, -0.236552f,  0.031931f, -0.550102f},  // alphaDot
  { 0.000414f,  0.302207f,  0.098732f,  0.053876f}   // thetaDot
};

// ============================================================
//  이산 모델 Ad (6×6), Bd (6×1)  — DT=0.002
// ============================================================
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

// 칼만 추정 상태 [phi, alpha, theta, phiDot, alphaDot, thetaDot]
float xHat[6] = {0};

// 오프셋/캘리브
int32_t encoder_offset = 0;
float imu_mount_offset = 0.0f;   // 부팅/0점 시 θ=0 기준
float gyro_bias_y = 0.0f;        // rad/s
float accel_ref_sq = 0.0f;       // 정지 1g 크기²

float prev_tau = 0.0f;           // 직전에 '실제로 명령된' 모델 τ (예측에 사용)
// ★이 플랜트는 개루프 불안정(도립진자) → gain이 너무 낮으면 균형이 아니라 *발산*한다.
//   시뮬상 최소 안정 게인 ≈ 0.2~0.3. gain 0.01/0.05/0.1 은 전부 발산(안전정지).
//   따라서 0.3부터 시작해 1.0까지 올린다. (MOTOR_DIR은 torque_cal로 먼저 확인할 것!)
float gain_scale = 0.30f;
unsigned long prev_loop_us = 0;

// 안전 한계
const float SAFETY_ANGLE_DEG = 80.0f;
const float DELTA_MAX_DEG    = 60.0f;   // 힙각 소프트 제한(추정·경고용)

// ============================================================
//  센서 읽기
// ============================================================
float getThetaAccel() {          // 가속도계 중력기울기 → 절대 θ (IMU_DIR로 규약정렬)
  float ax = imu.accData[0], az = imu.accData[2];
  float raw = -90.0f * DEG2RAD - atan2(ax, az);   // IMU 자연부호
  return IMU_DIR * (raw - imu_mount_offset);
}
float getThetaDot() {            // 자이로 Y (바이어스 보정, IMU_DIR로 규약정렬)
  return IMU_DIR * (imu.gyroData[1] * DEG2RAD - gyro_bias_y);
}
float getDeltaRad() {            // 엔코더 위치 = δ = θ − α
  int32_t raw = dxl.getPresentPosition(DXL_ID);
  return (float)(raw - encoder_offset) * (2.0f * PI / 4096.0f);
}
float getDeltaDotRad() {         // PRESENT_VELOCITY → δ̇ [rad/s]
  float rpm = dxl.getPresentVelocity(DXL_ID, UNIT_RPM);
  return rpm * 2.0f * PI / 60.0f;
}

// ============================================================
//  전류(토크) 명령. 반환: 실제로 적용된 모델 τ (current cap 반영)
// ============================================================
float commandTorque(float tau_model) {
  tau_model = constrain(tau_model, -TAU_MAX, TAU_MAX);
  int unit = (int)(MOTOR_DIR * tau_model * TAU2UNIT + (tau_model >= 0 ? 0.5f : -0.5f));
  unit = constrain(unit, -CUR_SAFE, CUR_SAFE);
  dxl.setGoalCurrent(DXL_ID, unit);
  return MOTOR_DIR * unit / TAU2UNIT;   // 실제 적용된 모델 τ
}

// === 행렬-벡터 곱 ===
void matVec6(const float A[6][6], const float x[6], float out[6]) {
  for (int i = 0; i < 6; i++) {
    out[i] = 0;
    for (int j = 0; j < 6; j++) out[i] += A[i][j] * x[j];
  }
}

// ============================================================
//  단일 칼만 1스텝 (물리측정 z=[θ_acc, θ̇_gyro, δ, δ̇])
//   accel_ok=false 면 θ_acc innovation 차단(게이팅)
// ============================================================
void kalmanStep(float tau, float z_thAcc, float z_thDot,
                float z_delta, float z_deltaDot, bool accel_ok) {
  // 1) 예측
  float xPred[6];
  matVec6(Ad_mat, xHat, xPred);
  for (int i = 0; i < 6; i++) xPred[i] += Bd_vec[i] * tau;

  // 2) innovation  e = z − C·xPred
  float innov[4];
  innov[0] = z_thAcc    -  xPred[2];                 // θ
  innov[1] = z_thDot    -  xPred[5];                 // θ̇
  innov[2] = z_delta    - (xPred[2] - xPred[1]);     // δ = θ − α
  innov[3] = z_deltaDot - (xPred[5] - xPred[4]);     // δ̇ = θ̇ − α̇

  // 게이팅: 선형가속 감지 시 가속도계 θ 측정 무시
  if (!accel_ok) innov[0] = 0.0f;

  // innovation 클램핑 (과대 보정 방지)  {θ, θ̇, δ, δ̇}
  const float innovLim[4] = {0.5f, 5.0f, 0.3f, 5.0f};
  for (int i = 0; i < 4; i++) innov[i] = constrain(innov[i], -innovLim[i], innovLim[i]);

  // 3) 보정  xHat = xPred + L·innov
  for (int i = 0; i < 6; i++) {
    xHat[i] = xPred[i];
    for (int j = 0; j < 4; j++) xHat[i] += L_kf[i][j] * innov[j];
  }

  // 상태 포화
  const float angleLim = PI / 2.0f, velLim = 30.0f;
  for (int i = 0; i < 3; i++) xHat[i] = constrain(xHat[i], -angleLim, angleLim);
  for (int i = 3; i < 6; i++) xHat[i] = constrain(xHat[i], -velLim, velLim);
}

// ============================================================
//  전류제어 모터 초기화
// ============================================================
void motorInitCurrent() {
  dxl.torqueOff(DXL_ID);  delay(50);
  dxl.torqueOff(DXL_ID);  delay(50);
  dxl.setOperatingMode(DXL_ID, OP_CURRENT);    // ★ 전류(토크) 제어
  delay(50);
  dxl.writeControlTableItem(CURRENT_LIMIT, DXL_ID, CUR_LIMIT);
  delay(20);
  encoder_offset = dxl.getPresentPosition(DXL_ID);  // 현재 δ=0 기준
  dxl.torqueOn(DXL_ID);
  dxl.setGoalCurrent(DXL_ID, 0);               // 0 토크로 시작
}

void calibrateIMU() {
  // 자이로 바이어스 + 마운트 오프셋 + 가속도 기준
  float gyro_sum = 0; int cnt = 0;
  for (int i = 0; i < 500; i++) {
    imu.update();
    if (i >= 100) { gyro_sum += imu.gyroData[1] * DEG2RAD; cnt++; }
    delay(4);
  }
  gyro_bias_y = gyro_sum / cnt;
  imu_mount_offset = -90.0f * DEG2RAD - atan2(imu.accData[0], imu.accData[2]);
  float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
  accel_ref_sq = ax*ax + ay*ay + az*az;
}

void setup() {
  Serial.begin(115200);
  imu.begin();
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);

  dxl.torqueOff(DXL_ID);
  delay(300);
  bool found = false;
  for (int i = 0; i < 5; i++) { if (dxl.ping(DXL_ID)) { found = true; break; } delay(300); }
  if (!found) { Serial.println("!!! Motor not found (12V/RS-485 확인)"); while (1) delay(1000); }

  Serial.print("IMU stabilizing + gyro bias cal...");
  calibrateIMU();
  motorInitCurrent();
  Serial.println(" done");
  Serial.print("  Gyro bias = "); Serial.print(gyro_bias_y * RAD2DEG, 3); Serial.println(" deg/s");
  Serial.print("  Accel ref = "); Serial.println(sqrt(accel_ref_sq), 1);

  Serial.println("==========================================");
  Serial.println("  Slackline LQR — CURRENT control + single Kalman");
  Serial.println("  v11_small params, R=0.30, tauMax=3.0");
  Serial.print  ("  CUR_SAFE="); Serial.print(CUR_SAFE);
  Serial.print  (" unit ("); Serial.print(CUR_SAFE*2.69f/1000.0f, 2); Serial.print("A)  MOTOR_DIR=");
  Serial.println(MOTOR_DIR);
  Serial.println("  z:zero  c:ctrl  +/-:gain  g:show  t:state  d:diag  x:STOP");
  Serial.println("==========================================");
  Serial.print("  Gain scale = "); Serial.println(gain_scale, 2);

  state = ST_IDLE;
  prev_loop_us = micros();
}

void loop() {
  imu.update();

  // === 시리얼 명령 ===
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'z' || c == 'Z') {
      motorInitCurrent();
      imu_mount_offset = -90.0f * DEG2RAD - atan2(imu.accData[0], imu.accData[2]);
      { float ax=imu.accData[0],ay=imu.accData[1],az=imu.accData[2]; accel_ref_sq=ax*ax+ay*ay+az*az; }
      memset(xHat, 0, sizeof(xHat));
      prev_tau = 0; ctrl_on = false; safety_stopped = false; state = ST_READY;
      Serial.print(">>> ZERO SET. thAcc="); Serial.print(getThetaAccel()*RAD2DEG, 2);
      Serial.print(" delta="); Serial.println(getDeltaRad()*RAD2DEG, 2);
      Serial.println("    Press 'c' to start (gain low!).");
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
        prev_tau = 0; state = ST_RUNNING; prev_loop_us = micros();
        Serial.print(">>> CONTROL ON (gain="); Serial.print(gain_scale, 2); Serial.println(")");
      } else {
        dxl.setGoalCurrent(DXL_ID, 0);   // 전류제어: OFF = 0토크(자유)
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
      Serial.print("[STATE] thAcc="); Serial.print(getThetaAccel()*RAD2DEG,2);
      Serial.print(" thDot="); Serial.print(getThetaDot()*RAD2DEG,1);
      Serial.print(" dl="); Serial.print(getDeltaRad()*RAD2DEG,2);
      Serial.print(" dlDot="); Serial.print(getDeltaDotRad()*RAD2DEG,1);
      Serial.print(" | xHat phi="); Serial.print(xHat[0]*RAD2DEG,1);
      Serial.print(" a="); Serial.print(xHat[1]*RAD2DEG,1);
      Serial.print(" th="); Serial.print(xHat[2]*RAD2DEG,1);
      Serial.println();
    }
    else if (c == 'x' || c == 'X') {
      dxl.setGoalCurrent(DXL_ID, 0); dxl.torqueOff(DXL_ID);
      ctrl_on = false; state = ST_IDLE; Serial.println(">>> EMERGENCY STOP!");
    }
    else if (c == 'd' || c == 'D') {
      diag_mode = !diag_mode;
      if (diag_mode) { dxl.setGoalCurrent(DXL_ID,0); ctrl_on=false; state=ST_IDLE;
        Serial.println(">>> DIAG ON (0 torque). 손으로 기울여 thAcc/delta 확인. 'd' 종료"); }
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
      Serial.print("[DIAG] thAcc="); Serial.print(getThetaAccel()*RAD2DEG,1);
      Serial.print(" thDot="); Serial.print(getThetaDot()*RAD2DEG,1);
      Serial.print(" delta="); Serial.print(getDeltaRad()*RAD2DEG,1);
      Serial.print(" dDot="); Serial.print(getDeltaDotRad()*RAD2DEG,1);
      Serial.print(" | aMag="); Serial.print(amag,0);
      Serial.print(" ratio="); Serial.println(accel_ref_sq>0 ? amag*amag/accel_ref_sq : 1.0f, 2);
    }
  }

  // === 제어 루프 ===
  if (state == ST_RUNNING && ctrl_on) {
    unsigned long now_us = micros();
    float dt = (float)(now_us - prev_loop_us) * 1e-6f;
    prev_loop_us = now_us;
    if (dt < 0.0001f || dt > 0.05f) dt = DT;

    // ★루프속도 계측 (Ad가 가정한 500Hz를 실제로 도는지 확인)
    static uint32_t loop_cnt = 0; static unsigned long rate_t0 = 0; static float loop_hz = 0;
    loop_cnt++;
    if (rate_t0 == 0) rate_t0 = millis();
    if (millis() - rate_t0 >= 1000) {
      loop_hz = loop_cnt * 1000.0f / (millis() - rate_t0);
      loop_cnt = 0; rate_t0 = millis();
      Serial.print("[RATE] "); Serial.print(loop_hz, 0); Serial.println(" Hz  (target 500)");
    }

    // --- 물리 측정 ---
    float z_thAcc   = getThetaAccel();
    float z_thDot   = getThetaDot();
    float z_delta   = getDeltaRad();
    float z_deltaDot= getDeltaDotRad();

    // 가속도 게이팅 (선형가속 시 θ_acc 무시)
    float ax=imu.accData[0],ay=imu.accData[1],az=imu.accData[2];
    float ratio = (accel_ref_sq>0) ? (ax*ax+ay*ay+az*az)/accel_ref_sq : 1.0f;
    bool accel_ok = (ratio > 0.64f && ratio < 1.56f);

    // --- 단일 칼만 (입력=직전 실제 적용 τ) ---
    kalmanStep(prev_tau, z_thAcc, z_thDot, z_delta, z_deltaDot, accel_ok);

    // --- LQR: τ = −K·x ---
    float tau = 0;
    for (int i = 0; i < 6; i++) tau -= K_lqr[i] * xHat[i];
    tau *= gain_scale;

    // --- 안전 체크 (추정 θ,α) ---
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

    // --- 토크 → 전류 명령 (실제 적용 τ를 예측에 환원) ---
    prev_tau = commandTorque(tau);

    // --- 출력 (★10Hz로 줄임: 115200에서 한 줄 ~7ms라 100Hz는 루프를 throttle함) ---
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

  delay(1);  // ~500Hz
}
