/*
 * ============================================
 * 슬랙라인 밸런싱 LQR + 칼만필터 — OpenCR + XM430
 * ============================================
 * 
 * v17 부호규약 + v11_small 파라미터
 * 
 * 부호 규약 (v17):
 *   θ : 상체 기울기 (연직 기준, 시계방향 +)
 *   α : 하체 기울기 (연직 기준, 시계방향 +)
 *   δ = θ − α : 힙 관절각 (앞으로 접기 = +)
 *   φ : 원호 위 발 각도 (칼만으로 추정)
 *   τ : 힙 토크 (α에 −τ, θ에 +τ)
 *   τ = −K·x (LQR 피드백)
 * 
 * [시리얼 명령]
 *   z : 0점 세팅 (직립 상태에서, 손으로 잡고)
 *   c : 제어 ON/OFF
 *   g : 게인 스케일 표시/변경 (0.0~1.0)
 *   t : 현재 상태 1회 출력
 *   x : 긴급 정지 (토크 OFF)
 *   + : 게인 스케일 +0.1
 *   - : 게인 스케일 -0.1
 */

#include <Dynamixel2Arduino.h>
#include <IMU.h>

// === 다이나믹셀 ===
#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     57600

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

// === IMU ===
cIMU imu;

// === 상수 ===
const float DEG2RAD = PI / 180.0f;
const float RAD2DEG = 180.0f / PI;
const float DT = 0.002f;   // 2ms 제어 주기 (500Hz)

// ============================================================
// 물리 파라미터 (v11_small + R=0.3)
// ============================================================
const float TAU_MAX = 4.0f;        // XM430 최대 토크 (N·m)
const float R_ARC   = 0.30f;       // 원호 반경 (m)

// φ 운동학 추정 계수 (IMU로 직접 관측 불가)
// φ ≈ −(C1·α + C2·θ) / R
const float C1_PHI = 0.175000f;    // (m1*ell1 + m2*L1) / Mt
const float C2_PHI = 0.093750f;    // m2*ell2 / Mt

// ============================================================
// LQR 게인 K (6×1) — compute_K.py 출력 (v17 규약)
// τ = −K·x = −(K₀φ + K₁α + K₂θ + K₃φ̇ + K₄α̇ + K₅θ̇)
// ============================================================
const float K_lqr[6] = {
  -36.9345f, -36.2186f, -20.8008f,
   -5.4460f,  -4.3731f,  -2.8358f
};

// ============================================================
// 칼만 게인 L (6×4) — 정상상태
// 행=상태(φ,α,θ,φ̇,α̇,θ̇), 열=관측(α,θ,α̇,θ̇)
// ============================================================
const float L_kf[6][4] = {
  {-0.084881f,  0.014342f,  0.221782f, -0.026472f},  // phi
  { 0.092297f,  0.002406f,  0.001504f, -0.006463f},  // alpha
  { 0.002406f,  0.092974f,  0.000564f,  0.006684f},  // theta
  {-0.197651f, -0.002785f,  0.408953f, -0.056555f},  // phiDot
  { 0.015040f,  0.005643f,  0.496211f, -0.062286f},  // alphaDot
  {-0.064628f,  0.066837f, -0.062286f,  0.115996f}   // thetaDot
};

// ============================================================
// 이산 시스템 행렬 Ad (6×6), Bd (6×1)
// ============================================================
const float Ad_mat[6][6] = {
  { 0.99978746f,-0.00021455f, 0.00001836f, 0.00099993f,-0.00000007f,-0.00000049f},
  { 0.00036780f, 1.00045044f,-0.00008263f, 0.00000012f, 0.00100015f, 0.00000222f},
  {-0.00005877f,-0.00015425f, 1.00009549f,-0.00000002f,-0.00000005f, 0.00099744f},
  {-0.42506446f,-0.42906918f, 0.03670241f, 0.99978746f,-0.00021455f,-0.00097932f},
  { 0.73554717f, 0.90069085f,-0.16513967f, 0.00036780f, 1.00045044f, 0.00440638f},
  {-0.11744771f,-0.30826071f, 0.19081236f,-0.00005877f,-0.00015425f, 0.99490861f}
};

const float Bd_vec[6] = {
  0.00036236f, -0.00088057f, 0.00048419f,
  0.72459640f, -1.76052361f, 0.96758961f
};

// === 상태머신 ===
enum State { ST_IDLE, ST_READY, ST_RUNNING };
State state = ST_IDLE;
bool ctrl_on = false;
bool safety_stopped = false;  // 안전 정지 후 READY 스팸 방지
bool diag_mode = false;       // IMU 진단 모드

// 칼만 필터 추정 상태
float xHat[6] = {0};  // [phi, alpha, theta, phiDot, alphaDot, thetaDot]

// 오프셋
int32_t encoder_offset = 0;

// 상보필터 (Complementary Filter) 상태
float theta_cf = 0.0f;               // 상보필터 출력 θ (rad)
float imu_mount_offset = 0.0f;       // IMU 장착 오프셋 (부팅 시 1회 캠리브)
float gyro_bias_y = 0.0f;            // 자이로 Y축 정적 바이어스 (rad/s)
float accel_ref_sq = 0.0f;           // 정지 시 가속도 크기² (기준값)
const float CF_ALPHA = 0.99f;        // 상보필터 계수 (0.99 = 자이로 99% 신뢰)

// 이전 토크 (예측에 사용)
float prev_tau = 0.0f;

// 게인 스케일 (0.0~1.0) — 점진적으로 올려서 테스트
float gain_scale = 0.01f;

// 타이밍
unsigned long prev_loop_us = 0;

// 안전 한계
const float SAFETY_ANGLE_DEG = 80.0f;   // 이 각도 초과 시 자동 정지
const float DELTA_MAX_DEG = 60.0f;       // 힙각 제한

// ============================================================
// θ 가속도계 순수 각도 (노이즈 많지만 드리프트 없음)
// ============================================================
float getThetaAccel() {
  float ax = imu.accData[0];
  float az = imu.accData[2];
  return -90.0f * DEG2RAD - atan2(ax, az) - imu_mount_offset;
}

// ============================================================
// θ 상보필터 업데이트 (매 루프 호출)
//   단기: 자이로 신뢰 (선형가속 내성)
//   장기: 가속도계 신뢰 (드리프트 보정)
// ============================================================
float updateThetaCF(float dt) {
  float gyro_rate = imu.gyroData[1] * DEG2RAD - gyro_bias_y;
  float theta_gyro = theta_cf + gyro_rate * dt;
  
  // 가속도 크기 검증: 모터 구동 시 선형가속 → 가속도계 무시
  float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
  float accel_sq = ax*ax + ay*ay + az*az;
  float ratio = (accel_ref_sq > 0) ? (accel_sq / accel_ref_sq) : 1.0f;
  
  if (ratio > 0.64f && ratio < 1.56f) {
    // 총 가속도 ≈ 1g (정상) → 가속도계 보정 적용
    float theta_accel = getThetaAccel();
    theta_cf = CF_ALPHA * theta_gyro + (1.0f - CF_ALPHA) * theta_accel;
  } else {
    // 선형가속 감지 → 자이로만 사용 (가속도계 오염 차단)
    theta_cf = theta_gyro;
  }
  return theta_cf;
}

// ============================================================
// θ̇: 자이로 (바이어스 보정)
// ============================================================
float getThetaDot() {
  return imu.gyroData[1] * DEG2RAD - gyro_bias_y;
}

// ============================================================
// δ: 힙 관절각 = θ − α (v17 규약: encoder 반전)
// ============================================================
float getDeltaRad() {
  int32_t raw = dxl.getPresentPosition(DXL_ID);
  return (float)(raw - encoder_offset) * (2.0f * PI / 4096.0f);  // 반전 제거
}

// ============================================================
// δ 명령 → 모터 위치 (v17: δ → encoder = -δ)
// ============================================================
void setDeltaRad(float delta_rad) {
  delta_rad = constrain(delta_rad, -DELTA_MAX_DEG * DEG2RAD, DELTA_MAX_DEG * DEG2RAD);
  int32_t raw = encoder_offset + (int32_t)(delta_rad * 4096.0f / (2.0f * PI));  // 반전 제거
  dxl.setGoalPosition(DXL_ID, raw);
}

// === 행렬-벡터 곱 (6×6) * (6) ===
void matVec6(const float A[6][6], const float x[6], float out[6]) {
  for (int i = 0; i < 6; i++) {
    out[i] = 0;
    for (int j = 0; j < 6; j++) {
      out[i] += A[i][j] * x[j];
    }
  }
}

// ============================================================
// 칼만 필터 1스텝: Predict + Correct
// ============================================================
void kalmanStep(float tau, float y_alpha, float y_theta, 
                float y_alphaDot, float y_thetaDot) {
  
  // === 1. Predict: x_pred = Ad * xHat + Bd * tau ===
  float xPred[6];
  matVec6(Ad_mat, xHat, xPred);
  for (int i = 0; i < 6; i++) {
    xPred[i] += Bd_vec[i] * tau;
  }
  
  // === 2. Innovation: e = y - C * x_pred ===
  float innov[4];
  innov[0] = y_alpha    - xPred[1];  // α
  innov[1] = y_theta    - xPred[2];  // θ
  innov[2] = y_alphaDot - xPred[4];  // α̇
  innov[3] = y_thetaDot - xPred[5];  // θ̇
  
  // Innovation 클램핑 (과대 보정 방지)
  const float innovLim[4] = {0.5f, 0.5f, 5.0f, 5.0f};
  for (int i = 0; i < 4; i++) {
    innov[i] = constrain(innov[i], -innovLim[i], innovLim[i]);
  }
  
  // === 3. Correct: xHat = xPred + L * innov ===
  for (int i = 0; i < 6; i++) {
    xHat[i] = xPred[i];
    for (int j = 0; j < 4; j++) {
      xHat[i] += L_kf[i][j] * innov[j];
    }
  }
  
  // 상태 포화 (물리적 한계)
  const float angleLim = PI / 2.0f;
  const float velLim = 30.0f;
  for (int i = 0; i < 3; i++) {
    xHat[i] = constrain(xHat[i], -angleLim, angleLim);
  }
  for (int i = 3; i < 6; i++) {
    xHat[i] = constrain(xHat[i], -velLim, velLim);
  }
}

// ============================================================
// 안전 모터 초기화
// ============================================================
void safeMotorInit() {
  dxl.torqueOff(DXL_ID);
  delay(50);
  dxl.torqueOff(DXL_ID);
  delay(50);
  
  dxl.setOperatingMode(DXL_ID, OP_EXTENDED_POSITION);
  delay(50);
  
  int32_t cur = dxl.getPresentPosition(DXL_ID);
  encoder_offset = cur;
  
  // 위치 제어 모드: 프로파일 속도 0 = 최대속도 (응답 빠르게)
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 0);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, 0);
  dxl.writeControlTableItem(GOAL_POSITION, DXL_ID, cur);
  delay(50);
  dxl.torqueOn(DXL_ID);
}

void setup() {
  Serial.begin(115200);
  
  // IMU 초기화
  imu.begin();
  
  // 모터 초기화
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);
  
  dxl.torqueOff(DXL_ID);
  delay(100);
  dxl.torqueOff(DXL_ID);
  
  delay(300);
  bool found = false;
  for (int i = 0; i < 5; i++) {
    if (dxl.ping(DXL_ID)) { found = true; break; }
    delay(300);
  }
  
  if (!found) {
    Serial.println("!!! Motor not found");
    while(1) delay(1000);
  }
  
  dxl.torqueOff(DXL_ID);
  
  // IMU 안정화 + 자이로 바이어스 캘리브레이션
  Serial.print("IMU stabilizing + gyro bias cal...");
  float gyro_sum = 0;
  int gyro_count = 0;
  for (int i = 0; i < 500; i++) {
    imu.update();
    if (i >= 100) {  // 처음 100회는 안정화 대기
      gyro_sum += imu.gyroData[1] * DEG2RAD;
      gyro_count++;
    }
    delay(4);
  }
  gyro_bias_y = gyro_sum / gyro_count;
  
  // IMU 장착 오프셋 (부팅 시 자세 기준)
  imu_mount_offset = -90.0f * DEG2RAD - atan2(imu.accData[0], imu.accData[2]);
  theta_cf = 0.0f;  // 부팅 시 대략 수직으로 가정
  
  // 가속도 기준값 (정지 상태의 1g 크기²)
  {
    float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
    accel_ref_sq = ax*ax + ay*ay + az*az;
  }
  
  Serial.println(" done");
  Serial.print("  Gyro bias = "); Serial.print(gyro_bias_y * RAD2DEG, 3); Serial.println(" deg/s");
  Serial.print("  Accel ref = "); Serial.println(sqrt(accel_ref_sq), 1);
  
  Serial.println("==========================================");
  Serial.println("  Slackline LQR Balance (v17 convention)");
  Serial.println("  v11_small params, R=0.3, tauMax=4.0");
  Serial.println("==========================================");
  Serial.println("  z : Zero-set (hold upright by hand)");
  Serial.println("  c : Control ON/OFF");
  Serial.println("  + : Gain scale +0.1");
  Serial.println("  - : Gain scale -0.1");
  Serial.println("  g : Show gain scale");
  Serial.println("  t : Print state once");
  Serial.println("  x : Emergency STOP");
  Serial.println("==========================================");
  Serial.print("  Gain scale = "); Serial.println(gain_scale, 1);
  Serial.println();
  
  state = ST_IDLE;
  prev_loop_us = micros();
}

void loop() {
  // IMU 업데이트
  imu.update();
  
  // === 시리얼 명령 ===
  if (Serial.available()) {
    char c = Serial.read();
    
    if (c == 'z' || c == 'Z') {
      // === 0점 세팅: 인코더 + 중력 기준 캘리브 ===
      // 사용자가 수직으로 잡은 상태에서 호출
      safeMotorInit();  // 인코더 offset → 0
      
      // IMU 마운트 오프셋 재캘리브 (이 자세 = 수직 = θ=0)
      imu_mount_offset = -90.0f * DEG2RAD - atan2(imu.accData[0], imu.accData[2]);
      theta_cf = 0.0f;  // 상보필터 초기화 (지금이 수직)
      
      // 가속도 기준값 갱신
      {
        float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
        accel_ref_sq = ax*ax + ay*ay + az*az;
      }
      
      memset(xHat, 0, sizeof(xHat));
      prev_tau = 0;
      ctrl_on = false;
      safety_stopped = false;
      state = ST_READY;
      
      Serial.println(">>> ZERO SET (encoder only)!");
      Serial.print("    theta(CF)=");
      Serial.print(theta_cf * RAD2DEG, 2);
      Serial.print("  delta=");
      Serial.println(getDeltaRad() * RAD2DEG, 2);
      Serial.println("    Press 'c' to start control.");
    }
    else if (c == 'c' || c == 'C') {
      if (state < ST_READY) {
        Serial.println("!!! Press 'z' first to zero-set");
        return;
      }
      ctrl_on = !ctrl_on;
      if (ctrl_on) {
        // 제어 시작: 칼만 상태를 현재 센서값으로 초기화
        float theta_now = theta_cf;
        float delta_now = getDeltaRad();
        float alpha_now = theta_now - delta_now;
        float thetaDot_now = getThetaDot();
        
        xHat[0] = -(C1_PHI * alpha_now + C2_PHI * theta_now) / R_ARC;  // phi 추정
        xHat[1] = alpha_now;
        xHat[2] = theta_now;
        xHat[3] = 0;
        xHat[4] = 0;
        xHat[5] = thetaDot_now;
        
        prev_tau = 0;
        state = ST_RUNNING;
        prev_loop_us = micros();
        
        Serial.print(">>> CONTROL ON (gain=");
        Serial.print(gain_scale, 1);
        Serial.println(")");
      } else {
        // 현재 위치 유지 (토크 끄지 않음)
        state = ST_READY;
        Serial.println(">>> CONTROL OFF (motor holds position)");
      }
    }
    else if (c == '+' || c == '=') {
      gain_scale = constrain(gain_scale + 0.1f, 0.0f, 1.0f);
      Serial.print(">>> Gain scale = ");
      Serial.println(gain_scale, 1);
    }
    else if (c == '-' || c == '_') {
      gain_scale = constrain(gain_scale - 0.1f, 0.0f, 1.0f);
      Serial.print(">>> Gain scale = ");
      Serial.println(gain_scale, 1);
    }
    else if (c == 'g' || c == 'G') {
      Serial.print(">>> Gain scale = ");
      Serial.println(gain_scale, 1);
    }
    else if (c == 't' || c == 'T') {
      float th = theta_cf;
      float dl = getDeltaRad();
      float al = th - dl;
      float td = getThetaDot();
      Serial.print("[STATE] th="); Serial.print(th * RAD2DEG, 2);
      Serial.print(" al="); Serial.print(al * RAD2DEG, 2);
      Serial.print(" dl="); Serial.print(dl * RAD2DEG, 2);
      Serial.print(" thDot="); Serial.print(td * RAD2DEG, 1);
      Serial.print(" | xHat: phi="); Serial.print(xHat[0] * RAD2DEG, 2);
      Serial.print(" a="); Serial.print(xHat[1] * RAD2DEG, 2);
      Serial.print(" th="); Serial.print(xHat[2] * RAD2DEG, 2);
      Serial.println();
    }
    else if (c == 'x' || c == 'X') {
      dxl.torqueOff(DXL_ID);
      ctrl_on = false;
      state = ST_IDLE;
      Serial.println(">>> EMERGENCY STOP!");
    }
    else if (c == 'd' || c == 'D') {
      // IMU 진단 모드 토글 (모터 없이 순수 센서 데이터)
      diag_mode = !diag_mode;
      if (diag_mode) {
        dxl.torqueOff(DXL_ID);
        ctrl_on = false;
        state = ST_IDLE;
        Serial.println(">>> DIAG MODE ON (motor off)");
        Serial.println("    Tilt robot by hand to verify theta");
        Serial.println("    ax  ay  az | gx  gy  gz | thAcc  thCF | aMag");
        Serial.println("    Press 'd' again to exit");
      } else {
        Serial.println(">>> DIAG MODE OFF");
      }
    }
  }
  
  // === IMU 진단 출력 (d 모드, 50Hz) ===
  if (diag_mode) {
    static unsigned long diag_print = 0;
    if (millis() - diag_print >= 20) {
      diag_print = millis();
      float ax = imu.accData[0], ay = imu.accData[1], az = imu.accData[2];
      float gx = imu.gyroData[0], gy = imu.gyroData[1], gz = imu.gyroData[2];
      float amag = sqrt(ax*ax + ay*ay + az*az);
      float th_acc = getThetaAccel() * RAD2DEG;
      float th_cf = theta_cf * RAD2DEG;
      float th_dot = getThetaDot() * RAD2DEG;
      
      Serial.print("[DIAG] ax="); Serial.print(ax, 0);
      Serial.print(" ay="); Serial.print(ay, 0);
      Serial.print(" az="); Serial.print(az, 0);
      Serial.print(" | gy="); Serial.print(gy, 1);
      Serial.print(" | thA="); Serial.print(th_acc, 1);
      Serial.print(" thCF="); Serial.print(th_cf, 1);
      Serial.print(" thD="); Serial.print(th_dot, 1);
      Serial.print(" | mag="); Serial.print(amag, 0);
      Serial.println();
    }
  }
  
  // === 상보필터 업데이트 (항상 실행, 제어 여부 무관) ===
  {
    static unsigned long cf_prev_us = 0;
    unsigned long cf_now = micros();
    float cf_dt = (cf_prev_us == 0) ? DT : (float)(cf_now - cf_prev_us) * 1e-6f;
    cf_prev_us = cf_now;
    if (cf_dt < 0.0001f || cf_dt > 0.05f) cf_dt = DT;
    updateThetaCF(cf_dt);
  }
  
  // === 제어 루프 ===
  if (state == ST_RUNNING && ctrl_on) {
    unsigned long now_us = micros();
    float dt = (float)(now_us - prev_loop_us) * 1e-6f;
    prev_loop_us = now_us;
    
    if (dt < 0.0001f || dt > 0.05f) dt = DT;
    
    // --- 센서 읽기 (v17 규약) ---
    float theta_meas = theta_cf;                  // θ (상보필터, 절대 중력 기준)
    float theta_dot_meas = getThetaDot();         // θ̇
    float delta_meas = getDeltaRad();             // δ = θ − α (힙각)
    float alpha_meas = theta_meas - delta_meas;   // α = θ − δ (하체)
    
    // α̇ 추정 (인코더 미분)
    static float prev_delta = 0;
    float delta_dot = (delta_meas - prev_delta) / dt;
    prev_delta = delta_meas;
    float alpha_dot_meas = theta_dot_meas - delta_dot;
    
    // --- 칼만 필터 ---
    kalmanStep(prev_tau, alpha_meas, theta_meas, alpha_dot_meas, theta_dot_meas);
    
    // --- LQR 제어: τ = −K·x ---
    float tau = 0;
    for (int i = 0; i < 6; i++) {
      tau -= K_lqr[i] * xHat[i];
    }
    
    // 게인 스케일 적용 (점진적 테스트용)
    tau *= gain_scale;
    
    // 토크 제한
    tau = constrain(tau, -TAU_MAX, TAU_MAX);
    
    // --- 안전 체크: 각도 초과 시 자동 정지 ---
    bool th_over = fabs(theta_meas) > SAFETY_ANGLE_DEG * DEG2RAD;
    bool al_over = fabs(alpha_meas) > SAFETY_ANGLE_DEG * DEG2RAD;
    if (th_over || al_over) {
      dxl.torqueOff(DXL_ID);
      ctrl_on = false;
      state = ST_READY;
      safety_stopped = true;
      
      Serial.println();
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Serial.println("  SAFETY STOP!");
      Serial.print("  Reason: ");
      if (th_over) { Serial.print("theta("); Serial.print(theta_meas * RAD2DEG, 1); Serial.print(") "); }
      if (al_over) { Serial.print("alpha("); Serial.print(alpha_meas * RAD2DEG, 1); Serial.print(") "); }
      Serial.print("> "); Serial.print(SAFETY_ANGLE_DEG, 0); Serial.println(" deg");
      Serial.println("  --- Last state ---");
      Serial.print("  theta = "); Serial.print(theta_meas * RAD2DEG, 2); Serial.println(" deg");
      Serial.print("  alpha = "); Serial.print(alpha_meas * RAD2DEG, 2); Serial.println(" deg");
      Serial.print("  delta = "); Serial.print(delta_meas * RAD2DEG, 2); Serial.println(" deg");
      Serial.print("  phi   = "); Serial.print(xHat[0] * RAD2DEG, 2); Serial.println(" deg");
      Serial.print("  tau   = "); Serial.print(tau, 3); Serial.println(" N.m");
      Serial.print("  gain  = "); Serial.println(gain_scale, 2);
      Serial.print("  thDot = "); Serial.print(theta_dot_meas * RAD2DEG, 1); Serial.println(" deg/s");
      Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      Serial.println("Press 'z' to re-zero, 'c' to retry.");
      return;
    }
    
    prev_tau = tau;
    
    // --- 토크 → δ 목표 변환 ---
    // τ = Kp_motor * (δ_goal − δ_current)
    // XM430 내부 PID가 위치 추종 → 토크 발생
    // δ_goal = δ_current + τ / Kp_equiv
    // Kp_equiv ≈ 모터 강성 (튜닝 파라미터)
    const float KP_EQUIV = 8.0f;  // N·m/rad (모터 내부 강성 근사)
    float delta_goal = delta_meas - tau / KP_EQUIV;  // 부호 반전: 모터 물리 방향 보정
    
    setDeltaRad(delta_goal);
    
    // --- 시리얼 출력 (100Hz) ---
    static unsigned long last_print = 0;
    if (millis() - last_print >= 10) {
      last_print = millis();
      
      Serial.print("[LQR] ");
      Serial.print("th=");   Serial.print(theta_meas * RAD2DEG, 1);
      Serial.print(" al=");  Serial.print(alpha_meas * RAD2DEG, 1);
      Serial.print(" dl=");  Serial.print(delta_meas * RAD2DEG, 1);
      Serial.print(" phi="); Serial.print(xHat[0] * RAD2DEG, 1);
      Serial.print(" tau="); Serial.print(tau, 2);
      Serial.print(" g=");   Serial.print(gain_scale, 1);
      Serial.println();
    }
  }
  
  // === IDLE 표시 ===
  if (state == ST_IDLE) {
    static unsigned long lp = 0;
    if (millis() - lp >= 2000) {
      lp = millis();
      Serial.print("[IDLE] th=");
      Serial.print(theta_cf * RAD2DEG, 1);
      Serial.println("  -- press 'z' to zero-set");
    }
  }
  
  // === READY 상태 (안전 정지 후는 스팸 없음) ===
  if (state == ST_READY && !ctrl_on && !safety_stopped) {
    static unsigned long lp2 = 0;
    if (millis() - lp2 >= 1000) {
      lp2 = millis();
      float th = theta_cf;
      float dl = getDeltaRad();
      Serial.print("[READY] th=");
      Serial.print(th * RAD2DEG, 1);
      Serial.print(" dl=");
      Serial.print(dl * RAD2DEG, 1);
      Serial.print("  -- 'c' to start (gain=");
      Serial.print(gain_scale, 1);
      Serial.println(")");
    }
  }
  
  delay(1);  // ~500Hz
}
