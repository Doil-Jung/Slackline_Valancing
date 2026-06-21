/*
 * Step 4c: 부호 규약 검증 (encoder 반전 — sim 매칭)
 * 
 * 규약 (시뮬레이션과 동일):
 *   θ = IMU (중력 기준 절대각도, + = 오른쪽)
 *   δ = −encoder (sim의 hip angle과 동일 부호)
 *       δ > 0: 상체가 하체보다 오른쪽 (sim hip > 0)
 *   α = θ − δ (하체 절대각도, + = 오른쪽)
 * 
 * [시리얼 명령]
 *   z : 힙 0점 세팅
 *   + : δ = +20 (상체가 하체보다 오른쪽으로)
 *   - : δ = -20
 *   0 : δ = 0 복귀
 *   c : 수직 유지 ON/OFF (α → 0)
 *   t : 현재 상태 출력
 *   x : 긴급 정지
 */

#include <Dynamixel2Arduino.h>
#include <IMU.h>

#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     57600

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;
cIMU imu;

int32_t encoder_offset = 0;
bool ctrl_on = false;

// ============================================================
// θ: IMU → 상체 절대각도 (중력 기준)
//    0 = 직립, + = 오른쪽, − = 왼쪽
// ============================================================
float getTheta() {
  float ax = imu.accData[0];
  float az = imu.accData[2];
  return -90.0f - atan2(ax, az) * (180.0f / PI);
}

// ============================================================
// δ: 힙 각도 (sim 매칭, encoder 반전)
//    δ = −encoder
//    δ > 0: 상체가 하체보다 오른쪽 (sim hip > 0)
// ============================================================
float getDelta() {
  int32_t raw = dxl.getPresentPosition(DXL_ID);
  return -(float)(raw - encoder_offset) * 360.0f / 4096.0f;  // 반전!
}

// δ 명령 → motor (encoder = −δ)
void setDelta(float deg) {
  int32_t raw = encoder_offset + (int32_t)(-deg * 4096.0f / 360.0f);  // 반전!
  dxl.setGoalPosition(DXL_ID, raw);
}

// ============================================================
// α = θ − δ (하체 절대각도, sim과 동일)
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
  
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 30);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, 10);
  dxl.writeControlTableItem(GOAL_POSITION, DXL_ID, cur);
  delay(50);
  dxl.torqueOn(DXL_ID);
}

void printState(const char* label) {
  float theta = getTheta();
  float delta = getDelta();
  float alpha = theta - delta;
  
  Serial.print(label);
  Serial.print("  th=");
  if (theta >= 0) Serial.print("+");
  Serial.print(theta, 1);
  Serial.print("  d=");
  if (delta >= 0) Serial.print("+");
  Serial.print(delta, 1);
  Serial.print("  a(=th-d)=");
  if (alpha >= 0) Serial.print("+");
  Serial.print(alpha, 1);
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  imu.begin();
  
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
    Serial.println("!!! 모터 못 찾음");
    while(1) delay(1000);
  }
  dxl.torqueOff(DXL_ID);
  
  Serial.print("IMU 안정화...");
  for (int i = 0; i < 500; i++) {
    imu.update();
    delay(4);
  }
  Serial.println(" 완료");
  
  Serial.println("==========================================");
  Serial.println("  부호 규약 검증 (encoder 반전, sim 매칭)");
  Serial.println("==========================================");
  Serial.println("  th = IMU (+ = 오른쪽)");
  Serial.println("  d  = -encoder (sim hip과 동일)");
  Serial.println("  a  = th - d (하체 절대각도)");
  Serial.println("------------------------------------------");
  Serial.println("  z:0점  +:d=+20  -:d=-20  0:복귀");
  Serial.println("  c:수직유지  t:값확인  x:정지");
  Serial.println("==========================================");
}

void loop() {
  imu.update();
  
  if (Serial.available()) {
    char c = Serial.read();
    
    if (c == 'z' || c == 'Z') {
      safeMotorInit();
      ctrl_on = false;
      Serial.println(">>> 0점 세팅!");
      printState("[세팅]");
    }
    else if (c == '+' || c == '=') {
      setDelta(+20.0f);
      Serial.println(">>> d = +20 명령 (상체가 하체보다 오른쪽)");
    }
    else if (c == '-' || c == '_') {
      setDelta(-20.0f);
      Serial.println(">>> d = -20 명령");
    }
    else if (c == '0') {
      setDelta(0);
      Serial.println(">>> d = 0 복귀");
    }
    else if (c == 't' || c == 'T') {
      printState("[확인]");
    }
    else if (c == 'c' || c == 'C') {
      ctrl_on = !ctrl_on;
      if (ctrl_on) {
        Serial.println(">>> 수직 유지 ON (d = th → a = 0)");
      } else {
        setDelta(0);
        Serial.println(">>> 수직 유지 OFF");
      }
    }
    else if (c == 'x' || c == 'X') {
      dxl.torqueOff(DXL_ID);
      ctrl_on = false;
      Serial.println(">>> 긴급 정지!");
    }
  }
  
  // === 수직 유지: α = 0 → δ = θ ===
  if (ctrl_on) {
    float theta = getTheta();
    float goal_delta = theta;  // α = θ − δ = 0 → δ = θ
    goal_delta = constrain(goal_delta, -90.0f, 90.0f);
    setDelta(goal_delta);
    
    static unsigned long lp = 0;
    if (millis() - lp >= 200) {
      lp = millis();
      printState("[제어]");
    }
  }
  
  // 대기 표시
  static unsigned long lp2 = 0;
  if (!ctrl_on && millis() - lp2 >= 2000) {
    lp2 = millis();
    printState("[대기]");
  }
  
  delay(2);
}
