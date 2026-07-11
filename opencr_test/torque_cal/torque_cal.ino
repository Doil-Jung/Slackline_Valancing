/*
 * ============================================
 * XM430 전류제어 토크 보정 (Torque Calibration)
 * ============================================
 *
 * 목적: 전류제어(OP_CURRENT) 모드에서 Goal Current를 직접 명령하고
 *       알려진 하중으로 실제 토크상수 Kt와 데드밴드를 실측한다.
 *
 * 배경: 기존 lqr_balance.ino 는 위치제어(OP_EXTENDED_POSITION)에
 *       KP_EQUIV 로 토크를 흉내냈다. 이 스케치는 "진짜 토크 제어"로,
 *       LQR이 설계한 τ를 전류로 직접 변환해 명령한다.
 *
 * XM430-W210-R:
 *   스톨 3.0 N·m @ 12V, 2.3A → 공칭 Kt = 1.304 N·m/A
 *   Goal/Present Current 단위 = 2.69 mA/unit
 *   Current Limit = 1193 unit (= 3.21 A)
 *
 * [측정 절차]
 *   1) 모터 출력축에 길이 r 수평 암 고정, 끝에 알려진 추 m 매달기
 *      → 부하 토크 τ_load = m·g·r
 *   2) 시리얼로 Goal Current(unit)를 0부터 천천히 올린다
 *   3) 암이 추를 들어 수평을 유지하기 시작하는 전류 I* 기록
 *   4) 추를 바꿔 (I*, τ_load) 여러 점 수집 → τ vs I 직선적합
 *      기울기 = 실측 Kt, 절편 = 데드밴드
 *
 * [시리얼 명령]
 *   숫자 입력 (예: 100)  : Goal Current = 그 값(unit)으로 설정 (부호 가능: -100)
 *   t  : 토크[N·m]로 입력 (공칭 Kt로 환산해 전류 명령). 예: t1.0
 *   r  : 0 전류 (정지, 토크 OFF 상태로 자유)
 *   p  : 현재 상태 1회 출력 (위치/속도/전류)
 *   m  : 모니터링 토글 (10Hz로 present current 연속 출력)
 *   x  : 비상정지 (토크 OFF)
 */

#include <Dynamixel2Arduino.h>

#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     1000000   // ★v2가 모터를 1Mbps로 승격시켜 놓았으므로 맞춤 (구 57600)

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

// === XM430 전류↔토크 상수 ===
const float CUR_UNIT   = 2.69f;     // mA per unit (Goal/Present Current)
const float KT_NOMINAL = 1.304f;    // N·m/A 공칭 토크상수
const int   CUR_LIMIT  = 1193;      // unit (= 3.21 A) 하드 상한
const int   CUR_SAFE   = 600;       // unit (≈1.6 A) 보정 중 안전 상한

bool monitoring = false;
int  goal_cur_unit = 0;

// 토크[N·m] → Goal Current[unit] (공칭 Kt 사용)
int torqueToCurrentUnit(float tau) {
  float amp = tau / KT_NOMINAL;        // A
  float unit = amp * 1000.0f / CUR_UNIT;
  return (int)(unit + (unit >= 0 ? 0.5f : -0.5f));
}

// Goal Current[unit] → 토크[N·m] (공칭)
float currentUnitToTorque(int unit) {
  float amp = unit * CUR_UNIT / 1000.0f;
  return amp * KT_NOMINAL;
}

void setGoalCurrentUnit(int unit) {
  unit = constrain(unit, -CUR_SAFE, CUR_SAFE);   // 보정 중 안전 상한
  goal_cur_unit = unit;
  dxl.setGoalCurrent(DXL_ID, unit);
}

void printStatus() {
  float pos = dxl.getPresentPosition(DXL_ID, UNIT_DEGREE);
  float vel = dxl.getPresentVelocity(DXL_ID, UNIT_RPM);
  int   curRaw = dxl.readControlTableItem(PRESENT_CURRENT, DXL_ID);  // signed unit
  float curA = curRaw * CUR_UNIT / 1000.0f;
  float tauEst = curRaw * CUR_UNIT / 1000.0f * KT_NOMINAL;

  Serial.print("  pos=");  Serial.print(pos, 1); Serial.print(" deg");
  Serial.print("  vel=");  Serial.print(vel, 1); Serial.print(" rpm");
  Serial.print("  | goalCur="); Serial.print(goal_cur_unit); Serial.print(" unit");
  Serial.print("  presentCur="); Serial.print(curRaw); Serial.print(" unit (");
  Serial.print(curA, 3); Serial.print(" A)");
  Serial.print("  tau_est="); Serial.print(tauEst, 3); Serial.println(" N.m");
}

void setup() {
  Serial.begin(115200);
  delay(300);

  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);

  bool found = false;
  for (int i = 0; i < 5; i++) { if (dxl.ping(DXL_ID)) { found = true; break; } delay(300); }
  if (!found) {
    Serial.println("!!! Motor not found. 12V 전원/RS-485 확인");
    while (1) delay(1000);
  }

  // 전류제어 모드 설정
  dxl.torqueOff(DXL_ID);
  delay(50);
  dxl.setOperatingMode(DXL_ID, OP_CURRENT);   // ★ 전류(토크) 제어 모드
  delay(50);
  dxl.writeControlTableItem(CURRENT_LIMIT, DXL_ID, CUR_LIMIT);
  dxl.torqueOn(DXL_ID);
  setGoalCurrentUnit(0);   // 시작은 0 토크

  Serial.println("==========================================");
  Serial.println("  XM430 Torque Calibration (CURRENT mode)");
  Serial.println("  Kt_nom=1.304 N.m/A, unit=2.69 mA");
  Serial.print  ("  Safe current cap = "); Serial.print(CUR_SAFE);
  Serial.print  (" unit ("); Serial.print(CUR_SAFE*CUR_UNIT/1000.0f, 2); Serial.println(" A)");
  Serial.println("------------------------------------------");
  Serial.println("  숫자        : Goal Current 설정 (unit, 부호가능)");
  Serial.println("  tX.X        : 토크 X.X N.m 명령 (예: t1.0)");
  Serial.println("  r           : 전류 0 (정지)");
  Serial.println("  p           : 상태 1회 출력");
  Serial.println("  m           : 모니터링 토글");
  Serial.println("  x           : 비상정지");
  Serial.println("==========================================");
}

void loop() {
  // === 시리얼 파싱 ===
  if (Serial.available()) {
    char c = Serial.peek();

    if (c == 'x' || c == 'X') {
      Serial.read();
      dxl.torqueOff(DXL_ID);
      goal_cur_unit = 0;
      Serial.println(">>> EMERGENCY STOP (torque off)");
    }
    else if (c == 'r' || c == 'R') {
      Serial.read();
      if (!dxl.getTorqueEnableStat(DXL_ID)) dxl.torqueOn(DXL_ID);
      setGoalCurrentUnit(0);
      Serial.println(">>> Goal Current = 0");
    }
    else if (c == 'p' || c == 'P') {
      Serial.read();
      printStatus();
    }
    else if (c == 'm' || c == 'M') {
      Serial.read();
      monitoring = !monitoring;
      Serial.print(">>> Monitoring "); Serial.println(monitoring ? "ON" : "OFF");
    }
    else if (c == 't' || c == 'T') {
      Serial.read();  // consume 't'
      float tau = Serial.parseFloat();
      if (!dxl.getTorqueEnableStat(DXL_ID)) dxl.torqueOn(DXL_ID);
      int unit = torqueToCurrentUnit(tau);
      setGoalCurrentUnit(unit);
      Serial.print(">>> tau="); Serial.print(tau, 3);
      Serial.print(" N.m → GoalCur="); Serial.print(goal_cur_unit);
      Serial.print(" unit ("); Serial.print(goal_cur_unit*CUR_UNIT/1000.0f, 3);
      Serial.println(" A)");
    }
    else if (c == '-' || (c >= '0' && c <= '9')) {
      int unit = Serial.parseInt();
      if (!dxl.getTorqueEnableStat(DXL_ID)) dxl.torqueOn(DXL_ID);
      setGoalCurrentUnit(unit);
      Serial.print(">>> GoalCur="); Serial.print(goal_cur_unit);
      Serial.print(" unit (");  Serial.print(goal_cur_unit*CUR_UNIT/1000.0f, 3);
      Serial.print(" A, tau_nom="); Serial.print(currentUnitToTorque(goal_cur_unit), 3);
      Serial.println(" N.m)");
    }
    else {
      Serial.read();  // 기타 문자 버림
    }
  }

  // === 모니터링 출력 (10Hz) ===
  if (monitoring) {
    static unsigned long last = 0;
    if (millis() - last >= 100) {
      last = millis();
      printStatus();
    }
  }
}
