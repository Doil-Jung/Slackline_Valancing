/*
 * ============================================
 * Step 4: IMU 기울기 → 모터 추종
 * ============================================
 * 
 * 목적: 보드를 기울이면 모터가 같은 각도로 회전
 * 전원: 12V 필요!
 * 모터: RS-485 4핀 포트에 XM430 연결
 * 
 * 동작 설명:
 *   - Roll(좌우 기울기) → 모터 위치로 매핑
 *   - 보드 수평 = 모터 180도 (중앙)
 *   - 보드 오른쪽 기울임 +90도 = 모터 270도
 *   - 보드 왼쪽 기울임 -90도 = 모터 90도
 *   
 * USER 버튼: 누르면 축 전환 (Roll ↔ Pitch)
 */

#include <Dynamixel2Arduino.h>
#include <IMU.h>

// === 다이나믹셀 설정 ===
#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     57600   // 공장 기본값

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

// === IMU ===
cIMU imu;

// === 설정 ===
#define MOTOR_CENTER   180.0   // 모터 중앙 위치 (도)
#define MOTOR_MIN      0.0     // 모터 최소 위치 (도)
#define MOTOR_MAX      360.0   // 모터 최대 위치 (도) - XM430은 360도 범위
#define IMU_RANGE      90.0    // IMU ±90도를 모터 범위로 매핑
#define DEADZONE       1.0     // 1도 미만 변화는 무시 (떨림 방지)

// === 상태 ===
// OpenCR USER 버튼 핀 (보드에 내장 정의됨)
// BDPIN_PUSH_SW_1 = 34, BDPIN_PUSH_SW_2 = 35
#define USER_BTN_1  34
#define USER_BTN_2  35
bool use_roll = true;    // true: Roll축, false: Pitch축
bool btn_prev = false;
float prev_target = MOTOR_CENTER;
unsigned long prev_print = 0;

// === 이동평균 필터 ===
#define FILTER_N  5
float filter_buf[FILTER_N] = {0};
int filter_i = 0;

float applyFilter(float val) {
  filter_buf[filter_i] = val;
  filter_i = (filter_i + 1) % FILTER_N;
  float sum = 0;
  for (int i = 0; i < FILTER_N; i++) sum += filter_buf[i];
  return sum / FILTER_N;
}

void setup() {
  Serial.begin(115200);
  pinMode(USER_BTN_1, INPUT);  // USER1: 축 전환
  pinMode(USER_BTN_2, INPUT);  // USER2: (예비)
  pinMode(22, OUTPUT);  // LED1
  pinMode(23, OUTPUT);  // LED2
  pinMode(24, OUTPUT);  // LED3
  pinMode(25, OUTPUT);  // LED4
  
  Serial.println("========================================");
  Serial.println("  IMU → Motor Follow - Step 4");
  Serial.println("  보드를 기울이면 모터가 따라갑니다!");
  Serial.println("========================================");
  Serial.println();
  
  // === IMU 초기화 ===
  Serial.print("[IMU] 초기화 중...");
  imu.begin();
  Serial.println(" 완료!");
  
  // === 다이나믹셀 초기화 ===
  Serial.print("[DXL] 초기화 중...");
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);
  delay(500);  // 모터 부팅 대기
  
  // 모터 검색 (최대 5회 재시도)
  bool motor_ok = false;
  for (int retry = 0; retry < 5; retry++) {
    if (dxl.ping(DXL_ID)) {
      motor_ok = true;
      break;
    }
    Serial.print(".");
    delay(500);
  }
  
  if (!motor_ok) {
    Serial.println(" 실패!");
    Serial.println("!!! 모터를 찾을 수 없습니다.");
    Serial.println("    전원과 RS-485 케이블을 확인하세요.");
    while(1) { 
      digitalWrite(22, HIGH); digitalWrite(23, HIGH);
      digitalWrite(24, HIGH); digitalWrite(25, HIGH);
      delay(200);
      digitalWrite(22, LOW); digitalWrite(23, LOW);
      digitalWrite(24, LOW); digitalWrite(25, LOW);
      delay(200);
    }
  }
  Serial.println(" 완료!");
  Serial.print("  모델: "); Serial.println(dxl.getModelNumber(DXL_ID));
  
  // 모터 설정
  dxl.torqueOff(DXL_ID);
  dxl.setOperatingMode(DXL_ID, OP_POSITION);
  
  // Profile Velocity 설정 (부드러운 움직임)
  // 값이 클수록 빠르게 이동, 0 = 최대 속도
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 100);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, 50);
  
  dxl.torqueOn(DXL_ID);
  
  // 초기 위치: 중앙
  dxl.setGoalPosition(DXL_ID, MOTOR_CENTER, UNIT_DEGREE);
  
  Serial.println();
  Serial.println("========================================");
  Serial.println("  준비 완료!");
  Serial.println("  보드를 기울여 보세요 (Roll축)");
  Serial.println("  USER 버튼: Roll ↔ Pitch 전환");
  Serial.println("========================================");
  Serial.println();
  
  // IMU 안정화 대기
  Serial.println("IMU 안정화 대기 중 (2초)...");
  for (int i = 0; i < 200; i++) {
    imu.update();
    delay(10);
  }
  Serial.println("시작!");
  Serial.println();
}

void loop() {
  // === IMU 업데이트 ===
  imu.update();
  
  // === 버튼으로 축 전환 (USER1, 눌리면 LOW) ===
  bool btn_now = !digitalRead(USER_BTN_1);  // active LOW
  if (btn_now && !btn_prev) {
    use_roll = !use_roll;
    Serial.print(">>> 축 전환: ");
    Serial.println(use_roll ? "Roll (좌우)" : "Pitch (앞뒤)");
    
    // 필터 초기화
    for (int i = 0; i < FILTER_N; i++) filter_buf[i] = 0;
  }
  btn_prev = btn_now;
  
  // === IMU 각도 읽기 ===
  float imu_angle = use_roll ? imu.rpy[0] : imu.rpy[1];
  
  // 필터 적용
  float filtered = applyFilter(imu_angle);
  
  // === IMU 각도 → 모터 위치 매핑 ===
  // IMU: -90도 ~ +90도  →  모터: 90도 ~ 270도
  float target = MOTOR_CENTER + filtered;
  
  // 범위 제한
  target = constrain(target, MOTOR_MIN, MOTOR_MAX);
  
  // 데드존 적용 (떨림 방지)
  if (abs(target - prev_target) > DEADZONE) {
    dxl.setGoalPosition(DXL_ID, target, UNIT_DEGREE);
    prev_target = target;
  }
  
  // === LED 표시 ===
  // 기울기 크기에 따라 LED 개수 증가
  float abs_angle = abs(filtered);
  digitalWrite(22, abs_angle > 5.0);    // LED1: 5도 이상
  digitalWrite(23, abs_angle > 15.0);   // LED2: 15도 이상
  digitalWrite(24, abs_angle > 30.0);   // LED3: 30도 이상
  digitalWrite(25, abs_angle > 45.0);   // LED4: 45도 이상
  
  // === 시리얼 출력 (20Hz) ===
  unsigned long now = millis();
  if (now - prev_print >= 50) {
    prev_print = now;
    
    float motor_pos = dxl.getPresentPosition(DXL_ID, UNIT_DEGREE);
    float motor_cur = dxl.getPresentCurrent(DXL_ID, UNIT_MILLI_AMPERE);
    
    Serial.print(use_roll ? "[Roll] " : "[Pitch] ");
    Serial.print("IMU: ");
    Serial.print(filtered, 1);
    Serial.print("°\t→ 목표: ");
    Serial.print(target, 1);
    Serial.print("°\t모터: ");
    Serial.print(motor_pos, 1);
    Serial.print("°\t전류: ");
    Serial.print(motor_cur, 0);
    Serial.println(" mA");
  }
  
  delay(2);  // ~500Hz 루프
}
