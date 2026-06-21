/*
 * 모터 0도 고정 + 오프셋 보정 — 조립용
 * 
 * USER1 버튼: 현재 위치를 0도로 재설정 (오프셋 보정)
 * 시리얼: 현재 각도 ±도 표시
 */

#include <Dynamixel2Arduino.h>

#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     57600
#define USER_BTN_1   34

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

int32_t offset_raw = 0;  // 오프셋 보정값
bool btn_prev = false;

// raw → signed 도 변환
float rawToDeg(int32_t raw) {
  return (float)(raw - offset_raw) * 360.0 / 4096.0;
}

void setup() {
  Serial.begin(115200);
  pinMode(USER_BTN_1, INPUT);
  
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);
  delay(500);
  
  for (int i = 0; i < 5; i++) {
    if (dxl.ping(DXL_ID)) break;
    delay(500);
  }
  
  dxl.torqueOff(DXL_ID);
  dxl.setOperatingMode(DXL_ID, OP_EXTENDED_POSITION);
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 30);
  dxl.torqueOn(DXL_ID);
  
  // 현재 위치를 0으로 유지
  int32_t cur = dxl.getPresentPosition(DXL_ID);
  offset_raw = cur;  // 현재 위치가 0도가 되도록
  dxl.setGoalPosition(DXL_ID, cur);  // raw값으로 현재 위치 유지
  
  Serial.println("=== 모터 0도 고정 (조립용) ===");
  Serial.println("현재 위치 = 0도로 설정됨");
  Serial.println("USER1 버튼: 현재 위치를 0도로 재설정");
  Serial.println();
}

void loop() {
  // USER1 버튼: 현재 위치를 0도로 재설정
  bool btn_now = !digitalRead(USER_BTN_1);
  if (btn_now && !btn_prev) {
    offset_raw = dxl.getPresentPosition(DXL_ID);
    dxl.setGoalPosition(DXL_ID, offset_raw);
    Serial.println(">>> 현재 위치를 0도로 재설정!");
  }
  btn_prev = btn_now;
  
  // 현재 위치 (signed 도)
  int32_t raw = dxl.getPresentPosition(DXL_ID);
  float deg = rawToDeg(raw);
  
  Serial.print("각도: ");
  if (deg >= 0) Serial.print("+");
  Serial.print(deg, 1);
  Serial.print("°  (raw: ");
  Serial.print(raw);
  Serial.println(")");
  
  delay(500);
}
