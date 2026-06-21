/*
 * ============================================
 * Step 3: 다이나믹셀 XM430 연결 테스트
 * ============================================
 * 
 * 목적: XM430-W210-R 모터 Ping, 기본 위치 제어
 * 전원: 12V 필요! (배터리 또는 직류전원장치)
 * 모터: RS-485 4핀 포트에 XM430 연결
 * 
 * !! 주의 !!
 *   - 12V 전원을 먼저 연결한 후 테스트
 *   - 모터가 움직이므로 손 조심!
 *   - 모터 ID 기본값은 1 (공장 출하)
 *   - 통신 속도 기본값은 57600bps (공장 출하)
 * 
 * 확인 사항:
 *   - 모터 Ping 응답
 *   - 위치 제어 (0도 ↔ 180도 왕복)
 *   - 모터 피드백 (위치, 속도, 전류) 읽기
 */

#include <Dynamixel2Arduino.h>

// OpenCR RS-485 포트 설정
#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84      // OpenCR RS-485 방향 제어 핀

// 모터 설정
#define DXL_ID       1       // 모터 ID (공장 기본값: 1)
#define DXL_BAUD     57600   // 통신 속도 (공장 기본값: 57600)

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);

// Dynamixel 제어 테이블 네임스페이스
using namespace ControlTableItem;

bool motor_found = false;

void setup() {
  Serial.begin(115200);
  
  Serial.println("========================================");
  Serial.println("  Dynamixel XM430 Test - Step 3");
  Serial.println("========================================");
  Serial.println();
  
  // 다이나믹셀 통신 초기화
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);  // XM430은 Protocol 2.0
  
  Serial.println("[1] 모터 검색 중...");
  Serial.println();
  
  // === 모터 스캔 (ID 0~10, Protocol 2.0) ===
  Serial.println("  [Protocol 2.0으로 검색]");
  for (int id = 0; id <= 10; id++) {
    if (dxl.ping(id)) {
      Serial.print("  >>> 모터 발견! ID = ");
      Serial.print(id);
      Serial.print(", 모델 번호 = ");
      Serial.println(dxl.getModelNumber(id));
      motor_found = true;
    }
  }
  
  // Protocol 1.0으로도 시도
  if (!motor_found) {
    Serial.println("  [Protocol 1.0으로 검색]");
    dxl.setPortProtocolVersion(1.0);
    for (int id = 0; id <= 10; id++) {
      if (dxl.ping(id)) {
        Serial.print("  >>> 모터 발견! (Protocol 1.0) ID = ");
        Serial.print(id);
        Serial.print(", 모델 번호 = ");
        Serial.println(dxl.getModelNumber(id));
        motor_found = true;
      }
    }
  }
  
  if (!motor_found) {
    Serial.println("  !!! 기본 속도에서 못 찾음. 다른 속도로 재시도...");
    Serial.println();
    
    // 다른 통신 속도 + 양쪽 프로토콜로 시도
    long bauds[] = {9600, 57600, 115200, 1000000, 2000000, 3000000};
    float protos[] = {2.0, 1.0};
    
    for (int b = 0; b < 6 && !motor_found; b++) {
      for (int p = 0; p < 2 && !motor_found; p++) {
        dxl.begin(bauds[b]);
        dxl.setPortProtocolVersion(protos[p]);
        delay(100);
        
        for (int id = 0; id <= 253 && !motor_found; id++) {
          if (dxl.ping(id)) {
            Serial.print("  >>> 모터 발견! Baud=");
            Serial.print(bauds[b]);
            Serial.print(", Protocol=");
            Serial.print(protos[p], 1);
            Serial.print(", ID=");
            Serial.println(id);
            motor_found = true;
          }
        }
      }
    }
    
    if (!motor_found) {
      Serial.println("  모든 조합에서 모터를 찾지 못했습니다.");
      Serial.println();
      Serial.println("  체크리스트:");
      Serial.println("  1. 12V 전원이 OpenCR에 연결되었나요?");
      Serial.println("  2. 모터가 RS-485 4핀 포트에 연결되었나요?");
      Serial.println("  3. 혹시 TTL(3핀) 모터인가요? → TTL 포트에 연결");
      Serial.println("  4. 케이블 양쪽 다 꽉 눌려 꽂혔나요?");
      while(1) { delay(1000); }  // 정지
    }
  }
  
  Serial.println();
  Serial.println("[2] 모터 설정 중...");
  
  // 토크 OFF (설정 변경을 위해)
  dxl.torqueOff(DXL_ID);
  
  // 위치 제어 모드 설정
  dxl.setOperatingMode(DXL_ID, OP_POSITION);
  
  // 토크 ON
  dxl.torqueOn(DXL_ID);
  
  Serial.println("  모드: 위치 제어");
  Serial.println("  토크: ON");
  Serial.println();
  Serial.println("[3] 위치 제어 테스트 시작!");
  Serial.println("  0도 ↔ 180도 왕복합니다.");
  Serial.println("  모터가 움직이니 손 조심하세요!");
  Serial.println("========================================");
  Serial.println();
  
  delay(1000);
}

void loop() {
  if (!motor_found) return;
  
  // === 위치 1: 0도 ===
  Serial.println(">>> 목표: 0도");
  dxl.setGoalPosition(DXL_ID, 0.0, UNIT_DEGREE);
  
  // 이동 중 피드백 출력
  for (int i = 0; i < 20; i++) {
    delay(100);
    printMotorStatus();
  }
  
  // === 위치 2: 180도 ===
  Serial.println(">>> 목표: 90도");
  dxl.setGoalPosition(DXL_ID, 180.0, UNIT_DEGREE);
  
  for (int i = 0; i < 20; i++) {
    delay(100);
    printMotorStatus();
  }
  
  // === 위치 3: 90도 (중앙) ===
  Serial.println(">>> 목표: 90도 (중앙)");
  dxl.setGoalPosition(DXL_ID, 90.0, UNIT_DEGREE);
  
  for (int i = 0; i < 20; i++) {
    delay(100);
    printMotorStatus();
  }
}

void printMotorStatus() {
  float pos = dxl.getPresentPosition(DXL_ID, UNIT_DEGREE);
  float vel = dxl.getPresentVelocity(DXL_ID, UNIT_RPM);
  float cur = dxl.getPresentCurrent(DXL_ID, UNIT_MILLI_AMPERE);
  
  Serial.print("  위치: ");
  Serial.print(pos, 1);
  Serial.print("°\t속도: ");
  Serial.print(vel, 1);
  Serial.print(" rpm\t전류: ");
  Serial.print(cur, 0);
  Serial.println(" mA");
}
