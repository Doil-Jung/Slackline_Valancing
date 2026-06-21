/*
 * 60도 킥 테스트 (안전 v3)
 * 
 * 동작: 0° → +60° → 0° (1초대기) → 반복
 * 
 * [시리얼 명령]
 *   z : 0도 설정 + 토크ON
 *   s : 시작 (3초 카운트다운)
 *   x : 정지 + 토크OFF
 */

#include <Dynamixel2Arduino.h>

#define DXL_SERIAL   Serial3
#define DXL_DIR_PIN  84
#define DXL_ID       1
#define DXL_BAUD     57600

Dynamixel2Arduino dxl(DXL_SERIAL, DXL_DIR_PIN);
using namespace ControlTableItem;

enum State { ST_IDLE, ST_READY, ST_COUNTDOWN, ST_RUNNING };
State state = ST_IDLE;

int32_t offset_raw = 0;
// 0: 0→60 이동중, 1: 60→0 복귀중, 2: 0에서 대기중
int phase = 0;
unsigned long pause_start = 0;
int countdown = 0;
unsigned long cd_time = 0;

int32_t degToRaw(float deg) {
  return offset_raw + (int32_t)(deg * 4096.0 / 360.0);
}
float rawToDeg(int32_t raw) {
  return (float)(raw - offset_raw) * 360.0 / 4096.0;
}

void safeMotorInit() {
  // 1) 토크 즉시 OFF (반복 전송)
  dxl.torqueOff(DXL_ID);
  delay(50);
  dxl.torqueOff(DXL_ID);
  delay(50);
  
  // 2) 확장 위치 모드 설정 (토크 OFF 상태에서만 가능)
  dxl.setOperatingMode(DXL_ID, OP_EXTENDED_POSITION);
  delay(50);
  
  // 3) 느린 프로파일 (토크 OFF 상태에서 기록)
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 20);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, 10);
  
  // 4) 현재 위치 읽기
  int32_t cur = dxl.getPresentPosition(DXL_ID);
  
  // 5) ★핵심★ Goal Position을 현재 위치로 먼저 기록
  //    토크 OFF 상태에서도 Goal Position 레지스터에 쓸 수 있음
  dxl.writeControlTableItem(GOAL_POSITION, DXL_ID, cur);
  delay(50);
  
  // 6) 이제 토크 ON → Goal == Present이므로 움직이지 않음
  dxl.torqueOn(DXL_ID);
  
  offset_raw = cur;
}

void setup() {
  Serial.begin(115200);
  
  dxl.begin(DXL_BAUD);
  dxl.setPortProtocolVersion(2.0);
  
  // ★ 즉시 토크 OFF (이전 코드의 잔여 동작 중단) ★
  dxl.torqueOff(DXL_ID);
  delay(100);
  dxl.torqueOff(DXL_ID);
  
  // 모터 핑 확인
  delay(300);
  bool found = false;
  for (int i = 0; i < 5; i++) {
    if (dxl.ping(DXL_ID)) { found = true; break; }
    delay(300);
  }
  
  if (!found) {
    Serial.println("!!! 모터 못 찾음. 전원/케이블 확인.");
    while(1) delay(1000);
  }
  
  // 모터 초기화 (토크 OFF 유지)
  dxl.torqueOff(DXL_ID);
  dxl.setOperatingMode(DXL_ID, OP_EXTENDED_POSITION);
  
  Serial.println("==========================================");
  Serial.println("  60도 킥 테스트");
  Serial.println("==========================================");
  Serial.println("  모터 토크 OFF (자유 상태)");
  Serial.println("  상하체를 일직선으로 정렬하세요.");
  Serial.println();
  Serial.println("  z : 0도 설정 + 토크ON");
  Serial.println("  s : 왕복 시작");
  Serial.println("  x : 정지 + 토크OFF");
  Serial.println("==========================================");
  Serial.println();
  
  state = ST_IDLE;
}

void loop() {
  // === 시리얼 명령 ===
  if (Serial.available()) {
    char c = Serial.read();
    
    if (c == 'z' || c == 'Z') {
      // 0도 설정 + 안전 토크ON
      safeMotorInit();
      state = ST_READY;
      Serial.println(">>> 0도 설정 완료! 토크 ON (위치 유지)");
      Serial.println("    's'로 왕복 시작");
    }
    else if (c == 's' || c == 'S') {
      if (state != ST_READY) {
        Serial.println("!!! 먼저 'z'로 0도 설정하세요");
        return;
      }
      state = ST_COUNTDOWN;
      countdown = 3;
      cd_time = millis();
      Serial.println(">>> 3초 후 시작! 손 치우세요!");
      Serial.print("    3");
    }
    else if (c == 'x' || c == 'X') {
      dxl.torqueOff(DXL_ID);
      state = ST_IDLE;
      Serial.println(">>> 토크 OFF. 정지.");
    }
  }
  
  // === 카운트다운 ===
  if (state == ST_COUNTDOWN) {
    if (millis() - cd_time >= 1000) {
      cd_time = millis();
      countdown--;
      if (countdown > 0) {
        Serial.print("...");
        Serial.print(countdown);
      } else {
        Serial.println("...GO!");
        
        // 최대 속도로 전환 (안전하게)
        dxl.torqueOff(DXL_ID);
        int32_t cur = dxl.getPresentPosition(DXL_ID);
        dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, 0);
        dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, 0);
        dxl.writeControlTableItem(GOAL_POSITION, DXL_ID, cur);
        delay(50);
        dxl.torqueOn(DXL_ID);
        delay(50);
        
        phase = 0;
        dxl.setGoalPosition(DXL_ID, degToRaw(60.0));
        state = ST_RUNNING;
      }
    }
    return;
  }
  
  // === 킥 동작: 0→60→0(1초대기) 반복 ===
  if (state == ST_RUNNING) {
    int32_t raw = dxl.getPresentPosition(DXL_ID);
    float deg = rawToDeg(raw);
    
    if (phase == 0) {
      // 0→60 이동중
      if (deg > 57.0) {
        phase = 1;
        dxl.setGoalPosition(DXL_ID, degToRaw(0.0));
      }
    } else if (phase == 1) {
      // 60→0 복귀중
      if (abs(deg) < 3.0) {
        phase = 2;
        pause_start = millis();
      }
    } else if (phase == 2) {
      // 0에서 1초 대기
      if (millis() - pause_start >= 1000) {
        phase = 0;
        dxl.setGoalPosition(DXL_ID, degToRaw(60.0));
      }
    }
    
    // 상태 출력 (5Hz)
    static unsigned long lp = 0;
    if (millis() - lp >= 200) {
      lp = millis();
      const char* ph[] = {"[0->60] ", "[60->0] ", "[대기]  "};
      Serial.print(ph[phase]);
      if (deg >= 0) Serial.print("+");
      Serial.print(deg, 1);
      Serial.println("°");
    }
  }
  
  // === IDLE 상태에서 현재 위치 표시 (1Hz) ===
  if (state == ST_IDLE) {
    static unsigned long lp2 = 0;
    if (millis() - lp2 >= 1000) {
      lp2 = millis();
      Serial.println("[대기] 토크OFF — 정렬 후 'z' 입력");
    }
  }
  
  delay(5);
}
