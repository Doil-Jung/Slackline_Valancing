/*
 * ============================================================
 *  FWE 데모 (개루프 스크립트) — 브리핑/시연용
 *  OpenCR 1.0 + Dynamixel XM430-W210-R
 * ============================================================
 *  ★목적: 닫힌루프(IMU 칼만·φ추정)가 아직 불안정하므로, 그것을 우회해
 *    FWE의 동작 시그니처 "접기(fold)→기다리기(wait)→펴기(expand)"를
 *    위치제어로 스크립트 재생한다. 센서 추정 문제와 무관하게 항상 같은
 *    동작이 나온다. 사람이 로봇을 적절히 기울여 놓으면 균형 잡는 것처럼 보인다.
 *
 *  ★정직성: 이건 *개루프 스크립트 시연*이지 닫힌루프 밸런싱이 아니다.
 *    (FWE 운동패턴 자체가 이 연구의 기여라 시연 가치는 충분.)
 *
 *  부호: δ>0 = 앞으로 접기(+전류=앞접기, torque_cal로 검증됨).
 *
 *  [시리얼 명령]
 *    z : 0점(현재 자세를 δ=0으로)        f : 앞으로 한 사이클(접기-대기-펴기)
 *    b : 뒤로 한 사이클                   a : IMU 자동트리거 ON/OFF (기울이면 발동)
 *    l : 리듬 반복 ON/OFF (앞↔뒤 번갈아)  x : 정지(δ=0 유지)
 *    + / - : 접기각 ±5°                   w / s : 대기시간 ±0.1s
 *    [ / ] : 접기/펴기 속도(프로파일) ↓/↑   t : 현재 설정·상태 출력
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
const float DEG2RAD = PI/180.0f, RAD2DEG = 180.0f/PI;

// === 조절 가능한 FWE 파라미터 ===
float fold_deg   = 35.0f;    // 접기 각도(δ_fold) [deg]
int   wait_ms    = 300;      // 대기시간 [ms]  (안정주기 ~한 박자)
int   fold_ms    = 150;      // 접기 슬루 시간 [ms]
int   expand_ms  = 150;      // 펴기 슬루 시간 [ms]
int   profile_vel= 0;        // 0=최대속도. 키울수록 느림(프로파일 속도 unit)
const float FOLD_MAX = 50.0f;

// === IMU 자동트리거 ===
bool  auto_trig = false;
float trig_deg  = 8.0f;      // 이 각도 넘으면 그쪽으로 사이클 발동
unsigned long last_cycle_ms = 0;
const int COOLDOWN_MS = 1200; // 사이클 후 재발동 금지시간

// === 리듬 반복 ===
bool loop_mode = false;
int  loop_dir  = +1;

int32_t enc0 = 0;            // δ=0 인코더 기준
float imu_mount = 0.0f;

// δ[deg] → Goal Position
void gotoDelta(float delta_deg) {
  delta_deg = constrain(delta_deg, -FOLD_MAX-5, FOLD_MAX+5);
  int32_t raw = enc0 + (int32_t)(delta_deg * DEG2RAD * 4096.0f / (2.0f*PI));
  dxl.setGoalPosition(DXL_ID, raw);
}
float getDeltaDeg() {
  int32_t raw = dxl.getPresentPosition(DXL_ID);
  return (float)(raw - enc0) * (2.0f*PI/4096.0f) * RAD2DEG;
}
float getThetaDeg() {       // raw IMU 기울기 (칼만 안 씀)
  float ax=imu.accData[0], az=imu.accData[2];
  return (-90.0f*DEG2RAD - atan2(ax,az) - imu_mount) * RAD2DEG;
}

void setProfile(int v) {
  dxl.writeControlTableItem(PROFILE_VELOCITY, DXL_ID, v);
  dxl.writeControlTableItem(PROFILE_ACCELERATION, DXL_ID, 0);
}

void zeroSet() {
  dxl.torqueOff(DXL_ID); delay(30);
  dxl.setOperatingMode(DXL_ID, OP_EXTENDED_POSITION); delay(30);
  enc0 = dxl.getPresentPosition(DXL_ID);
  setProfile(profile_vel);
  dxl.writeControlTableItem(GOAL_POSITION, DXL_ID, enc0);
  dxl.torqueOn(DXL_ID);
  imu_mount = -90.0f*DEG2RAD - atan2(imu.accData[0], imu.accData[2]);
}

// ★FWE 한 사이클: 접기 → 대기 → 펴기 (블로킹, 데모용)
void foldCycle(int dir) {
  float d = constrain(fold_deg, 5.0f, FOLD_MAX);
  Serial.print(">>> FWE cycle dir="); Serial.print(dir>0?"FWD(+)":"BACK(-)");
  Serial.print(" fold="); Serial.print(d,0); Serial.print("deg wait=");
  Serial.print(wait_ms); Serial.println("ms");
  setProfile(profile_vel);
  gotoDelta(dir * d);     delay(fold_ms);    // 접기(빠르게)
  delay(wait_ms);                            // 기다리기(한 박자)
  gotoDelta(0);           delay(expand_ms);  // 펴기(빠르게)
  last_cycle_ms = millis();
}

void printCfg() {
  Serial.print("[CFG] fold="); Serial.print(fold_deg,0);
  Serial.print("deg wait="); Serial.print(wait_ms);
  Serial.print("ms fold/expand="); Serial.print(fold_ms); Serial.print("/"); Serial.print(expand_ms);
  Serial.print("ms prof="); Serial.print(profile_vel);
  Serial.print(" | auto="); Serial.print(auto_trig?"ON":"off");
  Serial.print(" loop="); Serial.print(loop_mode?"ON":"off");
  Serial.print(" | theta="); Serial.print(getThetaDeg(),1);
  Serial.print(" delta="); Serial.println(getDeltaDeg(),1);
}

void setup() {
  Serial.begin(115200);
  imu.begin();
  dxl.begin(DXL_BAUD); dxl.setPortProtocolVersion(2.0);
  bool found=false;
  for(int i=0;i<5;i++){ if(dxl.ping(DXL_ID)){found=true;break;} delay(300);}
  if(!found){ Serial.println("!!! Motor not found (12V/RS-485)"); while(1) delay(1000); }
  for(int i=0;i<200;i++){ imu.update(); delay(3); }   // IMU 안정화
  zeroSet();
  Serial.println("==========================================");
  Serial.println("  FWE DEMO (open-loop scripted)");
  Serial.println("  z:zero  f:fwd  b:back  a:auto  l:loop  x:stop");
  Serial.println("  +/-:fold  w/s:wait  [/]:speed  t:status");
  Serial.println("==========================================");
  printCfg();
}

void loop() {
  imu.update();

  if (Serial.available()) {
    char c = Serial.read();
    if      (c=='z'||c=='Z'){ zeroSet(); Serial.println(">>> ZERO set (delta=0)"); }
    else if (c=='f'||c=='F'){ foldCycle(+1); }
    else if (c=='b'||c=='B'){ foldCycle(-1); }
    else if (c=='a'||c=='A'){ auto_trig=!auto_trig; Serial.print(">>> auto-trigger "); Serial.println(auto_trig?"ON (기울이면 발동)":"off"); }
    else if (c=='l'||c=='L'){ loop_mode=!loop_mode; Serial.print(">>> loop "); Serial.println(loop_mode?"ON":"off"); }
    else if (c=='x'||c=='X'){ loop_mode=false; auto_trig=false; gotoDelta(0); Serial.println(">>> STOP (hold delta=0)"); }
    else if (c=='+'||c=='='){ fold_deg=constrain(fold_deg+5,5,FOLD_MAX); printCfg(); }
    else if (c=='-'||c=='_'){ fold_deg=constrain(fold_deg-5,5,FOLD_MAX); printCfg(); }
    else if (c=='w'||c=='W'){ wait_ms=constrain(wait_ms+100,0,3000); printCfg(); }
    else if (c=='s'||c=='S'){ wait_ms=constrain(wait_ms-100,0,3000); printCfg(); }
    else if (c=='['){ profile_vel=constrain(profile_vel+20,0,500); setProfile(profile_vel); printCfg(); }
    else if (c==']'){ profile_vel=constrain(profile_vel-20,0,500); setProfile(profile_vel); printCfg(); }
    else if (c=='t'||c=='T'){ printCfg(); }
  }

  // IMU 자동트리거: 기울이면 그쪽으로 한 사이클 (raw 각도만 사용)
  if (auto_trig && (millis()-last_cycle_ms > COOLDOWN_MS)) {
    float th = getThetaDeg();
    if (fabs(th) > trig_deg) foldCycle(th>0 ? +1 : -1);
  }

  // 리듬 반복: 앞↔뒤 번갈아
  if (loop_mode && (millis()-last_cycle_ms > (unsigned)(fold_ms+wait_ms+expand_ms+600))) {
    foldCycle(loop_dir); loop_dir = -loop_dir;
  }

  delay(2);
}
