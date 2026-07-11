// =====================================================================
//  슬랙라인 균형 로봇 — 프레임 및 줄 시스템 v2 (2026-07-08)
//  2020 알루미늄 프로파일 프레임 + 사다리꼴 카본 줄 시스템 통합 모델
// ---------------------------------------------------------------------
//  좌표 규약 (단위: mm)
//    X = 깊이 (앞뒤 스윙 방향) = 400
//    Y = 폭 (피벗 축 방향)    = 600 (상단 피벗 이격 = 600 - 2*P)
//    Z = 높이                = 1200
//  회전축은 Y축과 평행. 상단 X레일 중앙을 지나는 단일 Y축을 공유.
//  사다리꼴 줄이 Y축을 중심으로 X-Z 평면에서 스윙(phi)함.
// =====================================================================

// ---------------- 1. 시스템 파라미터 ----------------
P   = 20;       // 2020 프로파일 단면 한 변 [mm]
X   = 400;      // 프레임 깊이 (앞뒤)
Y   = 600;      // 프레임 폭 (피벗 축 방향)
Z   = 1200;     // 프레임 높이
$fn = 32;

// --- 사용자 제어 및 스윙 시뮬레이션 각도 ---
phi = 15;       // 스윙 각도 (Y축 기준 회전각, X-Z 평면 상에서 스윙) [도]

show_removed_ghost = true;  // 제거한 상단 Y레일 2개를 반투명으로 표시
show_pivot_axis    = true;  // 피벗 회전축(Y) 시각화

// --- 카본 파이프 규격 (외경 8파이, 내경 6파이) ---
carbon_od  = 8;
carbon_id  = 6;
diagonal_L = 500;  // 빗변 카본 파이프 길이 [mm]
lower_w    = 100;  // 아랫변(발축 바) 폭 [mm]

// --- 피벗 좌표계 설정 ---
mount_drop = 25;                    // 프로파일 아랫면으로부터 베어링 축을 내리는 거리 [mm]
axis_z     = (Z - P) - mount_drop;  // 피벗 축 높이 (1200 - 20 - 25 = 1155)

// 앞뒤 베어링/엔코더 마운트의 배치 기준 Y 좌표 (프로파일 아랫면 중심선)
y_front_mount = P/2;      // 10
y_rear_mount  = Y - P/2;  // 590

// 마운트 설계 수치 (frame_v1과 정합)
plate_t  = 6;
boss_len = 16;
brg_w    = 7;
// 베어링 중심의 Y 좌표 (마운트 배치 Y 좌표 기준으로 자동 계산)
y_pivot_front = y_front_mount + plate_t + boss_len - brg_w/2;  // 10 + 6 + 16 - 3.5 = 28.5
y_pivot_rear  = y_rear_mount - (plate_t + boss_len - brg_w/2);   // 590 - 18.5 = 571.5

// --- 간섭 회피를 위해 줄 상단 끝점을 안쪽으로 오프셋 ---
clearance = 20;  // 마운트 끝단으로부터 확보할 틈새 여유 공간 [mm]
y_string_front = y_pivot_front + clearance;  // 58.5 mm
y_string_rear  = y_pivot_rear - clearance;   // 541.5 mm

// 윗변 줄 결합 간격 (실제 카본 줄 사이의 Y 간격)
upper_w = y_string_rear - y_string_front;  // 483 mm

// --- 처짐(sag) 자동 계산 (피타고라스 정리) ---
// dy = (윗변폭 - 아랫변폭) / 2
dy = (upper_w - lower_w) / 2;
// sag = sqrt(빗변^2 - dy^2)
sag = sqrt(pow(diagonal_L, 2) - pow(dy, 2));

// ---------------- 2. 프로파일 프레임 모델링 ----------------
xrail_len = X - 2*P;   // X방향 레일 = 360
yrail_len = Y - 2*P;   // Y방향 레일 = 560

module post()  color("silver") cube([P, P, Z]);           // 수직 기둥
module xrail() color("silver") cube([xrail_len, P, P]);    // X방향 가로재
module yrail() color("silver") cube([P, yrail_len, P]);    // Y방향 가로재

// 프레임 조립
module frame_assembly() {
    // 수직 기둥 4개
    translate([0,   0,   0]) post();
    translate([X-P, 0,   0]) post();
    translate([0,   Y-P, 0]) post();
    translate([X-P, Y-P, 0]) post();

    // X방향 가로재
    translate([P, 0,   0  ]) xrail();
    translate([P, Y-P, 0  ]) xrail(); // 하단 뒤쪽 레일
    translate([P, 0,   Z-P]) xrail(); // 상단 X레일 (Z-P=1180에 고정)
    translate([P, Y-P, Z-P]) xrail();

    // Y방향 가로재 (하단만 유지, 상단은 제거)
    translate([0,   P, 0  ]) yrail();
    translate([X-P, P, 0  ]) yrail();

    // 제거한 상단 Y레일 유령 표시 (원래 상단 높이 Z-P에 표시)
    if (show_removed_ghost) {
        color([1,0,0,0.15]) translate([0,   P, Z-P]) yrail();
        color([1,0,0,0.15]) translate([X-P, P, Z-P]) yrail();
    }
}

// ---------------- 3. 베어링 & 엔코더 마운트 (프로파일 하단 체결용 L자형 언더 마운트) ----------------
plate_w  = 52;
plate_h  = 44;
bolt_d   = 5.5;
bolt_dx  = 20;
boss_od  = 30;

module bearing_mount() {
    h_offset = mount_drop; // 축 중심에서 프로파일 아랫면까지의 거리 (25mm)
    difference() {
        union() {
            // 1) 프로파일 아랫면 슬롯 체결용 수평 플랜지 (프로파일 아랫면 Y=-10 ~ 10 영역을 덮음)
            translate([-plate_w/2, -P/2, h_offset - 6])
            cube([plate_w, P, 6]);
            
            // 2) 아래로 늘어뜨린 수직 지지판
            translate([-plate_w/2, 0, -plate_h/2]) 
            cube([plate_w, plate_t, plate_h + h_offset - plate_h/2]);
            
            // 3) 베어링 보스
            translate([0, plate_t, 0]) rotate([-90,0,0]) cylinder(h=boss_len, d=boss_od);
        }
        // 베어링 압입 포켓 (안쪽 끝)
        translate([0, plate_t+boss_len-brg_w, 0]) rotate([-90,0,0]) cylinder(h=brg_w+0.2, d=22); // 608 외경
        // 락 칼라 관통을 위해 샤프트 통로 지름을 18mm로 확대 (락 칼라 14mm와의 안전 마진 2mm 확보)
        translate([0,-1,0]) rotate([-90,0,0]) cylinder(h=plate_t+boss_len+2, d=18);
        
        // 레일 체결용 수직 볼트홀 2개 (수평 플랜지를 관통하여 레일 슬롯에 고정)
        for (dx=[-bolt_dx, bolt_dx])
            translate([dx, 0, h_offset - 8]) cylinder(h=10, d=bolt_d, $fn=16);
    }
}

module encoder_mount() {
    h_offset = mount_drop;
    difference() {
        union() {
            // 1) 프로파일 아랫면 슬롯 체결용 수평 플랜지 (프로파일 아랫면 Y=-10 ~ 10 영역을 덮음)
            translate([-plate_w/2, -P/2, h_offset - 6])
            cube([plate_w, P, 6]);
            
            // 2) 아래로 늘어뜨린 수직 지지판 (바깥쪽 Y=-15까지 살을 연장하여 외부 엔코더 포켓 수용)
            translate([-plate_w/2, -15, -plate_h/2]) 
            cube([plate_w, plate_t + 15, plate_h + h_offset - plate_h/2]);
            
            // 3) 엔코더 보스
            translate([0, plate_t, 0]) rotate([-90,0,0]) cylinder(h=boss_len, d=boss_od);
        }
        // 베어링 압입 포켓 (안쪽 끝인 y = plate_t + boss_len = 22 에서 깊이 7mm 파기)
        translate([0, plate_t+boss_len-brg_w, 0]) rotate([-90,0,0]) cylinder(h=brg_w+0.2, d=22); // 608 외경
        
        // 엔코더 기판 및 락 칼라 안착용 외부 포켓 (바깥쪽 Y = -15.2 에서 Y = 0.1 까지 관통하여 지름 29.2mm 포켓 파기)
        translate([0, -15.2, 0]) rotate([-90,0,0]) cylinder(h=15.3, d=29.2);
        
        // 락 칼라 관통을 위해 샤프트 통로 지름을 18mm로 확대 (포켓 사이를 관통)
        translate([0,-1,0]) rotate([-90,0,0]) cylinder(h=plate_t+boss_len+2, d=18);
        
        // ★ 조립성 확보를 위한 좌우 측면 컷아웃 개구부 (ㄷ자 오픈 윈도우) ★
        // 락 칼라가 매립되는 구간(y = -15.1 ~ 0.1)의 좌우 옆면을 완전히 뚫어내어
        // 조립 시 M3 관통 핀을 끼우고 드라이버나 렌치 툴을 대어 조일 수 있는 공간을 제공합니다.
        translate([-plate_w/2 - 1, -15.1, -15])
        cube([plate_w + 2, 15.3, 30]);
        
        // 레일 체결용 수직 볼트홀 2개
        for (dx=[-bolt_dx, bolt_dx])
            translate([dx, 0, h_offset - 8]) cylinder(h=10, d=bolt_d, $fn=16);
            
        // ★ 엔코더 기판 고정용 M2 나사홀 3개 (BCD 23mm, 120도 간격, 지름 1.8mm 태핑홀)
        // 외부 포켓 바닥면(y=0)에서 안쪽 방향으로 10mm 깊이 뚫기
        for (a=[0, 120, 240]) {
            rotate([0, a, 0]) 
            translate([0, 0, 11.5]) // BCD 반경 11.5mm, y=0 지점
            rotate([-90, 0, 0])
            cylinder(h=10, d=1.8, $fn=16);
        }
    }
}

// ---------------- 4. 사다리꼴 카본 줄 시스템 ----------------

// 두 점을 연결하는 카본 파이프 그리기 모듈 (hull 트릭 사용 - 회전 연산 없음)
module carbon_link(p1, p2, od=carbon_od) {
    color("dimgray")
    hull() {
        translate(p1) sphere(d=od, $fn=16);
        translate(p2) sphere(d=od, $fn=16);
    }
}


// 사다리꼴 줄 시스템 조립 모듈 (로컬 원점 = 상단 피벗 Y축 중심)
module trapezoid_slackline() {
    // 로컬 좌표계 기준 (간섭 회피 오프셋이 적용된 상단 두 시작점)
    p_top_front = [0, y_string_front, 0];
    p_top_rear  = [0, y_string_rear,  0];
    
    // 하단 발축 바 끝단 좌표 (로컬 Z = -sag)
    y_low_front = Y/2 - lower_w/2; // 300 - 50 = 250
    y_low_rear  = Y/2 + lower_w/2; // 300 + 50 = 350
    p_bot_front = [0, y_low_front, -sag];
    p_bot_rear  = [0, y_low_rear,  -sag];
    
    // 1) 앞쪽 빗변 카본 (500mm)
    carbon_link(p_bot_front, p_top_front); // 캡슐 실린더는 방향 상관없이 연결됨
    
    // 2) 뒤쪽 빗변 카본 (500mm)
    carbon_link(p_bot_rear, p_top_rear);
    
    // 3) 아랫변 발축 카본 (100mm)
    carbon_link(p_bot_front, p_bot_rear);
    
    // 4) 3D 프린팅 소켓 & 하우징 조립
    upper_socket([0, y_pivot_front, 0], p_top_front, is_front=true);
    upper_socket([0, y_pivot_rear,  0], p_top_rear,  is_front=false);
    lower_socket(p_bot_front, p_top_front, p_bot_rear);
    lower_socket(p_bot_rear,  p_top_rear,  p_bot_front);
    ankle_housing(p_bot_front, p_bot_rear);
}

// ---------------- 5. 3D 프린팅 소켓 및 하우징 모듈 정의 (hull 기반 - 각도오차 0%) ----------------

// 상단 소켓 (is_front = true 이면 앞쪽 베어링용, false 이면 뒤쪽 엔코더용)
module upper_socket(p_pivot_pos, p_string_pos, is_front=true) {
    dir = is_front ? 1 : -1;
    v_diag = p_string_pos - p_pivot_pos; // 윗변 clearance 20mm 벡터
    
    // 빗변 카본이 뻗어나가는 방향의 단위 벡터 계산
    y_low = is_front ? (Y/2 - lower_w/2) : (Y/2 + lower_w/2);
    p_bot = [0, y_low, -sag];
    v_link = p_bot - p_string_pos;
    u_link = v_link / norm(v_link); // 빗변 단위 벡터
    
    // 베어링 바깥쪽 접촉면 Y 좌표 (정밀 계산: 베어링 중심 기준 폭의 반만큼 바깥쪽)
    y_bearing_outer = p_pivot_pos[1] - dir * (brg_w / 2); // 앞쪽: 25, 뒤쪽: 575
    
    // M3 이탈방지 관통 핀홀 직경 및 위치 (락 칼라의 두께 4mm의 중앙인 y_bearing_outer - dir * 2 지점에 정렬)
    collar_t = 4;
    pin_d = 3.2; 
    y_pin = y_bearing_outer - dir * (collar_t / 2); // 앞쪽: 23, 뒤쪽: 577
    
    // 1) 메인 소켓 바디 및 샤프트 (일체형 crimson 파트)
    color("crimson") {
        difference() {
            union() {
                // 베어링 관통용 8파이 축 돌출부 (Y축과 똑바른 실린더)
                // 베어링 안쪽 끝(brg_w/2 안쪽)에서 시작해 락 칼라를 통과하는 바깥쪽 끝까지 연장
                p_shaft_inner = p_pivot_pos + [0, dir * (brg_w/2), 0]; // 앞쪽: 32, 뒤쪽: 568
                p_shaft_outer = [0, y_bearing_outer - dir * (collar_t + 2), 0]; // 앞쪽: 19, 뒤쪽: 581
                hull() {
                    translate(p_shaft_inner) sphere(d=8, $fn=16);
                    translate(p_shaft_outer) sphere(d=8, $fn=16);
                }
                
                // ★ 베어링 내륜 접촉용 단차 보스 (베어링 안쪽 면에만 접촉하도록 지름 10mm로 턱 형성)
                p_step_inner = p_pivot_pos + [0, dir * (brg_w/2), 0];
                hull() {
                    translate(p_step_inner) sphere(d=10, $fn=16);
                    translate(p_step_inner + [0, dir * 1.5, 0]) sphere(d=10, $fn=16);
                }
                
                // 소켓 바디 (clearance 20mm 구간을 꽉 채워주는 14mm 기둥)
                hull() {
                    translate(p_pivot_pos + [0, dir * (brg_w/2 + 1.5), 0]) sphere(d=14, $fn=16);
                    translate(p_string_pos) sphere(d=14, $fn=16);
                }
                
                // 카본 파이프 결합 컵 (p_string_pos에서 빗변 방향으로 15mm 뻗음)
                p_cup_end = p_string_pos + u_link * 15;
                hull() {
                    translate(p_string_pos) sphere(d=14, $fn=16);
                    translate(p_cup_end) sphere(d=14, $fn=16);
                }
            }
            // 이탈방지 관통 핀홀 (축을 Z축 방향으로 수직 관통)
            translate([0, y_pin, 0])
            cylinder(h=20, d=pin_d, center=true, $fn=16);
        }
    }
    
    // 2) ★ 분리형 이탈 방지 락 칼라 (별도 조립하는 gold 파트) ★
    // 베어링 외측면에 밀착하여 유격 차단
    color("gold") {
        translate([0, y_pin, 0]) // 축 핀홀 위치와 정밀 정렬
        rotate([-90, 0, 0])
        difference() {
            union() {
                // 락 칼라 본체 (두께 4mm, 외경 14mm)
                cylinder(h=collar_t, d=14, center=true, $fn=16); 
                
                // ★ 베어링 내륜 접촉용 돌출 단차 보스 ★
                translate([0, 0, collar_t/2 + 0.75]) 
                cylinder(h=1.5, d=10, center=true, $fn=16);
            }
            
            cylinder(h=collar_t + 4, d=8.1, center=true, $fn=16); // 8mm 축 끼움 구멍
            
            // 락 칼라 고정용 관통 핀홀
            rotate([90, 0, 0])
            cylinder(h=20, d=pin_d, center=true, $fn=16);
        }
    }
}

// 하단 L자 코너 소켓 (빗변-발축 바 코너)
module lower_socket(p_bot_pos, p_string_pos, p_bot_other_pos) {
    v_diag = p_string_pos - p_bot_pos;
    u_diag = v_diag / norm(v_diag); // 빗변 위쪽 방향 단위 벡터
    
    v_bar = p_bot_other_pos - p_bot_pos;
    u_bar = v_bar / norm(v_bar);   // 발축 바 방향 단위 벡터
    
    color("crimson") {
        // 빗변 결합 컵 (빗변 방향으로 15mm 뻗음)
        hull() {
            translate(p_bot_pos) sphere(d=14, $fn=16);
            translate(p_bot_pos + u_diag * 15) sphere(d=14, $fn=16);
        }
        
        // 발축 바 결합 컵 (발축 바 방향으로 15mm 뻗음)
        hull() {
            translate(p_bot_pos) sphere(d=14, $fn=16);
            translate(p_bot_pos + u_bar * 15) sphere(d=14, $fn=16);
        }
        
        // 보강 코너 코어
        translate(p_bot_pos) sphere(d=14, $fn=16);
    }
}

// 발목 베어링 하우징 (발축 바 중앙에 로봇 다리가 관통할 회전 조인트 형성)
module ankle_housing(p_bot_front_pos, p_bot_rear_pos) {
    p_ankle = (p_bot_front_pos + p_bot_rear_pos) / 2; // 발축 바 정확한 정중앙 (300)
    v_bar = p_bot_rear_pos - p_bot_front_pos;
    u_bar = v_bar / norm(v_bar); // Y축 평행 단위 벡터
    
    color("crimson") {
        // 1) 발축 바 관통용 칼라 (Y축을 따라 16mm 길이 형성)
        hull() {
            translate(p_ankle - u_bar * 8) sphere(d=16, $fn=16);
            translate(p_ankle + u_bar * 8) sphere(d=16, $fn=16);
        }
        
        // 2) 로봇 발목 베어링 하우징 (Y축 중심 회전을 위해 베어링 축을 Y와 나란히 배치)
        p_brg_center = p_ankle + [0, 0, -12]; // 아래로 12mm 이격
        
        // 하우징 넥 연결
        hull() {
            translate(p_ankle) sphere(d=16, $fn=16);
            translate(p_brg_center) sphere(d=16, $fn=16);
        }
        
        // 베어링 하우징 메인 디스크 (608 베어링 압입용)
        translate(p_brg_center)
        rotate([0, 90, 0]) // Y축 중심 회전을 위해 X축 기준 회전
        rotate([0, 0, 90])
        difference() {
            cylinder(h=10, d=30, center=true, $fn=32); // 하우징 외경
            cylinder(h=12, d=22.2, center=true, $fn=32); // 608 베어링 압입 포켓 (OD 22)
            cylinder(h=14, d=8.2, center=true, $fn=32);  // 관통 샤프트용 홀
        }
        
        // 하단 발목 베어링 608 조립 시각화
        translate(p_brg_center)
        rotate([0, 90, 0])
        bearing_608();
    }
}

// ---------------- 5.5 상용 부품 모델링 (베어링 & 엔코더) ----------------

// 표준 608 베어링 (OD 22mm, ID 8mm, 두께 7mm)
module bearing_608() {
    // 로컬 Z축 기준 렌더링 (center=true)
    color("darkgray") // 외륜 (고정부)
    difference() {
        cylinder(h=7, d=22, center=true, $fn=32);
        cylinder(h=8, d=19, center=true, $fn=32);
    }
    color("silver") // 내륜 (회전부)
    difference() {
        cylinder(h=7, d=12, center=true, $fn=32);
        cylinder(h=8, d=8, center=true, $fn=32);
    }
    color("dimgray") // 금속 실드/볼 리테이너
    difference() {
        cylinder(h=6.5, d=19, center=true, $fn=32);
        cylinder(h=8, d=12, center=true, $fn=32);
    }
}

// AS5048A 자기식 절대 엔코더 기판 (BCD 23mm, M2 나사홀 3개 포함)
module encoder_AS5048A() {
    difference() {
        union() {
            // 둥근 원형 PCB (지름 28mm, 두께 1.6mm)
            color("darkgreen") 
            cylinder(h=1.6, d=28, center=true, $fn=32);
            
            // 메인 감지 소자 칩 (TSSOP-14 칩 묘사)
            color("black")
            translate([0, 0, 1.2])
            cube([6, 5, 1], center=true);
            
            // SPI 배선 커넥터
            color("white")
            translate([0, 10, 1.5])
            cube([8, 6, 2], center=true);
        }
        // 120도 간격의 M2 볼트 통과용 구멍 3개 (BCD 23mm, 지름 2.2mm)
        for (a=[0, 120, 240]) {
            rotate([0, 0, a])
            translate([0, 11.5, 0])
            cylinder(h=5, d=2.2, center=true, $fn=16);
        }
    }
}

// ---------------- 6. 최종 조립 및 배치 ----------------

// 프레임 렌더링
frame_assembly();

// 마운트 렌더링 (프로파일 아랫면 중심선인 y_front_mount, y_rear_mount 에 정렬 배치)
color("orange") translate([X/2, y_front_mount, axis_z]) bearing_mount();
color("green")  translate([X/2, y_rear_mount,  axis_z]) mirror([0,1,0]) encoder_mount();

// --- 상용 부품 베어링 및 엔코더 조립 배치 ---
// 1) 앞쪽 베어링 마운트 내부 608 베어링
translate([X/2, y_pivot_front, axis_z])
rotate([-90, 0, 0])
bearing_608();

// 2) 뒤쪽 엔코더 마운트 내부 608 베어링
translate([X/2, y_pivot_rear, axis_z])
rotate([-90, 0, 0])
bearing_608();

// 3) 뒤쪽 엔코더 마운트 내측 카운터보어에 장착되는 AS5048A 엔코더 기판
// 마운트 플레이트 외측(y_rear_mount=590)에서 에어갭을 고려한 y_pivot_rear + 14.3 지점 정렬 배치
translate([X/2, y_pivot_rear + 14.3, axis_z])
rotate([-90, 0, 0])
encoder_AS5048A();

// 4) 회전하는 샤프트 끝단 자석 시각화 (엔코더 칩과 대면, 지름 6mm, 두께 2.5mm)
// 베어링 관통 샤프트 돌출부 끝단 단면(y_pivot_rear + 10.75mm 지점)에 본드로 접착 부착
translate([X/2, y_pivot_rear + 10.75, axis_z])
rotate([-90, 0, 0])
color("silver") cylinder(h=2.5, d=6, center=true, $fn=16);

// 피벗 회전축 시각화
if (show_pivot_axis) {
    color([1,0.4,0,0.6])
        translate([X/2, P/2, axis_z]) rotate([-90,0,0]) cylinder(h = Y - P, r = 1.5);
}

// 사다리꼴 줄 시스템 및 소켓/하우징 일괄 스윙 배치
translate([X/2, 0, axis_z])
rotate([0, phi, 0]) // Y축 기준 회전 (X-Z 평면 스윙)
trapezoid_slackline();

// --- 디버그 콘솔 출력 ---
echo("==================================================");
echo(str("프레임 폭 Y = ", Y, " mm, 줄 결합 간격 (윗변) = ", upper_w, " mm"));
echo(str("발축 바 폭 (아랫변) = ", lower_w, " mm"));
echo(str("빗변 카본 파이프 길이 = ", diagonal_L, " mm"));
echo(str("계산된 수직 처짐 (sag) = ", sag, " mm (", sag/10, " cm)"));
echo(str("현재 시뮬레이션 스윙 각도 (phi) = ", phi, "도"));
echo("==================================================");
