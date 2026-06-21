/*
 * ============================================
 * 슬랙라인 밸런싱 로봇 V18 — XM430 + OpenCR 모델
 * ============================================
 * 
 * v18 최적화 결과 적용:
 *   H = 800mm, L1:L2 = 30:70
 *   L1 = 240mm (하체), L2 = 560mm (상체)
 *   총 질량 목표: ~3kg
 * 
 * 모터: ROBOTIS XM430-W210-R
 *   본체: 28.5 x 46.5 x 34.0 mm
 *   무게: 82g
 *   출력축: 한쪽, 반대쪽 아이들혼
 * 
 * 보드: OpenCR 1.0 (105 x 75 mm)
 * 
 * 상/하체: 두께 25mm, 폭 80mm
 * 브라켓: 모터 폭 → 본체 폭 테이퍼
 * 
 * 좌표계: 관절(힙) 중심이 원점
 *   X = 좌우, Y = 전후 (모터 축 방향), Z = 상하
 */

$fn = 40;

// ==========================================
// 🚀 표시 모드
// ==========================================
// 0: 전체 어셈블리 (컬러)
// 1: 상체 프린트용
// 2: 하체 프린트용
// 3: 상체 브라켓 프린트용
// 4: 하체 브라켓 프린트용
export_mode = 0;

// 관절 각도 (시뮬레이션용, export_mode=0 에서만)
joint_angle = 0;  // -90 ~ +90

// ==========================================
// ⚙️ 로봇 파라미터 (v18 최적화)
// ==========================================
upper_length = 560;   // 상체 길이 (mm) — L2
lower_length = 240;   // 하체 길이 (mm) — L1

body_width   = 80;    // 상/하체 폭 (mm)
body_thick   = 25;    // 상/하체 두께 (mm)
wall_thick   = 3;     // 벽 두께 (mm) — 경량화용 중공

// ==========================================
// ⚙️ XM430-W210-R 모터 치수 (공식)
// ==========================================
motor_w = 28.5;       // 폭 (X)
motor_d = 46.5;       // 깊이 (Y) — 축 방향
motor_h = 34.0;       // 높이 (Z)

// 출력축 중심 위치 (모터 바닥에서)
shaft_center_z = 19.5;  // mm (모터 중앙보다 살짝 위)
shaft_center_y = 8.0;   // mm (모터 전면에서 축까지)

// 혼/아이들혼
horn_d = 20.0;        // 혼 직경
horn_h = 3.0;         // 혼 두께
idle_d = 16.0;        // 아이들혼 직경

// 모터 마운트 홀 (M2.5)
mount_hole_d = 2.8;   // M2.5 관통홀
mount_hole_spacing_x = 22.0;  // X방향 홀 간격
mount_hole_spacing_z = 22.0;  // Z방향 홀 간격 (측면)

// ==========================================
// ⚙️ 브라켓 파라미터
// ==========================================
bracket_thick = 3.0;      // 브라켓 두께
bracket_wall  = 3.0;      // 브라켓 벽 두께 
taper_length  = 30.0;     // 테이퍼 구간 길이

// 사이드 프레임 (모터 본체를 감싸는 부분)
side_frame_clearance = 0.5;  // 모터와의 틈새

// ==========================================
// ⚙️ OpenCR 보드 치수
// ==========================================
opencr_w = 75.0;     // 폭
opencr_h = 105.0;    // 높이
opencr_d = 15.0;     // 두께 (부품 포함)

// ==========================================
// ⚙️ 배터리 치수 (3S 11.1V 1800mAh 참고)
// ==========================================
bat_w = 35.0;
bat_h = 105.0;
bat_d = 25.0;

// ==========================================
// ⚙️ 파생 상수
// ==========================================
motor_half_h = motor_h / 2;
motor_half_d = motor_d / 2;
motor_half_w = motor_w / 2;

sf_inner_w = motor_w + side_frame_clearance * 2;
sf_inner_d = motor_d + side_frame_clearance * 2;
sf_outer_w = sf_inner_w + bracket_wall * 2;
sf_outer_d = sf_inner_d + bracket_wall * 2;


// ==========================================
// 👁️ 메인 렌더링
// ==========================================
if (export_mode == 0) {
    // --- 전체 어셈블리 뷰 ---
    
    // 모터 (중앙)
    color("DarkSlateGray") motor_body();
    color("Gold") horn_assembly();
    
    // 상체 (위로)
    color("SteelBlue", 0.85) upper_bracket();
    color("LightSkyBlue", 0.7) upper_body();
    
    // 하체 (아래로, 관절 각도 적용)
    rotate([0, joint_angle, 0]) {
        color("Silver", 0.85) lower_bracket();
        color("DimGray", 0.7) lower_body();
    }
    
    // 전자부품 (상체에 부착)
    electronics_assembly();

} else if (export_mode == 1) {
    upper_body();
} else if (export_mode == 2) {
    lower_body();
} else if (export_mode == 3) {
    upper_bracket();
} else if (export_mode == 4) {
    lower_bracket();
}


// ==========================================
// 🧱 모터 본체
// ==========================================
module motor_body() {
    difference() {
        // 본체
        cube([motor_w, motor_d, motor_h], center=true);
        
        // 케이블 홈 (뒤쪽 하단)
        translate([0, -motor_half_d + 3, -motor_half_h + 5])
            cube([12, 8, 10], center=true);
    }
    
    // 출력축 돌출 (Y+ 방향)
    translate([0, motor_half_d, 0])
        rotate([-90, 0, 0])
            cylinder(d=8, h=4);
    
    // 아이들축 돌출 (Y- 방향)
    translate([0, -motor_half_d - 3, 0])
        rotate([-90, 0, 0])
            cylinder(d=6, h=3);
}

// ==========================================
// 🧱 혼 어셈블리
// ==========================================
module horn_assembly() {
    // 출력 혼 (Y+ 방향)
    translate([0, motor_half_d + 4, 0])
        rotate([-90, 0, 0])
            cylinder(d=horn_d, h=horn_h);
    
    // 아이들 혼 (Y- 방향)
    translate([0, -motor_half_d - 3 - horn_h, 0])
        rotate([-90, 0, 0])
            cylinder(d=idle_d, h=horn_h);
}


// ==========================================
// 🧱 상체 브라켓 (모터 → 상체 연결)
// ==========================================
// 모터 본체 측면을 감싸고, 위로 테이퍼져서 상체 폭으로 넓어짐
module upper_bracket() {
    
    // 사이드 프레임: 모터 측면을 감싸는 U자형
    translate([0, 0, motor_half_h + bracket_thick/2]) {
        difference() {
            // 외곽
            cube([sf_outer_w, sf_outer_d, bracket_thick], center=true);
            // 내부 공간 (모터 통과)
            cube([sf_inner_w, sf_inner_d, bracket_thick + 2], center=true);
        }
    }
    
    // 양쪽 측벽
    for (y_sign = [-1, 1]) {
        translate([0, y_sign * (sf_inner_d/2 + bracket_wall/2), 0])
            cube([sf_outer_w, bracket_wall, motor_h], center=true);
    }
    
    // 테이퍼 구간: 모터 폭(28.5+여유) → 상체 폭(80)
    translate([0, 0, motor_half_h + bracket_thick]) {
        hull() {
            // 아래: 모터 폭
            cube([sf_outer_w, sf_outer_d, 0.1], center=true);
            // 위: 상체 폭
            translate([0, 0, taper_length])
                cube([body_thick, body_width, 0.1], center=true);
        }
    }
    
    // 상체 결합면
    translate([0, 0, motor_half_h + bracket_thick + taper_length]) {
        difference() {
            cube([body_thick, body_width, bracket_thick], center=true);
            // 결합 나사홀 4개 (M3)
            for (x = [-body_thick/4, body_thick/4])
                for (y = [-body_width/3, body_width/3])
                    translate([x, y, 0])
                        cylinder(d=3.2, h=bracket_thick + 2, center=true);
        }
    }
}


// ==========================================
// 🧱 하체 브라켓 (혼 → 하체 연결)
// ==========================================
// 혼에 볼트로 결합, 아래로 테이퍼져서 하체 폭으로 넓어짐
module lower_bracket() {
    
    // 혼 결합판 (양쪽)
    for (y_sign = [-1, 1]) {
        y_pos = y_sign * (motor_half_d + 4 + horn_h/2);
        
        translate([0, y_pos, 0])
            rotate([-90, 0, 0]) {
                difference() {
                    cylinder(d=horn_d + 8, h=bracket_thick, center=true);
                    // 축 관통 홀
                    cylinder(d=10, h=bracket_thick + 2, center=true);
                    // 혼 볼트홀 4개
                    for (a = [0, 90, 180, 270])
                        rotate([0, 0, a])
                            translate([horn_d/2 - 2, 0, 0])
                                cylinder(d=2.6, h=bracket_thick + 2, center=true);
                }
            }
    }
    
    // 양쪽 혼판을 연결하는 측벽 (아래로 내려감)
    for (y_sign = [-1, 1]) {
        y_pos = y_sign * (motor_half_d + 4 + horn_h/2);
        
        // 세로 벽
        translate([0, y_pos, -(horn_d/2 + 4)/2])
            cube([bracket_wall, bracket_thick, horn_d/2 + 4], center=true);
    }
    
    // 하부 연결 바
    bar_z = -(horn_d/2 + 4);
    translate([0, 0, bar_z]) {
        cube([bracket_wall, motor_d + 8 + horn_h, bracket_thick], center=true);
    }
    
    // 테이퍼 구간: 모터 부근 폭 → 하체 폭
    translate([0, 0, bar_z - bracket_thick/2]) {
        hull() {
            cube([bracket_wall + 6, motor_d + 8 + horn_h, 0.1], center=true);
            translate([0, 0, -taper_length])
                cube([body_thick, body_width, 0.1], center=true);
        }
    }
    
    // 하체 결합면
    lbf_z = bar_z - bracket_thick/2 - taper_length;
    translate([0, 0, lbf_z]) {
        difference() {
            cube([body_thick, body_width, bracket_thick], center=true);
            for (x = [-body_thick/4, body_thick/4])
                for (y = [-body_width/3, body_width/3])
                    translate([x, y, 0])
                        cylinder(d=3.2, h=bracket_thick + 2, center=true);
        }
    }
}


// ==========================================
// 🧱 상체 본체
// ==========================================
module upper_body() {
    // 브라켓 결합 위치부터 위로 연장
    start_z = motor_half_h + bracket_thick + taper_length + bracket_thick/2;
    body_len = upper_length - (start_z);
    
    translate([0, 0, start_z + body_len/2]) {
        difference() {
            // 외부
            cube([body_thick, body_width, body_len], center=true);
            
            // 내부 공간 (경량화)
            cube([body_thick - wall_thick*2, 
                  body_width - wall_thick*2, 
                  body_len - wall_thick*2], center=true);
            
            // 측면 경량화 구멍 (양쪽)
            for (y_sign = [-1, 1])
                for (z_step = [60 : 80 : body_len - 40])
                    translate([0, y_sign * body_width/2, -body_len/2 + z_step])
                        cube([body_thick - wall_thick*2, wall_thick + 2, 40], center=true);
            
            // 전면 경량화 구멍
            for (z_step = [60 : 80 : body_len - 40])
                translate([body_thick/2, 0, -body_len/2 + z_step])
                    cube([wall_thick + 2, body_width - wall_thick*4, 40], center=true);
        }
        
        // 내부 보강 리브 (가로)
        for (z_step = [40 : 120 : body_len - 20])
            translate([0, 0, -body_len/2 + z_step])
                cube([body_thick - wall_thick*2, body_width - wall_thick*2, wall_thick], center=true);
    }
}


// ==========================================
// 🧱 하체 본체
// ==========================================
module lower_body() {
    bar_z = -(horn_d/2 + 4);
    start_z = bar_z - bracket_thick/2 - taper_length - bracket_thick/2;
    body_len = lower_length - abs(start_z);
    
    translate([0, 0, start_z - body_len/2]) {
        difference() {
            // 외부
            cube([body_thick, body_width, body_len], center=true);
            
            // 내부 공간 (경량화)
            cube([body_thick - wall_thick*2, 
                  body_width - wall_thick*2, 
                  body_len - wall_thick*2], center=true);
            
            // 측면 경량화 구멍
            for (y_sign = [-1, 1])
                for (z_step = [30 : 50 : body_len - 20])
                    translate([0, y_sign * body_width/2, -body_len/2 + z_step])
                        cube([body_thick - wall_thick*2, wall_thick + 2, 25], center=true);
        }
        
        // 발 끝 (슬랙라인 접촉부)
        translate([0, 0, -body_len/2]) {
            difference() {
                hull() {
                    cube([body_thick, body_width, 0.1], center=true);
                    translate([0, 0, -15])
                        cube([body_thick, 20, 0.1], center=true);
                }
                // V자 홈 (줄에 걸리도록)
                translate([0, 0, -18])
                    rotate([0, 45, 0])
                        cube([10, 30, 10], center=true);
            }
        }
    }
}


// ==========================================
// 🧱 전자부품 배치
// ==========================================
module electronics_assembly() {
    start_z = motor_half_h + bracket_thick + taper_length + bracket_thick/2;
    
    // OpenCR 보드 (상체 전면에 부착)
    color("DodgerBlue", 0.6)
    translate([body_thick/2 + opencr_d/2 + 1, 0, start_z + 30 + opencr_h/2])
        cube([opencr_d, opencr_w, opencr_h], center=true);
    
    // 배터리 (상체 뒷면에 부착)
    color("OrangeRed", 0.6)
    translate([-body_thick/2 - bat_d/2 - 1, 0, start_z + 30 + bat_h/2])
        cube([bat_d, bat_w, bat_h], center=true);
    
    // 라벨
    if (export_mode == 0) {
        color("White")
        translate([body_thick/2 + opencr_d + 5, 0, start_z + 30 + opencr_h/2])
            rotate([0, 90, 0])
                linear_extrude(0.5)
                    text("OpenCR", size=8, halign="center", valign="center");
        
        color("White")
        translate([-body_thick/2 - bat_d - 5, 0, start_z + 30 + bat_h/2])
            rotate([0, -90, 0])
                linear_extrude(0.5)
                    text("Battery", size=8, halign="center", valign="center");
    }
}
