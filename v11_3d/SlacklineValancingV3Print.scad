/*
  2026 전람회 "줄타기 밸런싱 로봇" - PyBullet 시뮬레이션 및 제작 모델 (V12.6)
  - V12.6 패치: 상/하체 뼈대의 브라켓 결합용 4점 나사 구멍(Hole) 제거 (수동 드릴 타공용)
  - V12.5 유지: export_mode 3(상체 뼈대), 4(하체 뼈대) 단독 출력 지원
  - V12.4 유지: 비대칭 브라켓 실측 반영 (16mm/11mm)
*/

$fn = 30; 

// ==========================================
// 🚀 STL Export 설정
// ==========================================
// 0: 전체 보기 (시뮬레이션용)
// 1: 상체 어셈블리 
// 2: 하체 어셈블리 
// 3: 순수 상체 뼈대 (프린팅용)
// 4: 순수 하체 뼈대 (프린팅용)
export_mode = 0; 

// ==========================================
// ⚙️ 하드웨어 치수 파라미터 
// ==========================================
upper_length = 200; lower_length = 200; col_y_target = 60.0; 

// [모터 및 상/하체 브라켓 치수]
motor_x = 24.7; motor_y = 35.0; motor_z_up = 33.7; motor_z_down = 11.5; motor_z = 45.2;
horn_d = 20.0; horn_h = 3.0;
b_inner_y = 42.0; b_w = 24.0; b_arm_l = 30.0; b_thick = 2.0; 
b_hole_x = 14.0; b_hole_y = 20.0; 
ub_offset = 4.0; 
ub_top_z = motor_z_up + ub_offset;
ub_inner_y = 35.5; ub_w = 24.0; ub_thick = 2.0; 
ub_arm_front = 16.0; ub_arm_back = 11.0; ub_hole_margin = 4.0;
ub_hole_front_z = ub_top_z - ub_arm_front + ub_hole_margin;
ub_hole_back_z  = ub_top_z - ub_arm_back + ub_hole_margin;

// [뼈대 두께]
up_x_fixed = b_w; low_x_fixed = b_w; b_bot_z = -b_arm_l - b_thick;

// [전자부품 실측]
esp_w = 50.0; esp_h = 65.0; esp_d = 12.0;
urt_w = 37.0; urt_h = 56.0; urt_d = 12.0;
bat_w = 30.0; bat_h = 61.0; bat_d = 15.0;
pb_d  = 22.0; pb_h  = 92.0;
mpu_w = 16.0; mpu_h = 21.0; mpu_d = 3.0;

// ==========================================
// 👁️ 메인 병합 로직
// ==========================================
if (export_mode == 1) {
    union() { upper_body_mesh(); motor_body_mesh(); upper_bracket_mesh(); upper_electronics_mesh(); }
} else if (export_mode == 2) {
    union() { lower_body_mesh(); bracket_mesh(); horn_mesh(); lower_electronics_mesh(); }
} else if (export_mode == 3) {
    upper_body_mesh();
} else if (export_mode == 4) {
    lower_body_mesh();
} else if (export_mode == 0) {
    color("LightSkyBlue") upper_body_mesh();
    color("Silver") upper_bracket_mesh();
    color("DarkSlateGray") motor_body_mesh();
    upper_electronics_mesh();
    
    color("DimGray") lower_body_mesh();
    color("Silver") bracket_mesh();
    color("Gold") horn_mesh();
    lower_electronics_mesh();
    
    color("Red") {
        for(x = [-b_hole_x/2, b_hole_x/2]) {
            translate([x, ub_inner_y/2 + ub_thick/2, ub_hole_front_z]) rotate([90, 0, 0]) cylinder(d=2.6, h=6, center=true);
            translate([x, -ub_inner_y/2 - ub_thick/2, ub_hole_back_z]) rotate([90, 0, 0]) cylinder(d=2.6, h=6, center=true);
        }
    }
}

// ==========================================
// 🧱 세부 파트 모델링 
// ==========================================

module upper_body_mesh() {
    translate([0, 0, ub_top_z]) 
    difference() {
        union() {
            translate([0, 0, 2]) cube([up_x_fixed, ub_inner_y + ub_thick*2, 4], center=true);
            hull() {
                translate([0, 0, 4]) cube([up_x_fixed, ub_inner_y + ub_thick*2, 0.1], center=true);
                translate([0, 0, 24]) cube([up_x_fixed, col_y_target, 0.1], center=true);
            }
            h_len = upper_length - ub_top_z - 24;
            translate([0, 0, 24 + h_len/2]) cube([up_x_fixed, col_y_target, h_len], center=true);
        }
        
        // ⭐ [V12.6] 브라켓 결합용 4점 나사 구멍 제거 (주석 처리됨)
        // for(x = [-b_hole_x/2, b_hole_x/2]) for(y = [-b_hole_y/2, b_hole_y/2])
        //     translate([x, y, -10]) cylinder(d=3.2, h=30, center=true);
            
        // 중앙 배선 홀은 유지
        translate([0, 0, 2]) cylinder(d=12, h=10, center=true);
        
        h_len2 = upper_length - ub_top_z - 24 - 4;
        translate([0, 0, 24 + h_len2/2]) cube([up_x_fixed - 8, col_y_target - 8, h_len2], center=true);
            
        for(z = [40 : 40 : upper_length - ub_top_z - 20]) {
            translate([0, -col_y_target/2, z]) cube([up_x_fixed + 10, 4, 3], center=true);
            translate([0,  col_y_target/2, z]) cube([up_x_fixed + 10, 4, 3], center=true);
        }
        translate([0, col_y_target/2, 60 - ub_top_z]) {
            translate([0, 0, 7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
            translate([0, 0, -7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
        }
    }
}

module lower_body_mesh() {
    translate([0, 0, b_bot_z])
    difference() {
        union() {
            translate([0, 0, -2]) cube([low_x_fixed, b_inner_y + b_thick*2, 4], center=true);
            hull() {
                translate([0, 0, -4]) cube([low_x_fixed, b_inner_y + b_thick*2, 0.1], center=true);
                translate([0, 0, -24]) cube([low_x_fixed, col_y_target, 0.1], center=true);
            }
            h_len = lower_length - 24;
            translate([0, 0, -24 - h_len/2]) cube([low_x_fixed, col_y_target, h_len], center=true);
        }
        
        // ⭐ [V12.6] 하체 브라켓 결합용 4점 나사 구멍 제거 (주석 처리됨)
        // for(x = [-b_hole_x/2, b_hole_x/2]) for(y = [-b_hole_y/2, b_hole_y/2])
        //     translate([x, y, -10]) cylinder(d=3.2, h=30, center=true);
            
        h_len2 = lower_length - 24 - 4; 
        translate([0, 0, -24 - h_len2/2]) cube([low_x_fixed - 8, col_y_target - 8, h_len2], center=true);
        
        translate([0, 0, -lower_length - 1.5]) rotate([0, 45, 0]) cube([6, col_y_target + 10, 6], center=true);
        
        translate([0, col_y_target/2, -40]) {
            translate([0, 0, 7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
            translate([0, 0, -7.5]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
        }
    }
}

module upper_bracket_mesh() {
    difference() {
        union() {
            translate([0, 0, ub_top_z - ub_thick/2]) cube([ub_w, ub_inner_y + 2*ub_thick, ub_thick], center=true);
            translate([0, ub_inner_y/2 + ub_thick/2, ub_top_z - ub_arm_front/2]) cube([ub_w, ub_thick, ub_arm_front], center=true);
            translate([0, -ub_inner_y/2 - ub_thick/2, ub_top_z - ub_arm_back/2]) cube([ub_w, ub_thick, ub_arm_back], center=true);
        }
        for(x = [-b_hole_x/2, b_hole_x/2]) for(y = [-b_hole_y/2, b_hole_y/2])
            translate([x, y, ub_top_z - ub_thick/2]) cylinder(d=3.5, h=10, center=true);
        translate([0, 0, ub_top_z - ub_thick/2]) cylinder(d=12, h=10, center=true);
        
        for(x = [-b_hole_x/2, b_hole_x/2]) {
            translate([x, ub_inner_y/2 + ub_thick/2, ub_hole_front_z]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
            translate([x, -ub_inner_y/2 - ub_thick/2, ub_hole_back_z]) rotate([90, 0, 0]) cylinder(d=3.2, h=10, center=true);
        }
    }
}

module motor_body_mesh() {
    z_center = (motor_z_up - motor_z_down)/2;
    translate([0, 0, z_center]) cube([motor_x, motor_y, motor_z], center=true);
}

module upper_electronics_mesh() {
    translate([up_x_fixed/2 + esp_d/2 + 0.1, 0, 160]) cube([esp_d, esp_w, esp_h], center=true);
    translate([up_x_fixed/2 + urt_d/2 + 0.1, 0, 100]) cube([urt_d, urt_w, urt_h], center=true);
    translate([-up_x_fixed/2 - pb_d/2 - 0.1, 0, 160]) cylinder(d=pb_d, h=pb_h, center=true);
    translate([-up_x_fixed/2 - bat_d/2 - 0.1, 0, 100]) cube([bat_d, bat_w, bat_h], center=true);
    translate([0, col_y_target/2 + mpu_d/2 + 0.1, 60]) cube([mpu_w, mpu_d, mpu_h], center=true);
}

module bracket_mesh() {
    difference() {
        translate([0, 0, b_bot_z + b_thick/2]) cube([b_w, b_inner_y + 2*b_thick, b_thick], center=true);
        for(x = [-b_hole_x/2, b_hole_x/2]) for(y = [-b_hole_y/2, b_hole_y/2])
            translate([x, y, b_bot_z + b_thick/2]) cylinder(d=3.5, h=10, center=true);
    }
    translate([0, b_inner_y/2 + b_thick/2, -b_arm_l/2]) cube([b_w, b_thick, b_arm_l], center=true);
    translate([0, -b_inner_y/2 - b_thick/2, -b_arm_l/2]) cube([b_w, b_thick, b_arm_l], center=true);
}

module horn_mesh() {
    translate([0, motor_y/2 + 1.5, 0]) rotate([90, 0, 0]) cylinder(d=horn_d, h=horn_h, center=true);
    translate([0, -motor_y/2 - 1.5, 0]) rotate([90, 0, 0]) cylinder(d=horn_d, h=horn_h, center=true);
}

module lower_electronics_mesh() {
    translate([0, col_y_target/2 + mpu_d/2 + 0.1, b_bot_z - 40]) cube([mpu_w, mpu_d, mpu_h], center=true);
}