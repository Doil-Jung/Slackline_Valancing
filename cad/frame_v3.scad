// =====================================================================
// Slackline balancing robot frame and rope system v3
// 2026-07-08
//
// Design intent
//   - Keep the v2 pivot axis convention:
//       X = swing direction / depth
//       Y = pivot-axis direction / frame width
//       Z = vertical
//   - The slackline trapezoid swings about one Y-parallel top axis.
//   - Phi is measured at the upper frame pivot by a frame-fixed shaft encoder.
//   - The encoder is NOT a structural bearing:
//       each upper pivot mount carries two 608 bearings,
//       a short shaft extension drives the encoder through a printed Oldham
//       solid sleeve coupler.
//   - Lower ankle bearings belong to the robot lower body, not to the rope.
//     The 8 mm lower carbon bar passes through two bearings in the robot body.
//   - Encoder wiring uses service loops:
//       top loop: frame-fixed encoder -> rotating upper carbon tube
//       bottom loop: rotating lower bar -> robot lower body
// =====================================================================

fast_preview = is_undef(print_export) ? true : !print_export;
$fn = fast_preview ? 24 : 64;

// -------------------- Display controls --------------------
phi_demo          = 15;     // rope swing angle about the Y axis [deg]
alpha_demo       = 0;      // robot lower body absolute lean preview [deg]
show_removed_top = true;
show_pivot_axis  = true;
show_cables      = !fast_preview;
show_robot_stub  = !fast_preview;
show_bearing_cutaway = true; // cut windows in printed bearing housings so the pockets stay visible
show_lower_socket_cutaway = false; // use lower_socket_part.scad for socket cutaway inspection

// -------------------- Frame parameters --------------------
profile = 20;              // 2020 aluminum profile side [mm]
frame_x = 400;             // depth / swing direction [mm]
frame_y = 600;             // width / pivot-axis direction [mm]
frame_z = 1200;            // height [mm]

xrail_len = frame_x - 2 * profile;
yrail_len = frame_y - 2 * profile;

axis_x    = frame_x / 2;
mount_drop = 25;           // top pivot axis below the top profile underside [mm]
axis_z    = frame_z - profile - mount_drop;

// Upper bearing centers. These keep the v2 axis position.
top_mount_front_y = profile / 2;
top_mount_rear_y  = frame_y - profile / 2;

plate_t   = 6;
boss_len  = 16;
brg_w     = 7;
brg_od    = 22;
brg_id    = 8;
brg_outer_race_id = 19;
brg_inner_race_od = 12;
shaft_d   = 8;
shaft_clearance_d = 8.4;

top_boss_len = 24;         // upper pivot housing length for two 608 bearings [mm]
top_bearing_offset = top_boss_len/2 - brg_w/2; // bearings press in from both open ends
top_center_bore_len = top_boss_len - 2 * brg_w; // center land bore between the two 608 bearings
top_center_bore_d = brg_outer_race_id; // keeps the printed land touching only the outer races
inner_ring_contact_d = 11.4; // contact faces stay smaller than the 608 inner-ring OD [mm]
inner_ring_contact_len = 1.2;

lock_collar_t = 6;
lock_collar_od = 15;
lock_collar_shaft_clearance = 2.0;
lock_bolt_d = 3.2;         // M3 clearance for the lock collar clamp bolt
lock_slit_w = 1.2;
lock_lug_w = 4.0;          // one top lug width across X
lock_lug_t = 6.0;          // lug thickness along the shaft axis
lock_lug_h = 7.8;
lock_lug_bite = 3.0;       // how far the lugs sink into the collar OD
lock_lug_inset = 1.0;      // moves lugs inward so they blend into the collar
lock_bolt_z = lock_collar_od/2 + lock_bolt_d/2 + 0.2;

bearing_center_to_rail = plate_t + boss_len - brg_w / 2;
front_pivot_y = top_mount_front_y + bearing_center_to_rail;
rear_pivot_y  = top_mount_rear_y  - bearing_center_to_rail;

// -------------------- Rope system parameters --------------------
carbon_od  = 8;
carbon_id  = 6;
diag_len   = 530;          // diagonal carbon tube corner-to-corner length [mm]
lower_socket_outer_d = 18;
lower_socket_insert_len = 27;
lower_bar_exposed_len = 100; // lower bar length visible between the two socket mouths
lower_w    = lower_bar_exposed_len + 2 * lower_socket_insert_len;
lower_pipe_insert_len = 10;
lower_pipe_stop_s = lower_socket_insert_len - lower_pipe_insert_len;
lower_socket_bore_d = 8.2; // measured carbon pipe is about 7.9 mm OD
lower_wire_bore_d = 5;
lower_wire_extra_len = 6;
lower_wire_elbow_s = 7;
lower_wire_elbow_overlap = 2;
lower_wire_elbow_steps = fast_preview ? 6 : 12;
lower_side_wire_from_mouth = 15;
lower_side_wire_s = lower_socket_insert_len - lower_side_wire_from_mouth;
lower_side_wire_bore_d = 5;
lower_side_wire_hole_len = lower_socket_outer_d + 8;

top_socket_inset = 30;     // clearance between bearing center and tube entry [mm]
top_front_y = front_pivot_y + top_socket_inset;
top_rear_y  = rear_pivot_y  - top_socket_inset;
upper_w     = top_rear_y - top_front_y;
top_socket_outer_d = 16;
top_socket_cup_len = 22;
top_pipe_insert_len = 10;
top_pipe_stop_s = top_socket_cup_len - top_pipe_insert_len;
top_socket_bore_d = lower_socket_bore_d;
top_socket_overlap = 5;
top_socket_miter_extend = top_socket_outer_d;
top_socket_miter_overlap = 0.15; // tiny overlap across the miter plane for a solid union
top_encoder_wire_bore_d = 6;
top_encoder_wire_bore_extra = 2;
top_encoder_wire_exit_s = top_pipe_stop_s - 4;
top_encoder_wire_side_hole_d = 5;
top_encoder_wire_side_hole_lift_z = 1;
top_encoder_wire_side_hole_len = top_socket_outer_d + 8;

dy_sag = (upper_w - lower_w) / 2;
sag    = sqrt(pow(diag_len, 2) - pow(dy_sag, 2));

bottom_front_y = frame_y / 2 - lower_w / 2;
bottom_rear_y  = frame_y / 2 + lower_w / 2;

// -------------------- Encoder and coupling parameters --------------------
rear_shaft_out = 24;       // structural shaft beyond rear bearing [mm]
coupler_len = 26;
coupler_od = 14;
coupler_struct_bore_d = 8.1; // 8 mm structural shaft, close fit after sanding
coupler_encoder_bore_d = 6.1; // AN25 6 mm shaft, close fit after sanding
coupler_struct_bore_len = 12;
coupler_encoder_bore_len = 12;
coupler_clamp_bolt_d = 3.2; // M3 clearance, through-bolt and nut
coupler_slit_w = 1.0;
coupler_relief_slit_w = 1.0;
coupler_center_web_t = coupler_relief_slit_w;
coupler_lug_w = 4.0;
coupler_lug_t = 6.0;
coupler_lug_h = 7.8;
coupler_lug_bite = 3.0;
coupler_lug_inset = 1.0;

// Dream Solution AN25-Analog angle sensor, outline dimensions from datasheet.
enc_shaft_d    = 6;
enc_shaft_len  = 14.6;
enc_body_d     = 25;
enc_body_len   = 17;
enc_face_t     = 2.0;
enc_mount_bcd  = 20;
enc_mount_angles = [45, 135, 225, 315];
enc_bolt_d     = 3.2;      // M3 clearance / printed pilot hole
enc_center_clearance_d = 13.0;
enc_connector_w = 14;
enc_connector_len = 13.2;
enc_connector_h = 6.3;

enc_bracket_t = 5;
enc_driver_access_d = 12;       // screwdriver access through the lower face for M5 rail bolts
enc_rail_bolt_x = 20;
enc_driver_access_edge_meat = 3;
enc_bracket_w = max(
    enc_body_d + 22,
    enc_mount_bcd + 22,
    2 * (enc_rail_bolt_x + enc_driver_access_d/2 + enc_driver_access_edge_meat)
);
enc_bracket_h = 2 * mount_drop; // top face touches the underside of the top profile

rear_outer_bearing_y = rear_pivot_y + top_bearing_offset + brg_w / 2;
rear_shaft_tip_y     = rear_outer_bearing_y + rear_shaft_out;
coupler_start_y      = rear_shaft_tip_y;
coupler_end_y        = coupler_start_y + coupler_len;
enc_face_y           = coupler_end_y + enc_shaft_len;

// -------------------- Lower robot bearing stub parameters --------------------
robot_bearing_spacing = 34;
robot_stub_len        = 250;
robot_body_w          = 48;
robot_body_t          = 24;

// -------------------- Colors --------------------
c_profile = [0.74, 0.76, 0.78];
c_ghost   = [1.00, 0.20, 0.15, 0.18];
c_mount   = [0.95, 0.47, 0.12];
c_socket  = [0.76, 0.05, 0.08];
c_carbon  = [0.08, 0.08, 0.08];
c_bearing_outer = [0.18, 0.18, 0.18];
c_bearing_inner = [0.75, 0.75, 0.72];
c_encoder = [0.05, 0.18, 0.35];
c_coupler = [0.98, 0.72, 0.15];
c_robot   = [0.10, 0.35, 0.85, 0.45];
c_cable   = [0.02, 0.02, 0.02];
c_cable2  = [0.05, 0.18, 0.95];

// -------------------- Vector helpers --------------------
function vlen(v) = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
function unit(v) = v / vlen(v);
function add3(a,b) = [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
function rot_y(p,a) = [
    p[0] * cos(a) + p[2] * sin(a),
    p[1],
   -p[0] * sin(a) + p[2] * cos(a)
];

// -------------------- Primitive helpers --------------------
module cyl_y(h, d, center=true) {
    rotate([-90, 0, 0]) cylinder(h=h, d=d, center=center);
}

module cyl_x(h, d, center=true) {
    rotate([0, 90, 0]) cylinder(h=h, d=d, center=center);
}

module rod_between(p1, p2, d=4) {
    hull() {
        translate(p1) sphere(d=d);
        translate(p2) sphere(d=d);
    }
}

module capsule_between(p1, p2, d=4) {
    hull() {
        translate(p1) sphere(d=d);
        translate(p2) sphere(d=d);
    }
}

module cylinder_between(p1, p2, d=4) {
    v = p2 - p1;
    l = vlen(v);
    translate(p1)
        rotate(a=acos(v[2] / l), v=[-v[1], v[0], 0])
            cylinder(h=l, d=d, center=false);
}

module yz_halfspace(p, n, positive=true, size=1000) {
    nn = unit(n);
    a = atan2(nn[2], nn[1]);
    translate(p)
        rotate([a, 0, 0])
            translate([0, (positive ? 1 : -1) * size/2, 0])
                cube([size, size, size], center=true);
}

module mitered_cylinder_between(p1, p2, d, plane_p, plane_n, positive=true, overlap=0) {
    n = unit(plane_n);
    cut_p = positive ? plane_p - n * overlap : plane_p + n * overlap;
    intersection() {
        cylinder_between(p1, p2, d=d);
        yz_halfspace(cut_p, n, positive=positive);
    }
}

function bezier2(p0, p1, p2, t) =
    p0 * pow(1 - t, 2) + p1 * (2 * (1 - t) * t) + p2 * pow(t, 2);

module curved_wire_elbow(p_corner, u1, u2, radius_s, d=5, steps=12) {
    p0 = p_corner + u1 * radius_s;
    pc = p_corner;
    p2 = p_corner + u2 * radius_s;

    for (i = [0 : steps - 1]) {
        hull() {
            translate(bezier2(p0, pc, p2, i / steps))
                sphere(d=d);
            translate(bezier2(p0, pc, p2, (i + 1) / steps))
                sphere(d=d);
        }
    }
}

module tube_path(points, d=2, col=c_cable) {
    color(col)
    for (i = [0 : len(points) - 2])
        rod_between(points[i], points[i+1], d);
}

module arc_cable_y(center, radius, yoff, a0, a1, d=2, col=c_cable, n=24) {
    pts = [
        for (i = [0:n])
        [
            center[0] + radius * cos(a0 + (a1 - a0) * i / n),
            center[1] + yoff,
            center[2] + radius * sin(a0 + (a1 - a0) * i / n)
        ]
    ];
    tube_path(pts, d=d, col=col);
}

// -------------------- Frame profiles --------------------
module post() {
    color(c_profile) cube([profile, profile, frame_z]);
}

module xrail() {
    color(c_profile) cube([xrail_len, profile, profile]);
}

module yrail() {
    color(c_profile) cube([profile, yrail_len, profile]);
}

module frame_assembly() {
    // Vertical posts.
    translate([0, 0, 0]) post();
    translate([frame_x-profile, 0, 0]) post();
    translate([0, frame_y-profile, 0]) post();
    translate([frame_x-profile, frame_y-profile, 0]) post();

    // X direction rails. Top front/rear rails remain.
    translate([profile, 0, 0]) xrail();
    translate([profile, frame_y-profile, 0]) xrail();
    translate([profile, 0, frame_z-profile]) xrail();
    translate([profile, frame_y-profile, frame_z-profile]) xrail();

    // Y direction rails. Only bottom rails remain; upper Y rails are removed.
    translate([0, profile, 0]) yrail();
    translate([frame_x-profile, profile, 0]) yrail();

    if (show_removed_top) {
        color(c_ghost) translate([0, profile, frame_z-profile]) yrail();
        color(c_ghost) translate([frame_x-profile, profile, frame_z-profile]) yrail();
    }
}

// -------------------- Bearings and fixed upper mounts --------------------
module bearing_608_y() {
    // 608 bearing visual model: outer and inner races only.
    color(c_bearing_outer)
    difference() {
        cyl_y(brg_w, brg_od);
        cyl_y(brg_w + 0.2, brg_outer_race_id);
    }

    color(c_bearing_inner)
    difference() {
        cyl_y(brg_w + 0.1, brg_inner_race_od);
        cyl_y(brg_w + 0.3, brg_id);
    }
}

module upper_bearing_pair_y() {
    // Two independent 608 bearings in one printed upper pivot housing.
    // They are inserted from the two open ends; the center land bore is at the
    // outer-race inner diameter so the printed land preserves bearing spacing.
    for (yoff = [-top_bearing_offset, top_bearing_offset])
        translate([0, yoff, 0])
            bearing_608_y();
}

module top_pivot_bearing_mount(cutaway=show_bearing_cutaway) {
    plate_w = 58;
    plate_h = 52;
    flange_y = -bearing_center_to_rail - profile/2;
    web_y    = -bearing_center_to_rail + profile/2 - plate_t/2;

    color(c_mount)
    difference() {
        union() {
            // Flange bolted to the underside of the remaining top X rail.
            translate([-plate_w/2, flange_y, mount_drop - 6])
                cube([plate_w, profile, 6]);

            // Vertical web from the top rail underside down to the bearing boss.
            translate([-plate_w/2, web_y, -plate_h/2])
                cube([plate_w, plate_t, plate_h/2 + mount_drop]);

            // Bearing housing. One printed mount carries two separate 608 bearings.
            cyl_y(top_boss_len, brg_od + 10);

            // Small ribs make the printed part less floppy.
            for (sx = [-1, 1]) {
                hull() {
                    translate([sx * (plate_w/2 - 7), web_y + plate_t/2, mount_drop - 5])
                        sphere(d=6);
                    translate([sx * 14, 0, 0])
                        sphere(d=10);
                }
            }
        }

        // Two 608 bearing pockets, open from the two ends for real assembly.
        for (yoff = [-top_bearing_offset, top_bearing_offset])
            translate([0, yoff, 0])
                cyl_y(brg_w + 0.35, brg_od + 0.35);

        // The opening between the two bearings matches the outer race ID.
        // The remaining annular land contacts the outer races and preserves
        // the spacing between the two pressed-in bearings.
        cyl_y(top_center_bore_len + 0.35, top_center_bore_d);

        // Continuous clearance for the rotating 8 mm shaft.
        cyl_y(top_boss_len + 2, shaft_clearance_d);

        // Preview/render cutaway. This avoids relying on transparent draw order
        // and keeps both bearings visible as separate parts.
        if (cutaway)
            translate([-(brg_od + 18)/2, -top_boss_len/2 - 1, 1])
                cube([brg_od + 18, top_boss_len + 2, brg_od + 12]);

        // M5 rail bolts through the top flange.
        for (xoff = [-20, 20])
            translate([xoff, -bearing_center_to_rail, mount_drop - 9])
                cylinder(h=12, d=5.5);
    }
}

module rear_pivot_encoder_mount() {
    // Rear side is modeled as one clean printed part:
    // a side-open rectangular frame with the 608 bearing housing on one end
    // and the encoder mounting face on the other end.
    y_bearing_face = rear_pivot_y + top_boss_len/2;
    y0 = y_bearing_face - enc_bracket_t;
    y1 = enc_face_y;
    len_y = y1 - y0;
    w = enc_bracket_w;
    h = enc_bracket_h;
    t = enc_bracket_t;

    color(c_mount)
    difference() {
        union() {
            // Clean side-view square frame. The sides stay open for assembly;
            // no old bracket geometry is reused here.
            translate([axis_x - w/2, y0, axis_z + h/2 - t])
                cube([w, len_y, t]);
            translate([axis_x - w/2, y0, axis_z - h/2])
                cube([w, len_y, t]);

            // Bearing-side vertical plate, merged into the bearing housing.
            translate([axis_x - w/2, y0, axis_z - h/2])
                cube([w, t, h]);

            // Encoder face: the rear surface is the encoder reference plane.
            translate([axis_x - w/2, y1 - t, axis_z - h/2])
                cube([w, t, h]);

            // Integrated 608 bearing housing on the rope side.
            translate([axis_x, rear_pivot_y, axis_z])
                cyl_y(top_boss_len, brg_od + 10);
        }

        // Two 608 pockets press in from the open ends; the center bore leaves
        // an annular land that locates only the outer races.
        for (yoff = [-top_bearing_offset, top_bearing_offset])
            translate([axis_x, rear_pivot_y + yoff, axis_z])
                cyl_y(brg_w + 0.35, brg_od + 0.35);

        translate([axis_x, rear_pivot_y, axis_z])
            cyl_y(top_center_bore_len + 0.35, top_center_bore_d);

        translate([axis_x, rear_pivot_y, axis_z])
            cyl_y(top_boss_len + 2, shaft_clearance_d);

        // AN25 shaft clearance and 4-M3 front mounting pattern.
        translate([axis_x, y1 - t/2, axis_z])
            cyl_y(t + 0.4, enc_center_clearance_d);

        for (a = enc_mount_angles) {
            translate([
                axis_x + enc_mount_bcd/2 * cos(a),
                y1 - t/2,
                axis_z + enc_mount_bcd/2 * sin(a)
            ])
            cyl_y(t + 0.4, enc_bolt_d);
        }

        // M5 rail bolts through the top bar, directly under the rear profile.
        for (xoff = [-enc_rail_bolt_x, enc_rail_bolt_x])
            translate([axis_x + xoff, frame_y - profile/2, axis_z + h/2 - t - 0.2])
                cylinder(h=t + 0.4, d=5.5);

        // Matching access holes from the lower face so a driver can reach
        // the M5 rail bolts during assembly.
        for (xoff = [-enc_rail_bolt_x, enc_rail_bolt_x])
            translate([axis_x + xoff, frame_y - profile/2, axis_z - h/2 - 0.2])
                cylinder(h=t + 0.4, d=enc_driver_access_d);

    }
}

module upper_fixed_hardware() {
    // Front structural bearing bracket.
    translate([axis_x, front_pivot_y, axis_z])
        top_pivot_bearing_mount();

    // Rear bearing bracket and encoder support are one printed piece.
    rear_pivot_encoder_mount();

    // Bearings placed into the pockets: two 608 bearings per upper mount.
    translate([axis_x, front_pivot_y, axis_z]) upper_bearing_pair_y();
    translate([axis_x, rear_pivot_y, axis_z])  upper_bearing_pair_y();

    // Fixed AN25 analog encoder and printed coupler.
    translate([axis_x, enc_face_y, axis_z])
        encoder_AN25_analog();

    // Solid coupler between the structural shaft extension and encoder shaft.
    solid_coupler_y(coupler_start_y, coupler_end_y);
}

// -------------------- AN25 analog encoder and printed coupling --------------------
module encoder_AN25_analog() {
    // Local origin = encoder front face center. Shaft points toward -Y.
    color(c_bearing_inner)
        translate([0, -enc_shaft_len/2, 0])
        cyl_y(enc_shaft_len, enc_shaft_d);

    color([0.82, 0.82, 0.78])
    difference() {
        cyl_y(enc_face_t, enc_body_d);
        cyl_y(enc_face_t + 0.2, enc_center_clearance_d);
        for (a = enc_mount_angles) {
            translate([enc_mount_bcd/2 * cos(a), 0, enc_mount_bcd/2 * sin(a)])
                cyl_y(enc_face_t + 0.3, enc_bolt_d);
        }
    }

    color(c_encoder)
        translate([0, enc_body_len/2, 0])
        cyl_y(enc_body_len, enc_body_d);

    // Three-pin connector bump on the fixed rear side.
    color("white")
        translate([0, enc_body_len + enc_connector_len/2, enc_body_d/2 - enc_connector_h/2])
        cube([enc_connector_w, enc_connector_len, enc_connector_h], center=true);
}

module coupler_clamp_lugs_y(yoff) {
    lug_x = coupler_slit_w/2 + coupler_lug_w/2 - coupler_lug_inset;
    lug_z = coupler_od/2 - coupler_lug_bite + coupler_lug_h/2;

    for (sx = [-1, 1])
        translate([sx * lug_x, yoff, lug_z])
            cube([coupler_lug_w, coupler_lug_t, coupler_lug_h], center=true);
}

module coupler_clamp_cuts_y(bolt_y, slit_y, slit_len, bore_d) {
    lug_z = coupler_od/2 - coupler_lug_bite + coupler_lug_h/2;
    bolt_z = coupler_od/2 + coupler_clamp_bolt_d/2 + 0.2;
    slit_bottom_z = 0;
    slit_top_z = lug_z + coupler_lug_h/2 + 0.3;
    slit_z = (slit_bottom_z + slit_top_z) / 2;
    slit_h = slit_top_z - slit_bottom_z;

    translate([0, bolt_y, bolt_z])
        cyl_x(2 * coupler_lug_w + coupler_slit_w + 2, coupler_clamp_bolt_d);

    translate([0, slit_y, slit_z])
        cube([coupler_slit_w, slit_len, slit_h], center=true);
}

module coupler_relief_slit_y(yoff, bore_d) {
    slit_bottom_z = 0;
    slit_top_z = coupler_od/2 + 0.3;
    slit_z = (slit_bottom_z + slit_top_z) / 2;
    slit_h = slit_top_z - slit_bottom_z;

    translate([0, yoff, slit_z])
        cube([coupler_od + 2, coupler_relief_slit_w, slit_h], center=true);
}

module coupler_center_top_clearance_y(center_len) {
    // The lower half of the center web is enough as an insertion stop.
    // Remove the upper half so no semicircular boss remains between clamps.
    translate([-(coupler_od + 2)/2, -(center_len + 0.2)/2, 0])
        cube([coupler_od + 2, center_len + 0.2, coupler_od + coupler_lug_h + 2]);
}

module solid_coupler_local(total_len=coupler_len, cutaway=false) {
    struct_len = min(coupler_struct_bore_len, total_len/2 - 0.4);
    encoder_len = min(coupler_encoder_bore_len, total_len/2 - 0.4);
    center_stop_len = total_len - struct_len - encoder_len;
    struct_slit_len = struct_len + center_stop_len - coupler_center_web_t;
    encoder_slit_len = encoder_len + center_stop_len - coupler_center_web_t;
    struct_slit_y = -total_len/2 + struct_len/2;
    encoder_slit_y = total_len/2 - encoder_len/2;
    struct_relief_y = -center_stop_len/2 - coupler_relief_slit_w/2;
    encoder_relief_y = center_stop_len/2 + coupler_relief_slit_w/2;
    struct_clamp_y = (-total_len/2 + struct_relief_y - coupler_relief_slit_w/2) / 2;
    encoder_clamp_y = (total_len/2 + encoder_relief_y + coupler_relief_slit_w/2) / 2;

    color(c_coupler)
    difference() {
        union() {
            cyl_y(total_len, coupler_od);
            coupler_clamp_lugs_y(struct_clamp_y);
            coupler_clamp_lugs_y(encoder_clamp_y);
        }

        // Blind bore for the 8 mm structural shaft side.
        translate([0, -total_len/2 + struct_len/2 - 0.05, 0])
            cyl_y(struct_len + 0.2, coupler_struct_bore_d);

        // Blind bore for the 6 mm encoder shaft side. The uncut center web
        // works as an insertion stop and keeps the coupler torsionally rigid.
        translate([0, total_len/2 - encoder_len/2 + 0.05, 0])
            cyl_y(encoder_len + 0.2, coupler_encoder_bore_d);

        // Through-bolt clamp cuts. No printed thread is required.
        coupler_clamp_cuts_y(struct_clamp_y, struct_slit_y, struct_slit_len, coupler_struct_bore_d);
        coupler_clamp_cuts_y(encoder_clamp_y, encoder_slit_y, encoder_slit_len, coupler_encoder_bore_d);

        // X-direction relief slits just outside the center stopper shoulder.
        // These decouple each clamp end from the rigid center stop so tightening
        // the through-bolts can actually close the bore.
        coupler_relief_slit_y(struct_relief_y, coupler_struct_bore_d);
        coupler_relief_slit_y(encoder_relief_y, coupler_encoder_bore_d);

        // Leave only the lower half of the center stopper.
        coupler_center_top_clearance_y(center_stop_len);

        if (cutaway)
            translate([-coupler_od/2 - 0.5, -total_len/2 - 0.5, 0])
                cube([coupler_od + 1, total_len + 1, coupler_od/2 + 1]);
    }
}

module solid_coupler_y(y0, y1) {
    len = y1 - y0;
    yc  = (y0 + y1) / 2;

    translate([axis_x, yc, axis_z])
        solid_coupler_local(len);
}

module solid_coupler_part(cutaway=false) {
    solid_coupler_local(coupler_len, cutaway=cutaway);
}

// -------------------- Rotating carbon trapezoid and sockets --------------------
module carbon_tube_between(p1, p2, od=carbon_od) {
    color(c_carbon)
    rod_between(p1, p2, od);
}

module lock_collar_at_y(face_y, dir=1) {
    // Shaft lock collar. Only the small yellow nose touches the 608 inner ring;
    // the larger clamp body sits outside the bearing face.
    body_y = face_y + dir * (inner_ring_contact_len + lock_collar_t/2);
    total_t = inner_ring_contact_len + lock_collar_t;
    total_y = face_y + dir * total_t/2;
    lug_x = lock_slit_w/2 + lock_lug_w/2 - lock_lug_inset;
    lug_z = lock_collar_od/2 - lock_lug_bite + lock_lug_h/2;
    slit_bottom_z = (shaft_d + 0.25)/2 - 0.6;
    slit_top_z = lug_z + lock_lug_h/2 + 0.3;
    slit_z = (slit_bottom_z + slit_top_z) / 2;
    slit_h = slit_top_z - slit_bottom_z;
    color([1.0, 0.74, 0.10])
    difference() {
        union() {
            translate([0, face_y + dir * inner_ring_contact_len/2, 0])
                cyl_y(inner_ring_contact_len, inner_ring_contact_d);

            translate([0, body_y, 0])
                cyl_y(lock_collar_t, lock_collar_od);

            // Omega-style clamp lugs placed beside the top slit.
            // The bolt head and nut sit on the lug outer faces, not on a
            // shaved section of the round collar body.
            for (sx = [-1, 1])
                translate([sx * lug_x, body_y, lug_z])
                    cube([lock_lug_w, lock_lug_t, lock_lug_h], center=true);
        }

        translate([0, face_y + dir * (inner_ring_contact_len + lock_collar_t)/2, 0])
            cyl_y(inner_ring_contact_len + lock_collar_t + 0.4, shaft_d + 0.25);

        // Clamp bolt hole. It sits above the shaft bore and pulls the split
        // collar closed instead of passing through the shaft.
        translate([0, body_y, lock_bolt_z])
            cyl_x(2 * lock_lug_w + lock_slit_w + 2, lock_bolt_d);

        // Split cut from the OD into the shaft bore so tightening the bolt
        // actually clamps the collar onto the printed hub shaft.
        translate([0, total_y, slit_z])
            cube([lock_slit_w, total_t + 0.8, slit_h], center=true);
    }
}

module socket_inner_hub_face_at_y(face_y, inside_dir=1) {
    // This is part of the printed rope socket, not a loose spacer.
    // Its diameter is small enough to touch only the bearing inner ring.
    color(c_socket)
    translate([0, face_y + inside_dir * inner_ring_contact_len/2, 0])
        cyl_y(inner_ring_contact_len, inner_ring_contact_d);
}

module upper_socket(is_front=true, include_lock_collar=true) {
    pivot_y = is_front ? front_pivot_y : rear_pivot_y;
    tube_y  = is_front ? top_front_y : top_rear_y;
    dir     = is_front ? -1 : 1;  // direction to the outside of the frame
    inside_dir = -dir;
    outer_y = pivot_y + dir * (top_boss_len/2 + 5);
    inner_y = pivot_y - dir * (top_boss_len/2 + 6);
    outside_face_y = pivot_y + dir * top_boss_len/2;
    front_shaft_tip_y = outside_face_y + dir * (inner_ring_contact_len + lock_collar_t + lock_collar_shaft_clearance);
    shaft_tip_y = is_front ? front_shaft_tip_y : rear_shaft_tip_y;
    inside_face_y = pivot_y + inside_dir * top_boss_len/2;
    socket_start_y = inner_y - inside_dir * top_socket_overlap;

    color(c_socket)
    difference() {
        union() {
            // 8 mm shaft passing through both bearings in the mount.
            // Keep this as a flat-ended cylinder, not a sphere-ended hull.
            cylinder_between([0, inner_y, 0], [0, shaft_tip_y, 0], d=shaft_d);

            // Integrated socket hub face: the inside-side stop for the bearing
            // inner ring, assembled as one piece with the rope socket.
            socket_inner_hub_face_at_y(inside_face_y, inside_dir);

            y_bottom = is_front ? bottom_front_y : bottom_rear_y;
            p_top = [0, tube_y, 0];
            p_bot = [0, y_bottom, -sag];
            u = unit(p_bot - p_top);
            bridge_dir = [0, -inside_dir, 0];
            miter_n = bridge_dir - u;

            // Printed bridge from bearing axis to carbon tube entry.
            mitered_cylinder_between(
                [0, socket_start_y, 0],
                p_top - bridge_dir * top_socket_miter_extend,
                d=top_socket_outer_d,
                plane_p=p_top,
                plane_n=miter_n,
                positive=true,
                overlap=top_socket_miter_overlap
            );

            // Cylindrical carbon tube cup, directed down along the diagonal tube.
            mitered_cylinder_between(
                p_top - u * top_socket_miter_extend,
                p_top + u * top_socket_cup_len,
                d=top_socket_outer_d,
                plane_p=p_top,
                plane_n=miter_n,
                positive=false,
                overlap=top_socket_miter_overlap
            );
        }

        // Blind upper carbon-pipe socket. The pipe enters only
        // top_pipe_insert_len from the mouth and stops on a flat shoulder.
        y_bottom = is_front ? bottom_front_y : bottom_rear_y;
        p_top = [0, tube_y, 0];
        p_bot = [0, y_bottom, -sag];
        u = unit(p_bot - p_top);
        cylinder_between(
            p_top + u * top_pipe_stop_s,
            p_top + u * (top_socket_cup_len + 4),
            d=top_socket_bore_d
        );

        // Encoder-side wire passage through the diagonal tube stop. The
        // carbon tube bore already opens from the mouth to the stop; this
        // axial bore carries the wire past the shoulder and the side hole
        // gives it a short external exit.
        if (!is_front) {
            cylinder_between(
                p_top - u * top_encoder_wire_bore_extra,
                p_top + u * (top_pipe_stop_s + top_encoder_wire_bore_extra),
                d=top_encoder_wire_bore_d
            );

            translate(p_top + u * top_encoder_wire_exit_s + [0, 0, top_encoder_wire_side_hole_lift_z])
                rotate([0, 90, 0])
                    cylinder(
                        h=top_encoder_wire_side_hole_len,
                        d=top_encoder_wire_side_hole_d,
                        center=true
                    );
        }
    }

    // Retainer collar outside the bearing. This slides onto the shaft after
    // insertion and its small nose touches only the bearing inner ring.
    if (include_lock_collar)
        lock_collar_at_y(outside_face_y, dir);
}

module upper_socket_part(side="front") {
    // Standalone upper socket, using the same geometry as the full assembly.
    if (side == "rear")
        translate([0, -rear_pivot_y, 0])
            upper_socket(false, include_lock_collar=false);
    else
        translate([0, -front_pivot_y, 0])
            upper_socket(true, include_lock_collar=false);
}

module lock_collar_part(dir=1) {
    // Standalone lock collar, with the inner-ring contact face at Y=0.
    lock_collar_at_y(0, dir);
}

module encoder_bracket_part() {
    // Standalone rear encoder/bearing bracket. Origin is the rear bearing axis.
    translate([-axis_x, -rear_pivot_y, -axis_z])
        rear_pivot_encoder_mount();
}

module bearing_mount_part(cutaway=false) {
    // Standalone front bearing mount. Origin is the bearing axis.
    top_pivot_bearing_mount(cutaway=cutaway);
}

module lower_corner_socket(p_corner, p_top, p_other, cutaway=show_lower_socket_cutaway) {
    u_diag = unit(p_top - p_corner);
    u_bar  = unit(p_other - p_corner);

    color(c_socket)
    difference() {
        union() {
            // Diagonal carbon tube insertion cup.
            cylinder_between(
                p_corner,
                p_corner + u_diag * lower_socket_insert_len,
                d=lower_socket_outer_d
            );

            // Lower carbon bar insertion cup.
            cylinder_between(
                p_corner,
                p_corner + u_bar * lower_socket_insert_len,
                d=lower_socket_outer_d
            );

            // Slightly fuller corner node where the two cups meet.
            translate(p_corner) sphere(d=lower_socket_outer_d + 2);
        }

        // Blind carbon-pipe sockets. The pipe enters only lower_pipe_insert_len
        // from each mouth, then stops against a long internal shoulder.
        cylinder_between(
            p_corner + u_diag * lower_pipe_stop_s,
            p_corner + u_diag * (lower_socket_insert_len + 4),
            d=lower_socket_bore_d
        );
        cylinder_between(
            p_corner + u_bar * lower_pipe_stop_s,
            p_corner + u_bar * (lower_socket_insert_len + 4),
            d=lower_socket_bore_d
        );

        // Coaxial wire passages run from each mouth toward the corner, then
        // hand off to a smooth elbow so a pushed wire does not catch on a
        // sharp crossing.
        cylinder_between(
            p_corner + u_diag * (lower_wire_elbow_s - lower_wire_elbow_overlap),
            p_corner + u_diag * (lower_socket_insert_len + lower_wire_extra_len),
            d=lower_wire_bore_d
        );
        cylinder_between(
            p_corner + u_bar * (lower_wire_elbow_s - lower_wire_elbow_overlap),
            p_corner + u_bar * (lower_socket_insert_len + lower_wire_extra_len),
            d=lower_wire_bore_d
        );

        curved_wire_elbow(
            p_corner,
            u_diag,
            u_bar,
            radius_s=lower_wire_elbow_s,
            d=lower_wire_bore_d,
            steps=lower_wire_elbow_steps
        );

        // Side wire exits through the cylinder wall, about 15 mm in from each
        // pipe mouth. They intersect the coaxial wire passages above.
        translate(p_corner + u_diag * lower_side_wire_s)
            rotate([0, 90, 0])
                cylinder(h=lower_side_wire_hole_len, d=lower_side_wire_bore_d, center=true);
        translate(p_corner + u_bar * lower_side_wire_s)
            rotate([0, 90, 0])
                cylinder(h=lower_side_wire_hole_len, d=lower_side_wire_bore_d, center=true);

        // Inspection cutaway: remove the viewer-side half of the socket.
        if (cutaway) {
            translate([
                p_corner[0],
                p_corner[1] - lower_socket_insert_len - 4,
                p_corner[2] - lower_socket_insert_len - 8
            ])
            cube([
                lower_socket_outer_d,
                2 * lower_socket_insert_len + 8,
                2 * lower_socket_insert_len + 16
            ]);
        }
    }
}

module lower_socket_part(side="front", cutaway=false) {
    // Standalone lower socket. Dimensions are derived from the same rope
    // geometry used in the full assembly so parameter edits stay linked.
    p_corner = [0, 0, 0];
    p_top = side == "rear" ? [0, dy_sag, sag] : [0, -dy_sag, sag];
    p_other = side == "rear" ? [0, -lower_w, 0] : [0, lower_w, 0];

    lower_corner_socket(p_corner, p_top, p_other, cutaway=cutaway);
}

module robot_lower_body_stub(p_ankle) {
    // Local origin is the lower bar / ankle axis center.
    // The housing rotates relative to the lower bar, so it is shown at alpha_demo.
    rel_angle = alpha_demo - phi_demo;

    translate(p_ankle)
    rotate([0, rel_angle, 0])
    union() {
        color(c_robot)
        difference() {
            // Bearing carrier around the lower carbon bar.
            translate([0, 0, -8])
                cube([robot_body_w, robot_bearing_spacing + 26, robot_body_t], center=true);

            for (yoff = [-robot_bearing_spacing/2, robot_bearing_spacing/2]) {
                translate([0, yoff, -8]) cyl_y(brg_w + 2, brg_od + 0.6);
                translate([0, yoff, -8]) cyl_y(brg_w + 4, shaft_clearance_d);
            }
        }

        // The two lower-body bearings. These are in the robot, not in the rope.
        for (yoff = [-robot_bearing_spacing/2, robot_bearing_spacing/2])
            translate([0, yoff, -8]) bearing_608_y();

        // A simple lower-body placeholder so interferences are visible.
        color([0.10, 0.35, 0.85, 0.32])
        hull() {
            translate([0, 0, -18]) cube([36, 38, 20], center=true);
            translate([0, 0, -robot_stub_len]) cube([32, 30, 26], center=true);
        }

        // Mechanical rotation stop bosses. They keep cable wind-up bounded.
        color([0.02, 0.12, 0.35])
        for (sx = [-1, 1])
            translate([sx * 27, 0, -34])
                sphere(d=10);
    }
}

module lower_service_loop(p_ankle) {
    // A clock-spring-like loop around the ankle axis.
    // This is intentionally outside the bearing centerline to reduce cable torque.
    if (show_cables) {
        color(c_cable)
        union() {
            // Cable exiting the lower bar near the rear side.
            tube_path([
                p_ankle + [0, 28, 0],
                p_ankle + [8, 35, -10],
                p_ankle + [22, 35, -18]
            ], d=2, col=c_cable);

            arc_cable_y(p_ankle, radius=30, yoff=35, a0=210, a1=-120, d=2, col=c_cable, n=36);

            // Cable entering the robot body after the loop.
            tube_path([
                p_ankle + [-15, 35, -26],
                p_ankle + [-8, 18, -45],
                p_ankle + [0, 0, -72]
            ], d=2, col=c_cable);
        }
    }
}

module slackline_trapezoid() {
    p_top_front = [0, top_front_y, 0];
    p_top_rear  = [0, top_rear_y, 0];
    p_bot_front = [0, bottom_front_y, -sag];
    p_bot_rear  = [0, bottom_rear_y, -sag];
    p_ankle     = [0, frame_y/2, -sag];
    p_bot_front_diag_stop = p_bot_front + unit(p_top_front - p_bot_front) * lower_pipe_stop_s;
    p_bot_rear_diag_stop  = p_bot_rear  + unit(p_top_rear  - p_bot_rear)  * lower_pipe_stop_s;
    p_bot_front_bar_stop  = p_bot_front + unit(p_bot_rear  - p_bot_front) * lower_pipe_stop_s;
    p_bot_rear_bar_stop   = p_bot_rear  + unit(p_bot_front - p_bot_rear)  * lower_pipe_stop_s;
    p_top_front_diag_stop = p_top_front + unit(p_bot_front - p_top_front) * top_pipe_stop_s;
    p_top_rear_diag_stop  = p_top_rear  + unit(p_bot_rear  - p_top_rear)  * top_pipe_stop_s;

    // Top sockets and shaft stubs.
    upper_socket(true);
    upper_socket(false);

    // Carbon trapezoid: no physical upper bar.
    carbon_tube_between(p_top_front_diag_stop, p_bot_front_diag_stop);
    carbon_tube_between(p_top_rear_diag_stop,  p_bot_rear_diag_stop);
    carbon_tube_between(p_bot_front_bar_stop, p_bot_rear_bar_stop);

    // Lower corner sockets.
    lower_corner_socket(p_bot_front, p_top_front, p_bot_rear);
    lower_corner_socket(p_bot_rear,  p_top_rear,  p_bot_front);

    // Cable drawn on/inside the rear diagonal and lower bar.
    if (show_cables) {
        tube_path([
            p_top_rear + [1.8, 0, -4],
            p_bot_rear + [1.8, 0, 4],
            p_ankle    + [1.8, 24, 3]
        ], d=1.8, col=c_cable2);
    }

    // Lower robot bearing carrier placeholder and lower service loop.
    if (show_robot_stub)
        robot_lower_body_stub(p_ankle);

    lower_service_loop(p_ankle);
}

// -------------------- Top cable service loop --------------------
module top_encoder_service_loop() {
    if (show_cables) {
        // The moving entry point is a point slightly down the rear diagonal tube.
        local_entry = [0, top_rear_y, -34];
        entry = add3([axis_x, 0, axis_z], rot_y(local_entry, phi_demo));

        fixed_connector = [axis_x, enc_face_y + enc_body_len + 8, axis_z + 13];
        fixed_relief    = [axis_x - 38, frame_y + 28, axis_z + 30];
        loop_low        = [axis_x - 54, frame_y + 20, axis_z - 12];
        rotating_relief = entry + [-12, 0, 8];

        tube_path([
            fixed_connector,
            fixed_relief,
            loop_low,
            rotating_relief,
            entry
        ], d=2, col=c_cable);

        // Small strain-relief clamps, one fixed and one rotating.
        color([0.02, 0.02, 0.02])
            translate(fixed_relief) cube([14, 7, 7], center=true);

        color([0.02, 0.02, 0.02])
            translate(rotating_relief) cube([12, 6, 6], center=true);
    }
}

// -------------------- Render selection --------------------
module full_assembly() {
    frame_assembly();
    upper_fixed_hardware();

    if (show_pivot_axis) {
        color([1.0, 0.45, 0.0, 0.55])
        translate([axis_x, profile/2, axis_z])
            cyl_y(frame_y - profile, 2.2, center=false);
    }

    translate([axis_x, 0, axis_z])
        rotate([0, phi_demo, 0])
            slackline_trapezoid();

    top_encoder_service_loop();
}

if (is_undef(part_mode) || part_mode == "assembly") {
    full_assembly();

    // -------------------- Console summary --------------------
    echo("============================================================");
    echo("frame_v3.scad: upper frame encoder + solid shaft coupling");
    echo(str("Frame X/Y/Z [mm] = ", frame_x, " / ", frame_y, " / ", frame_z));
    echo(str("Pivot axis: X=", axis_x, "  Z=", axis_z, "  Y from ", front_pivot_y, " to ", rear_pivot_y));
    echo(str("Upper tube entry width [mm] = ", upper_w));
    echo(str("Lower bar exposed length [mm] = ", lower_bar_exposed_len));
    echo(str("Lower socket corner-to-corner width [mm] = ", lower_w));
    echo(str("Diagonal carbon tube [mm] = ", diag_len));
    echo(str("Computed sag R preview [mm] = ", sag, " (", sag/10, " cm)"));
    echo(str("Preview phi [deg] = ", phi_demo, "  alpha [deg] = ", alpha_demo));
    echo("Upper side: two 608 bearings per mount + solid shaft coupler + fixed shaft encoder");
    echo("Lower side: lower carbon bar passes through two bearings in robot lower body");
    echo("============================================================");
} else if (part_mode == "lower_socket") {
    lower_socket_part(
        side=is_undef(socket_side) ? "front" : socket_side,
        cutaway=is_undef(part_cutaway) ? false : part_cutaway
    );
} else if (part_mode == "upper_socket") {
    upper_socket_part(
        side=is_undef(socket_side) ? "front" : socket_side
    );
} else if (part_mode == "lock_collar") {
    lock_collar_part(
        dir=is_undef(collar_dir) ? 1 : collar_dir
    );
} else if (part_mode == "solid_coupler" || part_mode == "oldham_coupler") {
    solid_coupler_part(
        cutaway=is_undef(part_cutaway) ? false : part_cutaway
    );
} else if (part_mode == "encoder_bracket") {
    encoder_bracket_part();
} else if (part_mode == "bearing_mount") {
    bearing_mount_part(
        cutaway=is_undef(part_cutaway) ? false : part_cutaway
    );
}
